# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

from typing import Dict, Optional, Tuple, Union
import math
import numpy as np
from einops import rearrange
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from infinity.models.videovae.modules import DiagonalGaussianDistribution
from infinity.models.videovae.utils.misc import ptdtype
from infinity.models.videovae.modules.quantizer import MultiScaleBSQTP_AP as MultiScaleBSQTP_AP
from infinity.models.videovae.modules.quantizer import MultiScaleFSQTP
from infinity.models.videovae.modules.conv_wan import DCDownBlock2d, DCUpBlock2d, DCDownBlock3d, DCUpBlock3d, CogVideoXCausalConv3d, CogVideoXSafeConv3d
from infinity.models.videovae.modules.normalization_wan import get_norm
from infinity.models.videovae.utils.context_parallel import ContextParallelUtils as cp
from infinity.models.videovae.utils.context_parallel import dist_decoder_gather_result, dist_encoder_gather_result
from infinity.models.videovae.utils.dynamic_resolution_two_pyramid import get_ratio2hws_video_v2


def patchify(item):
    """把视频 latent 做 2x2 patch 重排。

    维度关系为：
    `(B, C, T, H, W) -> (B, 4C, T, H/2, W/2)`。
    本质上是先把时间维移到 `pixel_unshuffle` 兼容的位置，再把每个 `2x2`
    空间邻域折叠进通道维。这样后续量化器能在更低空间分辨率上工作。
    """
    assert item.ndim == 5
    # `(H, W)` 上的每个 `2x2` patch 会被重排到通道维，因此通道数变成原来的 4 倍。
    item = torch.nn.functional.pixel_unshuffle(item.permute(0,2,1,3,4), 2).permute(0,2,1,3,4)
    return item

def unpatchify(item):
    """把 patchified latent 还原回原始空间布局。

    维度关系为：
    `(B, 4C, T, H/2, W/2) -> (B, C, T, H, W)`。
    它正好是 `patchify` 的逆操作。
    """
    assert item.ndim == 5
    item = item.permute(0,2,1,3,4) # `(B, 4C, T, H/2, W/2)` 先换成 `pixel_shuffle` 期望的布局。
    item = torch.nn.functional.pixel_shuffle(item, 2) # 每个位置的 4 个子通道重新铺回 `2x2` 空间邻域。
    item = item.permute(0,2,1,3,4) # 恢复为 `(B, C, T, H, W)`。
    return item

class CogVideoXDownsample3D(nn.Module):
    r"""CogVideoX 风格的 3D 下采样层。

    该层可以只压缩空间，也可以同时压缩时间。对视频特征来说，这一步相当于把
    时空分辨率换成更粗的 latent 网格，供后续编码器或量化器继续处理。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time = None,
        down_layer = "conv",
        down_norm = False,
        pad_mode = "constant",
        norm_type=None,
    ):
        """根据 `down_layer` 选择 2D 卷积、2D DC 块或 3D DC 块实现。"""
        super().__init__()

        self.pad_mode = pad_mode
        self.down_layer = down_layer
        if down_layer == "conv":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif down_layer == "dc":
            self.conv = DCDownBlock2d(in_channels, out_channels, downsample=True, shortcut=True, pad_mode=pad_mode, group_norm=down_norm)
        elif down_layer == "3d-dc":
            self.conv = DCDownBlock3d(in_channels, out_channels, group_norm=down_norm, compress_time=compress_time, pad_mode=pad_mode, norm_type=norm_type)
        self.compress_time = compress_time

    def forward(self, x: torch.Tensor, conv_cache: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """执行下采样，并在需要时压缩时间维。"""
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        if self.down_layer == "3d-dc":
            x, new_conv_cache = self.conv(x, conv_cache=conv_cache)
        else:
            if self.compress_time == 2:
                batch_size, channels, frames, height, width = x.shape

                # 先把每个空间位置上的时间序列抽出来，再沿时间维做 1D 平均池化。
                x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

                if x.shape[-1] % 2 == 1:
                    x_first, x_rest = x[..., 0], x[..., 1:]
                    if x_rest.shape[-1] > 0:
                        # 若首帧被单独保留，其余帧按 2 倍时间步长做平均池化。
                        x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

                    x = torch.cat([x_first[..., None], x_rest], dim=-1)
                    # 把压缩后的时间序列重新还原回视频张量布局。
                    x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
                else:
                    x = F.avg_pool1d(x, kernel_size=2, stride=2)
                    x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
            elif self.compress_time == 3:
                batch_size, channels, frames, height, width = x.shape
                x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

                if x.shape[-1] % 2 == 1:
                    x_first, x_rest = x[..., 0], x[..., 1:]
                    if x_rest.shape[-1] > 0:
                        x_rest = F.avg_pool1d(x_rest, kernel_size=3, stride=3)

                    x = torch.cat([x_first[..., None], x_rest], dim=-1)
                    # 代码/形状说明：(batch_size * height * width, channels, (frames // 2) + 1) -> (batch_size, height, width, channels, (frames // 2) + 1) -> (batch_size, channels, (frames // 2) + 1, height, width)
                    x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
                else:
                    # 代码/形状说明：(batch_size * height * width, channels, frames) -> (batch_size * height * width, channels, frames // 2)
                    x = F.avg_pool1d(x, kernel_size=3, stride=3)
                    # 代码/形状说明：(batch_size * height * width, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2) -> (batch_size, channels, frames // 2, height, width)
                    x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)

            # 普通 2D 卷积路径需要手动补齐右侧和下侧，保证偶数下采样更平滑。
            if self.down_layer == "conv":
                pad = (0, 1, 0, 1)
                if self.pad_mode == "constant":
                    x = F.pad(x, pad, mode="constant", value=0)
                else:
                    _shape = x.shape
                    x = F.pad(x, pad, mode="replicate")
                    inputs = inputs.view(*_shape[:-2], *inputs.shape[-2:])

            batch_size, channels, frames, height, width = x.shape
            # 将视频拆成逐帧 2D 特征图后执行卷积，再拼回时间维。
            x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
            x = self.conv(x)
            x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x, new_conv_cache


class CogVideoXUpsample3D(nn.Module):
    r"""CogVideoX 风格的 3D 上采样层。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time = None,
        up_layer = "conv",
        up_norm = False,
        norm_type = None,
        pad_mode = "constant",
    ) -> None:
        """根据 `up_layer` 选择不同的上采样后端。"""
        super().__init__()

        self.up_layer = up_layer
        if up_layer == "conv":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif up_layer == "dc":
            self.conv = DCUpBlock2d(in_channels, out_channels, interpolate=False, shortcut=True, group_norm=up_norm, norm_type=norm_type, pad_mode=pad_mode)
        elif up_layer == "3d-dc":
            self.conv = DCUpBlock3d(in_channels, out_channels, group_norm=up_norm, compress_time=compress_time, norm_type=norm_type, pad_mode=pad_mode)
        self.compress_time = compress_time

    def forward(self, inputs: torch.Tensor, conv_cache: Optional[Dict[str, torch.Tensor]] = None, split_first=False) -> torch.Tensor:
        """执行上采样并返回新的卷积缓存。"""
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        if self.up_layer == "3d-dc":
            inputs, new_conv_cache = self.conv(inputs, conv_cache=conv_cache, split_first=split_first)
        else:
            raise NotImplementedError
            if self.up_layer == "conv":
                spatial_scale = (2., 2.)
            elif self.up_layer == "dc":
                spatial_scale = (1., 1.)
            if self.compress_time:
                temporal_scale = (float(self.compress_time), *spatial_scale)
                if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
                    # 单独处理第一帧。
                    x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]
                    x_first = F.interpolate(x_first, scale_factor=spatial_scale)
                    x_rest = F.interpolate(x_rest, scale_factor=temporal_scale)
                    x_first = x_first[:, :, None, :, :]
                    inputs = torch.cat([x_first, x_rest], dim=2)
                elif inputs.shape[2] > 1:
                    inputs = F.interpolate(inputs, scale_factor=temporal_scale)
                else:
                    inputs = inputs.squeeze(2)
                    inputs = F.interpolate(inputs, scale_factor=spatial_scale)
                    inputs = inputs[:, :, None, :, :]
            else:
                # 只在二维空间上插值。
                b, c, t, h, w = inputs.shape
                inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                inputs = F.interpolate(inputs, scale_factor=spatial_scale)
                inputs = inputs.reshape(b, t, c, *inputs.shape[2:]).permute(0, 2, 1, 3, 4)

            b, c, t, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            inputs = self.conv(inputs)
            inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3, 4)
        return inputs, new_conv_cache

class CogVideoXSpatialNorm3D(nn.Module):
    r"""3D 版本的空间条件归一化。

    先对主特征 `f` 做归一化，再由条件特征 `zq` 预测缩放项 `gamma` 与偏置项 `beta`，
    输出形式为 `Norm(f) * gamma + beta`。这与 SPADE/SpatialNorm 的思路一致。
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
        norm_type = None,
        pad_mode = "constant"
    ):
        """构造条件归一化所需的归一化层和两条 1x1 因果卷积分支。"""
        super().__init__()
        norm_layer = get_norm(norm_type)
        self.norm_layer = norm_layer(num_channels=f_channels, num_groups=groups, eps=1e-6, affine=True)
        self.conv_y = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1, pad_mode=pad_mode)
        self.conv_b = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1, pad_mode=pad_mode)

    def forward(
        self, f: torch.Tensor, zq: torch.Tensor, conv_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """用量化特征 `zq` 调制主特征 `f`。"""
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = F.interpolate(z_first, size=f_first_size)
            z_rest = F.interpolate(z_rest, size=f_rest_size)
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])

        conv_y, new_conv_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_conv_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))

        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        return new_f, new_conv_cache


class CogVideoXResnetBlock3D(nn.Module):
    r"""CogVideoX 使用的 3D ResNet 块。

    该块支持两类条件信息：
    1. `temb` 作为时间/步数嵌入，在第一层卷积后加到特征上；
    2. `zq` 作为空间条件特征，走 `CogVideoXSpatialNorm3D` 调制归一化输出。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        conv_shortcut: bool = False,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "constant",
        norm_type = None,
    ):
        """构造两层因果卷积、归一化和可选捷径投影。"""
        super().__init__()
        norm_layer = get_norm(norm_type)
        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = nn.SiLU()
        self.use_conv_shortcut = conv_shortcut
        self.spatial_norm_dim = spatial_norm_dim

        if spatial_norm_dim is None:
            self.norm1 = norm_layer(num_channels=in_channels, num_groups=groups, eps=eps)
            self.norm2 = norm_layer(num_channels=out_channels, num_groups=groups, eps=eps)
        else:
            self.norm1 = CogVideoXSpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
                norm_type=norm_type,
                pad_mode=pad_mode,
            )
            self.norm2 = CogVideoXSpatialNorm3D(
                f_channels=out_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
                norm_type=norm_type,
                pad_mode=pad_mode,
            )

        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(in_features=temb_channels, out_features=out_channels)

        self.dropout = nn.Dropout(dropout)
        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CogVideoXCausalConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
                )
            else:
                self.conv_shortcut = CogVideoXSafeConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(
        self,
        inputs: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """执行 3D ResNet 前向传播。"""
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states = inputs

        if zq is not None:
            hidden_states, new_conv_cache["norm1"] = self.norm1(hidden_states, zq, conv_cache=conv_cache.get("norm1"))
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states, new_conv_cache["conv1"] = self.conv1(hidden_states, conv_cache=conv_cache.get("conv1"))

        if temb is not None:
            hidden_states = hidden_states + self.temb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if zq is not None:
            hidden_states, new_conv_cache["norm2"] = self.norm2(hidden_states, zq, conv_cache=conv_cache.get("norm2"))
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states, new_conv_cache["conv2"] = self.conv2(hidden_states, conv_cache=conv_cache.get("conv2"))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                inputs, new_conv_cache["conv_shortcut"] = self.conv_shortcut(
                    inputs, conv_cache=conv_cache.get("conv_shortcut")
                )
            else:
                inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states, new_conv_cache


class CogVideoXDownBlock3D(nn.Module):
    r"""编码器中的 3D 下采样阶段。

    该模块先堆叠若干 `CogVideoXResnetBlock3D`，再可选接一个下采样层，
    用于逐步把视频压缩到更粗的 latent 网格。
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
        compress_time = None,
        compress_spatial = None,
        pad_mode: str = "constant",
        norm_type = None,
        down_layer = "conv",
        down_block_mode = "cogvideox",
        down_norm = False,
    ):
        """根据配置构造残差堆栈与可选下采样器。"""
        super().__init__()

        if down_block_mode == "cogvideox":
            resnets = []
            for i in range(num_layers):
                in_channel = in_channels if i == 0 else out_channels
                resnets.append(
                    CogVideoXResnetBlock3D(
                        in_channels=in_channel,
                        out_channels=out_channels,
                        dropout=dropout,
                        temb_channels=temb_channels,
                        groups=resnet_groups,
                        eps=resnet_eps,
                        pad_mode=pad_mode,
                        norm_type=norm_type
                    )
                )
            self.resnets = nn.ModuleList(resnets)
            self.downsamplers = None
            if add_downsample:
                self.downsamplers = nn.ModuleList(
                    [
                        CogVideoXDownsample3D(
                            out_channels, out_channels, padding=downsample_padding, compress_time=compress_time, down_layer=down_layer, down_norm=down_norm, pad_mode=pad_mode, norm_type=norm_type,
                        )
                    ]
                )
        elif down_block_mode == "dc":
            resnets = []
            for i in range(num_layers):
                resnets.append(
                    CogVideoXResnetBlock3D(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        dropout=dropout,
                        temb_channels=temb_channels,
                        groups=resnet_groups,
                        eps=resnet_eps,
                        pad_mode=pad_mode,
                        norm_type=norm_type
                    )
                )
            self.resnets = nn.ModuleList(resnets)
            self.downsamplers = None
            if add_downsample:
                self.downsamplers = nn.ModuleList(
                    [
                        CogVideoXDownsample3D(
                            in_channels, out_channels, padding=downsample_padding, compress_time=compress_time, down_layer=down_layer,down_norm=down_norm, pad_mode=pad_mode, norm_type=norm_type,
                        )
                    ]
                )
        else:
            raise NotImplementedError(f"遇到无效的 `down_block_mode`：{down_block_mode}。")

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""顺序执行残差块和下采样器。"""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    """为 gradient checkpointing 构造无关键字参数的包装函数。"""
                    def create_forward(*inputs):
                        """把 checkpoint 提供的位置参数转发给目标模块。"""
                        return module(*inputs)

                    return create_forward

                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    zq,
                    conv_cache.get(conv_cache_key),
                    use_reentrant=False
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
                )

        if self.downsamplers is not None:
            for i, downsampler in enumerate(self.downsamplers):
                conv_cache_key = f"downsampler_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = downsampler(hidden_states, conv_cache=conv_cache.get(conv_cache_key))

        return hidden_states, new_conv_cache


class CogVideoXMidBlock3D(nn.Module):
    r"""编码器/解码器中间瓶颈块。"""

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "constant",
        norm_type = None
    ):
        """构造中间阶段的若干 3D ResNet 块。"""
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    spatial_norm_dim=spatial_norm_dim,
                    pad_mode=pad_mode,
                    norm_type=norm_type,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""顺序执行中间阶段的所有残差块。"""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    """为 gradient checkpointing 构造包装函数。"""
                    def create_forward(*inputs):
                        """把 checkpoint 的位置参数直接转发给模块。"""
                        return module(*inputs)

                    return create_forward

                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, zq, conv_cache.get(conv_cache_key), use_reentrant=False
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
                )

        return hidden_states, new_conv_cache


class CogVideoXUpBlock3D(nn.Module):
    r"""解码器中的 3D 上采样阶段。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        spatial_norm_dim: int = 16,
        add_upsample: bool = True,
        upsample_padding: int = 1,
        compress_time = None,
        compress_spatial = None,
        pad_mode: str = "constant",
        norm_type = None,
        up_layer = "conv",
        up_block_mode="cogvideox",
        up_norm = False,
    ):
        """构造残差堆栈与可选上采样器。"""
        super().__init__()

        if up_block_mode == "cogvideox":
            resnets = []
            for i in range(num_layers):
                in_channel = in_channels if i == 0 else out_channels
                resnets.append(
                    CogVideoXResnetBlock3D(
                        in_channels=in_channel,
                        out_channels=out_channels,
                        dropout=dropout,
                        temb_channels=temb_channels,
                        groups=resnet_groups,
                        eps=resnet_eps,
                        spatial_norm_dim=spatial_norm_dim,
                        pad_mode=pad_mode,
                        norm_type=norm_type,
                    )
                )
            self.resnets = nn.ModuleList(resnets)
            self.upsamplers = None
            if add_upsample:
                self.upsamplers = nn.ModuleList(
                    [
                        CogVideoXUpsample3D(
                            out_channels, out_channels, padding=upsample_padding, compress_time=compress_time, up_layer=up_layer, up_norm=up_norm, norm_type=norm_type, pad_mode=pad_mode
                        )
                    ]
                )
        elif up_block_mode == "dc":
            resnets = []
            for i in range(num_layers):
                resnets.append(
                    CogVideoXResnetBlock3D(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        dropout=dropout,
                        temb_channels=temb_channels,
                        groups=resnet_groups,
                        eps=resnet_eps,
                        spatial_norm_dim=spatial_norm_dim,
                        pad_mode=pad_mode,
                        norm_type=norm_type,
                    )
                )
            self.resnets = nn.ModuleList(resnets)
            self.upsamplers = None
            if add_upsample:
                self.upsamplers = nn.ModuleList(
                    [
                        CogVideoXUpsample3D(
                            in_channels, out_channels, padding=upsample_padding, compress_time=compress_time, up_layer=up_layer, up_norm=up_norm, norm_type=norm_type, pad_mode=pad_mode
                        )
                    ]
                )
        else:
            raise NotImplementedError(f"遇到无效的 `up_block_mode`：{up_block_mode}。")

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
        split_first = False,
    ) -> torch.Tensor:
        r"""顺序执行残差块和上采样器。"""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    """为 gradient checkpointing 构造包装函数。"""
                    def create_forward(*inputs):
                        """把 checkpoint 的位置参数直接转发给模块。"""
                        return module(*inputs)

                    return create_forward

                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    zq,
                    conv_cache.get(conv_cache_key),
                    use_reentrant=False
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
                )

        if self.upsamplers is not None:
            for i, upsampler in enumerate(self.upsamplers):
                conv_cache_key = f"upsampler_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = upsampler(hidden_states, conv_cache=conv_cache.get(conv_cache_key), split_first=split_first)

        return hidden_states, new_conv_cache


class CogVideoXEncoder3D(nn.Module):
    """3D 视频编码器。

    输入视频会先经过因果 3D 卷积，再通过多个下采样阶段逐步压缩，最终输出
    `2 * latent_channels` 个通道，对应 VAE 的均值和对数方差。
    """
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "constant",
        temporal_compression_list: list = [],
        spatial_compression_list: list = [],
        norm_type=None,
        down_layer = "conv",
        down_block_mode = "cogvideox",
        down_norm=False,
    ):
        """根据给定的下采样配置构造编码器。"""
        super().__init__()

        norm_layer = get_norm(norm_type)
        # 中文说明：temporal_compress_times 的 log2。
        # 代码/形状说明：temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = CogVideoXCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)

        self.down_blocks = nn.ModuleList([])

        # 下采样 block。
        for i, down_block_type in enumerate(down_block_types):
            input_channel = block_out_channels[i]
            output_channel = block_out_channels[i+1]
            compress_time = temporal_compression_list[i] if i < len(temporal_compression_list) else None
            compress_spatial = spatial_compression_list[i] if i < len(spatial_compression_list) else None

            if down_block_type == "CogVideoXDownBlock3D":
                down_block = CogVideoXDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_groups=norm_num_groups,
                    add_downsample=compress_time or compress_spatial,
                    compress_time=compress_time,
                    compress_spatial=compress_spatial,
                    pad_mode=pad_mode,
                    norm_type=norm_type,
                    down_layer=down_layer,
                    down_block_mode=down_block_mode,
                    down_norm=down_norm,
                )
            else:
                raise ValueError("遇到无效的 `down_block_type`。必须为 `CogVideoXDownBlock3D`")

            self.down_blocks.append(down_block)

        # 中间 block。
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=block_out_channels[len(down_block_types)],
            temb_channels=0,
            dropout=dropout,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_groups=norm_num_groups,
            pad_mode=pad_mode,
            norm_type=norm_type,
        )

        self.norm_out = norm_layer(num_channels=block_out_channels[len(down_block_types)], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            block_out_channels[len(down_block_types)], 2 * out_channels, kernel_size=3, pad_mode=pad_mode
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""把输入视频编码为高斯分布参数张量。"""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states, new_conv_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                """为 checkpoint 包装编码器子模块。"""
                def custom_forward(*inputs):
                    """把位置参数转发给被包装模块。"""
                    return module(*inputs)

                return custom_forward

            # 中文说明：1. 下采样阶段
            for i, down_block in enumerate(self.down_blocks):
                conv_cache_key = f"down_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block),
                    hidden_states,
                    temb,
                    None,
                    conv_cache.get(conv_cache_key),
                    use_reentrant=False
                )

            # 2. 中间瓶颈阶段
            hidden_states, new_conv_cache["mid_block"] = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                hidden_states,
                temb,
                None,
                conv_cache.get("mid_block"),
                use_reentrant=False
            )
        else:
            # 中文说明：1. 下采样阶段
            for i, down_block in enumerate(self.down_blocks):
                conv_cache_key = f"down_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = down_block(
                    hidden_states, temb, None, conv_cache=conv_cache.get(conv_cache_key)
                )

            # 2. 中间瓶颈阶段
            hidden_states, new_conv_cache["mid_block"] = self.mid_block(
                hidden_states, temb, None, conv_cache=conv_cache.get("mid_block")
            )

        # 中文说明：3. 后处理阶段
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)

        hidden_states, new_conv_cache["conv_out"] = self.conv_out(hidden_states, conv_cache=conv_cache.get("conv_out"))

        return hidden_states, new_conv_cache


class CogVideoXDecoder3D(nn.Module):
    r"""3D 视频解码器。

    它接收潜变量 `z`，通过中间块和多个上采样阶段逐步恢复时空分辨率，
    最终输出重建视频帧。
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "constant",
        temporal_compression_list: list = [],
        spatial_compression_list: list = [],
        norm_type=None,
        up_layer="conv",
        up_block_mode="cogvideox",
        up_norm=False,
    ):
        """根据给定的上采样配置构造解码器。"""
        super().__init__()

        reversed_block_out_channels = list(reversed(block_out_channels))

        self.conv_in = CogVideoXCausalConv3d(
            in_channels, reversed_block_out_channels[0], kernel_size=3, pad_mode=pad_mode
        )

        # 中间 block。
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=reversed_block_out_channels[0],
            temb_channels=0,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_groups=norm_num_groups,
            spatial_norm_dim=in_channels,
            pad_mode=pad_mode,
            norm_type=norm_type,
        )

        # 上采样 block。
        self.up_blocks = nn.ModuleList([])

        # 输出通道取反向 block 通道列表的第一个值。
        # 代码/形状说明：temporal_compress_level = int(np.log2(temporal_compression_ratio))

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = reversed_block_out_channels[i]
            output_channel = reversed_block_out_channels[i+1]
            if up_block_mode == "cogvideox":
                raise NotImplementedError
                is_final_block = i == len(up_block_types) - 1
                compress_time = temporal_compression_list[i] if i < len(temporal_compression_list) else None
                compress_spatial = spatial_compression_list[i] if i < len(spatial_compression_list) else None
            elif up_block_mode == "dc":
                # 代码/形状说明：is_final_block = i == 0
                idx_temporal = i - (len(up_block_types) - len(temporal_compression_list))
                compress_time = temporal_compression_list[-idx_temporal] if idx_temporal >= 0 else None
                idx_spatial = i - (len(up_block_types) - len(spatial_compression_list))
                compress_spatial = spatial_compression_list[-idx_spatial] if idx_spatial >= 0 else None
                # 代码/形状说明：print(temporal_compression_list, idx_temporal, compress_time, spatial_compression_list, idx_spatial, compress_spatial, compress_time or compress_spatial)

            if up_block_type == "CogVideoXUpBlock3D":
                up_block = CogVideoXUpBlock3D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block + 1,
                    resnet_eps=norm_eps,
                    resnet_groups=norm_num_groups,
                    spatial_norm_dim=in_channels,
                    add_upsample=compress_time or compress_spatial,
                    compress_time=compress_time,
                    compress_spatial=compress_spatial,
                    pad_mode=pad_mode,
                    norm_type=norm_type,
                    up_layer=up_layer,
                    up_block_mode=up_block_mode,
                    up_norm=up_norm,
                )
                prev_output_channel = output_channel
            else:
                raise ValueError("遇到无效的 `up_block_type`。必须为 `CogVideoXUpBlock3D`")

            self.up_blocks.append(up_block)

        self.norm_out = CogVideoXSpatialNorm3D(reversed_block_out_channels[len(up_block_types)], in_channels, groups=norm_num_groups, norm_type=norm_type, pad_mode=pad_mode)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            reversed_block_out_channels[len(up_block_types)], out_channels, kernel_size=3, pad_mode=pad_mode
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
        split_first = False,
    ) -> torch.Tensor:
        r"""把潜变量解码为视频特征或重建帧。"""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states, new_conv_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                """为 checkpoint 包装解码器子模块。"""
                def custom_forward(*inputs):
                    """把位置参数转发给被包装模块。"""
                    return module(*inputs)

                return custom_forward

            # 1. 中间瓶颈阶段
            hidden_states, new_conv_cache["mid_block"] = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                hidden_states,
                temb,
                sample,
                conv_cache.get("mid_block"),
                use_reentrant=False
            )

            # 2. 上采样阶段
            for i, up_block in enumerate(self.up_blocks):
                conv_cache_key = f"up_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block),
                    hidden_states,
                    temb,
                    sample,
                    conv_cache.get(conv_cache_key),
                    split_first,
                    use_reentrant=False
                )
        else:
            # 1. 中间瓶颈阶段
            hidden_states, new_conv_cache["mid_block"] = self.mid_block(
                hidden_states, temb, sample, conv_cache=conv_cache.get("mid_block")
            )

            # 2. 上采样阶段
            for i, up_block in enumerate(self.up_blocks):
                conv_cache_key = f"up_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = up_block(
                    hidden_states, temb, sample, conv_cache=conv_cache.get(conv_cache_key), split_first=split_first
                )

        # 中文说明：3. 后处理阶段
        hidden_states, new_conv_cache["norm_out"] = self.norm_out(
            hidden_states, sample, conv_cache=conv_cache.get("norm_out")
        )
        hidden_states = self.conv_act(hidden_states)
        hidden_states, new_conv_cache["conv_out"] = self.conv_out(hidden_states, conv_cache=conv_cache.get("conv_out"))

        return hidden_states, new_conv_cache


class AutoencoderKLCogVideoX(nn.Module):
    """带 KL 采样与多尺度量化器的 VideoVAE 主模型。

    整体流程可以概括为：
    1. 编码器输出高斯分布参数；
    2. 通过 `DiagonalGaussianDistribution` 采样潜变量；
    3. 对 latent 做 patchify 与量化；
    4. 再 unpatchify 并送入解码器重建视频。
    """
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXResnetBlock3D"]

    def __init__(
        self,
        args
    ):
        """根据训练参数组装编码器、解码器和量化器。"""
        super().__init__()
        self.args = args
        self.embed_dim = args.latent_channels
        self.encoder_dtype = ptdtype[args.encoder_dtype]
        self.decoder_dtype = ptdtype[args.decoder_dtype]

        self.encoder = CogVideoXEncoder3D(
            in_channels=args.in_channels,
            out_channels=args.latent_channels,
            down_block_types=args.down_block_types,
            block_out_channels=args.block_out_channels,
            layers_per_block=args.layers_per_block,
            act_fn=args.act_fn,
            norm_eps=args.norm_eps,
            norm_num_groups=args.norm_num_groups,
            temporal_compression_list=args.temporal_compression_list,
            spatial_compression_list=args.spatial_compression_list,
            pad_mode=args.pad_mode,
            norm_type=args.norm_type,
            down_layer=args.down_layer,
            down_block_mode=args.down_block_mode,
            down_norm=args.down_norm,
        )
        self.decoder = CogVideoXDecoder3D(
            in_channels=args.latent_channels,
            out_channels=args.out_channels,
            up_block_types=args.up_block_types,
            block_out_channels=args.block_out_channels,
            layers_per_block=args.layers_per_block,
            act_fn=args.act_fn,
            norm_eps=args.norm_eps,
            norm_num_groups=args.norm_num_groups,
            temporal_compression_list=args.temporal_compression_list,
            spatial_compression_list=args.spatial_compression_list,
            pad_mode=args.pad_mode,
            norm_type=args.norm_type,
            up_layer=args.up_layer,
            up_block_mode=args.up_block_mode,
            up_norm=args.up_norm,
        )
        self.dropout_z_layer = nn.Dropout(p=args.dropout_z)
        if args.use_checkpoint:
            self._set_gradient_checkpointing(self.encoder, True)
            self._set_gradient_checkpointing(self.decoder, True)

        if args.fix_model != ["no"]:
            for _model in args.fix_model:
                if _model == "encoder":
                    self._set_no_grad(self.encoder)
                elif _model == "decoder":
                    self._set_no_grad(self.decoder)
                elif _model.startswith("down_blocks"):
                    fix_block_num = int(_model.split("_")[2])
                    self._set_no_grad(self.encoder.conv_in)
                    for idx in range(fix_block_num):
                        self._set_no_grad(self.encoder.down_blocks[idx])
                elif _model.startswith("up_blocks"):
                    fix_block_num = int(_model.split("_")[2])
                    self._set_no_grad(self.decoder.conv_out)
                    self._set_no_grad(self.decoder.norm_out)
                    for idx in range(fix_block_num):
                        total_num = len(self.decoder.up_blocks)
                        self._set_no_grad(self.decoder.up_blocks[total_num - idx - 1]) # 反向顺序修正。
                else:
                    raise NotImplementedError

            print("可学习参数：")
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(name)

        # 代码/形状说明：for down_block in self.encoder.down_blocks:
        # 代码/形状说明：if down_block.downsamplers is not None:
        # 代码/形状说明：print(f"下采样时间压缩倍数 {down_block.downsamplers[0].compress_time}")
        # 代码/形状说明：else:
        # 代码/形状说明：print(f"下采样器为空")
        # 代码/形状说明：for up_block in self.decoder.up_blocks:
        # 代码/形状说明：if up_block.upsamplers is not None:
        # 代码/形状说明：print(f"上采样时间压缩倍数 {up_block.upsamplers[0].compress_time}")
        # 代码/形状说明：else:
        # 代码/形状说明：print("上采样器为空")

        self.quant_conv = CogVideoXSafeConv3d(2 * args.out_channels, 2 * args.out_channels, 1) if args.use_quant_conv else None
        self.post_quant_conv = CogVideoXSafeConv3d(args.out_channels, args.out_channels, 1) if args.use_post_quant_conv else None

        self.use_slicing = False
        self.use_tiling = False

        # 这里固定一次解码 2 个 latent 帧，是为了和训练时的时间卷积缓存语义保持一致。
        # 若一次处理 `X` 个 latent 帧，因果卷积缓存和两级时间上采样会额外引入若干输出帧，
        # 粗略关系可写成：`输出帧数 ≈ X + 2 + 2 + 4 - 2 = X + 6`。
        # 继续增大这个批量虽能提高吞吐，但会显著增加显存，而且会偏离模型训练时见过的时间模式。
        self.num_latent_frames_batch_size = 2
        self.num_sample_frames_batch_size = 2 * int(math.prod([float(a) for a in self.args.temporal_compression_list]))

        # tile 阈值通常取官方推荐分辨率的一半，便于在大分辨率下做分块编解码。
        self.tile_sample_min_height = args.sample_height // 2
        self.tile_sample_min_width = args.sample_width // 2
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / 8
        )
        self.tile_latent_min_width = int(self.tile_sample_min_width / 8)

        # overlap 系数控制相邻 tile 的重叠区域，用于后续平滑拼接，减少块状边界。
        self.tile_overlap_factor_height = 1 / 6
        self.tile_overlap_factor_width = 1 / 5

        if cp.is_cp_initialized():
            self.cp_size = cp.get_cp_size()
            self.cp_rank = cp.get_cp_rank()

        self.lfq_weight = args.lfq_weight
        self.commitment_loss_weight = args.commitment_loss_weight
        self.compute_all_commitment = args.compute_all_commitment # 中文标题：compute commitment between input and rq-output
        if args.quantizer_type == 'MultiScaleBSQ':
            quantizer_class = MultiScaleBSQ
        elif args.quantizer_type == 'MultiScaleBSQTP':
            quantizer_class = MultiScaleBSQTP_AP
        elif args.quantizer_type == 'MultiScaleFSQ':
            quantizer_class = MultiScaleFSQ
        elif args.quantizer_type == 'MultiScaleFSQTP':
            quantizer_class = MultiScaleFSQTP
        elif args.quantizer_type == 'MultiScaleFSQSIM':
            quantizer_class = MultiScaleFSQSIM
        else:
            raise NotImplementedError

        ratio2hws_video_common_v2, total_pixels2scales = get_ratio2hws_video_v2()
        scales_256 = total_pixels2scales['0.06M']
        h_div_w2hw = {}
        for h_div_w in ratio2hws_video_common_v2:
            h_div_w2hw[h_div_w] = ratio2hws_video_common_v2[h_div_w][scales_256-1]
            h_div_w2hw[1/h_div_w] = (h_div_w2hw[h_div_w][1], h_div_w2hw[h_div_w][0])
        self.h_div_w2hw = h_div_w2hw
        self.h_div_w_templates = np.array(list(self.h_div_w2hw.keys()))
        self.scales_256 = scales_256
        args.h_div_w2hw = h_div_w2hw
        args.h_div_w_templates = self.h_div_w_templates
        args.scales_256 = scales_256
        dim = args.codebook_dim if args.codebook_dim_low < 0 else args.codebook_dim_low * 4
        self.quantizer = quantizer_class(
            dim = args.codebook_dim_low * 4, # 中文说明：这里是输入特征维度；未显式指定时，默认使用 log2(codebook_size)。
            entropy_loss_weight = args.entropy_loss_weight, # 中文标题：entropy loss 的权重大小
            commitment_loss_weight=args.commitment_loss_weight, # 中文标题：commitment loss 的权重大小
            use_stochastic_depth=args.use_stochastic_depth,
            drop_rate=args.drop_rate,
            schedule_mode=args.schedule_mode,
            keep_first_quant=args.keep_first_quant,
            keep_last_quant=args.keep_last_quant,
            remove_residual_detach=args.remove_residual_detach,
            use_out_phi=args.use_out_phi,
            use_out_phi_res=args.use_out_phi_res,
            random_flip = args.random_flip,
            flip_prob = args.flip_prob,
            flip_mode = args.flip_mode,
            max_flip_lvl = args.max_flip_lvl,
            random_flip_1lvl = args.random_flip_1lvl,
            flip_lvl_idx = args.flip_lvl_idx,
            drop_when_test = args.drop_when_test,
            drop_lvl_idx = args.drop_lvl_idx,
            drop_lvl_num = args.drop_lvl_num,
            random_short_schedule = args.random_short_schedule,
            short_schedule_prob = args.short_schedule_prob,
            use_bernoulli = args.use_bernoulli,
            use_rot_trick = args.use_rot_trick,
            disable_flip_prob = args.disable_flip_prob,
            casual_multi_scale = args.casual_multi_scale,
            temporal_slicing = args.temporal_slicing,
            last_scale_repeat_n = args.last_scale_repeat_n,
            num_lvl_fsq = args.num_lvl_fsq,
            other_args=args,
        )
        self.quantize = self.quantizer
        self.codebook_dim_continuous = args.codebook_dim
        assert args.codebook_dim_low > 0
        self.codebook_dim = args.codebook_dim_low * 4
        self.vocab_size = 2**self.codebook_dim

        if args.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if args.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.origin_dim = 64
        assert args.use_feat_proj in [0, 1, 2], f'use_feat_proj 只支持 0、1、2，实际为 {args.use_feat_proj}'
        if args.use_feat_proj > 0:
            if args.use_feat_proj == 1:
                self.proj_down = nn.Linear(self.origin_dim*2, self.origin_dim*2)
                self.proj_down_two = nn.Linear(self.origin_dim*2, self.origin_dim*2)
            elif args.use_feat_proj == 2:
                self.proj_down = nn.Linear(self.origin_dim, self.origin_dim)
                self.proj_down_two = nn.Linear(self.origin_dim, self.origin_dim)
            self.proj_up = nn.Linear(self.origin_dim, self.origin_dim)
            self.proj_up_two = nn.Linear(self.origin_dim, self.origin_dim)
        else:
            self.proj_down, self.proj_up, self.proj_down_two, self.proj_up_two = nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()
        self.other_args = args
        self.scale_learnable_parameters = nn.Parameter(torch.ones(4))

    def _set_gradient_checkpointing(self, module, value=False, subset=True):
        """为编码器/解码器及其子模块统一开关 gradient checkpointing。"""
        if isinstance(module, (CogVideoXEncoder3D, CogVideoXDecoder3D)):
            module.gradient_checkpointing = value

        for n, m in module.named_modules():
            if hasattr(m, 'gradient_checkpointing') and subset:
                m.gradient_checkpointing = value

    def _set_no_grad(self, module):
        """冻结指定模块的全部参数。"""
        for param in module.parameters():
            param.requires_grad = False

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_overlap_factor_height: Optional[float] = None,
        tile_overlap_factor_width: Optional[float] = None,
    ) -> None:
        r"""开启分块编解码。

        当输入分辨率过大时，VAE 会把视频切成带重叠的小块分别编解码，以显著降低显存占用。
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / 8
        )
        self.tile_latent_min_width = int(self.tile_sample_min_width / 8)
        self.tile_overlap_factor_height = tile_overlap_factor_height or self.tile_overlap_factor_height
        self.tile_overlap_factor_width = tile_overlap_factor_width or self.tile_overlap_factor_width

    def disable_tiling(self) -> None:
        r"""关闭分块编解码，恢复整图/整视频处理。"""
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""开启 batch slicing，用逐样本切片的方式节省显存。"""
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""关闭 batch slicing。"""
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """执行底层编码流程，并处理时间分片与 context-parallel 缓存。"""
        batch_size, num_channels, num_frames, height, width = x.shape
        self.raw_height = height
        self.raw_width = width

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        frame_batch_size = self.num_sample_frames_batch_size
        # 说明：帧数应为 `1`、`frame_batch_size * k` 或 `frame_batch_size * k + 1`，其中 k 为整数。
        # 中文说明：多出来的单帧会在循环内部处理，所以这里不需要向上取整。
        num_batches = max(num_frames // frame_batch_size, 1)
        if num_batches > 1:
            if cp.is_cp_initialized():
                frame_batch_size = num_frames // self.cp_size
                num_batches = self.cp_size
                cp.set_cp_on(True)
        else:
            cp.set_cp_on(False)


        conv_cache = None
        enc = []

        for i in range(num_batches):
            if cp.cp_on() and i != self.cp_rank:
                continue

            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            x_intermediate = x[:, :, start_frame:end_frame]


            torch._dynamo.mark_dynamic(x_intermediate, 0)
            torch._dynamo.mark_dynamic(x_intermediate, 2)
            if conv_cache is not None:
                for key, tensor in conv_cache.items():
                    if tensor is not None and isinstance(tensor, torch.Tensor):
                        torch._dynamo.mark_dynamic(tensor, 0)

            x_intermediate, conv_cache = self.encoder(x_intermediate, conv_cache=conv_cache)

            if self.quant_conv is not None:
                x_intermediate = self.quant_conv(x_intermediate)

            enc.append(x_intermediate)

        if cp.cp_on():
            enc = dist_encoder_gather_result(enc[0])

        enc = torch.cat(enc, dim=2)

        return enc

    def encode_for_raw_features(
        self, x: torch.Tensor,
        scale_schedule,
        return_residual_norm_per_scale=False,
        slice=None,
    ):
        """返回未量化的连续 latent 特征，供外部模块直接消费。"""
        is_image = x.ndim == 4
        if not is_image:
            B, C, T, H, W = x.shape
        else:
            B, C, H, W = x.shape
            T = 1
            x = x.unsqueeze(2)

        with torch.amp.autocast("cuda", dtype=self.encoder_dtype):
            h = self.encode(x)
        # 先做 patchify，把空间上的 `2x2` 邻域折叠进通道维，便于后续量化。
        h = patchify(h) # (B,c,t,H,W) -> (B,4c,t,H/2,W/2)

        posterior = DiagonalGaussianDistribution(h)
        z = posterior.sample()
        z = self.dropout_z_layer(z)
        if self.other_args.use_feat_proj == 2:
            z = self.proj_down(z.permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,24,t,H/2,W/2)
        z = z * self.scale_learnable_parameters[0]
        return z, None, None


    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ):
        """编码输入视频，返回连续 latent 参数张量。"""
        h = None
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        if not return_dict:
            return (h,)
        return h

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        """执行底层解码流程，并处理时间分片与 context-parallel 缓存。"""
        batch_size, num_channels, num_frames, height, width = z.shape

        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        frame_batch_size = self.num_latent_frames_batch_size

        num_batches = max(num_frames // frame_batch_size, 1)
        split_first = False
        if num_frames % frame_batch_size == 0 and num_batches:
            split_first = True
            num_batches -= 1
        if num_batches > 1:
            if cp.is_cp_initialized():
                frame_batch_size = num_frames // self.cp_size
                num_batches = self.cp_size
                cp.set_cp_on(True)
        else:
            cp.set_cp_on(False)

        conv_cache = None
        dec = []

        start_frame = 0
        remaining_frames = num_frames % frame_batch_size
        if split_first:
            remaining_frames += frame_batch_size
        for i in range(num_batches):
            if cp.cp_on() and i != self.cp_rank:
                continue

            end_frame = frame_batch_size * (i + 1) + remaining_frames
            z_intermediate = z[:, :, start_frame:end_frame]
            start_frame = end_frame
            if self.post_quant_conv is not None:
                z_intermediate = self.post_quant_conv(z_intermediate)


            torch._dynamo.mark_dynamic(z_intermediate, 0)
            torch._dynamo.mark_dynamic(z_intermediate, 2)
            torch._dynamo.mark_dynamic(z_intermediate, 3)
            torch._dynamo.mark_dynamic(z_intermediate, 4)
            if conv_cache is not None:
                for key, tensor in conv_cache.items():
                    if tensor is not None and isinstance(tensor, torch.Tensor):
                        torch._dynamo.mark_dynamic(tensor, 0)

            z_intermediate, conv_cache = self.decoder(z_intermediate, conv_cache=conv_cache, split_first=split_first)
            split_first = False

            dec.append(z_intermediate)

        if cp.cp_on():
            dec = dist_decoder_gather_result(dec[0])

        dec = torch.cat(dec, dim=2)

        if not return_dict:
            return (dec,)

        return dec

    def decode(self, z: torch.Tensor, return_dict: bool = True, **kwargs):
        """把量化后的 latent 解码回视频。

        这里会先撤销尺度放缩与线性投影，再执行 `unpatchify` 恢复空间布局，
        最后进入 3D 解码器。
        """

        z = z / self.scale_learnable_parameters[0]
        z = self.proj_up(z.permute(0,2,3,4,1)).permute(0,4,1,2,3)

        z = unpatchify(z)
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)
        return decoded

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """在垂直方向平滑拼接两个重叠 tile。"""
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """在水平方向平滑拼接两个重叠 tile。"""
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""按 tile 编码视频。

        输入会被切成带重叠的小块分别送入编码器，再通过 `blend_v/blend_h`
        在重叠区域做线性混合，减少块边界造成的接缝。
        """
        # 显存估算可参考下方 `tiled_decode` 的注释。
        batch_size, num_channels, num_frames, height, width = x.shape

        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_latent_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_latent_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_latent_min_height - blend_extent_height
        row_limit_width = self.tile_latent_min_width - blend_extent_width
        frame_batch_size = self.num_sample_frames_batch_size

        # 逐 tile 编码，并保留重叠区域，后续通过线性混合消除拼接缝。
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                # 说明：帧数应为 `1`、`frame_batch_size * k` 或 `frame_batch_size * k + 1`，其中 k 为整数。
                # 中文说明：多出来的单帧会在循环内部处理，所以这里不需要向上取整。
                num_batches = max(num_frames // frame_batch_size, 1)
                conv_cache = None
                time = []

                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = x[
                        :,
                        :,
                        start_frame:end_frame,
                        i : i + self.tile_sample_min_height,
                        j : j + self.tile_sample_min_width,
                    ]
                    tile, conv_cache = self.encoder(tile, conv_cache=conv_cache)
                    if self.quant_conv is not None:
                        tile = self.quant_conv(tile)
                    time.append(tile)

                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # 与上方、左侧 tile 的重叠区域做平滑混合。
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))

        enc = torch.cat(result_rows, dim=3)
        return enc

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
        """按 tile 解码 latent。

        整图解码时，中间 3D 卷积激活会非常大；按半分辨率 tile 处理后，
        激活量大致会随空间面积缩小到原来的四分之一，因此显存占用显著下降。
        """

        batch_size, num_channels, num_frames, height, width = z.shape

        overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_sample_min_height - blend_extent_height
        row_limit_width = self.tile_sample_min_width - blend_extent_width
        frame_batch_size = self.num_latent_frames_batch_size

        # 逐 tile 解码，并保留重叠区域供后续平滑拼接。
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                num_batches = max(num_frames // frame_batch_size, 1)
                conv_cache = None
                time = []

                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = z[
                        :,
                        :,
                        start_frame:end_frame,
                        i : i + self.tile_latent_min_height,
                        j : j + self.tile_latent_min_width,
                    ]
                    if self.post_quant_conv is not None:
                        tile = self.post_quant_conv(tile)
                    tile, conv_cache = self.decoder(tile, conv_cache=conv_cache)
                    time.append(tile)

                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # 与上方、左侧 tile 的重叠区域做平滑混合。
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)

        if not return_dict:
            return (dec,)

        return dec

    ### 中文标题：原始 CogVideoX forward 示例
    # 代码/形状说明：def forward(
    # 中文说明：self,
    # 中文说明：sample: torch.Tensor,
    # 代码/形状说明：sample_posterior: bool = False,
    # 代码/形状说明：return_dict: bool = True,
    # 代码/形状说明：generator: Optional[torch.Generator] = None,
    # 代码/形状说明：) -> Union[torch.Tensor, torch.Tensor]:
    # 中文标题：x = sample
    # 代码/形状说明：posterior = self.encode(x).latent_dist
    # 代码/形状说明：if sample_posterior:
    # 代码/形状说明：z = posterior.sample(generator=generator)
    # 代码/形状说明：else:
    # 代码/形状说明：z = posterior.mode()
    # 代码/形状说明：dec = self.decode(z)
    # 代码/形状说明：if not return_dict:
    # 代码/形状说明：return (dec,)
    # 中文标题：return dec

    def forward(self, x, disc_factor, image_disc=None, video_disc=None, image_perceptual_model=None, video_perceptual_model=None, is_train=True):
        """执行训练态或推理态前向。

        训练态下会返回重建结果以及重建损失、感知损失、GAN 损失等统计；
        推理态下则直接返回输入与重建结果。
        """
        device = x.device
        is_image = x.ndim == 4
        if not is_image:
            B, C, T, H, W = x.shape
        else:
            B, C, H, W = x.shape
            T = 1
            x = x.unsqueeze(2)

        semantic_enlarge_factor = torch.clamp(self.scale_learnable_parameters, min=0.01)[0] # 低分辨率语义分支的可学习放缩。
        detail_enlarge_factor = torch.clamp(self.scale_learnable_parameters, min=0.01)[1] # 高分辨率细节分支的可学习放缩。

        h_div_w = H / W
        h_div_w_template = self.h_div_w_templates[np.argmin(np.abs(self.h_div_w_templates - h_div_w))]
        hh, ww = self.h_div_w2hw[h_div_w_template]
        is_high_resolution = H*W > hh*ww*256
        x_list = []
        if self.other_args.use_multi_scale and is_high_resolution:
            x_list.append(F.interpolate(x, size=(T, hh*16, ww*16), mode=self.quantizer.z_interplote_down))
        x_list.append(x)
        assert len(x_list) <= 2
        z_list = []
        for i, x in enumerate(x_list):
            with torch.amp.autocast("cuda", dtype=self.encoder_dtype):
                h = self.encode(x)
            # 先做 patchify，把空间上的 `2x2` 邻域折叠进通道维。
            h = patchify(h) # (B,c,t,H,W) -> (B,4c,t,H/2,W/2)

            if self.other_args.use_feat_proj == 1:
                if i==0:
                    h = self.proj_down(h.permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,24,t,H/2,W/2)
                elif i==1:
                    h = self.proj_down_two(h.permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,24,t,H/2,W/2)

            posterior = DiagonalGaussianDistribution(h)
            z = posterior.sample()
            z = self.dropout_z_layer(z)

            if self.other_args.use_feat_proj == 2:
                if i==0:
                    z = self.proj_down(z.permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,24,t,H/2,W/2)
                elif i==1:
                    z = self.proj_down_two(z.permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,24,t,H/2,W/2)

            if i == 0:
                z_list.append(z.clone() * semantic_enlarge_factor)
            elif i==1:
                z_list.append(z.clone() * detail_enlarge_factor)

        # 把连续 latent 送入多尺度量化器，得到离散近似与码本索引。
        z_list, all_indices, all_loss = self.quantizer(z_list) # (B,24,t,H/2,W/2)

        x_recon_list = []
        for i in range(len(z_list)):
            if i==0:
                z_list[i] = z_list[i] / semantic_enlarge_factor
                z_list[i] = self.proj_up(z_list[i].permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,64,t,H/2,W/2)
            elif i==1:
                z_list[i] = z_list[i] / detail_enlarge_factor
                z_list[i] = self.proj_up_two(z_list[i].permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,64,t,H/2,W/2)

            z_list[i] = unpatchify(z_list[i]) # (B,4c,t,H/2,W/2) -> (B,c,t,H,W)

            with torch.amp.autocast("cuda", dtype=self.decoder_dtype):
                x_recon = self.decode(z_list[i]).to(torch.float32)
            x_recon_list.append(x_recon)

        loss_dict, log_dict = {}, {}
        log_dict['semantic_enlarge_factor'] = torch.tensor(self.scale_learnable_parameters[0].item(), device=device)
        log_dict['detail_enlarge_factor'] = torch.tensor(self.scale_learnable_parameters[1].item(), device=device)

        if "FSQ" in self.args.quantizer_type:
            vq_output = {"encodings": all_indices}
        else:
            vq_output = {
                "commitment_loss": torch.mean(all_loss) * self.lfq_weight, # 中文标题：这里的 commitment loss 是 commitment loss 与 entropy penalty 的总和
                "encodings": all_indices,
            }

        if is_train == False:
            if self.other_args.return_256_res:
                return x_list[0], x_recon_list[0]
            else:
                return x_list[-1], x_recon_list[-1]

        # 代码/形状说明：if is_high_resolution_video:
        # 代码/形状说明：x_recon_list, x_list = x_recon_list[1:], x_list[1:]
        if "FSQ" not in self.args.quantizer_type:
            loss_dict["train/commitment_loss"] = vq_output['commitment_loss']
            # 代码/形状说明：loss_dict["train/all_commitment_loss"] = vq_output['all_commitment_loss']
        for (x_recon, x) in zip(x_recon_list, x_list):
            if self.args.recon_loss_type == 'l1':
                recon_loss = F.l1_loss(x_recon, x) * self.args.l1_weight
            else:
                recon_loss = F.mse_loss(x_recon, x) * self.args.l1_weight
            if 'train/recon_loss' not in loss_dict:
                loss_dict['train/recon_loss'] = recon_loss
            else:
                loss_dict['train/recon_loss'] += recon_loss

            if is_image: # 图像输入会临时补成单帧视频，这里再压回 4D。
                flat_frames = x = x.squeeze(2)
                flat_frames_recon = x_recon = x_recon.squeeze(2)
            else:
                flat_frames = rearrange(x, "B C T H W -> (B T) C H W")
                flat_frames_recon = rearrange(x_recon, "B C T H W -> (B T) C H W")

            # 感知损失更关注纹理与结构相似性，而不只是逐像素误差。
            if is_image:
                image_perceptual_loss = image_perceptual_model(flat_frames, flat_frames_recon).mean() * self.args.perceptual_weight
                if "train/image_perceptual_loss" not in loss_dict:
                    loss_dict["train/image_perceptual_loss"] = image_perceptual_loss
                else:
                    loss_dict["train/image_perceptual_loss"] += image_perceptual_loss
            else:
                if self.args.lpips_model == "swin3d_t":
                    video_perceptual_loss = video_perceptual_model(x, x_recon).mean() * self.args.video_perceptual_weight
                else:
                    video_perceptual_loss = video_perceptual_model(flat_frames, flat_frames_recon).mean() * self.args.video_perceptual_weight
                if "train/video_perceptual_loss" not in loss_dict:
                    loss_dict["train/video_perceptual_loss"] = video_perceptual_loss
                else:
                    loss_dict["train/video_perceptual_loss"] += video_perceptual_loss

            ### GAN 损失鼓励重建结果在判别器看来更像真实样本。
            if self.args.image_gan_weight > 0 and (self.args.gan_image4video == "yes" or is_image):
                logits_image_fake = image_disc(flat_frames_recon)
                g_image_loss = -torch.mean(logits_image_fake) * self.args.image_gan_weight * disc_factor
                if 'train/g_image_loss' not in loss_dict:
                    loss_dict["train/g_image_loss"] = g_image_loss
                else:
                    loss_dict["train/g_image_loss"] += g_image_loss
            if T > 1 and self.args.video_gan_weight > 0:
                logits_video_fake = video_disc(x_recon)
                g_video_loss = -torch.mean(logits_video_fake) * self.args.video_gan_weight * disc_factor
                if 'train/g_video_loss' not in loss_dict:
                    loss_dict["train/g_video_loss"] = g_video_loss
                else:
                    loss_dict["train/g_video_loss"] += g_video_loss

        loss_dict['train/recon_loss'] /= len(x_list)
        if "train/image_perceptual_loss" in loss_dict:
            loss_dict["train/image_perceptual_loss"] /= len(x_list)
        if "train/video_perceptual_loss" in loss_dict:
            loss_dict["train/video_perceptual_loss"] /= len(x_list)

        x_recon1, flat_frames1, flat_frames_recon1 = x_recon.detach(), flat_frames.detach(), flat_frames_recon.detach()

        return (x, x_recon1, flat_frames1, flat_frames_recon1, loss_dict, log_dict)


    @staticmethod
    def add_model_specific_args(parent_parser):
        """向外部 `ArgumentParser` 注册 VideoVAE 相关命令行参数。"""
        from infinity.models.videovae.utils import str2bool

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--out_channels", type=int, default=3)
        parser.add_argument("--down_block_types", type=str, nargs='+', default=[
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ])
        parser.add_argument("--down_block_mode", type=str, default="cogvideox", choices=["cogvideox", "dc"])
        parser.add_argument("--up_block_types", type=str, nargs='+', default=[
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ])
        parser.add_argument("--up_block_mode", type=str, default="cogvideox", choices=["cogvideox", "dc"])
        parser.add_argument("--block_out_channels", type=int, nargs='+', default=[128, 128, 256, 256, 512, 512])
        parser.add_argument("--layers_per_block", type=int, default=3)
        parser.add_argument("--latent_channels", type=int, default=16)
        parser.add_argument("--act_fn", type=str, default="silu")
        parser.add_argument("--norm_eps", type=float, default=1e-6)
        parser.add_argument("--norm_num_groups", type=int, default=32)
        # 中文说明：parser.add_argument("--temporal_compression_ratio", type=float, default=4) # 已废弃
        parser.add_argument("--spatial_compression_list", type=int, nargs='+', default=[2, 2, 2], choices=[2])
        parser.add_argument("--temporal_compression_list", type=int, nargs='+', default=[2, 2], choices=[2, 3])
        parser.add_argument("--sample_height", type=int, default=480)
        parser.add_argument("--sample_width", type=int, default=720)
        parser.add_argument("--use_quant_conv", action="store_true")
        parser.add_argument("--use_post_quant_conv", action="store_true")
        parser.add_argument("--down_layer", type=str, default="conv", choices=["conv", "dc", "3d-dc"])
        parser.add_argument('--down_norm', type=str2bool, default=False)
        parser.add_argument("--up_layer", type=str, default="conv", choices=["conv", "dc", "3d-dc"])
        parser.add_argument('--up_norm', type=str2bool, default=False)
        parser.add_argument("--pad_mode", type=str, default="constant", choices=["constant", "replicate"])
        parser.add_argument("--dropout_z", type=float, default=0.0)
        return parser

if __name__ == '__main__':
    pass
