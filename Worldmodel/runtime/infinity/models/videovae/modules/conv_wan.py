# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from .misc import swish
from infinity.models.videovae.modules.normalization_wan import get_norm
from infinity.models.videovae.utils.context_parallel import ContextParallelUtils as cp
from infinity.models.videovae.utils.context_parallel import dist_conv_cache_send, dist_conv_cache_recv


class DCDownBlock3d(nn.Module):
    """WAN 版 3D 下采样残差块。"""
    def __init__(self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        group_norm=False,
        compress_time=False,
        norm_type=None,
        pad_mode="constant",
    ) -> None:
        """构造 WAN 版 3D 下采样残差块。"""
        super().__init__()
        self.shortcut = shortcut
        self.compress_time = compress_time
        if group_norm:
            norm_layer = get_norm(norm_type)
            self.norm = norm_layer(num_channels=in_channels, num_groups=32, eps=1e-6, affine=True)
            self.nonlinearity = swish
        else:
            self.norm = nn.Identity()
            self.nonlinearity = nn.Identity()
        self.spatial_factor = 2
        self.temporal_factor = int(compress_time) if compress_time else 1
        out_ratio = self.spatial_factor**2
        assert out_channels % out_ratio == 0
        out_channels = out_channels // out_ratio

        # 代码/形状说明：self.conv = nn.Conv3d(
        # 中文说明：in_channels,
        # 中文说明：out_channels,
        # 中文说明：kernel_size=3,
        # 中文说明：stride=(1, 1, 1),
        # 中文说明：padding=0,
        # )
        self.conv = CogVideoXCausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)

    def forward(self, hidden_states: torch.Tensor, conv_cache: Optional[Dict[str, torch.Tensor]] = None, temporal_compress = True) -> torch.Tensor:
        """执行 3D 下采样，并按需要压缩时间维。"""
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        x = hidden_states
        x = self.nonlinearity(self.norm(x))
        assert x.ndim == 5, f"x.ndim 必须对应 (B C T H W)"

        ### 旧的 `nn.Conv3d` 路径保留在注释中，便于对照。
        # 代码/形状说明：x = F.pad(x, (1, 1, 1, 1, 2, 0))  # 因果 padding：左、右、上、下、前、后
        # 代码/形状说明：x[:, :, :2, 1:-1, 1:-1] = x[:, :, 2:3, 1:-1, 1:-1].clone() # 广播第一个有效时间值
        # 代码/形状说明：x = self.conv(x)

        ### 当前实现使用带缓存的因果 3D 卷积。
        x, new_conv_cache["conv"] = self.conv(x, conv_cache=conv_cache.get("conv"))

        if x.shape[2] > 1:
            if x.shape[2] % 2 == 1:
                x_first, x_rest = x[:, :, 0, ...], x[:, :, 1:, ...]
                y_first, y_rest = hidden_states[:, :, 0, ...], hidden_states[:, :, 1:, ...]
            else:
                x_first, x_rest = None, x
                y_first, y_rest = None, hidden_states
        elif x.shape[2] == 1:
            x_first, x_rest = x[:, :, 0, ...], None
            y_first, y_rest = hidden_states[:, :, 0, ...], None
        else:
            raise NotImplementedError
        if x_first is not None:
            x_first = rearrange(x_first, "b c (h ph) (w pw) -> b (ph pw c) h w", ph=self.spatial_factor, pw=self.spatial_factor)
            y_first = rearrange(y_first, "b c (h ph) (w pw) -> b (ph pw c) h w", ph=self.spatial_factor, pw=self.spatial_factor)
        if x_rest is not None:
            if temporal_compress:
                x_rest = rearrange(x_rest, "b c (t pt) (h ph) (w pw) -> b (ph pw c) t pt h w", pt=self.temporal_factor, ph=self.spatial_factor, pw=self.spatial_factor)
                x_rest = x_rest.mean(dim=3)
                y_rest = rearrange(y_rest, "b c (t pt) (h ph) (w pw) -> b (ph pw c) t pt h w", pt=self.temporal_factor, ph=self.spatial_factor, pw=self.spatial_factor)
                y_rest = y_rest.mean(dim=3)
            else:
                x_rest = rearrange(x_rest, "b c (t pt) (h ph) (w pw) -> b (ph pw c) (t pt) h w", pt=self.temporal_factor, ph=self.spatial_factor, pw=self.spatial_factor)
                y_rest = rearrange(y_rest, "b c (t pt) (h ph) (w pw) -> b (ph pw c) (t pt) h w", pt=self.temporal_factor, ph=self.spatial_factor, pw=self.spatial_factor)
        if x_first is not None and x_rest is not None:
            x = torch.cat([x_first[:,:, None,...], x_rest], dim=2)
            y = torch.cat([y_first[:,:, None,...], y_rest], dim=2)
        else:
            x = x_first[:,:, None,...] if x_first is not None else x_rest
            y = y_first[:,:, None,...] if y_first is not None else y_rest
        if self.shortcut:
            y = rearrange(y, "b (g c) t h w -> b g c t h w", c=x.shape[1]).mean(dim=1)
            hidden_states = x + y
        else:
            hidden_states = x
        return hidden_states, new_conv_cache

class DCUpBlock3d(nn.Module):
    """WAN 版 3D 上采样残差块。"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
        group_norm=False,
        compress_time=False,
        norm_type=None,
        pad_mode="constant",
    ) -> None:
        """构造 WAN 版 3D 上采样残差块。"""
        super().__init__()

        self.compress_time = compress_time
        if group_norm:
            norm_layer = get_norm(norm_type)
            self.norm = norm_layer(num_channels=in_channels, num_groups=32, eps=1e-6, affine=True)
            self.nonlinearity = swish
        else:
            self.norm = nn.Identity()
            self.nonlinearity = nn.Identity()
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.spatial_factor = 2
        self.temporal_factor = int(compress_time) if compress_time else 1
        out_channels = out_channels * self.spatial_factor**2 * self.temporal_factor
        # 代码/形状说明：self.conv = nn.Conv3d(in_channels, out_channels, 3, (1, 1, 1), 0)
        self.conv = CogVideoXCausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        assert out_channels % in_channels == 0
        self.repeats = out_channels // in_channels

    def forward(self, hidden_states: torch.Tensor, conv_cache: Optional[Dict[str, torch.Tensor]] = None, split_first=False) -> torch.Tensor:
        """执行 3D 上采样并恢复时空分辨率。"""
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        x = hidden_states
        x = self.nonlinearity(self.norm(x))

        compress_first = False
        if x.shape[2] % 2 == 1 or split_first:
            compress_first = True

        ### 旧的 `nn.Conv3d` 路径保留在注释中，便于对照。
        # 代码/形状说明：x = F.pad(x, (1, 1, 1, 1, 2, 0))  # 因果 padding：左、右、上、下、前、后
        # 代码/形状说明：x[:, :, :2, 1:-1, 1:-1] = x[:, :, 2:3, 1:-1, 1:-1].clone() # 广播第一个有效时间值
        # 代码/形状说明：x = self.conv(x)

        ### 当前实现使用带缓存的因果 3D 卷积。
        x, new_conv_cache["conv"] = self.conv(x, conv_cache=conv_cache.get("conv"))

        x = rearrange(x, "b (pt ph pw c) t h w -> b c (t pt) (h ph) (w pw)", pt=self.temporal_factor, ph=self.spatial_factor, pw=self.spatial_factor)
        y = repeat(hidden_states, "b c t h w -> b (r c) t h w", r=self.repeats)
        y = rearrange(y, "b (pt ph pw c) t h w -> b c (t pt) (h ph) (w pw)", pt=self.temporal_factor, ph=self.spatial_factor, pw=self.spatial_factor)

        # 若首帧被单独保留，则把时间展开后的前 `pt` 帧压回 1 帧。
        if self.temporal_factor > 1 and compress_first:
            if x.shape[2] > 1:
                x_first, x_rest = x[:, :, :self.temporal_factor, ...], x[:, :, self.temporal_factor:, ...]
                y_first, y_rest = y[:, :, :self.temporal_factor, ...], y[:, :, self.temporal_factor:, ...]
            elif x.shape[2] == 1:
                assert x.shape[2] == y.shape[2] == self.temporal_factor
                x_first, x_rest = x, None
                y_first, y_rest = y, None
            else:
                raise NotImplementedError
            x = torch.cat([x_first.mean(dim=2, keepdim=True), x_rest], dim=2)
            y = torch.cat([y_first.mean(dim=2, keepdim=True), y_rest], dim=2)
        if self.shortcut:
            hidden_states = x + y
        else:
            hidden_states = x
        return hidden_states, new_conv_cache

class DCDownBlock2d(nn.Module):
    """WAN 版 2D 下采样残差块。"""
    def __init__(self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        shortcut: bool = True,
        group_norm=False,
        pad_mode="contant",
    ) -> None:
        """构造 WAN 版 2D 下采样残差块。"""
        super().__init__()
        if group_norm:
            self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=32, eps=1e-6, affine=True)
            self.nonlinearity = swish
        else:
            self.norm = nn.Identity()
            self.nonlinearity = nn.Identity()
        self.downsample = downsample
        self.factor = 2
        self.stride = 1 if downsample else 2
        self.group_size = in_channels * self.factor**2 // out_channels
        self.shortcut = shortcut

        out_ratio = self.factor**2
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行 2D 下采样。"""
        x = self.nonlinearity(self.norm(hidden_states))
        x = self.conv(x)
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)

        if self.shortcut:
            y = F.pixel_unshuffle(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states

class DCUpBlock2d(nn.Module):
    """WAN 版 2D 上采样残差块。"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
        group_norm=False,
        pad_mode="constant",
    ) -> None:
        """构造 WAN 版 2D 上采样残差块。"""
        super().__init__()

        if group_norm:
            self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=32, eps=1e-6, affine=True)
            self.nonlinearity = swish
        else:
            self.norm = nn.Identity()
            self.nonlinearity = nn.Identity()
        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor**2 // in_channels

        out_ratio = self.factor**2

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode=pad_mode)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行 2D 上采样，并与捷径分支相加。"""
        x = self.nonlinearity(self.norm(hidden_states))
        if self.interpolate:
            x = F.interpolate(x, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states

class CogVideoXSafeConv3d(nn.Conv3d):
    r"""安全版 3D 卷积。

    当输入激活过大时，沿时间维分块卷积，以避免显存溢出。
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """在必要时按时间块执行卷积，降低单次显存峰值。"""
        memory_count = (
            (input.shape[1] * input.shape[2] * input.shape[3] * input.shape[4]) * 2 / 1024**3
        )

        # 以约 2GB 激活量为阈值切块，兼顾 CuDNN 效率与稳定性。
        if memory_count > 2:
            kernel_size = self.kernel_size[0]
            part_num = int(memory_count / 2) + 1
            input_chunks = torch.chunk(input, part_num, dim=2)

            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)
                    for i in range(1, len(input_chunks))
                ]

            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super().forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super().forward(input)


class CogVideoXCausalConv3d(nn.Module):
    r"""因果 3D 卷积。

    时间维只向左补齐，因此当前输出不会访问未来帧。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        """预计算时空 padding，并构造底层安全卷积。"""
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        self.temporal_dim = 2
        self.time_kernel_size = time_kernel_size

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = CogVideoXSafeConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def fake_context_parallel_forward(
        self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """拼接缓存的历史帧，补足因果卷积所需的上下文。"""
        kernel_size = self.time_kernel_size

        if cp.cp_on():
            conv_cache = dist_conv_cache_recv()

        if kernel_size > 1:
            cached_inputs = [conv_cache.to(inputs.device)] if conv_cache is not None else [inputs[:, :, :1]] * (kernel_size - 1)
            inputs = torch.cat(cached_inputs + [inputs], dim=2)
        return inputs

    def forward(self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        """执行带缓存的因果卷积。"""
        inputs = self.fake_context_parallel_forward(inputs, conv_cache)

        if cp.cp_on():
            dist_conv_cache_send(inputs[:, :, -self.time_kernel_size + 1 :])
        else:
            conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone()


        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
        if self.pad_mode == "constant":
            inputs = F.pad(inputs, padding_2d, mode="constant", value=0)
        else:
            _shape = inputs.shape
            inputs = F.pad(inputs.view(-1, *inputs.shape[-2:]), padding_2d, mode="replicate")
            inputs = inputs.view(*_shape[:-2], *inputs.shape[-2:])

        output = self.conv(inputs)
        return output, conv_cache

class FluxConv(nn.Module):
    """WAN 路径中使用的 2D/3D 分片卷积包装器。"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, cnn_type="2d", cnn_slice_seq_len=17, causal_offset=0, temporal_down=False):
        """构造兼容 2D/3D 的分片卷积包装器。"""
        super().__init__()
        self.cnn_type = cnn_type
        self.slice_seq_len = cnn_slice_seq_len

        if cnn_type == "2d":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if cnn_type == "3d":
            if temporal_down == False:
                stride = (1, stride, stride)
            else:
                stride = (stride, stride, stride)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size, kernel_size)
            self.padding = (
                kernel_size[0] - 1 + causal_offset,  # 时间维因果 padding
                padding,  # 高度方向 padding
                padding  # 宽度方向 padding
            )
        self.causal_offset = causal_offset
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        """根据 `cnn_type` 选择 2D 或 3D 卷积路径。"""
        if self.cnn_type == "2d":
            if type(x) == list:
                for i in range(len(x)):
                    x[i] = self.forward(x[i])
                return x
            if x.ndim == 5:
                B, C, T, H, W = x.shape
                x = rearrange(x, "B C T H W -> (B T) C H W")
                x = self.conv(x)
                x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
                return x
            else:
                return self.conv(x)
        if self.cnn_type == "3d":
            if x.ndim == 5:
                assert self.stride[0] == 1 or self.stride[0] == 2, f"仅支持 temporal stride = 1 或 2"
                if self.stride[0] == 1:
                    for i in reversed(range(0, x.shape[2], self.slice_seq_len+self.stride[0]-1)):
                        st = i
                        en = min(i+self.slice_seq_len, x.shape[2])
                        _x = x[:,:,st:en,:,:]
                        if i == 0:
                            _x = F.pad(_x, (self.padding[2], self.padding[2],  # 宽度维。
                                    self.padding[1], self.padding[1],   # 高度维。
                                    self.padding[0], 0))                # 时间维。
                            _x[:,:,:self.padding[0],
                                self.padding[1]:_x.shape[-2]-self.padding[1],
                                self.padding[2]:_x.shape[-1]-self.padding[2]] = x[:,:,0:1,:,:].clone() # 广播第一个时间值。
                        else:
                            padding_0 = self.kernel_size[0] - 1
                            _x = F.pad(_x, (self.padding[2], self.padding[2],  # 宽度维。
                                    self.padding[1], self.padding[1],   # 高度维。
                                    padding_0, 0))                      # 时间维。
                            _x[:,:,:padding_0,
                                self.padding[1]:_x.shape[-2]-self.padding[1],
                                self.padding[2]:_x.shape[-1]-self.padding[2]] = x[:,:,i-padding_0:i,:,:].clone()
                        try:
                            _x = self.conv(_x)
                        except:
                            xs = [_x[:,:,:,:,i-1:i+2] for i in range(1,_x.shape[-1]-1)]
                            for i in range(len(xs)):
                                xs[i] = self.conv(xs[i])
                            _x = torch.cat(xs, dim=-1)
                        if i == 0:
                            x[:,:,st-self.causal_offset:en,:,:] = _x
                            x = x[:,:,1:,:,:]
                        else:
                            x[:,:,st:en,:,:] = _x
                else:
                    xs = []
                    for i in range(0, x.shape[2], self.slice_seq_len+self.stride[0]-1):
                        st = i
                        en = min(i+self.slice_seq_len, x.shape[2])
                        _x = x[:,:,st:en,:,:]
                        if i == 0:
                            _x = F.pad(_x, (self.padding[2], self.padding[2],  # 宽度维。
                                    self.padding[1], self.padding[1],   # 高度维。
                                    self.padding[0], 0))                # 时间维。
                            _x[:,:,:self.padding[0],
                                self.padding[1]:_x.shape[-2]-self.padding[1],
                                self.padding[2]:_x.shape[-1]-self.padding[2]] = x[:,:,0:1,:,:].clone() # 广播第一个时间值。
                        else:
                            padding_0 = self.kernel_size[0] - 1
                            _x = F.pad(_x, (self.padding[2], self.padding[2],  # 宽度维。
                                    self.padding[1], self.padding[1],   # 高度维。
                                    padding_0, 0))                      # 时间维。
                            _x[:,:,:padding_0,
                                self.padding[1]:_x.shape[-2]-self.padding[1],
                                self.padding[2]:_x.shape[-1]-self.padding[2]] = x[:,:,i-padding_0:i,:,:].clone()
                        _x = self.conv(_x)
                        xs.append(_x)
                    try:
                        x = torch.cat(xs, dim=2)
                    except:
                        device = x.device
                        del x
                        xs = [_x.cpu().pin_memory() for _x in xs]
                        torch.cuda.empty_cache()
                        x = torch.cat([_x for _x in xs], dim=2).to(device=device)
            else:
                x = F.pad(x, (self.padding[2], self.padding[2],  # 宽度维。
                            self.padding[1], self.padding[1]))   # 高度维。
                weight = torch.sum(self.conv.weight, dim=2)
                bias = self.conv.bias
                x = F.conv2d(x, weight=weight, bias=bias,stride=self.conv.stride[1:])
            return x
