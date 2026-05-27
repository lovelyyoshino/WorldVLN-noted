from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_to_192x640(x: torch.Tensor) -> torch.Tensor:
    """把 VAE 解码器特征调整到 TimesFormer 训练使用的 192x640 输入网格。"""
    return F.interpolate(x, size=(192, 640), mode="bilinear", align_corners=False)


class Vae96ToTSformerEmbedAdapter(nn.Module):
    """
        把 InfinityStar 解码器的 `up_block_3` 特征 `(B,96,T,256,256)` 映射成 TSformer patch tokens。

        输出 token 与 TSformer `PatchEmbed` 的输出保持一致：
          - 输出 `patch_tokens`：形状 `(B*T, N=12*40=480, D=384)`
          - 返回 `(patch_tokens, T, W_grid=40)`

        形状公式：
          公式/形状说明：(B,96,T,H,W) -> (B*T,96,H,W) -> (B*T,D,H_patch,W_patch) -> (B*T,N,D)。
          公式/形状说明：`H_patch = 192 / patch_size`，`W_patch = 640 / patch_size`，
          `N = H_patch * W_patch`；每个 token 对应调整尺寸后图像网格上的一个 patch。

        中文导读：
        这是动作解码器中最关键的“翻译层”。世界模型给出的 latent 先经过 VAE 解码器，
        但代码不会使用最终 RGB 输出，而是在解码器中间层取 96 通道时空特征。Adapter
        把这些 VAE 特征重排成 TimesFormer 能直接消费的 patch tokens，使后续动作头可以
        从预测到的潜在世界状态转移中恢复 6D 动作。

    """

    def __init__(self, embed_dim: int = 384, patch_size: int = 16, use_skip: bool = False):
        """构造 Adapter：96 通道 VAE 特征 -> 384 维、16x16 patch token。"""
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.use_skip = bool(use_skip)

        self.conv_a = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
        )
        self.patch = nn.Conv2d(128, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.skip = nn.Conv2d(96, self.embed_dim, kernel_size=1, padding=0, bias=False) if self.use_skip else None
        self.out_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, f96_up3: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        将一段 VAE 解码器特征 clip 转成 TimesFormer patch tokens。

        中文导读：
        输入的时间维 `T` 保留下来，但批量维和时间维会临时合并成 `B*T` 做 2D patch 化；
        输出 token 再交给 TimesFormer 的 `forward_features_from_patch_tokens()`。

        token 网格含义：192x640 调整尺寸后用 16x16 patch 时，网格是 12x40，
        所以 `N = 12 * 40 = 480` 个空间 token。
        """
        if f96_up3.ndim != 5:
            raise ValueError(f"期望 f96_up3 形状为 (B,96,T,H,W)，实际收到 {tuple(f96_up3.shape)}")

        bsz, channels, num_frames, height, width = f96_up3.shape
        if int(channels) != 96:
            raise ValueError(f"期望通道数 channel=96，实际 C={channels}")

        # 先把时间维并入批量维：每一帧单独做 2D patch，形状 (B,96,T,H,W) -> (B*T,96,H,W)。
        x = f96_up3.permute(0, 2, 1, 3, 4).reshape(bsz * num_frames, channels, height, width)
        x = _resize_to_192x640(x)

        hidden = self.conv_a(x)
        # patch conv 后得到空间网格 (H_patch,W_patch)=(12,40)，总 token 数 N=H_patch*W_patch=480。
        hidden = self.patch(hidden)

        if self.skip is not None:
            skip = self.skip(x)
            skip = F.avg_pool2d(skip, kernel_size=self.patch_size, stride=self.patch_size)
            hidden = hidden + 0.1 * skip

        # flatten(2) 把 12x40 网格拉平成 N=480，再转成 TimesFormer 需要的 (B*T,N,D)。
        tokens = hidden.flatten(2).transpose(1, 2).contiguous()
        tokens = self.out_norm(tokens)
        grid_w = int(hidden.shape[-1])
        return tokens, int(num_frames), grid_w
