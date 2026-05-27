# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
有限标量量化（Finite Scalar Quantization, FSQ）。

论文见 https://arxiv.org/abs/2309.15505。

FSQ 不显式维护一个可学习码本，而是把每个维度直接量化到有限个标量等级。
因此 `codes <-> indices` 的转换本质上是“离散等级值”和“整型等级索引”的互转，
非常适合给初学者理解“离散化不一定需要查表 codebook”这一点。
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.cuda.amp import autocast

from einops import rearrange, pack, unpack

# 辅助函数。

def exists(v):
    """返回值是否不是 `None`。"""
    return v is not None

def default(*args):
    """返回第一个非空默认值。"""
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    """对单个张量执行 `einops.pack`。"""
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    """把 `pack_one` 的结果按原始形状还原。"""
    return unpack(t, ps, pattern)[0]

# 张量辅助函数。

def round_ste(z: Tensor) -> Tensor:
    """带 straight-through estimator（直通估计器）的四舍五入。

        前向等价于 `round(z)`，反向近似恒等映射：
        公式/形状说明：`z + (round(z) - z).detach()`。

    """
    zhat = z.round()
    return z + (zhat - z).detach()

# 主类定义。

class FSQ(Module):
    """FSQ 量化器。

    每个维度只允许落在有限个等级上，例如 `L=5` 时是
    `{-1, -1/2, 0, 1/2, 1}`。与 VQ 不同，这里没有显式查表的 codebook，
    而是通过逐维离散等级来得到 `indices -> codes` 的映射。
    """

    def __init__(
        self,
        num_lvl: int,
        # 中文说明：levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None
    ):
        """根据量化等级数与特征维初始化 FSQ。"""
        super().__init__()
        levels = [num_lvl] * dim
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        # 代码/形状说明：_basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        # 代码/形状说明：self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        # 代码/形状说明：self.codebook_size = self._levels.prod().item()

        # 代码/形状说明：implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out = False)
        # 代码/形状说明：self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """把连续输入平滑约束到可量化范围内，避免边界处梯度过硬。"""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """将输入量化到有限等级，并归一化回 `[-1, 1]` 区间。"""
        quantized = round_ste(self.bound(z)) # L=5 时是 -2, -1, 0, 1, 2。
        half_width = self._levels // 2 # 代码/形状说明：重新归一化到 [-1, 1]；L=5 时 half_width = 2。
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        """把归一化等级 `[-1, 1]` 平移缩放为非负整数等级。"""
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        """把非负整数等级恢复为归一化后的连续码值。"""
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """把逐维离散码值转换成整型等级索引。"""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat) # {-1, -1/2, 0, 1/2, 1} -> {-2, -1, 0, 1, 2} -> {0, 1, 2, 3, 4}
        # 代码/形状说明：return (zhat * self._basis).sum(dim=-1).to(int32)
        return zhat.to(int32)

    def indices_to_codes(
        self,
        indices: Tensor,
        project_out = True,
        **kwargs,
    ) -> Tensor:
        """把整型等级索引恢复成连续码值，可选投影回原特征维。"""

        # 代码/形状说明：is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        # 代码/形状说明：if is_img_or_video:
        # 代码/形状说明：indices = rearrange(indices, 'b d ... -> b ... d')

        # 代码/形状说明：indices = rearrange(indices, '... -> ... 1')
        # 代码/形状说明：codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(indices) # {0, 1, 2, 3, 4} -> {-1, -1/2, 0, 1/2, 1}

        # 代码/形状说明：if self.keep_num_codebooks_dim:
        # 代码/形状说明：codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        # 代码/形状说明：if is_img_or_video:
        codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    @autocast(enabled = False)
    def forward(self, z: Tensor) -> Tensor:
        """
        FSQ 前向过程。

        记号说明：

        - `b`：batch 维度。
        - `n`: 序列长度，或展平后的时空位置数
        - `d`: 特征维
        - `c`: codebook 维度数

        输入先标准化成 `(batch, seq, dim)`，再逐维量化为有限等级，
        输出 `(quantized, None, indices, None)` 以兼容项目里其他量化器接口。
        """

        is_img_or_video = z.ndim >= 4

        # 中文说明：把图像/视频统一整理成 (batch, seq, dimension)。

        if is_img_or_video:
            z = rearrange(z, 'b d ... -> b ... d') # (b, c, t, h, w) -> (b, t, h, w, c)
        # 代码/形状说明：z, ps = pack_one(z, 'b * d') # (b, thw, c), (t, h, w)

        assert z.shape[-1] == self.dim, f'期望最后一维 dimension 为 {self.dim}，但实际为 {z.shape[-1]}'

        z = self.project_in(z)

        # 代码/形状说明：z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks) # (b, thw, 1, c)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        # 代码/形状说明：codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # 还原图像/视频维度。

        if is_img_or_video:
        # 代码/形状说明：out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')
            # 代码/形状说明：indices = rearrange(indices, 'b ... d -> b d ...')


        # 代码/形状说明：# indices = unpack_one(indices, ps, 'b * c')

        # 代码/形状说明：if not self.keep_num_codebooks_dim:
        # 代码/形状说明：# indices = rearrange(indices, '... 1 -> ...')
        # 中文说明：pass

        return out, None, indices, None


if __name__ == "__main__":
    num_lvl = 5
    dim = 16
    T, H, W = 21, 32, 32
    quantizer = FSQ(num_lvl, dim)
    z = torch.randn(2, dim, T, H, W)
    out, indices = quantizer(z)
