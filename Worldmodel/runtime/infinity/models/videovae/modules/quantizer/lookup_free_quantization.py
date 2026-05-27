# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
免查表量化（Lookup-Free Quantization, LFQ）。

论文见 https://arxiv.org/abs/2310.05737。

LFQ 用 bit 组合来隐式表示码本，而不是显式维护 embedding 表：

- 每个维度先量化到 `{-1, 1}`。
- 多个 bit 按位权 `mask` 打包为整数 `indices`。
- `indices -> codes` 时再把整数按位展开回 bit，并映射成连续码向量。

训练时同样引入熵正则，兼顾“单位置预测足够确定”和“batch 内码本使用足够均匀”。
"""

from math import log2, ceil
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module
from torch.cuda.amp import autocast

from einops import rearrange, reduce, pack, unpack

# 常量区。

Return = namedtuple('Return', ['quantized', 'indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# 辅助函数。

def exists(v):
    """返回值是否不是 `None`。"""
    return v is not None

def default(*args):
    """返回第一个有效参数；如果该参数可调用则先执行。"""
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def pack_one(t, pattern):
    """对单个张量执行 `einops.pack`。"""
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    """把 `pack_one` 的输出按原始形状还原。"""
    return unpack(t, ps, pattern)[0]

# 熵相关计算。

def log(t, eps = 1e-5):
    """带数值保护的对数。"""
    return t.clamp(min = eps).log()

def entropy(prob):
    """计算熵 `H(p) = -sum(p * log(p))`。"""
    return (-prob * log(prob)).sum(dim=-1)

# 类定义。

class LFQ(Module):
    """LFQ 量化器。

    初学者可以把它理解成“用 bit 串编码 codebook”：

    - `bits_to_codes` 负责 `0/1 -> -scale/+scale`；
    - `indices_to_codes` 负责 `整数索引 -> bit -> 连续码`；
    - `forward` 则把连续特征量化、打包整数索引，并计算熵辅助损失。
    """

    def __init__(
        self,
        *,
        dim = None,
        codebook_size = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 0.25,
        diversity_gamma = 1.,
        straight_through_activation = nn.Identity(),
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,            # 代码/形状说明：residual LFQ 中每层 codebook 按 2 倍逐级缩小。
        frac_per_sample_entropy = 1.    # 中文说明：设为小于 1 时，只随机抽取部分概率来估计 per-sample entropy。
    ):
        """初始化 LFQ 所需的隐式 bit-codebook 与损失权重。"""
        super().__init__()

        # 若干 assert 校验。

        assert exists(dim) or exists(codebook_size), 'LFQ 必须指定 dim 或 codebook_size'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'LFQ 要求 codebook_size 是 2 的幂，建议值为 {2 ** ceil(log2(codebook_size))}'

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        codebook_dim = int(log2(codebook_size))

        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.project_in = nn.Linear(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # straight-through activation（直通激活）。

        self.activation = straight_through_activation

        # 熵辅助损失相关权重。

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook 缩放系数。

        self.codebook_scale = codebook_scale

        # commitment 损失。

        self.commitment_loss_weight = commitment_loss_weight

        # 推理阶段不计算辅助损失时使用。

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # 码本 codes。

        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.register_buffer('codebook', codebook, persistent = False)

    def bits_to_codes(self, bits):
        """把 bit `{0, 1}` 映射到 `{-scale, +scale}`。"""
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        """返回隐式码本张量的数据类型。"""
        return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        project_out = True
    ):
        """把整数索引解包成 bit，再还原为连续码向量。"""
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... -> ... 1')

        # 中文说明：把 indices 展开为 bit，再映射成 -scale/+scale 的 codes。

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = rearrange(codes, '... c d -> ... (c d)')

        # 如果输入特征维不等于 log2(codebook_size)，就把 codes 投影回原始维度。

        if project_out:
            codes = self.project_out(codes)

        # 把 codes 排回原始形状。

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    @autocast(enabled = False)
    def forward(
        self,
        x,
        inv_temperature = 100.,
        mask = None,
    ):
        """
        LFQ 前向过程。

        记号说明：

        - `b`：batch 维度。
        - `n`: 序列长度，或展平后的时空位置数
        - `d`: 特征维，也等于 `log2(codebook_size)`
        - `c`: codebook 个数

        训练阶段的熵辅助项分成两部分：

        - `per_sample_entropy`：让单个位置的 bit 预测更确定。
        - `codebook_entropy`：让 batch 平均分布更接近均匀，提升 codebook usage。
        """
        x = x.float()

        is_img_or_video = x.ndim >= 4

        # 中文说明：把图像/视频统一整理成 (batch, seq, dimension)。

        if is_img_or_video:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim, f'期望最后一维 dimension 为 {self.dim}，但收到 {x.shape[-1]}'

        x = self.project_in(x)

        # 拆出 num_codebooks 维。

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # 中文说明：按论文公式 3 量化。

        original_input = x

        codebook_value = torch.ones_like(x) * self.codebook_scale
        quantized = torch.where(x > 0, codebook_value, -codebook_value)

        # 中文说明：训练时使用 straight-through 梯度，可叠加自定义 activation。

        if self.training:
            x = self.activation(x)
            x = x + (quantized - x).detach()
        else:
            x = quantized

        # 计算 indices。

        indices = reduce((x > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

        # 熵辅助损失。

        if self.training:
            # 除了常数项之外，这等价于欧氏距离。
            distance = -2 * einsum('... i d, j d -> ... i j', original_input, self.codebook)

            prob = (-distance * inv_temperature).softmax(dim = -1)

            # 应用 mask。

            if exists(mask):
                prob = prob[mask]
            else:
                prob = rearrange(prob, 'b n ... -> (b n) ...')

            # 中文说明：是否只抽取一部分概率，以减少显存占用。

            if self.frac_per_sample_entropy < 1.:
                num_tokens = prob.shape[0]
                num_sampled_tokens = int(num_tokens * self.frac_per_sample_entropy)
                rand_mask = torch.randn(num_tokens).argsort(dim = -1) < num_sampled_tokens
                per_sample_probs = prob[rand_mask]
            else:
                per_sample_probs = prob

            # 计算 per-sample entropy。

            per_sample_entropy = entropy(per_sample_probs).mean()

            # 统计 batch 中所有可用 token 的平均分布。

            avg_prob = reduce(per_sample_probs, '... c d -> c d', 'mean')
            codebook_entropy = entropy(avg_prob).mean()

            # 中文说明：1. 降低每个位置的 entropy，鼓励网络输出更确定的 code。
            # 中文说明：2. 提高 codebook entropy，鼓励 batch 内所有 code 更均匀地被使用。

            entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
        else:
            # 非训练阶段返回占位 0。
            entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

        # commitment 损失。

        if self.training:
            commit_loss = F.mse_loss(original_input, quantized.detach(), reduction = 'none')

            if exists(mask):
                commit_loss = commit_loss[mask]

            commit_loss = commit_loss.mean()
        else:
            commit_loss = self.zero

        # 合并回 codebook 维。

        x = rearrange(x, 'b n c d -> b n (c d)')

        # 需要时投影回特征维。

        x = self.project_out(x)

        # 还原图像/视频维度。

        if is_img_or_video:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        # 只有一个 codebook 时去掉该维度。

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        # 合成最终辅助损失。
        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

        # 代码/形状说明：ret = Return(x, indices, aux_loss)
        # 中文说明：返回 ret。
        # 代码/形状说明：return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)

        return dict(embeddings=x, encodings=indices, commitment_loss=aux_loss)
