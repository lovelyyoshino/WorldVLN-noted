# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
二值球面量化（Binary Spherical Quantization, BSQ）。

论文见 https://arxiv.org/abs/2406.07548。

这个实现把每个潜变量维度离散到 {-1, 1}，再乘以 `1 / sqrt(d)` 投到单位球面附近。
训练时使用 straight-through estimator（STE）近似离散化梯度，额外用熵正则约束码字使用情况：

- 单样本熵希望更低，让每个位置的二值决策更确定。
- 批量平均熵希望更高，让不同 bit / code 在 batch 内被更均匀地使用。
"""

import random
import copy
from math import log2, ceil
from functools import partial, cache
from collections import namedtuple
from contextlib import nullcontext

import torch.distributed as dist
from torch.distributed import nn as dist_nn

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module
from torch.amp import autocast
import numpy as np

from einops import rearrange, reduce, pack, unpack

# 中文标题：from einx import get_at

# 代码/形状说明：print(f"{dynamic_resolution_thw=}")

# 常量区。

Return = namedtuple('Return', ['quantized', 'indices', 'bit_indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# 分布式辅助函数。

@cache
def is_distributed():
    """判断当前进程是否处于已初始化的分布式训练环境。"""
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    """在分布式场景对张量做 all-reduce 后取均值，单卡时直接原样返回。"""
    if not is_distributed():
        return t

    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t

# 辅助函数。

def exists(v):
    """返回值是否不是 `None`。"""
    return v is not None

def identity(t):
    """恒等映射，常用作可选模块的默认占位。"""
    return t

def default(*args):
    """返回第一个有效参数；若该参数可调用，则先调用再返回。"""
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def round_up_multiple(num, mult):
    """把 `num` 向上取整到 `mult` 的整数倍，即 `ceil(num / mult) * mult`。"""
    return ceil(num / mult) * mult

def pack_one(t, pattern):
    """对单个张量执行 `einops.pack`，便于把时空维打平成一维序列。"""
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    """把 `pack_one` 打平后的张量按记录的 shape 信息还原。"""
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    """沿最后一维做 L2 归一化，使向量落到单位球面上。"""
    return F.normalize(t, dim = -1)

# 熵相关计算。

def log(t, eps = 1e-5):
    """带下界裁剪的对数，避免 `log(0)` 导致数值溢出。"""
    return t.clamp(min = eps).log()

def entropy(prob):
    """计算离散分布熵 `H(p) = -sum(p * log(p))`。"""
    return (-prob * log(prob)).sum(dim=-1)

# 余弦相似度线性层。

class CosineSimLinear(Module):
    """先归一化输入与权重，再用 cosine similarity 做线性投影。"""

    def __init__(
        self,
        dim_in,
        dim_out,
        scale = 1.
    ):
        """初始化余弦相似度投影层。"""
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        """输出 `normalize(x) @ normalize(W)`，再乘以可选缩放系数。"""
        x = F.normalize(x, dim = -1)
        w = F.normalize(self.weight, dim = 0)
        return (x @ w) * self.scale

def repeat_schedule(scale_schedule, repeat_scales_num, times):
    """把前若干个尺度重复多次，构造更密集的多尺度量化日程。"""
    new_scale_schedule = []
    for i in range(repeat_scales_num):
        new_scale_schedule.extend([scale_schedule[i] for _ in range(times)])
    new_scale_schedule.extend(scale_schedule[repeat_scales_num:])
    return new_scale_schedule


class BSQ(Module):
    """BSQ 量化器。

    关键数据流如下：

    1. 输入特征先按 codebook 维重排。
    2. 每一维根据符号量化为 bit：`z > 0 -> 1`，否则为 `0 / -1`。
    3. bit 通过 `bits_to_codes` 变成码向量，再做球面归一化。
    4. 反向传播时使用 STE：`z + (q - z).detach()`，前向离散、反向近似恒等。

    这里 `indices`、`bit_indices`、`codes` 的关系是：

    - `bit_indices`: 每个离散维上的二值标签。
    - `indices`: 把 bit 按位权 `mask` 打包后的整数索引。
    - `codes`: 把 bit 映射回 `{-scale, +scale}` 后得到的连续码向量。
    """

    def __init__(
        self,
        *,
        dim = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 0.25,
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,                        # 代码/形状说明：用于 residual LFQ，每层 codebook 缩小 2x
        frac_per_sample_entropy = 1.,               # 中文说明：小于 1 时，只随机使用一部分概率来计算逐样本熵（per sample entropy）
        soft_clamp_input_value = None,
        channel_first = None,
        experimental_softplus_entropy_loss = False,
        entropy_loss_offset = 5.,                   # 中文标题：softplus 前对损失（loss）的平移量
        spherical = True,                          # 代码/形状说明：参考 https://arxiv.org/abs/2406.07548
        force_quantization_f32 = True,               # 中文标题：强制量化步骤使用全精度（full precision）
        inv_temperature = 100.0,
        gamma0=1.0, gamma=1.0, zeta=1.0,
        use_out_phi = False, # 中文标题：使用输出 phi 网络
        use_out_phi_res = False, # 中文标题：残差式 out phi
        use_bernoulli = False,
        use_rot_trick = False,
    ):
        """初始化 BSQ 量化器及熵/承诺损失相关超参数。"""
        super().__init__()

        # 若干 assert 校验。
        assert exists(dim) , '必须为 BSQ 指定 dim'

        codebook_dim = dim
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        self.codebook_dims = codebook_dims

        self.out_phi = nn.Linear(codebook_dims, codebook_dims) if use_out_phi else nn.Identity()
        self.use_out_phi_res = use_out_phi_res
        if self.use_out_phi_res:
            self.out_phi_scale = nn.Parameter(torch.zeros(codebook_dims), requires_grad=True) # 初始化为 0。

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # 通道维在前。
        self.channel_first = channel_first

        # 中文说明：对于 BSQ（binary spherical quantization）
        if not spherical:
            raise ValueError("对于 BSQ，spherical 必须为 True。")
        self.persample_entropy_compute = 'analytical'
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0  # 中文标题：熵惩罚（entropy penalty）的损失（loss）权重
        self.gamma = gamma  # 中文标题：熵惩罚（entropy penalty）的损失（loss）权重
        self.zeta = zeta    # 中文标题：整体熵惩罚（entropy penalty）的损失（loss）权重
        self.use_bernoulli = use_bernoulli
        self.use_rot_trick = use_rot_trick

        # 中文标题：熵辅助损失（entropy aux loss）相关权重

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.entropy_loss_weight = entropy_loss_weight

        # 中文标题：codebook 缩放

        self.codebook_scale = codebook_scale

        # commitment 损失。

        self.commitment_loss_weight = commitment_loss_weight

        # 中文标题：是否把输入值软裁剪（soft clamp）到 [-value, value]

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # 中文说明：是否通过 softplus 让熵损失（entropy loss）为正（实验性选项）

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # 代码/形状说明：推理时没有辅助损失（auxiliary loss）

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # 中文说明：是否强制量化步骤使用 f32

        self.force_quantization_f32 = force_quantization_f32

    def bits_to_codes(self, bits):
        """把 bit `{0, 1}` 映射到码值 `{-scale, +scale}`。"""
        return bits * self.codebook_scale * 2 - self.codebook_scale

    # 中文说明：@property
    # 代码/形状说明：def dtype(self):
    # 代码/形状说明：return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        label_type = 'int_label',
        project_out = True
    ):
        """把整数索引或 bit 标签还原成连续码向量。

        `int_label` 路径会先按位与 `mask` 解包成 bit，再映射成 `{-scale, +scale}`。
        对 BSQ 来说，得到的 codes 还要做一次 L2 归一化，保证它们位于单位球面。
        """
        assert label_type in ['int_label', 'bit_label']
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            if label_type == 'int_label':
                indices = rearrange(indices, '... -> ... 1')
            else:
                indices = indices.unsqueeze(-2)

        # 中文说明：indices 转 codes，codes 是 -1 或 1 的 bit

        if label_type == 'int_label':
            assert indices[..., None].int().min() > 0
            bits = ((indices[..., None].int() & self.mask) != 0).float() # 中文说明：.to(self.dtype)
        else:
            bits = indices

        codes = self.bits_to_codes(bits).float()

        codes = l2norm(codes) # 中文标题：使用 BSQ 时必须归一化

        codes = rearrange(codes, '... c d -> ... (c d)')

        # 中文标题：是否把 codes 投影回原始维度
        # 代码/形状说明：如果输入特征维度不是 log2(codebook size)

        # 中文标题：把 codes 重排回原始 shape

        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def quantize(self, z):
        """把输入按符号量化到单位球面，并用 STE 保留梯度。"""
        assert z.shape[-1] == self.codebook_dims, f"期望 {self.codebook_dims} 维，实际 {z.shape[-1]} 维"

        zhat = torch.where(z > 0,
                           torch.tensor(1, dtype=z.dtype, device=z.device),
                           torch.tensor(-1, dtype=z.dtype, device=z.device))

        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat # 中文标题：投到单位球面（unit sphere）上

        return z + (zhat - z).detach()

    def quantize_new_bernoulli(self, z, prob_z):
        """按给定 Bernoulli 概率采样 bit，再映射成球面上的二值码。"""
        assert z.shape[-1] == self.codebook_dims, f"期望 {self.codebook_dims} 维，实际 {z.shape[-1]} 维"

        zhat = (torch.bernoulli(prob_z) - 0.5) * 2.0

        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat # 中文标题：投到单位球面（unit sphere）上

        return z + (zhat - z).detach()

    def rot_quantize(self, z, inference=False):
        """使用 rotation trick 的 BSQ 量化。

        推理时直接返回离散码；训练时构造旋转后的代理梯度，
        让梯度方向更贴近量化前后的几何关系。
        """
        assert z.shape[-1] == self.codebook_dims, f"期望 {self.codebook_dims} 维，实际 {z.shape[-1]} 维"
        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = torch.where(z > 0,
                            torch.tensor(1, dtype=z.dtype, device=z.device),
                            torch.tensor(-1, dtype=z.dtype, device=z.device)) * q_scale
        if inference:
            return zhat

        w = ((z + zhat) / torch.norm(z + zhat, dim=-1, keepdim=True)).detach()
        z = z.unsqueeze(1) - 2*torch.bmm(torch.bmm(z.unsqueeze(1), w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
            torch.bmm(z.unsqueeze(1), z.unsqueeze(-1).detach()), zhat.unsqueeze(1).detach())
        return z.squeeze()

    def soft_entropy_loss(self, z):
        """解析近似二值分布的熵。

        先用 sigmoid 估计每个 bit 取正/负的概率，再计算：

        - 单样本熵：`-sum(p * log(p))`，希望更小。
        - 平均码本熵：对 batch 平均概率再求熵，希望更大。
        """
        if self.persample_entropy_compute == 'analytical':
            # 代码/形状说明：if self.l2_norm:
            p = torch.sigmoid(-4 * z / (self.codebook_dims ** 0.5) * self.inv_temperature)
            # 代码/形状说明：else:
            # 代码/形状说明：p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1-p], dim=-1) # (b, h, w, 18, 2)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean() # 代码/形状说明：(b,h,w,18)->(b,h,w)->scalar
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        # 中文标题：每个子组（subgroup）概率的宏平均
        avg_prob = reduce(prob, '... g d ->g d', 'mean') # (18, 2)
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        # 中文标题：熵近似为各子组（subgroup）熵之和
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        """根据计数或概率计算熵；`normalize=True` 时先归一化为概率。"""
        if normalize: # 中文说明：False
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim =True)
        else: # 中文说明：True
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def forward(
        self,
        x,
        return_loss_breakdown = False,
        mask = None,
        entropy_weight=0.1
    ):
        """
        BSQ 前向过程。

        记号说明：

        - `b`：批量（batch）维度。
        - `n`: 序列长度，或展平后的时空位置数
        - `d`: 特征维，也是单个 codebook 的 bit 数
        - `c`: codebook 个数

        主要步骤：

        1. 把 `(b, d, ...)` 规范化为 `(b, n, c, d)`。
        2. 对每个 `d` 维向量做 L2 归一化。
        3. 执行 BSQ 离散化，得到 bit 与量化码。
        4. 根据 bit 通过位权求整数 `indices`，并计算熵正则与 commitment loss。
        """

        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # 中文说明：将图像或视频标准化为 (batch, seq, dimension)

        if should_transpose:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d') # 代码/形状说明：x.shape [b, hwt, c]

        assert x.shape[-1] == self.dim, f'期望维度为 {self.dim}，实际收到 {x.shape[-1]}'

        # 中文标题：拆出 codebook 个数维度

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        if self.use_bernoulli:
            prob_x = torch.sigmoid(x)

        x = l2norm(x)

        # 中文标题：是否强制量化步骤使用全精度（full precision）

        force_f32 = self.force_quantization_f32

        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            # 中文标题：使用直通梯度（straight-through gradients）
            if self.use_rot_trick:
                x_f = x.flatten(end_dim=-2) # 代码/形状说明：(b, hwt, 1, d) -> (bhwt, d)
                q_f = self.rot_quantize(x_f, inference= not self.training)
                quantized = q_f.reshape(x.shape)
            elif self.use_bernoulli:
                quantized = self.quantize_new_bernoulli(x, prob_x)
            else:
                quantized = self.quantize(x)

            # 中文标题：计算 indices
            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')
            bit_indices = (quantized > 0).int()

            # 中文标题：熵辅助损失（entropy aux loss）
            if self.training:
                persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(x) # 中文标题：计算熵（entropy）
                entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
            else:
                # 代码/形状说明：如果不是训练模式，只返回占位 0
                entropy_penalty = persample_entropy = cb_entropy = self.zero

            # 中文标题：commit 损失

            if self.training and self.commitment_loss_weight > 0.:

                commit_loss = F.mse_loss(x, quantized.detach(), reduction = 'none')

                if exists(mask):
                    commit_loss = commit_loss[mask]

                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            # 中文标题：必要时把输入转回原始 dtype

            if force_f32:
                x = x.type(orig_dtype)

        # 中文标题：合并回 codebook 维度
        x = quantized # 中文标题：将 quantized 重命名为 x 作为输出

        if self.use_out_phi_res:
            x = x + self.out_phi_scale * self.out_phi(x) # 中文标题：把 out_phi 作为残差应用到量化输出
        else:
            x = self.out_phi(x) # 中文标题：把 out_phi 应用到量化输出

        x = rearrange(x, 'b n c d -> b n (c d)')

        # 中文标题：还原 image/video 维度

        if should_transpose:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            bit_indices = unpack_one(bit_indices, ps, 'b * c d')

        # 中文标题：是否移除单 codebook 维度

        if not self.keep_num_codebooks_dim:
            bit_indices = rearrange(bit_indices, '... 1 d -> ... d')

        # 中文标题：完整辅助损失（aux loss）

        aux_loss = commit_loss * self.commitment_loss_weight + (self.zeta * entropy_penalty / self.inv_temperature)*entropy_weight
        # 中文说明：返回 Return(x, indices, bit_indices, aux_loss)，分别是量化输出、整数索引、bit 索引和辅助损失。

        ret = Return(x, indices, bit_indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(persample_entropy, cb_entropy, commit_loss)
