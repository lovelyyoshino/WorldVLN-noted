# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
多尺度二值球面量化（多尺度 BSQ）。

这里把 BSQ 放进残差金字塔量化流程中：先在粗尺度量化，再逐步细化残差。
每一级量化器都输出 bit / index / code，最后通过上采样把各级码叠加回完整时空分辨率。

可把核心流程记成：

步骤说明：1. `residual_0 = x`
2. 第 `s` 层在尺度 `scale_s` 上量化 `residual_s`
3. 上采样回原分辨率后更新 `residual_{s+1} = residual_s - stopgrad(q_s)`
4. 所有 `q_s` 累加得到最终量化输出
"""

import random
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

from .dynamic_resolution import predefined_HW_Scales_dynamic

# 常量区。

Return = namedtuple('Return', ['quantized', 'indices', 'bit_indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# 分布式辅助函数。

@cache
def is_distributed():
    """判断当前进程是否处于已初始化的分布式训练环境。"""
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    """分布式下对张量做 all-reduce 均值，单卡时直接返回。"""
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
    """恒等映射，占位用。"""
    return t

def default(*args):
    """返回第一个有效参数；若其可调用则先执行。"""
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def round_up_multiple(num, mult):
    """把整数向上补齐到指定倍数。"""
    return ceil(num / mult) * mult

def pack_one(t, pattern):
    """对单个张量执行 `einops.pack`。"""
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    """把 `pack_one` 打平后的张量还原回原始形状。"""
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    """沿最后一维做 L2 归一化。"""
    return F.normalize(t, dim = -1)

# 熵相关计算。

def log(t, eps = 1e-5):
    """带下界保护的对数。"""
    return t.clamp(min = eps).log()

def entropy(prob):
    """计算熵 `H(p) = -sum(p * log(p))`。"""
    return (-prob * log(prob)).sum(dim=-1)

# 余弦相似度线性层。

class CosineSimLinear(Module):
    """基于余弦相似度的线性层。"""

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
        """归一化输入和权重后做矩阵乘法。"""
        x = F.normalize(x, dim = -1)
        w = F.normalize(self.weight, dim = 0)
        return (x @ w) * self.scale


def get_latent2scale_schedule(T: int, H: int, W: int, mode="original"):
    """根据时空分辨率生成多尺度量化日程。

    返回值是 `[(t_1, h_1, w_1), ..., (t_s, h_s, w_s)]`。
    这个日程决定每一层量化器看到的时空窗口大小，也决定残差是如何逐级细化的。
    """
    assert mode in ["original", "dynamic", "dense", "same1", "same2", "same3"], f"量化 scale schedule mode 不支持：{mode}"
    predefined_HW_Scales = {
        # 256 * 256
        (32, 32): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 9), (13, 13), (18, 18), (24, 24), (32, 32)],
        (16, 16): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)],
        # 1024x1024
        (64, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (9, 9), (12, 12), (16, 16), (21, 21), (27, 27), (36, 36), (48, 48), (64, 64)],

        (36, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 12), (13, 16), (18, 24), (24, 32), (32, 48), (36, 64)],
    }
    if mode == "dynamic":
        predefined_HW_Scales.update(predefined_HW_Scales_dynamic)
    elif mode == "dense":
        predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(32, 32)] = predefined_HW_Scales[(16, 16)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (56, 56), (64, 64)]
    elif mode.startswith("same"):
        num_quant = int(mode[len("same"):])
        predefined_HW_Scales[(16, 16)] = [(16, 16) for _ in range(num_quant)]
        predefined_HW_Scales[(32, 32)] = [(32, 32) for _ in range(num_quant)]
        predefined_HW_Scales[(64, 64)] = [(64, 64) for _ in range(num_quant)]

    predefined_T_Scales = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 17, 17, 17, 17]
    patch_THW_shape_per_scale = predefined_HW_Scales[(H, W)]
    if len(predefined_T_Scales) < len(patch_THW_shape_per_scale):
        # 代码/形状说明：print("警告：predefined_T_Scales 的长度小于 patch_THW_shape_per_scale 的长度！")
        predefined_T_Scales += [predefined_T_Scales[-1]] * (len(patch_THW_shape_per_scale) - len(predefined_T_Scales))
    patch_THW_shape_per_scale = [(min(T, t), h, w ) for (h, w), t in zip(patch_THW_shape_per_scale, predefined_T_Scales[:len(patch_THW_shape_per_scale)])]
    return patch_THW_shape_per_scale

class LayerNorm(nn.Module):
    r"""支持 `channels_last` 与 `channels_first` 的 LayerNorm。

    - `channels_last` 对应 `(batch, ..., channel)`
    - `channels_first` 对应 `(batch, channel, ...)`

    这个封装主要用于量化前对特征做稳定化处理，降低不同尺度之间数值分布漂移。
    """
    def __init__(self, normalized_shape, norm_weight=False, eps=1e-6, data_format="channels_first"):
        """初始化适配不同数据格式的 LayerNorm。"""
        super().__init__()
        if norm_weight:
            self.weight = nn.Parameter(torch.ones(normalized_shape)/(normalized_shape**0.5))
        else:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        """根据数据排布选择标准 LayerNorm 或手写 channels-first 版本。"""
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if x.ndim == 4: # (b, c, h, w)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif x.ndim == 5: # (b, c, t, h, w)
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            else:
                raise ValueError("输入维数应为 4 或 5")
            return x

class MultiScaleBSQ(Module):
    """多尺度 BSQ 残差量化器。

    实现思想接近残差量化：粗尺度先编码全局结构，细尺度继续编码剩余误差。
    每个尺度都会产出一组 `indices`，这些索引可以通过 `get_codes_from_indices` 重建连续 codes。
    """

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        soft_clamp_input_value = None,
        aux_loss = False, # 中文标题：中间辅助损失
        ln_before_quant=False, # 中文标题：在多尺度 RQ 前添加 LN
        ln_init_by_sqrt=False, # 中文说明：按 1/sqrt(d) 初始化权重
        use_decay_factor=False,
        use_stochastic_depth=False,
        drop_rate=0.,
        schedule_mode="original", # 代码/形状说明：["original", "dynamic", "dense"]
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach=False,
        random_flip = False,
        flip_prob = 0.5,
        flip_mode = "stochastic", # 代码/形状说明："stochastic", "deterministic"
        max_flip_lvl = 1,
        random_flip_1lvl = False, # 中文标题：每次随机翻转一个层级
        flip_lvl_idx = None,
        drop_when_test=False,
        drop_lvl_idx=None,
        drop_lvl_num=0,
        casual_multi_scale=False,
        **kwargs
    ):
        """初始化多尺度 BSQ 调度、随机丢层、随机翻转等策略。"""
        super().__init__()
        codebook_dim = int(log2(codebook_size))

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.layernorm = LayerNorm(codebook_dim, norm_weight=ln_init_by_sqrt) if ln_before_quant else nn.Identity()
        self.use_stochastic_depth = use_stochastic_depth
        self.drop_rate = drop_rate
        self.remove_residual_detach = remove_residual_detach
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.flip_mode = flip_mode
        self.max_flip_lvl = max_flip_lvl
        self.random_flip_1lvl = random_flip_1lvl
        self.flip_lvl_idx = flip_lvl_idx
        assert (random_flip and random_flip_1lvl) == False
        self.drop_when_test = drop_when_test
        self.drop_lvl_idx = drop_lvl_idx
        self.drop_lvl_num = drop_lvl_num
        self.casual_multi_scale = casual_multi_scale
        print(f"{casual_multi_scale=}")
        if self.drop_when_test:
            assert drop_lvl_idx is not None
            assert drop_lvl_num > 0

        self.lfq = BSQ(
            dim = codebook_dim,
            codebook_scale = 1/np.sqrt(codebook_dim),
            soft_clamp_input_value = soft_clamp_input_value,
            # 中文说明：experimental_softplus_entropy_loss=True,
            # 中文说明：entropy_loss_offset=2,
            **kwargs
        )

        self.z_interplote_up = 'trilinear'
        self.z_interplote_down = 'area'

        self.use_decay_factor = use_decay_factor
        self.schedule_mode = schedule_mode
        self.keep_first_quant = keep_first_quant
        self.keep_last_quant = keep_last_quant
        if self.use_stochastic_depth and self.drop_rate > 0:
            assert self.keep_first_quant or self.keep_last_quant

    @property
    def codebooks(self):
        """暴露底层 BSQ 的隐式码本接口，便于外部统一访问。"""
        return self.lfq.codebook

    def get_codes_from_indices(self, indices_list):
        """把每一级 `indices` 重建成 codes，并上采样到最终分辨率后累加。"""
        all_codes = []
        for indices in indices_list:
            codes = self.lfq.indices_to_codes(indices)
            all_codes.append(codes)
        _, _, T, H, W = all_codes[-1].size()
        summed_codes = 0
        for code in all_codes:
            summed_codes += F.interpolate(code, size=(T, H, W), mode=self.z_interplote_up)
        return summed_codes

    def get_output_from_indices(self, indices):
        """直接根据索引重建量化输出，常用于离线可视化或调试重放。"""
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def flip_quant(self, x):
        """按给定概率翻转量化码的符号，用于量化扰动实验。"""
        assert self.flip_mode == 'stochastic'
        flip_mask = torch.rand_like(x) < self.flip_prob
        x = x.clone()
        x[flip_mask] = -x[flip_mask]
        return x

    def forward(
        self,
        x,
        scale_schedule=None,
        mask = None,
        return_all_codes = False,
        return_residual_norm_per_scale = False
    ):
        """执行多尺度残差量化。

        每层流程为：

        1. 把当前残差下采样到目标尺度。
        2. 用底层 BSQ 量化得到 `quantized / indices / bit_indices / loss`。
        3. 把量化结果上采样回原始时空分辨率。
        4. 用 `residual = residual - stopgrad(quantized)` 更新残差。

        `return_residual_norm_per_scale=True` 时会额外统计各尺度残差的稀疏程度，
        用于观察由粗到细调度是否有效。
        """
        if x.ndim == 4:
            x = x.unsqueeze(2)
        B, C, T, H, W = x.size()

        if scale_schedule is None:
            if self.schedule_mode.startswith("same"):
                scale_num = int(self.schedule_mode[len("same"):])
                assert T == 1
                scale_schedule = [(1, H, W)] * scale_num
            else:
                scale_schedule = get_latent2scale_schedule(T, H, W, mode=self.schedule_mode)
                scale_num = len(scale_schedule)

        # 代码/形状说明：x = self.project_in(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous() # (b, c, t, h, w) => (b, t, h, w, c)
        x = self.project_in(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous() # (b, t, h, w, c) => (b, c, t, h, w)
        x = self.layernorm(x)

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_bit_indices = []
        var_inputs = []
        residual_norm_per_scale = []

        # 中文标题：遍历各层
        out_fact = init_out_fact = 1.0
        # 代码/形状说明：residual_list = []
        # 代码/形状说明：interpolate_residual_list = []
        # 代码/形状说明：quantized_list = []
        if self.drop_when_test:
            drop_lvl_start = self.drop_lvl_idx
            drop_lvl_end = self.drop_lvl_idx + self.drop_lvl_num
        scale_num = len(scale_schedule)
        with autocast('cuda', enabled = False):
            for si, (pt, ph, pw) in enumerate(scale_schedule):
                out_fact = max(0.1, out_fact) if self.use_decay_factor else init_out_fact
                if self.casual_multi_scale and (pt, ph, pw) != (T, H, W):
                    interpolate_residual = F.interpolate(residual[:, :, :pt, :, :], size=(pt, ph, pw), mode=self.z_interplote_down)
                elif (pt, ph, pw) != (T, H, W):
                    interpolate_residual = F.interpolate(residual, size=(pt, ph, pw), mode=self.z_interplote_down)
                else:
                    interpolate_residual = residual
                if return_residual_norm_per_scale:
                    residual_norm_per_scale.append((torch.abs(interpolate_residual) < 0.05 * self.lfq.codebook_scale).sum() / interpolate_residual.numel())
                # 中文说明：residual_list.append(torch.norm(residual.detach(), dim=1).mean())
                # 中文说明：interpolate_residual_list.append(torch.norm(interpolate_residual.detach(), dim=1).mean())
                if self.training and self.use_stochastic_depth and random.random() < self.drop_rate:
                    if (si == 0 and self.keep_first_quant) or (si == scale_num - 1 and self.keep_last_quant):
                        quantized, indices, _, loss = self.lfq(interpolate_residual)
                        quantized = quantized * out_fact
                        all_indices.append(indices)
                        all_losses.append(loss)
                    else:
                        quantized = torch.zeros_like(interpolate_residual)
                elif self.drop_when_test and drop_lvl_start <= si < drop_lvl_end:
                    continue
                else:
                    # 代码/形状说明：residual_norm = torch.norm(interpolate_residual.detach(), dim=1) # (b, t, h, w)
                    # 代码/形状说明：print(si, residual_norm.min(), residual_norm.max(), residual_norm.mean())
                    quantized, indices, bit_indices, loss = self.lfq(interpolate_residual)
                    if self.random_flip and si < self.max_flip_lvl:
                        quantized = self.flip_quant(quantized)
                    if self.random_flip_1lvl and si == self.flip_lvl_idx:
                        quantized = self.flip_quant(quantized)
                    quantized = quantized * out_fact
                    all_indices.append(indices)
                # 中文说明：quantized_list.append(torch.norm(quantized.detach(), dim=1).mean())
                if (pt, ph, pw) != (T, H, W):
                    quantized = F.interpolate(quantized, size=(T, H, W), mode=self.z_interplote_up).contiguous()

                if self.remove_residual_detach:
                    residual = residual - quantized
                else:
                    residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_bit_indices.append(bit_indices)
                all_losses.append(loss)
                if si != scale_num - 1:
                    var_inputs.append(F.interpolate(quantized_out, size=scale_schedule[si+1], mode=self.z_interplote_down).contiguous())

                if self.use_decay_factor:
                    out_fact -= 0.1
        # 代码/形状说明：print("residual_list:", residual_list)
        # 代码/形状说明：print("interpolate_residual_list:", interpolate_residual_list)
        # 代码/形状说明：print("quantized_list:", quantized_list)
        # 代码/形状说明：import ipdb; ipdb.set_trace()
        # 中文说明：必要时投影回输出维度
        quantized_out = quantized_out.permute(0, 2, 3, 4, 1).contiguous() # (b, c, t, h, w) => (b, t, h, w, c)
        quantized_out = self.project_out(quantized_out)
        quantized_out = quantized_out.permute(0, 4, 1, 2, 3).contiguous() # (b, t, h, w, c) => (b, c, t, h, w)

        # 中文说明：图像
        if quantized_out.size(2) == 1:
            quantized_out = quantized_out.squeeze(2)

        # 中文标题：堆叠所有损失和索引

        all_losses = torch.stack(all_losses, dim = -1)

        ret = (quantized_out, all_indices, all_bit_indices, residual_norm_per_scale, all_losses, var_inputs)

        if not return_all_codes:
            return ret

        # 中文标题：是否返回跨层所有 codebook 的全部 codes
        all_codes = self.get_codes_from_indices(all_indices)

        # 中文说明：将以 (quantizer, batch, sequence length, codebook dimension) 形状返回全部 codes

        return (*ret, all_codes)


class BSQ(Module):
    """支持投影、多 codebook 与多种熵正则选项的 BSQ 实现。"""

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
        codebook_scale = 1.,                        # 代码/形状说明：用于残差 LFQ，每层 codebook 缩小 2x
        frac_per_sample_entropy = 1.,               # 中文说明：小于 1 时，只随机使用一部分概率来计算单样本熵
        has_projections = None,
        projection_has_bias = True,
        soft_clamp_input_value = None,
        cosine_sim_project_in = False,
        cosine_sim_project_in_scale = None,
        channel_first = None,
        experimental_softplus_entropy_loss = False,
        entropy_loss_offset = 5.,                   # 中文标题：softplus 前对损失的平移量
        spherical = True,                          # 代码/形状说明：来源：https://arxiv.org/abs/2406.07548
        force_quantization_f32 = True,               # 中文标题：强制量化步骤使用全精度
        inv_temperature = 100.0,
        gamma0=1.0, gamma=1.0, zeta=1.0,
        preserve_norm = False, # 中文标题：是否保留原始范数信息
        new_quant = True, # 中文说明：新的量化函数，
        mask_out = False, # 中文说明：在某些条件下把输出 mask 为 0
        use_out_phi = False, # 中文标题：使用输出 phi 网络
        use_out_phi_res = False, # 中文标题：残差式 out phi
        **kwargs,
    ):
        """初始化多尺度版本底层 BSQ 的投影、熵损失与量化策略。"""
        super().__init__()

        # 若干 assert 校验。

        assert exists(dim) or exists(codebook_size), '必须为 LFQ 指定 dim 或 codebook_size'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'无查找表量化的 codebook_size 必须是 2 的幂（建议 {2 ** ceil(log2(codebook_size))}）'

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        self.codebook_size = codebook_size

        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        self.codebook_dims = codebook_dims

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale = cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias = projection_has_bias)

        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity() # 代码/形状说明：nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim, bias = projection_has_bias) if has_projections else nn.Identity() # 代码/形状说明：nn.Identity()
        self.has_projections = has_projections

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

        # 中文标题：straight-through 激活

        self.activation = straight_through_activation

        # 中文说明：对于 BSQ（二值球面量化）
        if not spherical:
            raise ValueError("对于 BSQ，spherical 必须为 True。")
        self.persample_entropy_compute = 'analytical'
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0  # 中文标题：熵惩罚的损失权重
        self.gamma = gamma  # 中文标题：熵惩罚的损失权重
        self.zeta = zeta    # 中文标题：整体熵惩罚的损失权重
        self.preserve_norm = preserve_norm
        self.new_quant = new_quant
        self.mask_out = mask_out

        # 中文标题：熵辅助损失相关权重

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # 中文标题：codebook 缩放

        self.codebook_scale = codebook_scale

        # 承诺损失。

        self.commitment_loss_weight = commitment_loss_weight

        # 中文标题：是否把输入值软裁剪到 -value 到 value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # 中文说明：是否通过 softplus 让熵损失为正（实验性选项）

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # 代码/形状说明：推理时没有辅助损失

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # 中文说明：是否强制量化步骤使用 f32

        self.force_quantization_f32 = force_quantization_f32

        # 码本 codes。

        # 代码/形状说明：all_codes = torch.arange(codebook_size)
        # 代码/形状说明：bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        # 代码/形状说明：codebook = self.bits_to_codes(bits)

        # 代码/形状说明：self.register_buffer('codebook', codebook.float(), persistent = False)

    def bits_to_codes(self, bits):
        """把 bit `{0, 1}` 映射为 `{-scale, +scale}` 码值。"""
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
        """把整数索引或 bit 标签恢复为连续码，并按需投影回特征维。"""
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

        codes = self.bits_to_codes(bits)

        codes = l2norm(codes) # 中文标题：使用 BSQ 时必须归一化

        codes = rearrange(codes, '... c d -> ... (c d)')

        # 中文标题：是否把 codes 投影回原始维度
        # 代码/形状说明：如果输入特征维度不是 log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # 中文标题：把 codes 重排回原始形状

        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def quantize(self, z):
        """基础符号量化：输出 `{-1, +1}`，再通过 STE 保留反向梯度。"""
        assert z.shape[-1] == self.codebook_dims, f"期望 {self.codebook_dims} 维，实际 {z.shape[-1]} 维"

        zhat = torch.where(z > 0,
                           torch.tensor(1, dtype=z.dtype, device=z.device),
                           torch.tensor(-1, dtype=z.dtype, device=z.device))
        return z + (zhat - z).detach()

    def quantize_new(self, z):
        """把符号量化结果再缩放到单位球面半径 `1 / sqrt(d)`。"""
        assert z.shape[-1] == self.codebook_dims, f"期望 {self.codebook_dims} 维，实际 {z.shape[-1]} 维"

        zhat = torch.where(z > 0,
                           torch.tensor(1, dtype=z.dtype, device=z.device),
                           torch.tensor(-1, dtype=z.dtype, device=z.device))

        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat # 中文标题：归一化到单位球面

        return z + (zhat - z).detach()

    def soft_entropy_loss(self, z):
        """用解析近似计算 bit 熵与 batch 级 codebook 使用率熵。"""
        if self.persample_entropy_compute == 'analytical':
            # 代码/形状说明：if self.l2_norm:
            p = torch.sigmoid(-4 * z / (self.codebook_dims ** 0.5) * self.inv_temperature)
            # 代码/形状说明：else:
            # 代码/形状说明：p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1-p], dim=-1) # (b, h, w, 18, 2)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean() # 代码/形状说明：(b,h,w,18)->(b,h,w)->scalar
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        # 中文标题：每个子组概率的宏平均
        avg_prob = reduce(prob, '... g d ->g d', 'mean') # (18, 2)
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        # 中文标题：熵近似为各子组熵之和
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        """根据计数或概率计算熵 `-sum(p * log(p))`。"""
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

        - `b`：batch 维度。
        - `n`: 序列长度，或展平后的时空位置数
        - `d`: 单个 codebook 的 bit 维度
        - `c`: codebook 个数

        对初学者最重要的几个中间量：

        - `bit_indices`: 每个位置、每个 codebook 的 bit 决策。
        - `indices`: 可选的整数索引表示，用于压缩存储。
        - `quantized`: 量化后的连续码，可继续参与网络前向。
        """

        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # 中文说明：将图像或视频标准化为 (batch, seq, dimension)

        if should_transpose:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d') # 代码/形状说明：x.shape [b, hwt, c]

        assert x.shape[-1] == self.dim, f'期望维度为 {self.dim}，实际收到 {x.shape[-1]}'

        x = self.project_in(x)

        # 中文标题：拆出 codebook 数量维

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        x = l2norm(x)

        # 中文标题：是否强制量化步骤使用全精度

        force_f32 = self.force_quantization_f32

        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        indices = None
        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            # 中文说明：训练时使用 straight-through 梯度（可选自定义激活函数）
            if self.new_quant:
                quantized = self.quantize_new(x)

            # 中文标题：计算 indices
            bit_indices = (quantized > 0).int()
            entropy_penalty = persample_entropy = cb_entropy = self.zero
            commit_loss = self.zero

            # 中文标题：必要时把输入转回原始 dtype

            if force_f32:
                x = x.type(orig_dtype)

        # 中文标题：合并回 codebook 维度
        x = quantized # 中文标题：将 quantized 重命名为 x 作为输出
        x = rearrange(x, 'b n c d -> b n (c d)')

        # 中文标题：必要时投影回特征维度

        x = self.project_out(x)

        # 中文标题：还原图像或视频维度

        if should_transpose:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            bit_indices = unpack_one(bit_indices, ps, 'b * c d')

        # 中文标题：是否移除单 codebook 维度

        if not self.keep_num_codebooks_dim:
            bit_indices = rearrange(bit_indices, '... 1 d -> ... d')

        # 中文标题：完整辅助损失

        aux_loss = commit_loss * self.commitment_loss_weight + (self.zeta * entropy_penalty / self.inv_temperature)*entropy_weight
        # 中文说明：返回值

        ret = Return(x, indices, bit_indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(persample_entropy, cb_entropy, commit_loss)
