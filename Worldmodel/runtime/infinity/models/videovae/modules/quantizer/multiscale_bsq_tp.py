"""
双塔多尺度二值球面量化（双塔多尺度 BSQ）。

该版本把首帧语义信息与后续视频细节信息放进两座量化金字塔中处理：

- 外层负责生成 multi-scale schedule，并区分两座塔各自的时空分辨率。
- 内层仍然使用 BSQ，把连续特征按符号量化为 bit，再打包成整数索引。
- 每一级量化结果都会上采样回目标分辨率，逐层更新残差。
"""
import json
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

from infinity.models.videovae.utils.dynamic_resolution import predefined_HW_Scales_dynamic
from infinity.models.videovae.utils.dynamic_resolution_two_pyramid import dynamic_resolution_thw, total_pixels2scales
from infinity.models.videovae.modules.quantizer.finite_scalar_quantization import FSQ

# 代码/形状说明：print(f"{dynamic_resolution_thw=}")

# 常量区。

Return = namedtuple('Return', ['quantized', 'indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# 分布式辅助函数。

@cache
def is_distributed():
    """判断当前是否处于已初始化的分布式训练环境。"""
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    """分布式下对张量执行 all-reduce 均值；单卡时直接返回。"""
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
    """返回第一个有效参数；若该参数可调用则先执行。"""
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def round_up_multiple(num, mult):
    """把 `num` 向上补齐为 `mult` 的整数倍。"""
    return ceil(num / mult) * mult

def pack_one(t, pattern):
    """对单个张量执行 `einops.pack`。"""
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    """把 `pack_one` 打平后的张量按记录的形状还原。"""
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    """沿最后一维做 L2 归一化。"""
    return F.normalize(t, dim = -1)

# 熵相关计算。

def log(t, eps = 1e-5):
    """带数值保护的对数。"""
    return t.clamp(min = eps).log()

def entropy(prob):
    """计算熵 `H(p) = -sum(p * log(p))`。"""
    return (-prob * log(prob)).sum(dim=-1)

# 余弦相似度线性层。

class CosineSimLinear(Module):
    """先归一化输入和权重，再做余弦相似度线性投影。"""

    def __init__(
        self,
        dim_in,
        dim_out,
        scale = 1.
    ):
        """初始化余弦相似度线性层。"""
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        """返回归一化后的输入与权重的点积结果。"""
        x = F.normalize(x, dim = -1)
        w = F.normalize(self.weight, dim = 0)
        return (x @ w) * self.scale

def repeat_schedule(scale_schedule, repeat_scales_num, times):
    """把前若干个尺度重复多次，生成更密集的量化日程。"""
    new_scale_schedule = []
    for i in range(repeat_scales_num):
        new_scale_schedule.extend([scale_schedule[i] for _ in range(times)])
    new_scale_schedule.extend(scale_schedule[repeat_scales_num:])
    return new_scale_schedule

def get_latent2scale_schedule(T: int, H: int, W: int, mode="original", last_scale_repeat_n=0, args=None):
    """根据输入时空大小生成双塔多尺度量化日程。"""
    predefined_HW_Scales = {}
    if mode.startswith("infinity_video_two_pyramid"):
        if 'elegant' in mode:
            base_scale_schedule = copy.deepcopy(dynamic_resolution_thw[(H, W)]['scales'])
            image_scale_repetition = json.loads(args.image_scale_repetition)
            video_scale_repetition = json.loads(args.video_scale_repetition)
            # 代码/形状说明：print(f'{image_scale_repetition=} {video_scale_repetition=}')
            base_scale_schedule = copy.deepcopy(dynamic_resolution_thw[(H, W)]['scales'])
            def repeat_scales(base_scale_schedule, scale_repetition):
                """按重复次数展开基础尺度表。"""
                scale_schedule = []
                for i in range(len(base_scale_schedule)):
                    scale_schedule.extend([base_scale_schedule[i] for _ in range(scale_repetition[i])])
                return scale_schedule
            image_scale_schedule = repeat_scales(base_scale_schedule, image_scale_repetition)
            spatial_time_schedule = []
            spatial_time_schedule.extend(image_scale_schedule)
            firstframe_scalecnt = len(image_scale_schedule)
            if T > 1:
                scale_schedule = repeat_scales(base_scale_schedule, video_scale_repetition)
                spatial_time_schedule.extend([(T-1, h, w) for i, (_, h, w) in enumerate(scale_schedule)])
            # 中文标题：将 h 和 w 翻倍
            tower_split_index = firstframe_scalecnt
            # 代码/形状说明：print(f'{spatial_time_schedule=}')
            return spatial_time_schedule, tower_split_index
        if "motion_boost_v2" in mode:
            times = 6
            base_scale_schedule = copy.deepcopy(dynamic_resolution_thw[(H, W)]['scales'])
            image_scale_schedule = repeat_schedule(base_scale_schedule, 3, times)
            spatial_time_schedule = []
            spatial_time_schedule.extend(image_scale_schedule)
            firstframe_scalecnt = len(image_scale_schedule)
            if T > 1:
                scale_schedule = repeat_schedule(base_scale_schedule, 7, times)
                predefined_t = [T - 1 for _ in range(len(scale_schedule))]
                spatial_time_schedule.extend([(min(int(np.round(predefined_t[i])), T - 1), h, w) for i, (_, h, w) in enumerate(scale_schedule)])
            # 中文标题：将 h 和 w 翻倍
            spatial_time_schedule_double = [(t, 2*h, 2*w) for (t, h, w) in spatial_time_schedule]
            tower_split_index = firstframe_scalecnt
            return spatial_time_schedule_double, tower_split_index
        spatial_time_schedule = copy.deepcopy(dynamic_resolution_thw[(H, W)]['scales'])
        spatial_time_schedule.extend(spatial_time_schedule[-1:] * last_scale_repeat_n)
        tower_split_index = dynamic_resolution_thw[(H, W)]['tower_split_index'] + last_scale_repeat_n
        if T > 1:
            # 代码/形状说明：predefined_t = np.linspace(1, compressed_frames - 1, len(scale_schedule))
            if mode == "infinity_video_two_pyramid_full_time":
                spatial_time_schedule.extend([(T - 1, h, w) for i, (_, h, w) in enumerate(spatial_time_schedule)])
            else:
                predefined_t = np.linspace(1, T - 1, total_pixels2scales['0.06M']-3).tolist() + [T - 1] * (len(spatial_time_schedule)-total_pixels2scales['0.06M']+3)
                spatial_time_schedule.extend([(min(int(np.round(predefined_t[i])), T - 1), h, w) for i, (_, h, w) in enumerate(spatial_time_schedule)])
            spatial_time_schedule.extend(spatial_time_schedule[-1:] * last_scale_repeat_n)
        # 中文标题：将 h 和 w 翻倍
        spatial_time_schedule_double = [(t, 2*h, 2*w) for (t, h, w) in spatial_time_schedule]
        return spatial_time_schedule_double, tower_split_index
    if mode == "original":
        predefined_HW_Scales = {
            # 256x256
            (16, 16): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)],
            (36, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 12), (13, 16), (18, 24), (24, 32), (32, 48), (36, 64)],
            (18, 32): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 8), (8, 10), (10, 14), (12, 18), (14, 22), (16, 26), (18, 32)],
            (30, 53): [(1, 1), (2, 2), (3, 3), (4, 7), (6, 11), (8, 14), (12, 21), (16, 28), (20, 35), (22, 39), (24, 42), (26, 46), (28, 50), (30, 53)]
        }
        predefined_HW_Scales[(32, 32)] = predefined_HW_Scales[(16, 16)] + [(20, 20), (24, 24), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (64, 64)]
    elif mode == "dynamic":
        predefined_HW_Scales.update(predefined_HW_Scales_dynamic)
    elif mode == "dense":
        predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(32, 32)] = predefined_HW_Scales[(16, 16)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (56, 56), (64, 64)]
    elif mode == "dense_f8":
        # 代码/形状说明：predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(32, 32)] = [(x, x) for x in range(1, 16+1)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (56, 56), (64, 64)]
        predefined_HW_Scales[(128, 128)] = predefined_HW_Scales[(64, 64)] + [(80, 80), (96, 96), (112, 112), (128, 128)]
    elif mode == "dense_f8_double":
        # 代码/形状说明：从 dense f16 设置翻倍后的 predefined_HW_Scales
        predefined_HW_Scales[(32, 32)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(96, 96)] = predefined_HW_Scales[(64, 64)] + [(40, 40), (48, 48)]
        predefined_HW_Scales[(128, 128)] = predefined_HW_Scales[(64, 64)] + [(40, 40), (48, 48), (56, 56), (64, 64)]

        predefined_HW_Scales[(24, 42)] = [(1, 1), (2, 2), (3, 3), (3, 4), (3, 5), (4, 6), (4, 7), (5, 8), (6, 9), (6, 10), (6, 11), (7, 12), (7, 13), (8, 14), (9, 15), (9, 16), (12, 21)]
        predefined_HW_Scales[(36, 64)] = predefined_HW_Scales[(24, 42)] + [(14, 26), (18, 32)]
        predefined_HW_Scales[(60, 108)] = predefined_HW_Scales[(36, 64)] + [(24, 42), (30, 54)]
        predefined_HW_Scales[(90, 160)] = predefined_HW_Scales[(60, 108)] + [(38, 66),(45, 80)]

        for k, v in predefined_HW_Scales.items():
            predefined_HW_Scales[k] = [(2*x, 2*y) for (x, y) in v]
    elif mode.startswith("same"):
        num_quant = int(mode[len("same"):])
        predefined_HW_Scales[(16, 16)] = [(16, 16) for _ in range(num_quant)]
        predefined_HW_Scales[(32, 32)] = [(32, 32) for _ in range(num_quant)]
        predefined_HW_Scales[(64, 64)] = [(64, 64) for _ in range(num_quant)]
    elif mode == "half":
        predefined_HW_Scales[(32, 32)] = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)]
        predefined_HW_Scales[(64, 64)] = [(1,1),(2,2),(4,4),(6,6),(8,8),(12,12),(16,16)]
    else:
        raise NotImplementedError

    # 代码/形状说明：predefined_T_Scales = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 17, 17, 17, 17, 17, 17]
    # 代码/形状说明：predefined_T_Scales = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    predefined_T_Scales = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    # 代码/形状说明：predefined_T_Scales = [1, 2, 3, 5, 6, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    patch_THW_shape_per_scale = predefined_HW_Scales[(H, W)]
    if len(predefined_T_Scales) < len(patch_THW_shape_per_scale):
        # 代码/形状说明：print("警告：predefined_T_Scales 的长度小于 patch_THW_shape_per_scale 的长度！")
        predefined_T_Scales += [predefined_T_Scales[-1]] * (len(patch_THW_shape_per_scale) - len(predefined_T_Scales))
    patch_THW_shape_per_scale = [(min(T, t), h, w ) for (h, w), t in zip(patch_THW_shape_per_scale, predefined_T_Scales[:len(patch_THW_shape_per_scale)])]
    return patch_THW_shape_per_scale

def interpolate(tensor, size, mode):
    """
    在 `(B, C, T, H, W)` 与目标尺度 `(C1, T, H1, W1)` 之间做插值。

    这里先把时间维挪到通道维前面，再调用 `F.interpolate`，
    便于同时处理通道数变化与空间分辨率变化。
    """
    C1, T, H1, W1 = size
    tensor = tensor.permute(0,2,1,3,4) # (B,C,T,H,W) -> (B,T,C,H,W)
    tensor = F.interpolate(tensor, size=(C1, H1, W1), mode=mode)
    tensor = tensor.permute(0,2,1,3,4) # (B,T,C1,H1,W1) -> (B,C1,T,H1,W1)
    return tensor

# 中文说明：TP：双塔
class MultiScaleBSQTP(Module):
    """双塔多尺度 BSQ 量化器。

    可以把它理解成“两套残差量化日程串接在一起”：

    - 塔 1 通常面向首帧或静态语义信息。
    - 塔 2 通常面向后续帧或动态细节信息。

    每一级都遵循“下采样残差 -> 量化 -> 上采样回写”的流程。
    """

    def __init__(
        self,
        *,
        dim,
        soft_clamp_input_value = None,
        aux_loss = False, # 中文标题：中间辅助损失
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
        random_short_schedule = False, # 中文说明：随机使用较短日程（用于 256x256 图像的日程）
        short_schedule_prob = 0.5,
        disable_flip_prob = 0.0, # 中文标题：在当前图像禁用随机翻转
        casual_multi_scale = False,  # 中文标题：因果多尺度
        temporal_slicing = False,
        last_scale_repeat_n = 0,
        num_lvl_fsq = None,
        other_args = None,
        **kwargs
    ):
        """初始化双塔多尺度调度、随机短日程与底层量化器。"""
        super().__init__()
        codebook_dim = dim
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
        self.disable_flip_prob = disable_flip_prob
        self.casual_multi_scale = casual_multi_scale
        self.temporal_slicing = temporal_slicing
        self.last_scale_repeat_n = last_scale_repeat_n
        # 代码/形状说明：print(f"{casual_multi_scale=}")

        self.drop_when_test = drop_when_test
        self.drop_lvl_idx = drop_lvl_idx
        self.drop_lvl_num = drop_lvl_num
        if self.drop_when_test:
            assert drop_lvl_idx is not None
            assert drop_lvl_num > 0
        self.random_short_schedule = random_short_schedule
        self.short_schedule_prob = short_schedule_prob
        self.z_interplote_up = 'trilinear'
        self.z_interplote_down = 'area'

        self.schedule_mode = schedule_mode
        self.keep_first_quant = keep_first_quant
        self.keep_last_quant = keep_last_quant
        if self.use_stochastic_depth and self.drop_rate > 0:
            assert self.keep_first_quant or self.keep_last_quant

        self.full2short = {7:7, 10:7, 13:7, 16:16, 20:16, 24:16}
        if self.schedule_mode == 'dense_f8':
            self.full2short_f8 = {20:20, 24:24, 28:24}
        elif self.schedule_mode == 'dense_f8_double':
            self.full2short_f8 = {16: 14, 17: 14, 19: 14, 20:14, 21:14, 22:14, 24:14}
        elif self.schedule_mode.startswith("infinity_video_two_pyramid"):
            self.full2short_f8 = {11: 11, 13: 11, 14: 11, 16: 11, 29: 26, 28: 26, 26: 26}

        self.other_args = other_args
        print(f'{self.other_args=}')
        self.origin_C = self.other_args.detail_scale_dim
        self.detail_scale_dim, self.semantic_scale_dim = self.other_args.detail_scale_dim, self.other_args.semantic_scale_dim
        self.semantic_scales = other_args.semantic_scales

        if self.other_args.semantic_num_lvl == 2:
            self.lfq_semantic = BSQ(
                dim = self.semantic_scale_dim,
                codebook_scale = 1,
                soft_clamp_input_value = soft_clamp_input_value,
                **kwargs,
            )
        else:
            assert self.other_args.semantic_num_lvl >= 2, f'不支持 {self.other_args.semantic_num_lvl=}'
            self.lfq_semantic = FSQ(
                num_lvl = self.other_args.semantic_num_lvl,
                dim = self.semantic_scale_dim,
            )
        if self.other_args.detail_num_lvl == 2:
            self.lfq_detail = BSQ(
                dim = self.detail_scale_dim,
                codebook_scale = 1,
                soft_clamp_input_value = soft_clamp_input_value,
                **kwargs,
            )
        else:
            assert self.other_args.detail_num_lvl >= 2, f'不支持 {self.other_args.semantic_num_lvl=}'
            self.lfq_detail = FSQ(
                num_lvl = self.other_args.detail_num_lvl,
                dim = self.detail_scale_dim,
            )

    @property
    def codebooks(self):
        """暴露细节塔的底层码本接口，便于调试与可视化。"""
        return self.lfq_detail.codebook

    def get_codes_from_indices(self, indices_list):
        """把各尺度索引恢复为连续 codes，并上采样到最终时空分辨率。"""
        all_codes = []
        for indices in indices_list:
            # 中文说明：indices: [B,t,h,w,d]
            if indices.shape[-1] == self.origin_C:
                codes = self.lfq.indices_to_codes(indices)
            elif indices.shape[-1] == self.semantic_scale_dim:
                codes = self.lfq_semantic.indices_to_codes(indices)
            else:
                raise NotImplementedError(f'不支持 indices 形状 {indices.shape}')
            all_codes.append(codes)
        _, _, T, H, W = all_codes[-1].size()
        summed_codes = 0
        for code in all_codes:
            summed_codes += F.interpolate(code, size=(T, H, W), mode=self.z_interplote_up)
        return summed_codes

    def get_output_from_indices(self, indices):
        """仅根据索引重建量化输出，不重新执行量化器前向。"""
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return codes_summed

    def flip_quant(self, x):
        """以随机符号翻转方式扰动量化码，模拟 bit 噪声。"""
        # 代码/形状说明：assert self.flip_mode in ['stochastic', 'stochastic_dynamic']
        if self.flip_mode == 'stochastic':
            flip_mask = torch.rand_like(x) < self.flip_prob
        elif self.flip_mode == 'stochastic_dynamic':
            flip_prob = random.uniform(0, self.flip_prob)
            flip_mask = torch.rand_like(x) < flip_prob
        else:
            raise NotImplementedError
        x = x.clone()
        x[flip_mask] = -x[flip_mask]
        return x

    def forward(
        self,
        x_list,
        mask = None,
        return_all_codes = False,
    ):
        """执行双塔多尺度残差量化。

        `x_list` 最多包含两路输入特征，分别对应两座塔。
        返回的 `all_indices` 记录每个尺度的离散索引，便于后续做
        `indices -> codes -> output` 的重建和调度分析。
        """
        assert len(x_list) <= 2
        multi_scale = len(x_list) == 2
        for i in range(len(x_list)):
            if x_list[i].ndim == 4:
                x_list[i] = x_list[i].unsqueeze(2)
        B, C, T, H, W = x_list[-1].size()

        if self.schedule_mode.startswith("same"):
            scale_num = int(self.schedule_mode[len("same"):])
            assert T == 1
            scale_schedule = [(1, H, W)] * scale_num
        elif self.schedule_mode.startswith("infinity_video_two_pyramid"):
            scale_schedule, tower_split_index = get_latent2scale_schedule(T, H, W, mode=self.schedule_mode, last_scale_repeat_n=self.last_scale_repeat_n, args=self.other_args)
            scale_num = len(scale_schedule)
        else:
            scale_schedule = get_latent2scale_schedule(T, H, W, mode=self.schedule_mode, args=self.other_args)
            scale_num = len(scale_schedule)

        quantized_out = torch.zeros((B, C, 1, 1, 1), device=x_list[-1].device, dtype=x_list[-1].dtype)
        quantized_out_firstframe = None

        all_losses = []
        all_indices = []

        # 中文标题：计算 scale_in_one_clip
        unique_scale_schedule = [scale_schedule[0]]
        scale_in_one_clip = 1
        for si in range(1, len(scale_schedule)):
            if np.array(scale_schedule[si]).prod() < np.array(scale_schedule[si-1]).prod():
                break
            if scale_schedule[si] != scale_schedule[si-1]:
                unique_scale_schedule.append(scale_schedule[si])
                scale_in_one_clip += 1

        current_scale_in_one_clip = 0
        must_preserve_scales = []
        if self.other_args.quant_not_rely_256:
            must_preserve_scales = [11]
        with autocast('cuda', enabled = False):
            for si, (pt, ph, pw) in enumerate(scale_schedule):
                if si > 0 and scale_schedule[si] != scale_schedule[si-1]:
                    current_scale_in_one_clip += 1
                    current_scale_in_one_clip = current_scale_in_one_clip % scale_in_one_clip

                last_step_in_one_scale = False
                if si < len(scale_schedule)-1 and (scale_schedule[si] != scale_schedule[si+1]):
                    last_step_in_one_scale = True
                if si == len(scale_schedule)-1:
                    last_step_in_one_scale = True

                if si < tower_split_index:
                    ss, ee = 0, 1
                else:
                    ss, ee = 1, T

                if multi_scale and current_scale_in_one_clip < self.other_args.scales_256:
                    target = x_list[0][:,:,ss:ee]
                else:
                    target = x_list[-1][:,:,ss:ee]
                tgt_shape = target.shape[-4:]

                skip_this_scale = False
                if current_scale_in_one_clip < self.semantic_scales:
                    C1 = self.semantic_scale_dim
                    lfq = self.lfq_semantic
                else:
                    C1 = self.detail_scale_dim
                    lfq = self.lfq_detail
                    if current_scale_in_one_clip not in must_preserve_scales:
                        skip_this_scale = random.random() < self.other_args.skip_detail_scales_prob

                if not skip_this_scale:
                    quantized_out = interpolate(quantized_out, size=tgt_shape, mode=self.z_interplote_up)
                    interpolate_residual = interpolate(target-quantized_out, size=(C1, pt, ph, pw), mode=self.z_interplote_down)
                    quantized, indices, loss = lfq(interpolate_residual)
                    quantized = interpolate(quantized, size=tgt_shape, mode=self.z_interplote_up)
                    all_indices.append(indices)
                    all_losses.append(loss)
                    quantized_out = quantized_out + quantized

                if si == tower_split_index - 1:
                    quantized_out_firstframe = quantized_out.clone()
                    quantized_out = quantized_out * 0. # 中文标题：置零

                if multi_scale and si < tower_split_index and last_step_in_one_scale and current_scale_in_one_clip == self.other_args.scales_256-1:
                    quantized_out_firstframe_256 = quantized_out.clone()
                    if self.other_args.quant_not_rely_256:
                        quantized_out = quantized_out * 0.
                if multi_scale and si >= tower_split_index and last_step_in_one_scale and current_scale_in_one_clip == self.other_args.scales_256-1:
                    quantized_out_256 = quantized_out.clone()
                    if self.other_args.quant_not_rely_256:
                        quantized_out = quantized_out * 0.

            quantized_out_list = []
            if T == 1:
                if multi_scale:
                    quantized_out_list.append(quantized_out_firstframe_256)
                    quantized_out_list.append(quantized_out_firstframe)
                else:
                    quantized_out_list.append(quantized_out_firstframe)
            else:
                if multi_scale:
                    quantized_out_256 = torch.cat([quantized_out_firstframe_256, quantized_out_256], dim=2)
                    quantized_out_list.append(quantized_out_256)
                    quantized_out = torch.cat([quantized_out_firstframe, quantized_out], dim=2)
                    quantized_out_list.append(quantized_out)
                else:
                    quantized_out = torch.cat([quantized_out_firstframe, quantized_out], dim=2)
                    quantized_out_list.append(quantized_out)

        all_losses = torch.stack(all_losses, dim = -1)

        ret = (quantized_out_list, all_indices, all_losses)

        if not return_all_codes:
            return ret

        # 中文标题：是否返回跨层所有 codebook 的全部 codes
        all_codes = self.get_codes_from_indices(all_indices)

        # 中文说明：将以 (quantizer, batch, sequence length, codebook dimension) 形状返回全部 codes

        return (*ret, all_codes)


class BSQ(Module):
    """双塔多尺度版本底层使用的 BSQ 量化器。"""

    def __init__(
        self,
        *,
        dim = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 0.25,
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,                        # 代码/形状说明：用于残差 LFQ，每层 codebook 缩小 2x
        frac_per_sample_entropy = 1.,               # 中文说明：小于 1 时，只随机使用一部分概率来计算单样本熵
        soft_clamp_input_value = None,
        channel_first = None,
        experimental_softplus_entropy_loss = False,
        entropy_loss_offset = 5.,                   # 中文标题：softplus 前对损失的平移量
        spherical = True,                          # 代码/形状说明：来源：https://arxiv.org/abs/2406.07548
        force_quantization_f32 = True,               # 中文标题：强制量化步骤使用全精度
        inv_temperature = 100.0,
        gamma0=1.0, gamma=1.0, zeta=1.0,
        use_out_phi = False, # 中文标题：使用输出 phi 网络
        use_out_phi_res = False, # 中文标题：残差式 out phi
        use_bernoulli = False,
        use_rot_trick = False,
    ):
        """初始化球面二值量化、熵正则和可选 Bernoulli / 旋转技巧。"""
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

        # 中文说明：对于 BSQ（二值球面量化）
        if not spherical:
            raise ValueError("对于 BSQ，spherical 必须为 True。")
        self.persample_entropy_compute = 'analytical'
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0  # 中文标题：熵惩罚的损失权重
        self.gamma = gamma  # 中文标题：熵惩罚的损失权重
        self.zeta = zeta    # 中文标题：整体熵惩罚的损失权重
        self.use_bernoulli = use_bernoulli
        self.use_rot_trick = use_rot_trick

        # 中文标题：熵辅助损失相关权重

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

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

    def bits_to_codes(self, bits):
        """把 bit `{0, 1}` 映射为 `{-scale, +scale}`。"""
        return bits * self.codebook_scale * 2 - self.codebook_scale

    # 中文说明：@property
    # 代码/形状说明：def dtype(self):
    # 代码/形状说明：return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        project_out = True
    ):
        """把整数索引按位解包成 bit，再恢复为连续码向量。"""
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... -> ... 1')

        # 中文说明：indices 转 codes，codes 是 -1 或 1 的 bit

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = l2norm(codes) # 中文标题：使用 BSQ 时必须归一化

        codes = rearrange(codes, '... c d -> ... (c d)')

        # 中文标题：是否把 codes 投影回原始维度
        # 代码/形状说明：如果输入特征维度不是 log2(codebook size)

        # 中文标题：把 codes 重排回原始形状

        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def quantize(self, z):
        """按符号量化到球面二值码，并通过 STE 保留梯度。"""
        assert z.shape[-1] == self.codebook_dims, f"期望 {self.codebook_dims} 维，实际 {z.shape[-1]} 维"

        zhat = torch.where(z > 0,
                           torch.tensor(1, dtype=z.dtype, device=z.device),
                           torch.tensor(-1, dtype=z.dtype, device=z.device))

        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat # 中文标题：归一化到单位球面

        return z + (zhat - z).detach()

    def quantize_new_bernoulli(self, z, prob_z):
        """依据 Bernoulli 概率采样 bit，再映射到球面二值码。"""
        assert z.shape[-1] == self.codebook_dims, f"期望 {self.codebook_dims} 维，实际 {z.shape[-1]} 维"

        zhat = (torch.bernoulli(prob_z) - 0.5) * 2.0

        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat # 中文标题：归一化到单位球面

        return z + (zhat - z).detach()

    def rot_quantize(self, z, inference=False):
        """使用旋转技巧构造更平滑的量化代理梯度。"""
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
        """解析估计 bit 熵与 batch 级 codebook 使用率熵。"""
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
        """计算熵 `-sum(p * log(p))`，可选先做概率归一化。"""
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

        量化后通过位权 `mask` 把 bit 串打包成整数索引，
        因此 `indices -> codes` 的反向重建只需再做一次按位展开。
        """

        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # 中文说明：将图像或视频标准化为 (batch, seq, dimension)

        if should_transpose:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d') # 代码/形状说明：x.shape [b, hwt, c]

        assert x.shape[-1] == self.dim, f'期望维度为 {self.dim}，实际收到 {x.shape[-1]}'

        # 中文标题：拆出 codebook 数量维

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        if self.use_bernoulli: # 中文说明：False
            prob_x = torch.sigmoid(x)

        x = l2norm(x)

        # 中文标题：是否强制量化步骤使用全精度

        force_f32 = self.force_quantization_f32

        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            # 中文标题：使用 straight-through 梯度
            if self.use_rot_trick: # 中文说明：False
                x_f = x.flatten(end_dim=-2) # 代码/形状说明：(b, hwt, 1, d) -> (bhwt, d)
                q_f = self.rot_quantize(x_f, inference= not self.training)
                quantized = q_f.reshape(x.shape)
            elif self.use_bernoulli: # 中文说明：False
                quantized = self.quantize_new_bernoulli(x, prob_x)
            else:
                quantized = self.quantize(x)

            # 中文标题：计算 indices
            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

            # 中文标题：熵辅助损失
            if self.training:
                persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(x) # 中文标题：计算熵
                entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
            else:
                # 代码/形状说明：如果不是训练模式，只返回占位 0
                entropy_penalty = persample_entropy = cb_entropy = self.zero

            # 中文标题：承诺损失

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

        if self.use_out_phi_res: # 中文说明：False
            x = x + self.out_phi_scale * self.out_phi(x) # 中文标题：把 out_phi 作为残差应用到量化输出
        else:
            x = self.out_phi(x) # 中文标题：把 out_phi 应用到量化输出

        x = rearrange(x, 'b n c d -> b n (c d)')

        # 中文标题：还原图像或视频维度

        if should_transpose:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        # 中文标题：是否移除单 codebook 维度

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        # 中文标题：完整辅助损失

        aux_loss = commit_loss * self.commitment_loss_weight + (self.zeta * entropy_penalty / self.inv_temperature)*entropy_weight
        # 中文说明：返回值

        ret = Return(x, indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(persample_entropy, cb_entropy, commit_loss)


# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
