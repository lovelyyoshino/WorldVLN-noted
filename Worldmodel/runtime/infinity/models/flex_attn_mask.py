# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
from functools import partial
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._dynamo.exc import CacheLimitExceeded

from infinity.schedules.dynamic_resolution import get_full_spatial_size_scale_indices, get_first_full_spatial_size_scale_index


def _length_to_offsets(lengths, device):
    """把每段长度前缀和化，得到 packed sequence 的边界 offsets。"""
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets

def _offsets_to_doc_ids_tensor(offsets):
    """把 offsets 展开成每个 token 所属的文档/尺度编号。"""
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    visual = torch.repeat_interleave(torch.arange(len(counts), device=device, dtype=torch.int32), counts)
    return visual

def _generate_video_tower_mask(offsets, context_frames, full_resolution_scales, prefix_lens):
    """生成 video tower 结构下的可见性函数。"""
    document_id = _offsets_to_doc_ids_tensor(offsets)
    visual_tokens = offsets[-2]
    def _mask_prefix_valid(b, h, q_idx, kv_idx):
        """文本前缀只能彼此互看。"""
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx >= visual_tokens) & (q_idx < text_token_ends) & (kv_idx >= visual_tokens) & (kv_idx < text_token_ends)
    def _mask_visual(b, h, q_idx, kv_idx):
        """视觉 token 可看同尺度、文本前缀，以及有限历史上下文中的全分辨率尺度。"""
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx < visual_tokens) & (
                        (document_id[q_idx] == document_id[kv_idx]) |
                        ((kv_idx >= visual_tokens) & (kv_idx < text_token_ends)) |
                        (
                            (document_id[q_idx] > document_id[kv_idx]) & (document_id[q_idx] - document_id[kv_idx] < context_frames) & (document_id[kv_idx] in full_resolution_scales)
                        )
                    )
    def video_tower_mask(b, h, q_idx, kv_idx):
        """合并前缀 mask 与视觉 mask。"""
        mask_prefix_valid = _mask_prefix_valid(b, h, q_idx, kv_idx)
        mask_visual = _mask_visual(b, h, q_idx, kv_idx)
        return mask_prefix_valid | mask_visual
    return video_tower_mask

def _generate_two_pyramid_mask(offsets, first_full_spatial_size_scale_index, prefix_lens):
    """生成双金字塔结构的可见性函数。"""
    document_id = _offsets_to_doc_ids_tensor(offsets)
    visual_tokens = offsets[-2]
    def _mask_prefix_valid(b, h, q_idx, kv_idx):
        """文本前缀内部全可见。"""
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx >= visual_tokens) & (q_idx < text_token_ends) & (kv_idx >= visual_tokens) & (kv_idx < text_token_ends)
    def _mask_visual(b, h, q_idx, kv_idx):
        """视觉 token 额外可以回看第一层全分辨率尺度。"""
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx < visual_tokens) & (
                        (document_id[q_idx] == document_id[kv_idx]) |
                        ((kv_idx >= visual_tokens) & (kv_idx < text_token_ends)) |
                        (document_id[q_idx] > document_id[kv_idx]) & (document_id[kv_idx] == first_full_spatial_size_scale_index)
                    )
    def video_two_pyramid_mask(b, h, q_idx, kv_idx):
        """合并文本前缀与视觉层级规则。"""
        mask_prefix_valid = _mask_prefix_valid(b, h, q_idx, kv_idx)
        mask_visual = _mask_visual(b, h, q_idx, kv_idx)
        return mask_prefix_valid | mask_visual
    return video_two_pyramid_mask

def _generate_inner_scale_only_mask(offsets, prefix_lens):
    """生成“仅同尺度可见”的最保守 mask。"""
    document_id = _offsets_to_doc_ids_tensor(offsets)
    visual_tokens = offsets[-2]
    def _mask_prefix_valid(b, h, q_idx, kv_idx):
        """文本前缀内部全可见。"""
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx >= visual_tokens) & (q_idx < text_token_ends) & (kv_idx >= visual_tokens) & (kv_idx < text_token_ends)
    def _mask_visual(b, h, q_idx, kv_idx):
        """视觉 token 只能看同尺度和文本前缀。"""
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx < visual_tokens) & (
                        (document_id[q_idx] == document_id[kv_idx]) |
                        ((kv_idx >= visual_tokens) & (kv_idx < text_token_ends))
                    )
    def overall_mask(b, h, q_idx, kv_idx):
        """组合成最终的布尔 mask。"""
        mask_prefix_valid = _mask_prefix_valid(b, h, q_idx, kv_idx)
        mask_visual = _mask_visual(b, h, q_idx, kv_idx)
        return mask_prefix_valid | mask_visual
    return overall_mask

def _generate_infinity_pack(offsets, querysid_refsid):
    """按预先打包好的尺度引用表生成 mask。"""
    document_id = _offsets_to_doc_ids_tensor(offsets) # 中文标题：to scale_ind
    def overall_mask(b, h, q_idx, kv_idx):
        """查询 token 所属尺度是否允许访问 key/value 所属尺度。"""
        querysid = document_id[q_idx]
        kv_sid = document_id[kv_idx]
        return querysid_refsid[querysid][kv_sid]
    return overall_mask

def causal(b, h, q_idx, kv_idx):
    """标准因果 mask：当前位置只能看见自己及之前的 token。"""
    return q_idx >= kv_idx

def build_flex_attn_func(
        flex_attention,
        seq_l,
        prefix_lens,
        args,
        device,
        batch_size,
        heads,
        pad_seq_len,
        sequece_packing_scales,
        super_scale_lengths,
        super_querysid_super_refsid,
):
    """
        为给定尺度日程构建 flex attention mask 函数。
        参数：
            flex_attention：编译后的 flex attention 函数。
            说明：seq_l: seq length
            公式/形状说明：prefix_lens: valid text prefix lens, [bs]
            说明：args: arguments
            说明：device: device
            说明：batch_size: batch size
            说明：heads: heads
            说明：pad_seq_len: pad_seq_len
            说明：sequece_packing_scales: list of scale schedule
            说明：querysid_refsid: list of scale_pack_info
        返回：
            说明：attn_fn: flex attn function

    """
    assert sum(super_scale_lengths) == seq_l, f'{sum(super_scale_lengths)}!= {seq_l}'
    offsets = _length_to_offsets(super_scale_lengths, device=device)
    mask_mod = _generate_infinity_pack(offsets, super_querysid_super_refsid)
    # 中文说明：TorchDynamo compilation of block masks can hit CacheLimitExceeded when Q_LEN varies a lot.
    # 中文说明：When it happens, fall back to eager mask creation and disable compilation thereafter
    # 中文说明：to keep training running (at the cost of performance).
    if not hasattr(build_flex_attn_func, "_compile_block_mask"):
        build_flex_attn_func._compile_block_mask = True
    try:
        block_mask = create_block_mask(
            mask_mod,
            B=batch_size,
            H=heads,
            Q_LEN=seq_l,
            KV_LEN=seq_l,
            device=device,
            _compile=bool(build_flex_attn_func._compile_block_mask),
        )
    except CacheLimitExceeded:
        build_flex_attn_func._compile_block_mask = False
        block_mask = create_block_mask(
            mask_mod,
            B=batch_size,
            H=heads,
            Q_LEN=seq_l,
            KV_LEN=seq_l,
            device=device,
            _compile=False,
        )
    attn_fn = partial(flex_attention, block_mask=block_mask)
    return attn_fn
