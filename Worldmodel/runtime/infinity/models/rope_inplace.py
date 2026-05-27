# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import math
import os
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, drop_path
from torch.utils.checkpoint import checkpoint


def precompute_rope2d_freqs_grid(dim, dynamic_resolution_h_w, rope2d_normalized_by_hw, pad_to_multiplier=1, max_height=2048 // 16, max_width=2048 // 16, base=10000.0, device=None, scaling_factor=1.0):
    """预计算 2D Rope 频率表，供 inplace 旋转实现复用。"""
    # 2D Rope 把一半维度分给 y，另一半分给 x。
    half_dim = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.int64).float().to(device) / half_dim)) # Rope 的 `theta = 1 / 10000^(i / half_dim)`。
    t_height = torch.arange(max_height, device=device, dtype=torch.int64).type_as(inv_freq)
    t_width = torch.arange(max_width, device=device, dtype=torch.int64).type_as(inv_freq)
    t_height = t_height / scaling_factor
    freqs_height = torch.outer(t_height, inv_freq)  # 代码/形状说明：(max_height, dim / (1d 用 1，2d 用 2，3d 用 3) / 2)，即 y*theta。
    t_width = t_width / scaling_factor
    freqs_width = torch.outer(t_width, inv_freq)  # 代码/形状说明：(max_width, dim / (1d 用 1，2d 用 2，3d 用 3) / 2)，即 x*theta。
    freqs_grid_map = torch.concat([
        freqs_height[:, None, :].expand(-1, max_width, -1), # 代码/形状说明：(max_height, max_width, dim / (1d 用 1，2d 用 2，3d 用 3) / 2)
        freqs_width[None, :, :].expand(max_height, -1, -1), # 代码/形状说明：(max_height, max_width, dim / (1d 用 1，2d 用 2，3d 用 3) / 2)
    ], dim=-1)  # 代码/形状说明：(max_height, max_width, dim / (1d 用 1，2d 用 2，3d 用 3))
    freqs_grid_map = torch.stack([torch.cos(freqs_grid_map), torch.sin(freqs_grid_map)], dim=0)
    # 代码/形状说明：(2, max_height, max_width, dim / (1d 用 1，2d 用 2，3d 用 3))

    rope2d_freqs_grid = {}
    for h_div_w in dynamic_resolution_h_w:
        scale_schedule = dynamic_resolution_h_w[h_div_w]['1M']['scales']
        _, ph, pw = scale_schedule[-1]
        max_edge_length = freqs_grid_map.shape[1]
        if ph >= pw:
            uph, upw = max_edge_length, int(max_edge_length / ph * pw)
        else:
            uph, upw = int(max_edge_length / pw * ph), max_edge_length
        rope_cache_list = []
        for (_, ph, pw) in scale_schedule:
            ph_mul_pw = ph * pw
            if rope2d_normalized_by_hw == 1: # 下采样到目标尺寸。
                rope_cache = F.interpolate(freqs_grid_map[:, :uph, :upw, :].permute([0,3,1,2]), size=(ph, pw), mode='bilinear', align_corners=True)
                rope_cache = rope_cache.permute([0,2,3,1]) # 代码/形状说明：(2, ph, pw, half_head_dim)
            elif rope2d_normalized_by_hw == 2: # 星形风格索引映射。
                _, uph, upw = scale_schedule[-1]
                indices = torch.stack([
                    (torch.arange(ph) * (uph / ph)).reshape(ph, 1).expand(ph, pw),
                    (torch.arange(pw) * (upw / pw)).reshape(1, pw).expand(ph, pw),
                ], dim=-1).round().int() # (ph, pw, 2)
                indices = indices.reshape(-1, 2) # (ph*pw, 2)
                rope_cache = freqs_grid_map[:, indices[:,0], indices[:,1], :] # 代码/形状说明：(2, ph*pw, half_head_dim)
                rope_cache = rope_cache.reshape(2, ph, pw, -1)
            elif rope2d_normalized_by_hw == 0:
                rope_cache = freqs_grid_map[:, :ph, :pw, :] # 代码/形状说明：(2, ph, pw, half_head_dim)
            else:
                raise ValueError(f'未知 rope2d_normalized_by_hw: {rope2d_normalized_by_hw}')
            rope_cache_list.append(rope_cache.reshape(2, ph_mul_pw, -1))
        cat_rope_cache = torch.cat(rope_cache_list, 1) # 代码/形状说明：(2, seq_len, half_head_dim)
        if cat_rope_cache.shape[1] % pad_to_multiplier:
            pad = torch.zeros(2, pad_to_multiplier - cat_rope_cache.shape[1] % pad_to_multiplier, half_dim)
            cat_rope_cache = torch.cat([cat_rope_cache, pad], dim=1)
        cat_rope_cache = cat_rope_cache[:,None,None,None] # 代码/形状说明：(2, 1, 1, 1, seq_len, half_dim)
        for pn in dynamic_resolution_h_w[h_div_w]:
            scale_schedule = dynamic_resolution_h_w[h_div_w][pn]['scales']
            tmp_scale_schedule = [(1, h, w) for _, h, w in scale_schedule]
            rope2d_freqs_grid[str(tuple(tmp_scale_schedule))] = cat_rope_cache
    return rope2d_freqs_grid


def precompute_rope3d_freqs_grid(dim, dynamic_resolution_h_w, rope2d_normalized_by_hw, pad_to_multiplier=1, max_frames=129 // 4, max_height=2048 // 8, max_width=2048 // 8, base=10000.0, device=None):
    """预计算 3D Rope，供 inplace 旋转实现直接索引。"""
    # 3D Rope 需要把频率维拆给 t / y / x 三个方向。
    assert dim % 2 == 0, f'目前只支持 dim % 2 == 0，但收到 dim={dim}'
    dim_div_2 = dim // 2
    num_of_freqs = int(np.ceil(dim_div_2 / 3))
    inv_freq = 1.0 / (base ** (torch.arange(num_of_freqs, dtype=torch.int64).float().to(device) / num_of_freqs)) # Rope 的 `theta = 1 / 10000^(i / num_of_freqs)`。
    t_height = torch.arange(max_height, device=device, dtype=torch.int64).type_as(inv_freq)
    t_width = torch.arange(max_width, device=device, dtype=torch.int64).type_as(inv_freq)
    t_frames = torch.arange(max_frames, device=device, dtype=torch.int64).type_as(inv_freq)
    freqs_height = torch.outer(t_height, inv_freq)  # 代码/形状说明：(max_height, ceil(dim_div_2 / 3))，即 y*theta。
    freqs_width = torch.outer(t_width, inv_freq)  # 代码/形状说明：(max_width, ceil(dim_div_2 / 3))，即 x*theta。
    freqs_frames = torch.outer(t_frames, inv_freq)  # 代码/形状说明：(max_width, ceil(dim_div_2 / 3))，即 x*theta。
    if (num_of_freqs*3) - dim_div_2 == 0:
        offset_t, offset_h, offset_w = num_of_freqs, num_of_freqs, num_of_freqs
    elif (num_of_freqs*3) - dim_div_2 == 2: # 需要裁掉 2 个频率元素。
        offset_t, offset_h, offset_w = num_of_freqs, num_of_freqs-1, num_of_freqs-1
    else: # 需要裁掉 1 个频率元素。
        offset_t, offset_h, offset_w = num_of_freqs-1, num_of_freqs, num_of_freqs
    freqs_grid_map = torch.concat([
        freqs_frames[:, None, None, :offset_t].expand(-1, max_height, max_width, -1), # 代码/形状说明：(max_frames, max_height, max_width, ceil(dim_div_2 / 3))
        freqs_height[None, :, None, :offset_h].expand(max_frames, -1, max_width, -1), # 代码/形状说明：(max_frames, max_height, max_width, ceil(dim_div_2 / 3))
        freqs_width[None, None, :, :offset_w].expand(max_frames, max_height, -1, -1), # 代码/形状说明：(max_frames, max_height, max_width, ceil(dim_div_2 / 3))
    ], dim=-1)  # 代码/形状说明：(max_frames, max_height, max_width, dim / 2)
    freqs_grid_map = torch.stack([torch.cos(freqs_grid_map), torch.sin(freqs_grid_map)], dim=0)
    # 代码/形状说明：(2, max_frames, max_height, max_width, dim / 2)

    rope2d_freqs_grid = {}
    for h_div_w in dynamic_resolution_h_w:
        scale_schedule = dynamic_resolution_h_w[h_div_w]['1M']['scales']
        pt, ph, pw = scale_schedule[-1]
        rope_cache_list4image, rope_cache_list4video = [], []
        for (pt, ph, pw) in scale_schedule:
            mul_pt_ph_pw = pt * ph * pw
            mul_ph_pw = ph * pw
            if rope2d_normalized_by_hw == 2: # 星形风格索引映射。
                upt, uph, upw = scale_schedule[-1]
                indices = torch.stack([
                    (torch.arange(pt) * (upt / pt)).reshape(pt, 1, 1).expand(pt, ph, pw),
                    (torch.arange(ph) * (uph / ph)).reshape(1, ph, 1).expand(pt, ph, pw),
                    (torch.arange(pw) * (upw / pw)).reshape(1, 1, pw).expand(pt, ph, pw),
                ], dim=-1).round().int() # (pt, ph, pw, 3)
                indices = indices.reshape(-1, 3) # (pt*ph*pw, 3)
                rope_cache = freqs_grid_map[:, indices[:,0], indices[:,1], indices[:,2], :] # (2, pt*ph*pw, dim / 2)
                rope_cache = rope_cache.reshape(2, pt, ph, pw, -1)
            elif rope2d_normalized_by_hw == 0:
                rope_cache = freqs_grid_map[:, :pt, :ph, :pw, :] # (2, pt, ph, pw, dim / 2)
            else:
                raise ValueError(f'未知 rope2d_normalized_by_hw: {rope2d_normalized_by_hw}')
            rope_cache_list4image.append(rope_cache[:,:1].reshape(2, mul_ph_pw, -1)) # (2, 1*ph*pw, dim / 2)
            rope_cache_list4video.append(rope_cache.reshape(2, mul_pt_ph_pw, -1)) # (2, pt*ph*pw, dim / 2)
        cat_rope_cache4image = torch.cat(rope_cache_list4image, 1) # (2, seq_len, dim / 2)
        cat_rope_cache4video = torch.cat(rope_cache_list4video, 1) # (2, seq_len, dim / 2)
        if cat_rope_cache4image.shape[1] % pad_to_multiplier:
            pad = torch.zeros(2, pad_to_multiplier - cat_rope_cache4image.shape[1] % pad_to_multiplier, dim//2)
            cat_rope_cache4image = torch.cat([cat_rope_cache4image, pad], dim=1)
        if cat_rope_cache4video.shape[1] % pad_to_multiplier:
            pad = torch.zeros(2, pad_to_multiplier - cat_rope_cache4video.shape[1] % pad_to_multiplier, dim//2)
            cat_rope_cache4video = torch.cat([cat_rope_cache4video, pad], dim=1)
        cat_rope_cache4image = cat_rope_cache4image[:,None,None,None] # (2, 1, 1, 1, seq_len, dim / 2)
        cat_rope_cache4video = cat_rope_cache4video[:,None,None,None] # (2, 1, 1, 1, seq_len, dim / 2)
        for pn in dynamic_resolution_h_w[h_div_w]:
            video_scale_schedule = dynamic_resolution_h_w[h_div_w][pn]['scales']
            image_scale_schedule = [(1, ph, pw) for pt, ph, pw in video_scale_schedule]
            rope2d_freqs_grid[str(tuple(video_scale_schedule))] = cat_rope_cache4video
            rope2d_freqs_grid[str(tuple(image_scale_schedule))] = cat_rope_cache4image
    return rope2d_freqs_grid


def apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, scale_ind):
    """对 query / key 原地施加 Rope 旋转，减少中间张量占用。"""
    # 代码/形状说明：qk = torch.stack((q, k), dim=0)  #(2, batch_size, heads, seq_len, head_dim)
    device_type = q.device.type # 代码/形状说明：(batch_size, heads, seq_len, head_dim))
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        seq_len = q.shape[2]
        start = 0
        if scale_ind >= 1:
            assert len(scale_schedule[0]) == 3
            start = np.sum([item[0] * item[1] * item[2] for item in scale_schedule[:scale_ind]])
        try:
            rope2d_freqs_grid[str(tuple(scale_schedule))] = rope2d_freqs_grid[str(tuple(scale_schedule))].to(q.device)
        except:
            print(rope2d_freqs_grid.keys())
            print(str(tuple(scale_schedule)))
            import pdb; pdb.set_trace()
        assert start+seq_len <= rope2d_freqs_grid[str(tuple(scale_schedule))].shape[4]
        rope_cache = rope2d_freqs_grid[str(tuple(scale_schedule))][:, 0, :, :, start:start+seq_len] # 中文说明：rope_cache 形状为 [2, 1, 1, seq_len, dim]

        qk = [q, k]
        for i in range(len(qk)):
            qk[i] = qk[i].reshape(*qk[i].shape[:-1], -1, 2) # 代码/形状说明：(batch_size, heads, seq_len, half_head_dim, 2)
            # 代码/形状说明：qk = torch.stack([
            # 代码/形状说明：rope_cache[0] * qk[...,0] - rope_cache[1] * qk[...,1],
            # 代码/形状说明：rope_cache[1] * qk[...,0] + rope_cache[0] * qk[...,1],
            # 中文说明：], dim=-1) # (2, batch_size, heads, seq_len, half_head_dim, 2)，这里必须先 stack 再 reshape，不能直接 concat。

            # 这是节省 vRAM 的 in-place 版本。
            tmp1 = qk[i][..., 1] * rope_cache[1]
            tmp2 = qk[i][..., 0] * rope_cache[1]
            qk[i][..., 0].mul_(rope_cache[0]).sub_(tmp1)
            qk[i][..., 1].mul_(rope_cache[0]).add_(tmp2)

        q, k = qk
        q = q.reshape(*q.shape[:-2], -1)
        k = k.reshape(*k.shape[:-2], -1)
    return q, k
