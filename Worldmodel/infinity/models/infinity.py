# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Infinity Transformer 世界模型定义。

小白导读：
这个文件是 WorldVLN 世界模型的 Transformer 主体。训练时它接收 compact 文本条件和
video VAE latent token，预测下一组多尺度 latent bit labels；推理时它按动态分辨率
schedule 自回归生成 `summed_codes`，再交给 VAE 解码或动作头使用。
"""

import math
import random
import time
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any
import json

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from torch.utils.checkpoint import checkpoint
import numpy as np
from torch.nn.attention.flex_attention import flex_attention

import infinity.utils.dist as dist
from infinity.utils.dist import for_visualize
from infinity.models.basic import flash_fused_op_installed, SelfAttnBlock, FastRMSNorm
from infinity.models.rope import precompute_rope4d_freqs_grid
from infinity.models.flex_attn_mask import build_flex_attn_func
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index, get_activated_h_div_w_templates
from infinity.models.apg import normalized_guidance
from infinity.utils.sequence_parallel import sp_split_sequence_by_dim, sp_gather_sequence_by_dim, SequenceParallelManager as sp_manager

try:
    from infinity.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None


class MultiInpIdentity(nn.Module):
    """兼容多输入模块签名的 Identity，用在无需额外条件的分支。"""

    def forward(self, x, *args, **kwargs):
        """忽略额外参数并原样返回主输入。"""
        return x

class SharedAdaLin(nn.Linear):
    """把条件向量投影成 6 组 AdaLN/AdaRMSNorm 参数。"""

    def forward(self, cond_BD):
        """返回形状 `(B,1,6,C)` 的共享自适应归一化参数。"""
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C

class MultipleLayers(nn.Module):
    """把连续 Transformer blocks 包成一个 chunk，便于 FSDP/分块执行。"""

    def __init__(self, ls, num_blocks_in_a_chunk, index):
        """从 block 列表中截取 `[index, index+num_blocks_in_a_chunk)` 作为一个 chunk。"""
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None, scale_ind=None, context_info=None, last_repetition_step=True, ref_text_scale_inds=[]):
        """按顺序执行 chunk 内 blocks，并按需启用 activation checkpoint。"""
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_schedule, scale_ind, context_info, last_repetition_step, ref_text_scale_inds, use_reentrant=False)
            else:
                h = m(h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_schedule, scale_ind, context_info, last_repetition_step, ref_text_scale_inds)
        return h

def get_timestep_embedding(dim, timesteps=1000, max_period=10000):
    """
    创建正弦/余弦时间步 embedding。

    参数说明：
    - `timesteps`: 时间步数量或时间索引范围；
    - `dim`: 输出 embedding 维度，必须是偶数；
    - `max_period`: 控制最低频率，数值越大，低频覆盖越长。
    返回形状为 `[timesteps, dim]` 的位置/时间编码。
    """
    assert dim % 2 == 0, "embedding 维度 dim 必须是偶数"
    half = dim // 2
    timesteps = torch.arange(timesteps, dtype=torch.float32)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

class Infinity(nn.Module):
    """
    文本条件 video latent 自回归世界模型。

    中文导读：
    `forward()` 用于训练，输入是已打包的视觉 token 和 compact 文本 KV；推理路径主要是
    `ar_infer_infinity_elegant()`，它根据 scale schedule 逐尺度采样 latent code，并累计成
    `summed_codes`。WorldVLN 在线服务直接复用这些 `summed_codes` 做动作解码。
    """

    def __init__(
        self, vae_local,
        arch='qwen',                         # 模型结构风格：当前主要使用 qwen
        qwen_qkvo_bias=False,               # Qwen attention 的 q/k/v/o 是否带 bias
        text_channels=0, text_maxlen=0,     # 文本条件维度和最大长度
        embed_dim=1024, depth=16,
        num_key_value_heads=-1,
        num_heads=16, mlp_ratio=4.,   # Transformer 主体结构参数
        norm_eps=1e-6, rms_norm=False,      # 归一化层配置
        cond_drop_rate=0.1,                 # classifier-free guidance 的条件丢弃概率
        rand_uncond=False,
        drop_path_rate=0.1,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        top_p=0.0,
        top_k=0.0,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        add_lvl_embeding_on_first_block=1,
        num_of_label_value=2,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        apply_spatial_patchify = 0,
        inference_mode=False,
        other_args=None,
    ):
        """初始化文本投影、视觉 token embedding、Transformer blocks、输出 head 和 RoPE cache。"""
        super().__init__()
        # 记录核心超参数，后续训练、推理、checkpoint 加载都会依赖这些字段。
        self.C = embed_dim
        self.vae_embed_dim = vae_local.codebook_dim
        self.detail_scale_min_tokens = other_args.detail_scale_min_tokens
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.codebook_dim * 4
        else:
            self.d_vae = vae_local.codebook_dim
        self.other_args = other_args
        self.mask_type = other_args.mask_type
        self.context_frames = other_args.context_frames
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(other_args.dynamic_scale_schedule, other_args.video_frames)
        self.num_of_label_value = num_of_label_value
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * self.num_of_label_value) if self.num_of_label_value else vae_local.vocab_size
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.image_batch_size = other_args.image_batch_size
        self.video_batch_size = other_args.video_batch_size
        self.arch = arch
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = get_activated_h_div_w_templates(train_h_div_w_list, self.h_div_w_templates)
        self.video_frames = video_frames


        assert add_lvl_embeding_on_first_block in [0,1]
        self.add_lvl_embeding_on_first_block = add_lvl_embeding_on_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        self.image_scale_repetition = json.loads(other_args.image_scale_repetition)
        self.video_scale_repetition = json.loads(other_args.video_scale_repetition)
        print(f'arch: {arch}, self.pn: {self.pn}, self.codebook_dim: {self.codebook_dim}, self.add_lvl_embeding_on_first_block: {self.add_lvl_embeding_on_first_block}, \
            self.num_of_label_value: {self.num_of_label_value}, self.rope2d_each_sa_layer: {rope2d_each_sa_layer}, self.rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw} \
            self.train_h_div_w_list: {self.train_h_div_w_list}, self.image_scale_repetition: {self.image_scale_repetition}, self.video_scale_repetition: {self.video_scale_repetition}')
        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule} 与 word_patch_size={word_patch_size} 不兼容'

        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)

        self.raw_scale_schedule = raw_scale_schedule    # raw 表示还没有经过空间 patchify 的原始尺度表
        # 规范化 top-p/top-k 采样参数；top_k 可以传比例，也可以传绝对数量。
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0

        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'

        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0

        # 输入 token 和位置编码相关模块。
        self.norm0_cond = nn.Identity()
        self.selecting_idx = None
        self.num_classes = 0
        self.D = self.C

        cfg_uncond = torch.empty(512, self.Ct5)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(0)
        torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
        cfg_uncond /= self.Ct5 ** 0.5
        if rand_uncond:
            self.register_buffer('cfg_uncond', cfg_uncond)
        else:
            self.cfg_uncond = nn.Parameter(cfg_uncond)

        if other_args.simple_text_proj:
            self.text_norm = nn.Identity()
            self.text_proj = nn.Linear(self.Ct5, self.D)
        else:
            self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
            self.text_proj = nn.Sequential(
                nn.Linear(self.Ct5, self.D),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.D, self.D),
            )
        self.sos_token = nn.Parameter(torch.empty(1, 1, self.D))

        if self.rope2d_each_sa_layer:
            if other_args.rope_type == '4d':
                tmp_h_div_w_template = self.train_h_div_w_list[0]
                scales_in_one_clip = self.dynamic_resolution_h_w[tmp_h_div_w_template][self.pn]['scales_in_one_clip']
                max_video_scales = self.dynamic_resolution_h_w[tmp_h_div_w_template][self.pn]['max_video_scales']
                if other_args.dynamic_scale_schedule == 'infinity_star_interact':
                    max_scales = 1000
                else:
                    max_scales = sum(self.image_scale_repetition) + sum(self.video_scale_repetition) * (max_video_scales//scales_in_one_clip-1)
                    max_scales = max(max_scales, max_video_scales)
                rope2d_freqs_grid = precompute_rope4d_freqs_grid(dim=self.C//self.num_heads,
                                                                 pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                                                                 activated_h_div_w_templates=self.train_h_div_w_list,
                                                                 steps_per_frame=other_args.steps_per_frame,
                                                                 max_scales=max_scales+10,
                                                                 max_frames=int(self.video_frames/other_args.temporal_compress_rate+1),
                                                                 max_height=1800 // 8, max_width=1800 // 8,
                                                                 text_maxlen=self.text_maxlen,
                                                                 pn=self.pn,
                                                                 args=other_args,)
            else:
                raise ValueError(f'self.rope_type == {self.rope_type} 暂不支持')
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} 暂未实现')

        # 输入层：对 VAE latent token 做归一化和线性嵌入。
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        if self.arch == 'qwen':
            self.norm_hidden_sates = FastRMSNorm(self.C)
        else:
            raise ValueError(f'arch={self.arch} 暂未实现')

        # Transformer 主干和输出 head。
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        if self.use_flex_attn:
            self.flex_attention = torch.compile(flex_attention)

        self.unregistered_blocks = []
        for _ in range(depth):
            block = SelfAttnBlock(
                embed_dim=self.C,
                cond_dim=self.D,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                mlp_ratio=mlp_ratio,
                use_flex_attn=use_flex_attn,
                pad_to_multiplier=pad_to_multiplier,
                rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                mask_type=other_args.mask_type,
                context_frames=other_args.context_frames,
                steps_per_frame=other_args.steps_per_frame,
                arch=self.arch,
                qwen_qkvo_bias=qwen_qkvo_bias,
                inject_sync=other_args.inject_sync,
            )
            # 如需节省显存，可在这里把单个 block 转成 bfloat16。
            self.unregistered_blocks.append(block)

        # 输出 head：detail/semantic 两套 LFQ bit-label 预测头。
        self.head = nn.Linear(self.C, self.other_args.detail_scale_dim*self.other_args.num_of_label_value)
        if self.other_args.use_two_stage_lfq:
            self.semantic_head2 = nn.Linear(self.C, self.other_args.semantic_scale_dim*self.other_args.num_of_label_value)

        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(f"{self.num_blocks_in_a_chunk=}, {depth=}, {block_chunks=}")
        assert self.num_blocks_in_a_chunk * block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))
        print(
            f'    [Infinity config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}, num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}\n',
            end='\n\n', flush=True
        )

    def get_loss_acc(self, x_BLC, sequece_packing_scales, gt):
        """
        根据视觉 token hidden states 计算逐 token 的训练 loss 和准确率。

        `sequece_packing_scales` 记录每段视觉 token 对应的 `(pt, ph, pw)` 尺度；
        two-stage LFQ 模式下，小空间尺度走 semantic head，大空间尺度走 detail head。
        """
        if self.arch == 'qwen':
            x_BLC = self.norm_hidden_sates(x_BLC)

        with torch.amp.autocast('cuda', enabled=False):
            x_BLC = x_BLC.float()
            logits_full = self.head(x_BLC)
            if self.other_args.use_two_stage_lfq:
                logits_semantic_full = self.semantic_head2(x_BLC)
                global_token_ptr, global_scale_ptr = 0, 0
                loss_list, acc_list = [], []
                for i in range(len(sequece_packing_scales)):
                    for j in range(len(sequece_packing_scales[i])):
                        pt, ph, pw = sequece_packing_scales[i][j]
                        mul_pt_ph_pw = pt * ph * pw
                        if ph * pw >= self.detail_scale_min_tokens:
                            logits = logits_full[:,global_token_ptr:global_token_ptr+mul_pt_ph_pw]
                        else:
                            logits = logits_semantic_full[:,global_token_ptr:global_token_ptr+mul_pt_ph_pw]
                        logits = logits.reshape(x_BLC.shape[0], mul_pt_ph_pw, -1, self.other_args.num_of_label_value)
                        logits = logits.permute(0,3,1,2) # 交叉熵要求类别维在前：[1, num_of_label_value, mul_pt_ph_pw, d]
                        # 当前尺度的监督 bit labels，形状为 [1, mul_pt_ph_pw, d]。
                        loss_this_scale = F.cross_entropy(logits, gt[global_scale_ptr], reduction='none').mean(-1)[0] # 输出 [mul_pt_ph_pw]
                        acc_this_scale = (logits.argmax(1) == gt[global_scale_ptr]).float().mean(-1)[0] # 输出 [mul_pt_ph_pw]

                        loss_list.append(loss_this_scale)
                        acc_list.append(acc_this_scale)
                        global_scale_ptr += 1
                        global_token_ptr += mul_pt_ph_pw
                loss_list = torch.cat(loss_list)
                acc_list = torch.cat(acc_list)
            else:
                gt = torch.cat(gt, 1) # 把所有尺度标签拼成 [B, L, d]
                logits = logits_full
                logits = logits.reshape(x_BLC.shape[0], x_BLC.shape[1], -1, self.other_args.num_of_label_value)
                logits = logits.permute(0,3,1,2) # 交叉熵布局：[B, num_of_label_value, L, d]
                if self.other_args.num_of_label_value > 1:
                    loss_list = F.cross_entropy(logits, gt, reduction='none').mean(-1)[0] # 每个视觉 token 一个 loss
                    acc_list = (logits.argmax(1) == gt).float().mean(-1)[0] # 每个视觉 token 一个 bit-level accuracy
                elif self.other_args.num_of_label_value == 1:
                    loss_list = torch.nn.functional.mse_loss(logits.squeeze(1), gt[global_scale_ptr], reduction='none').mean(-1)[0] # 连续值回归兼容路径
                    acc_list = loss_list
            return loss_list, acc_list

    def get_logits_during_infer(self, x_BLC, is_semantic_scale):
        """推理阶段把 hidden states 投影成 LFQ bit label logits。"""
        if self.arch == 'qwen':
            x_BLC = self.norm_hidden_sates(x_BLC)
        with torch.amp.autocast('cuda', enabled=False):
            x_BLC = x_BLC.float()
            if self.other_args.use_two_stage_lfq:
                if is_semantic_scale:
                    logits = self.semantic_head2(x_BLC)
                else:
                    logits = self.head(x_BLC)
            else:
                logits = self.head(x_BLC)
        return logits

    def pick_visual_tokens(
        self,
        x_BLC,
        sequece_packing_scales,
        visual_tokens_len,
        args,
    ):
        """从拼接后的视觉+文本序列中截取视觉 token 部分用于计算训练 loss。"""
        visual_tokens = x_BLC[:,:visual_tokens_len]
        return visual_tokens

    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC: torch.Tensor,
        visual_rope_cache = None,
        sequece_packing_scales = None, # 每个样本的多尺度视觉 token 范围，例如 [(1,1,1)..(5,5,5)]
        super_scale_lengths = None,
        super_querysid_super_refsid = None,
        other_info_by_scale = None,
        gt_BL = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # 返回逐 token loss、acc 和有效序列比例
        """
        训练阶段前向传播。

        参数说明：
        - `label_B_or_BLT`: compact 文本条件，通常是 `(kv_compact, lens, cu_seqlens_k, max_seqlen_k)`；
        - `x_BLC`: 已打包的视觉 latent token；
        - `visual_rope_cache`: 视觉 token 对应的 RoPE 位置编码；
        - `gt_BL`: 每个尺度的监督 bit labels。

        返回逐 token loss、逐 token accuracy 和有效序列比例。
        """

        x_BLC= x_BLC.float()       # 训练入口统一使用 float32，混精逻辑在内部控制。
        B = x_BLC.shape[0]
        cond_BD_or_gss, ca_kv = None, None

        # 第一步：构造视觉 token + 文本 token 的完整训练序列。
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            # compact 文本表示由 run_infinity/train.py 构建：kv_compact + 每条文本长度。

            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact[0, 0] += must_on_graph
            # 训练时随机把部分文本条件替换成 unconditional token，用于 classifier-free guidance。
            total = 0
            for le in lens:
                if random.random() < self.cond_drop_rate:
                    kv_compact[total:total+le] = self.cfg_uncond[:le]
                total += le

            visual_tokens_len = x_BLC.shape[1]
            # 文本 token 先做归一化/投影，再和视觉 token 拼接。
            kv_compact = self.text_norm(kv_compact)
            kv_compact = self.text_proj(kv_compact).contiguous()
            x_BLC = self.word_embed(self.norm0_ve(x_BLC)) # norm0_ve 当前是 Identity，保留接口兼容。
            x_BLC = torch.cat((x_BLC, kv_compact.unsqueeze(0)), dim=1)

            if self.other_args.train_with_var_seq_len:
                pad_seq_len = int(np.ceil(x_BLC.shape[1]/self.pad_to_multiplier))*self.pad_to_multiplier - x_BLC.shape[1]
            else:
                pad_seq_len = self.other_args.train_max_token_len - x_BLC.shape[1]
            if pad_seq_len > 0:
                x_BLC = F.pad(x_BLC, (0, 0, 0, pad_seq_len), value=0.0)

            # 旧写法按固定 train_max_token_len 计算；当前按真实 padding 后长度计算更稳。
            valid_sequence_ratio = 1 - pad_seq_len / x_BLC.shape[1]
            assert self.use_flex_attn
            attn_bias_or_two_vector = None

        attn_fn = build_flex_attn_func(
            flex_attention=self.flex_attention,
            seq_l=x_BLC.shape[1],
            prefix_lens=lens,
            args=self.other_args,
            device=x_BLC.device,
            batch_size=B,
            heads=None,
            pad_seq_len=pad_seq_len,
            sequece_packing_scales=sequece_packing_scales,
            super_scale_lengths=super_scale_lengths,
            super_querysid_super_refsid=super_querysid_super_refsid,
        )

        # 为本次视觉+文本序列拼出 RoPE cache；视觉部分由外部 schedule 生成，文本部分取固定表。
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(x_BLC.device)
        rope_cache_list = [visual_rope_cache]
        for i in range(len(lens)):
            rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[i]])
        rope_cache = torch.cat(rope_cache_list, dim=4)
        if pad_seq_len > 0:
            rope_cache = F.pad(rope_cache, (0,0,0,pad_seq_len), 'constant', 0.)
        assert rope_cache.shape[4] == x_BLC.shape[1], f'{rope_cache.shape[4]} != {x_BLC.shape[1]}'
        # 第二步：经过 Transformer blocks。
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training

        if sp_manager.sp_on():
            # 序列并行时，先把 token 序列按长度维切给不同 rank。
            x_BLC = sp_split_sequence_by_dim(x_BLC, 1)

        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, attn_fn, rope_cache, use_reentrant=False)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, rope2d_freqs_grid=rope_cache)
        else:
            for i, chunk in enumerate(self.block_chunks): # 大模型通常走分块 block_chunks 路径。
                x_BLC = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=rope_cache)

        if sp_manager.sp_on():
            # 序列并行结束后，把各 rank 的 token 拼回完整序列。
            x_BLC = sp_gather_sequence_by_dim(x_BLC, 1)

        # 第三步：只取视觉 token，计算训练监督 loss/acc。
        x_BLC = self.pick_visual_tokens(x_BLC, sequece_packing_scales, visual_tokens_len, self.other_args)
        loss_list, acc_list = self.get_loss_acc(x_BLC, sequece_packing_scales, gt_BL)
        return loss_list, acc_list, valid_sequence_ratio

    def prepare_text_conditions(
        self,
        label_B_or_BLT,
        cfg_list,
        B,
        negative_label_B_or_BLT,
        vae_scale_schedule=None,
        text_token_only=False,
        text_maxlen_this_iter=512,
    ):
        """
        准备推理阶段的文本 prefix token。

        当 cfg 不等于 1 时，会把正向文本和 unconditional/negative 文本拼成两倍 batch，
        后续 logits 再按 CFG/APG 做引导。

        CFG 公式导读：
          CFG 公式：`logits_guided = cfg*logits_cond + (1-cfg)*logits_uncond`。
        所以 cfg!=1 时需要同时前向 cond/uncond 两条分支；cfg=1 时只保留 cond 分支。
        """
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        bs = B
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
                lens = lens + lens
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
                lens = lens + lens_un
        kv_compact = self.text_norm(kv_compact)
        kv_compact = self.text_proj(kv_compact).contiguous()
        assert B == 1
        prefix_tokens = torch.zeros((bs, text_maxlen_this_iter, self.C), dtype=kv_compact.dtype, device=kv_compact.device)
        total = 0
        for i, le in enumerate(lens):
            assert le <= text_maxlen_this_iter
            prefix_tokens[i,:le] = kv_compact[total:total+le]
            total += le
        return prefix_tokens, lens

    @torch.no_grad()
    def autoregressive_infer(
        self,
        args=None,
        **kwargs,
    ):
        """根据动态 schedule 名称分派到对应的自回归推理实现。"""
        if 'infinity_elegant' in args.dynamic_scale_schedule:
            infer_func = self.ar_infer_infinity_elegant
        elif 'infinity_star_interact' in args.dynamic_scale_schedule:
            infer_func = self.ar_infer_infinity_star_interact
        else:
            infer_func = self.autoregressive_infer_cfg
        return infer_func(args=args, **kwargs)

    # ---------------------------
    # 流式 KV-cache 辅助函数：在线闭环服务会导出/导入这些 cache。
    # cache key 约定：'t0' 是文本前缀，整数 si 是第 si 个视觉尺度，
    # 'gt_obs' / 其他字符串是外部写入的真实观测或条件；is_pred 标记决定清理时是否保留。
    # ---------------------------
    def set_cache_write_is_pred(self, is_pred: bool):
        """把当前写入的是预测 token 还是 GT/text token 的标记同步到所有 attention block。"""
        for b in self.unregistered_blocks:
            b.attn.set_cache_write_is_pred(is_pred)

    def clear_pred_cache(self):
        """只清理所有 block 中预测 token 对应的 KV cache，保留可复用的上下文 cache。"""
        for b in self.unregistered_blocks:
            b.attn.clear_pred_cache()

    def export_kv_cache(self):
        """导出所有 block 的 KV cache，便于在线会话跨步复用。"""
        return [b.attn.export_kv_cache() for b in self.unregistered_blocks]

    def import_kv_cache(self, caches, overwrite: bool = True):
        """把外部保存的 KV cache 写回每个 block。"""
        assert len(caches) == len(self.unregistered_blocks)
        for b, c in zip(self.unregistered_blocks, caches):
            b.attn.import_kv_cache(c, overwrite=overwrite)

    def embeds_codes2input(
        self,
        last_stage, # 输入 latent code，形状 [B, d, t, h, w]
        repeat=1,
    ):
        """把 VAE latent code tensor 展平并嵌入成 Transformer 下一步输入 token。"""
        if self.apply_spatial_patchify: # 空间 patchify：把 2x2 空间邻域折到通道维。
            last_stage = last_stage.permute(0,2,1,3,4) # [B, t, d, 2h, 2w]
            last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, t, 4d, h, w]
            last_stage = last_stage.permute(0,2,1,3,4) # [B, 4d, t, h, w]
        last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # 展平成 [B, d, t*h*w] 或 [B, 4d, t*h*w]
        last_stage = torch.permute(last_stage, [0,2,1]) # 转成 Transformer token 布局 [B, token数, 通道]
        last_stage = self.word_embed(self.norm0_ve(last_stage))
        last_stage = last_stage.repeat(repeat, 1, 1)
        return last_stage

    @torch.no_grad()
    def ar_infer_infinity_elegant(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None,
        g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        low_vram_mode=False,
        args=None,
        get_visual_rope_embeds=None,
        context_info=None,
        return_summed_code_only=False,
        kv_cache_reset: bool = True,
        skip_text_forward: bool = False,
        cache_text_as_gt: bool = False,
        extra_ref_text_scale_inds: Optional[List[Union[int, str]]] = None,
        **kwargs,
    ):   # 返回采样 token 或 decoded video，取决于 return_summed_code_only。
        """
        Infinity Elegant schedule 的主推理路径。

        逐尺度采样 latent bit labels，按 VAE quantizer 转成 code 后累加到 `summed_codes`。
        在线闭环服务常用 `return_summed_code_only=True`，避免解码 RGB，直接把 latent 给动作头。

        公式/流程导读：
        - scale_schedule = [(pt, ph, pw), ...]，第 si 个尺度会生成 pt*ph*pw 个视觉 token；
        - 自回归顺序是从粗到细：x_si -> logits -> sample idx -> codes -> summed_codes；
        - CFG/APG 在 logits 上做条件分支和无条件分支的组合；
        - KV cache 用 scale_ind 区分 key：'t0' 是文本，整数 si 是视觉尺度，字符串 key 可指向 GT 观测。
        """
        from infinity.schedules.infinity_elegant import interpolate
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        assert args.use_cfg + args.use_apg == 1
        device = label_B_or_BLT[0].device
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        # 准备本次推理需要的 RoPE cache 和文本 prefix token。
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(device)
        text_maxlen_this_iter = label_B_or_BLT[-1] # compact 文本 tuple 的最后一项是当前 batch 最大文本长度
        prefix_tokens, lens = self.prepare_text_conditions(label_B_or_BLT, cfg_list, B, negative_label_B_or_BLT, vae_scale_schedule, text_token_only=False, text_maxlen_this_iter=text_maxlen_this_iter)
        bs = prefix_tokens.shape[0]
        ca_kv, cond_BD_or_gss, attn_mask = None, None, None
        ret, idx_Bl_list = [], []  # 兼容旧返回格式；主路径主要使用 summed_codes。
        # 开启 KV cache 后，每个 block 会按 scale_ind 写入 K/V；kv_cache_reset=False 时保留旧 key。
        for b in self.unregistered_blocks: b.attn.kv_caching(True, reset=kv_cache_reset)
        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        image_scale_repetition = np.array(json.loads(args.image_scale_repetition))
        video_scale_repetition = np.array(json.loads(args.video_scale_repetition))
        scales_in_one_clip = first_full_spatial_size_scale_index + 1
        assert len(image_scale_repetition) == len(video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
        assert len(image_scale_repetition) == scales_in_one_clip, f'{len(image_scale_repetition)} != {scales_in_one_clip}'
        total_steps = image_scale_repetition.sum() + video_scale_repetition.sum() * (len(scale_schedule)//len(video_scale_repetition)-1)
        if not skip_text_forward:
            total_steps += 1  # 额外计算一次文本 prefix token 的 forward/cache。
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks

        noise_shape = vae_scale_schedule[0]
        if self.other_args.noise_input:
            noise = torch.randn((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)
        else:
            noise = torch.zeros((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)

        summed_codes = [noise[0:1]]
        sos_token = self.embeds_codes2input(noise, bs//1)
        # 文本 token 前向传播：cache key 是 't0'；在线服务会把它作为 GT cache 固定下来。
        if not skip_text_forward:
            if cache_text_as_gt:
                self.set_cache_write_is_pred(False)
            rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:text_maxlen_this_iter]
            last_stage = prefix_tokens
            pbar.update(1)
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind='t0', context_info=context_info, last_repetition_step=True)
            if cache_text_as_gt:
                self.set_cache_write_is_pred(True)

        # 视觉 token 自回归前向传播：按 scale_schedule 逐尺度采样 latent code。
        # ref_text_scale_inds 是当前视觉尺度允许额外 attention 的 cache key 列表。
        ref_text_scale_inds = ['t0']
        if extra_ref_text_scale_inds:
            ref_text_scale_inds.extend(extra_ref_text_scale_inds)
        last_stage = sos_token
        cum_scales = 0
        for si, pn in enumerate(scale_schedule):   # si 表示当前尺度/clip 内位置。
            rel_si_in_one_clip = si % scales_in_one_clip
            if si < scales_in_one_clip: # 第一组尺度是 image/首帧相关尺度。
                repeat_times = image_scale_repetition[si%scales_in_one_clip]
                target_pn = vae_scale_schedule[first_full_spatial_size_scale_index]
            else:
                repeat_times = video_scale_repetition[si%scales_in_one_clip]
                target_pn = vae_scale_schedule[-1]
            cfg = cfg_list[si]
            infer_repeat_times = min(repeat_times, args.max_repeat_times)
            for repeat_idx in range(infer_repeat_times):
                # 调试用：真实展开后的尺度步编号是 cum_scales + repeat_idx。
                rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule, si, cum_scales+repeat_idx, device, args, context_info, first_full_spatial_size_scale_index)
                pbar.update(1)
                last_repetition_step = (repeat_idx == (infer_repeat_times-1))
                for block_idx, b in enumerate(block_chunks):
                    last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds)
                logits_BlV = self.get_logits_during_infer(last_stage, is_semantic_scale=rel_si_in_one_clip < args.semantic_scales).mul(1/tau_list[si])
                if cfg != 1:
                    # 在 logits 上做 CFG/APG 引导，增强文本条件约束。
                    if args.use_cfg:
                        # CFG 线性公式：guided = cfg*cond + (1-cfg)*uncond。
                        logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                    elif args.use_apg:
                        pred_cond = logits_BlV[:B]
                        pred_uncond = logits_BlV[B:]
                        pred_guided = normalized_guidance(pred_cond, pred_uncond, guidance_scale=cfg, momentum_buffer=None, eta=0, norm_threshold=args.apg_norm_threshold)
                        # 普通 CFG 可写成线性组合；APG 会额外做归一化约束。
                        logits_BlV = pred_guided
                else:
                    logits_BlV = logits_BlV[:B]

                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)
                probs_Bld = logits_BlV.softmax(dim=-1) # 每个 bit label 的二分类概率。
                idx_Bld = torch.multinomial(probs_Bld.view(-1, self.num_of_label_value), num_samples=1, replacement=True, generator=rng).view(tmp_bs, -1) # 采样后的展平 bit labels。
                probs_Bld = torch.gather(probs_Bld, dim=2, index=idx_Bld.unsqueeze(-1)).squeeze(-1)

                def Bld2Bthwd(item):
                    """把展平的 bit-label token 还原成 `(B,t,h,w,d)` latent 网格。"""
                    item = item.reshape(tmp_bs, tmp_seq_len, -1) # [B, thw, d] 或 patchify 后 [B, thw, 4d]
                    item = item.reshape(B, pn[0], pn[1], pn[2], -1) # 还原到 [B, t, h, w, d/4d]
                    if self.apply_spatial_patchify: # 反 patchify：把通道维 4d 还原回 2 倍空间分辨率。
                        item = item.permute(0,1,4,2,3) # [B, t, 4d, h, w]
                        item = torch.nn.functional.pixel_shuffle(item, 2) # [B, t, d, 2h, 2w]
                        item = item.permute(0,1,3,4,2) # [B, t, 2h, 2w, d]
                    return item

                idx_Bld = Bld2Bthwd(idx_Bld)
                probs_Bld = Bld2Bthwd(probs_Bld)
                # 调试用：查看当前尺度采样出的 bit-label 网格形状。

                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[cum_scales+repeat_idx]
                # idx_Bld 已经是时空 latent 网格，形状 [B,t,h,w,d] 或 patchify 还原后的 [B,t,2h,2w,d]。
                if self.other_args.use_two_stage_lfq:
                    if pn[1] * pn[2] >= vae.quantizer.detail_scale_min_tokens:
                        is_semantic_scale = False
                        lfq = vae.quantizer.lfq_detail
                    else:
                        is_semantic_scale = True
                        lfq = vae.quantizer.lfq_semantic
                    codes = lfq.indices_to_codes(idx_Bld, 'bit_label')
                    codes = interpolate(codes, size=(self.vae_embed_dim, *target_pn), mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                else:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, 'bit_label')
                    codes = F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)
                summed_codes[-1] = F.interpolate(summed_codes[-1], size=target_pn, mode=vae.quantizer.z_interplote_up)
                summed_codes[-1] += codes
                if repeat_idx < repeat_times - 1:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_down)
                    last_stage = self.embeds_codes2input(last_stage, bs//B)
            cum_scales += repeat_times
            if si < len(scale_schedule)-1:
                if scale_schedule[si][-2:] == scale_schedule[-1][-2:]:
                    if self.other_args.noise_input:
                        summed_codes.append(torch.randn((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    else:
                        summed_codes.append(torch.zeros((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    last_stage = summed_codes[-1]
                else:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_down)
                last_stage = self.embeds_codes2input(last_stage, bs//B)
        summed_codes = torch.cat(summed_codes, dim=-3)
        for b in self.unregistered_blocks: b.attn.kv_caching(False, reset=kv_cache_reset)
        if return_summed_code_only:
            return summed_codes
        else:
            if low_vram_mode: vae.to('cuda')
            img = self.summed_codes2images(vae, summed_codes)
            return idx_Bl_list, img


    @torch.no_grad()
    def ar_infer_infinity_star_interact(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None,
        g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        low_vram_mode=False,
        args=None,
        get_visual_rope_embeds=None,
        context_info=None,
        return_summed_code_only=False,
        mode='',
        former_clip_features=None,
        first_frame_features=None,
        semantic_scale_ind = 7,
        detail_frame_inds = [18,19],
        kv_cache_reset: bool = True,
        skip_text_forward: bool = False,
        cache_text_as_gt: bool = False,
        extra_ref_text_scale_inds: Optional[List[Union[int, str]]] = None,
        **kwargs,
    ):   # 返回交互式推理的 summed_codes 和 decoded video。
        """
        Infinity Star 交互式推理路径，支持前一 clip/首帧特征作为额外视觉条件。

        cache key 流程和 elegant 路径一致：'t0' 存文本，整数 si 存视觉尺度；
        交互式路径还可能写入 'semantic_condition' 和 'detail_condition' 作为额外视觉条件。
        """
        from infinity.schedules.infinity_star_interact import interpolate
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        assert args.use_apg + args.use_cfg == 1
        device = label_B_or_BLT[0].device
        if g_seed is None:
            rng = None
        else:
            self.rng = torch.Generator(device=device)
            self.rng.manual_seed(g_seed)
            rng = self.rng

        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        # 准备本次推理需要的 RoPE cache 和文本 prefix token。
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(device)
        text_maxlen_this_iter = label_B_or_BLT[-1] # 当前文本最大长度来自 compact 文本 tuple。
        prefix_tokens, _ = self.prepare_text_conditions(label_B_or_BLT, cfg_list, B, negative_label_B_or_BLT, vae_scale_schedule, text_token_only=False, text_maxlen_this_iter=text_maxlen_this_iter)
        bs = prefix_tokens.shape[0]

        ca_kv, cond_BD_or_gss, attn_mask = None, None, None
        for b in self.unregistered_blocks: b.attn.kv_caching(True, reset=kv_cache_reset)
        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        image_scale_repetition = np.array(json.loads(args.image_scale_repetition))
        video_scale_repetition = np.array(json.loads(args.video_scale_repetition))
        scales_in_one_clip = first_full_spatial_size_scale_index + 1
        assert len(image_scale_repetition) == len(video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
        assert len(image_scale_repetition) == scales_in_one_clip, f'{len(image_scale_repetition)} != {scales_in_one_clip}'
        total_steps = image_scale_repetition.sum() + video_scale_repetition.sum() * (len(scale_schedule)//len(video_scale_repetition)-1)
        if not skip_text_forward:
            total_steps += 1  # 额外计算一次文本 prefix token 的 forward/cache。
        if mode == 'second_v_clip':
            total_steps += 2
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks

        noise_shape = vae_scale_schedule[0]
        if self.other_args.noise_input:
            noise = torch.randn((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)
        else:
            noise = torch.zeros((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)

        summed_codes = [noise[0:1]]
        sos_token = self.embeds_codes2input(noise, bs//1)
        # 可选文本 token 前向传播，用于提前写入 KV cache。
        if not skip_text_forward:
            if cache_text_as_gt:
                self.set_cache_write_is_pred(False)
            rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:text_maxlen_this_iter]
            last_stage = prefix_tokens
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=f't0', context_info=context_info, last_repetition_step=True)
            pbar.update(1)
            if cache_text_as_gt:
                self.set_cache_write_is_pred(True)

        ref_text_scale_inds = ['t0']
        if extra_ref_text_scale_inds:
            ref_text_scale_inds.extend(extra_ref_text_scale_inds)

        # 可选视觉条件前向传播：第二段视频可引用上一段 clip 的特征。
        if mode == 'second_v_clip':
            # 用 args.frames_inner_clip 定义压缩时间上的 clip，不包含边界帧。
            # 历史约定：frames_inner_clip=20 时，former_clip_features 的 T=21（边界 + 20）。
            clip_inner_t = int(getattr(args, "frames_inner_clip", 20))
            clip_total_t = clip_inner_t + 1  # 总长度 = 边界帧 + clip 内部帧。
            assert former_clip_features.shape[-3] == clip_total_t, f"former_clip_features 的时间维 T 期望为 {clip_total_t}，实际为 {former_clip_features.shape=}"
            # 丢掉边界帧，只保留上一段 clip 内部特征。
            former_clip_features = former_clip_features[:, :, 1:]
            last_stage = F.interpolate(
                former_clip_features,
                size=(clip_inner_t, *vae_scale_schedule[semantic_scale_ind][-2:]),
                mode=vae.quantizer.z_interplote_down,
            )
            rope_cache = get_visual_rope_embeds(
                self.rope2d_freqs_grid,
                scale_schedule[-1],
                last_stage.shape[-3:],
                list(range(1, clip_inner_t + 1)),
                800,
                device,
            )
            last_stage = self.embeds_codes2input(last_stage, bs//B)
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=f'semantic_condition', context_info=context_info, last_repetition_step=True)
            pbar.update(1)

            # detail 条件使用首帧 + 上一段内部 clip 的最后 2 帧。
            if clip_inner_t >= 2:
                detail_frame_inds = [clip_inner_t - 2, clip_inner_t - 1]
            else:
                detail_frame_inds = [clip_inner_t - 1]
            last_stage = torch.cat([first_frame_features, former_clip_features[:, :, detail_frame_inds]], dim=2)
            rope_cache = get_visual_rope_embeds(
                self.rope2d_freqs_grid,
                scale_schedule[-1],
                last_stage.shape[-3:],
                [0] + [item + 1 for item in detail_frame_inds],
                801,
                device,
            )
            last_stage = self.embeds_codes2input(last_stage, bs//B)
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=f'detail_condition', context_info=context_info, last_repetition_step=True)
            pbar.update(1)

            ref_text_scale_inds.extend(['semantic_condition', 'detail_condition'])

        # 视觉 token 自回归前向传播。
        last_stage = sos_token
        cum_scales = 0
        for si, pn in enumerate(scale_schedule):   # si 表示当前尺度/clip 内位置。
            rel_si_in_one_clip = si % scales_in_one_clip
            if si < scales_in_one_clip: # 第一组尺度是 image/首帧相关尺度。
                repeat_times = image_scale_repetition[rel_si_in_one_clip]
                target_pn = vae_scale_schedule[first_full_spatial_size_scale_index]
            else:
                repeat_times = video_scale_repetition[rel_si_in_one_clip]
                target_pn = vae_scale_schedule[-1]
            cfg = cfg_list[si]
            infer_repeat_times = min(repeat_times, args.max_repeat_times)
            for repeat_idx in range(infer_repeat_times):
                frame_ss, frame_ee = context_info[si]['frame_ss'], context_info[si]['frame_ee']
                rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule[-1], scale_schedule[si], list(range(frame_ss, frame_ee)), cum_scales+repeat_idx, device)
                last_repetition_step = (repeat_idx == (infer_repeat_times-1))
                for block_idx, b in enumerate(block_chunks):
                    last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds)
                logits_BlV = self.get_logits_during_infer(last_stage, is_semantic_scale=rel_si_in_one_clip < args.semantic_scales).mul(1/tau_list[si])
                if cfg != 1:
                    # 在 logits 上做 CFG/APG 引导，增强文本条件约束。
                    if args.use_cfg:
                        # CFG 线性公式：guided = cfg*cond + (1-cfg)*uncond。
                        logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                    elif args.use_apg:
                        pred_cond = logits_BlV[:B]
                        pred_uncond = logits_BlV[B:]
                        pred_guided = normalized_guidance(pred_cond, pred_uncond, guidance_scale=cfg, momentum_buffer=None, eta=0, norm_threshold=args.apg_norm_threshold)
                        # 普通 CFG 可写成线性组合；APG 会额外做归一化约束。
                        logits_BlV = pred_guided
                else:
                    logits_BlV = logits_BlV[:B]

                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)
                probs_Bld = logits_BlV.softmax(dim=-1) # 每个 bit label 的二分类概率。
                idx_Bld = torch.multinomial(probs_Bld.view(-1, self.num_of_label_value), num_samples=1, replacement=True, generator=rng).view(tmp_bs, -1) # 采样后的展平 bit labels。
                probs_Bld = torch.gather(probs_Bld, dim=2, index=idx_Bld.unsqueeze(-1)).squeeze(-1)

                def Bld2Bthwd(item):
                    """把采样到的展平 bit labels 还原到时空 latent 网格。"""
                    item = item.reshape(tmp_bs, tmp_seq_len, -1) # [B, thw, d] 或 patchify 后 [B, thw, 4d]
                    item = item.reshape(B, pn[0], pn[1], pn[2], -1) # 还原到 [B, t, h, w, d/4d]
                    if self.apply_spatial_patchify: # 反 patchify：把通道维 4d 还原回 2 倍空间分辨率。
                        item = item.permute(0,1,4,2,3) # [B, t, 4d, h, w]
                        item = torch.nn.functional.pixel_shuffle(item, 2) # [B, t, d, 2h, 2w]
                        item = item.permute(0,1,3,4,2) # [B, t, 2h, 2w, d]
                    return item

                idx_Bld = Bld2Bthwd(idx_Bld)
                probs_Bld = Bld2Bthwd(probs_Bld)

                if si < gt_leak:
                    acc = (idx_Bld==gt_ls_Bl[cum_scales+repeat_idx]).float().mean() * 100.
                    idx_Bld = gt_ls_Bl[cum_scales+repeat_idx]
                    print(f'{si=} {repeat_idx=} idx_Bld.shape={idx_Bld.shape} {acc=}%')

                # idx_Bld 已经是时空 latent 网格，形状 [B,t,h,w,d] 或 patchify 还原后的 [B,t,2h,2w,d]。
                if self.other_args.use_two_stage_lfq:
                    if si >= args.semantic_scales:
                        is_semantic_scale = False
                        lfq = vae.quantizer.lfq_detail
                    else:
                        is_semantic_scale = True
                        lfq = vae.quantizer.lfq_semantic
                    codes = lfq.indices_to_codes(idx_Bld, 'bit_label')
                    codes = interpolate(codes, size=(self.vae_embed_dim, *target_pn), mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                else:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, 'bit_label')
                    codes = F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)
                summed_codes[-1] = F.interpolate(summed_codes[-1], size=target_pn, mode=vae.quantizer.z_interplote_up)
                summed_codes[-1] += codes
                if repeat_idx < repeat_times - 1:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_down)
                    last_stage = self.embeds_codes2input(last_stage, bs//B)
                pbar.update(1)
            cum_scales += repeat_times
            if si < len(scale_schedule)-1:
                if scale_schedule[si][-2:] == scale_schedule[-1][-2:]:
                    if self.other_args.noise_input:
                        summed_codes.append(torch.randn((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    else:
                        summed_codes.append(torch.zeros((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    last_stage = summed_codes[-1]
                else:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_down)
                last_stage = self.embeds_codes2input(last_stage, bs//B)
        summed_codes = torch.cat(summed_codes, dim=-3)
        for b in self.unregistered_blocks: b.attn.kv_caching(False, reset=kv_cache_reset)
        if mode == 'second_v_clip':
            # 把压缩时间长度换算回像素帧长度，用于截取当前 clip 的 decoded video。
            this_clip_frames = summed_codes.shape[2] * int(getattr(args, "temporal_compress_rate", 4))
            summed_codes = torch.cat([former_clip_features, summed_codes], dim=-3)
            img = self.summed_codes2images(vae, summed_codes) # 代码/形状说明：[bs, t, h, w, 3]，BGR uint8。
            img = img[:,-this_clip_frames:]
            clip_inner_t = int(getattr(args, "frames_inner_clip", 20))
            clip_total_t = clip_inner_t + 1
            summed_codes = summed_codes[:, :, -clip_total_t:]
            assert summed_codes.shape[2] == clip_total_t, f'形状不正确：{summed_codes.shape=}'
        else:
            img = self.summed_codes2images(vae, summed_codes)

        if low_vram_mode: vae.to('cuda')
        return summed_codes, img

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None,
        g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
        returns_vemb=0,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        low_vram_mode=False,
        args=None,
        get_visual_rope_embeds=None,
        **kwargs,
    ):   # 旧路径返回采样索引和 decoded video。
        """旧版 CFG 自回归推理路径，保留给非 elegant schedule 或兼容实验使用。"""
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        assert args.use_cfg + args.use_apg == 1
        device = label_B_or_BLT[0].device
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        # 准备本次推理的 RoPE cache。
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(device)
        text_maxlen_this_iter = self.text_maxlen
        last_stage, lens, _ = self.prepare_text_conditions(label_B_or_BLT, cfg_list, B, negative_label_B_or_BLT, args.input_noise, vae_scale_schedule)
        bs = last_stage.shape[0]
        ca_kv, cond_BD_or_gss = None, None
        ret, idx_Bl_list = [], []  # 兼容旧返回格式。
        for b in self.unregistered_blocks: b.attn.kv_caching(True)
        summed_codes = 0
        for si, pn in enumerate(scale_schedule):   # si 表示当前尺度。
            visual_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule, si, device, args)
            if si == 0:
                rope_cache = torch.cat([self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:text_maxlen_this_iter], visual_rope_cache], dim=4)
            else:
                rope_cache = visual_rope_cache
            attn_mask = torch.ones((last_stage.shape[0], 1, last_stage.shape[1], text_maxlen_this_iter+np.array(pn).prod()), device=last_stage.device).bool() # q_heads=1，依靠广播扩展到所有 head。
            assert len(attn_mask) == len(lens)
            for tmp_i, le in enumerate(lens):
                attn_mask[tmp_i, :, :, le:text_maxlen_this_iter] = False
                if si == 0:
                    attn_mask[tmp_i, :, :text_maxlen_this_iter, text_maxlen_this_iter:] = False
            cfg = cfg_list[si]
            if si >= trunk_scale:
                break
            for block_idx, b in enumerate(self.block_chunks):
                for m in b.module:
                    last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si)
            if si == 0:
                last_stage = last_stage[:, text_maxlen_this_iter:]
            # 调试断点保留位置：可在这里查看单尺度 attention mask 和 logits。
            if cfg != 1:
                # 在 logits 上做 CFG/APG 引导。
                logits_BlV = self.get_logits(last_stage).mul(1/tau_list[si])
                if args.use_cfg:
                    # CFG 线性公式：guided = cfg*cond + (1-cfg)*uncond。
                    logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                elif args.use_apg:
                    pred_cond = logits_BlV[:B]
                    pred_uncond = logits_BlV[B:]
                    pred_guided = normalized_guidance(pred_cond, pred_uncond, guidance_scale=cfg, momentum_buffer=None, eta=0, norm_threshold=10)
                    # 普通 CFG 可写成线性组合；APG 会额外做归一化约束。
                    logits_BlV = pred_guided
            else:
                logits_BlV = self.get_logits(last_stage[:B]).mul(1/tau_list[si])
            if self.num_of_label_value == 1:
                idx_Bld = logits_BlV
            elif self.num_of_label_value > 1:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            elif self.num_of_label_value == 0:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            assert returns_vemb
            if si < gt_leak:
                idx_Bld = gt_ls_Bl[si]
            else:
                idx_Bld = idx_Bld.reshape(B, pn[0], pn[1], pn[2], -1) # 还原到 [B, t, h, w, d/4d]
                if self.apply_spatial_patchify: # 反 patchify：把通道维 4d 还原回 2 倍空间分辨率。
                    idx_Bld = idx_Bld.permute(0,1,4,2,3) # [B, t, 4d, h, w]
                    idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, t, d, 2h, 2w]
                    idx_Bld = idx_Bld.permute(0,1,3,4,2) # [B, t, 2h, 2w, d]
                # idx_Bld 已经是时空 latent 网格。

            # 旧版本会把每个尺度的 idx_Bld 放入列表，这里保留变量用于兼容。
            if self.num_of_label_value == 1:
                if si < gt_leak:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, label_type='bit_label') # 转成 [B,d,t,h,w] latent code。
                else:
                    codes = idx_Bld.permute(0,4,1,2,3)
            else:
                codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, label_type='bit_label') # 转成 [B,d,t,h,w] latent code。
            if vae_scale_schedule[si] != vae_scale_schedule[-1]:
                codes = F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            summed_codes += codes
            if si < len(scale_schedule)-1:
                last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_down) # 下采样到下一尺度。
                if self.apply_spatial_patchify: # 空间 patchify：把 2x2 空间邻域折到通道维。
                    last_stage = last_stage.permute(0,2,1,3,4) # [B, t, d, 2h, 2w]
                    last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, t, 4d, h, w]
                    last_stage = last_stage.permute(0,2,1,3,4) # [B, 4d, t, h, w]
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # 展平成 [B, d/4d, t*h*w]
                last_stage = torch.permute(last_stage, [0,2,1]) # 转成 [B, token数, 通道]
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)
        for b in self.unregistered_blocks: b.attn.kv_caching(False)
        if low_vram_mode: vae.to('cuda')
        img = self.summed_codes2images(vae, summed_codes)
        return ret, idx_Bl_list, img

    def summed_codes2images(self, vae, summed_codes):
        """把累计 latent `summed_codes` 通过 VAE decode 成 uint8 BGR 视频帧。"""
        t1 = time.time()

        img = vae.decode(summed_codes, slice=True)
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        img = img.permute(0,2,3,4,1) # 从 [bs,3,t,h,w] 转成视频常用布局 [bs,t,h,w,3]
        img = img.mul_(255).to(torch.uint8).flip(dims=(4,))

        # 简单平滑首帧：避免第一帧 decode 抖动影响视频观感。
        img[:, 0:1, :, :, :] = img[:, 1:2, :, :, :]

        print(f'Decode 耗时 {time.time()-t1:.1f}s')
        return img

    @for_visualize
    def vis_key_params(self, ep):
        """可视化装饰器回调占位，目前不额外返回关键参数。"""
        return

    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        """兼容不同文本长度/旧 buffer 名称的 checkpoint 加载。"""
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]

        for buf_name in ('lvl_1L', 'attn_bias_for_masking', 'Infinity_visible_kvlen', 'Infinity_invisible_qlen'):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)

        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

    def special_init(self):
        """按 Qwen 风格初始化 Linear/Embedding 权重。"""
        if self.arch == 'qwen':
            std = 0.02
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
        else:
            raise ValueError(f'未知的 arch：{self.arch}')

    def extra_repr(self):
        """保持模块 repr 简洁，避免打印超长配置。"""
        return f''

    def get_layer_id_and_scale_exp(self, para_name: str):
        """预留给分层学习率/scale decay 的接口，当前 Infinity 未实现。"""
        raise NotImplementedError


def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # 返回形状为 (B,l,num_samples) 的采样 id
    """对 logits 原地应用 top-k/top-p 过滤后 multinomial 采样 token id。"""
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # multinomial 只能处理 2D 分布，因此先把 B 和 l 合并再采样。
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def sampling_with_top_k_top_p_also_inplace_modifying_probs_(probs_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # 返回形状为 (B,l,num_samples) 的采样 id
    """对概率分布原地应用 top-k/top-p 过滤并重新归一化后采样。"""
    B, l, V = probs_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = probs_BlV < probs_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        probs_BlV.masked_fill_(idx_to_remove, 0)
    if top_p > 0:
        sorted_probs, sorted_idx = probs_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_probs.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        probs_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 0)
    # multinomial 只能处理 2D 分布，因此先把 B 和 l 合并再采样。
    probs_BlV = probs_BlV / probs_BlV.sum(-1, keepdims=True)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(probs_BlV.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def get_params_num(d, w, mlp):
    """按深度、宽度和 MLP ratio 粗略估计 Infinity 参数规模。"""
    m = round(mlp * w / 256) * 256
    s = d * (w**2 * 8 + w*m * 2)    # self-attn/cross-attn 和 MLP 的粗略参数量
    s += w**2 * 6       # 自适应归一化相关参数
    s += 4096 * w       # 预测 head
    s += 32 * w         # word embedding 近似项

    Ct5 = 4096
    s += Ct5*w * 4      # T5 文本注意力池化近似项
    s += Ct5*w + w*w    # T5 文本投影 MLP 近似项
    return f'{s/1e9:.2f}B'


TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool', 'cache_dir'}

@register_model
def infinity_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs):
    """注册 2B 级 Infinity 配置，供 timm.create_model 按名称构建。"""
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa2b(depth=28, block_chunks=7, embed_dim=2560, num_heads=2560//128, drop_path_rate=0.1, **kwargs):
    """注册分块 self-attention 2B 级 Infinity 配置。"""
    return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa8b(depth=42, block_chunks=7, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs):
    """注册 8B 级分块 Infinity 配置。"""
    return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa14b(depth=40, block_chunks=8, embed_dim=5120, num_heads=5120//128, drop_path_rate=0.1, mlp_ratio=3.4, **kwargs):
    """注册 14B 级分块 Infinity 配置。"""
    return Infinity(
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )
    # 备用配置记录：曾使用 GQA KV heads 的 14B 变体。

@register_model
def infinity_sa12b(depth=60, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs):
    """注册深层 12B 级 Infinity 配置。"""
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa16b(depth=42, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs):
    """注册 16B 命名的 Infinity 配置变体。"""
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_v2b(depth=32, embed_dim=2016, num_heads=2016//126, drop_path_rate=0.1, **kwargs):
    """注册早期 v2B Infinity 配置。"""
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_8b(depth=40, block_chunks=1, embed_dim=3584, num_heads=3584//128, drop_path_rate=0.1, **kwargs):
    """注册 8B Infinity 基础配置。"""
    return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_qwen7b(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096//128, num_key_value_heads=4096//128//4, mlp_ratio=12288/4096, drop_path_rate=0, **kwargs):
    """注册 Qwen 风格 7B 配置，使用 GQA KV heads。"""
    return Infinity(
        arch='qwen',
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen8b(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096//128, num_key_value_heads=4096//128//4, mlp_ratio=4, drop_path_rate=0, **kwargs):
    """注册 WorldVLN 常用的 Qwen 风格 8B Infinity 配置。"""
    return Infinity(
        arch='qwen',
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen_wide14b(depth=36, block_chunks=6, embed_dim=5632, num_heads=5632//128, num_key_value_heads=5632//128//4, drop_path_rate=0, **kwargs):
    """注册更宽 hidden size 的 Qwen 14B 配置。"""
    return Infinity(
        arch='qwen',
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.4,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen13bMHA(depth=40, block_chunks=8, embed_dim=5120, num_heads=5120//128, num_key_value_heads=5120//128, drop_path_rate=0, **kwargs):
    """注册 Qwen 13B MHA 配置，KV heads 等于 attention heads。"""
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=True,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.4,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen2_2b(depth=28, block_chunks=7, embed_dim=2304, num_heads=2304//128, num_key_value_heads=2304//128, drop_path_rate=0, **kwargs):
    """注册小型 Qwen2 风格 2B 配置。"""
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen0b(depth=4, block_chunks=2, embed_dim=512, num_heads=512//128, num_key_value_heads=512//128, drop_path_rate=0, **kwargs):
    """注册极小 Qwen 配置，主要用于 smoke test 或结构调试。"""
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen2_30b(depth=54, block_chunks=27, embed_dim=6144, num_heads=6144//128, num_key_value_heads=6144//128//4, drop_path_rate=0, **kwargs):
    """注册 Qwen2 风格 30B 大模型配置。"""
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=4, # 旧实验用过 mlp_ratio=3.55，这里固定为 4。
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen14b(depth=48, block_chunks=24, embed_dim=4608, num_heads=4608//128, num_key_value_heads=4608//128//4, drop_path_rate=0, **kwargs):
    """注册 Qwen 14B 配置。"""
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=4,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_20b(depth=58, embed_dim=4608, num_heads=4608//128, drop_path_rate=0.25, **kwargs):
    """注册 20B 级 Infinity 配置。"""
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

# 用于缩放实验的小/中型 Infinity Transformer 配置。
@register_model
def infinity_layer12(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, **kwargs):
    """注册 12 层小型缩放实验配置。"""
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer16(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, **kwargs):
    """注册 16 层缩放实验配置。"""
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer24(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, **kwargs):
    """注册 24 层缩放实验配置。"""
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer32(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, **kwargs):
    """注册 32 层缩放实验配置。"""
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer40(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, **kwargs):
    """注册 40 层缩放实验配置。"""
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer48(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, **kwargs):
    """注册 48 层缩放实验配置。"""
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
