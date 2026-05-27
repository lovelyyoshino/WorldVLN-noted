# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import os
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

def interpolate(tensor, size, mode, quantizer, is_semantic_scale):
    """
    把 5 维视频特征插值到指定尺寸，并在需要时同时调整通道数。

    参数:
        tensor: 输入特征，形状为 (B,C,T,H,W)。
        size: 目标形状 (C1,T,H1,W1)，其中 C1 是目标通道数。
        mode: 插值方式，例如 nearest 或 trilinear。
        quantizer: VAE 的量化器，提供通道投影层和插值配置。
        is_semantic_scale: 当前尺度是否属于语义尺度；否则按细节尺度处理。
    返回:
        调整后的特征，形状为 (B,*size)。
    """
    B, C, T, H, W = tensor.shape
    C1, T, H1, W1 = size
    if quantizer.other_args.use_learnable_dim_proj:
        if is_semantic_scale:
            if C > C1:
                proj = quantizer.semantic_proj_down
            elif C < C1:
                proj = quantizer.semantic_proj_up
        else:
            if C > C1:
                proj = quantizer.detail_proj_down
            elif C < C1:
                proj = quantizer.detail_proj_up
        if C != C1:
            tensor = tensor.permute(0,2,3,4,1) # 把通道放到最后，便于线性投影: (B,C,T,H,W) -> (B,T,H,W,C)
            tensor = proj(tensor) # 用语义/细节投影层把通道数变成 C1: (B,T,H,W,C1)
            tensor = tensor.permute(0,4,1,2,3) # 再把通道放回第二维: (B,T,H,W,C1) -> (B,C1,T,H,W)
        tensor = F.interpolate(tensor, size=(T, H1, W1), mode=mode) # 只缩放时间/空间尺寸: (B,C1,T,H,W) -> (B,C1,T,H1,W1)
        return tensor
    else:
        tensor = tensor.permute(0,2,1,3,4) # 没有可学习投影时，把 T 临时放到通道前: (B,C,T,H,W) -> (B,T,C,H,W)
        tensor = F.interpolate(tensor, size=(C1, H1, W1), mode=mode)
        tensor = tensor.permute(0,2,1,3,4) # 插值后恢复标准布局: (B,T,C1,H1,W1) -> (B,C1,T,H1,W1)
    return tensor

def get_scale_pack_info(scale_schedule, first_full_spatial_size_scale_index, args):
    """
    为 Infinity Elegant 的多尺度生成顺序整理每个尺度的帧范围和可参考尺度。

    scale_schedule 按 clip 展开，每个 clip 内有若干从粗到细的尺度。
    这个函数会记录每个尺度属于哪个 clip、覆盖哪些压缩帧，以及它在注意力里
    可以看见哪些历史尺度，后续会用这些信息构造 querysid_refsid 掩码。
    """
    meta = {}
    sid2clipid_innsid = {}
    clipid_innsid2sid = {}
    scales_per_clip = first_full_spatial_size_scale_index + 1
    compress_frames_inner_clip = args.frames_inner_clip
    total_clips = len(scale_schedule) // scales_per_clip
    context_clips = args.context_frames // args.frames_inner_clip
    for si in range(len(scale_schedule)):
        clipid = si // scales_per_clip
        if clipid == 0:
            frame_ss, frame_ee = 0, scale_schedule[scales_per_clip*(clipid+1)-1][0] # 第一个 clip 的压缩帧起止位置
        else:
            frame_ss = scale_schedule[0][0] + (clipid-1) * compress_frames_inner_clip
            frame_ee = frame_ss + scale_schedule[scales_per_clip*(clipid+1)-1][0]
            if context_clips < total_clips-1:
                assert scale_schedule[si][0] == compress_frames_inner_clip
        sid2clipid_innsid[si] = (clipid, si % scales_per_clip)
        clipid_innsid2sid[(clipid, si % scales_per_clip)] = si
        # 记录当前尺度所属 clip，并准备它可以参考的历史 clip。
        if si <= first_full_spatial_size_scale_index:
            meta[si] = {
                'clipid': clipid,
                'frame_ss': frame_ss,
                'frame_ee': frame_ee,
                'left_ref': [-1],
                'right_ref': [-1],
            }
        else:
            meta[si] = {
                'clipid': clipid,
                'frame_ss': frame_ss,
                'frame_ee': frame_ee,
                # 把历史 clip 当作“记忆”来参考。
                # 默认先列出所有更早的 clip，下面再用 context_clips 截断数量。
                # 这样更接近流式推理：过去看到的内容就是当前生成的记忆。
                'left_ref': list(range(clipid-1, -1, -1)),
                'right_ref': [-1],
            }
            meta[si]['left_ref'] = meta[si]['left_ref'][:context_clips]
        # 如果只想从每个历史 clip 的大尺度取上下文，这里把 clip 编号和 clip 内尺度编号打包在一起。
        if args.context_from_largest_no > 0:
            meta[si]['left_ref'] = [(meta[si]['left_ref'][i], max(0, scales_per_clip - args.context_from_largest_no - args.context_interval*i)) for i in range(len(meta[si]['left_ref']))]
            meta[si]['right_ref'] = [(meta[si]['right_ref'][i], max(0, scales_per_clip - args.context_from_largest_no - args.context_interval*i)) for i in range(len(meta[si]['right_ref']))]
    for si in meta:
        if args.context_from_largest_no > 0:
            meta[si]['left_ref_sids'], meta[si]['right_ref_sids'] = [], []
            for clipid, innsid in (meta[si]['left_ref']):
                if clipid != -1:
                    meta[si]['left_ref_sids'].append(clipid_innsid2sid[(clipid, innsid)])
            for fid, innsid in (meta[si]['right_ref']):
                if fid != -1:
                    meta[si]['right_ref_sids'].append(clipid_innsid2sid[(clipid, innsid)])
            meta[si]['ref_sids'] = meta[si]['left_ref_sids'] + meta[si]['right_ref_sids']
        else:
            meta[si]['ref_sids'] = list(range(si))
    return meta


def video_encode(
    vae,
    inp_B3HW,
    vae_features=None,
    self_correction=None,
    device='cuda',
    args=None,
    infer_mode=False,
    rope2d_freqs_grid=None,
    dynamic_resolution_h_w=None,
    tokens_remain=9999999,
    text_lens=[],
    **kwargs,
):
    """
    对外的编码入口，保持旧接口不变，并转到全局 BSC 编码实现。

    这里不直接写逻辑，是为了让调用方继续使用 video_encode 这个名字，
    同时把真正的 Infinity Elegant 多尺度打包流程集中在
    video_encode_global_bsc 中维护。
    """
    return video_encode_global_bsc(
        vae,
        inp_B3HW,
        vae_features,
        self_correction,
        device,
        args,
        infer_mode,
        rope2d_freqs_grid,
        dynamic_resolution_h_w,
        tokens_remain,
        text_lens,
        **kwargs,
    )

def video_encode_global_bsc(
    vae,
    inp_B3HW,
    vae_features=None,
    self_correction=None,
    device='cuda',
    args=None,
    infer_mode=False,
    rope2d_freqs_grid=None,
    dynamic_resolution_h_w=None,
    tokens_remain=9999999,
    text_lens=[],
    **kwargs,
):
    """
    将输入视频特征编码成 Infinity Elegant 训练/推理需要的多尺度 token 序列。

    核心流程是：先按分辨率和帧数找到 scale_schedule，再按尺度从粗到细量化残差；
    每个尺度会保存预测索引、教师索引、VAR 输入、视觉 RoPE 和可参考尺度信息。
    训练时返回打包后的 x_BLC、标签、RoPE、尺度长度和注意力参考掩码；
    推理时提前返回噪声、重建图和索引，供解码阶段复用。

    公式导读：
    - scale_schedule = [(pt, ph, pw), ...]，每一项的 token 数是 pt*ph*pw；
    - 列表顺序从粗到细，越靠后的尺度空间网格越大、细节越多；
    - sequence packing 会把多个尺度的 token 串起来，长度约为 sum(pt*ph*pw)+text_len。
    """
    if vae_features is None:
        raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW, scale_schedule=None, slice=True)
        raw_features_list = [raw_features]
        x_recon_raw = vae.decode(raw_features, slice=True)
        x_recon_raw = torch.clamp(x_recon_raw, min=-1, max=1)
        print(f'raw_features.shape: {raw_features.shape}')
    else:
        raw_features_list = vae_features
    # raw_features_list 是样本列表，每个元素通常形如 [1,d,t,h,w]。
    gt_all_bit_indices = []
    pred_all_bit_indices = []
    var_input_list = []
    sequece_packing_scales = [] # 每个样本实际保留下来的多尺度表，包含主干尺度。
    flatten_packing_scales = []
    h_div_w_template_list = np.array(list(dynamic_resolution_h_w.keys()))
    visual_rope_cache_list = []
    noise_list = []
    scale_pack_info_list = []
    image_scale_repetition = json.loads(args.image_scale_repetition)
    video_scale_repetition = json.loads(args.video_scale_repetition)
    scales_in_one_clip = dynamic_resolution_h_w[h_div_w_template_list[0]][args.pn]['scales_in_one_clip']
    other_info_by_scale = []
    tokens_remain = tokens_remain-sum(text_lens)
    examples = len(raw_features_list)
    assert len(image_scale_repetition) == len(video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
    with torch.amp.autocast('cuda', enabled = False):
        for example_ind, raw_features in enumerate(raw_features_list):
            t, h, w = raw_features.shape[-3:]
            h_div_w = h / w
            mapped_h_div_w_template = h_div_w_template_list[np.argmin(np.abs(h_div_w-h_div_w_template_list))]
            min_t = min(dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'].keys())
            image_scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'][min_t]
            # 按 latent 的 h/w 最近邻选择模板，再用 t 查出从粗到细的 (pt,ph,pw) 尺度表。
            scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'][t]

            if args.apply_spatial_patchify:
                vae_scale_schedule = [(pt, ph + (ph % 2), pw + (pw % 2)) for pt, ph, pw in scale_schedule]
            else:
                vae_scale_schedule = scale_schedule
            first_full_spatial_size_scale_index = len(image_scale_schedule) - 1
            scale_pack_info = get_scale_pack_info(vae_scale_schedule, first_full_spatial_size_scale_index, args)
            scale_pack_info_list.append(scale_pack_info)

            if raw_features.dim() == 4:
                codes_out = raw_features.unsqueeze(2) # 单帧特征补上时间维，变成 [B, d, t, h, w]。
            else:
                codes_out = raw_features # 视频特征本来就是 [B, d, t, h, w]。
            # 调试时可打印 raw_features.shape 和 scale_schedule，默认保持关闭。
            v_d = codes_out.shape[1]
            B, C, T, H, W = codes_out.shape
            if args.noise_input:
                noise = torch.randn((B, v_d, *vae_scale_schedule[0]), device=device, dtype=raw_features.dtype)
            else:
                noise = torch.zeros((B, v_d, *vae_scale_schedule[0]), device=device, dtype=raw_features.dtype)
            if infer_mode: noise_list.append(noise)
            next_var_input = noise
            valid_scales = len(vae_scale_schedule)
            assert len(image_scale_repetition) == len(image_scale_schedule), f'{len(image_scale_repetition)} != {len(image_scale_schedule)}'
            real_si = 0
            noise_apply_strength = self_correction.noise_apply_strength
            if args.noise_apply_random_one:
                image_scale_cnt = len(image_scale_schedule)
                video_scale_cnt = len(vae_scale_schedule)
                keep_image_si = random.randint(0, image_scale_cnt-1)
                if video_scale_cnt == image_scale_cnt:
                    keep_video_si = keep_image_si
                else:
                    keep_video_si = random.randint(image_scale_cnt, video_scale_cnt-1)
                noise_apply_strength = [noise_prob if i == keep_image_si or i == keep_video_si else 0 for i, noise_prob in enumerate(noise_apply_strength)]
            for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
                tokens_remain = tokens_remain - np.array(scale_schedule[si]).prod()
                if tokens_remain < 0 and (not args.allow_less_one_elem_in_seq or examples > 1):
                    valid_scales = si
                    break

                rel_si_in_one_clip = si % len(image_scale_schedule)
                if si < len(image_scale_schedule): # 图像/首帧尺度使用 image_scale_repetition。
                    repeat_times = image_scale_repetition[rel_si_in_one_clip]
                else:
                    repeat_times = video_scale_repetition[rel_si_in_one_clip]
                select_repeat_idx = np.random.randint(0, repeat_times)
                frame_ss, frame_ee = scale_pack_info[si]['frame_ss'], scale_pack_info[si]['frame_ee']
                target = codes_out[:,:,frame_ss:frame_ee]
                for repeat_idx in range(repeat_times):
                    if (not infer_mode) and (repeat_idx==select_repeat_idx):
                        visual_rope_cache_list.append(get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule, si, real_si, device, args, scale_pack_info, first_full_spatial_size_scale_index))

                    if next_var_input.shape[-3:] != target.shape[-3:]:
                        next_var_input = F.interpolate(next_var_input, size=target.shape[-3:], mode=vae.quantizer.z_interplote_up).contiguous()
                    cum_var_input = next_var_input
                    this_scale_var_input = F.interpolate(cum_var_input, size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_down).contiguous()
                    if repeat_idx > 0 and args.inner_scale_boost:
                        residual = residual - quantized
                    else:
                        residual = target - cum_var_input
                    if args.use_two_stage_lfq:
                        if ph * pw >= vae.quantizer.detail_scale_min_tokens:
                            is_semantic_scale = False
                            C1 = vae.quantizer.detail_scale_dim
                            lfq = vae.quantizer.lfq_detail
                        else:
                            is_semantic_scale = True
                            C1 = vae.quantizer.semantic_scale_dim
                            lfq = vae.quantizer.lfq_semantic
                        residual = interpolate(residual, size=(C1, *vae_scale_schedule[si]), mode=vae.quantizer.z_interplote_down, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                    else:
                        residual = F.interpolate(residual, size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_down).contiguous()
                        try:
                            lfq = vae.quantizer.lfq_detail
                        except:
                            lfq = vae.quantizer.lfq
                    quantized, _, bit_indices, loss = lfq(residual) # quantized 是重建残差 [B,d,t,h,w]，bit_indices 是离散码 [B,t,h,w,d]。

                    if args.reduce_accumulate_error_method == 'bsc':
                        if si < min(len(vae_scale_schedule)-1, self_correction.noise_apply_layers):
                            pred_bit_indices, quantized = self_correction.apply_noise_requant(bit_indices, quantized, args, device, si, lfq, noise_apply_strength)
                        else:
                            pred_bit_indices = bit_indices
                    else:
                        raise NotImplementedError(args.reduce_accumulate_error_method)

                    if infer_mode or (repeat_idx==select_repeat_idx):
                        pred_all_bit_indices.append(pred_bit_indices)
                        var_input_list.append(this_scale_var_input)
                        gt_all_bit_indices.append(bit_indices)
                        other_info_by_scale.append({'largest_scale': scale_schedule[-1], 'real_si': si})
                    if args.use_two_stage_lfq:
                        quantized_scaled = interpolate(quantized, size=target.shape[-4:], mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                    else:
                        quantized_scaled = F.interpolate(quantized, size=target.shape[-3:], mode=vae.quantizer.z_interplote_up).contiguous()
                    next_var_input = cum_var_input + quantized_scaled
                    real_si += 1

                if si < len(vae_scale_schedule)-1: # 第一个尺度相当于起始噪声；这里只需要为后续尺度准备累计输入。
                    if vae_scale_schedule[si][-2:] == vae_scale_schedule[-1][-2:]:
                        if args.noise_input:
                            next_var_input = torch.randn((B, v_d, *vae_scale_schedule[si+1]), device=device, dtype=raw_features.dtype)
                        else:
                            next_var_input = torch.zeros((B, v_d, *vae_scale_schedule[si+1]), device=device, dtype=raw_features.dtype)
                        if infer_mode: noise_list.append(next_var_input)

            sequece_packing_scales.append(scale_schedule[:valid_scales])
            flatten_packing_scales.extend(scale_schedule[:valid_scales])
            if infer_mode:
                return noise_list, x_recon_raw, pred_all_bit_indices, None, None, scale_pack_info

    # 当完整尺度 token 超过训练上限时，只训练部分尺度，让 480p 等大分辨率样本也能进入训练。
    if args.allow_less_one_elem_in_seq and len(sequece_packing_scales) == 1 and np.array(sequece_packing_scales[0]).prod(-1).sum() > args.train_max_token_len:
        scale_schedule = sequece_packing_scales[0]

        if args.train_with_var_seq_len:
            if len(scale_schedule) == scales_in_one_clip * 4: # 49 帧 clip4 表示 4 个 clip：1 个图像/首帧 clip + 3 个视频 clip。
                S = scales_in_one_clip
                outcomes = [
                        # --- 只选 clip 0，也就是图像/首帧部分 ---
                        lambda: list(range(S)),
                        # --- clip 0 加上 clip 1 的语义尺度 ---
                        lambda: list(range(S + 8)),
                        lambda: list(range(S + 11)),
                        # --- clip 0 加上 clip 1 的细节尺度 ---
                        lambda: list(range(S + 11)) + [S+11],
                        lambda: list(range(S + 11)) + [S+12],
                        lambda: list(range(S + 11)) + [S+13],
                        # --- 跨 clip：clip0 锚点 -> clip1 锚点 -> clip2 语义尺度 ---
                        lambda: [S-1] + [2*S-1] + list(range(2*S, 2*S + 11)),
                        # --- 跨 clip：clip0 锚点 -> clip1 锚点 -> clip2 细节尺度 ---
                        lambda: [S-1] + [2*S-1] + [2*S + 11],
                        lambda: [S-1] + [2*S-1] + [2*S + 12],
                        lambda: [S-1] + [2*S-1] + [2*S + 13],
                        # --- 跨 clip：clip0 锚点 -> clip1 锚点 -> clip2 锚点 -> clip3 语义尺度 ---
                        lambda: [S-1] + [2*S-1] + [3*S-1] + list(range(3*S, 3*S + 11)),
                        # --- 跨 clip：clip0 锚点 -> clip1 锚点 -> clip2 锚点 -> clip3 细节尺度 ---
                        lambda: [S-1] + [2*S-1] + [3*S-1] + [3*S + 11],
                        lambda: [S-1] + [2*S-1] + [3*S-1] + [3*S + 12],
                        lambda: [S-1] + [2*S-1] + [3*S-1] + [3*S + 13],
                        # --- 完整语义：4 个 clip 的语义尺度都参与，增强跨 clip 连贯性 ---
                        lambda: list(range(S)) + list(range(S, S+11)) + list(range(2*S, 2*S+11)) + list(range(3*S, 3*S+11)),
                    ]
            elif len(scale_schedule) == scales_in_one_clip * 3: # 训练 10 秒视频时的候选尺度组合。
                outcomes = [
                        lambda: list(range(scales_in_one_clip)),
                        lambda: list(range(scales_in_one_clip + 8)),
                        lambda: list(range(scales_in_one_clip + 11)),
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+11],
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+12],
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+13],
                        lambda: [scales_in_one_clip-1] + [2*scales_in_one_clip-1] + list(range(2*scales_in_one_clip, 2*scales_in_one_clip + 11)),
                        lambda: [scales_in_one_clip-1] + [2*scales_in_one_clip-1] + list(range(2*scales_in_one_clip, 2*scales_in_one_clip + 11)),
                        lambda: [scales_in_one_clip-1] + [2*scales_in_one_clip-1] + [2*scales_in_one_clip + 11],
                        lambda: [scales_in_one_clip-1] + [2*scales_in_one_clip-1] + [2*scales_in_one_clip + 12],
                        lambda: [scales_in_one_clip-1] + [2*scales_in_one_clip-1] + [2*scales_in_one_clip + 13],
                    ]
            else:
                if args.drop_720p_last_scale:
                    outcomes = [
                        lambda: list(range(scales_in_one_clip)),
                        lambda: list(range(scales_in_one_clip + 8)),
                        lambda: list(range(scales_in_one_clip + 11)),
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+11],
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+12],
                        lambda: list(range(scales_in_one_clip + 8)) + [scales_in_one_clip+13],
                    ]
                else:
                    outcomes = [
                        lambda: list(range(scales_in_one_clip)),
                        lambda: list(range(scales_in_one_clip + 8)),
                        lambda: list(range(scales_in_one_clip + 11)),
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+11],
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+12],
                        lambda: [scales_in_one_clip-1] + [scales_in_one_clip+13],
                        lambda: [scales_in_one_clip-1] + [scales_in_one_clip+14],
                    ]

            # outcomes 候选数量会随 schedule 变化，例如 7 个或 11 个。
            # args.video_var_len_prob 可能给得更短；NumPy 要求候选和概率长度一致。
            # 因此这里截到共同长度，并重新归一化概率，避免采样报错。
            raw_probs = json.loads(args.video_var_len_prob)
            probabilities = np.array(raw_probs, dtype=np.float32)
            n = min(len(outcomes), len(probabilities))
            if n <= 0:
                # 兜底：没有可用概率时，总是选择第一个候选。
                select_si_list = outcomes[0]()
            else:
                outcomes = outcomes[:n]
                probabilities = probabilities[:n]
                s = float(probabilities.sum())
                if (not np.isfinite(s)) or s <= 0:
                    probabilities = np.ones(n, dtype=np.float32) / n
                else:
                    probabilities /= s
                # 按概率选中一个候选函数，并立即执行得到尺度编号列表。
                select_si_list = np.random.choice(outcomes, p=probabilities)()

        else:
            select_si_list = [scales_in_one_clip-1] # 必选首帧/上下文的最大尺度，作为后续视频尺度的锚点。
            if args.train_192pshort:
                # 可选：曾用于追加短视频的某个视频尺度，当前保留为调试参考。
                if args.train_192pshort > 1:
                    select_si_list = list(range(0, scales_in_one_clip+args.train_192pshort))
                else:
                    select_si_list = list(range(0, scales_in_one_clip+11))
            else:
                select_si_list = list(range(0, scales_in_one_clip)) # 首帧的所有尺度都必须选中，保证图像上下文完整。
                select_si_list.append(scales_in_one_clip + np.random.choice([11, 12, 13], p=[0.7, 0.2, 0.1]))

            other_si_list = list(range(scales_in_one_clip-1)) + list(range(scales_in_one_clip, 2*scales_in_one_clip))
            other_si_list = list(set(other_si_list) - set(select_si_list))
            np.random.shuffle(other_si_list)
            train_token_len = np.array(scale_schedule)[select_si_list].prod(-1).sum() + text_lens[0]
            for si in other_si_list:
                token_len = np.array(scale_schedule[si]).prod(-1).sum()
                if train_token_len + token_len <= args.train_max_token_len:
                    train_token_len += token_len
                    select_si_list.append(si)

            # 安全兜底：
            # 有些 schedule（例如总帧数较短的 clip16）里，必选或随机选中的视频尺度可能太大。
            # 为了始终不超过 train_max_token_len，优先丢弃可选的视频尺度，
            # 同时保留第一个 clip 的尺度作为锚点和上下文。
            selected_tokens = int(np.array(scale_schedule)[select_si_list].prod(-1).sum() + text_lens[0])
            if selected_tokens > args.train_max_token_len:
                first_clip_set = set(range(scales_in_one_clip))
                # 先丢 token 数最多的可选尺度，最快把序列长度降下来。
                optional_sorted = sorted(
                    [si for si in select_si_list if si not in first_clip_set],
                    key=lambda si: int(np.array(scale_schedule[si]).prod()),
                    reverse=True,
                )
                for si in optional_sorted:
                    if selected_tokens <= args.train_max_token_len:
                        break
                    select_si_list.remove(si)
                    selected_tokens = int(np.array(scale_schedule)[select_si_list].prod(-1).sum() + text_lens[0])

        select_si_list.sort()
        new_si_2_real_si, real_si_2_new_si = {}, {}
        for new_si, real_si in enumerate(select_si_list):
            new_si_2_real_si[new_si] = real_si
            real_si_2_new_si[real_si] = new_si

        sequece_packing_scales = [[scale_schedule[si] for si in select_si_list]]
        flatten_packing_scales = [flatten_packing_scales[si] for si in select_si_list]
        gt_all_bit_indices = [gt_all_bit_indices[si] for si in select_si_list]
        pred_all_bit_indices = [pred_all_bit_indices[si] for si in select_si_list]
        var_input_list = [var_input_list[si] for si in select_si_list]
        visual_rope_cache_list = [visual_rope_cache_list[si] for si in select_si_list]
        other_info_by_scale = [other_info_by_scale[si] for si in select_si_list]

        # 重新映射 scale_pack_info：删掉未选尺度后，参考关系也要改成新的尺度编号。
        new_scale_pack_info = {}
        for new_query_sid in new_si_2_real_si:
            real_query_sid = new_si_2_real_si[new_query_sid]
            new_scale_pack_info[new_query_sid] = {'ref_sids': []}
            for real_ref_sid in scale_pack_info_list[0][real_query_sid]['ref_sids']:
                # 注意：
                # 为了适配 train_max_token_len，我们可能只保留 select_si_list 里的部分尺度。
                # 原始 ref_sids 可能指向未保留的尺度，它们在 real_si_2_new_si 中不存在。
                # 这些引用需要安全丢弃；querysid_refsid 后面仍会补上自引用和文本引用。
                new_ref_sid = real_si_2_new_si.get(real_ref_sid, None)
                if new_ref_sid is None:
                    continue
                new_scale_pack_info[new_query_sid]['ref_sids'].append(new_ref_sid)
        scale_pack_info_list = [new_scale_pack_info]

    # 每个尺度展平成 token 后长度就是 pt*ph*pw；packing 总长度再加文本 token 和 padding。
    scale_lengths = [ pt * ph * pw for pt,ph,pw in flatten_packing_scales]
    scale_lengths = scale_lengths + text_lens
    valid_scales = len(flatten_packing_scales) + len(text_lens)

    cur_seq_len = np.sum(scale_lengths)
    if args.train_with_var_seq_len:
        pad_seq_len = int(np.ceil(cur_seq_len/args.pad_to_multiplier))*args.pad_to_multiplier - cur_seq_len
    else:
        pad_seq_len = args.train_max_token_len - cur_seq_len
    assert pad_seq_len >= 0, f'pad_seq_len: {pad_seq_len} < 0，{scale_lengths=}'
    if pad_seq_len:
        scale_lengths = scale_lengths + [pad_seq_len]
    max_sid_nums = 2000
    querysid_refsid = torch.zeros((max_sid_nums, max_sid_nums), device=args.device, dtype=torch.bool) # 注意：不同迭代里这个形状必须保持一致。
    for i in range(valid_scales):
        querysid_refsid[i][i] = True
    base = 0
    for ind, scale_schedule in enumerate(sequece_packing_scales):
        scale_pack_info = scale_pack_info_list[ind]
        for local_querysid in range(len(scale_schedule)):
            global_querysid = local_querysid + base
            global_text_sid = len(flatten_packing_scales) + ind
            querysid_refsid[global_querysid][global_text_sid] = True
            for local_refsid in (scale_pack_info[local_querysid]['ref_sids']):
                global_refsid = base + local_refsid
                querysid_refsid[global_querysid][global_refsid] = True
        base += len(scale_schedule)

    gt_ms_idx_Bl = []
    for item in gt_all_bit_indices:
        if args.apply_spatial_patchify:
            # item 的形状是 (B,t,H,W,d)。
            item = item.permute(0,1,4,2,3) # 把码本维 d 放到通道位: (B,t,d,H,W)。
            # 空间 patchify 会把 2x2 空间块折到通道里: (B,t,d,H,W) -> (B,t,4d,H/2,W/2)。
            item = torch.nn.functional.pixel_unshuffle(item, 2)
            _, tt, dd, hh, ww = item.shape
            # 再铺平成 token 序列: (B,t,4d,H/2,W/2) -> (B,t,H/2,W/2,4d) -> (B,t*H/2*w/2,4d)。
            item = item.permute(0,1,3,4,2).reshape(B, tt*hh*ww, dd)
        else:
            _, tt, hh, ww, dd = item.shape
            item = item.reshape(B, tt*hh*ww, dd)
        gt_ms_idx_Bl.append(item.type(torch.long))
    gt_BLC = gt_ms_idx_Bl # 保持逐尺度标签列表；如需拼接，可用 torch.cat(gt_ms_idx_Bl, 1)。
    for i in range(len(var_input_list)):
        if args.apply_spatial_patchify:
            # VAR 输入也按同样规则 patchify: (B,d,t,H,W) -> (B,t,d,H,W) -> (B,t,4d,H/2,W/2) -> (B,t,H/2,W/2,4d)。
            var_input_list[i] = torch.nn.functional.pixel_unshuffle(var_input_list[i].permute(0,2,1,3,4), 2).permute(0,1,3,4,2)
            var_input_list[i] = var_input_list[i].reshape(B, -1, 4*vae.codebook_dim)
        else:
            # 不做 patchify 时，只把通道维移到最后: (B,d,t,H,W) -> (B,t,H,W,d)。
            var_input_list[i] = var_input_list[i].permute(0,2,3,4,1)
            var_input_list[i] = var_input_list[i].reshape(B, -1, vae.codebook_dim)
    x_BLC = torch.cat(var_input_list, 1)
    visual_rope_cache = torch.cat(visual_rope_cache_list, dim=4)
    x_BLC_mask = None
    return x_BLC, x_BLC_mask, gt_BLC, pred_all_bit_indices, visual_rope_cache, sequece_packing_scales, scale_lengths, querysid_refsid, other_info_by_scale


def video_decode(
    vae,
    all_indices,
    scale_schedule,
    label_type,
    args=None,
    noise_list=None,
    trunc_scales=-1,
    **kwargs,
):
    """
    按 Infinity Elegant 的多尺度顺序，把离散索引逐尺度还原成视频特征并解码成图像。

    解码时会复用编码阶段保存的 noise_list，从粗尺度开始逐步叠加每个尺度的码本残差；
    每个尺度可能重复多次，对应 image_scale_repetition 或 video_scale_repetition。
    最后把各段特征在时间维拼接，并交给 VAE decoder 输出重建视频。
    """
    image_scale_repetition = json.loads(args.image_scale_repetition)
    video_scale_repetition = json.loads(args.video_scale_repetition)
    assert len(image_scale_repetition) == len(video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
    real_si = 0
    noise_ptr = 0
    summed_codes = [noise_list[noise_ptr]]
    noise_ptr += 1
    v_d = summed_codes[0].shape[1]
    for si, (pt, ph, pw) in enumerate(scale_schedule):
        if trunc_scales > 0 and si >= trunc_scales:
            break
        if si < len(image_scale_repetition): # 图像/首帧尺度使用 image_scale_repetition。
            repeat_times = image_scale_repetition[si%len(image_scale_repetition)]
        else:
            repeat_times = video_scale_repetition[si%len(image_scale_repetition)]
        for repeat_idx in range(repeat_times):
            tgt_shape = (pt, scale_schedule[-1][-2], scale_schedule[-1][-1])
            if args.use_two_stage_lfq:
                if ph * pw >= vae.quantizer.detail_scale_min_tokens:
                    is_semantic_scale = False
                    lfq = vae.quantizer.lfq_detail
                else:
                    is_semantic_scale = True
                    lfq = vae.quantizer.lfq_semantic
                codes = lfq.indices_to_codes(all_indices[real_si], label_type)
                codes = interpolate(codes, size=(v_d, *tgt_shape), mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
            else:
                codes = vae.quantizer.lfq_detail.indices_to_codes(all_indices[real_si], label_type)
                codes = F.interpolate(codes, size=tgt_shape, mode=vae.quantizer.z_interplote_up).contiguous()

            summed_codes[-1] = F.interpolate(summed_codes[-1], size=tgt_shape, mode=vae.quantizer.z_interplote_up).contiguous()
            summed_codes[-1] += codes
            real_si += 1
        if si < len(scale_schedule) - 1:
            if scale_schedule[si][-3:] == tgt_shape:
                summed_codes.append(noise_list[noise_ptr])
                noise_ptr += 1
    if trunc_scales < 0:
        assert real_si == len(all_indices), f'all_repeated_scales={real_si} 与 len(all_indices)={len(all_indices)} 不一致'
    summed_codes = torch.cat(summed_codes, dim=-3)
    x_recon = vae.decode(summed_codes, slice=True)
    x_recon = torch.clamp(x_recon, min=-1, max=1)
    return x_recon

def get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule, sid, real_sid, device=None, args=None, scale_pack_info=None, first_full_spatial_size_scale_index=None):
    """
    为当前尺度生成视觉 RoPE 位置编码，覆盖尺度、帧、高度和宽度四个方向。

    Infinity Elegant 把一个视频 token 的位置拆成四部分：它属于第几个生成尺度、
    对应哪些压缩帧、位于哪个高度网格、位于哪个宽度网格。这里把四部分频率拼起来，
    得到后续 Transformer 可以直接使用的 RoPE 缓存。
    """
    # freqs_scales 记录尺度方向的频率，形状为 (2, max_scales, ceil(dim_div_2 / 4))。
    # freqs_frames 记录帧方向的频率，形状为 (2, max_frames, ceil(dim_div_2 / 4))。
    rope2d_freqs_grid['freqs_scales'] = rope2d_freqs_grid['freqs_scales'].to(device)
    rope2d_freqs_grid['freqs_frames'] = rope2d_freqs_grid['freqs_frames'].to(device)
    rope2d_freqs_grid['freqs_height'] = rope2d_freqs_grid['freqs_height'].to(device)
    rope2d_freqs_grid['freqs_width'] = rope2d_freqs_grid['freqs_width'].to(device)
    upt, uph, upw = scale_schedule[-1]
    pt, ph, pw = scale_schedule[sid]
    dim_div_2_div_4 = rope2d_freqs_grid['freqs_scales'].shape[2]
    dim_div_2 = dim_div_2_div_4 * 4
    f_scales = rope2d_freqs_grid['freqs_scales'][:, real_sid].reshape(2, 1, dim_div_2_div_4)
    frame_ss, frame_ee = scale_pack_info[sid]['frame_ss'], scale_pack_info[sid]['frame_ee']
    f_frames = rope2d_freqs_grid['freqs_frames'][:, frame_ss:frame_ee]
    f_height = rope2d_freqs_grid['freqs_height'][:, (torch.arange(ph) * (uph / ph)).round().int()]
    f_width = rope2d_freqs_grid['freqs_width'][:, (torch.arange(pw) * (upw / pw)).round().int()]
    rope_embeds = torch.cat([
        f_scales[   :,     :,  None,   None,   None,   :].expand(-1, -1, pt, ph, pw, -1),
        f_frames[   :,  None,      :,  None,   None,   :].expand(-1,  1, -1, ph, pw, -1),
        f_height[   :,  None,  None,      :,   None,   :].expand(-1,  1, pt, -1, pw, -1),
        f_width[    :,  None,  None,   None,      :,   :].expand(-1,  1, pt, ph, -1, -1),
    ], dim=-1)  # 拼接后每个 token 都有完整的四方向位置频率: (2,1,pt,ph,pw,dim_div_2)。
    rope_embeds = rope_embeds.reshape(2, 1, 1, 1, 1*pt*ph*pw, dim_div_2)  # 展平成 Transformer 使用的 token 维度。
    return rope_embeds
