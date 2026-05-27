# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
#!/usr/bin/python3
from __future__ import annotations

from typing import Any

import torch

from infinity.models import Infinity

def load_visual_tokenizer(args, device=None):
    """中文说明：`load_visual_tokenizer` 根据 `vae_type`/`videovae` 实例化视觉 VAE tokenizer，并把它放到目标设备。

    新手提示：这里不会做训练 checkpoint 恢复，而是直接构建“视觉编码/解码器”这一半模型。
    阅读重点：先看 `vae_type` 和 `videovae` 怎样决定具体实现，再看返回的 `vae_local` 如何被 GPT/Infinity 复用。
    """
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.vae_type in [8,12,14,16,18,20,24,32,48,64,128]:
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type # 18
        print(f'从 {args.vae_path} 加载视觉 VAE/tokenizer')

        if args.videovae == 10: # 使用 absorb patchify 变体。
            from infinity.models.videovae.models.load_vae_bsq_wan_absorb_patchify import video_vae_model
            vae_local = video_vae_model(args.vae_path, schedule_mode, codebook_dim, global_args=args, test_mode=True).to(device)
        else:
            raise ValueError(f"vae_type {args.vae_type} 暂不支持")
    else:
        raise ValueError(f"vae_type {args.vae_type} 暂不支持")
    return vae_local

def build_vae_gpt(args: Any, force_flash: bool = False, device: str = 'cuda'):
    """中文说明：`build_vae_gpt` 一次性构建视觉 VAE 和 Infinity/GPT 主模型，并把两者按当前参数拼起来。

    新手提示：它本身只负责“造模型骨架”，真正的 checkpoint 权重恢复通常发生在外层训练/推理脚本。
    阅读重点：先看 `gpt_kw` 怎么从 `args` 提炼出模型结构参数，再看 `create_model()` 返回的 `gpt_wo_ddp`。
    """
    vae_local = load_visual_tokenizer(args, device)

    if force_flash: args.flash = True
    gpt_kw = dict(
        text_channels=args.Ct5,
        text_maxlen=args.tlen,
        norm_eps=args.norm_eps,
        rms_norm=args.rms_norm,
        cond_drop_rate=args.cfg,
        rand_uncond=args.rand_uncond,
        raw_scale_schedule=args.scale_schedule,
        top_p=args.topp,
        top_k=args.topk,
        checkpointing=args.enable_checkpointing,
        pad_to_multiplier=args.pad_to_multiplier,
        use_flex_attn=args.use_flex_attn,
        add_lvl_embeding_on_first_block=args.add_lvl_embeding_on_first_block,
        num_of_label_value=args.num_of_label_value,
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        pn=args.pn,
        train_h_div_w_list=None,
        apply_spatial_patchify=args.apply_spatial_patchify,
        video_frames=args.video_frames,
        other_args=args,
    )

    print(f'[create gpt_wo_ddp] 构造参数 kw={gpt_kw}\n')
    gpt_kw['vae_local'] = vae_local

    model_str = args.model.replace('vgpt', 'infinity')   # 兼容旧版本逻辑。
    print(f"{model_str=}")
    if model_str.rsplit('c', maxsplit=1)[-1].isdecimal():
        model_str, _ = model_str.rsplit('c', maxsplit=1)
    from timm.models import create_model
    gpt_wo_ddp: Infinity = create_model(model_str, **gpt_kw)
    vae_local = vae_local.to('cuda')
    assert all(not p.requires_grad for p in vae_local.parameters())
    assert all(p.requires_grad for n, p in gpt_wo_ddp.named_parameters())
    return vae_local, gpt_wo_ddp
