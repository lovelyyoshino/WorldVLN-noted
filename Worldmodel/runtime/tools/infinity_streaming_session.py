# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
InfinityStar KV-cache 工作流的 streaming/session 封装。

把底层 InfinityStar 世界模型封装成“在线闭环 streaming session”

  reset(prompt)
    -> 写文本条件 t0

  compute_kv_cache_gt(real_obs)
    -> 把真实第一视角观测编码进 gt_obs cache

  infer_chunk()
    -> 基于 t0 + gt_obs 预测下一段 latent/video

  correction_clear_pred()
    -> 清掉预测 cache，只保留真实文本/观测 cache

目标：对齐 lingbot-va 的“计算 KV cache -> 推理当前 chunk -> 用真实观测校正”的闭环语义。

关键约定：
- 用 't0' 作为文本前缀 cache key，并按 GT 写入（is_pred=False），这样 clear_pred_cache() 不会删它。
- 用 'gt_obs' 作为观测帧 cache key，并按 GT 写入（is_pred=False）。
- 推理过程中写入的 KV-cache 条目都视作 Pred（is_pred=True），在 correction 阶段一次性清理。

中文导读：
- 这个文件解释 WorldVLN 在线闭环为什么不会“闭眼滚完整条路线”。文本前缀和真实观测
  写入 GT cache，预测时新增的 cache 标成 Pred。
- 每一轮 segment 推理结束后，`correction_clear_pred()` 会清理 Pred cache；下一轮再把真实
  新观测写回 `gt_obs`。因此模型的预测只服务当前动作，不会被永久当作历史事实。

给小白的核心比喻：
- `t0` 像任务说明书：一条路线开始时写一次，之后一直保留。
- `gt_obs` 像行车记录仪：客户端真实看到什么，就把这些真实帧编码后写进去。
- `Pred cache` 像草稿纸：模型为了预测下一段会临时写草稿；一段结束后必须擦掉。

如果不区分这三类 cache，闭环推理会很容易出错：
模型上一轮“猜”出来的未来会在下一轮继续被当成历史，误差会被不断放大。这个 session
封装的主要价值，就是把“真实历史”和“预测草稿”分开管理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from tools.run_infinity import encode_prompt


@dataclass
class StreamingSchedule:
    """
    一次 chunk 推理对应的动态分辨率计划、目标尺寸和 scale 元信息。

    阅读提示：
    - `scale_schedule` 告诉 InfinityStar 这一轮按哪些时间/空间尺度生成 token；
    - `context_info` 告诉 attention/RoPE 每个 scale 的 token 范围；
    - `tgt_h/tgt_w` 是 RGB 帧最终会被 resize 到的空间尺寸；
    - `tower_split_index` 用来区分 image tower 和 video tower，外层会据此拼 `tau_list`。
    """

    scale_schedule: List[Tuple[int, int, int]]
    context_info: Dict[int, Dict[str, Any]]
    tgt_h: int
    tgt_w: int
    tower_split_index: int
    first_full_spatial_size_scale_index: int


def _count_cache_entries(infinity_model) -> Tuple[int, int, int]:
    """统计所有 block 的 cache 条目数，返回 (total_entries, pred_entries, gt_entries)。"""
    total = pred = gt = 0
    for blk in infinity_model.unregistered_blocks:
        meta = getattr(blk.attn, "cached_is_pred", {})
        total += len(meta)
        pred += sum(1 for v in meta.values() if v)
        gt += sum(1 for v in meta.values() if not v)
    return total, pred, gt


class InfinityStreamingSession:
    """
    封装 InfinityStar 文本 cache、真实观测 cache、chunk 推理和预测 cache 清理。

    可以把它理解成服务端闭环推理的“四步机”：
    1. `reset()`：写入文本条件 `t0`；
    2. `compute_kv_cache_gt()`：写入真实观测 `gt_obs`；
    3. `infer_chunk()`：基于 `t0 + gt_obs` 预测下一段 latent，并把新增 cache 标成 Pred；
    4. `correction_clear_pred()`：清掉本轮预测 cache，避免把“想象的未来”留到下一轮。

    这样设计的核心目的是：
    允许模型短程预演世界变化，但下一轮必须重新依赖真实观测，不能让预测无限自我滚动。

    和 `infer/server.py` 的关系：
    - `server.py::_ensure_traj_infinity_session()` 为每条 `TrajectoryState` 创建一个本类实例；
    - `server.py::_infer_summed_codes_for_step()` 会调用本类的 schedule/cache 能力；
    - 服务端跨请求保存的是 `TrajectoryState.kv_cache`，而本类负责把它导入底层模型 block。
    """

    def __init__(
        self,
        *,
        args,
        infinity_model,
        vae,
        text_tokenizer,
        text_encoder,
        h_div_w_template: float = 0.571,
        gt_obs_cache_key: str = "gt_obs",
        gt_obs_rope_real_sid: int = 850,
    ):
        """保存模型组件并初始化动态分辨率编解码函数。"""
        self.args = args
        self.infinity = infinity_model
        self.vae = vae
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder

        self.h_div_w_template = float(h_div_w_template)
        self.gt_obs_cache_key = gt_obs_cache_key
        self.gt_obs_rope_real_sid = int(gt_obs_rope_real_sid)

        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = get_encode_decode_func(
            args.dynamic_scale_schedule
        )

        self._text_cond_tuple = None
        self.bs = 1  # cache 写入使用的 batch size：no-CFG 为 1，CFG 为 2。

    def build_schedule_for_num_frames(self, num_frames: int) -> StreamingSchedule:
        """
        为当前 chunk 构建 `scale_schedule / context_info`，并和官方推理脚本对齐。

        核心公式：
        `pt = (num_frames - 1)//temporal_compress_rate + 1`

        直觉解释：
        video VAE 不会为每个 RGB 帧都保留一个独立 latent 时间步，而是按时间压缩率分组。
        默认 `temporal_compress_rate=4` 时：
        - 1 帧对应 1 个 latent 时间步；
        - 17 帧对应 5 个 latent 时间步；
        - 81 帧对应 21 个 latent 时间步。

        这个 `pt` 会进一步决定动态分辨率表该选哪一条 `scale_schedule`。

        输出怎么看：
        - 返回的 `scale_schedule` 决定世界模型本轮输出的 latent 时间长度；
        - 返回的 `tgt_h/tgt_w` 会被服务端用来 resize 客户端上传的 RGB 帧；
        - 如果 `num_frames` 变小（tail-window），这里得到的 schedule 也会变短。

        """
        args = self.args
        # 动态分辨率表来自训练/推理配置。它按宽高比模板和 latent 时间步数组织，
        # 例如“0.5625 宽高比 + 21 个 latent 时间步”对应一条具体 scale_schedule。
        dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - self.h_div_w_template))]

        # 视频 token 时间轴会被时间压缩：pt = (frames-1)//4 + 1（默认 temporal_compress_rate=4）。
        pt = (num_frames - 1) // args.temporal_compress_rate + 1
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"][pt]

        # 第一个完整空间尺度之后通常进入 video tower；外层用 tower_split_index 把 image/video
        # 两段使用不同的 tau 温度，和官方 InfinityStar 推理脚本保持一致。
        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        args.first_full_spatial_size_scale_index = first_full_spatial_size_scale_index
        args.tower_split_index = first_full_spatial_size_scale_index + 1
        context_info = self.get_scale_pack_info(scale_schedule, first_full_spatial_size_scale_index, args)

        tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
        return StreamingSchedule(
            scale_schedule=scale_schedule,
            context_info=context_info,
            tgt_h=tgt_h,
            tgt_w=tgt_w,
            tower_split_index=args.tower_split_index,
            first_full_spatial_size_scale_index=first_full_spatial_size_scale_index,
        )

    @torch.no_grad()
    def reset(self, prompt: str, negative_prompt: str = "", cfg_scale: float = 1.0):
        """
        清空所有 KV cache，然后把文本前缀 cache（'t0'）按 GT 写入。

        中文导读：
        每条新 session 先把语言指令编码进 cache。`t0` 被标记为 GT，后续清理预测 cache
        时不会删掉它，所以同一条轨迹可以持续复用同一份语言条件。

        为什么要先写 `t0`：
        后续 `infer_chunk(skip_text_forward=True)` 不会再重复跑文本前向。如果没有这一步，
        模型虽然能看到视觉 cache，却看不到“要往哪里走”的语言条件。

        CFG 的 batch 语义：
        - `cfg_scale == 1.0` 时只有条件分支，`bs=1`；
        - `cfg_scale != 1.0` 时同时有条件/无条件分支，`bs=2`；
        - 真实观测 cache 也要 repeat 到同样 batch size，否则后续 attention batch 对不上。
        """
        args = self.args
        model_dtype = next(iter(self.infinity.parameters())).dtype
        self.bs = 2 if float(cfg_scale) != 1.0 else 1

        # 1) 重置所有 block 的 cache。reset 只在新 session 或新 run 开始时调用；
        # 普通 segment 推进不会调用它，否则历史真实观测会被清掉。
        for blk in self.infinity.unregistered_blocks:
            blk.attn.kv_caching(True, reset=True)

        # 2) 编码 prompt；如果使用 CFG，同时准备 cond/uncond。
        text_cond_tuple = encode_prompt(args.text_encoder_ckpt, self.text_tokenizer, self.text_encoder, prompt, enable_positive_prompt=False, low_vram_mode=False)
        if negative_prompt:
            neg_tuple = encode_prompt(args.text_encoder_ckpt, self.text_tokenizer, self.text_encoder, negative_prompt, enable_positive_prompt=False, low_vram_mode=False)
        else:
            neg_tuple = None
        self._text_cond_tuple = (text_cond_tuple, neg_tuple)

        # 3) 按 GT 写入 text cache，避免被 clear_pred_cache 清掉。
        # 这一行之后，底层 attention block 新写的 K/V 都会记录为 is_pred=False。
        self.infinity.set_cache_write_is_pred(False)

        # 复用模型 helper 构造 prefix tokens，然后用 scale_ind='t0' 前向传播一次。
        # 这对应 `ar_infer_infinity_*` 里的 "text tokens forward" 代码块。
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
        text_maxlen_this_iter = max_seqlen_k
        prefix_tokens, _ = self.infinity.prepare_text_conditions(
            label_B_or_BLT=text_cond_tuple,
            cfg_list=[float(cfg_scale)],
            B=1,
            # 注意：这里要和后续 skip_text_forward=True 的推理保持一致。
            # 如果提供 negative_prompt 且 cfg_scale != 1，无条件分支应使用
            # negative prompt tokens，而不是 cfg_uncond。
            negative_label_B_or_BLT=neg_tuple,
            vae_scale_schedule=None,
            text_token_only=False,
            text_maxlen_this_iter=text_maxlen_this_iter,
        )

        device = prefix_tokens.device
        self.infinity.rope2d_freqs_grid["freqs_text"] = self.infinity.rope2d_freqs_grid["freqs_text"].to(device)
        rope_cache = self.infinity.rope2d_freqs_grid["freqs_text"][:, :, :, :, :text_maxlen_this_iter]

        block_chunks = self.infinity.block_chunks if getattr(self.infinity, "num_block_chunks", 1) > 1 else self.infinity.blocks
        last_stage = prefix_tokens.to(dtype=model_dtype)
        with torch.amp.autocast("cuda", dtype=model_dtype):
            for b in block_chunks:
                # scale_ind='t0' 是文本 cache 的稳定 key。后续视觉 token 可以引用它，
                # 但 `clear_pred_cache()` 不会删除它。
                last_stage = b(
                    x=last_stage,
                    cond_BD=None,
                    ca_kv=None,
                    attn_bias_or_two_vector=None,
                    attn_fn=None,
                    scale_schedule=None,
                    rope2d_freqs_grid=rope_cache.to(dtype=model_dtype),
                    scale_ind="t0",
                    context_info=None,
                    last_repetition_step=True,
                    ref_text_scale_inds=[],
                )

        # reset 结束后恢复默认写 Pred。之后推理产生的新 cache 默认都当作临时预测。
        self.infinity.set_cache_write_is_pred(True)

    @torch.no_grad()
    def compute_kv_cache_gt(self, obs_video_bcthw: torch.Tensor):
        """
        Step 1 / Step 4：把已观测帧编码成 latent tokens，并写入 GT cache。

        输入形状：
        `obs_video_bcthw = [B,3,T,H,W]`

        写入规则：
        - cache key 使用 `gt_obs_cache_key`，默认就是 `gt_obs`；
        - `is_pred=False`，表示这是可信的真实观测，不会被 `clear_pred_cache()` 删除。

        中文导读：
        这里写入的是“真实已经看到的第一视角观测”，不是上一步预测出来的 latent。
        闭环纠偏正是靠这个函数把真实视频前缀重新锚定到 Transformer KV cache 中。
        你可以把它理解成：每次客户端回传了真实新帧，服务端都要重新说一遍
        “下面这些视觉 token 才是历史事实”。

        为什么 GT cache 也必须带 RoPE：
        Transformer 的 KV cache 里不只存“这个 token 的内容是什么”，还隐含“这个 token 在视频
        时间/空间网格里的位置”。Infinity 的视觉注意力使用 RoPE 表示位置；如果真实观测写入
        cache 时不带对应的 RoPE，后续生成阶段虽然能读到 `gt_obs` 的 K/V 内容，却不知道这些
        token 属于第几帧、哪个空间 patch、哪个 scale。结果就是预测 token attend 真实观测时
        位置坐标对不上，轻则时空关系变乱，重则和模型训练时的注意力分布不一致。

        这个函数的主要作用：
        把“客户端刚刚确认过的真实历史”编码成视觉 latent，并用正确 RoPE 写入 `gt_obs` KV cache。
        后续 `infer_chunk()` 预测未来 chunk 时，会把 `t0` 文本 cache 和 `gt_obs` 真实观测 cache
        一起作为上下文；预测结束后只清理 Pred cache，不清理这里写入的 GT cache。

        何时会被调用：
        - 默认闭环：某个 segment 的右边界真实帧已经到达后，服务端调用它推进真实历史；
        - hybrid leak 模式：真实前缀既作为 leak 条件，也写入 `gt_obs` cache；
        - future emission：提前输出动作时不会调用它，因为那些未来帧还不是真实观测。

        """
        assert obs_video_bcthw.ndim == 5 and obs_video_bcthw.shape[1] == 3, "obs_video_bcthw 期望形状为 [B,3,T,H,W]"
        device = next(iter(self.infinity.parameters())).device
        dtype = next(iter(self.infinity.parameters())).dtype

        obs_video_bcthw = obs_video_bcthw.to(device=device, dtype=torch.float32)
        # VAE 需要输入 float，范围 [-1,1]。这里不做随机增强，也不做训练时的 teacher forcing；
        # 它只把真实 RGB 前缀变成世界模型可读的 latent token。
        features, _, _ = self.vae.encode_for_raw_features(obs_video_bcthw, scale_schedule=None, slice=True)  # [B,d,t,h,w]

        # VAE 输出的 latent 时间长度同样满足 t_latent = (T_obs-1)//4 + 1（默认压缩率 4）。
        pt, ph, pw = features.shape[-3:]
        scale_schedule = [(pt, ph, pw)]
        mini_scale_pack_info = {0: {"frame_ss": 0, "frame_ee": pt}}

        # 在预计算范围内选择一个合法的 RoPE scale index。
        # 这里的 RoPE 不是为了“重新编码内容”，而是给真实观测 token 标注时空位置：
        # - pt/ph/pw 告诉模型这批 token 覆盖多少 latent 帧和空间 patch；
        # - real_sid 选择预计算好的视觉 RoPE 频率表；
        # - mini_scale_pack_info 告诉 RoPE helper 当前 cache 片段对应的帧范围。
        # 没有这一步，`gt_obs` cache 只剩内容向量，后续自回归生成时无法正确对齐真实历史的位置。
        max_scales = int(self.infinity.rope2d_freqs_grid["freqs_scales"].shape[1])
        real_sid = min(self.gt_obs_rope_real_sid, max_scales - 1)

        rope_cache = self.get_visual_rope_embeds(
            self.infinity.rope2d_freqs_grid,
            scale_schedule,
            0,  # sid
            real_sid,
            device,
            self.args,
            mini_scale_pack_info,
            0,  # 第一个完整空间尺寸 scale 的索引。
        )

        # 写入 GT cache。
        # 从这里开始到写完 block 前向，所有新 K/V 都会被标记为真实观测。
        self.infinity.set_cache_write_is_pred(False)
        # repeat 到和 bs 一致；CFG 使用 bs=2。
        last_stage = self.infinity.embeds_codes2input(features.to(dtype=dtype), repeat=self.bs)
        block_chunks = self.infinity.block_chunks if getattr(self.infinity, "num_block_chunks", 1) > 1 else self.infinity.blocks
        with torch.amp.autocast("cuda", dtype=dtype):
            for b in block_chunks:
                last_stage = b(
                    x=last_stage,
                    cond_BD=None,
                    ca_kv=None,
                    attn_bias_or_two_vector=None,
                    attn_fn=None,
                    scale_schedule=None,
                    rope2d_freqs_grid=rope_cache.to(dtype=dtype),
                    scale_ind=self.gt_obs_cache_key,
                    context_info=None,
                    last_repetition_step=True,
                    ref_text_scale_inds=[],
                )
        # 写完真实观测后立刻恢复 Pred 默认值，避免后续生成 cache 被误标成 GT。
        self.infinity.set_cache_write_is_pred(True)

    @torch.no_grad()
    def infer_chunk(
        self,
        *,
        num_frames: int,
        cfg_list: List[float],
        tau_list: List[float],
        top_k: int = 0,
        top_p: float = 0.0,
        seed: Optional[int] = None,
        negative_prompt: str = "",
        low_vram_mode: bool = True,
        gt_leak: int = -1,
        gt_ls_Bl=None,
    ):
        """
        Step 2：基于当前 cache 推理未来 chunk，并把新增 cache 标成 Pred。

        这里使用官方 `autoregressive_infer()`，但刻意固定了三点：
        - `kv_cache_reset=False`：保留历史 cache（包括 GT `t0` / `gt_obs`）；
        - `skip_text_forward=True`：不要重复写文本 prefix；
        - `extra_ref_text_scale_inds=['gt_obs']`：允许视觉生成阶段继续 attend 真实观测 cache。

        中文导读：
        这是“预测下一段 latent 世界状态变化”的地方。它不会清空历史，而是在
        `t0 + gt_obs` 这份上下文上继续自回归，生成本轮新的 latent。
        但这些新增 cache 全部会被标成 Pred，意味着它们只是“当前一轮的临时想象结果”，
        等 segment 结束后就会被清掉。

        和 `compute_kv_cache_gt()` 的区别：
        - `compute_kv_cache_gt()` 写入真实历史，cache key 是 `gt_obs`，is_pred=False；
        - `infer_chunk()` 生成未来 latent，is_pred=True；
        - `correction_clear_pred()` 只清掉第二类，不清掉第一类。

        """
        assert self._text_cond_tuple is not None, "请先调用 reset() 写入文本条件 cache"
        text_cond_tuple, neg_tuple = self._text_cond_tuple
        if negative_prompt and neg_tuple is None:
            neg_tuple = encode_prompt(self.args.text_encoder_ckpt, self.text_tokenizer, self.text_encoder, negative_prompt, enable_positive_prompt=False, low_vram_mode=False)

        # 本次预测窗口的 schedule。`num_frames` 可以是完整视频长度，也可以是 tail-window 长度。
        sched = self.build_schedule_for_num_frames(num_frames)

        # 确保列表长度和 schedule 一致。
        if not isinstance(cfg_list, list):
            cfg_list = [cfg_list] * len(sched.scale_schedule)
        if not isinstance(tau_list, list):
            tau_list = [tau_list] * len(sched.scale_schedule)

        # 后续写入都标为 pred。也就是说，本轮 autoregressive_infer 产生的 KV cache
        # 都是临时预测上下文，不能长期留在闭环历史里。
        self.infinity.set_cache_write_is_pred(True)

        # 只有 gt_obs cache 已写入时才引用它。
        # 在 step-0 direct-like 推理里，我们有意不写 gt_obs cache，以避免重复条件注入伪影；
        # 这里可以避开对应的 KeyError。
        has_gt_obs_cache = False
        for blk in self.infinity.unregistered_blocks:
            cached_k = getattr(blk.attn, "cached_k", {})
            if self.gt_obs_cache_key in cached_k:
                has_gt_obs_cache = True
                break
        extra_ref_text_scale_inds = [self.gt_obs_cache_key] if has_gt_obs_cache else []

        model_dtype = next(iter(self.infinity.parameters())).dtype
        with torch.amp.autocast("cuda", dtype=model_dtype):
            # 关键参数：
            # - kv_cache_reset=False：保留 reset()/compute_kv_cache_gt() 写入的 GT cache；
            # - skip_text_forward=True：不重复写 t0，因为 reset() 已经写过；
            # - extra_ref_text_scale_inds：如果有 gt_obs，就让生成 token attend 真实观测 cache。
            return self.infinity.autoregressive_infer(
                vae=self.vae,
                scale_schedule=sched.scale_schedule,
                label_B_or_BLT=text_cond_tuple,
                negative_label_B_or_BLT=neg_tuple,
                B=1,
                g_seed=seed,
                cfg_list=cfg_list,
                tau_list=tau_list,
                top_k=int(top_k or 0),
                top_p=float(top_p or 0.0),
                trunk_scale=1000,
                gt_leak=gt_leak,
                gt_ls_Bl=gt_ls_Bl,
                low_vram_mode=low_vram_mode,
                args=self.args,
                get_visual_rope_embeds=self.get_visual_rope_embeds,
                context_info=sched.context_info,
                kv_cache_reset=False,
                skip_text_forward=True,
                cache_text_as_gt=False,
                extra_ref_text_scale_inds=extra_ref_text_scale_inds,
            )

    def correction_clear_pred(self):
        """
        Step 4：一次性清理 Pred KV cache，保留 GT cache。

        中文导读：
        这一步是闭环推理的安全阀：模型可以短程想象下一段世界变化，但想象产生的 cache
        不会长期污染历史。下一轮决策必须重新依赖 `gt_obs` 中的真实观测。

        如果跳过这一步，会发生什么：
        模型会把自己上一轮预测出来的视觉上下文继续当作真实历史使用，误差会在多轮后持续累积，
        最终变成“模型在和自己对话”，而不是和真实环境闭环。
        """
        self.infinity.clear_pred_cache()

    def cache_stats(self) -> Tuple[int, int, int]:
        """返回当前 KV cache 的 `(total, pred, gt)` 条目数量。"""
        return _count_cache_entries(self.infinity)
