#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
InfinityStar 闭环 / streaming 推理辅助脚本。

在开源目录里，这个文件主要用于：

- 提供共享参数构造函数，供 `infer/server.py` 和 `action_aware_grpo/grpo_server.py` 复用。
- 做离线调试：把一段真实帧序列送进 streaming 流程重放。

准备一个真实帧目录（文件名按时间顺序命名）和对应 prompt。
脚本会自动排序帧，并打印推理结果。

示例：
```bash
python3 tools/closed_loop_streaming_infer_480p_81f.py \
  --ckpt ./checkpoints/model.pth \
  --route_dir ./data/reference_route \
  --prompt_key instruction \
  --out_dir ./output_streaming_closed_loop
```

小白阅读顺序建议：
1. 先看 `_make_args()`，理解一个 CLI 命令会被补齐成哪些 runtime 参数；
2. 再看 `main()` 里“加载模型 -> 初始化 session -> 构造 schedule”的准备段；
3. 最后看 `_write_gt_obs()`、`_infer_full()` 和离线/watch 两条循环，理解缓存如何被 GT 覆盖。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import sys
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
from PIL import Image

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.infinity_streaming_session import InfinityStreamingSession
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, save_video, transform


def _is_safetensors_shard_dir(path: str) -> bool:
    """判断目录是否是 HuggingFace/safetensors 分片 checkpoint。"""
    if not path or not osp.isdir(path):
        return False
    return bool(glob.glob(osp.join(path, "*.safetensors"))) or bool(glob.glob(osp.join(path, "*.safetensors.index.json")))


def _resolve_checkpoint_layout(ckpt: str) -> dict[str, str]:
    """
    识别 checkpoint 布局并转成 Infinity loader 需要的字段。

    支持三类输入：HF repo 风格的 `gpt/` + `vae/` 分片目录、单独 GPT safetensors
    分片目录、以及普通 `global_step_*.pth` torch checkpoint。
    """
    ckpt = osp.abspath(ckpt)

    gpt_dir = osp.join(ckpt, "gpt")
    vae_dir = osp.join(ckpt, "vae")
    if osp.isdir(ckpt) and _is_safetensors_shard_dir(gpt_dir) and _is_safetensors_shard_dir(vae_dir):
        return {
            "checkpoint_layout": "hf_repo",
            "checkpoint_type": "torch_shard",
            "model_path": gpt_dir,
            "vae_model_path": vae_dir,
        }

    if _is_safetensors_shard_dir(ckpt):
        return {
            "checkpoint_layout": "torch_shard",
            "checkpoint_type": "torch_shard",
            "model_path": ckpt,
            "vae_model_path": "",
        }

    return {
        "checkpoint_layout": "torch",
        "checkpoint_type": "torch",
        "model_path": ckpt,
        "vae_model_path": "",
    }


def _sorted_images(image_dir: str) -> List[str]:
    """按文件名顺序收集观测帧，保持离线 replay 的时间顺序稳定。"""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(osp.join(image_dir, e)))
    files = sorted(files)
    return files


def _read_prompt(prompt: Optional[str], prompt_json: Optional[str], prompt_key: str) -> str:
    """优先使用命令行 prompt，否则从 meta/prompt json 中读取导航指令。"""
    if prompt and prompt.strip():
        return prompt.strip()
    if not prompt_json:
        raise ValueError("必须提供 --prompt 或 --prompt_json")
    with open(prompt_json, "r", encoding="utf-8") as f:
        pj = json.load(f)
    p = pj.get(prompt_key) or pj.get("instruction_unified") or pj.get("instruction") or pj.get("prompt")
    if not isinstance(p, str) or not p.strip():
        raise ValueError(f"在 {prompt_json} 中找不到有效 prompt（key={prompt_key}）")
    return p.strip()


def _make_args(
    *,
    ckpt: str,
    pn: str,
    fps: int,
    num_frames: int,
    seed: int,
    dynamic_scale_schedule: str,
    mask_type: str,
    cfg: float,
    tau_image: float,
    tau_video: float,
) -> SimpleNamespace:
    """
    构造 InfinityStar 推理参数对象。

    中文导读：
    这里不是普通 argparse，而是把训练/推理脚本散落依赖的字段集中补齐。`infer/server.py`
    和 GRPO server 都会复用这个函数，因此新增推理参数时要确认这些服务端路径是否也需要同步。
    """
    ckpt_dir = osp.join(REPO_ROOT, "checkpoint")
    ckpt_layout = _resolve_checkpoint_layout(ckpt)
    # 优先使用仓库里的 Args（默认字段更完整），避免缺字段崩溃。
    # 如果当前 Python 环境没有 Tap/Args，就退回 SimpleNamespace。
    try:
        from infinity.utils.arg_util import Args as _Args  # type: ignore
        a = _Args()
    except Exception:
        a = SimpleNamespace()
    a.pn = pn
    a.fps = int(fps)
    # 同时保留两个名字；部分工具会读取 video_fps。
    a.video_fps = int(fps)
    a.video_frames = int(num_frames)  # 按模型最大长度配置（我们的用例里是 81）。
    a.temporal_compress_rate = 4
    a.videovae = 10
    a.vae_type = 64
    a.vae_path = osp.join(ckpt_dir, "infinitystar_videovae.pth")
    a.text_encoder_ckpt = osp.join(ckpt_dir, "text_encoder", "flan-t5-xl-official")
    a.text_channels = 2048
    # 保留仓库其他脚本会用到的别名。
    a.Ct5 = a.text_channels
    a.tlen = 512
    a.simple_text_proj = 1
    a.model_type = "infinity_qwen8b"
    a.model_path = ckpt_layout["model_path"]
    a.checkpoint_type = ckpt_layout["checkpoint_type"]
    a.checkpoint_layout = ckpt_layout["checkpoint_layout"]
    a.vae_model_path = ckpt_layout["vae_model_path"]

    # 匹配 finetune 使用的 schedule family。
    a.dynamic_scale_schedule = dynamic_scale_schedule
    a.mask_type = mask_type

    # 推理控制参数：对齐 tools/infer_video_480p.py 和我们的 finetune 推理脚本。
    a.use_flex_attn = True
    a.bf16 = 1
    a.use_apg = 1
    a.use_cfg = 0
    a.cfg = float(cfg)
    a.tau_image = float(tau_image)
    a.tau_video = float(tau_video)
    a.apg_norm_threshold = 0.05
    a.append_duration2caption = 1
    a.use_two_stage_lfq = 1
    a.detail_scale_min_tokens = 350
    a.semantic_scales = 11
    # VideoVAE 构造函数（global_args.*）需要这些字段，Infinity heads 也会读取。
    # 这里对齐仓库默认值和 finetune 脚本。
    a.semantic_scale_dim = 16
    a.detail_scale_dim = 64
    a.use_learnable_dim_proj = 0
    a.use_feat_proj = 2
    # Infinity __init__、attention mask 和 RoPE helper 会额外访问这些参数。
    a.context_frames = getattr(a, "context_frames", 10000)
    a.steps_per_frame = getattr(a, "steps_per_frame", 3)
    a.inject_sync = getattr(a, "inject_sync", 0)
    a.rope_type = getattr(a, "rope_type", "4d")
    a.image_batch_size = getattr(a, "image_batch_size", 0)
    a.video_batch_size = getattr(a, "video_batch_size", 1)
    a.train_with_var_seq_len = getattr(a, "train_with_var_seq_len", 0)
    a.train_max_token_len = getattr(a, "train_max_token_len", -1)
    a.noise_input = getattr(a, "noise_input", 0)
    a.max_repeat_times = 10000
    a.apply_spatial_patchify = 0
    a.num_of_label_value = 2
    a.rope2d_each_sa_layer = 1
    a.rope2d_normalized_by_hw = 2
    a.pad_to_multiplier = 128
    a.seed = int(seed)

    # repetition 配置（0.40M 使用 14 个 scale）。
    a.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    a.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
    return a


def _load_obs_video_bcthw(frame_paths: List[str], tgt_h: int, tgt_w: int) -> torch.Tensor:
    """
    把真实观测帧目录读成 world model 需要的 `[1,3,T,H,W]` 张量。

    输出仍在 CPU，范围是 `[-1,1]`。这样调用方可以先决定何时搬到 GPU，
    避免在 watch 模式下每次轮询都过早占用显存。
    """
    frames = []
    for p in frame_paths:
        pil = Image.open(p).convert("RGB")
        frames.append(transform(pil, tgt_h, tgt_w))  # [3,H,W]，范围 [-1,1]。
    video_T3HW = torch.stack(frames, dim=0)  # [T,3,H,W]
    return video_T3HW.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]

def _take_with_pad(paths: List[str], n: int, pad_short_real: bool) -> List[str]:
    """截取前 n 帧；若真实帧不足且允许 padding，则重复最后一帧补齐。"""
    if len(paths) >= n:
        return paths[:n]
    if not paths:
        raise ValueError("没有找到任何真实观测帧")
    if not pad_short_real:
        raise ValueError(f"真实帧数量不足：需要 {n} 帧，但只有 {len(paths)} 帧（可启用 --pad_short_real 进行补齐）")
    # 重复最后一帧路径，补到 n 帧。
    return paths + [paths[-1]] * (n - len(paths))


def _obs_points(total_gt_frames: int, pred_num_frames: int, step: int) -> List[int]:
    """
        生成闭环观测边界，例如 1,17,33,...，并裁剪到真实帧和预测帧上限。

        points 表示每一轮已经看到的真实帧数量：
          公式/形状说明：points = [1, 1+step, 1+2*step, ..., end]
        推理先用 points[i] 帧预测未来，再等到 points[i+1] 帧到齐后写回 GT cache。

    """
    end = min(int(total_gt_frames), int(pred_num_frames))
    if end <= 0:
        return []
    pts = [1]
    k = 1
    while True:
        v = 1 + k * int(step)
        if v >= end:
            break
        pts.append(v)
        k += 1
    if pts[-1] != end:
        pts.append(end)
    return pts


def main() -> None:
    """
    命令行闭环 streaming 推理入口。

    离线模式会按 segment 保存预测视频；watch 模式会等待新观测帧到齐后再写 GT cache，
    用来模拟“预测动作 -> 执行 -> 获得新帧 -> 继续预测”的在线流程。

    可以把主流程记成 5 步：
    1. 解析 CLI，并把 checkpoint/route/prompt/帧目录整理成统一输入；
    2. `_make_args()` 补齐 Infinity runtime 依赖的参数字段；
    3. `load_tokenizer/load_visual_tokenizer/load_transformer` 加载文本编码器、VAE、世界模型；
    4. `InfinityStreamingSession.reset()` 建立文本 cache，再用 `_write_gt_obs()` 写真实观测 cache；
    5. 反复执行“完整预测 -> 保存 segment -> `correction_clear_pred()` -> 用更长 GT 覆盖 cache”。
    """
    ap = argparse.ArgumentParser(
        description=(
            "离线或 watch 模式下的 InfinityStar 闭环 streaming 推理调试脚本。"
            "它会把真实观测帧分段写入 GT cache，并保存每一段会在下一轮被覆盖的预测视频。"
        )
    )
    ap.add_argument("--ckpt", type=str, required=True, help="世界模型 checkpoint 路径；通常是 `global_step_*.pth` 或 sharded 目录。")
    ap.add_argument("--route_dir", type=str, default="", help="可选 route 目录；若提供，默认从 `route_dir/images` 读帧、从 `route_dir/meta.json` 读 prompt。")
    ap.add_argument("--frames_dir", type=str, default="", help="真实观测帧目录，按文件名排序读取；若提供了 `--route_dir`，这里可以省略。")
    ap.add_argument("--prompt", type=str, default="", help="直接给定导航 prompt；和 `--prompt_json` 二选一即可。")
    ap.add_argument("--prompt_json", type=str, default="", help="含 prompt 的 JSON 路径；常见是 route 的 `meta.json`。")
    ap.add_argument("--prompt_key", type=str, default="instruction_unified", help="当从 `--prompt_json` 读取 prompt 时优先尝试的字段名。")
    ap.add_argument("--negative_prompt", type=str, default="", help="可选 negative prompt，会传给世界模型采样。")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录；每次运行会在其中创建一个时间戳子目录保存视频和 run_args。")

    ap.add_argument("--num_frames", type=int, default=81, help="每轮完整预测的视频总帧数；默认 81，对应本项目常见的 4n+1 时间规则。")
    ap.add_argument("--step", type=int, default=16, help="闭环每轮前进多少帧；默认 16，对应 `points=[1,17,33,...]`。")
    ap.add_argument("--save_full_pred", action="store_true", help="除 segment 之外，额外保存每轮完整的 N 帧预测视频，便于排查 segment 之外的生成质量。")
    ap.add_argument("--pad_short_real", action="store_true", default=True, help="真实帧不足 `num_frames` 时，是否重复最后一帧补齐；默认开启。")
    ap.add_argument("--no_pad_short_real", action="store_false", dest="pad_short_real", help="关闭补齐；若真实帧不足，会提前报错或停止。")
    ap.add_argument("--watch", action="store_true", help="监视模式（watch）：持续等待 `frames_dir` 下出现更多真实帧，每凑够一段就继续下一轮推理。")
    ap.add_argument("--poll_interval", type=float, default=0.5, help="监视模式（watch）的轮询间隔，单位秒。")
    ap.add_argument("--fps", type=int, default=16, help="输出视频和时长标签使用的帧率。")
    ap.add_argument("--pn", type=str, default="0.40M", help="动态分辨率预设名，会影响 scale schedule 和 patch 预算。")
    ap.add_argument("--h_div_w_template", type=float, default=0.562, help="目标高宽比模板 H/W；会影响 schedule 推导出的目标分辨率。")

    ap.add_argument("--seed", type=int, default=0, help="随机种子基值；每个 step 会在此基础上再加 step_i。")
    ap.add_argument("--cfg", type=float, default=34.0, help="CFG（Classifier-Free Guidance）强度。")
    ap.add_argument("--tau_image", type=float, default=1.0, help="图像/首帧相关尺度使用的采样温度。")
    ap.add_argument("--tau_video", type=float, default=0.4, help="视频续帧相关尺度使用的采样温度。")
    ap.add_argument("--top_k", type=int, default=0, help="top-k 采样；0 通常表示不启用额外 top-k 截断。")
    ap.add_argument("--top_p", type=float, default=0.0, help="top-p 采样；0 通常表示不启用额外 nucleus 截断。")

    ap.add_argument("--dynamic_scale_schedule", type=str, default="infinity_elegant_clip20frames_v2_allpt", help="动态尺度计划名，决定每个自回归尺度的 patch 网格和顺序。")
    ap.add_argument("--mask_type", type=str, default="infinity_elegant_clip20frames_v2_allpt", help="注意力 mask 类型，通常与 dynamic_scale_schedule 成对出现。")
    args_cli = ap.parse_args()

    # 支持 torchrun 注入的 local rank。
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    ckpt = osp.abspath(args_cli.ckpt)
    route_dir = osp.abspath(args_cli.route_dir) if args_cli.route_dir else ""
    frames_dir = osp.abspath(args_cli.frames_dir) if args_cli.frames_dir else ""
    if route_dir:
        if not frames_dir:
            frames_dir = osp.join(route_dir, "images")
        if not args_cli.prompt_json:
            args_cli.prompt_json = osp.join(route_dir, "meta.json")
    out_dir = osp.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    prompt = _read_prompt(args_cli.prompt, args_cli.prompt_json or None, args_cli.prompt_key)
    prompt = prompt.strip()
    negative_prompt = (args_cli.negative_prompt or "").strip()

    if not frames_dir:
        raise ValueError("必须提供 --frames_dir 或 --route_dir")

    frame_paths = _sorted_images(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"在 {frames_dir} 下没有找到任何图像帧（至少先写入 1 帧）")
    if (not args_cli.watch) and (len(frame_paths) < int(args_cli.num_frames)) and bool(args_cli.pad_short_real):
        print(
            f"[警告] 真实帧数 {len(frame_paths)} 小于 num_frames={args_cli.num_frames}；"
            f"写 GT cache 时会通过重复最后一帧补到 {args_cli.num_frames} 帧"
        )

    # 第 1 段：CLI args -> runtime args。
    # 这里把命令行参数补成 Infinity 各组件都能读取的一大组字段。
    a = _make_args(
        ckpt=ckpt,
        pn=args_cli.pn,
        fps=args_cli.fps,
        num_frames=args_cli.num_frames,
        seed=args_cli.seed,
        dynamic_scale_schedule=args_cli.dynamic_scale_schedule,
        mask_type=args_cli.mask_type,
        cfg=args_cli.cfg,
        tau_image=args_cli.tau_image,
        tau_video=args_cli.tau_video,
    )

    # 第 2 段：runtime args -> tokenizer / VAE / world model。
    # 文本编码器提供语言条件，VAE 负责视频 latent，Infinity 负责在已有历史上继续生成未来。
    text_tokenizer, text_encoder = load_tokenizer(t5_path=a.text_encoder_ckpt)
    vae = load_visual_tokenizer(a).float().to("cuda")
    infinity = load_transformer(vae, a)
    infinity.eval().requires_grad_(False)

    # 第 3 段：模型组件 -> streaming session。
    # session 会统一管理文本 cache、GT 观测 cache 和预测 cache。
    session = InfinityStreamingSession(
        args=a,
        infinity_model=infinity,
        vae=vae,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        h_div_w_template=float(args_cli.h_div_w_template),
    )

    # 第 4 段：session -> 输出 schedule。
    # schedule 不只决定“预测多少帧”，还决定目标高宽、每个尺度的 token 网格，以及
    # tau_image/tau_video 应该如何按 tower_split_index 拼成整段 tau 列表。
    sched_out = session.build_schedule_for_num_frames(num_frames=int(args_cli.num_frames))
    tgt_h, tgt_w = sched_out.tgt_h, sched_out.tgt_w
    tau = [float(a.tau_image)] * int(sched_out.tower_split_index) + [float(a.tau_video)] * (len(sched_out.scale_schedule) - int(sched_out.tower_split_index))

    # 给 prompt 加时长标签，以匹配训练时的 prompt 格式。
    # 公式：duration_seconds = (num_frames - 1) // fps。
    dur_s = (int(args_cli.num_frames) - 1) // int(args_cli.fps)
    prompt_infer = f"<<<t={dur_s}s>>>{prompt}" if int(getattr(a, "append_duration2caption", 0)) else prompt

    # 初始化 session（把 text cache 作为 GT）。注意：cfg != 1 时 bs=2，因此 GT cache 也会按 bs=2 写入。
    session.reset(prompt_infer, negative_prompt=negative_prompt, cfg_scale=float(a.cfg))

    # 离线模式的循环边界：1, 17, 33, ... , min(81, len(frames))。
    total_for_points = int(args_cli.num_frames) if (not args_cli.watch and bool(args_cli.pad_short_real)) else len(frame_paths)
    points = _obs_points(total_gt_frames=total_for_points, pred_num_frames=int(args_cli.num_frames), step=int(args_cli.step))
    if not args_cli.watch:
        print(f"[信息] 离线模式：真实帧数={len(frame_paths)} 预测帧数={args_cli.num_frames} points={points}")
    print(f"[信息] tgt_h={tgt_h} tgt_w={tgt_w} cfg={a.cfg} tau_image={a.tau_image} tau_video={a.tau_video} top_k={args_cli.top_k} top_p={args_cli.top_p}")

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = osp.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(osp.join(run_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": ckpt,
                "route_dir": route_dir,
                "frames_dir": frames_dir,
                "real_frame_count": len(frame_paths),
                "num_frames": int(args_cli.num_frames),
                "step": int(args_cli.step),
                "pad_short_real": bool(args_cli.pad_short_real),
                "fps": int(args_cli.fps),
                "pn": str(args_cli.pn),
                "h_div_w_template": float(args_cli.h_div_w_template),
                "cfg": float(a.cfg),
                "tau_image": float(a.tau_image),
                "tau_video": float(a.tau_video),
                "top_k": int(args_cli.top_k),
                "top_p": float(args_cli.top_p),
                "dynamic_scale_schedule": str(args_cli.dynamic_scale_schedule),
                "mask_type": str(args_cli.mask_type),
                "prompt": prompt,
                "prompt_infer": prompt_infer,
                "negative_prompt": negative_prompt,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    def _write_gt_obs(cur_frame_paths: List[str], obs_len: int) -> None:
        """
        用累计真实帧 `[1..obs_len]` 覆盖 GT obs cache。

        注意这里写入的是“从第 1 帧到当前边界的完整真实历史”，不是只追加最后新到的那几帧。
        这样做虽然更重，但语义最稳：下一轮预测一定基于真实观测前缀，而不是夹杂旧预测残留。
        """
        padded = _take_with_pad(cur_frame_paths, int(obs_len), bool(args_cli.pad_short_real))
        obs_cpu = _load_obs_video_bcthw(padded, tgt_h, tgt_w)  # [1,3,T,H,W] CPU
        obs = obs_cpu.to("cuda", non_blocking=True)
        session.compute_kv_cache_gt(obs)

    def _infer_full(step_i: int) -> torch.Tensor:
        """
        推理完整 N 帧，返回第一个样本的 uint8 BGR tensor `[T,H,W,3]`。

        尽管闭环最终只会保存当前 segment 对应的那一小段未来帧，这里仍先生成完整 `num_frames`
        视频，是为了与世界模型的标准 streaming 协议保持一致，也方便在 `--save_full_pred` 时
        检查“超出当前 segment 的更远未来”长什么样。
        """
        seed = int(args_cli.seed) + int(step_i)
        t0 = time.time()
        _, img = session.infer_chunk(
            num_frames=int(args_cli.num_frames),
            cfg_list=float(a.cfg),
            tau_list=tau,
            top_k=int(args_cli.top_k),
            top_p=float(args_cli.top_p),
            seed=seed,
            negative_prompt=negative_prompt,
            low_vram_mode=True,
        )
        dt = time.time() - t0
        vid = img[0] if isinstance(img, torch.Tensor) and img.dim() == 5 else img
        print(f"[推理] step={step_i} seed={seed} 耗时={dt:.2f}s cache_stats={session.cache_stats()}")
        return vid

    def _save_segment(vid: torch.Tensor, *, step_i: int, obs_len: int, next_obs_len: int) -> None:
        """
        保存下一次写入 GT 时会被覆盖的预测片段：
        按从 1 开始编号的约定，对应帧 (obs_len+1 .. next_obs_len)。
        """
        # vid: [T,H,W,3]，其中 T == num_frames。
        start0 = int(obs_len)  # 0-based：obs_len=1 时，从 frame index 1（第 2 帧）开始。
        end0 = int(next_obs_len)
        seg = vid[start0:end0]
        seg_start_1idx = int(obs_len) + 1
        seg_end_1idx = int(next_obs_len)
        save_path = osp.join(run_dir, f"seg_{step_i:02d}_pred_{seg_start_1idx:03d}_{seg_end_1idx:03d}.mp4")
        save_video(seg, fps=int(args_cli.fps), save_filepath=save_path, force_all_keyframes=True)
        print(f"[保存] segment step={step_i} 预测帧={seg_start_1idx}-{seg_end_1idx} 路径={save_path}")

    def _save_full(vid: torch.Tensor, *, step_i: int, obs_len: int) -> None:
        """保存当前 step 的完整预测视频，主要用于排查某段 segment 之外的生成质量。"""
        save_path = osp.join(run_dir, f"full_{step_i:02d}_obs{obs_len:03d}_pred{int(args_cli.num_frames):03d}.mp4")
        save_video(vid, fps=int(args_cli.fps), save_filepath=save_path, force_all_keyframes=True)
        print(f"[保存] full step={step_i} 路径={save_path}")

    # 总是先写入第一帧的 GT obs（obs_len=1）。
    session.correction_clear_pred()
    _write_gt_obs(frame_paths, obs_len=1)

    if not args_cli.watch:
        # 离线模式时序：
        # 1. 用当前 GT 历史预测完整 N 帧；
        # 2. 只保存“下一轮会被 GT 覆盖掉”的那一小段 segment；
        # 3. 清掉 pred cache；
        # 4. 用更长的真实历史重写 GT cache；
        # 5. 进入下一段。
        if len(points) < 2:
            raise ValueError(f"points={points} 太短，无法执行分段保存（至少需要 2 个边界点）")
        for i in range(len(points) - 1):
            obs_len = int(points[i])
            next_obs_len = int(points[i + 1])
            vid = _infer_full(step_i=i)
            if args_cli.save_full_pred:
                _save_full(vid, step_i=i, obs_len=obs_len)
            _save_segment(vid, step_i=i, obs_len=obs_len, next_obs_len=next_obs_len)

            # 现在用到 next_obs_len 为止的 GT 覆盖 cache。
            session.correction_clear_pred()
            _write_gt_obs(frame_paths, obs_len=next_obs_len)
        return

    # watch 模式和离线模式的核心差别是：
    # 离线模式一开始就拿到了全部真实帧；watch 模式则需要边等边推理。
    points_watch = _obs_points(total_gt_frames=10**9, pred_num_frames=int(args_cli.num_frames), step=int(args_cli.step))
    # 至少要有第一帧；上面已经完成 obs_len=1 的 GT 写入。
    for i in range(len(points_watch) - 1):
        obs_len = int(points_watch[i])
        next_obs_len = int(points_watch[i + 1])
        vid = _infer_full(step_i=i)
        if args_cli.save_full_pred:
            _save_full(vid, step_i=i, obs_len=obs_len)
        _save_segment(vid, step_i=i, obs_len=obs_len, next_obs_len=next_obs_len)

        # 等到 next_obs_len 所需帧数到齐，再覆盖 GT cache。
        while True:
            cur = _sorted_images(frames_dir)
            if len(cur) >= next_obs_len:
                frame_paths = cur
                break
            time.sleep(float(args_cli.poll_interval))
        session.correction_clear_pred()
        _write_gt_obs(frame_paths, obs_len=next_obs_len)


if __name__ == "__main__":
    main()
