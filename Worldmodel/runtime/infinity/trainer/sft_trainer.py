# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
"""
Runtime 版 InfinityTrainer：同时服务 SFT 和 Action-aware GRPO。

先把概念分清：

- SFT（Supervised Fine-Tuning / 监督微调）：
  给定文本条件和真实视频 VAE token，用 teacher-forcing 训练世界模型预测真实视频 token。
  它的监督信号是“真实视频 token”，不是动作标签，也不是 reward。

- GRPO / RL：
  先由 StageA rollout 采样出一批 token/latent 轨迹，reward_uavflow.py 给轨迹打分；
  StageB 再回放这些采样 token，计算当前模型的 new_logprob，并和 rollout 时的 old_logprob
  做 PPO/GRPO ratio，使用 grpo_adv_final 作为权重更新世界模型。

这个 runtime 文件和 `Worldmodel/infinity/trainer/sft_trainer.py` 的区别：

- 基础版主要是 Stage-1 SFT 世界模型训练；
- runtime 版额外接收 `grpo_*` 字段、trace 文件、old/ref logprob，能在同一个 `train_step()`
  里切换 SFT loss 或 GRPO/PPO loss；
- 即使走 GRPO，代码仍需要 SFT 的 video_encode / teacher-forcing forward 能力：
  它们负责把 video latent/token 打包、重算 logprob、提供 CE 诊断和可选 aux SFT 稳定项。

WorldVLN 是否“需要 SFT”：

- 如果只是加载已经训练好的 checkpoint 做推理，不需要再运行 SFT 训练；
- 如果要训练/微调世界模型，SFT 是基础阶段，负责让 InfinityStar 先学会生成合理未来 latent；
- 如果要做 GRPO，SFT 仍然是底座：GRPO 不是从零学世界模型，而是在 SFT 世界模型基础上用 reward
  调整采样分布，让高奖励 latent/token 更容易被采到。

换句话说：

- SFT 解决“世界模型会不会生成像真实视频的 latent”；
- Stage-2 / TSformer 动作头解决“这些 latent 对应什么动作”；
- GRPO 解决“在能生成的基础上，哪些生成结果更符合导航/动作 reward，应该被提高概率”。
"""

from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import nullcontext
import math
import os
import os.path as osp
import json
import time
import importlib
import sys

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch.distributed as tdist

import infinity.utils.dist as dist
from infinity.models import Infinity
from infinity.models.ema import update_ema
from infinity.models.self_correction import SelfCorrection
from infinity.utils import arg_util, misc, wandb_utils
from infinity.utils.amp_opt import AmpOptimizer
from infinity.schedules import get_encode_decode_func
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

import queue
import threading

def save_token():
    """
    后台线程：异步把新提取的 VAE token 保存到磁盘缓存。

    SFT/GRPO 的 batch 都可能先拿到 RGB，再编码成 VAE raw_features。这个编码开销大，
    所以训练时会把 raw_features 缓存到磁盘。主训练线程只把待保存项放入队列，
    后台线程负责实际写盘，避免 GPU 等 I/O。
    """
    while True:
        try:
            raw_features, feature_cache_files4images = save_token_queue.get()
            for i in range(len(feature_cache_files4images)):
                if not osp.exists(feature_cache_files4images[i]):
                    os.makedirs(osp.dirname(feature_cache_files4images[i]), exist_ok=True)
                    torch.save(raw_features[i], feature_cache_files4images[i])
                    print(f'保存 VAE token 到: {feature_cache_files4images[i]}')
                else:
                    print(f'{feature_cache_files4images[i]} 已存在，跳过保存')
        except Exception as e:
            print(f"保存 VAE token 失败: {e}")
        finally:
            save_token_queue.task_done()

save_token_queue = queue.Queue()
saver = threading.Thread(target=save_token, daemon=True)
saver.start()


def _obs_points(pred_num_frames: int, step: int):
    """生成观测锚点 `points = [1, 1+step, 1+2*step, ..., end]`。"""
    end = int(pred_num_frames)
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

class InfinityTrainer(object):
    """
    Infinity 的统一训练器，兼容 SFT、GRPO 与 teacher-forcing 调试导出。

    可以把它看成三层：

    1. 公共世界模型前向层：
       RGB/cache latent -> VAE raw_features -> video_encode() -> Infinity.forward()。
       SFT 和 GRPO 都需要这层，因为二者都要把视频 token 打包成模型可消费的序列。

    2. SFT loss 层：
       直接使用 Infinity.forward() 返回的 token CE loss，按 scale 加权后反传。
       这是普通监督微调路径。

    3. GRPO/RL loss 层：
       使用 StageA 写入的 `grpo_old_logprob`、`grpo_adv_final`、`grpo_trace_files`，
       重新计算 `new_logprob`，构造 PPO/GRPO 目标。
       这条路径更新的仍然是 Infinity 世界模型，不是动作头。
    """
    def __init__(
        self,
        device,
        raw_scale_schedule: Tuple[int, ...],
        vae_local,
        gpt_wo_ddp: Infinity, gpt: DDP,
        gpt_opt: AmpOptimizer,
        label_smooth: float,
        zero=0,
        vae_type=True,
        reweight_loss_by_scale=0,
        gpt_wo_ddp_ema=None,
        gpt_ema=None,
        use_fsdp_model_ema=False,
        other_args=None,
    ):
        """
        保存训练依赖对象，并初始化损失、EMA、GRPO 统计与调试工具。

        关键成员：
        - `vae_local`：把 RGB 转成世界模型训练/回放用的 raw_features；
        - `video_encode`：把 raw_features 打包成 `x_BLC/gt_BLC/RoPE/scale_info`；
        - `gpt` / `gpt_wo_ddp`：真正被训练的 InfinityStar 世界模型；
        - `get_visual_rope_embeds/get_scale_pack_info`：GRPO trace replay / trace CE 复现 StageA
          采样路径时需要的 schedule 辅助函数；
        - `_stable_metric_*`：GRPO 指标更噪，所以这里维护一些 EMA/step 级统计，便于看趋势。
        """
        super(InfinityTrainer, self).__init__()

        self.zero = zero
        self.vae_type = vae_type

        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local = gpt, vae_local
        self.dynamic_scale_schedule = other_args.dynamic_scale_schedule
        self.steps_per_frame = other_args.steps_per_frame
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(other_args.dynamic_scale_schedule, other_args.video_frames)
        self.gpt_opt: AmpOptimizer = gpt_opt
        self.gpt_wo_ddp: Union[Infinity, torch._dynamo.eval_frame.OptimizedModule] = gpt_wo_ddp  # 中文说明：after torch.compile
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema
        self.self_correction = SelfCorrection(self.vae_local, other_args)
        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.batch_size, self.seq_len = 0, 0
        self.reweight_loss_by_scale = reweight_loss_by_scale
        print(f'self.reweight_loss_by_scale: {self.reweight_loss_by_scale}')
        video_encode, _, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(other_args.dynamic_scale_schedule)
        self.video_encode = video_encode
        # 这两个函数给 GRPO trace-replay / trace-ce 复现 StageA 采样路径时使用。
        # 因为 PPO ratio 必须在“同一批 sampled tokens”上比较 new/old logprob，
        # StageB 不能随便重新打包 schedule，而要尽量复现 StageA 的尺度和 RoPE 语义。
        self.get_visual_rope_embeds = get_visual_rope_embeds
        self.get_scale_pack_info = get_scale_pack_info

        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        del gpt_uncompiled.rng
        gpt_uncompiled.rng = torch.Generator(device=device)
        del gpt_uncompiled

        self.label_smooth = label_smooth

        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
        self.loss_weight = {0:{}, 1:{}}

        # teacher-forcing 调试导出：把 latent clip 解码，并用 TSformer 导出轨迹。
        self._tf_dump_enable = bool(int(getattr(other_args, "tf_dump_tsformer_enable", 0)))
        self._tf_dump_interval = int(getattr(other_args, "tf_dump_interval", 0))
        self._tf_dump_step = int(getattr(other_args, "tf_dump_step_frames", 16))
        self._tf_dump_max_samples = max(1, int(getattr(other_args, "tf_dump_max_samples_per_step", 1)))
        self._tf_dump_save_video = bool(int(getattr(other_args, "tf_dump_save_video", 1)))
        self._tf_dump_out_dir = str(getattr(other_args, "tf_dump_out_dir", "") or "").strip()
        self._tf_ts_repo_root = str(getattr(other_args, "tf_dump_tsformer_repo_root", "") or "").strip()
        self._tf_ts_ckpt = str(getattr(other_args, "tf_dump_tsformer_ckpt", "") or "").strip()
        self._tf_ts_stats = str(getattr(other_args, "tf_dump_tsformer_stats", "") or "").strip()
        self._tf_ref_pose_json = str(getattr(other_args, "tf_dump_ref_pose_json", "") or "").strip()
        self._tf_expand_factor = max(1, int(getattr(other_args, "tf_dump_expand_factor", 4)))
        self._tf_ts_model = None
        self._tf_ts_mean = None
        self._tf_ts_std = None
        self._tf_ts_ready = False
        self._tf_ts_init_err = ""
        if self._tf_dump_enable and self._tf_dump_interval <= 0:
            # 给一个保守默认值，避免每个 iteration 都做重 I/O。
            self._tf_dump_interval = 200

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.generator = np.random.default_rng(0)
        self._stable_metric_beta = min(0.999, max(0.0, float(getattr(other_args, "grpo_metric_ema_beta", 0.9) or 0.9)))
        self._stable_metric_ema: Dict[str, float] = {}
        self._stable_metric_last: Dict[str, float] = {}
        self._optstep_metric_sums: Dict[str, float] = {}
        self._optstep_metric_counts: Dict[str, int] = {}
        self._optstep_metric_last: Dict[str, float] = {}
        self._optstep_metric_ema: Dict[str, float] = {}
        self._optstep_metric_ema_last: Dict[str, float] = {}

    def _update_scalar_ema(self, store: Dict[str, float], key: str, value: Optional[float]) -> Optional[float]:
        """更新某个标量指标的 EMA，忽略空值和非有限值。"""
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        prev = store.get(key, None)
        if prev is None:
            store[key] = value
        else:
            beta = self._stable_metric_beta
            store[key] = beta * prev + (1.0 - beta) * value
        return store[key]

    def _update_stable_metric_trackers(self, metrics: Dict[str, Optional[float]], stepping: bool) -> None:
        """维护逐 iter 与逐 optimizer-step 的稳定指标统计。"""
        touched: List[str] = []
        for key, value in metrics.items():
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            ema_val = self._update_scalar_ema(self._stable_metric_ema, key, value)
            if ema_val is not None:
                self._stable_metric_last[key] = ema_val
            self._optstep_metric_sums[key] = self._optstep_metric_sums.get(key, 0.0) + value
            self._optstep_metric_counts[key] = self._optstep_metric_counts.get(key, 0) + 1
            touched.append(key)

        if not stepping:
            return

        for key in touched:
            cnt = self._optstep_metric_counts.get(key, 0)
            if cnt <= 0:
                continue
            optstep_mean = self._optstep_metric_sums[key] / float(cnt)
            self._optstep_metric_last[key] = optstep_mean
            optstep_ema_val = self._update_scalar_ema(self._optstep_metric_ema, key, optstep_mean)
            if optstep_ema_val is not None:
                self._optstep_metric_ema_last[key] = optstep_ema_val
            self._optstep_metric_sums[key] = 0.0
            self._optstep_metric_counts[key] = 0

    def _collect_stable_metrics(self) -> Dict[str, float]:
        """导出当前稳定指标快照，供日志系统统一记录。"""
        stable_metrics: Dict[str, float] = {}
        for key, value in self._stable_metric_last.items():
            stable_metrics[f"{key}_ema"] = value
        for key, value in self._optstep_metric_last.items():
            stable_metrics[f"{key}_optstep"] = value
        for key, value in self._optstep_metric_ema_last.items():
            stable_metrics[f"{key}_optstep_ema"] = value
        return stable_metrics

    def _tf_to_cm_deg(self, deltas_m_rad: torch.Tensor) -> torch.Tensor:
        """把轨迹增量从米/弧度转换成厘米/角度。"""
        out = deltas_m_rad.clone()
        # 平移分量乘 100 变成厘米；旋转分量乘 `180 / pi` 变成角度。
        out[..., 0:3] = out[..., 0:3] * 100.0
        out[..., 3:6] = out[..., 3:6] * (180.0 / math.pi)
        return out

    def _tf_integrate(self, actions_cm_deg):
        """把逐步增量轨迹积分成绝对位姿序列。"""
        pose = [0.0] * 6
        poses = []
        for a in actions_cm_deg:
            pose = [float(pose[i]) + float(a[i]) for i in range(6)]
            poses.append(pose)
        return {"start_pose": [0.0] * 6, "poses": poses, "final_pose": (poses[-1] if poses else [0.0] * 6)}

    def _tf_expand_actions4_to16(self, actions4):
        """把 4 个 clip-level 动作均分并展开成 16 个 frame-level 动作。"""
        # 关键公式：每个 clip 动作先除以 `expand_factor`，再重复 `expand_factor` 次。
        fac = int(self._tf_expand_factor)
        out = []
        for a in actions4:
            per_frame = [float(v) / float(fac) for v in a]
            for _ in range(fac):
                out.append(list(per_frame))
        return out

    def _tf_clip_positions_from_start(self, start_pose6, actions4):
        """从给定起点位姿出发，把一个 clip 的动作序列展开成逐帧位姿。"""
        frame_deltas = self._tf_expand_actions4_to16(actions4)
        cur = [float(v) for v in start_pose6]
        poses16 = []
        for d in frame_deltas:
            cur = [cur[i] + float(d[i]) for i in range(6)]
            poses16.append(list(cur))
        return poses16

    def _tf_clip_mse_vs_ref(self, clip_actions4, points, ref_poses):
        """把预测 clip 轨迹与参考位姿对齐，计算 MSE 诊断指标。"""
        # `points` 使用 1-based 帧锚点，例如 `[1, 17, 33, 49]`。
        per_clip = []
        for seg, a4 in enumerate(clip_actions4):
            if seg >= len(points) - 1:
                break
            start_fid = int(points[seg])  # 中文说明：1-based
            if start_fid - 1 >= len(ref_poses):
                break
            gt_start = ref_poses[start_fid - 1]
            pred16 = self._tf_clip_positions_from_start(gt_start, a4)
            gt16 = ref_poses[start_fid : start_fid + 16]
            if len(gt16) < len(pred16):
                pred16 = pred16[: len(gt16)]
            if len(pred16) == 0:
                continue
            p = np.asarray(pred16, dtype=np.float64)
            g = np.asarray(gt16, dtype=np.float64)
            mse_all = float(np.mean((p - g) ** 2))
            mse_xyz = float(np.mean((p[:, :3] - g[:, :3]) ** 2))
            mse_yaw = float(np.mean((p[:, 4] - g[:, 4]) ** 2))
            per_clip.append(
                {
                    "segment_index": int(seg),
                    "start_frame_1based": int(start_fid),
                    "pred_frames": int(len(pred16)),
                    "mse_all6": mse_all,
                    "mse_xyz": mse_xyz,
                    "mse_yaw": mse_yaw,
                }
            )
        if not per_clip:
            return {"per_clip": [], "global_mse_all6": None, "global_mse_xyz": None, "global_mse_yaw": None}
        return {
            "per_clip": per_clip,
            "global_mse_all6": float(np.mean([x["mse_all6"] for x in per_clip])),
            "global_mse_xyz": float(np.mean([x["mse_xyz"] for x in per_clip])),
            "global_mse_yaw": float(np.mean([x["mse_yaw"] for x in per_clip])),
        }

    def _tf_maybe_init_tsformer(self, device: str):
        """按需加载 TSformer 轨迹模型及其归一化统计。"""
        if self._tf_ts_ready or self._tf_ts_init_err:
            return
        if not self._tf_ts_ckpt:
            self._tf_ts_init_err = "tf_dump_tsformer_ckpt 为空"
            return
        repo_root = self._tf_ts_repo_root.strip()
        if not repo_root:
            self._tf_ts_init_err = "tf_dump_tsformer_repo_root 为空"
            return
        try:
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            m = importlib.import_module("pretrain_latent_p2p")
            build_p2p_model = getattr(m, "build_p2p_model")
            args = type("A", (), {})()
            args.window_size = 2
            args.hidden_dim = 96
            args.num_layers = 2
            args.device = device
            args.checkpoint = self._tf_ts_ckpt
            args.stats_path = self._tf_ts_stats
            model = build_p2p_model(args)
            model.to(device).eval()

            ckpt = torch.load(self._tf_ts_ckpt, map_location="cpu")
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("module."):
                    new_sd[k[7:]] = v
                else:
                    new_sd[k] = v
            model.load_state_dict(new_sd, strict=False)

            mean_t = std_t = None
            if self._tf_ts_stats and osp.exists(self._tf_ts_stats):
                with open(self._tf_ts_stats, "r", encoding="utf-8") as f:
                    st = json.load(f)
                mean_t = torch.tensor(st["mean"], dtype=torch.float32, device=device)
                std_t = torch.tensor(st["std"], dtype=torch.float32, device=device)
            self._tf_ts_model = model
            self._tf_ts_mean = mean_t
            self._tf_ts_std = std_t
            self._tf_ts_ready = True
            print(f"[TF-TS] TSformer 初始化完成: ckpt={self._tf_ts_ckpt}")
        except Exception as e:
            self._tf_ts_init_err = str(e)
            print(f"[TF-TS] 初始化失败: {e}")

    @torch.no_grad()
    def _tf_predict_actions_4(self, lat5_BCTHW: torch.Tensor, device: str) -> torch.Tensor:
        """用 TSformer 从 5 帧 latent 窗口预测 4 个动作增量。"""
        assert self._tf_ts_model is not None
        x = lat5_BCTHW
        if int(x.shape[1]) == 64:
            t = x.permute(0, 2, 1, 3, 4).contiguous()
            t = torch.nn.functional.pixel_shuffle(t, 2)
            x = t.permute(0, 2, 1, 3, 4).contiguous()
        if int(x.shape[1]) != 16:
            raise ValueError(f"TSformer 期望输入通道 C=16/64，实际收到 shape={tuple(x.shape)}")
        lat_TCHW = x[0].permute(1, 0, 2, 3).contiguous()
        windows = torch.stack([lat_TCHW[:-1], lat_TCHW[1:]], dim=1).to(device=device, dtype=torch.float32)
        out = self._tf_ts_model(windows)
        if self._tf_ts_mean is not None and self._tf_ts_std is not None:
            out = out * self._tf_ts_std + self._tf_ts_mean
        if int(out.shape[0]) >= 4:
            out = out[-4:]
        else:
            out = torch.cat([out] + [out[-1:]] * (4 - int(out.shape[0])), dim=0)
        return self._tf_to_cm_deg(out).detach().cpu()

    @torch.no_grad()
    def _tf_decode_and_save_clip(self, z5_BCTHW: torch.Tensor, out_mp4: str, fps: int, drop_first: bool):
        """把 latent clip 解码成视频并保存，便于肉眼检查。"""
        if not self._tf_dump_save_video:
            return
        try:
            imageio = importlib.import_module("imageio")
        except Exception:
            return
        try:
            gpt_raw = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, "_orig_mod") else self.gpt_wo_ddp
            model_dtype = next(iter(gpt_raw.parameters())).dtype
            vae_device = next(iter(self.vae_local.parameters())).device
            z = z5_BCTHW.to(device=vae_device, dtype=model_dtype)
            bgr = gpt_raw.summed_codes2images(self.vae_local, z)
            if isinstance(bgr, torch.Tensor):
                bgr = bgr.detach().cpu().numpy()
            clip = bgr[0]
            if drop_first and int(clip.shape[0]) > 1:
                clip = clip[1:]
            if int(clip.shape[0]) <= 0:
                return
            rgb = clip[:, :, :, ::-1]
            os.makedirs(osp.dirname(out_mp4), exist_ok=True)
            imageio.mimsave(out_mp4, rgb, fps=int(fps))
        except Exception as e:
            print(f"[TF-TS] 解码或保存 clip 失败: {e}")

    @torch.no_grad()
    def _tf_dump_step_trajectory(self, raw_features_list, g_it: int, args):
        """定期导出 teacher-forcing 轨迹调试结果。"""
        if not self._tf_dump_enable:
            return
        if self._tf_dump_interval <= 0 or ((int(g_it) + 1) % int(self._tf_dump_interval) != 0):
            return
        self._tf_maybe_init_tsformer(args.device)
        if not self._tf_ts_ready:
            if self._tf_ts_init_err:
                print(f"[TF-TS] 跳过导出: {self._tf_ts_init_err}")
            return

        out_root = self._tf_dump_out_dir or osp.join(args.local_out_path, "tf_tsformer_debug")
        points = _obs_points(int(args.video_frames), int(self._tf_dump_step))
        if len(points) < 2:
            return
        ts_tag = time.strftime("%Y%m%d_%H%M%S")
        run_dir = osp.join(out_root, f"it_{int(g_it)+1:08d}_{ts_tag}")
        os.makedirs(run_dir, exist_ok=True)

        for sample_idx, raw in enumerate(raw_features_list[: self._tf_dump_max_samples]):
            z = raw.detach()
            if z.ndim == 4:
                z = z.unsqueeze(0)
            z = z.to("cpu").contiguous()
            if z.shape[0] != 1:
                z = z[:1]
            t_lat = int(z.shape[2])
            actions_all = []
            clip_actions4 = []
            clip_positions16 = []
            sample_dir = osp.join(run_dir, f"sample_{sample_idx:02d}")
            os.makedirs(sample_dir, exist_ok=True)

            for seg in range(len(points) - 1):
                abs_end_lat = (int(points[seg + 1]) - 1) // 4 + 1
                abs_start_lat = max(1, int(abs_end_lat) - 4)
                if abs_start_lat > t_lat:
                    break
                end_lat = min(abs_end_lat, t_lat)
                z5 = z[:, :, (abs_start_lat - 1) : end_lat].contiguous()
                if int(z5.shape[2]) < 2:
                    break
                if int(z5.shape[2]) < 5:
                    rep = z5[:, :, -1:].repeat(1, 1, 5 - int(z5.shape[2]), 1, 1)
                    z5 = torch.cat([z5, rep], dim=2)

                a4 = self._tf_predict_actions_4(z5, args.device).tolist()
                actions_all.extend(a4)
                clip_actions4.append(a4)
                if len(clip_positions16) == 0:
                    # 没有 GT 锚点时，退回到原点开始积分。
                    start_pose = [0.0] * 6
                else:
                    start_pose = clip_positions16[-1][-1]
                clip_positions16.append(self._tf_clip_positions_from_start(start_pose, a4))
                out_mp4 = osp.join(sample_dir, f"seg{seg:02d}_latent5.mp4")
                self._tf_decode_and_save_clip(z5, out_mp4=out_mp4, fps=int(args.video_fps), drop_first=(seg > 0))

            ref_pose_eval = None
            if self._tf_ref_pose_json and osp.exists(self._tf_ref_pose_json):
                try:
                    with open(self._tf_ref_pose_json, "r", encoding="utf-8") as f:
                        ref_poses = json.load(f)
                    if isinstance(ref_poses, list) and len(ref_poses) > 1 and isinstance(ref_poses[0], list):
                        # 计算 MSE 前，把每个 clip 都重新锚定到 GT 的 clip 起点。
                        clip_positions16 = []
                        for seg, a4 in enumerate(clip_actions4):
                            if seg >= len(points) - 1:
                                break
                            start_fid = int(points[seg])
                            if start_fid - 1 >= len(ref_poses):
                                break
                            clip_positions16.append(self._tf_clip_positions_from_start(ref_poses[start_fid - 1], a4))
                        ref_pose_eval = self._tf_clip_mse_vs_ref(clip_actions4, points, ref_poses)
                except Exception as e:
                    ref_pose_eval = {"error": str(e)}

            payload = {
                "global_it": int(g_it) + 1,
                "video_frames": int(args.video_frames),
                "step_frames": int(self._tf_dump_step),
                "points": points,
                "expand_factor": int(self._tf_expand_factor),
                "num_actions": len(actions_all),
                "units": {"translation": "cm", "angles": "deg"},
                "actions": actions_all,
                "clip_actions4": clip_actions4,
                "clip_positions16": clip_positions16,
                "relative_poses": self._tf_integrate(actions_all),
                "ref_pose_json": self._tf_ref_pose_json if self._tf_ref_pose_json else None,
                "clip_mse_vs_ref": ref_pose_eval,
                "note": "Teacher-forcing debug export from VAE raw_features (not autoregressive sampled video).",
            }
            with open(osp.join(sample_dir, "trajectory.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[TF-TS] 轨迹调试结果已导出到: {run_dir}")

    def train_step(
        self, epoch: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float, metric_lg: misc.MetricLogger,
        raw_features_bcthw: FTen, feature_cache_files4images: list, media: str,
        inp_B3HW: FTen, text_cond_tuple: Union[ITen, FTen], args: arg_util.Args,
        grpo_rewards: Optional[list] = None,
        grpo_old_logprobs: Optional[list] = None,
        grpo_adv_finals: Optional[list] = None,
        grpo_reward_acts: Optional[list] = None,
        grpo_reward_tasks: Optional[list] = None,
        grpo_reward_task_raws: Optional[list] = None,
        grpo_reward_task_dense_raws: Optional[list] = None,
        grpo_reward_task_success_raws: Optional[list] = None,
        grpo_succs: Optional[list] = None,
        grpo_succ_trajs: Optional[list] = None,
        grpo_task_final_costs: Optional[list] = None,
        grpo_task_final_pos_errs: Optional[list] = None,
        grpo_task_final_yaw_errs: Optional[list] = None,
        grpo_reward_ces: Optional[list] = None,
        grpo_ref_logprobs: Optional[list] = None,
        grpo_group_ids: Optional[list] = None,
        grpo_clip_ids: Optional[list] = None,
        grpo_trace_files: Optional[list] = None,
        traj_ids: Optional[list] = None,
        hybrid_roles: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        执行一次训练 step，内部可走 SFT 或 GRPO 分支。

        推荐阅读顺序：
        1. 先看 `trainer_type / hybrid_roles / use_grpo`，确认这一批样本走 SFT 还是 GRPO；
        2. 再看 `video_encode()` 如何把原始视觉输入打包成 GPT 可消费的视觉 token；
        3. 如果走 GRPO，再看 `old_logprob -> new_logprob -> ratio -> clipped objective -> backward` 这条主链。

        对小白最重要的区别：
        - SFT：核心是 teacher-forcing 下的视觉 token 监督；
        - GRPO：核心是“拿 rollout 时记录的旧 logprob 和 reward/advantage，重新计算新策略 logprob，再做 PPO/GRPO 目标”。

        读 `grpo_*` 参数时可以这样记：
        - `grpo_old_logprobs`：StageA 采样时，旧模型对“实际采到的 token”的概率；
        - `grpo_trace_files`：StageA 采样路径记录，里面保存了每一步采到的 token、scale、CFG、tau 等；
        - `grpo_adv_finals`：reward_uavflow.py 汇总后的最终训练权重，越大越鼓励模型以后更容易采到这条轨迹；
        - `grpo_reward_* / grpo_succ* / grpo_task_*`：reward 的拆分项，主要用于日志和兜底重建权重。

        这个函数可以按 7 段读：

        1. 公共输入准备：
           把 RGB 或已缓存 `raw_features_bcthw` 整理成 `raw_features_list`。

        2. 公共 token packing：
           调 `video_encode()` 得到 `x_BLC`、`gt_BLC`、mask、RoPE 和 scale packing 信息。

        3. 公共 teacher-forcing 前向：
           调 `self.gpt(...)` 得到每个 token 的 CE loss/acc。SFT 会直接反传它；
           GRPO 严格模式下它主要作为 metric/aux，真正梯度来自后面的 replay logprob。

        4. GRPO metadata 解析：
           把 batch 里的 `grpo_old_logprob`、`grpo_adv_final`、reward 分项、trace 文件转成 tensor。

        5. GRPO new_logprob 计算：
           `trace_replay` 逐样本回放 KV-cache 采样路径；`trace_ce` 用 teacher-forcing 单次前向近似重算。

        6. loss 组合：
           SFT 用 scale-weighted CE；GRPO 用 PPO/GRPO policy objective，可加 KL 和 aux SFT。

        7. 反向与记录：
           `AmpOptimizer.backward_clip_step()` 负责 backward/clip/step，再记录 SFT 和 GRPO 指标。
        """
        device = args.device
        B = len(inp_B3HW) + len(raw_features_bcthw)

        if media == 'images':
            is_image_batch = 1
        else:
            is_image_batch = 0
        # 公共前向准备：先把图像编码成 VAE raw_features，再做 packed transformer 前向。
        # SFT 和 GRPO 都需要这一步，因为二者都基于“文本条件 + 视频 token 序列”训练世界模型。
        with self.gpt_opt.amp_ctx:
            with torch.amp.autocast('cuda', enabled=False):
                raw_features_list = []
                if len(inp_B3HW):
                    with torch.no_grad():
                        for inp_ind, inp in enumerate(inp_B3HW):
                            # VAE 在这里是 token/latent 提供者，不是本 step 的优化对象。
                            raw_features_, _, _ = self.vae_local.encode_for_raw_features(inp.unsqueeze(0), scale_schedule=None, slice=args.use_slice)
                            raw_features_list.append(raw_features_)
                            if args.use_vae_token_cache and args.save_vae_token_cache and (not osp.exists(feature_cache_files4images[inp_ind])):
                                os.makedirs(osp.dirname(feature_cache_files4images[inp_ind]), exist_ok=True)
                                save_token_queue.put((raw_features_.cpu().data, [feature_cache_files4images[inp_ind]]))
                if len(raw_features_bcthw):
                    raw_features_bcthw = [item.unsqueeze(0) for item in raw_features_bcthw]
                    raw_features_list = raw_features_list + raw_features_bcthw

            full_pts_this_batch = [item.shape[-3] for item in raw_features_list]
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
            # video_encode 是 SFT/GRPO 共用的数据打包入口：
            # - x_BLC：模型输入视觉 token；
            # - gt_BLC：teacher-forcing 目标 token；
            # - sequece_packing_scales / other_info_by_scale：后面把扁平 loss 切回各尺度用。
            #
            # 这里是理解 SFT 作用的关键：
            # 世界模型训练时并不是直接看 RGB，也不是直接预测动作，而是预测 VAE/BSQ token。
            # SFT 就是在这里把真实视频 token 当答案，让 Infinity 学会“下一个 latent/token 应该是什么”。
            x_BLC, x_BLC_mask, gt_BLC, pred_all_bit_indices, visual_rope_cache, sequece_packing_scales, super_scale_lengths, super_querysid_super_refsid, other_info_by_scale = self.video_encode(
                vae=self.vae_local,
                inp_B3HW=None,
                vae_features=raw_features_list,
                self_correction=self.self_correction,
                args=args,
                device=device,
                rope2d_freqs_grid=self.gpt.rope2d_freqs_grid,
                dynamic_resolution_h_w=self.dynamic_resolution_h_w,
                text_lens=lens,
                tokens_remain=args.train_max_token_len,
            )

            # 严格 GRPO 的梯度来自“replay 后的新 logprob”，而不是普通 CE 图。
            # 如果 replay 图和 packed CE 图同时保留，显存会非常大，因此这里先做模式切分。
            trainer_type = str(getattr(args, "trainer_type", "sft") or "sft").strip().lower()
            # hybrid 模式允许数据集按 batch 指定当前应走 SFT 还是 GRPO。
            # 例如 12 条 GRPO clip + 1 条 SFT anchor 的混训数据，可以在 batch 级别切换 role。
            if isinstance(hybrid_roles, list) and len(hybrid_roles) > 0:
                uniq = {str(x or "").strip().lower() for x in hybrid_roles}
                uniq.discard("")
                if len(uniq) == 1:
                    role0 = next(iter(uniq))
                    if role0 == "sft":
                        trainer_type = "sft"
                    elif role0 == "grpo":
                        trainer_type = "grpo"
            use_grpo = trainer_type == "grpo"
            new_mode = str(getattr(args, "grpo_new_logprob_mode", "trace_replay") or "trace_replay").strip().lower()
            aux = float(getattr(args, "grpo_aux_sft_coef", 0.0) or 0.0)
            pg_only_flag = int(getattr(args, "grpo_pg_only", 1) or 0) == 1
            # 严格 GRPO 中，真正带梯度的是 PPO 目标。
            # 因此大 packed forward 可以先放进 no_grad，只保留统计值，把显存让给后面的 replay/new-logprob 分支。
            pg_only = bool(use_grpo and new_mode in ("trace_replay", "trace_ce") and pg_only_flag)

            if pg_only:
                with torch.no_grad():
                    loss, acc_bit, valid_sequence_ratio = self.gpt(
                        text_cond_tuple,
                        x_BLC,
                        gt_BL=gt_BLC,
                        is_image_batch=is_image_batch,
                        visual_rope_cache=visual_rope_cache,
                        sequece_packing_scales=sequece_packing_scales,
                        super_scale_lengths=super_scale_lengths,
                        super_querysid_super_refsid=super_querysid_super_refsid,
                        other_info_by_scale=other_info_by_scale,
                    )  # 中文说明：loss & acc_bit: [seq_len] (metrics only)
            else:
                loss, acc_bit, valid_sequence_ratio = self.gpt(
                    text_cond_tuple,
                    x_BLC,
                    gt_BL=gt_BLC,
                    is_image_batch=is_image_batch,
                    visual_rope_cache=visual_rope_cache,
                    sequece_packing_scales=sequece_packing_scales,
                    super_scale_lengths=super_scale_lengths,
                    super_querysid_super_refsid=super_querysid_super_refsid,
                    other_info_by_scale=other_info_by_scale,
                )  # 中文说明：loss & acc_bit: [seq_len]

            # 多尺度损失重加权：按尺度体积比调节不同 level 的贡献。
            # SFT 时它决定最终 CE loss 怎么平均；GRPO 时它决定每个样本/scale 的 PPO 目标怎么广播和聚合。
            acc_pt2scale_acc = {}
            acc_pt2scale_acc_counter = {}
            for full_pt, scale_schedule in self.dynamic_resolution_h_w[self.h_div_w_templates[0]][args.pn]['pt2scale_schedule'].items():
                acc_pt2scale_acc[full_pt] = [[] for _ in range(len(scale_schedule))]
                acc_pt2scale_acc_counter[full_pt] = [0 for _ in range(len(scale_schedule))]

            flatten_L_list, flatten_acc_bit_list, flatten_weight_list = [], [], []
            flatten_pg_obj_list = []
            flatten_sample_ind_list: List[int] = []
            ptr = 0
            global_scale_ind = 0
            # 中文说明：NOTE: use_grpo already computed above (keep consistent).
            reward_t = None
            reward_act_t = None
            reward_task_t = None
            reward_task_raw_t = None
            reward_task_dense_raw_t = None
            reward_task_success_raw_t = None
            succ_t = None
            succ_traj_t = None
            task_cost_t = None
            task_pos_err_t = None
            task_yaw_err_t = None
            reward_ce_t = None
            weight_t = None
            oldlp_t = None
            pg_by_sample: Optional[torch.Tensor] = None
            aux_sft_by_sample: Optional[torch.Tensor] = None
            if use_grpo:
                # -------------------------
                # GRPO metadata 入口
                # -------------------------
                # StageA/reward_uavflow.py 已经把每条 replay row 的 reward、advantage、old logprob
                # 和 trace 文件写进 batch。这里把它们转成 tensor，供后面 PPO 目标使用。
                #
                # 通俗地说，StageA 做的是“试着生成很多条未来 latent，并给每条打分”；
                # StageB 到这里拿到的是每条样本的训练账本：
                # - oldlp_t：当时旧模型多大概率会生成这条；
                # - weight_t：这条轨迹值不值得鼓励；
                # - trace：这条轨迹具体采了哪些 token。
                n_s = len(raw_features_list)
                if isinstance(grpo_rewards, list) and len(grpo_rewards) == n_s:
                    reward_t = torch.tensor(grpo_rewards, dtype=loss.dtype, device=loss.device)
                if isinstance(grpo_old_logprobs, list) and len(grpo_old_logprobs) == len(raw_features_list):
                    oldlp_t = torch.tensor(grpo_old_logprobs, dtype=loss.dtype, device=loss.device)
                if bool(int(getattr(args, "grpo_require_old_logprob", 1))):
                    # strict GRPO 必须有 old logprob 和 trace。
                    # 没有 old logprob，就无法算 ratio = exp(new-old)；
                    # 没有 trace，就无法保证 new logprob 是在同一批 sampled tokens 上重算的。
                    if oldlp_t is None:
                        raise RuntimeError(
                            "GRPO strict mode 需要每个样本都有 grpo_old_logprob；当前列表缺失或长度不匹配。"
                        )
                    if isinstance(grpo_trace_files, list) and len(grpo_trace_files) == n_s:
                        for i_tf, tf in enumerate(grpo_trace_files):
                            if not isinstance(tf, list) or len(tf) <= 0:
                                raise RuntimeError(f"GRPO strict mode 需要 grpo_trace_files；sample[{i_tf}] 缺失")
                    else:
                        raise RuntimeError("GRPO strict mode 需要与 batch 对齐的 grpo_trace_files 列表")
                # 首选直接消费 StageA 预先算好的 final advantage。
                # 兼容路径则在当前 batch 内按 reward 字段重建权重。
                if isinstance(grpo_adv_finals, list) and len(grpo_adv_finals) == n_s:
                    # `grpo_adv_final` 通常来自 reward_uavflow.py：
                    # 它已经做过非负裁剪、整轨迹 gate 和安全检查，是训练最优先使用的权重。
                    w_np = np.asarray([float(x) for x in grpo_adv_finals], dtype=np.float64)
                    adv_clip = float(getattr(args, "grpo_adv_clip", 0.0))
                    if adv_clip and adv_clip > 0:
                        w_np = np.clip(w_np, -adv_clip, adv_clip)
                    weight_t = torch.tensor(w_np, dtype=loss.dtype, device=loss.device)
                if (
                    isinstance(grpo_reward_acts, list)
                    and len(grpo_reward_acts) == n_s
                    and isinstance(grpo_reward_tasks, list)
                    and len(grpo_reward_tasks) == n_s
                ):
                    r_act_np = np.asarray([float(x) for x in grpo_reward_acts], dtype=np.float64)
                    r_task_np = np.asarray([float(x) for x in grpo_reward_tasks], dtype=np.float64)
                    r_ce_np = (
                        np.asarray([float(x) for x in grpo_reward_ces], dtype=np.float64)
                        if isinstance(grpo_reward_ces, list) and len(grpo_reward_ces) == n_s
                        else np.zeros((n_s,), dtype=np.float64)
                    )
                    # reward 分项主要用于日志、兜底权重重建和诊断：
                    # r_act 约束动作轨迹像专家，r_task 看任务进展，r_ce/reference 限制偏离旧策略。
                    reward_act_t = torch.tensor(r_act_np, dtype=loss.dtype, device=loss.device)
                    reward_task_t = torch.tensor(r_task_np, dtype=loss.dtype, device=loss.device)
                    if isinstance(grpo_reward_task_raws, list) and len(grpo_reward_task_raws) == n_s:
                        reward_task_raw_t = torch.tensor(
                            np.asarray([float(x) for x in grpo_reward_task_raws], dtype=np.float64),
                            dtype=loss.dtype,
                            device=loss.device,
                        )
                    if isinstance(grpo_reward_task_dense_raws, list) and len(grpo_reward_task_dense_raws) == n_s:
                        reward_task_dense_raw_t = torch.tensor(
                            np.asarray([float(x) for x in grpo_reward_task_dense_raws], dtype=np.float64),
                            dtype=loss.dtype,
                            device=loss.device,
                        )
                    if isinstance(grpo_reward_task_success_raws, list) and len(grpo_reward_task_success_raws) == n_s:
                        reward_task_success_raw_t = torch.tensor(
                            np.asarray([float(x) for x in grpo_reward_task_success_raws], dtype=np.float64),
                            dtype=loss.dtype,
                            device=loss.device,
                        )
                    if isinstance(grpo_succs, list) and len(grpo_succs) == n_s:
                        succ_t = torch.tensor(
                            np.asarray([float(x) for x in grpo_succs], dtype=np.float64),
                            dtype=loss.dtype,
                            device=loss.device,
                        )
                    if isinstance(grpo_succ_trajs, list) and len(grpo_succ_trajs) == n_s:
                        succ_traj_t = torch.tensor(
                            np.asarray([float(x) for x in grpo_succ_trajs], dtype=np.float64),
                            dtype=loss.dtype,
                            device=loss.device,
                        )
                    if isinstance(grpo_task_final_costs, list) and len(grpo_task_final_costs) == n_s:
                        task_cost_t = torch.tensor(
                            np.asarray([float(x) for x in grpo_task_final_costs], dtype=np.float64),
                            dtype=loss.dtype,
                            device=loss.device,
                        )
                    if isinstance(grpo_task_final_pos_errs, list) and len(grpo_task_final_pos_errs) == n_s:
                        task_pos_err_t = torch.tensor(
                            np.asarray([float(x) for x in grpo_task_final_pos_errs], dtype=np.float64),
                            dtype=loss.dtype,
                            device=loss.device,
                        )
                    if isinstance(grpo_task_final_yaw_errs, list) and len(grpo_task_final_yaw_errs) == n_s:
                        task_yaw_err_t = torch.tensor(
                            np.asarray([float(x) for x in grpo_task_final_yaw_errs], dtype=np.float64),
                            dtype=loss.dtype,
                            device=loss.device,
                        )
                    reward_ce_t = torch.tensor(r_ce_np, dtype=loss.dtype, device=loss.device)
                    lam_act = float(getattr(args, "grpo_lambda_act", 1.0))
                    lam_task = float(getattr(args, "grpo_lambda_task", 1.0))
                    lam_ce = float(getattr(args, "grpo_lambda_ce", 0.0))
                    # 组合 reward 只是兜底/诊断分数；如果前面已有 grpo_adv_final，训练权重仍优先用 weight_t。
                    r_np = lam_act * r_act_np + lam_task * r_task_np + lam_ce * r_ce_np
                    if reward_t is None:
                        reward_t = torch.tensor(r_np, dtype=loss.dtype, device=loss.device)
                    if weight_t is not None:
                        mode = ""
                    else:
                        mode = str(getattr(args, "grpo_weight_mode", "raw_reward") or "raw_reward").strip().lower()

                    # 如果数据提供 group_id，则按组计算排名/门控；否则每个样本单独成组。
                    groups: Dict[str, List[int]] = {}
                    if isinstance(grpo_group_ids, list) and len(grpo_group_ids) == n_s:
                        for i, gid in enumerate(grpo_group_ids):
                            key = str(gid) if str(gid) else f"__single_{i}"
                            groups.setdefault(key, []).append(i)
                    else:
                        for i in range(n_s):
                            groups[f"__single_{i}"] = [i]

                    w_np = np.asarray(r_np, dtype=np.float64).copy()
                    if mode == "gate_mean":
                        for _, inds in groups.items():
                            idx = np.asarray(inds, dtype=np.int64)
                            mu = float(np.mean(r_np[idx])) if idx.size > 0 else 0.0
                            m = (r_np[idx] >= mu).astype(np.float64)
                            w_np[idx] = w_np[idx] * m
                    elif mode == "rank_gate":
                        clip_ids_np = np.asarray(
                            [int(x) if x is not None else 1 for x in (grpo_clip_ids or [1] * n_s)],
                            dtype=np.int64,
                        )

                        def _rank01_average_ties(vals: np.ndarray) -> np.ndarray:
                            """把组内 reward 排名归一化到 `[0, 1]`，并让并列项共享平均名次。"""
                            n = int(vals.shape[0])
                            if n <= 1:
                                return np.zeros((n,), dtype=np.float64)
                            order = np.argsort(vals, kind="mergesort")
                            ranks = np.zeros((n,), dtype=np.float64)
                            i = 0
                            while i < n:
                                j = i
                                while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                                    j += 1
                                avg_rank = 0.5 * (i + j)
                                ranks[order[i : j + 1]] = avg_rank
                                i = j + 1
                            return ranks / float(max(1, n - 1))

                        alpha = float(getattr(args, "grpo_alpha_decay", 0.9))
                        w_rank = np.zeros((n_s,), dtype=np.float64)
                        for _, inds in groups.items():
                            idx = np.asarray(inds, dtype=np.int64)
                            # 在组内按合成 reward 排名；并列项共享平均名次。
                            s = _rank01_average_ties(r_np[idx])
                            mu = float(np.mean(r_np[idx])) if idx.size > 0 else 0.0
                            m = (r_np[idx] >= mu).astype(np.float64)
                            decay = np.power(alpha, np.maximum(0, clip_ids_np[idx] - 1))
                            w_rank[idx] = decay * m * s
                        w_np = w_rank
                    elif mode == "raw_reward":
                        # raw_reward：保留 w_np 为原始组合 reward。
                        pass
                    if mode:
                        # 可选安全裁剪：不改变排序，只限制数值幅度。
                        adv_clip = float(getattr(args, "grpo_adv_clip", 0.0))
                        if adv_clip and adv_clip > 0:
                            w_np = np.clip(w_np, -adv_clip, adv_clip)
                        weight_t = torch.tensor(w_np, dtype=loss.dtype, device=loss.device)
                elif reward_t is not None and weight_t is None:
                    # 兜底逻辑：只有 grpo_reward 时使用。
                    w = reward_t
                    adv_clip = float(getattr(args, "grpo_adv_clip", 0.0))
                    if adv_clip and adv_clip > 0:
                        w = w.clamp(min=-adv_clip, max=adv_clip)
                    weight_t = w
                # 安全兜底：任何 NaN/Inf 都先转成 0，避免污染 PPO loss。
                if weight_t is not None:
                    weight_t = torch.nan_to_num(weight_t, nan=0.0, posinf=0.0, neginf=0.0)
                    if bool(int(getattr(args, "grpo_require_nonnegative_adv", 1))):
                        if torch.any(weight_t < -1e-6):
                            raise RuntimeError("在 0410 非负权重方案下，grpo_adv_final 不应包含负权重")
                if reward_t is not None:
                    reward_t = torch.nan_to_num(reward_t, nan=0.0, posinf=0.0, neginf=0.0)
                if reward_act_t is not None:
                    reward_act_t = torch.nan_to_num(reward_act_t, nan=0.0, posinf=0.0, neginf=0.0)
                if reward_task_t is not None:
                    reward_task_t = torch.nan_to_num(reward_task_t, nan=0.0, posinf=0.0, neginf=0.0)
                if reward_task_raw_t is not None:
                    reward_task_raw_t = torch.nan_to_num(reward_task_raw_t, nan=0.0, posinf=0.0, neginf=0.0)
                if task_cost_t is not None:
                    task_cost_t = torch.nan_to_num(task_cost_t, nan=0.0, posinf=0.0, neginf=0.0)
                if task_pos_err_t is not None:
                    task_pos_err_t = torch.nan_to_num(task_pos_err_t, nan=0.0, posinf=0.0, neginf=0.0)
                if task_yaw_err_t is not None:
                    task_yaw_err_t = torch.nan_to_num(task_yaw_err_t, nan=0.0, posinf=0.0, neginf=0.0)
                if reward_ce_t is not None:
                    reward_ce_t = torch.nan_to_num(reward_ce_t, nan=0.0, posinf=0.0, neginf=0.0)
                if oldlp_t is not None:
                    oldlp_t = torch.nan_to_num(oldlp_t, nan=0.0, posinf=0.0, neginf=0.0)
                # PPO 比率 `exp(logp_new - logp_old)` 必须在同一条采样 token 轨迹上计算。
                # 所以严格路径默认回放 StageA 保存的 trace。
                newlp_t: Optional[torch.Tensor] = None
                new_mode = str(getattr(args, "grpo_new_logprob_mode", "trace_replay") or "trace_replay").strip().lower()
                if oldlp_t is not None and weight_t is not None and new_mode in ("trace_replay", "trace_ce"):
                    # -------------------------
                    # GRPO new_logprob 计算
                    # -------------------------
                    # old_logprob 是 StageA rollout 时的策略概率；
                    # new_logprob 是当前正在训练的模型对同一批 token 重新打分。
                    # 二者差值决定 PPO ratio。这里支持两种打分模式：
                    # - trace_replay：严格复现采样路径和 KV cache，最准确但显存最重；
                    # - trace_ce：用 teacher-forcing 单次前向重算选中 token 的 CE/logprob，更省显存。
                    #
                    # 为什么不能直接拿上面 SFT 的 CE loss 当 new_logprob？
                    # 因为 PPO 比较的是“同一条采样轨迹”在旧模型和新模型下的概率。
                    # 普通 SFT forward 看的是真实视频 token；GRPO 要重算的是 StageA 采出来的 token。
                    # 如果两批 token 不一致，ratio = exp(new-old) 就没有训练意义。
                    if not (isinstance(grpo_trace_files, list) and len(grpo_trace_files) == n_s):
                        raise RuntimeError("trace_replay 需要与 batch 对齐的 grpo_trace_files")
                    # 从 packed 的 text_cond_tuple 中切出单样本版本，供逐样本 replay 使用。
                    def _slice_text_cond_tuple(tup, sample_i: int):
                        """从 packed 文本条件里切出第 `sample_i` 个样本。"""
                        kv_compact, lens, cu_seqlens_k, max_seqlen_k = tup
                        le = int(lens[sample_i])
                        st = int(cu_seqlens_k[sample_i].item()) if hasattr(cu_seqlens_k[sample_i], "item") else int(cu_seqlens_k[sample_i])
                        ed = st + le
                        kv_i = kv_compact[st:ed]
                        lens_i = [le]
                        cu_i = torch.tensor([0, le], device=kv_compact.device, dtype=cu_seqlens_k.dtype)
                        return (kv_i, lens_i, cu_i, le)

                    # 构造一个最小 infer-args，仅包含推理回放真正需要的字段。
                    import types as _types
                    base = vars(args).copy()
                    base.setdefault("use_cfg", 1)
                    base.setdefault("use_apg", 0)
                    base.setdefault("max_repeat_times", int(getattr(args, "max_repeat_times", 999999)))
                    base.setdefault("apg_norm_threshold", float(getattr(args, "apg_norm_threshold", 0.0)))
                    infer_args = _types.SimpleNamespace(**base)

                    # trace-replay 不是普通 `model.forward`，所以 FSDP 根模块上的参数
                    # 可能仍是分片状态。这里仅临时 unshard 根模块，避免整模型全量 gather。
                    try:
                        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
                    except Exception:
                        FSDP = None  # type: ignore
                    gpt_fsdp = getattr(self, "gpt", None)
                    use_root_fsdp_unshard = FSDP is not None and gpt_fsdp is not None and isinstance(gpt_fsdp, FSDP)
                    if use_root_fsdp_unshard:
                        gpt_replay = gpt_fsdp.module
                        fsdp_ctx = FSDP.summon_full_params(gpt_fsdp, recurse=False, writeback=False)
                    else:
                        gpt_replay = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, "_orig_mod") else self.gpt_wo_ddp
                        fsdp_ctx = nullcontext()

                    model_dtype = next(iter(gpt_replay.parameters())).dtype
                    newlp_t = torch.zeros((n_s,), dtype=loss.dtype, device=loss.device)
                    tok_t = torch.ones((n_s,), dtype=loss.dtype, device=loss.device)
                    if float(aux) > 0 and new_mode == "trace_ce":
                        aux_sft_by_sample = torch.zeros((n_s,), dtype=loss.dtype, device=loss.device)
                    with fsdp_ctx:
                        for si in range(n_s):
                            tfl = grpo_trace_files[si]
                            tf0 = tfl[0] if isinstance(tfl, list) and len(tfl) > 0 else ""
                            if not tf0 or (not osp.exists(tf0)):
                                raise RuntimeError(f"trace_replay 缺少 sample[{si}] 的 trace 文件: {tf0}")
                            tr = torch.load(tf0, map_location="cpu")
                            idx_trace = tr.get("idx_trace", None)
                            if idx_trace is None:
                                raise RuntimeError(f"trace_replay 在 {tf0} 中缺少 idx_trace")
                            # idx_trace 是 StageA 采样时真正选中的 token。
                            # PPO/GRPO 不能在另一批 token 上算 new logprob，否则 ratio 没有意义。
                            clipid_target = tr.get("clipid_target", None)
                            try:
                                clipid_target_i = int(clipid_target) if clipid_target is not None else None
                            except Exception:
                                clipid_target_i = None
                            # 选中 token 的数量，用来做 KL / logprob 归一化。
                            tok_cnt = 0
                            try:
                                # 若 trace 记录了 step_clipids，则只统计目标 clip 的 token 数。
                                step_clipids = tr.get("step_clipids", None)
                                use_step_clipids = (
                                    clipid_target_i is not None
                                    and isinstance(step_clipids, list)
                                    and isinstance(idx_trace, list)
                                    and len(step_clipids) == len(idx_trace)
                                )
                                if isinstance(idx_trace, list):
                                    for k, t in enumerate(idx_trace):
                                        if use_step_clipids and int(step_clipids[k]) != int(clipid_target_i):
                                            continue
                                        if isinstance(t, torch.Tensor):
                                            tok_cnt += int(t.numel())
                                        else:
                                            tok_cnt += int(np.asarray(t).size)
                                else:
                                    tok_cnt = int(np.asarray(idx_trace).size)
                                tok_t[si] = float(max(1, tok_cnt))
                            except Exception:
                                tok_t[si] = 1.0
                            scale_schedule = tr.get("scale_schedule", None)
                            context_info = tr.get("context_info", None)
                            if scale_schedule is None or context_info is None:
                                raise RuntimeError(
                                    f"trace_replay 在 {tf0} 中需要 scale_schedule/context_info（请在服务端更新后重跑 StageA）"
                                )
                            cfg_scale = float(tr.get("infinity_cfg", 1.0))
                            # 调试开关：在 StageB 强制 CFG=1，可把 cond/uncond 双分支缩成单分支，
                            # 但会破坏严格 PPO 语义，只适合排查 OOM。
                            try:
                                import os as _os
                                v = _os.environ.get("INFINITY_GRPO_FORCE_CFG", "").strip()
                                if v:
                                    cfg_scale = float(v)
                            except Exception:
                                pass
                            tau_list = tr.get("tau_list", None)
                            if not isinstance(tau_list, list) or len(tau_list) < len(scale_schedule):
                                raise RuntimeError(f"trace_replay 需要 {tf0} 中的 tau_list 与 scale_schedule 对齐")
                            top_k = int(tr.get("top_k", 0))
                            top_p = float(tr.get("top_p", 0.0))
                            gt_leak = int(tr.get("gt_leak", -1))
                            gt_ls_Bl = tr.get("gt_ls_Bl", None)
                            if gt_leak > 0 and gt_ls_Bl is None:
                                raise RuntimeError(
                                    f"trace_replay 在 gt_leak>0 时需要 {tf0} 提供 gt_ls_Bl（请在服务端更新后重跑 StageA）"
                                )
                            # 可选调试：打印当前回放的是哪个 clip，方便定位 clip3 之类的特定 OOM。
                            try:
                                import os as _os
                                if int(_os.environ.get("INFINITY_GRPO_DEBUG_REPLAY", "0") or 0) == 1:
                                    rk = int(_os.environ.get("RANK", "-1") or -1)
                                    if rk in (-1, 0):
                                        print(
                                            f"[grpo][replay] si={si} clipid_target={clipid_target_i} "
                                            f"cfg={cfg_scale} gt_leak={gt_leak} scales={len(scale_schedule)} "
                                            f"tok_cnt={tok_cnt} trace={tf0}"
                                        )
                            except Exception:
                                pass
                            # `gt_ls_Bl` 可能非常大，先保持在 CPU，让 replay 按 step 懒加载到 GPU。
                            gt_ls_gpu = gt_ls_Bl
                            cfg_list = [cfg_scale] * int(len(scale_schedule))
                            label_i = _slice_text_cond_tuple(text_cond_tuple, si)
                            if new_mode == "trace_replay":
                                # 严格 trace-replay 的显存保护：把保存的 autograd tensor 挪到 CPU。
                                use_save_on_cpu = int(getattr(args, "grpo_replay_save_on_cpu", 1) or 0) == 1
                                pin_save_on_cpu = int(getattr(args, "grpo_replay_save_on_cpu_pin", 1) or 0) == 1
                                # 可选：只给最重的 clip 开 CPU offload，减少主机内存压力。
                                try:
                                    import os as _os
                                    if int(_os.environ.get("INFINITY_GRPO_SAVE_ON_CPU_CLIP3_ONLY", "0") or 0) == 1:
                                        use_save_on_cpu = bool(use_save_on_cpu and int(clipid_target_i or -1) == 3)
                                except Exception:
                                    pass
                                save_ctx = nullcontext()
                                if use_save_on_cpu:
                                    try:
                                        save_ctx = torch.autograd.graph.save_on_cpu(pin_memory=pin_save_on_cpu)
                                    except Exception:
                                        save_ctx = nullcontext()
                                with save_ctx:
                                    with torch.amp.autocast("cuda", dtype=model_dtype):
                                        lp = gpt_replay.ar_infer_infinity_elegant_replay_logprob(
                                            vae=self.vae_local,
                                            scale_schedule=scale_schedule,
                                            label_B_or_BLT=label_i,
                                            B=1,
                                            negative_label_B_or_BLT=None,
                                            g_seed=None,
                                            cfg_list=cfg_list,
                                            tau_list=[float(x) for x in tau_list],
                                            top_k=top_k,
                                            top_p=top_p,
                                            trunk_scale=1000,
                                            gt_leak=gt_leak,
                                            gt_ls_Bl=gt_ls_gpu,
                                            low_vram_mode=False,
                                            args=infer_args,
                                            get_visual_rope_embeds=self.get_visual_rope_embeds,
                                            context_info=context_info,
                                            forced_idx_trace=idx_trace,
                                            logprob_clipid_target=clipid_target_i,
                                            kv_cache_reset=True,
                                            skip_text_forward=False,
                                            cache_text_as_gt=False,
                                        )
                                newlp_t[si] = lp[0]
                            else:
                                # `trace_ce` 走 teacher-forcing 单次前向，不回放 KV cache，
                                # 属于更传统的 PPO / RLHF 新 logprob 计算方式。
                                # 它牺牲部分“采样过程完全复现”的严格性，换来更低显存和更稳定吞吐。
                                import math as _math
                                import json as _json
                                import torch.nn.functional as _F
                                from infinity.schedules.dynamic_resolution import get_first_full_spatial_size_scale_index as _ffssi
                                from infinity.schedules.infinity_elegant import interpolate as _interp

                                gpt_eval = getattr(self, "gpt", None)
                                if gpt_eval is None:
                                    raise RuntimeError("trace_ce 需要 self.gpt (FSDP/DDP) 才能执行一次 forward")

                                # 建立“尺度 -> 最后一次重复 step”的映射，确保和 StageA 的重复策略严格对齐。
                                try:
                                    img_rep_s = str(tr.get("image_scale_repetition", getattr(infer_args, "image_scale_repetition", "[1]"))).strip()
                                    vid_rep_s = str(tr.get("video_scale_repetition", getattr(infer_args, "video_scale_repetition", "[1]"))).strip()
                                    image_rep = np.array(_json.loads(img_rep_s))
                                    video_rep = np.array(_json.loads(vid_rep_s))
                                except Exception as e:
                                    raise RuntimeError(f"trace_ce 需要合法的 image/video_scale_repetition: {e}")
                                first_full = _ffssi(scale_schedule)
                                scales_in_one_clip = int(first_full) + 1
                                max_repeat_times = int(getattr(infer_args, "max_repeat_times", 999999))

                                rep_counts: List[int] = []
                                cache_step_id: Dict[int, int] = {}
                                step_ptr0 = 0
                                for _si, _pn in enumerate(scale_schedule):
                                    if _si < scales_in_one_clip:
                                        _rt = int(image_rep[_si % scales_in_one_clip])
                                    else:
                                        _rt = int(video_rep[_si % scales_in_one_clip])
                                    _rt = int(min(_rt, max_repeat_times))
                                    _rt = max(1, _rt)
                                    rep_counts.append(_rt)
                                    cache_step_id[int(_si)] = int(step_ptr0 + _rt - 1)
                                    step_ptr0 += _rt
                                total_steps = int(step_ptr0)
                                if not (isinstance(idx_trace, list) and len(idx_trace) >= total_steps):
                                    raise RuntimeError(
                                        f"trace_ce 期望 idx_trace 列表长度 >= total_steps ({len(idx_trace) if isinstance(idx_trace, list) else 'NA'} < {total_steps})"
                                    )

                                # 根据是否 patchify，准备 VAE 视角下的尺度表。
                                apply_patchify = bool(getattr(gpt_replay, "apply_spatial_patchify", False))
                                if apply_patchify:
                                    vae_scale_schedule = [(int(pt), int(2 * ph), int(2 * pw)) for (pt, ph, pw) in scale_schedule]
                                else:
                                    vae_scale_schedule = [(int(pt), int(ph), int(pw)) for (pt, ph, pw) in scale_schedule]

                                # 辅助函数：把 latent `[B, d, t, h, w]` 还原成原始视觉 token 序列。
                                def _latent_to_raw_tokens(lat: torch.Tensor) -> torch.Tensor:
                                    """把 latent 重排成 `[B, token_num, dim]` 的 token 序列。"""
                                    if apply_patchify:
                                        _x = lat.permute(0, 2, 1, 3, 4)  # [B, t, d, 2h, 2w]
                                        _x = torch.nn.functional.pixel_unshuffle(_x, 2)  # [B, t, 4d, h, w]
                                        _x = _x.permute(0, 2, 1, 3, 4)  # [B, 4d, t, h, w]
                                    else:
                                        _x = lat
                                    _x = _x.reshape(_x.shape[0], _x.shape[1], -1).permute(0, 2, 1).contiguous()
                                    return _x

                                # 按尺度拼装输入 / 标签 / rope，并用 idx_trace 更新 latent 状态。
                                # 这一段看起来复杂，是因为 `trace_ce` 要“手工还原一次 StageA 的生成现场”：
                                # 1. 从初始噪声/零 latent 开始；
                                # 2. 每个 scale 读取 StageA 真实采到的 idx_trace；
                                # 3. 把 idx_trace 反查成 codes，加回 summed_code；
                                # 4. 用更新后的 summed_code 作为后续 scale 的条件；
                                # 5. 最后只对目标 clip 的 token 求 logprob。
                                # 这样虽然没有回放 KV-cache，但至少保证当前模型打分的是“采样轨迹本身”。
                                B1 = 1
                                # latent 运算尽量沿用模型 dtype；真正前向时会再转成 fp32。
                                lat_dtype = model_dtype
                                device0 = loss.device
                                vae_embed_dim = int(
                                    getattr(gpt_replay, "vae_embed_dim", 0)
                                    or getattr(self.vae_local, "embed_dim", 0)
                                    or getattr(self.vae_local, "vae_embed_dim", 0)
                                    or 64
                                )
                                if getattr(gpt_replay.other_args, "noise_input", 0):
                                    noise0 = torch.randn((1, vae_embed_dim, *vae_scale_schedule[0]), device=device0, dtype=lat_dtype)
                                else:
                                    noise0 = torch.zeros((1, vae_embed_dim, *vae_scale_schedule[0]), device=device0, dtype=lat_dtype)
                                summed_code = noise0[0:1]
                                # 尺度选择必须确定性复现，优先使用 StageA 保存的选择结果。
                                select_si_list = None
                                try:
                                    sel = tr.get("trace_ce_select_si_list", None)
                                    if isinstance(sel, list) and len(sel) > 0:
                                        select_si_list = [int(x) for x in sel]
                                except Exception:
                                    select_si_list = None
                                if select_si_list is None:
                                    select_si_list = list(range(len(scale_schedule)))
                                try:
                                    total_tokens = int(np.array(scale_schedule).prod(-1).sum())
                                    tmax = int(tr.get("trace_ce_tmax", getattr(args, "train_max_token_len", 20480) or 20480) or 20480)
                                    if total_tokens > tmax and len(select_si_list) == len(scale_schedule):
                                        S = int(scales_in_one_clip)
                                        L = int(len(scale_schedule))
                                        c = int(clipid_target_i) if clipid_target_i is not None else 1
                                        if L == S * 4:
                                            if c <= 1:
                                                select_si_list = list(range(min(L, S + 11)))
                                            elif c == 2:
                                                select_si_list = [S - 1, 2 * S - 1] + list(range(2 * S, min(L, 2 * S + 11)))
                                            else:
                                                select_si_list = [S - 1, 2 * S - 1, 3 * S - 1] + list(range(3 * S, min(L, 3 * S + 11)))
                                        elif L == S * 3:
                                            if c <= 1:
                                                select_si_list = list(range(min(L, S + 11)))
                                            else:
                                                select_si_list = [S - 1, 2 * S - 1] + list(range(2 * S, min(L, 2 * S + 11)))
                                        else:
                                            # 兜底策略：保留完整首个 clip，再尽量从目标 clip 保留一个高分辨率尺度。
                                            select_si_list = list(range(min(L, S)))
                                            tgt = min(L - 1, c * S + (S - 1))
                                            if tgt not in select_si_list:
                                                select_si_list.append(tgt)
                                        # 确保唯一且不越界。
                                        select_si_list = sorted({int(x) for x in select_si_list if 0 <= int(x) < L})
                                except Exception:
                                    select_si_list = list(range(len(scale_schedule)))

                                # 把原始 context_info 里的 ref_sids 重映射到“当前选中的尺度子集”；
                                # 没有被选中的引用尺度会在这里被丢掉。
                                # 选择子集的原因是省显存：长视频所有 scale 全部 teacher-forcing 会非常重。
                                # 但如果丢掉引用关系，attention 依赖会错位，所以这里必须同步重映射 ref_sids。
                                real_si_2_new_si: Dict[int, int] = {int(r): int(i2) for i2, r in enumerate(select_si_list)}
                                new_scale_pack_info: Dict[int, Dict[str, Any]] = {}
                                for new_q, real_q in enumerate(select_si_list):
                                    new_scale_pack_info[int(new_q)] = {"ref_sids": []}
                                    try:
                                        refs = context_info[int(real_q)].get("ref_sids", [])
                                    except Exception:
                                        refs = []
                                    for rr in refs:
                                        nn = real_si_2_new_si.get(int(rr), None)
                                        if nn is not None:
                                            new_scale_pack_info[int(new_q)]["ref_sids"].append(int(nn))

                                # 为 `trace_ce` 的 teacher-forcing 单次前向构造选中尺度输入：
                                # `x_BLC` 是视觉 token，`gt_BL` 是对应 bit-label 监督，
                                # `visual_rope_cache` 则保证这些尺度的位置编码仍然对齐。
                                x_scales = []
                                gt_scales = []
                                rope_scales = []
                                dlabels = []
                                muls = []
                                clipids = []
                                for _si, _pn in enumerate(scale_schedule):
                                    pt, ph, pw = int(_pn[0]), int(_pn[1]), int(_pn[2])
                                    # 计算当前尺度输入：把累计 latent 下采样到当前尺度。
                                    this_lat = summed_code
                                    if tuple(this_lat.shape[-3:]) != tuple(vae_scale_schedule[_si]):
                                        this_lat = _F.interpolate(this_lat, size=vae_scale_schedule[_si], mode=self.vae_local.quantizer.z_interplote_down).contiguous()
                                    if int(_si) in real_si_2_new_si:
                                        x_scales.append(_latent_to_raw_tokens(this_lat))
                                        # RoPE cache 使用当前尺度最后一次重复对应的真实 step 编号。
                                        rope_scales.append(
                                            self.get_visual_rope_embeds(
                                                gpt_replay.rope2d_freqs_grid,
                                                scale_schedule,
                                                int(_si),
                                                int(cache_step_id[int(_si)]),
                                                device0,
                                                infer_args,
                                                context_info,
                                                int(first_full),
                                            )
                                        )
                                        mul = int(pt * ph * pw)
                                        muls.append(mul)
                                        d_label = int(
                                            getattr(gpt_replay.other_args, "detail_scale_dim", 64)
                                            if (ph * pw) >= int(getattr(self.vae_local.quantizer, "detail_scale_min_tokens", 350))
                                            else getattr(gpt_replay.other_args, "semantic_scale_dim", 16)
                                        )
                                        dlabels.append(d_label)
                                        clipids.append(int(_si // max(1, scales_in_one_clip)))
                                        forced = idx_trace[int(cache_step_id[int(_si)])]
                                        if not isinstance(forced, torch.Tensor):
                                            forced = torch.tensor(forced, dtype=torch.long, device=device0)
                                        else:
                                            forced = forced.to(device=device0, dtype=torch.long)
                                        if forced.ndim == 1:
                                            forced = forced.unsqueeze(0)
                                        gt_scales.append(forced.reshape(B1, mul, d_label).contiguous())

                                    # 用该尺度缓存 token 更新 latent 状态，保持后续条件一致。
                                    # codes 累加使用的目标空间尺寸。
                                    # 这就是世界模型 latent 的“逐尺度求和”逻辑：
                                    # 当前 scale 预测/采样出的 code 不是单独存在，而是上采样到目标分辨率后
                                    # 加到 `summed_code` 里；下一 scale 再基于这个累计 latent 继续预测。
                                    if _si < scales_in_one_clip:
                                        target_pn = vae_scale_schedule[int(first_full)]
                                    else:
                                        target_pn = vae_scale_schedule[-1]
                                    forced_upd = idx_trace[int(cache_step_id[int(_si)])]
                                    if not isinstance(forced_upd, torch.Tensor):
                                        forced_upd = torch.tensor(forced_upd, dtype=torch.long, device=device0)
                                    else:
                                        forced_upd = forced_upd.to(device=device0, dtype=torch.long)
                                    if forced_upd.ndim == 1:
                                        forced_upd = forced_upd.unsqueeze(0)
                                    mul = int(pt * ph * pw)
                                    d_label = int(
                                        getattr(gpt_replay.other_args, "detail_scale_dim", 64)
                                        if (ph * pw) >= int(getattr(self.vae_local.quantizer, "detail_scale_min_tokens", 350))
                                        else getattr(gpt_replay.other_args, "semantic_scale_dim", 16)
                                    )
                                    idx_Bld = forced_upd.reshape(B1, -1)
                                    idx_Bthwd = idx_Bld.reshape(B1, pt, ph, pw, d_label)
                                    if apply_patchify:
                                        _t = idx_Bthwd.permute(0, 1, 4, 2, 3)
                                        _t = torch.nn.functional.pixel_shuffle(_t, 2)
                                        idx_Bthwd = _t.permute(0, 1, 3, 4, 2)
                                    if gt_leak > 0 and gt_ls_Bl is not None and int(_si) < int(gt_leak):
                                        try:
                                            idx_Bthwd = gt_ls_Bl[int(cache_step_id[int(_si)])].to(device=device0, dtype=idx_Bthwd.dtype)
                                        except Exception:
                                            pass
                                    if getattr(gpt_replay.other_args, "use_two_stage_lfq", 0):
                                        if (ph * pw) >= int(getattr(self.vae_local.quantizer, "detail_scale_min_tokens", 350)):
                                            is_sem = False
                                            lfq = self.vae_local.quantizer.lfq_detail
                                        else:
                                            is_sem = True
                                            lfq = self.vae_local.quantizer.lfq_semantic
                                        codes = lfq.indices_to_codes(idx_Bthwd, "bit_label")
                                        codes = _interp(
                                            codes,
                                            size=(vae_embed_dim, *target_pn),
                                            mode=self.vae_local.quantizer.z_interplote_up,
                                            quantizer=self.vae_local.quantizer,
                                            is_semantic_scale=is_sem,
                                        ).contiguous()
                                    else:
                                        codes = self.vae_local.quantizer.lfq_detail.indices_to_codes(idx_Bthwd, "bit_label")
                                        codes = _F.interpolate(codes, size=target_pn, mode=self.vae_local.quantizer.z_interplote_up)
                                    summed_code = _F.interpolate(summed_code, size=target_pn, mode=self.vae_local.quantizer.z_interplote_up).contiguous()
                                    summed_code = summed_code + codes
                                    # 下一尺度输入会在下一轮循环开头统一处理，需要时再下采样。

                                    if _si < len(scale_schedule) - 1:
                                        if tuple(scale_schedule[int(_si)][-2:]) == tuple(scale_schedule[-1][-2:]):
                                            if getattr(gpt_replay.other_args, "noise_input", 0):
                                                summed_code = torch.randn((B1, summed_code.shape[1], *vae_scale_schedule[int(_si) + 1]), device=device0, dtype=summed_code.dtype)
                                            else:
                                                summed_code = torch.zeros((B1, summed_code.shape[1], *vae_scale_schedule[int(_si) + 1]), device=device0, dtype=summed_code.dtype)

                                x_vis = torch.cat(x_scales, dim=1) if len(x_scales) else torch.zeros((B1, 1, vae_embed_dim), device=device0, dtype=lat_dtype)
                                rope_vis = torch.cat(rope_scales, dim=4) if len(rope_scales) else self.get_visual_rope_embeds(
                                    gpt_replay.rope2d_freqs_grid,
                                    scale_schedule,
                                    0,
                                    0,
                                    device0,
                                    infer_args,
                                    context_info,
                                    int(first_full),
                                )

                                # 构造 super_scale_lengths 和 querysid_refsid，逻辑与 `video_encode` 保持一致。
                                kv_i, lens_i, cu_i, le = label_i
                                text_lens = list(lens_i)
                                scale_lengths = [int(np.array(scale_schedule[si]).prod()) for si in select_si_list] + [int(x) for x in text_lens]
                                valid_scales = int(len(select_si_list) + len(text_lens))
                                # 对齐 Infinity.forward 的 padding 规则：视觉+文本拼接后，
                                # train_with_var_seq_len=1 时会补齐到 pad_to_multiplier。
                                # 代码/形状说明：build_flex_attn_func asserts sum(super_scale_lengths) == padded_seq_len.
                                cur_seq_len = int(np.sum(scale_lengths))
                                try:
                                    if int(getattr(args, "train_with_var_seq_len", 0) or 0) == 1:
                                        pad_to = int(getattr(args, "pad_to_multiplier", 128) or 128)
                                        pad_to = max(1, pad_to)
                                        pad_seq_len = int(_math.ceil(cur_seq_len / float(pad_to)) * pad_to - cur_seq_len)
                                    else:
                                        pad_seq_len = int(getattr(args, "train_max_token_len", -1) or -1) - cur_seq_len
                                    pad_seq_len = int(max(0, pad_seq_len))
                                    if pad_seq_len > 0:
                                        scale_lengths = scale_lengths + [int(pad_seq_len)]
                                except Exception:
                                    pass
                                max_sid_nums = 2000
                                qref = torch.zeros((max_sid_nums, max_sid_nums), device=device0, dtype=torch.bool)
                                for i_sid in range(valid_scales):
                                    qref[i_sid][i_sid] = True
                                base = 0
                                # 代码/形状说明：Only one packed sample (B=1): ind=0, global_text_sid = len(flatten_packing_scales)+0.
                                for local_q in range(len(select_si_list)):
                                    global_q = local_q + base
                                    global_text_sid = len(select_si_list) + 0
                                    qref[global_q][global_text_sid] = True
                                    for local_r in new_scale_pack_info[int(local_q)]["ref_sids"]:
                                        qref[global_q][base + int(local_r)] = True

                                # 中文说明：Disable condition-drop randomness during policy scoring (keep deterministic ratio).
                                orig_cdr = float(getattr(gpt_replay, "cond_drop_rate", 0.0) or 0.0)
                                try:
                                    gpt_replay.cond_drop_rate = 0.0
                                except Exception:
                                    pass
                                try:
                                    # 保持 training mode，使 `full-block` checkpointing 仍然启用。
                                    with torch.amp.autocast("cuda", dtype=model_dtype):
                                        loss_tok, _, _ = gpt_eval(
                                            label_i,
                                            x_vis,
                                            gt_BL=gt_scales,
                                            is_image_batch=0,
                                            visual_rope_cache=rope_vis,
                                            sequece_packing_scales=[[tuple(map(int, scale_schedule[si])) for si in select_si_list]],
                                            super_scale_lengths=scale_lengths,
                                            super_querysid_super_refsid=qref,
                                            other_info_by_scale=None,
                                        )
                                finally:
                                    try:
                                        gpt_replay.cond_drop_rate = orig_cdr
                                    except Exception:
                                        pass

                                # 把按 d 维求均值的 token loss 转成 logprob，并只选目标 clip。
                                nll_target = torch.zeros((1,), dtype=loss.dtype, device=device0)
                                # 中文说明：Token counts:
                                # tok_cnt_elems 是 bit 元素数（mul*d_label），与求和 logprob 的单位一致，
                                # 用于选中 token 的 KL 归一化。
                                tok_cnt_elems = 0
                                tok_ptr = 0
                                for j, si_real in enumerate(select_si_list):
                                    mul = int(muls[j])
                                    seg = loss_tok[tok_ptr : tok_ptr + mul]
                                    tok_ptr += mul
                                    dlab = float(dlabels[j])
                                    if clipid_target_i is not None and int(clipids[j]) != int(clipid_target_i):
                                        continue
                                    nll_target = nll_target + seg.sum() * dlab
                                    tok_cnt_elems += int(mul) * int(dlabels[j])
                                newlp_t[si] = (-nll_target)[0]
                                # 可选辅助稳定项：只在目标 clip token 上计算 teacher-forcing CE 均值。
                                # 这是轻量的“质量/防坍缩”约束，不需要完整 packed backward。
                                if aux_sft_by_sample is not None:
                                    # `nll_target` 已按 (mul*d_label) 元素求和，这里再除以元素数做归一化。
                                    aux_sft_by_sample[si] = (nll_target / float(max(1, tok_cnt_elems)))[0]
                                # 用 trace_ce packing 得到的确定性计数覆盖 tok_t。
                                # 这样即使 idx_trace/step_clipids 计数失败，也不会造成 KL 归一化尖峰。
                                try:
                                    if int(tok_cnt_elems) > 0:
                                        tok_t[si] = float(tok_cnt_elems)
                                except Exception:
                                    pass

                    # 这里进入标准 PPO / GRPO 主目标：
                    # - `ratio = exp(new_logprob - old_logprob)`；
                    # - 优势/权重 >= 0 时取 `min(ratio*A, clip(ratio)*A)`；
                    # - 优势/权重 < 0 时取 `max(ratio*A, clip(ratio)*A)`。
                    # ratio 强制用 fp32 计算，避免 fp16/bf16 下 exp 溢出成 inf，
                    # 再与 0 相乘产生 nan。
                    delta = (newlp_t - oldlp_t).to(torch.float32).clamp(min=-60.0, max=60.0)
                    ratio = torch.exp(delta)
                    eps = float(getattr(args, "grpo_ratio_eps", 0.2))
                    ratio_clip = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
                    wt32 = weight_t.to(torch.float32)
                    # PPO clipped objective 的作用：
                    # - advantage/weight 高的样本，希望当前策略提高它的概率；
                    # - 但 ratio 不能涨太猛，所以和 clipped ratio 取保守值；
                    # - 如果未来允许负 advantage，则负样本会用 max 形式做对称处理。
                    unclipped = ratio * wt32
                    clipped = ratio_clip * wt32
                    obj = torch.where(wt32 >= 0, torch.minimum(unclipped, clipped), torch.maximum(unclipped, clipped))
                    pg_by_sample = (-obj).to(loss.dtype)
                    beta = float(getattr(args, "grpo_kl_beta", 0.0) or 0.0)
                    # 这里的 KL 不是全分布精确 KL，而是“离线采样 token 上的便宜近似”：
                    # 只在被选中的 token 上估计 `E[logpi_new - logpi_ref]`。
                    if beta > 0:
                        if isinstance(grpo_ref_logprobs, list) and len(grpo_ref_logprobs) == n_s:
                            ref_t = torch.tensor([float(x) for x in grpo_ref_logprobs], dtype=loss.dtype, device=loss.device)
                        else:
                            # 兜底：若没有显式 ref_logprob，就把 old policy 当成 ref。
                            # 这在 rollout checkpoint 与参考 checkpoint 相同的场景下是成立的。
                            ref_t = oldlp_t
                        ref_t = torch.nan_to_num(ref_t, nan=0.0, posinf=0.0, neginf=0.0)
                        tok_safe = torch.nan_to_num(tok_t, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1.0)
                        kl_per_token = (newlp_t - ref_t) / tok_safe
                        kl_per_token = torch.nan_to_num(kl_per_token, nan=0.0, posinf=0.0, neginf=0.0)
                        approx_kl = kl_per_token.mean()
                        metric_lg.update(approx_kl=approx_kl)
                        # 重要：这只是基于离线采样 token 的近似 KL。
                        # 受采样噪声和策略不匹配影响，它可能出现轻微负值；
                        # 负的“KL 惩罚”会反过来奖励策略发散，所以这里把惩罚项裁成非负。
                        kl_pen = kl_per_token.clamp_min(0.0)
                        pg_by_sample = pg_by_sample + beta * kl_pen
                    # `trace_ce` 模式下可以再给每条样本加一个辅助稳定项 `aux_sft`。
                    # 这里故意把它直接并入 strict PPO 目标，而不是额外跑一次完整 packed backward，
                    # 这样显存更省。
                    if aux_sft_by_sample is not None and float(aux) > 0:
                        pg_by_sample = pg_by_sample + float(aux) * aux_sft_by_sample.to(pg_by_sample.dtype)
            for sample_ind, item in enumerate(sequece_packing_scales):
                full_pt = full_pts_this_batch[sample_ind]
                for si, (pt, ph, pw) in enumerate(item):
                    mul_pt_ph_pw = pt * ph * pw
                    start, end = ptr, ptr+mul_pt_ph_pw
                    ptr = end
                    if x_BLC_mask is None:
                        loss_this_scale = loss[start:end].mean()
                        acc_this_scale = acc_bit[start:end].mean()
                    else:
                        pred_elem_num = x_BLC_mask[start:end].sum()
                        assert pred_elem_num > 0
                        loss_this_scale = loss[start:end].sum() / pred_elem_num
                        acc_this_scale = acc_bit[start:end].sum() / pred_elem_num
                    real_si = other_info_by_scale[global_scale_ind]['real_si']
                    volume_times = np.array(other_info_by_scale[global_scale_ind]['largest_scale']).prod() / mul_pt_ph_pw
                    acc_pt2scale_acc[full_pt][real_si].append(acc_this_scale)
                    acc_pt2scale_acc_counter[full_pt][real_si] += 1
                    if self.reweight_loss_by_scale == 0:
                        weight = 1 * mul_pt_ph_pw
                    else:
                        reweight_value = min(args.max_reweight_value, np.power(volume_times, 1/(1+self.reweight_loss_by_scale)))
                        weight = reweight_value * mul_pt_ph_pw
                    flatten_weight_list.append(weight)
                    flatten_L_list.append(loss_this_scale)
                    if use_grpo and weight_t is not None:
                        adv = weight_t[sample_ind]
                        if pg_by_sample is not None:
                            # Trace-replay 的逐样本 PPO 目标，广播到所有尺度用于加权。
                            # 这里不是每个 scale 重新算一遍 PPO，而是把该样本的策略梯度目标
                            # 放进 scale 聚合框架，保持和 SFT 的 loss 汇总/日志路径一致。
                            pg_obj = pg_by_sample[sample_ind]
                        elif oldlp_t is not None:
                            # 兼容旧逻辑：把逐尺度负 CE 当作 logp_new 估计。
                            logp_new = -loss_this_scale
                            delta = (logp_new - oldlp_t[sample_ind]).to(torch.float32).clamp(min=-60.0, max=60.0)
                            ratio = torch.exp(delta)
                            eps = float(getattr(args, "grpo_ratio_eps", 0.2))
                            ratio_clip = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
                            adv32 = adv.to(torch.float32) if isinstance(adv, torch.Tensor) else torch.tensor(float(adv), device=loss.device, dtype=torch.float32)
                            unclipped = ratio * adv32
                            clipped = ratio_clip * adv32
                            obj = torch.minimum(unclipped, clipped) if float(adv32.item()) >= 0 else torch.maximum(unclipped, clipped)
                            pg_obj = (-obj).to(loss.dtype)
                            if float(getattr(args, "grpo_kl_beta", 0.0)) > 0:
                                kl_proxy = (logp_new - oldlp_t[sample_ind]) ** 2
                                pg_obj = pg_obj + float(getattr(args, "grpo_kl_beta", 0.0)) * kl_proxy
                        else:
                            # 兜底逻辑：不做 ratio clipping 的 reward 加权 policy-gradient 近似。
                            pg_obj = adv * loss_this_scale
                        if pg_obj is not None:
                            flatten_pg_obj_list.append(pg_obj)
                    flatten_acc_bit_list.append(acc_this_scale)
                    flatten_sample_ind_list.append(sample_ind)
                    global_scale_ind += 1
            flatten_weight_list = torch.tensor(flatten_weight_list, dtype=loss.dtype, device=loss.device)
            flatten_weight_list = flatten_weight_list / flatten_weight_list.sum()
            sft_loss = (torch.stack(flatten_L_list) * flatten_weight_list).sum()
            if use_grpo and len(flatten_pg_obj_list) == len(flatten_L_list) and len(flatten_pg_obj_list) > 0:
                # 这里进入“最终 loss 选择”：
                # - 严格 GRPO：`pg_by_sample` 已经是逐样本 PPO loss，最终只反传它；
                # - 兼容/兜底 GRPO：没有严格 new/old logprob 时，用 reward 加权 CE 近似；
                # - 普通 SFT：完全不走这个分支，只反传 `sft_loss`。
                rl_loss = (torch.stack(flatten_pg_obj_list) * flatten_weight_list).sum()
                # 可选混合缩放：当 GRPO 与更强的 SFT anchor 混训时，压低 RL 更新幅度。
                rl_coef = float(getattr(args, "grpo_hybrid_rl_coef", 1.0) or 1.0)
                if rl_coef != 1.0:
                    rl_loss = rl_loss * rl_coef
                aux = float(getattr(args, "grpo_aux_sft_coef", 0.0) or 0.0)
                # 若严格 PPO 目标存在（pg_by_sample != None），辅助项已经在 trace_ce 分支里合入，
                # 不能再通过 sft_loss 强制触发完整 packed backward。
                if pg_by_sample is not None:
                    # 严格 GRPO：最终 loss 就是 PPO/GRPO 目标。
                    # 此时 SFT CE forward 主要用于 metric/aux，不再额外反传完整 sft_loss。
                    final_loss = rl_loss
                else:
                    # 兼容旧逻辑：没有严格 replay 目标时，可以把 aux SFT CE 混进 RL loss，
                    # 用来防止模型在 reward 信号下快速偏离基础视频建模能力。
                    final_loss = rl_loss + aux * sft_loss
            else:
                # 普通 SFT 路径：没有 GRPO 权重，最终 loss 就是 teacher-forcing CE。
                final_loss = sft_loss
            final_acc_bit = (torch.stack(flatten_acc_bit_list) * flatten_weight_list).sum()

        # 反向阶段：统一由 AmpOptimizer 负责 mixed precision / grad clip / step。
        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(ep=epoch, it=it, g_it=g_it, stepping=stepping, loss=final_loss, clip_decay_ratio=clip_decay_ratio)

        # EMA 更新公式仍是 `ema = decay * ema + (1 - decay) * param`。
        if args.use_fsdp_model_ema and (args.model_ema_decay < 1):
            update_ema(self.gpt_ema, self.gpt)

        # 只有真的发生 optimizer.step 时才清空梯度。
        if stepping:
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)

        # 可选导出：把当前 step 的 latent clip 解码并做 TSformer 轨迹分析。
        self._tf_dump_step_trajectory(raw_features_list=raw_features_list, g_it=g_it, args=args)

        # 指标聚合：先按尺度整理，再跨卡 all_reduce。
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
            def sum_dict(acc_pt2scale_acc):
                """把按尺度收集的 tensor 列表压成逐尺度总和。"""
                for full_pt in acc_pt2scale_acc:
                    for si in range(len(acc_pt2scale_acc[full_pt])):
                        acc_pt2scale_acc[full_pt][si] = torch.tensor(acc_pt2scale_acc[full_pt][si]).sum()
                return acc_pt2scale_acc

            def dict2list(acc_pt2scale_acc):
                """把嵌套字典拍平成列表，方便组成一个 all_reduce 张量。"""
                flatten_acc_pt2scale_acc = []
                for key, val in acc_pt2scale_acc.items():
                    flatten_acc_pt2scale_acc.extend(val)
                return flatten_acc_pt2scale_acc

            def list2dict(acc_pt2scale_acc, flatten_acc_pt2scale_acc):
                """把 all_reduce 后的扁平列表按原结构写回字典。"""
                ptr = 0
                for key in acc_pt2scale_acc:
                    for ind in range(len(acc_pt2scale_acc[key])):
                        acc_pt2scale_acc[key][ind] = flatten_acc_pt2scale_acc[ptr]
                        ptr += 1
                return acc_pt2scale_acc

            acc_pt2scale_acc = sum_dict(acc_pt2scale_acc)
            flatten_acc_pt2scale_acc = dict2list(acc_pt2scale_acc)
            flatten_acc_pt2scale_acc_counter = dict2list(acc_pt2scale_acc_counter)

            train_loss = final_loss.item()
            train_acc = final_acc_bit.item()
            grad_norm_scalar = 0.0
            if grad_norm_t is not None:
                grad_norm_scalar = float(torch.nan_to_num(grad_norm_t.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).item())
            metrics = torch.tensor(flatten_acc_pt2scale_acc + flatten_acc_pt2scale_acc_counter + [grad_norm_scalar, train_loss, train_acc, is_image_batch, valid_sequence_ratio], device=loss.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            flatten_acc_pt2scale_acc, flatten_acc_pt2scale_acc_counter = metrics[:len(flatten_acc_pt2scale_acc)], metrics[len(flatten_acc_pt2scale_acc):2*len(flatten_acc_pt2scale_acc)]
            flatten_acc_pt2scale_acc = flatten_acc_pt2scale_acc / (flatten_acc_pt2scale_acc_counter + 1e-16)
            acc_pt2scale_acc = list2dict(acc_pt2scale_acc, flatten_acc_pt2scale_acc)
            acc_pt2scale_acc_counter = list2dict(acc_pt2scale_acc_counter, flatten_acc_pt2scale_acc_counter)
            grad_norm_t, train_loss, train_acc, is_image_batch, valid_sequence_ratio = metrics[2*len(flatten_acc_pt2scale_acc):] / (dist.get_world_size() + 1e-16)
            if args.num_of_label_value == 1:
                key, base = 'Loss', 1
            else:
                key, base = 'Acc', 100
            rew_mean = None
            reward_act_mean = None
            reward_task_mean = None
            reward_task_raw_mean = None
            reward_task_dense_raw_mean = None
            reward_task_success_raw_mean = None
            succ_hit_clip_ratio = None
            succ_hit_traj_ratio = None
            task_cost_mean = None
            task_pos_err_mean = None
            task_yaw_err_mean = None
            reward_ce_mean = None
            adv_mean = None
            pos_ratio = None
            neff_count = None
            success_bonus_hit_ratio = None
            log_weight_stats = bool(int(getattr(args, "grpo_log_weight_stats", 1)))
            weight_zero_ratio = None
            weight_neg_ratio = None
            weight_mean = None
            weight_min = None
            weight_max = None
            success_negative_ratio = None
            if use_grpo and weight_t is not None:
                # GRPO 日志统计只做观测，不参与梯度。
                # 这些指标回答的是“reward/advantage 数据质量怎么样”：
                # - rew/r_act/r_task：这批 rollout 的奖励均值；
                # - adv/weight：最终用于训练的鼓励强度；
                # - pos/w_zero/w_neg/N_eff：有效样本有多少，避免一批里大多数权重为 0；
                # - succ/task_cost/pos_err/yaw_err：导航任务层面的诊断。
                def _sum_and_count(tensor):
                    """把一个指标张量安全地转换成“总和 + 样本数”。"""
                    if tensor is None:
                        return (
                            torch.tensor(0.0, dtype=torch.float32, device=loss.device),
                            torch.tensor(0.0, dtype=torch.float32, device=loss.device),
                        )
                    tensor = torch.nan_to_num(tensor.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    return tensor.sum(), torch.tensor(float(tensor.numel()), dtype=torch.float32, device=loss.device)

                rew_sum, rew_cnt = _sum_and_count(reward_t)
                reward_act_sum, reward_act_cnt = _sum_and_count(reward_act_t)
                reward_task_sum, reward_task_cnt = _sum_and_count(reward_task_t)
                reward_task_raw_sum, reward_task_raw_cnt = _sum_and_count(reward_task_raw_t)
                reward_task_dense_raw_sum, reward_task_dense_raw_cnt = _sum_and_count(reward_task_dense_raw_t)
                reward_task_success_raw_sum, reward_task_success_raw_cnt = _sum_and_count(reward_task_success_raw_t)
                task_cost_sum, task_cost_cnt = _sum_and_count(task_cost_t)
                task_pos_err_sum, task_pos_err_cnt = _sum_and_count(task_pos_err_t)
                task_yaw_err_sum, task_yaw_err_cnt = _sum_and_count(task_yaw_err_t)
                reward_ce_sum, reward_ce_cnt = _sum_and_count(reward_ce_t)
                adv_sum, adv_cnt = _sum_and_count(weight_t)
                finite_mask = torch.isfinite(weight_t)
                pos_sum = torch.sum((finite_mask & (weight_t > 0)).to(torch.float32))
                zero_sum = torch.sum((finite_mask & (weight_t == 0)).to(torch.float32))
                neg_sum = torch.sum((finite_mask & (weight_t < 0)).to(torch.float32))
                finite_cnt = torch.sum(finite_mask.to(torch.float32))
                # `weight_t` 是 GRPO 真正乘到 policy objective 上的 advantage。
                # 如果 pos_ratio/N_eff 很低，说明这一批可学习的“好轨迹”太少，RL 更新会很弱。
                finite_weight_t = torch.masked_select(weight_t.detach().to(torch.float32), finite_mask)
                weight_min_t = torch.tensor(float("inf"), dtype=torch.float32, device=loss.device)
                weight_max_t = torch.tensor(float("-inf"), dtype=torch.float32, device=loss.device)
                if finite_weight_t.numel() > 0:
                    weight_min_t = torch.min(finite_weight_t)
                    weight_max_t = torch.max(finite_weight_t)
                success_hit_sum = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                success_hit_cnt = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                success_neg_sum = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                succ_clip_hit_sum = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                succ_clip_hit_cnt = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                succ_traj_hit_sum = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                succ_traj_hit_cnt = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                if reward_task_success_raw_t is not None:
                    success_mask = reward_task_success_raw_t > 0
                    success_hit_sum = torch.sum(success_mask.to(torch.float32))
                    success_hit_cnt = torch.tensor(float(reward_task_success_raw_t.numel()), dtype=torch.float32, device=loss.device)
                    success_neg_sum = torch.sum((success_mask & (weight_t < 0)).to(torch.float32))
                if succ_t is not None:
                    succ_clip_hit_sum = torch.sum((succ_t > 0).to(torch.float32))
                    succ_clip_hit_cnt = torch.tensor(float(succ_t.numel()), dtype=torch.float32, device=loss.device)
                if succ_traj_t is not None:
                    succ_traj_hit_sum = torch.sum((succ_traj_t > 0).to(torch.float32))
                    succ_traj_hit_cnt = torch.tensor(float(succ_traj_t.numel()), dtype=torch.float32, device=loss.device)
                reward_stats = torch.stack(
                    [
                        rew_sum, rew_cnt,
                        reward_act_sum, reward_act_cnt,
                        reward_task_sum, reward_task_cnt,
                        reward_task_raw_sum, reward_task_raw_cnt,
                        reward_task_dense_raw_sum, reward_task_dense_raw_cnt,
                        reward_task_success_raw_sum, reward_task_success_raw_cnt,
                        task_cost_sum, task_cost_cnt,
                        task_pos_err_sum, task_pos_err_cnt,
                        task_yaw_err_sum, task_yaw_err_cnt,
                        reward_ce_sum, reward_ce_cnt,
                        adv_sum, adv_cnt,
                        pos_sum, zero_sum, neg_sum, finite_cnt,
                        success_hit_sum, success_hit_cnt,
                        success_neg_sum,
                        succ_clip_hit_sum, succ_clip_hit_cnt,
                        succ_traj_hit_sum, succ_traj_hit_cnt,
                    ]
                )
                # 多卡训练时每张卡只看到局部 batch；这里把 reward/advantage 统计合并成全局均值。
                # 注意这只是日志聚合，不会改变前面已经计算好的本地 loss。
                tdist.all_reduce(reward_stats, op=tdist.ReduceOp.SUM)
                tdist.all_reduce(weight_min_t, op=tdist.ReduceOp.MIN)
                tdist.all_reduce(weight_max_t, op=tdist.ReduceOp.MAX)

                def _safe_mean(sum_idx, cnt_idx):
                    """从 `reward_stats` 中安全读取均值；计数为 0 时返回 None。"""
                    denom = reward_stats[cnt_idx].item()
                    if denom <= 0:
                        return None
                    return float((reward_stats[sum_idx] / reward_stats[cnt_idx].clamp_min(1.0)).item())

                rew_mean = _safe_mean(0, 1)
                reward_act_mean = _safe_mean(2, 3)
                reward_task_mean = _safe_mean(4, 5)
                reward_task_raw_mean = _safe_mean(6, 7)
                reward_task_dense_raw_mean = _safe_mean(8, 9)
                reward_task_success_raw_mean = _safe_mean(10, 11)
                task_cost_mean = _safe_mean(12, 13)
                task_pos_err_mean = _safe_mean(14, 15)
                task_yaw_err_mean = _safe_mean(16, 17)
                reward_ce_mean = _safe_mean(18, 19)
                adv_mean = _safe_mean(20, 21)
                if reward_stats[25].item() > 0:
                    pos_ratio = float((reward_stats[22] / reward_stats[25].clamp_min(1.0) * 100.0).item())
                    neff_count = float(reward_stats[22].item())
                    if log_weight_stats:
                        weight_zero_ratio = float((reward_stats[23] / reward_stats[25].clamp_min(1.0) * 100.0).item())
                        weight_neg_ratio = float((reward_stats[24] / reward_stats[25].clamp_min(1.0) * 100.0).item())
                        weight_mean = adv_mean
                        weight_min = float(weight_min_t.item()) if torch.isfinite(weight_min_t) else 0.0
                        weight_max = float(weight_max_t.item()) if torch.isfinite(weight_max_t) else 0.0
                if reward_stats[27].item() > 0:
                    success_bonus_hit_ratio = float((reward_stats[26] / reward_stats[27].clamp_min(1.0) * 100.0).item())
                    if log_weight_stats and reward_stats[26].item() > 0:
                        success_negative_ratio = float((reward_stats[28] / reward_stats[26].clamp_min(1.0) * 100.0).item())
                if reward_stats[30].item() > 0:
                    succ_hit_clip_ratio = float((reward_stats[29] / reward_stats[30].clamp_min(1.0) * 100.0).item())
                if reward_stats[32].item() > 0:
                    succ_hit_traj_ratio = float((reward_stats[31] / reward_stats[32].clamp_min(1.0) * 100.0).item())

            stable_metric_inputs = {
                'rew': rew_mean,
                'r_act': reward_act_mean,
                'r_task': reward_task_mean,
                'task_cost': task_cost_mean,
                'pos_err_m': task_pos_err_mean,
                'yaw_err_deg': task_yaw_err_mean,
                'adv': adv_mean,
            }
            # GRPO reward 抖动通常比 SFT CE 大得多，所以额外维护 EMA/optimizer-step 级平滑值。
            # 看训练曲线时优先看 *_ema / *_optstep_ema，少被单个 rollout batch 的噪声误导。
            self._update_stable_metric_trackers(stable_metric_inputs, stepping=stepping)
            stable_metrics = self._collect_stable_metrics()

            # MetricLogger 是控制台/本地日志；wandb_log_dict 是远端可视化。
            # SFT 路径主要看 L/Acc/seq_usage；GRPO 路径还要同时看 reward、advantage 和 success 指标。
            metric_lg.update(
                L=train_loss,
                Acc=train_acc*base,
                L_i=0.,
                Acc_i=0.,
                L_v=0.,
                Acc_v=0.,
                tnm=grad_norm_t,
                seq_usage=valid_sequence_ratio*100.,
                rew=rew_mean,
                r_act=reward_act_mean,
                r_task=reward_task_mean,
                r_task_raw=reward_task_raw_mean,
                r_task_dense_raw=reward_task_dense_raw_mean,
                r_task_success_raw=reward_task_success_raw_mean,
                succ_hit=success_bonus_hit_ratio,
                succ_hit_clip=succ_hit_clip_ratio,
                succ_hit_traj_diag=succ_hit_traj_ratio,
                task_cost=task_cost_mean,
                pos_err_m=task_pos_err_mean,
                yaw_err_deg=task_yaw_err_mean,
                r_ce=reward_ce_mean,
                adv=adv_mean,
                pos=pos_ratio,
                w_zero=weight_zero_ratio if log_weight_stats else None,
                w_neg=weight_neg_ratio if log_weight_stats else None,
                w_mean=weight_mean if log_weight_stats else None,
                w_min=weight_min if log_weight_stats else None,
                w_max=weight_max if log_weight_stats else None,
                succ_neg=success_negative_ratio if log_weight_stats else None,
                neff=neff_count,
            )    # 中文说明：todo: Accm, Acct
            if stable_metrics:
                metric_lg.update(**stable_metrics)
            wandb_log_dict = {
                'Overall/train_loss': train_loss,
                'Overall/train_acc': train_acc*base,
                'Overall/grad_norm_t': grad_norm_t,
                'Overall/video_batch_ratio': (1-is_image_batch)*100.,
                'Overall/valid_sequence_ratio': valid_sequence_ratio*100.,
            }
            if reward_task_raw_mean is not None:
                wandb_log_dict['GRPO/reward_task_raw_mean'] = reward_task_raw_mean
            if reward_task_dense_raw_mean is not None:
                wandb_log_dict['GRPO/reward_task_dense_raw_mean'] = reward_task_dense_raw_mean
            if reward_task_success_raw_mean is not None:
                wandb_log_dict['GRPO/reward_task_success_raw_mean'] = reward_task_success_raw_mean
            if success_bonus_hit_ratio is not None:
                wandb_log_dict['GRPO/success_bonus_hit_ratio'] = success_bonus_hit_ratio
            if log_weight_stats and weight_zero_ratio is not None:
                wandb_log_dict['GRPO/weight_zero_ratio'] = weight_zero_ratio
            if log_weight_stats and weight_neg_ratio is not None:
                wandb_log_dict['GRPO/weight_neg_ratio'] = weight_neg_ratio
            if log_weight_stats and weight_mean is not None:
                wandb_log_dict['GRPO/weight_mean'] = weight_mean
            if log_weight_stats and weight_min is not None:
                wandb_log_dict['GRPO/weight_min'] = weight_min
            if log_weight_stats and weight_max is not None:
                wandb_log_dict['GRPO/weight_max'] = weight_max
            if log_weight_stats and success_negative_ratio is not None:
                wandb_log_dict['GRPO/success_negative_ratio'] = success_negative_ratio
            if succ_hit_clip_ratio is not None:
                wandb_log_dict['GRPO/success_clip_hit_ratio'] = succ_hit_clip_ratio
            if succ_hit_traj_ratio is not None:
                wandb_log_dict['GRPO/success_traj_hit_ratio'] = succ_hit_traj_ratio
            if task_cost_mean is not None:
                wandb_log_dict['GRPO/task_cost_mean'] = task_cost_mean
            if task_pos_err_mean is not None:
                wandb_log_dict['GRPO/task_final_pos_err_mean_m'] = task_pos_err_mean
            if task_yaw_err_mean is not None:
                wandb_log_dict['GRPO/task_final_yaw_err_mean_deg'] = task_yaw_err_mean
            if use_grpo and ('weight_t' in locals()) and (weight_t is not None):
                with torch.no_grad():
                    # N_eff 这里定义为正 advantage 样本数。
                    # 它不是严格统计学 effective sample size，而是一个直观诊断：
                    # 当前 batch 里到底有多少条轨迹会推动模型“更容易采到它”。
                    neff = torch.sum(torch.isfinite(weight_t) & (weight_t > 0)).float()
                    wandb_log_dict['GRPO/N_eff'] = float(neff.item())
            for stable_key, stable_value in stable_metrics.items():
                wandb_log_dict[f'GRPOStable/{stable_key}'] = stable_value
            for full_pt in acc_pt2scale_acc:
                for si in range(len(acc_pt2scale_acc[full_pt])):
                    if acc_pt2scale_acc_counter[full_pt][si] > 0:
                        duration = (full_pt-1) / args.temporal_compress_rate
                        wandb_log_dict[f'Details/{key}/t{duration:04.1f}s/s{si+1:03d}'] = acc_pt2scale_acc[full_pt][si].item() * base
                        wandb_log_dict[f'Details/Num/t{duration:04.1f}s/s{si+1:03d}'] = acc_pt2scale_acc_counter[full_pt][si]
            wandb_utils.log(wandb_log_dict, step=g_it)
        return grad_norm_t, scale_log2_t

    def __repr__(self):
        """返回训练器配置与结构摘要，便于日志打印。"""
        return (
            f'\n'
            f'[VGPTTr.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[VGPTTr.structure]: {super(InfinityTrainer, self).__repr__().replace(InfinityTrainer.__name__, "")}'
        )

    def ema_load(self):
        """把在线模型参数临时替换成 EMA 参数，常用于验证前切换。"""
        self.cached_state_not_ema = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            self.gpt_opt.paras[pi].data.copy_(p_ema)
        for pi, para in enumerate(self.gpt_opt.paras):
            dist.broadcast(para, src_rank=pi % dist.get_world_size())

    def ema_recover(self):
        """从缓存中恢复非 EMA 的原始训练参数。"""
        self.gpt_wo_ddp.load_state_dict(self.cached_state_not_ema)
        del self.cached_state_not_ema
        self.cached_state_not_ema = None

    def get_config(self):
        """导出需要随 checkpoint 一起保存的轻量训练状态。"""
        return {
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }

    def state_dict(self):
        """
        打包训练器状态，包括配置、VAE、模型和优化器。

        不管当前是在 SFT 还是 GRPO 训练，checkpoint 保存的核心都是同一个 `gpt` 世界模型。
        GRPO 没有在这里额外保存一个“强化学习模型”；它只是用 PPO/GRPO loss 改写世界模型参数。
        """
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config(), 'vae_local': m.state_dict()}

        if self.zero:   # 中文说明：待修复；zero 路径需要单独处理 state_dict 加载细节。
            # FSDP/ZeRO 下参数被切分在多张卡上，保存前需要进入 FULL_STATE_DICT 上下文，
            # 由 PyTorch 负责把分片参数/优化器状态还原成可落盘的完整 checkpoint。
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                state['gpt_fsdp'] = self.gpt.state_dict()
                if self.use_fsdp_model_ema:
                    state['gpt_ema_fsdp'] = self.gpt_ema.state_dict()
                state['gpt_fsdp_opt'] = FSDP.optim_state_dict(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=self.gpt_opt.optimizer.state_dict())
            if self.gpt_opt.scaler is not None:
                state['gpt_opt_scaler'] = self.gpt_opt.scaler.state_dict()

        else:

            for k in ('gpt_wo_ddp', 'gpt_opt'):
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True, skip_vae=False):
        """
        从 checkpoint 恢复模型、优化器和训练进度状态。

        这也是理解 SFT/GRPO 关系的地方：
        - 加载 SFT checkpoint 后继续训练，可以走普通 SFT；
        - 加载同一个 SFT checkpoint 后切到 `trainer_type=grpo`，就是在 SFT 底座上做 RL 微调；
        - 推理时只需要加载训练好的世界模型权重，不需要运行本 trainer。
        """
        if self.zero:
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                gpt_state = state['gpt_fsdp']
                # FSDP 恢复也必须遵守 `strict`。
                # 旧 checkpoint 可能含有来自略有差异的 head 的 key（例如 semantic_head2.*），
                # 这里统一交给 PyTorch 的 strict/missing/unexpected 机制处理。
                ret = self.gpt.load_state_dict(gpt_state, strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VGPTTr.load_state_dict][zero] gpt 缺失 keys:  {missing}')
                    print(f'[VGPTTr.load_state_dict][zero] gpt 未预期 keys:  {unexpected}')
                if self.use_fsdp_model_ema:
                    ema_state = state.get('gpt_ema_fsdp', None)
                    if ema_state is not None:
                        ret_ema = self.gpt_ema.load_state_dict(ema_state, strict=strict)
                        if ret_ema is not None:
                            missing, unexpected = ret_ema
                            print(f'[VGPTTr.load_state_dict][zero] gpt_ema 缺失 keys:  {missing}')
                            print(f'[VGPTTr.load_state_dict][zero] gpt_ema 未预期 keys:  {unexpected}')
                one_group_opt_state = state.get('gpt_fsdp_opt', None)
                """
                AdamW state['gpt_fsdp_opt']:
                {
                    'state': { <para_name>: {'exp_avg': <unsharded_tensor>, 'exp_avg_sq': <unsharded_tensor>, 'step': <int>} },
                    'param_groups': [
                        {
                            'wd_sc': 1.0, 'lr_sc': 1.0, 'lr': xxx, 'betas': (0.9, 0.97), 'eps': 1e-08, 'weight_decay': 0.02,
                            'amsgrad': False, 'foreach': None, 'maximize': False, 'capturable': False, 'differentiable': False, 'fused': True,
                            'params': [<para_name> x m]
                        } x n
                    ]
                }
                one_group_opt_state['param_groups'] = self.gpt_opt.optimizer.state_dict()['param_groups']
                """
                if one_group_opt_state is not None:
                    try:
                        optim_state_dict = FSDP.optim_state_dict_to_load(
                            model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=one_group_opt_state
                        )
                        self.gpt_opt.optimizer.load_state_dict(optim_state_dict)
                    except Exception as e:
                        if strict:
                            raise
                        print(f'[VGPTTr.load_state_dict][zero] optimizer state 不匹配，已跳过: {e}')

            if self.gpt_opt.scaler is not None:
                try: self.gpt_opt.scaler.load_state_dict(state['gpt_opt_scaler'])
                except Exception as e: print(f'[fp16 load_state_dict 错误] {e}')
        else:
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                if skip_vae and 'vae' in k: continue
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    ret = m.load_state_dict(state[k], strict=strict)
                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[VGPTTr.load_state_dict] {k} 缺失 keys:  {missing}')
                        print(f'[VGPTTr.load_state_dict] {k} 未预期 keys:  {unexpected}')

        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VGPT.load_state_dict] config 不匹配:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
