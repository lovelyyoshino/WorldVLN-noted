#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
action-aware GRPO 在线推理 API 服务端，采用权重常驻的流式架构。

目标：
- 启动时只加载一次 world model 和 action head，并常驻内存/GPU。
- 客户端按轨迹（`session_id`）流式上传 RGB 帧：
  - 首次请求通常上传 1 帧预热图像，并可附带 instruction/prompt。
  - 后续请求通常每次上传 `step` 帧，直到累计到 `num_frames`。
- 服务端先运行 world model 生成 summed_codes（latents），再预测增量动作。
  为兼容旧流程，这个入口仍保留 TSformer(P2P, window_size=2) 路径。
- 输出增量动作使用 cm/deg，顺序为：[dx, dy, dz, droll, dyaw, dpitch]。

示例：
  环境变量示例：export INFINITY_CKPT=/path/to/global_step_xxx.pth
  说明：uvicorn grpo_server:app --host 0.0.0.0 --port 8002

自测需要真实 checkpoint 和 route_dir：
  说明：python3 grpo_server.py --self_test     --infinity_ckpt "$INFINITY_CKPT"     --route_dir /path/to/route_dir
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# 可选依赖：只有 actionhead reference-video 模式才需要。
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

# 可选服务端依赖：未安装 fastapi/pydantic 时仍允许运行离线评估。
FASTAPI_AVAILABLE = True
try:
    from fastapi import FastAPI, HTTPException  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
except Exception:
    FASTAPI_AVAILABLE = False

    class HTTPException(RuntimeError):  # 最小桩实现。
        """FastAPI 不可用时的最小异常桩，方便离线脚本复用同一套错误路径。"""

        def __init__(self, status_code: int = 500, detail: str = ""):
            """记录 HTTP 状态码和错误详情，行为上尽量贴近 FastAPI.HTTPException。"""
            super().__init__(f"HTTP {status_code}: {detail}")
            self.status_code = status_code
            self.detail = detail

    class BaseModel:  # 最小桩实现。
        """Pydantic 不可用时的占位基类，仅用于让离线路径完成导入。"""

        pass

    def Field(default=None, **kwargs):  # noqa: N802
        """Pydantic Field 的轻量占位实现：离线模式只需要默认值。"""
        return default


# -------------------------
# 0) 路径 / sys.path
# -------------------------
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

TSFORMER_ROOT = REPO / "Worldmodel" / "action_decoder" / "actionhead_runtime"

if not TSFORMER_ROOT.exists():
    raise FileNotFoundError(f"找不到 TSformer repo：{TSFORMER_ROOT}")

# TSformer 模块
sys.path.insert(0, str(TSFORMER_ROOT))

# -------------------------
# 1) InfinityStar 动态导入（支持 INFINITY_REPO_ROOT）
# -------------------------
# 注意：Worldmodel repo 在运行时选择，因此同一服务可以嵌入不同副本。
DEFAULT_INFINITY_REPO_ROOT = REPO / "Worldmodel" / "runtime"

# 由 _import_infinity_modules() 填充。
InfinityStreamingSession = None  # type: ignore
SelfCorrection = None  # type: ignore
get_dynamic_resolution_meta = None  # type: ignore
_make_infinity_args = None  # type: ignore
load_tokenizer = None  # type: ignore
load_transformer = None  # type: ignore
load_visual_tokenizer = None  # type: ignore
infinity_transform = None  # type: ignore
infinity_save_video = None  # type: ignore
infinity_gen_one_example = None  # type: ignore


def _get_infinity_repo_root() -> Path:
    """解析 GRPO 服务使用的 InfinityStar runtime 根目录，优先读 INFINITY_REPO_ROOT。"""
    p = os.environ.get("INFINITY_REPO_ROOT", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return DEFAULT_INFINITY_REPO_ROOT


def _import_infinity_modules(repo_root: Path) -> None:
    """
    从 `repo_root` 动态导入 InfinityStar Python 模块。
    使用 Infinity 相关符号前必须先调用本函数。
    """
    global InfinityStreamingSession, SelfCorrection, get_dynamic_resolution_meta
    global _make_infinity_args, load_tokenizer, load_transformer, load_visual_tokenizer, infinity_transform, infinity_save_video, infinity_gen_one_example

    if InfinityStreamingSession is not None:
        return
    if not repo_root.exists():
        raise FileNotFoundError(f"找不到 InfinityStar repo：{repo_root}")
    # 把选中的 repo 放到最高导入优先级。
    sys.path.insert(0, str(repo_root))

    from tools.closed_loop_streaming_infer_480p_81f import _make_args as __make_args  # type: ignore
    from tools.infinity_streaming_session import InfinityStreamingSession as __ISS  # type: ignore
    from tools.run_infinity import (  # type: ignore
        load_tokenizer as __load_tokenizer,
        load_transformer as __load_transformer,
        load_visual_tokenizer as __load_visual_tokenizer,
        gen_one_example as __gen_one_example,
        save_video as __save_video,
        transform as __transform,
    )
    from infinity.models.self_correction import SelfCorrection as __SelfCorrection  # type: ignore
    from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta as __get_dyn  # type: ignore

    _make_infinity_args = __make_args
    InfinityStreamingSession = __ISS
    load_tokenizer = __load_tokenizer
    load_transformer = __load_transformer
    load_visual_tokenizer = __load_visual_tokenizer
    infinity_gen_one_example = __gen_one_example
    infinity_save_video = __save_video
    infinity_transform = __transform
    SelfCorrection = __SelfCorrection
    get_dynamic_resolution_meta = __get_dyn


# -------------------------
# 2) 仅 Action head 模式
# -------------------------


# -------------------------
# 3) 配置（环境变量默认值）
# -------------------------
DEFAULT_TS_CKPT = str(TSFORMER_ROOT / "adapter_p2p" / "new_stage2_resume70_to100_bs256" / "p2p_epoch_100.pth")
DEFAULT_TS_STATS = str(TSFORMER_ROOT / "adapter_p2p" / "uav-flow_p2p" / "p2p_target_stats.json")

DEFAULT_NUM_FRAMES = int(os.environ.get("INFINITY_NUM_FRAMES", "81"))
DEFAULT_STEP = int(os.environ.get("INFINITY_STEP", "16"))
DEFAULT_FPS = int(os.environ.get("INFINITY_FPS", "16"))
DEFAULT_PN = os.environ.get("INFINITY_PN", "0.40M")
DEFAULT_H_DIV_W = float(os.environ.get("INFINITY_H_DIV_W_TEMPLATE", "0.562"))

DEFAULT_DYNAMIC_SCHEDULE = os.environ.get("INFINITY_DYNAMIC_SCALE_SCHEDULE", "infinity_elegant_clip20frames_v2_allpt")
DEFAULT_MASK_TYPE = os.environ.get("INFINITY_MASK_TYPE", "infinity_elegant_clip20frames_v2_allpt")
DEFAULT_CFG = float(os.environ.get("INFINITY_CFG", "34.0"))
DEFAULT_TAU_IMAGE = float(os.environ.get("INFINITY_TAU_IMAGE", "1.0"))
DEFAULT_TAU_VIDEO = float(os.environ.get("INFINITY_TAU_VIDEO", "0.4"))
DEFAULT_TOP_K = int(os.environ.get("INFINITY_TOP_K", "900"))
DEFAULT_TOP_P = float(os.environ.get("INFINITY_TOP_P", "0.97"))
DEFAULT_GT_LEAK_FIRST = int(os.environ.get("INFINITY_GT_LEAK_FIRST", "14"))

# 默认 config 文件位置，可用 INFINITY_SERVER_CONFIG 覆盖。
DEFAULT_SERVER_CONFIG_JSON = str((ROOT / "config.json").resolve())


def _obs_points(pred_num_frames: int, step: int) -> List[int]:
    """
    生成闭环 segment 边界。

    初学者公式：`points = [1, 1+step, 1+2*step, ..., num_frames]`。
    例如 49 帧、step=16 对应 `[1,17,33,49]`。
    服务端默认就绪条件是 `ready_default = n >= points[seg+1]`：
    已收到帧数 `n` 到达当前 segment 的右边界，才发射该段动作。
    """
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


@dataclass
class InfinityConfig:
    """GRPO rollout 世界模型配置：checkpoint、帧数、采样参数和 cache 策略。"""

    ckpt: str = ""
    num_frames: int = DEFAULT_NUM_FRAMES
    step: int = DEFAULT_STEP
    fps: int = DEFAULT_FPS
    pn: str = DEFAULT_PN
    h_div_w_template: float = DEFAULT_H_DIV_W
    dynamic_scale_schedule: str = DEFAULT_DYNAMIC_SCHEDULE
    mask_type: str = DEFAULT_MASK_TYPE
    cfg: float = DEFAULT_CFG
    tau_image: float = DEFAULT_TAU_IMAGE
    tau_video: float = DEFAULT_TAU_VIDEO
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    gt_leak_first: int = DEFAULT_GT_LEAK_FIRST

    # 闭环 / rolling-tail 控制项，与 batch_closed_loop_streaming_infer_routes.py 对齐。
    rolling_tail_infer: bool = False
    rolling_infer_mode: str = "stable_full"  # 可选窗口策略：stable_full 或 tail_window。
    tail_window_frames: int = 33
    tail_window_start_step: int = 1
    v2v_history_injection: str = "gt_obs"  # 可选观测来源模式：gt_obs、official_leak 或 hybrid_leak_gtobs。
    late_v2v_history_injection: Optional[str] = None
    late_step_start: int = 2
    late_top_k: int = 300
    late_top_p: float = 0.90
    lock_seed_across_steps: bool = False

    def points(self) -> List[int]:
        """返回 RGB 帧时间线上的闭环 segment 边界。"""
        return _obs_points(pred_num_frames=int(self.num_frames), step=int(self.step))

    def pt_total(self) -> int:
        """返回整段视频对应的 latent 时间长度，默认 temporal_compress_rate=4。"""
        # 帧号 f 映射到 latent 下标：latent_index(f) = (f - 1)//temporal_compress_rate + 1。
        # 本 repo 通常 temporal_compress_rate=4，所以总帧数 num_frames 对应下面这个 pt。
        return (int(self.num_frames) - 1) // 4 + 1


@dataclass
class TSformerConfig:
    """旧 P2P TSformer 动作头配置，保留给兼容路径。"""

    ckpt: str = DEFAULT_TS_CKPT
    stats: str = DEFAULT_TS_STATS


@dataclass
class ServerConfig:
    """GRPO 推理服务总配置：world model、动作头和 runtime 根目录。"""

    infinity: InfinityConfig = field(default_factory=InfinityConfig)
    tsformer: TSformerConfig = field(default_factory=TSformerConfig)
    infinity_repo_root: Path = field(default_factory=_get_infinity_repo_root)


_SRV_CFG: Optional[ServerConfig] = None


def _load_server_config_from_json(path: str) -> ServerConfig:
    """从 config.json 读取 GRPO server 配置，兼容顶层即 infinity 配置的格式。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    inf = raw.get("infinity", raw) if isinstance(raw, dict) else {}
    ts = raw.get("tsformer", {}) if isinstance(raw, dict) else {}

    inf_cfg = InfinityConfig(
        ckpt=str(inf.get("ckpt") or inf.get("checkpoint") or "").strip(),
        num_frames=int(inf.get("num_frames", DEFAULT_NUM_FRAMES)),
        step=int(inf.get("step", DEFAULT_STEP)),
        fps=int(inf.get("fps", DEFAULT_FPS)),
        pn=str(inf.get("pn", DEFAULT_PN)),
        h_div_w_template=float(inf.get("h_div_w_template", DEFAULT_H_DIV_W)),
        dynamic_scale_schedule=str(inf.get("dynamic_scale_schedule", DEFAULT_DYNAMIC_SCHEDULE)),
        mask_type=str(inf.get("mask_type", DEFAULT_MASK_TYPE)),
        cfg=float(inf.get("cfg", DEFAULT_CFG)),
        tau_image=float(inf.get("tau_image", DEFAULT_TAU_IMAGE)),
        tau_video=float(inf.get("tau_video", DEFAULT_TAU_VIDEO)),
        top_k=int(inf.get("top_k", DEFAULT_TOP_K)),
        top_p=float(inf.get("top_p", DEFAULT_TOP_P)),
        gt_leak_first=int(inf.get("gt_leak_first", DEFAULT_GT_LEAK_FIRST)),
        rolling_tail_infer=bool(inf.get("rolling_tail_infer", False)),
        rolling_infer_mode=str(inf.get("rolling_infer_mode", "stable_full")),
        tail_window_frames=int(inf.get("tail_window_frames", 33)),
        tail_window_start_step=int(inf.get("tail_window_start_step", 1)),
        v2v_history_injection=str(inf.get("v2v_history_injection", "gt_obs")),
        late_v2v_history_injection=(str(inf.get("late_v2v_history_injection")).strip() if inf.get("late_v2v_history_injection") is not None else None),
        late_step_start=int(inf.get("late_step_start", 2)),
        late_top_k=int(inf.get("late_top_k", 300)),
        late_top_p=float(inf.get("late_top_p", 0.90)),
        lock_seed_across_steps=bool(inf.get("lock_seed_across_steps", False)),
    )

    ts_cfg = TSformerConfig(
        ckpt=str(ts.get("ckpt", DEFAULT_TS_CKPT)).strip(),
        stats=str(ts.get("stats", DEFAULT_TS_STATS)).strip(),
    )

    return ServerConfig(infinity=inf_cfg, tsformer=ts_cfg, infinity_repo_root=_get_infinity_repo_root())


def _get_server_config() -> ServerConfig:
    """懒加载配置；GRPO 允许环境变量覆盖 checkpoint 以便迭代切换策略。"""
    global _SRV_CFG
    if _SRV_CFG is not None:
        return _SRV_CFG

    cfg_path = os.environ.get("INFINITY_SERVER_CONFIG", "").strip()
    if not cfg_path:
        cfg_path = DEFAULT_SERVER_CONFIG_JSON if os.path.exists(DEFAULT_SERVER_CONFIG_JSON) else ""

    if cfg_path:
        cfg = _load_server_config_from_json(cfg_path)
    else:
        cfg = ServerConfig()

    # 允许环境变量覆盖 checkpoint 路径。
    # 这里故意让环境变量优先，方便 offline rollout 每轮切换策略而不改 config.json
    # （例如 StageA->StageB->StageA 循环）。
    env_inf_ckpt = os.environ.get("INFINITY_CKPT", "").strip()
    if env_inf_ckpt:
        cfg.infinity.ckpt = env_inf_ckpt
    env_ts_ckpt = os.environ.get("TS_P2P_CKPT", "").strip()
    if env_ts_ckpt:
        cfg.tsformer.ckpt = env_ts_ckpt
    env_ts_stats = os.environ.get("TS_P2P_STATS", "").strip()
    if env_ts_stats:
        cfg.tsformer.stats = env_ts_stats

    # 即使 config.json 已提供 runtime 参数，也允许环境变量覆盖。
    # GRPO 实验中 StageA 必须和 StageB 的 logprob 打分模式对齐，
    # 例如 teacher-forcing `trace_ce` 期望 cfg=1、tau=1。
    try:
        v = os.environ.get("INFINITY_CFG", "").strip()
        if v:
            cfg.infinity.cfg = float(v)
    except Exception:
        pass
    try:
        v = os.environ.get("INFINITY_TAU_IMAGE", "").strip()
        if v:
            cfg.infinity.tau_image = float(v)
    except Exception:
        pass
    try:
        v = os.environ.get("INFINITY_TAU_VIDEO", "").strip()
        if v:
            cfg.infinity.tau_video = float(v)
    except Exception:
        pass
    try:
        v = os.environ.get("INFINITY_TOP_K", "").strip()
        if v:
            cfg.infinity.top_k = int(float(v))
    except Exception:
        pass
    try:
        v = os.environ.get("INFINITY_TOP_P", "").strip()
        if v:
            cfg.infinity.top_p = float(v)
    except Exception:
        pass

    _SRV_CFG = cfg
    return cfg


# -------------------------
# 4) 工具函数
# -------------------------
_DATA_URL_SPLIT_RE = re.compile(r"^data:image/[^;]+;base64,", flags=re.IGNORECASE)


def _load_image_from_base64(s: str) -> Image.Image:
    """把客户端上传的 base64 或 data URL 图片解码成 RGB PIL 图像。"""
    if not isinstance(s, str) or not s.strip():
        raise ValueError("图片字符串为空")
    b64 = _DATA_URL_SPLIT_RE.sub("", s.strip())
    raw = base64.b64decode(b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _sorted_image_paths(images_dir: str) -> List[str]:
    """按文件名稳定排序读取目录中的常见图片文件，供自测或离线评估使用。"""
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    names = [n for n in os.listdir(images_dir) if n.lower().endswith(exts)]
    names.sort()
    return [os.path.join(images_dir, n) for n in names]


def _to_cm_deg(deltas_m_rad: torch.Tensor) -> torch.Tensor:
    """
    deltas: [..., 6] = [dx,dy,dz,droll,dyaw,dpitch]，单位为 (m, rad)。
    返回单位转换后的 (cm, deg)。

    单位换算公式：
    - meters -> cm：乘以 100；
    - rad -> deg：乘以 180/pi。
    """
    out = deltas_m_rad.clone()
    out[..., 0:3] = out[..., 0:3] * 100.0
    out[..., 3:6] = out[..., 3:6] * (180.0 / math.pi)
    return out


def _prompt_with_duration(prompt: str, *, num_frames: int, fps: int, append_tag: bool = True) -> str:
    """按 InfinityStar 约定把视频时长标签追加到 prompt 前缀中。"""
    if not append_tag:
        return prompt
    dur_s = (int(num_frames) - 1) // max(1, int(fps))
    return f"<<<t={dur_s}s>>>{prompt}"


# -------------------------
# 5) 模型持有对象（只加载一次）
# -------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.bfloat16 if (_DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16 if _DEVICE == "cuda" else torch.float32

_infinity_args = None
_infinity_session_template: Optional[InfinityStreamingSession] = None
_infinity_self_correction: Optional[SelfCorrection] = None

_ts_model: Optional[torch.nn.Module] = None
_ts_mean: Optional[torch.Tensor] = None
_ts_std: Optional[torch.Tensor] = None

# ActionHead（reference-video TimesFormer）可选模式：
# - 输入：4-frame windows（stride=1），再聚合成逐帧 deltas。
# - 输出：逐帧 6D deltas，之后转换成 API 使用的 cm/deg。
_ah_vit_cls = None  # type: ignore
_ah_model: Optional[torch.nn.Module] = None
_ah_stats: Optional[Dict[str, "np.ndarray"]] = None  # type: ignore[name-defined]
_ah_preprocess = None  # type: ignore
_AH_KITTI_MEAN = [0.34721234, 0.36705238, 0.36066107]
_AH_KITTI_STD = [0.30737526, 0.31515116, 0.32020183]
_AH_TARGET_H = 192
_AH_TARGET_W = 640

DEFAULT_ACTIONHEAD_REPO_ROOT = REPO / "Worldmodel" / "action_decoder" / "actionhead_runtime"


def _get_actionhead_repo_root() -> Path:
    """解析 reference-video ActionHead 代码根目录，允许 ACTIONHEAD_REPO_ROOT 覆盖。"""
    p = os.environ.get("ACTIONHEAD_REPO_ROOT", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return DEFAULT_ACTIONHEAD_REPO_ROOT


def _import_actionhead_modules(repo_root: Path) -> None:
    """
    为 actionhead reference-video 模式导入 TimesFormer VisionTransformer。
    注意：这里故意不导入任何 `datasets.*` 模块，避免和 latent TSformer repo
    中同名的 `datasets` package 冲突。
    """
    global _ah_vit_cls, _ah_preprocess
    if _ah_vit_cls is not None and _ah_preprocess is not None:
        return
    if np is None:
        raise RuntimeError("actionhead 模式需要安装 numpy")
    if not repo_root.exists():
        raise FileNotFoundError(f"找不到 ActionHead repo：{repo_root}")
    if str(repo_root) not in sys.path:
        # 使用 append 而不是 insert(0)，尽量减少导入遮蔽。
        sys.path.append(str(repo_root))
    try:
        from torchvision import transforms as T  # type: ignore
    except Exception as e:
        raise RuntimeError(f"actionhead 模式需要安装 torchvision：{e}")
    from timesformer.models.vit import VisionTransformer  # type: ignore

    _ah_vit_cls = VisionTransformer
    # 对齐 predict_reference_videos_batch copy.py 的预处理：
    # ToPILImage -> Resize((H,W)) -> ToTensor -> Normalize（不 crop）
    _ah_preprocess = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((int(_AH_TARGET_H), int(_AH_TARGET_W))),
            T.ToTensor(),
            T.Normalize(mean=_AH_KITTI_MEAN, std=_AH_KITTI_STD),
        ]
    )


def _default_action_head_mode() -> str:
    """
    从环境变量读取默认 action-head mode。
    启动初始化模型时用它判断是否需要提前加载 TSformer(P2P)。
    """
    return os.environ.get("ACTION_HEAD_MODE", "").strip().lower()


def _use_actionhead_ref_mode_by_default() -> bool:
    """判断启动时默认是否进入 decoded-video + TimesFormer ViT 动作头路径。"""
    mode = _default_action_head_mode()
    return mode in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead")


def _load_actionhead_stats(run_config_path: str) -> Dict[str, "np.ndarray"]:  # type: ignore[name-defined]
    """从 ActionHead 训练 run_config.json 读取反归一化均值/方差。"""
    assert np is not None
    with open(run_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    stats = cfg.get("label_stats") or {}
    need = ("mean_angles", "std_angles", "mean_t", "std_t")
    for k in need:
        if k not in stats:
            raise ValueError(f"run_config.json 缺少 label_stats.{k}")
    out: Dict[str, "np.ndarray"] = {}
    out["mean_angles"] = np.asarray(stats["mean_angles"], dtype=np.float32)
    out["std_angles"] = np.asarray(stats["std_angles"], dtype=np.float32)
    out["mean_t"] = np.asarray(stats["mean_t"], dtype=np.float32)
    out["std_t"] = np.asarray(stats["std_t"], dtype=np.float32)
    return out


def _init_actionhead_model(*, ckpt_path: str, run_config_path: str) -> None:
    """懒加载 reference-video TimesFormer ViT 动作头和标签统计量。"""
    global _ah_model, _ah_stats
    if _ah_model is not None and _ah_stats is not None:
        return
    repo_root = _get_actionhead_repo_root()
    _import_actionhead_modules(repo_root)
    assert _ah_vit_cls is not None

    device = torch.device("cuda" if _DEVICE == "cuda" else "cpu")
    model = _ah_vit_cls(  # type: ignore[misc]
        img_size=(int(_AH_TARGET_H), int(_AH_TARGET_W)),
        num_classes=18,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=lambda *a, **kw: torch.nn.LayerNorm(*a, eps=1e-6, **kw),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        num_frames=4,
        attention_type="divided_space_time",
    )
    ckpt = torch.load(os.path.abspath(ckpt_path), map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[ActionHead] 以 strict=False 加载权重：missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()

    _ah_model = model
    _ah_stats = _load_actionhead_stats(os.path.abspath(run_config_path))

# 并发控制：单进程测试阶段只保留一把 GPU pipeline 锁。
try:
    import asyncio

    _LOCK: "asyncio.Lock" = asyncio.Lock()
except Exception:
    _LOCK = None  # type: ignore


def _load_tsformer_p2p(
    *,
    ckpt_path: str,
    stats_path: str,
    device: str,
) -> Tuple[torch.nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """加载旧版 latent P2P TSformer 动作头，并返回可选的输出反归一化统计。"""
    try:
        from pretrain_latent_p2p import build_p2p_model  # type: ignore
    except Exception as e:
        raise RuntimeError(f"旧版 TSformer(P2P) 导入失败（请安装 fvcore 等依赖）：{e}")
    args = argparse.Namespace(window_size=2, hidden_dim=96, num_layers=2, device=device, checkpoint=ckpt_path, stats_path=stats_path)
    model = build_p2p_model(args)
    model.to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    new_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        # 保持 strict=False：本 repo 有多个变体，adapter 层不匹配很常见，通常不是致命问题。
        print(f"[TSformer] 以 strict=False 加载权重：missing={len(missing)} unexpected={len(unexpected)}")

    mean_t = std_t = None
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
        mean_t, std_t = mean, std
        print(f"[TSformer] 已加载 stats：{stats_path}")
    else:
        print("[TSformer] 找不到 stats；将输出归一化后的 deltas")
    return model, mean_t, std_t


def _init_models(
    *,
    cfg: ServerConfig,
) -> None:
    """
    初始化常驻模型：InfinityStar 世界模型、streaming session 模板，以及可选 TSformer。

    这个函数是服务的重量级入口，所有 checkpoint 和 runtime 参数都在这里落到真实模型对象。
    """
    global _infinity_args, _infinity_session_template, _infinity_self_correction, _ts_model, _ts_mean, _ts_std

    skip_p2p = _use_actionhead_ref_mode_by_default()
    if _infinity_session_template is not None and (_ts_model is not None or skip_p2p):
        return

    print("[Service] 正在初始化模型...")
    print(f"[Service] 运行设备：device={_DEVICE} dtype={_DTYPE}")

    # 选择 InfinityStar repo 并导入其模块。
    _import_infinity_modules(Path(cfg.infinity_repo_root))

    def _resolve_path(p: str) -> str:
        """把配置里的相对路径解析到 action_aware_grpo 目录下，绝对路径保持不变。"""
        if not p:
            return p
        if os.path.isabs(p):
            return p
        return str((ROOT / p).resolve())

    cfg.infinity.ckpt = _resolve_path(cfg.infinity.ckpt)
    cfg.tsformer.ckpt = _resolve_path(cfg.tsformer.ckpt)
    cfg.tsformer.stats = _resolve_path(cfg.tsformer.stats)

    if not cfg.infinity.ckpt:
        raise ValueError("InfinityStar checkpoint 路径为空（请在 config.json 或 INFINITY_CKPT 环境变量中设置）")

    # InfinityStar：构建 args，并且只加载一次模型。
    a = _make_infinity_args(  # type: ignore[misc]
        ckpt=os.path.abspath(cfg.infinity.ckpt),
        pn=str(cfg.infinity.pn),
        fps=int(cfg.infinity.fps),
        num_frames=int(cfg.infinity.num_frames),
        seed=0,
        dynamic_scale_schedule=str(cfg.infinity.dynamic_scale_schedule),
        mask_type=str(cfg.infinity.mask_type),
        cfg=float(cfg.infinity.cfg),
        tau_image=float(cfg.infinity.tau_image),
        tau_video=float(cfg.infinity.tau_video),
    )

    # 对齐 StageB 训练默认值，这对 flex-attn packing 正确性很关键。
    # 当 train_with_var_seq_len=1 时，Infinity forward 会把（visual+text）序列 pad 到 `pad_to_multiplier`。
    # trace_ce old_logprob 路径依赖相同的 padding 规则，才能匹配 flex-attn mask 构造。
    try:
        a.train_with_var_seq_len = 1
        a.pad_to_multiplier = int(getattr(a, "pad_to_multiplier", 128) or 128) if hasattr(a, "pad_to_multiplier") else 128
    except Exception:
        pass
    try:
        a.train_max_token_len = int(getattr(a, "train_max_token_len", 20480) or 20480)
        a.allow_less_one_elem_in_seq = int(getattr(a, "allow_less_one_elem_in_seq", 1) or 1)
    except Exception:
        pass
    try:
        a.use_flex_attn = True
    except Exception:
        pass

    # 关键：`infinity_elegant` schedules 依赖 `args.frames_inner_clip`
    # 计算 scale_pack_info.frame_ss/frame_ee。如果它和 schedule family 不匹配
    # （例如 clip4frames 与 clip20frames），`freqs_frames[:, frame_ss:frame_ee]`
    # 可能为空，导致 get_visual_rope_embeds() 因 size-0 tensor 失败。
    try:
        sched_name = str(cfg.infinity.dynamic_scale_schedule)
        if "clip4frames" in sched_name:
            a.frames_inner_clip = 4
        elif "clip20frames" in sched_name:
            a.frames_inner_clip = 20
    except Exception:
        pass

    text_tokenizer, text_encoder = load_tokenizer(t5_path=a.text_encoder_ckpt)  # type: ignore[misc]
    vae = load_visual_tokenizer(a).float().to(_DEVICE)  # type: ignore[misc]
    infinity = load_transformer(vae, a).to(_DEVICE)  # type: ignore[misc]
    infinity.eval().requires_grad_(False)
    self_correction = SelfCorrection(vae, a)  # type: ignore[misc]

    session = InfinityStreamingSession(  # type: ignore[misc]
        args=a,
        infinity_model=infinity,
        vae=vae,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        h_div_w_template=float(cfg.infinity.h_div_w_template),
    )

    _infinity_args = a
    _infinity_session_template = session
    _infinity_self_correction = self_correction

    # TSformer(P2P)：除非默认模式是 actionhead_ref_vit，否则只加载一次。
    # actionhead_ref_vit 模式会从解码后的视频窗口预测动作，
    # 因此不需要 p2p 权重，且权重可能不兼容。
    if skip_p2p:
        _ts_model, _ts_mean, _ts_std = None, None, None
        print("[Service] ACTION_HEAD_MODE=actionhead_ref_vit*，跳过加载 TSformer(P2P)。")
    else:
        ts_model, mean_t, std_t = _load_tsformer_p2p(
            ckpt_path=os.path.abspath(cfg.tsformer.ckpt),
            stats_path=os.path.abspath(cfg.tsformer.stats) if cfg.tsformer.stats else "",
            device=_DEVICE,
        )
        _ts_model, _ts_mean, _ts_std = ts_model, mean_t, std_t

    print("[Service] 模型初始化完成。")


# -------------------------
# 6) 每条轨迹的状态
# -------------------------
@dataclass
class TrajectoryState:
    """
    单条在线轨迹的服务端状态。

    它同时保存客户端已经上传的真实帧、InfinityStar streaming cache、跨 segment 复用的
    latent 边界帧，以及本次 session 的输出进度。GRPO rollout 依赖这些状态把多次 HTTP
    请求拼成一条连续闭环轨迹。
    """

    session_id: str
    prompt_raw: str
    negative_prompt: str = ""
    created_at: float = field(default_factory=lambda: time.time())

    # 已接收帧，已经按 (tgt_h,tgt_w) transform 到 [-1,1]；每帧是 [3,H,W] CPU tensor。
    frames_cpu: List[torch.Tensor] = field(default_factory=list)

    # 每条轨迹的 Infinity session wrapper（保存 text tuple）；cache 在模型里，这里保存导出的副本。
    stream: Optional[InfinityStreamingSession] = None
    kv_cache: Optional[Any] = None

    # 首帧 i2v 对齐辅助信息（可选）。
    gt_ls_Bl_first: Optional[Any] = None

    # closed-loop 辅助状态。
    dyn_res: Optional[Any] = None
    h_sel: Optional[str] = None
    firstframe_prepared: bool = False

    # TSformer latent 记忆：跨 segment 携带一个 latent。
    last_latent_1: Optional[torch.Tensor] = None  # [1,16,1,H,W]，位于 CPU（float16/float32）。
    latent_dir: Optional[str] = None  # 磁盘 cache 目录："<session_id>_infinity_latnet"。

    # transform 目标空间尺寸，只确定一次。
    tgt_h: Optional[int] = None
    tgt_w: Optional[int] = None
    h_div_w_template: float = float(DEFAULT_H_DIV_W)

    # 动作发射进度记录。
    last_emitted_segment: int = -1
    # 请求模式提示；prefix_mode 表示每次请求都带完整 prefix [1..K]。
    last_req_prefix_mode: bool = False

    def num_frames(self) -> int:
        """返回当前 session 已接收并完成 transform 的真实 RGB 帧数量。"""
        return len(self.frames_cpu)


_TRAJ: Dict[str, TrajectoryState] = {}
_SESSION_ALIAS: Dict[str, str] = {}


def _make_run_session_id(external_session_id: str) -> str:
    """为 reset_session 生成内部唯一 session id，避免复用外部 route id 时覆盖旧缓存。"""
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    # 添加 ns 后缀，避免同一秒内碰撞。
    suffix = str(time.time_ns() % 1_000_000_000).rjust(9, "0")
    return f"{external_session_id}__{ts}_{suffix}"


def _get_or_create_traj(session_id: str, prompt: str, negative_prompt: str) -> TrajectoryState:
    """获取或创建轨迹状态，并在首次创建时准备 latent cache 目录和断点 latent。"""
    cfg = _get_server_config()
    if session_id in _TRAJ:
        st = _TRAJ[session_id]
        # 允许客户端在后续请求中省略 prompt。
        if prompt and prompt.strip():
            st.prompt_raw = prompt.strip()
        if negative_prompt is not None:
            st.negative_prompt = (negative_prompt or "").strip()
        return st

    st = TrajectoryState(
        session_id=session_id,
        prompt_raw=prompt.strip(),
        negative_prompt=(negative_prompt or "").strip(),
        h_div_w_template=float(cfg.infinity.h_div_w_template),
    )
    # latent cache 目录：可选，但默认启用。
    root = os.environ.get("INFINITY_LATENT_CACHE_ROOT", "").strip()
    if not root:
        root = str((ROOT / "cache").resolve())
    try:
        os.makedirs(root, exist_ok=True)
        st.latent_dir = os.path.join(root, f"{session_id}_infinity_latnet")
        os.makedirs(st.latent_dir, exist_ok=True)
        # 尽力续跑：如果存在 last_latent.pt 就加载。
        last_path = os.path.join(st.latent_dir, "last_latent.pt")
        if os.path.exists(last_path):
            try:
                t = torch.load(last_path, map_location="cpu")
                if isinstance(t, torch.Tensor) and t.ndim == 5 and t.shape[0] == 1 and t.shape[1] == 16 and t.shape[2] == 1:
                    st.last_latent_1 = t.contiguous()
            except Exception:
                pass
    except Exception:
        st.latent_dir = None
        st.last_latent_1 = None
    _TRAJ[session_id] = st
    return st


def _ensure_traj_infinity_session(st: TrajectoryState) -> None:
    """为指定轨迹创建轻量 streaming session，并初始化 prompt/text cache。"""
    assert _infinity_session_template is not None
    assert _infinity_args is not None
    cfg = _get_server_config()

    if st.stream is not None:
        return

    # 创建轻量级的逐轨迹 wrapper；它和模板共享 model/vae/text 组件。
    tpl = _infinity_session_template
    st.stream = InfinityStreamingSession(  # type: ignore[misc]
        args=_infinity_args,
        infinity_model=tpl.infinity,
        vae=tpl.vae,
        text_tokenizer=tpl.text_tokenizer,
        text_encoder=tpl.text_encoder,
        h_div_w_template=float(st.h_div_w_template),
    )

    prompt_infer = _prompt_with_duration(
        st.prompt_raw,
        num_frames=int(cfg.infinity.num_frames),
        fps=int(cfg.infinity.fps),
        append_tag=bool(getattr(_infinity_args, "append_duration2caption", 0)),
    )
    st.stream.reset(prompt_infer, negative_prompt=st.negative_prompt, cfg_scale=float(cfg.infinity.cfg))
    st.kv_cache = st.stream.infinity.export_kv_cache()


def _import_kv_cache_for_traj(st: TrajectoryState) -> None:
    """把轨迹保存的 KV cache 导回 Infinity 模型，供下一段闭环推理继续使用。"""
    assert st.stream is not None
    if st.kv_cache is None:
        return
    # 先重置 cache 存储以清掉上一个 session 的 cache，再导入当前轨迹 cache。
    for blk in st.stream.infinity.unregistered_blocks:
        blk.attn.kv_caching(True, reset=True)
    st.stream.infinity.import_kv_cache(st.kv_cache, overwrite=True)


def _prepare_firstframe_condition_if_needed(st: TrajectoryState) -> None:
    """
    对齐 batch_closed_loop_streaming_infer_routes.py：
    - Step0 使用首帧 gt_leak 注入，并且故意不把 obs1 写入 gt_obs cache。
    - 每条轨迹只预计算一次 gt_ls_Bl_first 与 dyn_res/h_sel。
    """
    cfg = _get_server_config()
    assert st.stream is not None
    assert _infinity_args is not None
    assert _infinity_self_correction is not None
    assert get_dynamic_resolution_meta is not None

    if st.firstframe_prepared:
        return
    if st.num_frames() <= 0:
        raise ValueError("还没有收到任何帧")

    dyn_res, _ = get_dynamic_resolution_meta(_infinity_args.dynamic_scale_schedule, _infinity_args.video_frames)  # type: ignore[misc]
    st.dyn_res = dyn_res

    # 在 dynamic-resolution 表中选择最接近的 h/w template key。
    try:
        import numpy as np  # 局部导入；InfinityStar 工具已使用 numpy。

        h_keys = list(dyn_res.keys())
        h_vals = np.array([float(k) for k in h_keys], dtype=np.float64)
        st.h_sel = h_keys[int(np.argmin(np.abs(h_vals - float(st.h_div_w_template))))]
    except Exception:
        # 兜底：直接取第一个 key。
        st.h_sel = list(dyn_res.keys())[0]

    # 把首帧编码成 gt_ls_Bl_first，用于严格 i2v 对齐。
    obs1 = st.frames_cpu[0].unsqueeze(0).to(_DEVICE, non_blocking=True)  # [1,3,H,W] in [-1,1]
    with torch.no_grad():
        _, _, gt_ls_Bl_first, _, _, _ = st.stream.video_encode(
            vae=st.stream.vae,
            inp_B3HW=obs1,
            vae_features=None,
            self_correction=_infinity_self_correction,
            args=_infinity_args,
            infer_mode=True,
            dynamic_resolution_h_w=dyn_res,
        )
    st.gt_ls_Bl_first = gt_ls_Bl_first
    st.firstframe_prepared = True


def _update_gt_obs_cache_to(st: TrajectoryState, n_frames: int) -> None:
    """用 prefix [1..n_frames] 覆盖 gt_obs cache（B=1）。"""
    assert st.stream is not None
    if n_frames <= 0:
        return
    obs = torch.stack(st.frames_cpu[:n_frames], dim=0)  # [T,3,H,W]
    obs_bcthw = obs.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1,3,T,H,W]
    st.stream.compute_kv_cache_gt(obs_bcthw.to(_DEVICE, non_blocking=True))


def _infer_summed_codes_for_step(
    st: TrajectoryState,
    *,
    step_i: int,
    obs_len: int,
    infer_num_frames: int,
    seed: int,
    top_k: int,
    top_p: float,
    injection: str,
    need_pred_video: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float, Optional[str]]:
    """
    为单个 closed-loop step 运行 InfinityStar 推理，并返回 summed_codes [1,16,pt,H,W]。
    控制流与 batch_closed_loop_streaming_infer_routes.py 保持一致，但跳过 VAE decode。
    """
    cfg = _get_server_config()
    assert st.stream is not None
    assert _infinity_args is not None

    # 确保 session 使用正确的 aspect template。
    st.stream.h_div_w_template = float(st.h_div_w_template)
    st.stream.correction_clear_pred()

    gt_leak = -1
    gt_ls_Bl = None

    if int(step_i) == 0:
        _prepare_firstframe_condition_if_needed(st)
        gt_leak = int(cfg.infinity.gt_leak_first)
        gt_ls_Bl = st.gt_ls_Bl_first
    else:
        inj = str(injection)
        if inj in ("official_leak", "hybrid_leak_gtobs"):
            if not st.dyn_res or not st.h_sel:
                _prepare_firstframe_condition_if_needed(st)
            assert st.dyn_res is not None and st.h_sel is not None
            assert _infinity_self_correction is not None
            # 编码连续 prefix [1..obs_len]，并使用自动 leak 深度注入。
            prefix = torch.stack(st.frames_cpu[:obs_len], dim=0)  # [T,3,H,W]
            prefix_obs = prefix.permute(1, 0, 2, 3).unsqueeze(0).contiguous().to(_DEVICE, non_blocking=True)  # [1,3,T,H,W]
            with torch.no_grad():
                _, _, gt_ls_Bl_prefix, _, _, _ = st.stream.video_encode(
                    vae=st.stream.vae,
                    inp_B3HW=prefix_obs,
                    vae_features=None,
                    self_correction=_infinity_self_correction,
                    args=_infinity_args,
                    infer_mode=True,
                    dynamic_resolution_h_w=st.dyn_res,
                )
            if inj == "hybrid_leak_gtobs":
                # Hybrid：同时写入 gt_obs cache，帮助稳定后段 segment。
                st.stream.compute_kv_cache_gt(prefix_obs)

            # 帧到 latent 的压缩索引：latent_index(frame f) = (f - 1)//temporal_compress_rate + 1；
            # 这里用真实前缀长度 obs_len 算出该前缀覆盖到第几个 latent。
            pt_obs = (int(obs_len) - 1) // int(getattr(_infinity_args, "temporal_compress_rate", 4)) + 1
            pt2sched = st.dyn_res[st.h_sel][_infinity_args.pn]["pt2scale_schedule"]
            leak_auto = len(pt2sched[int(pt_obs)])
            gt_leak = int(leak_auto)
            gt_ls_Bl = gt_ls_Bl_prefix
        else:
            # gt_obs 模式：把 prefix 写入 cache，并在无 leak 的情况下推理。
            _update_gt_obs_cache_to(st, obs_len)

    sched = st.stream.build_schedule_for_num_frames(int(infer_num_frames))
    tau_list = [float(cfg.infinity.tau_image)] * int(sched.tower_split_index) + [float(cfg.infinity.tau_video)] * (
        len(sched.scale_schedule) - int(sched.tower_split_index)
    )

    # 沿用旧的独立推理 wrapper 风格，通过 gen_one_example() 调用；
    # 它内部会规范化 cfg/tau 列表，并按请求处理 prompt 编码。
    try:
        trace_sample_logprob = 0.0
        trace_path: Optional[str] = None
        if infinity_gen_one_example is None:
            raise RuntimeError("InfinityStar gen_one_example 尚未导入")
        assert _infinity_args is not None

        prompt_infer = _prompt_with_duration(
            st.prompt_raw,
            num_frames=int(cfg.infinity.num_frames),
            fps=int(cfg.infinity.fps),
            append_tag=bool(getattr(_infinity_args, "append_duration2caption", 0)),
        )

        with torch.no_grad():
            if _DEVICE == "cuda":
                with torch.cuda.amp.autocast(enabled=True, dtype=next(iter(st.stream.infinity.parameters())).dtype):
                    out_gen = infinity_gen_one_example(  # type: ignore[misc]
                        st.stream.infinity,
                        st.stream.vae,
                        st.stream.text_tokenizer,
                        st.stream.text_encoder,
                        prompt_infer,
                        negative_prompt=str(st.negative_prompt or ""),
                        g_seed=int(seed),
                        gt_leak=int(gt_leak),
                        gt_ls_Bl=gt_ls_Bl,
                        cfg_list=float(cfg.infinity.cfg),
                        tau_list=tau_list,
                        scale_schedule=sched.scale_schedule,
                        top_k=int(top_k),
                        top_p=float(top_p),
                        cfg_insertion_layer=[0],
                        vae_type=int(getattr(_infinity_args, "vae_type", 64)),
                        sampling_per_bits=1,
                        enable_positive_prompt=0,
                        low_vram_mode=True,
                        args=_infinity_args,
                        get_visual_rope_embeds=st.stream.get_visual_rope_embeds,
                        context_info=sched.context_info,
                        noise_list=None,
                        return_summed_code_only=True,
                        return_trace=True,
                    )
            else:
                out_gen = infinity_gen_one_example(  # type: ignore[misc]
                    st.stream.infinity,
                    st.stream.vae,
                    st.stream.text_tokenizer,
                    st.stream.text_encoder,
                    prompt_infer,
                    negative_prompt=str(st.negative_prompt or ""),
                    g_seed=int(seed),
                    gt_leak=int(gt_leak),
                    gt_ls_Bl=gt_ls_Bl,
                    cfg_list=float(cfg.infinity.cfg),
                    tau_list=tau_list,
                    scale_schedule=sched.scale_schedule,
                    top_k=int(top_k),
                    top_p=float(top_p),
                    cfg_insertion_layer=[0],
                    vae_type=int(getattr(_infinity_args, "vae_type", 64)),
                    sampling_per_bits=1,
                    enable_positive_prompt=0,
                    low_vram_mode=True,
                    args=_infinity_args,
                    get_visual_rope_embeds=st.stream.get_visual_rope_embeds,
                    context_info=sched.context_info,
                    noise_list=None,
                    return_summed_code_only=True,
                    return_trace=True,
                )

        summed_codes = out_gen
        if isinstance(out_gen, dict):
            summed_codes = out_gen.get("summed_codes", None)
            try:
                slp = out_gen.get("sample_logprob", 0.0)
                if isinstance(slp, torch.Tensor):
                    trace_sample_logprob = float(slp.detach().to("cpu").item())
                else:
                    trace_sample_logprob = float(slp)
            except Exception:
                trace_sample_logprob = 0.0
            # 如果有按 clip 对齐的 logprob，优先使用当前 segment 对应值。
            try:
                clipid_target = int(step_i) + 1
                byc = out_gen.get("sample_logprob_by_clip", None)
                if isinstance(byc, list) and len(byc) > clipid_target:
                    v = byc[clipid_target]
                    if isinstance(v, torch.Tensor):
                        trace_sample_logprob = float(v.detach().to("cpu").item())
                    else:
                        trace_sample_logprob = float(v)
            except Exception:
                pass

            # 保留一份 sampling-time logprob（通常受 cfg/tau 和完整 schedule 影响）。
            trace_sample_logprob_sampling = float(trace_sample_logprob)
            trace_sample_logprob_trace_ce = None

            # 可选：用 teacher-forcing 单次 forward 计算 old_logprob，兼容 StageB trace_ce。
            # 通过环境变量启用：
            # 中文标题：INFINITY_STAGEA_OLD_LOGPROB_MODE=trace_ce
            mode = (os.environ.get("INFINITY_STAGEA_OLD_LOGPROB_MODE", "") or "").strip().lower()
            if mode == "trace_ce":
                    strict = (os.environ.get("INFINITY_STAGEA_OLD_LOGPROB_STRICT", "0") or "0").strip()
                    strict = int(strict) == 1
                    import math as _math
                    import json as _json
                    import numpy as _np
                    import torch.nn.functional as _F
                    from infinity.schedules.dynamic_resolution import (  # type: ignore
                        get_first_full_spatial_size_scale_index as _ffssi,
                    )
                    from infinity.schedules.infinity_elegant import (  # type: ignore
                        get_visual_rope_embeds as _get_rope,
                        interpolate as _interp,
                    )

                    idx_trace = out_gen.get("idx_trace", None)
                    if not isinstance(idx_trace, list) or len(idx_trace) <= 0:
                        raise RuntimeError("trace_ce 需要 out_gen 中包含 idx_trace list")

                    assert st.stream is not None
                    gpt = st.stream.infinity
                    vae = st.stream.vae
                    device0 = next(iter(gpt.parameters())).device
                    model_dtype = next(iter(gpt.parameters())).dtype if _DEVICE == "cuda" else torch.float32

                    # 在 policy scoring 期间关闭 cond-drop 随机性。
                    orig_cdr = float(getattr(gpt, "cond_drop_rate", 0.0) or 0.0)
                    try:
                        gpt.cond_drop_rate = 0.0
                    except Exception:
                        pass
                    try:
                        text_pair = getattr(st.stream, "_text_cond_tuple", None)
                        if not (isinstance(text_pair, tuple) and len(text_pair) >= 1):
                            raise RuntimeError("trace_ce 需要先调用 stream.reset()（缺少 _text_cond_tuple）")
                        text_cond_tuple = text_pair[0]

                        scale_schedule = sched.scale_schedule
                        first_full = int(_ffssi(scale_schedule))
                        scales_in_one_clip = int(first_full) + 1
                        clipid_target = int(step_i) + 1

                        # 使用 rollout args 中的 repetition，必须匹配 StageB scoring。
                        img_rep_s = str(getattr(_infinity_args, "image_scale_repetition", "[1]")).strip()
                        vid_rep_s = str(getattr(_infinity_args, "video_scale_repetition", "[1]")).strip()
                        image_rep = _np.array(_json.loads(img_rep_s), dtype=_np.int64)
                        video_rep = _np.array(_json.loads(vid_rep_s), dtype=_np.int64)

                        cache_step_id: Dict[int, int] = {}
                        step_ptr0 = 0
                        for _si in range(len(scale_schedule)):
                            if _si < scales_in_one_clip:
                                _rt = int(image_rep[_si % scales_in_one_clip])
                            else:
                                _rt = int(video_rep[_si % scales_in_one_clip])
                            _rt = max(1, _rt)
                            cache_step_id[int(_si)] = int(step_ptr0 + _rt - 1)
                            step_ptr0 += _rt
                        if len(idx_trace) < int(step_ptr0):
                            raise RuntimeError(f"trace_ce 需要 idx_trace 长度 >= {step_ptr0}，实际为 {len(idx_trace)}")

                        # 确定性选择 scale，与 StageB trace_ce 保持一致。
                        tmax = int(float(os.environ.get("INFINITY_GRPO_TRACE_CE_TMAX", "20480")))
                        total_tokens = int(_np.array(scale_schedule).prod(-1).sum())
                        select_si_list = list(range(len(scale_schedule)))
                        if total_tokens > tmax:
                            S = int(scales_in_one_clip)
                            L = int(len(scale_schedule))
                            c = int(clipid_target)
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
                                select_si_list = list(range(min(L, S)))
                                tgt = min(L - 1, c * S + (S - 1))
                                if tgt not in select_si_list:
                                    select_si_list.append(tgt)
                            select_si_list = sorted({int(x) for x in select_si_list if 0 <= int(x) < L})
                        # 保留用于精确 StageB replay / debugging。
                        trace_ce_select_si_list = [int(x) for x in select_si_list]

                        # 将 context refs 重映射到选中的子集。
                        scale_pack_info = sched.context_info
                        real_si_2_new_si: Dict[int, int] = {int(r): int(i2) for i2, r in enumerate(select_si_list)}
                        new_scale_pack_info: Dict[int, Dict[str, Any]] = {}
                        for new_q, real_q in enumerate(select_si_list):
                            new_scale_pack_info[int(new_q)] = {"ref_sids": []}
                            try:
                                refs = scale_pack_info[int(real_q)].get("ref_sids", [])
                            except Exception:
                                refs = []
                            for rr in refs:
                                nn = real_si_2_new_si.get(int(rr), None)
                                if nn is not None:
                                    new_scale_pack_info[int(new_q)]["ref_sids"].append(int(nn))

                        apply_patchify = bool(getattr(gpt, "apply_spatial_patchify", False))
                        if apply_patchify:
                            vae_scale_schedule = [(int(pt), int(2 * ph), int(2 * pw)) for (pt, ph, pw) in scale_schedule]
                        else:
                            vae_scale_schedule = [(int(pt), int(ph), int(pw)) for (pt, ph, pw) in scale_schedule]

                        def _latent_to_raw_tokens(lat: torch.Tensor) -> torch.Tensor:
                            """把当前尺度 latent 展平成 Infinity forward 需要的 token 序列。"""
                            if apply_patchify:
                                _x = lat.permute(0, 2, 1, 3, 4)
                                _x = torch.nn.functional.pixel_unshuffle(_x, 2)
                                _x = _x.permute(0, 2, 1, 3, 4)
                            else:
                                _x = lat
                            return _x.reshape(_x.shape[0], _x.shape[1], -1).permute(0, 2, 1).contiguous()

                        vae_embed_dim = int(getattr(gpt, "vae_embed_dim", 0) or getattr(vae, "embed_dim", 0) or 64)
                        if getattr(_infinity_args, "noise_input", 0):
                            summed_code = torch.randn((1, vae_embed_dim, *vae_scale_schedule[0]), device=device0, dtype=model_dtype)
                        else:
                            summed_code = torch.zeros((1, vae_embed_dim, *vae_scale_schedule[0]), device=device0, dtype=model_dtype)

                        x_scales: List[torch.Tensor] = []
                        gt_scales: List[torch.Tensor] = []
                        rope_scales: List[torch.Tensor] = []
                        dlabels: List[int] = []
                        muls: List[int] = []
                        clipids: List[int] = []

                        for si, pn in enumerate(scale_schedule):
                            pt, ph, pw = int(pn[0]), int(pn[1]), int(pn[2])
                            this_lat = summed_code
                            if tuple(this_lat.shape[-3:]) != tuple(vae_scale_schedule[int(si)]):
                                this_lat = _F.interpolate(this_lat, size=vae_scale_schedule[int(si)], mode=vae.quantizer.z_interplote_down).contiguous()

                            if int(si) in real_si_2_new_si:
                                x_scales.append(_latent_to_raw_tokens(this_lat))
                                rope_scales.append(
                                    _get_rope(
                                        gpt.rope2d_freqs_grid,
                                        scale_schedule,
                                        int(si),
                                        int(cache_step_id[int(si)]),
                                        device=device0,
                                        args=_infinity_args,
                                        scale_pack_info=scale_pack_info,
                                        first_full_spatial_size_scale_index=int(first_full),
                                    )
                                )
                                mul = int(pt * ph * pw)
                                muls.append(mul)
                                d_label = int(
                                    getattr(gpt.other_args, "detail_scale_dim", 64)
                                    if (ph * pw) >= int(getattr(vae.quantizer, "detail_scale_min_tokens", 350))
                                    else getattr(gpt.other_args, "semantic_scale_dim", 16)
                                )
                                dlabels.append(d_label)
                                clipids.append(int(si // max(1, scales_in_one_clip)))
                                forced = idx_trace[int(cache_step_id[int(si)])]
                                if not isinstance(forced, torch.Tensor):
                                    forced = torch.tensor(forced, dtype=torch.long, device=device0)
                                else:
                                    forced = forced.to(device=device0, dtype=torch.long)
                                if forced.ndim == 1:
                                    forced = forced.unsqueeze(0)
                                gt_scales.append(forced.reshape(1, mul, d_label).contiguous())

                            # 用缓存 token 更新 latent 状态；这是近似过程，但必须匹配 StageB trace_ce。
                            target_pn = vae_scale_schedule[int(first_full)] if int(si) < scales_in_one_clip else vae_scale_schedule[-1]
                            forced_upd = idx_trace[int(cache_step_id[int(si)])]
                            if not isinstance(forced_upd, torch.Tensor):
                                forced_upd = torch.tensor(forced_upd, dtype=torch.long, device=device0)
                            else:
                                forced_upd = forced_upd.to(device=device0, dtype=torch.long)
                            if forced_upd.ndim == 1:
                                forced_upd = forced_upd.unsqueeze(0)
                            mul = int(pt * ph * pw)
                            d_label = int(
                                getattr(gpt.other_args, "detail_scale_dim", 64)
                                if (ph * pw) >= int(getattr(vae.quantizer, "detail_scale_min_tokens", 350))
                                else getattr(gpt.other_args, "semantic_scale_dim", 16)
                            )
                            idx_Bld = forced_upd.reshape(1, -1)
                            idx_Bthwd = idx_Bld.reshape(1, pt, ph, pw, d_label)
                            if apply_patchify:
                                _t = idx_Bthwd.permute(0, 1, 4, 2, 3)
                                _t = torch.nn.functional.pixel_shuffle(_t, 2)
                                idx_Bthwd = _t.permute(0, 1, 3, 4, 2)
                            if gt_leak > 0 and isinstance(gt_ls_Bl, list) and int(si) < int(gt_leak):
                                try:
                                    idx_Bthwd = gt_ls_Bl[int(cache_step_id[int(si)])].to(device=device0, dtype=idx_Bthwd.dtype)
                                except Exception:
                                    pass
                            if getattr(gpt.other_args, "use_two_stage_lfq", 0):
                                if (ph * pw) >= int(getattr(vae.quantizer, "detail_scale_min_tokens", 350)):
                                    is_sem = False
                                    lfq = vae.quantizer.lfq_detail
                                else:
                                    is_sem = True
                                    lfq = vae.quantizer.lfq_semantic
                                codes = lfq.indices_to_codes(idx_Bthwd, "bit_label")
                                codes = _interp(
                                    codes,
                                    size=(vae_embed_dim, *target_pn),
                                    mode=vae.quantizer.z_interplote_up,
                                    quantizer=vae.quantizer,
                                    is_semantic_scale=is_sem,
                                ).contiguous()
                            else:
                                codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bthwd, "bit_label")
                                codes = _F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)
                            summed_code = _F.interpolate(summed_code, size=target_pn, mode=vae.quantizer.z_interplote_up).contiguous()
                            summed_code = summed_code + codes
                            if int(si) < len(scale_schedule) - 1 and tuple(scale_schedule[int(si)][-2:]) == tuple(scale_schedule[-1][-2:]):
                                if getattr(_infinity_args, "noise_input", 0):
                                    summed_code = torch.randn((1, summed_code.shape[1], *vae_scale_schedule[int(si) + 1]), device=device0, dtype=summed_code.dtype)
                                else:
                                    summed_code = torch.zeros((1, summed_code.shape[1], *vae_scale_schedule[int(si) + 1]), device=device0, dtype=summed_code.dtype)

                        if not x_scales:
                            raise RuntimeError("trace_ce 选中的 scales 为空")
                        x_vis = torch.cat(x_scales, dim=1)
                        rope_vis = torch.cat(rope_scales, dim=4)

                        # 构造 querysid_refsid。
                        # 重要：`lens` 可能因 `_text_cond_tuple` 的缓存方式而是 padded/max length。
                        # `gpt(...)` 内部做严格 packing 长度检查时，必须使用每个 sample 的真实
                        # text token 长度；这个长度可以从 `cu_seqlens_k` / `kv_compact` 稳定推导。
                        kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
                        try:
                            # StageA server 中 B=1（脚本里强制 cfg=1.0），因此取 [0:1]。
                            if hasattr(cu_seqlens_k, "__len__") and len(cu_seqlens_k) >= 2:
                                text_len0 = int(cu_seqlens_k[1].item() if hasattr(cu_seqlens_k[1], "item") else cu_seqlens_k[1]) - int(
                                    cu_seqlens_k[0].item() if hasattr(cu_seqlens_k[0], "item") else cu_seqlens_k[0]
                                )
                            else:
                                text_len0 = int(kv_compact.shape[0])
                        except Exception:
                            # 保守兜底：使用 kv_compact 长度。
                            text_len0 = int(getattr(kv_compact, "shape", [0])[0] or 0)
                        text_len0 = int(max(0, text_len0))
                        # 构造 super_scale_lengths，用来匹配 Infinity forward padding：
                        # train_with_var_seq_len=1 时，Infinity.forward 会拼接（visual + text），
                        # 再 pad 到 pad_to_multiplier。build_flex_attn_func 会断言：
                        # 代码/形状说明：sum(super_scale_lengths) == padded_seq_len
                        scale_lengths = [int(m) for m in muls] + [int(text_len0)]
                        valid_scales = int(len(muls) + 1)
                        try:
                            pad_to = int(getattr(_infinity_args, "pad_to_multiplier", 128) or 128)
                            pad_to = max(1, pad_to)
                            cur_seq_len = int(_np.sum(scale_lengths))
                            pad_seq_len = int(_math.ceil(cur_seq_len / float(pad_to)) * pad_to - cur_seq_len)
                            pad_seq_len = int(max(0, pad_seq_len))
                            if pad_seq_len > 0:
                                scale_lengths = scale_lengths + [int(pad_seq_len)]
                        except Exception:
                            pass
                        max_sid_nums = 2000
                        qref = torch.zeros((max_sid_nums, max_sid_nums), device=device0, dtype=torch.bool)
                        for i_sid in range(valid_scales):
                            qref[i_sid][i_sid] = True
                        for local_q in range(len(muls)):
                            global_text_sid = len(muls)
                            qref[local_q][global_text_sid] = True
                            for local_r in new_scale_pack_info[int(local_q)]["ref_sids"]:
                                qref[local_q][int(local_r)] = True

                        with torch.cuda.amp.autocast(enabled=(_DEVICE == "cuda"), dtype=model_dtype):
                            loss_tok, _, _ = gpt(
                                text_cond_tuple,
                                x_vis,
                                gt_BL=gt_scales,
                                is_image_batch=0,
                                visual_rope_cache=rope_vis,
                                sequece_packing_scales=[[tuple(map(int, scale_schedule[si])) for si in select_si_list]],
                                super_scale_lengths=scale_lengths,
                                super_querysid_super_refsid=qref,
                                other_info_by_scale=None,
                            )

                        nll_target = torch.zeros((1,), dtype=loss_tok.dtype, device=loss_tok.device)
                        tok_ptr = 0
                        for j, mul in enumerate(muls):
                            seg = loss_tok[tok_ptr : tok_ptr + int(mul)]
                            tok_ptr += int(mul)
                            if int(clipids[j]) != int(clipid_target):
                                continue
                            nll_target = nll_target + seg.sum() * float(dlabels[j])
                        trace_sample_logprob_trace_ce = float((-nll_target)[0].detach().to("cpu").item())
                    except Exception as e:
                        # trace_ce 失败时回退到 sampling-time logprob。
                        print(f"[trace_ce] old_logprob 计算失败，回退到 sampling logprob：{e}")
                        if strict:
                            raise
                        trace_sample_logprob_trace_ce = None
                    finally:
                        try:
                            st.stream.infinity.cond_drop_rate = orig_cdr  # type: ignore[attr-defined]
                        except Exception:
                            pass
            if trace_sample_logprob_trace_ce is not None:
                trace_sample_logprob = float(trace_sample_logprob_trace_ce)
            try:
                if st.latent_dir:
                    os.makedirs(st.latent_dir, exist_ok=True)
                    trace_path = os.path.join(st.latent_dir, f"seg{int(step_i):02d}_trace.pt")
                    torch.save(
                        {
                            "segment_index": int(step_i),
                            "obs_len": int(obs_len),
                            "infer_num_frames": int(infer_num_frames),
                            "injection": str(injection),
                            "infinity_cfg": float(cfg.infinity.cfg),
                            "tau_list": [float(x) for x in tau_list],
                            "top_k": int(top_k),
                            "top_p": float(top_p),
                            "pn": str(cfg.infinity.pn),
                            "h_div_w_template": float(st.h_div_w_template),
                            "dynamic_scale_schedule": str(cfg.infinity.dynamic_scale_schedule),
                            "mask_type": str(cfg.infinity.mask_type),
                            "scale_schedule": sched.scale_schedule,
                            "context_info": sched.context_info,
                            "gt_leak": int(gt_leak),
                            "gt_ls_Bl": (
                                [t.detach().to("cpu") for t in gt_ls_Bl]
                                if isinstance(gt_ls_Bl, list)
                                else None
                            ),
                            # 当前 segment 的 clip-id 目标（49f clip4 schedule：clip0=image，clip1..3=video clips）。
                            # 代码/形状说明：seg00->clip1 (2..17)，seg01->clip2 (18..33)，seg02->clip3 (34..49)。
                            "clipid_target": int(step_i) + 1,
                            "step_clipids": out_gen.get("step_clipids", None),
                            "sample_logprob": float(trace_sample_logprob),
                            "sample_logprob_sampling": float(trace_sample_logprob_sampling),
                            "sample_logprob_trace_ce": (
                                float(trace_sample_logprob_trace_ce)
                                if trace_sample_logprob_trace_ce is not None
                                else None
                            ),
                            "trace_ce_select_si_list": trace_ce_select_si_list if "trace_ce_select_si_list" in locals() else None,
                            "trace_ce_total_tokens": int(total_tokens) if "total_tokens" in locals() else None,
                            "trace_ce_tmax": int(tmax) if "tmax" in locals() else None,
                            "sample_logprob_by_clip": out_gen.get("sample_logprob_by_clip", None),
                            "idx_trace": out_gen.get("idx_trace", None),
                            "image_scale_repetition": str(getattr(_infinity_args, "image_scale_repetition", "")),
                            "video_scale_repetition": str(getattr(_infinity_args, "video_scale_repetition", "")),
                        },
                        trace_path,
                    )
            except Exception as e:
                print(f"[trace->file] 跳过保存：{e}")
                trace_path = None
        if not isinstance(summed_codes, torch.Tensor):
            raise RuntimeError(f"Infinity 输出类型不符合预期：{type(out_gen)}")

        pred_vid: Optional[torch.Tensor] = None
        want_decode = bool(st.latent_dir) or bool(need_pred_video)
        if want_decode:
            try:
                with torch.no_grad():
                    pred_vid = st.stream.infinity.summed_codes2images(st.stream.vae, summed_codes)  # 代码/形状说明：[1,T,H,W,3], uint8(BGR)
                if st.latent_dir:
                    total_num_frames = int(cfg.infinity.num_frames)
                    _save_pred_video(st, f"seg{int(step_i):02d}_pred_full_{int(total_num_frames):03d}f.mp4", pred_vid)
            except Exception as e:
                pred_vid = None
                print(f"[pred->video] 跳过解码/保存：{e}")
    except Exception:
        # 向服务端日志打印更完整的 debug 信息（FastAPI 会把异常包装进 HTTP 500 detail）。
        print("[InfinityStar] infer_chunk 失败。正在打印 debug 信息...")
        print(traceback.format_exc())
        try:
            blk0 = st.stream.infinity.unregistered_blocks[0]
            ck = getattr(blk0.attn, "cached_k", {})
            cv = getattr(blk0.attn, "cached_v", {})
            keys = list(ck.keys())
            print(f"[InfinityStar] cached_k keys（第一个 block）：{keys}")
            for k in keys[:10]:
                vk = ck.get(k, None)
                vv = cv.get(k, None)
                sk = tuple(vk.shape) if isinstance(vk, torch.Tensor) else type(vk).__name__
                sv = tuple(vv.shape) if isinstance(vv, torch.Tensor) else type(vv).__name__
                print(f"  - key={k!r} k={sk} v={sv}")
        except Exception:
            print("[InfinityStar]（debug 信息打印失败）")
        raise

    st.stream.correction_clear_pred()
    return summed_codes, pred_vid, float(trace_sample_logprob), trace_path


def _save_latent_tensor(st: TrajectoryState, name: str, t: torch.Tensor) -> None:
    """把调试用 latent 张量以 float16 CPU 格式保存到该轨迹的 cache 目录。"""
    if not st.latent_dir:
        return
    try:
        p = os.path.join(st.latent_dir, name)
        # 保存为 float16 CPU，减少磁盘占用。
        torch.save(t.detach().to("cpu", dtype=torch.float16).contiguous(), p)
    except Exception:
        return


def _save_latent_video_clip(
    st: TrajectoryState,
    name: str,
    latents_B16THW: torch.Tensor,
    *,
    drop_first_frame: bool,
) -> None:
    """
    解码 latent clip，并把 mp4 保存到同一个 latent 目录下。
    - seg0：5 个 latents -> 17 帧（drop_first_frame=False）
    - seg>0：解码 latent5 后丢掉第一个边界帧 -> 16 个新帧
    """
    if not st.latent_dir or st.stream is None or infinity_save_video is None:
        return
    try:
        model_dtype = next(iter(st.stream.infinity.parameters())).dtype if _DEVICE == "cuda" else torch.float32
        z = latents_B16THW.to(_DEVICE, dtype=model_dtype, non_blocking=(_DEVICE == "cuda"))
        with torch.no_grad():
            frames = st.stream.infinity.summed_codes2images(st.stream.vae, z)  # 代码/形状说明：[1,T,H,W,3], uint8
        clip = frames[0] if isinstance(frames, torch.Tensor) else frames[0]
        if drop_first_frame and int(clip.shape[0]) > 1:
            clip = clip[1:]
        clip_np = clip.detach().cpu().numpy() if isinstance(clip, torch.Tensor) else clip
        if int(clip_np.shape[0]) <= 0:
            return
        # Infinity.summed_codes2images 返回 uint8 BGR（它会翻转通道维）。
        # tools.run_infinity.save_video 期望 BGR，并在写入时内部转成 RGB。
        out_path = os.path.join(st.latent_dir, name)
        cfg = _get_server_config()
        infinity_save_video(clip_np, fps=int(cfg.infinity.fps), save_filepath=out_path, force_all_keyframes=True)
    except Exception as e:
        print(f"[latent->video] 跳过 {name}：{e}")


def _save_pred_video(
    st: TrajectoryState,
    name: str,
    pred_video_BTHWC: Any,
) -> None:
    """把 Infinity 解码得到的预测视频保存到 latent_dir；文件名必须唯一以避免覆盖。"""
    if not st.latent_dir or infinity_save_video is None:
        return
    try:
        vid = pred_video_BTHWC
        if isinstance(vid, torch.Tensor):
            vid = vid.detach().cpu().numpy()
        # 期望 [B,T,H,W,3] 或 [T,H,W,3]。
        if getattr(vid, "ndim", 0) == 5:
            vid = vid[0]
        if getattr(vid, "ndim", 0) != 4 or int(vid.shape[-1]) != 3:
            return
        # Infinity.summed_codes2images 返回 BGR；tools.run_infinity.save_video 也期望 BGR。
        out_path = os.path.join(st.latent_dir, name)
        cfg = _get_server_config()
        infinity_save_video(vid, fps=int(cfg.infinity.fps), save_filepath=out_path, force_all_keyframes=True)
    except Exception as e:
        print(f"[pred->video] 跳过 {name}：{e}")


def _slice_abs_latents_from_summed_codes(
    summed_codes: torch.Tensor,
    *,
    abs_lat_start: int,
    abs_lat_end: int,
    infer_num_frames: int,
    total_num_frames: int,
) -> torch.Tensor:
    """
        summed_codes: [1,16,pt_local,H,W]，可能来自 full-horizon
                    （infer_num_frames==total_num_frames），也可能来自右对齐 tail-window
                    说明：（infer_num_frames < total_num_frames）。
        abs_lat_start/end: 完整视频时间线上的 1-indexed 绝对 latent 下标。
        返回：[1,16,T,H,W]，其中 T == abs_lat_end-abs_lat_start+1（前提是落在窗口内）。

    """
    if abs_lat_end < abs_lat_start:
        raise ValueError(f"abs_lat 范围错误：{abs_lat_start}..{abs_lat_end}")
    t_local = int(summed_codes.shape[2])

    local_start = int(abs_lat_start)
    local_end = int(abs_lat_end)
    if int(infer_num_frames) != int(total_num_frames):
        window_start_abs = int(total_num_frames) - int(infer_num_frames) + 1  # 1-indexed 绝对帧 id
        # tail-window 从某个绝对帧开始；先用 latent_index(f) = (f - 1)//4 + 1
        # 找到这个窗口在整条 latent 时间线上的起点，再把绝对 latent 下标平移成本地切片下标。
        abs_lat_start_window = (int(window_start_abs) - 1) // 4 + 1
        local_start = int(abs_lat_start) - int(abs_lat_start_window) + 1
        local_end = int(abs_lat_end) - int(abs_lat_start_window) + 1

    s0 = max(1, int(local_start))
    e0 = min(int(local_end), int(t_local))
    if e0 < s0:
        raise ValueError(f"latent 切片越界：abs [{abs_lat_start}..{abs_lat_end}] -> 本地 [{local_start}..{local_end}]，t_local={t_local}")
    out = summed_codes[:, :, (s0 - 1) : e0].contiguous()
    # 如果请求的 slice 部分落在窗口外，当前先按错误处理。
    if int(out.shape[2]) != int(abs_lat_end - abs_lat_start + 1):
        raise ValueError(
            f"latent 切片长度不匹配：需要 {abs_lat_end-abs_lat_start+1}，实际 {out.shape[2]} "
            f"(abs [{abs_lat_start}..{abs_lat_end}] -> 本地 [{local_start}..{local_end}], infer_num_frames={infer_num_frames})"
        )
    return out


def _slice_abs_frames_from_pred_video_bgr(
    pred_video_BTHWC: object,
    *,
    abs_frame_start: int,
    abs_frame_end: int,
    infer_num_frames: int,
    total_num_frames: int,
) -> "np.ndarray":  # type: ignore[name-defined]
    """
    pred_video_BTHWC: [1,T,H,W,3] 或 [T,H,W,3] uint8(BGR)，其中 T==infer_num_frames
    （full 或 tail-window）。
    abs_frame_start/end: 完整视频时间线上的 1-indexed 绝对像素帧下标。
    返回：[Tslice,H,W,3] uint8(BGR)。
    """
    if np is None:
        raise RuntimeError("actionhead 模式需要安装 numpy")
    if abs_frame_end < abs_frame_start:
        raise ValueError(f"abs_frame 范围错误：{abs_frame_start}..{abs_frame_end}")

    vid = pred_video_BTHWC
    if isinstance(vid, torch.Tensor):
        vid = vid.detach().cpu().numpy()
    if getattr(vid, "ndim", 0) == 5:
        vid = vid[0]
    if getattr(vid, "ndim", 0) != 4 or int(vid.shape[-1]) != 3:
        raise ValueError(f"pred_video 形状错误：{getattr(vid,'shape',None)}")
    t_local = int(vid.shape[0])

    if int(infer_num_frames) == int(total_num_frames):
        local_start = int(abs_frame_start) - 1
        local_end = int(abs_frame_end) - 1
    else:
        window_start_abs = int(total_num_frames) - int(infer_num_frames) + 1  # 1-indexed 绝对帧 id
        local_start = int(abs_frame_start) - int(window_start_abs)
        local_end = int(abs_frame_end) - int(window_start_abs)

    s0 = max(0, int(local_start))
    e0 = min(int(local_end), int(t_local) - 1)
    if e0 < s0:
        raise ValueError(
            f"frame 切片越界：abs [{abs_frame_start}..{abs_frame_end}] -> 本地 [{local_start}..{local_end}]，t_local={t_local} "
            f"(infer_num_frames={infer_num_frames}, total_num_frames={total_num_frames})"
        )
    out = vid[s0 : (e0 + 1)]
    if int(out.shape[0]) != int(abs_frame_end - abs_frame_start + 1):
        raise ValueError(
            f"frame 切片长度不匹配：需要 {abs_frame_end-abs_frame_start+1}，实际 {out.shape[0]} "
            f"(abs [{abs_frame_start}..{abs_frame_end}] -> 本地 [{local_start}..{local_end}])"
        )
    return out


def _frame_tensor_chw_neg1to1_to_bgr_uint8(fr_3hw: torch.Tensor) -> "np.ndarray":  # type: ignore[name-defined]
    """
    fr_3hw: torch.Tensor [3,H,W]，取值范围 [-1,1]（RGB）。
    返回：uint8 [H,W,3]，通道顺序 BGR。
    """
    if np is None:
        raise RuntimeError("需要安装 numpy")
    x = fr_3hw.detach().to("cpu", dtype=torch.float32).clamp(-1.0, 1.0)
    x = (x + 1.0) * 0.5 * 255.0
    x = x.round().clamp(0.0, 255.0).to(torch.uint8)
    rgb = x.permute(1, 2, 0).contiguous().numpy()  # HWC RGB
    return rgb[..., ::-1].copy()  # BGR


@dataclass
class SegmentInferResult:
    """
    单个闭环 segment 的中间结果。

    既包含给动作头使用的 5 个 latent，也保留本段世界模型输出、采样 logprob 和 trace
    路径，方便 GRPO StageB 回放或排查 old_logprob 对齐问题。
    """

    latent5_input: torch.Tensor
    summed_codes: torch.Tensor
    pred_vid_bgr: Optional[torch.Tensor]
    infer_num_frames: int
    obs_len: int
    next_obs_len: int
    total_num_frames: int
    sample_logprob: float
    trace_path: Optional[str]


def _infer_latents_for_actions_and_advance_cache(
    st: TrajectoryState,
    *,
    segment_index: int,
    seed: int,
    advance_gt_obs_to_next: bool = True,
    need_pred_video: bool = False,
) -> SegmentInferResult:
    """
    针对给定 segment i 执行：
    - 按闭环/rolling-tail 配置运行 InfinityStar 推理。
    - seg0：截取预测 4 个动作所需的 5 个 latent step。
    - seg>0：只截取 4 个新的 latent step，再和保存的 last_latent_1 拼成 5 个 latents。
    - 将 gt_obs cache 覆盖到新揭示的前缀（points[i+1]）。
    返回：latent5_input [1,16,5,H,W]。
    """
    cfg = _get_server_config()
    points = cfg.infinity.points()
    if segment_index < 0 or segment_index >= len(points) - 1:
        raise ValueError(f"segment_index 错误：segment_index={segment_index}, points={points}")

    obs_len = int(points[segment_index])
    next_obs_len = int(points[segment_index + 1])

    # 当前 step 专用控制项。
    lock_seed = bool(cfg.infinity.lock_seed_across_steps)
    local_seed = int(seed) + (0 if lock_seed else int(segment_index))
    use_late = int(segment_index) >= int(cfg.infinity.late_step_start)
    step_top_k = int(cfg.infinity.late_top_k) if use_late else int(cfg.infinity.top_k)
    step_top_p = float(cfg.infinity.late_top_p) if use_late else float(cfg.infinity.top_p)
    inj = str(cfg.infinity.late_v2v_history_injection or cfg.infinity.v2v_history_injection) if use_late else str(cfg.infinity.v2v_history_injection)

    # early-stop rollout：每个 segment 只推理到自己的 next_obs_len。
    # seg00：推到 17f（预测 2..17），seg01：推到 33f（预测 18..33），seg02：推到 49f（预测 34..49）。
    infer_num_frames = int(next_obs_len)

    summed_codes, pred_vid, step_logprob, step_trace_path = _infer_summed_codes_for_step(
        st,
        step_i=int(segment_index),
        obs_len=obs_len,
        infer_num_frames=infer_num_frames,
        seed=int(local_seed),
        top_k=int(step_top_k),
        top_p=float(step_top_p),
        injection=inj,
        need_pred_video=bool(need_pred_video),
    )  # 代码/形状说明：[1,16,pt_local,H,W]

    # 对 early-stop inference，切片时把预测 horizon 当作 “total” 时间线。
    total_num_frames = int(infer_num_frames)
    # segment 右边界是 RGB 帧 `next_obs_len`；用 latent_index(f) = (f - 1)//4 + 1
    # 转成动作头要读取的绝对 latent 右边界。
    abs_end_lat = (int(next_obs_len) - 1) // 4 + 1  # 当前 segment 结束后的绝对 latent。

    latent5_input: torch.Tensor
    force_full_window = bool(getattr(st, "last_req_prefix_mode", False))
    if int(segment_index) == 0 or st.last_latent_1 is None or force_full_window:
        # seg0（或续跑失败）：提供完整 5-latent 窗口 [end-4..end]。
        abs_start_lat = max(1, int(abs_end_lat) - 4)
        latent5_input = _slice_abs_latents_from_summed_codes(
            summed_codes,
            abs_lat_start=int(abs_start_lat),
            abs_lat_end=int(abs_end_lat),
            infer_num_frames=int(infer_num_frames),
            total_num_frames=int(total_num_frames),
        )
        # 如果视频太短、不足 5 个 latents，就重复最后一个补齐。
        if int(latent5_input.shape[2]) < 5:
            rep = latent5_input[:, :, -1:].repeat(1, 1, 5 - int(latent5_input.shape[2]), 1, 1)
            latent5_input = torch.cat([latent5_input, rep], dim=2)
    else:
        # seg>0：先把上一段 RGB 边界 `prev_obs_len` 转成 latent_index(prev_obs_len)，
        # 再只取 4 个新 latents（prev_end+1 .. cur_end），并和 last_latent_1 拼接。
        prev_obs_len = int(points[segment_index])  # 等于当前 obs_len。
        prev_end_lat = (int(prev_obs_len) - 1) // 4 + 1
        abs_start_lat_new = int(prev_end_lat) + 1
        abs_end_lat_new = int(abs_end_lat)
        new4 = _slice_abs_latents_from_summed_codes(
            summed_codes,
            abs_lat_start=int(abs_start_lat_new),
            abs_lat_end=int(abs_end_lat_new),
            infer_num_frames=int(infer_num_frames),
            total_num_frames=int(total_num_frames),
        )  # 期望 [1,16,4,H,W]。
        # 保持 latents 在 CPU 上，方便下游 TSformer 和磁盘保存；
        # 同时避免和存于 CPU 的 st.last_latent_1 拼接时发生 device mismatch。
        if isinstance(new4, torch.Tensor):
            new4 = new4.detach().to("cpu").contiguous()
        if int(new4.shape[2]) < 4:
            rep = new4[:, :, -1:].repeat(1, 1, 4 - int(new4.shape[2]), 1, 1)
            new4 = torch.cat([new4, rep], dim=2)
        last1 = st.last_latent_1
        if last1 is None:
            # 理论上不会发生，但保留兜底。
            last1 = new4[:, :, :1].clone()
        # 确保空间维度 (H,W) 对齐。
        if last1.shape[-2:] != new4.shape[-2:]:
            raise ValueError(f"latent 空间尺寸不匹配：last1={tuple(last1.shape)} new4={tuple(new4.shape)}")
        latent5_input = torch.cat([last1.to(new4.dtype), new4], dim=2).contiguous()

    # 统一 latent5 tensor 的放置位置；TSformer 内部会从 CPU 转到 cuda。
    latent5_input = latent5_input.detach().to("cpu").contiguous()

    # 推进状态：用新揭示的 GT prefix [1..next_obs_len] 覆盖 cache。
    # 最后一个 segment 可由调用方关闭此步骤，避免把非真实帧写入 gt_obs cache
    # （例如 34-49 完全是预测帧时）。
    if bool(advance_gt_obs_to_next):
        st.stream.correction_clear_pred()  # type: ignore[union-attr]
        _update_gt_obs_cache_to(st, int(next_obs_len))
        st.stream.correction_clear_pred()  # type: ignore[union-attr]
        st.kv_cache = st.stream.infinity.export_kv_cache()  # type: ignore[union-attr]

    # 更新记忆状态并保存到磁盘。
    st.last_latent_1 = latent5_input[:, :, -1:].detach().to("cpu").contiguous()
    _save_latent_tensor(st, f"seg{int(segment_index):02d}_latent5_input.pt", latent5_input)
    if int(segment_index) == 0:
        _save_latent_tensor(st, "seg00_latent5.pt", latent5_input)
        _save_latent_video_clip(st, "seg00_latent5_17f.mp4", latent5_input, drop_first_frame=False)
        # 额外显式拆分并保存边界 latent（frame 1）和 new4 latents（frames 2..17）。
        _save_latent_tensor(st, "seg00_first1.pt", latent5_input[:, :, 0:1].contiguous())
        _save_latent_tensor(st, "seg00_new4.pt", latent5_input[:, :, 1:].contiguous())
        _save_latent_video_clip(st, "seg00_new4_16f.mp4", latent5_input, drop_first_frame=True)
    else:
        # 另外只保存 new4，便于检查。
        _save_latent_tensor(st, f"seg{int(segment_index):02d}_new4.pt", latent5_input[:, :, 1:].contiguous())
        _save_latent_video_clip(
            st,
            f"seg{int(segment_index):02d}_new4_16f.mp4",
            latent5_input,
            drop_first_frame=True,
        )
    _save_latent_tensor(st, "last_latent.pt", st.last_latent_1)
    return SegmentInferResult(
        latent5_input=latent5_input,
        summed_codes=summed_codes,
        pred_vid_bgr=pred_vid,
        infer_num_frames=int(infer_num_frames),
        obs_len=int(obs_len),
        next_obs_len=int(next_obs_len),
        total_num_frames=int(total_num_frames),
        sample_logprob=float(step_logprob),
        trace_path=step_trace_path,
    )


def _tsformer_predict_actions_from_summed_codes(
    summed_codes_BCTHW: torch.Tensor,
    *,
    prefix_latents: int,
) -> torch.Tensor:
    """
    返回最后 4 个动作 (4,6)，单位 cm/deg。
    """
    assert _ts_model is not None
    assert summed_codes_BCTHW.ndim == 5 and summed_codes_BCTHW.shape[0] == 1, f"期望 [1,C,T,H,W]，实际为 {tuple(summed_codes_BCTHW.shape)}"

    # InfinityStar 的 WAN VAE 常使用 patchified codes：(B, 4*C0, T, H/2, W/2)。
    # TSformer adapter 用未 patchify 的表示训练（C0=16）。
    # 如果看到 C=64，就反 patchify -> C=16，并把空间尺寸放大 2 倍。
    if int(summed_codes_BCTHW.shape[1]) == 64:
        x = summed_codes_BCTHW.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,C,H,W]
        x = torch.nn.functional.pixel_shuffle(x, 2)  # [B,T,C/4,H*2,W*2]
        summed_codes_BCTHW = x.permute(0, 2, 1, 3, 4).contiguous()  # [B,C/4,T,H*2,W*2]

    assert int(summed_codes_BCTHW.shape[1]) == 16, f"TSformer 需要 16 通道 latents，实际 C={int(summed_codes_BCTHW.shape[1])}"

    t_lat = int(summed_codes_BCTHW.shape[2])
    k = int(prefix_latents)
    if k > t_lat:
        k = t_lat
    if k < 2:
        raise ValueError(f"prefix_latents 太小：{k}")

    # [1,16,T,H,W] -> [T,16,H,W]
    lat_TCHW = summed_codes_BCTHW[0].permute(1, 0, 2, 3).contiguous()  # [T,16,H,W]
    lat_TCHW = lat_TCHW[:k]

    # 中文说明：windows: (k-1, 2, 16, H, W)
    windows = torch.stack([lat_TCHW[:-1], lat_TCHW[1:]], dim=1)
    windows = windows.to(_DEVICE, dtype=torch.float32)

    with torch.no_grad():
        out = _ts_model(windows)  # (N, 6)
        if _ts_mean is not None and _ts_std is not None:
            out = out * _ts_std + _ts_mean

    # 取最后 4 个动作；不足时补齐。
    if out.shape[0] >= 4:
        last4 = out[-4:]
    else:
        # 重复最后一个动作补齐。
        pads = [out[-1:]] * (4 - int(out.shape[0]))
        last4 = torch.cat([out] + pads, dim=0)

    return _to_cm_deg(last4).detach().cpu()


def _ah_denorm_window_preds(pred_norm: "np.ndarray", stats: Dict[str, "np.ndarray"]) -> "np.ndarray":  # type: ignore[name-defined]
    """把 ActionHead 窗口级归一化输出还原成训练标签原始单位。"""
    assert np is not None
    b = int(pred_norm.shape[0])
    pred = pred_norm.reshape(b, 3, 6).astype(np.float32)
    mean_a, std_a = stats["mean_angles"], stats["std_angles"]
    mean_t, std_t = stats["mean_t"], stats["std_t"]
    pred[:, :, 0:3] = pred[:, :, 0:3] * std_a[None, None, :] + mean_a[None, None, :]
    pred[:, :, 3:6] = pred[:, :, 3:6] * std_t[None, None, :] + mean_t[None, None, :]
    return pred


def _ah_aggregate_overlapping_windows(
    *,
    num_frames: int,
    window_starts: List[int],
    window_deltas: "np.ndarray",  # (N,3,6)
    window_size: int = 4,
) -> "np.ndarray":  # (T,6)
    """将重叠 4 帧窗口预测的 3 个 delta 平均回逐帧动作序列。"""
    assert np is not None
    acc = np.zeros((int(num_frames), 6), dtype=np.float32)
    cnt = np.zeros((int(num_frames),), dtype=np.int32)
    for i, s in enumerate(window_starts):
        for j in range(1, int(window_size)):
            t = int(s) + int(j)
            if 0 <= t < int(num_frames):
                acc[t] += window_deltas[i, j - 1]
                cnt[t] += 1
    out = np.zeros((int(num_frames), 6), dtype=np.float32)
    mask = cnt > 0
    out[mask] = acc[mask] / cnt[mask, None]
    return out


def _actionhead_ref_predict_actions_cm_deg(
    *,
    frames_rgb_uint8: List["np.ndarray"],  # 中文说明：长度为 1(prev)+16(clip)=17，RGB uint8
    batch_size: int = 8,
    stride: int = 1,
    pre_resize_hw: int = 0,
) -> List[List[float]]:
    """
    Reference-video actionhead 模式（TimesFormer ViT）：
    - 输入长度 T>=4 的 RGB 帧序列（uint8）。
    - 按 size=4 和 stride 滑动窗口运行。
    - 每个窗口预测 3 个 deltas（对应后 3 帧），再聚合成逐帧 deltas。
    - 返回 frames[1:] 对应动作（长度 T-1），API 顺序为 [dx,dy,dz,droll,dyaw,dpitch]，单位 cm/deg。
    """
    if np is None:
        raise RuntimeError("actionhead 模式需要安装 numpy")
    if _ah_model is None or _ah_stats is None or _ah_preprocess is None:
        raise RuntimeError("actionhead 模型尚未初始化")
    if len(frames_rgb_uint8) < 4:
        return []

    # 可选的中间 resize（debug 桥接用）：
    # 有些 pipeline 会先强制 480p -> 256x256，再替换成原生 480p actionhead。
    # 注意：当前 reference actionhead checkpoint 使用 img_size=(192,640) 训练，
    # 因此这里仅是预 resize；_ah_preprocess 之后模型实际仍接收 192x640。
    if int(pre_resize_hw) <= 0:
        env_pre = os.environ.get("ACTIONHEAD_PRE_RESIZE_HW", "").strip()
        if env_pre:
            try:
                pre_resize_hw = int(env_pre)
            except Exception:
                pre_resize_hw = 0
    # 默认不做中间预 resize，直接把 848x480 预处理成 actionhead 输入。

    # 预处理成已归一化的 tensors (C,H,W)。
    frames_t: List[torch.Tensor] = []
    for f in frames_rgb_uint8:
        if int(pre_resize_hw) > 0:
            try:
                pil = Image.fromarray(f)
                pil = pil.resize((int(pre_resize_hw), int(pre_resize_hw)), resample=Image.BILINEAR)
                f = np.asarray(pil, dtype=np.uint8)  # type: ignore[assignment]
            except Exception:
                pass
        frames_t.append(_ah_preprocess(f))  # type: ignore[misc]

    window_size = 4
    t = int(len(frames_t))
    starts = list(range(0, t - window_size + 1, max(1, int(stride))))
    clips: List[torch.Tensor] = []
    for s in starts:
        # 堆成 (C,T,H,W)。
        x = torch.stack([frames_t[s + i] for i in range(window_size)], dim=0).transpose(0, 1).contiguous()
        clips.append(x)

    preds = []
    device = torch.device("cuda" if _DEVICE == "cuda" else "cpu")
    with torch.no_grad():
        for i in range(0, len(clips), max(1, int(batch_size))):
            batch = torch.stack(clips[i : i + int(batch_size)], dim=0).to(device)  # (B,C,T,H,W)
            out = _ah_model(batch.float())
            preds.append(out.detach().cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0) if preds else np.zeros((0, 18), dtype=np.float32)
    window_deltas = _ah_denorm_window_preds(pred_norm, _ah_stats) if pred_norm.shape[0] > 0 else np.zeros((0, 3, 6), dtype=np.float32)
    deltas = _ah_aggregate_overlapping_windows(num_frames=t, window_starts=starts, window_deltas=window_deltas, window_size=window_size)

    # 只把 frames[1:] 的逐帧 deltas 转成 API actions（cm/deg）。
    # 这里假设 delta 格式为 [dz, dy, dx, tx, ty, tz]，角度单位 rad、平移单位 m；
    # 换算公式是 meter*100 -> cm，rad*180/pi -> deg。
    out_actions: List[List[float]] = []
    for i in range(1, t):
        dz, dy, dx = [float(x) for x in deltas[i, 0:3]]
        tx, ty, tz = [float(x) for x in deltas[i, 3:6]]
        out_actions.append(
            [
                tx * 100.0,
                ty * 100.0,
                tz * 100.0,
                dx * (180.0 / math.pi),  # 中文说明：roll (x)
                dz * (180.0 / math.pi),  # yaw (z)
                dy * (180.0 / math.pi),  # 中文说明：pitch (y)
            ]
        )
    return out_actions


# -------------------------
# 7) FastAPI schema（可选）
# -------------------------
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="InfinityStar+TSformer 动作 API",
        description="把 InfinityStar 预测出的 summed_codes（latents）转换成 TSformer(P2P) 动作增量（cm/deg）。num_frames/step 可通过 config.json 配置。",
        version="0.1.0",
    )

    class PredictDeltaActionsRequest(BaseModel):
        """
        `/v1/predict_delta_actions` 请求体。

        客户端可以按增量帧或 prefix_mode 上传帧；服务端据此判断哪个 segment 已经可发射动作。
        """

        session_id: str = Field(..., description="轨迹/session 标识符")
        instruction: Optional[str] = Field(None, description="导航指令 / prompt；首次调用或更新 prompt 时使用")
        prompt: Optional[str] = Field(None, description="`instruction` 的兼容别名（兼容旧客户端）")
        negative_prompt: Optional[str] = Field("", description="可选负向 prompt")
        images_base64: List[str] = Field(..., description="RGB 图片的 base64 字符串；首次调用通常传 1 帧，之后通常每次传 16 帧")
        reset_session: bool = Field(
            False,
            description="如果为 true，即使之前用过同一个 session_id，也强制开始新的运行（丢弃内存状态，并用新的内部 run session id 避免覆盖旧结果）。",
        )
        action_head_mode: str = Field(
            "tsformer_latent",
            description=(
                "动作头模式。"
                "'tsformer_latent'（默认）：5 个 latents -> 每个 segment 输出 4 个动作。"
                "'actionhead_ref_vit'：把 Infinity 预测视频解码成 RGB 帧，再运行 4 帧滑窗 ViT "
                "（stride=1，并聚合重叠窗口）来为每个 16 帧 clip 输出 16 个动作。"
            ),
        )
        action_head_batch_size: int = Field(8, description="`actionhead_ref_vit` 滑窗推理时的批大小。")
        action_head_stride: int = Field(1, description="`actionhead_ref_vit` 滑窗推理的 stride（默认 1）。")
        action_head_pre_resize_hw: int = Field(
            0,
            description=(
                "actionhead_ref_vit 预处理前的可选中间预 resize。"
                "如果 >0，每个解码后的 RGB 帧会先 resize 到 (N,N)（例如 256）；设为 0 表示关闭。"
                "注意：参考 actionhead 模型内部随后仍会用 torchvision Resize 到 (192,640)（不裁剪），以匹配 predict_reference_videos_batch*.py。"
            ),
        )
        allow_future_segments: bool = Field(
            False,
            description=(
                "如果为 true，当真实前缀到达 points[i] 时，服务端就可以发射 segment i 的动作 "
                "（不再强制要求到达 points[i+1]）。这用于严格闭环协议："
                "发送 1 帧+prompt -> 得到 4 个动作 -> 执行并收集 16 帧 -> 发送 16 帧 -> 得到下一组 4 个动作，以此类推。"
            ),
        )
        prefix_mode: bool = Field(
            False,
            description="如果为 true，images_base64 每次都包含完整 prefix [1..K]。服务端只会追加新的尾部帧，避免重复。",
        )
        allow_future_last_segment: bool = Field(
            False,
            description="如果为 true，最后一个 segment 在真实 prefix 到达 points[seg]（例如 33）时即可发射（例如 points [1,17,33,49] 的 seg02），不要求 points[seg+1]（49）张真实帧。这对应“34-49 帧由模型预测补齐”的语义。",
        )
        seed: Optional[int] = Field(
            None,
            description="可选采样基准 seed。省略时服务端使用 0。若 lock_seed_across_steps=true，此 seed 会用于该 session 的所有 segments（官方批处理脚本使用 seed=base_seed + global_idx*1000）。",
        )
        debug: bool = False

    class PredictDeltaActionsResponse(BaseModel):
        """动作预测响应体，包含本次发射的 segment、动作、logprob trace 和 session 进度。"""

        actions: List[List[float]] = Field(
            ...,
            description="动作增量列表；每个元素是 [dx_cm,dy_cm,dz_cm,droll_deg,dyaw_deg,dpitch_deg]。长度取决于 action_head_mode（tsformer_latent：每个 segment 4 个；actionhead_ref_vit：每个 16 帧 clip 16 个）。",
        )
        segment_index: int = Field(
            ...,
            description="本次输出对应哪个 segment（0..S-1，其中 S=len(points)-1，来自 config）。-1 表示没有发射新的 segment。",
        )
        num_received_frames: int
        prefix_latents: int
        done: bool
        used_prompt: Optional[str] = None
        segment_old_logprob: Optional[float] = None
        segment_trace_path: Optional[str] = None

    @app.get("/health")
    async def health():
        """返回模型加载状态、闭环 points 和当前在线 session 数量。"""
        cfg = _get_server_config()
        tgt_h = tgt_w = None
        try:
            if _infinity_session_template is not None:
                sched = _infinity_session_template.build_schedule_for_num_frames(int(cfg.infinity.num_frames))
                tgt_h, tgt_w = int(sched.tgt_h), int(sched.tgt_w)
        except Exception:
            tgt_h = tgt_w = None
        return {
            "status": "ok",
            "device": _DEVICE,
            "dtype": str(_DTYPE),
            "ts_ckpt_loaded": _ts_model is not None,
            "infinity_loaded": _infinity_session_template is not None,
            "active_sessions": len(_TRAJ),
            "num_frames": int(cfg.infinity.num_frames),
            "step": int(cfg.infinity.step),
            "points": cfg.infinity.points(),
            "h_div_w_template": float(cfg.infinity.h_div_w_template),
            "tgt_h": tgt_h,
            "tgt_w": tgt_w,
            "rolling_tail_infer": bool(cfg.infinity.rolling_tail_infer),
            "rolling_infer_mode": str(cfg.infinity.rolling_infer_mode),
            "v2v_history_injection": str(cfg.infinity.v2v_history_injection),
        }

    @app.on_event("startup")
    async def _startup_load_models():
        """
        可选的提前加载：如果环境变量已经设置好，就在启动时加载权重。
        这样即使首个请求尚未到达，也能保持“权重常驻”行为。
        """
        cfg = _get_server_config()
        if not cfg.infinity.ckpt:
            # 允许未设置 ckpt 时先启动服务；请求会快速失败，直到 config/env 设置完成。
            print("[Service] startup：未设置 InfinityStar ckpt，跳过启动时预加载模型。")
            return
        _init_models(cfg=cfg)

    @app.post("/v1/predict_delta_actions", response_model=PredictDeltaActionsResponse)
    async def predict_delta_actions(req: "PredictDeltaActionsRequest"):
        """HTTP 动作预测入口；只负责串行化 GPU 调用，实际逻辑在同步实现函数中。"""
        # 全局锁：保证单进程内 GPU 调用安全。
        if _LOCK is not None:
            async with _LOCK:
                return _predict_delta_actions_impl(req)
        return _predict_delta_actions_impl(req)

else:
    app = None  # type: ignore


def _predict_delta_actions_impl(req) -> "PredictDeltaActionsResponse":
    """
    执行动作预测主流程。

    流程顺序是：解析/重置 session -> 解码并缓存新帧 -> 判断 segment 是否就绪 ->
    调 InfinityStar 生成 latent/视频 -> 调动作头 -> 更新发射进度并返回。
    """
    cfg = _get_server_config()
    if not cfg.infinity.ckpt:
        raise HTTPException(status_code=500, detail="必须提供 InfinityStar ckpt（请在 config.json 或 INFINITY_CKPT 环境变量中设置）")
    _init_models(cfg=cfg)

    external_session_id = (req.session_id or "").strip()
    if not external_session_id:
        raise HTTPException(status_code=400, detail="必须提供 session_id")

    raw_prompt = (req.instruction or "").strip() or (req.prompt or "").strip()
    allow_future_segments = bool(getattr(req, "allow_future_segments", False))
    # 自动 “new run” 规则，用来避免和陈旧内存状态冲突：
    # 如果前端用恰好 1 帧 + prompt/instruction 开始一条 route，
    # 即使复用了同一个外部 session_id，也按新 run 处理。
    auto_reset_on_one_frame = os.environ.get("INFINITY_RESET_SESSION_ON_ONE_FRAME", "1").strip() in ("1", "true", "True")
    one_frame_with_prompt = bool(raw_prompt) and int(len(getattr(req, "images_base64", []) or [])) == 1
    want_reset = bool(getattr(req, "reset_session", False)) or (auto_reset_on_one_frame and one_frame_with_prompt)
    if want_reset and not raw_prompt:
        raise HTTPException(status_code=400, detail="reset_session 需要同时提供 instruction/prompt")

    if want_reset:
        old_key = _SESSION_ALIAS.get(external_session_id, external_session_id)
        try:
            if old_key in _TRAJ:
                del _TRAJ[old_key]
        except Exception:
            pass
        try:
            # 同时删除旧逻辑直接存到 external_session_id 下的状态。
            if external_session_id in _TRAJ and external_session_id != old_key:
                del _TRAJ[external_session_id]
        except Exception:
            pass
        _SESSION_ALIAS[external_session_id] = _make_run_session_id(external_session_id)

    session_id = _SESSION_ALIAS.get(external_session_id, external_session_id)
    if session_id not in _TRAJ and not raw_prompt:
        raise HTTPException(status_code=400, detail="一个 session 的第一次调用必须提供 instruction/prompt")

    st = _get_or_create_traj(session_id, raw_prompt, req.negative_prompt or "")

    # 解码并追加帧。
    if not req.images_base64:
        raise HTTPException(status_code=400, detail="必须提供 images_base64")

    new_imgs: List[Image.Image] = []
    try:
        for s in req.images_base64:
            new_imgs.append(_load_image_from_base64(s))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"images_base64 解码失败：{e}")

    # 重要：
    # 许多 UAVFlow 风格数据集把帧存成 256x256，但推理时仍需要固定的训练期模板
    # （例如 h_div_w_template=0.562 -> 848x480）。
    # 因此默认不使用原始帧宽高比覆盖 `st.h_div_w_template`。
    #
    # 如果确实要根据首帧宽高比自动检测，可通过环境变量启用：
    # 中文说明：INFINITY_AUTO_H_DIV_W_TEMPLATE=1
    if st.num_frames() == 0 and os.environ.get("INFINITY_AUTO_H_DIV_W_TEMPLATE", "0").strip() in ("1", "true", "True"):
        w, h = new_imgs[0].size
        if w > 0 and h > 0:
            st.h_div_w_template = float(h) / float(w)

    _ensure_traj_infinity_session(st)

    # 只确定一次 (tgt_h,tgt_w)；schedule 来自配置中的 num_frames。
    if st.tgt_h is None or st.tgt_w is None:
        assert st.stream is not None
        sched = st.stream.build_schedule_for_num_frames(int(cfg.infinity.num_frames))
        st.tgt_h, st.tgt_w = int(sched.tgt_h), int(sched.tgt_w)
        # 可选的目标分辨率强校验，强制 640x640 模板等场景会用到。
        req_hw = os.environ.get("INFINITY_REQUIRE_TGT_HW", "").strip()
        if req_hw:
            try:
                parts = [p.strip() for p in str(req_hw).split(",")]
                req_h = int(parts[0])
                req_w = int(parts[1])
                if req_h > 0 and req_w > 0 and (int(st.tgt_h), int(st.tgt_w)) != (int(req_h), int(req_w)):
                    raise HTTPException(
                        status_code=500,
                        detail=f"目标分辨率不匹配：实际 {(int(st.tgt_h), int(st.tgt_w))}，但 INFINITY_REQUIRE_TGT_HW={(req_h, req_w)}。请检查 h_div_w_template/dynamic_scale_schedule。",
                    )
            except HTTPException:
                raise
            except Exception:
                # 忽略格式错误的环境变量。
                pass

    # 如果客户端每次都发送完整 prefix，只保留新增 tail frames。
    st.last_req_prefix_mode = bool(getattr(req, "prefix_mode", False))
    if bool(st.last_req_prefix_mode):
        already = int(st.num_frames())
        if already > int(len(new_imgs)):
            raise HTTPException(
                status_code=400,
                detail=f"prefix_mode 要求 prefix 长度不能变短，但 server 已有 {already} 帧，本次请求只有 {len(new_imgs)} 帧",
            )
        new_imgs = new_imgs[already:]

    # 将新增帧按目标尺寸 transform 到 [-1,1]，并存到 CPU。
    for pil in new_imgs:
        if st.num_frames() >= int(cfg.infinity.num_frames):
            break
        if infinity_transform is None:
            raise HTTPException(status_code=500, detail="InfinityStar modules 尚未导入（请检查 INFINITY_REPO_ROOT）")
        fr = infinity_transform(pil, int(st.tgt_h), int(st.tgt_w))  # type: ignore[misc]  # [3,H,W] in [-1,1]
        st.frames_cpu.append(fr.cpu())

    n = st.num_frames()
    done = n >= int(cfg.infinity.num_frames)

    points = cfg.infinity.points()
    if len(points) < 2:
        raise HTTPException(status_code=500, detail=f"配置 points 错误：points={points}（num_frames={cfg.infinity.num_frames}, step={cfg.infinity.step}）")

    # 预热：首帧用于准备 first-frame conditioning。
    # 默认情况下预热请求不返回动作。
    # 如果开启 allow_future_segments，则继续执行，并可能只凭首帧立即发射 seg0 动作。
    if n == 1 and st.last_emitted_segment < 0:
        try:
            _prepare_firstframe_condition_if_needed(st)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"InfinityStar 预热失败：{e}")
        if not allow_future_segments:
            return PredictDeltaActionsResponse(
                actions=[],
                segment_index=-1,
                num_received_frames=n,
                prefix_latents=0,
                done=done,
                used_prompt=st.prompt_raw if req.debug else None,
            )

    next_seg = int(st.last_emitted_segment) + 1
    if next_seg >= (len(points) - 1):
        return PredictDeltaActionsResponse(
            actions=[],
            segment_index=-1,
            num_received_frames=n,
            prefix_latents=0,
            done=done,
            used_prompt=st.prompt_raw if req.debug else None,
        )

    # Segment 就绪条件：
    # 代码/形状说明：- points = [1, 1+step, 1+2*step, ..., num_frames]；
    # - 默认公式：ready_default = n >= points[seg+1]，
    #   即已收到帧数 n 到达当前 segment 的右边界时才发射该段动作；
    # - 默认：真实 prefix 需要达到 points[seg+1]（例如 49）才发射 seg2。
    # - 特例：最后一个 segment 的 prefix 只要达到 points[seg]（例如 33）即可发射，
    #   因为 frames 34..49 完全由模型预测。
    seg = int(next_seg)
    is_last_seg = int(seg) == (len(points) - 2)
    ready_default = n >= int(points[seg + 1])
    ready_last_future = bool(getattr(req, "allow_future_last_segment", False)) and is_last_seg and n >= int(points[seg])
    ready_future = bool(allow_future_segments) and n >= int(points[seg])
    if not (ready_default or ready_last_future or ready_future):
        return PredictDeltaActionsResponse(
            actions=[],
            segment_index=-1,
            num_received_frames=n,
            prefix_latents=0,
            done=done,
            used_prompt=st.prompt_raw if req.debug else None,
        )

    # prefix_latents_abs 是右边界 RGB 帧 points[seg+1] 对应的 latent 下标：
    # latent_index(frame f) = (f - 1)//temporal_compress_rate + 1，本 repo 通常 temporal_compress_rate=4。
    prefix_latents_abs = (int(points[seg + 1]) - 1) // 4 + 1

    # 选择 action head mode。
    mode = str(getattr(req, "action_head_mode", "tsformer_latent") or "tsformer_latent").strip().lower()
    if mode in ("", "default", "tsformer_latent"):
        env_mode = os.environ.get("ACTION_HEAD_MODE", "").strip().lower()
        if env_mode:
            mode = env_mode
    use_actionhead_ref_vit = mode in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead")

    # InfinityStar 闭环推理：生成 latents，并按需解码预测视频；
    # 在允许时，把 gt_obs cache 推进到新揭示的前缀（points[seg+1]）。
    try:
        # 只有确实拥有到 points[seg+1] 的真实帧时才推进 GT cache。
        # 如果带 future（预测）tail 发射，就不要把非真实帧写入 GT cache。
        advance_gt = bool(ready_default)
        base_seed = 0
        try:
            if getattr(req, "seed", None) is not None:
                base_seed = int(getattr(req, "seed"))
        except Exception:
            base_seed = 0
        infer_res = _infer_latents_for_actions_and_advance_cache(
            st,
            segment_index=seg,
            seed=int(base_seed),
            advance_gt_obs_to_next=advance_gt,
            need_pred_video=bool(use_actionhead_ref_vit),
        )
    except Exception as e:
        print("[Service] _infer_latents_for_actions_and_advance_cache 失败。")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"InfinityStar 推理失败：{e}")

    actions: List[List[float]] = []
    if use_actionhead_ref_vit:
        # ActionHead（reference-video）模式：解码预测视频 -> 4 帧滑窗 -> 逐帧动作。
        ckpt_path = os.environ.get("ACTIONHEAD_CKPT", "").strip() or os.environ.get("ACTIONHEAD_REF_CKPT", "").strip()
        run_cfg = os.environ.get("ACTIONHEAD_RUN_CONFIG", "").strip() or os.environ.get("ACTIONHEAD_REF_RUN_CONFIG", "").strip()
        if not ckpt_path or not run_cfg:
            raise HTTPException(
                status_code=500,
                detail="actionhead_ref_vit 需要环境变量 ACTIONHEAD_CKPT 和 ACTIONHEAD_RUN_CONFIG（或 ACTIONHEAD_REF_CKPT/ACTIONHEAD_REF_RUN_CONFIG）",
            )
        try:
            _init_actionhead_model(ckpt_path=ckpt_path, run_config_path=run_cfg)
            pred_vid = infer_res.pred_vid_bgr
            if pred_vid is None:
                # 尽力兜底；need_pred_video=True 时理论上不会发生。
                assert st.stream is not None
                with torch.no_grad():
                    pred_vid = st.stream.infinity.summed_codes2images(st.stream.vae, infer_res.summed_codes)
            # 重要：为了匹配 `predict_reference_videos_batch*.py`（window_size=4）的跨 clip 边界行为，
            # clip 开始前最多需要提供 (window_size-1)=3 帧历史。否则 clip 内最前面的几个
            # deltas 会因为平均次数不足而偏离离线脚本。
            #
            # 对 seg i（points=[1,17,33,49]）：
            # - 输出 transitions [obs_len->obs_len+1 .. next_obs_len-1->next_obs_len] 的动作，共 16 个。
            # - actionhead 输入帧取绝对帧 [ctx_start .. next_obs_len]，其中
            # 代码/形状说明：ctx_start = max(1, (obs_len+1) - 3) = max(1, obs_len-2)。
            obs_len = int(infer_res.obs_len)
            next_obs_len = int(infer_res.next_obs_len)
            clip_abs_start = int(obs_len) + 1
            clip_abs_end = int(next_obs_len)
            ctx_start_abs = max(1, int(clip_abs_start) - 3)

            frames_rgb: List["np.ndarray"] = []  # type: ignore[name-defined]
            for abs_i in range(int(ctx_start_abs), int(clip_abs_end) + 1):
                # 如果已有真实帧则优先使用；prefix observations 是真实帧，应和离线脚本一致。
                if 1 <= int(abs_i) <= int(st.num_frames()):
                    bgr = _frame_tensor_chw_neg1to1_to_bgr_uint8(st.frames_cpu[int(abs_i) - 1])
                else:
                    bgr = _slice_abs_frames_from_pred_video_bgr(
                        pred_vid,
                        abs_frame_start=int(abs_i),
                        abs_frame_end=int(abs_i),
                        infer_num_frames=int(infer_res.infer_num_frames),
                        total_num_frames=int(infer_res.total_num_frames),
                    )[0]
                frames_rgb.append(bgr[..., ::-1].copy())

            actions_all = _actionhead_ref_predict_actions_cm_deg(
                frames_rgb_uint8=frames_rgb,
                batch_size=int(getattr(req, "action_head_batch_size", 8) or 8),
                stride=int(getattr(req, "action_head_stride", 1) or 1),
                pre_resize_hw=int(getattr(req, "action_head_pre_resize_hw", 0) or 0),
            )

            # 精确切出该 clip 的 16 个动作。
            # frames_rgb 中的 transitions 连续；action index 对应 “to-frame” 位置减 1。
            start_idx = int(obs_len) - int(ctx_start_abs)
            end_idx = int(start_idx) + (int(clip_abs_end) - int(obs_len))
            actions = actions_all[int(start_idx) : int(end_idx)]
            if len(actions) != int(clip_abs_end) - int(obs_len):
                raise ValueError(f"actionhead 动作长度不匹配：实际 {len(actions)}，需要 {int(clip_abs_end)-int(obs_len)}")
        except HTTPException:
            raise
        except Exception as e:
            print("[Service] actionhead_ref_vit 推理失败。")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"actionhead_ref_vit 推理失败：{e}")
    else:
        # TSformer(P2P)：5 个 latents -> 4 个动作（cm/deg）。
        try:
            latent5 = infer_res.latent5_input
            actions_t = _tsformer_predict_actions_from_summed_codes(latent5, prefix_latents=int(latent5.shape[2]))  # (4,6)
            actions = actions_t.tolist()
        except Exception as e:
            print("[Service] TSformer 推理失败。")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"TSformer 推理失败：{e}")

    st.last_emitted_segment = seg
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("未安装 FastAPI/pydantic；服务模式不可用。")
    return PredictDeltaActionsResponse(  # type: ignore[name-defined]
        actions=actions,
        segment_index=seg,
        num_received_frames=n,
        prefix_latents=int(prefix_latents_abs),
        done=bool(done or ((ready_last_future or (ready_future and is_last_seg)) and is_last_seg)),
        used_prompt=st.prompt_raw if getattr(req, "debug", False) else None,
        segment_old_logprob=float(getattr(infer_res, "sample_logprob", 0.0)),
        segment_trace_path=getattr(infer_res, "trace_path", None),
    )


# -------------------------
# 8) 内部 self-test（不走 HTTP）
# -------------------------
def _self_test(
    *,
    infinity_ckpt: str,
    route_dir: str,
    ts_ckpt: str,
    ts_stats: str,
    prompt_key: str = "instruction",
) -> None:
    """用 route_dir 中的 meta/images 模拟真实 streaming 请求，验证端到端服务路径。"""
    route_dir = os.path.abspath(route_dir)
    meta_path = os.path.join(route_dir, "meta.json")
    images_dir = os.path.join(route_dir, "images")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到 meta.json：{meta_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"找不到 images 目录：{images_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = str(meta.get(prompt_key) or meta.get("instruction_unified") or meta.get("instruction") or meta.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"meta.json 中找不到 prompt（key={prompt_key}）")

    paths = _sorted_image_paths(images_dir)
    if not paths:
        raise FileNotFoundError("找不到任何图像")

    cfg0 = _get_server_config()
    num_frames = int(cfg0.infinity.num_frames)
    step = int(cfg0.infinity.step)
    # 为保证 deterministic test，把帧数 pad/trim 到配置的 num_frames。
    if len(paths) < num_frames:
        paths = paths + [paths[-1]] * (num_frames - len(paths))
    else:
        paths = paths[:num_frames]

    cfg = ServerConfig(
        infinity=InfinityConfig(**{**cfg0.infinity.__dict__, "ckpt": os.path.abspath(infinity_ckpt)}),
        tsformer=TSformerConfig(ckpt=os.path.abspath(ts_ckpt), stats=os.path.abspath(ts_stats)),
        infinity_repo_root=cfg0.infinity_repo_root,
    )
    _init_models(cfg=cfg)

    sid = f"selftest_{int(time.time())}"
    st = _get_or_create_traj(sid, prompt, "")

    # 模拟 streaming：先发 1 帧，然后按 `step` 分块，直到 num_frames。
    chunks: List[List[str]] = []
    chunks.append(paths[:1])
    idx = 1
    while idx < num_frames:
        chunks.append(paths[idx : min(num_frames, idx + step)])
        idx += step
    for i, ch in enumerate(chunks):
        imgs = [Image.open(p).convert("RGB") for p in ch]
        # 临时 base64 编码后复用内部实现，保持代码路径一致。
        b64s = []
        for pil in imgs:
            buf = BytesIO()
            pil.save(buf, format="PNG")
            b64s.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        req = PredictDeltaActionsRequest(session_id=sid, instruction=prompt if i == 0 else None, images_base64=b64s, debug=True)
        resp = _predict_delta_actions_impl(req)
        print(f"[自测] step={i} 已接收帧数={resp.num_received_frames} segment={resp.segment_index} prefix_latents={resp.prefix_latents} done={resp.done}")
        if resp.actions:
            print(f"[自测] 最近 4 个动作 = {resp.actions}")


def _init_tsformer_only(*, ts_ckpt: str, ts_stats: str) -> None:
    """只初始化 TSformer 权重/统计量，跳过 InfinityStar。"""
    global _ts_model, _ts_mean, _ts_std
    if _ts_model is not None:
        return
    ts_model, mean_t, std_t = _load_tsformer_p2p(
        ckpt_path=os.path.abspath(ts_ckpt),
        stats_path=os.path.abspath(ts_stats) if ts_stats else "",
        device=_DEVICE,
    )
    _ts_model, _ts_mean, _ts_std = ts_model, mean_t, std_t


def _integrate_relative_pose_points(actions_cm_deg: List[List[float]]) -> Dict[str, object]:
    """
    `actions` 是 6D deltas 列表，单位为 (cm, deg)。这里用简单加法积分得到相对 pose。

    返回：
    - 起点位姿 `start_pose`: [0,0,0,0,0,0]
    - `poses`: 长度等于 len(actions)，表示每个动作后的 pose。
    - `final_pose`: 最终位姿。
    """
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    poses = []
    for a in actions_cm_deg:
        if len(a) != 6:
            raise ValueError(f"action 维度必须是 6，实际为 {len(a)}")
        pose = [pose[i] + float(a[i]) for i in range(6)]
        poses.append(pose)
    return {"start_pose": [0.0] * 6, "poses": poses, "final_pose": poses[-1] if poses else [0.0] * 6}


def _offline_eval_from_precomputed_summed_codes(
    *,
    route_dir: str,
    ts_ckpt: str,
    ts_stats: str,
    out_dir: str,
    prompt_key: str = "instruction",
    take_first_pixel_frames: Optional[int] = None,
) -> str:
    """
    不加载 InfinityStar 权重的离线评估：
    - 从 route_dir 加载 meta.json 和 video_summed_codes.npy。
    - 切出与前 `take_first_pixel_frames` 帧匹配的 summed_codes（pt = (N-1)//4 + 1）。
    - 生成 20 个动作（5 segments * 4 actions）及其积分 pose。
    - 在 out_dir 下写两个 json 文件。
    返回输出目录路径。
    """
    route_dir = os.path.abspath(route_dir)
    meta_path = os.path.join(route_dir, "meta.json")
    summed_path = os.path.join(route_dir, "reshape_actionhead_data", "video_summed_codes.npy")
    images_dir = os.path.join(route_dir, "images")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到 meta.json：{meta_path}")
    if not os.path.exists(summed_path):
        raise FileNotFoundError(f"找不到 video_summed_codes.npy：{summed_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"找不到 images 目录：{images_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = str(meta.get(prompt_key) or meta.get("instruction_unified") or meta.get("instruction") or meta.get("prompt") or "").strip()

    cfg0 = _get_server_config()
    if take_first_pixel_frames is None:
        take_first_pixel_frames = int(cfg0.infinity.num_frames)

    # 即使这里不运行 Infinity，也要确保有 >=N 张图以符合“流式”语义。
    img_paths = _sorted_image_paths(images_dir)
    if len(img_paths) < int(take_first_pixel_frames):
        raise ValueError(f"本次离线评估至少需要 {take_first_pixel_frames} 张图像，实际为 {len(img_paths)}")

    import numpy as np

    z = np.load(summed_path)  # 中文说明：期望形状为 (1,16,T_lat,H,W)
    if z.ndim != 5 or z.shape[0] != 1 or z.shape[1] != 16:
        raise ValueError(f"summed_codes 形状不符合预期：{z.shape}（期望 (1,16,T,H,W)）")

    # 按前 N 个像素帧切到 pt：
    # latent_index(frame f) = (f - 1)//temporal_compress_rate + 1，这里 temporal_compress_rate=4。
    pt = (int(take_first_pixel_frames) - 1) // 4 + 1
    if z.shape[2] < pt:
        raise ValueError(f"summed_codes 时间长度太短：T_lat={z.shape[2]}，但需要 pt={pt}")
    z = z[:, :, :pt]

    summed_codes = torch.from_numpy(z).to(_DEVICE, dtype=torch.float32)

    _init_tsformer_only(ts_ckpt=ts_ckpt, ts_stats=ts_stats)

    # 按 points(num_frames, step) 模拟多个 segment。
    all_actions: List[List[float]] = []
    points = _obs_points(pred_num_frames=int(take_first_pixel_frames), step=int(cfg0.infinity.step))
    for seg in range(len(points) - 1):
        # 每个 segment 的右边界 points[seg+1] 也用同一条 frame->latent 公式换成 latent 右边界。
        abs_end_lat = (int(points[seg + 1]) - 1) // 4 + 1
        abs_start_lat = max(1, int(abs_end_lat) - 4)
        z5 = summed_codes[:, :, (abs_start_lat - 1) : abs_end_lat].contiguous()
        a4 = _tsformer_predict_actions_from_summed_codes(z5, prefix_latents=int(z5.shape[2])).tolist()
        all_actions.extend(a4)
    poses_info = _integrate_relative_pose_points(all_actions)

    route_id = os.path.basename(route_dir.rstrip("/"))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_run = os.path.join(out_dir, f"offline_eval_{route_id}_{int(time.time())}")
    os.makedirs(out_run, exist_ok=True)

    actions_json = {
        "route_id": route_id,
        "route_dir": route_dir,
        "prompt": prompt,
        "take_first_pixel_frames": int(take_first_pixel_frames),
        "pt_used": int(pt),
        "points": points,
        "ts_ckpt": os.path.abspath(ts_ckpt),
        "ts_stats": os.path.abspath(ts_stats) if ts_stats else "",
        "units": {"translation": "cm", "angles": "deg"},
        "actions": all_actions,
        "num_actions": int(len(all_actions)),
    }
    poses_json = {
        "route_id": route_id,
        "units": {"translation": "cm", "angles": "deg"},
        # poses 长度等于 num_actions（每个动作后的 pose）；start_pose 单独保留。
        **poses_info,
        "note": "poses 长度等于 num_actions（每个动作后的 pose）；start_pose 单独提供。",
    }

    with open(os.path.join(out_run, "actions.json"), "w", encoding="utf-8") as f:
        json.dump(actions_json, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_run, "relative_poses.json"), "w", encoding="utf-8") as f:
        json.dump(poses_json, f, ensure_ascii=False, indent=2)

    return out_run


def main():
    """命令行入口：运行 HTTP 外的自测或 precomputed latent 离线评估。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--offline_eval_precomputed", action="store_true", help="使用 route_dir/reshape_actionhead_data/video_summed_codes.npy 做离线评估（不需要 InfinityStar 权重）。")
    ap.add_argument("--infinity_ckpt", type=str, default=os.environ.get("INFINITY_CKPT", ""))
    ap.add_argument("--route_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "cache"), help="离线评估 JSON 文件的输出目录。")
    ap.add_argument("--ts_ckpt", type=str, default=DEFAULT_TS_CKPT)
    ap.add_argument("--ts_stats", type=str, default=DEFAULT_TS_STATS)
    ap.add_argument("--prompt_key", type=str, default="instruction")
    args = ap.parse_args()

    if args.self_test:
        if not args.infinity_ckpt:
            raise SystemExit("--self_test 需要 --infinity_ckpt 或环境变量 INFINITY_CKPT")
        if not args.route_dir:
            raise SystemExit("--self_test 需要 --route_dir")
        _self_test(
            infinity_ckpt=args.infinity_ckpt,
            route_dir=args.route_dir,
            ts_ckpt=args.ts_ckpt,
            ts_stats=args.ts_stats,
            prompt_key=args.prompt_key,
        )
    elif args.offline_eval_precomputed:
        if not args.route_dir:
            raise SystemExit("--offline_eval_precomputed 需要 --route_dir")
        out_run = _offline_eval_from_precomputed_summed_codes(
            route_dir=args.route_dir,
            ts_ckpt=args.ts_ckpt,
            ts_stats=args.ts_stats,
            out_dir=args.out_dir,
            prompt_key=args.prompt_key,
        )
        print(f"[offline_eval_precomputed] 已写入 json 文件到：{out_run}")


if __name__ == "__main__":
    main()
