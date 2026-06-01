#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WorldVLN 在线推理 API 服务端：权重常驻内存/GPU，按流式方式处理一条轨迹。

这里的 stage2 指的是 Stage-2 latent2action 推理阶段：先由世界模型产出 summed_codes / latent，再由 Stage-2 动作头把这些 latent 解码成一段动作序列。默认服务走的是 “Stage2 latent2action” 路径；建议用 `rg "def _predict_delta_actions_impl|def _stage2_predict_16_actions_for_segment_cm_deg" infer/server.py` 定位当前入口。

更准确地说，这个文件是默认的在线闭环推理服务。它的主链路可以理解成：真实图像/指令 -> 世界模型 latent -> stage2 latent2action -> 16 个动作

如果要看强化学习 / GRPO 流程，它是单独拆出去的，不是这个 infer/server.py。对应入口是 `action_aware_grpo/grpo_server.py`，文档里也明确区分了默认 infer/ 服务端和 GRPO rollout 服务端

小白导读：
- 这个文件是理解 WorldVLN 在线闭环推理的首选入口。客户端每次只上传当前真实观测段，
  服务端用 `session_id` 找到同一条轨迹的历史状态，再预测下一小段 latent 世界转移。
- 预测出的 `summed_codes` 不是为了直接展示成视频，而是交给 Stage-2 latent-to-action
  流程解码为 6D 航点动作。
- 每个 segment 执行动作后，下一次请求会上传真实新帧；服务端会把真实观测重新写入
  `gt_obs` cache，覆盖上一轮预测带来的不确定性。这是“预测服务当前动作、下一轮回到真实观测”
  的闭环协议。

运行示例：
  环境变量示例：export INFINITY_CKPT=./checkpoints/infinity/global_step_xxx.pth
  说明：uvicorn server:app --host 0.0.0.0 --port 8002

自测示例（需要真实 checkpoint 和 route_dir）：
  说明：python3 server.py --self_test     --infinity_ckpt "$INFINITY_CKPT"     --route_dir /path/to/route_dir
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

# 可选依赖：只有 actionhead reference-video 模式需要 numpy。
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

# 可选服务端依赖：没有 fastapi/pydantic 时，离线评估仍可 import 本文件。
FASTAPI_AVAILABLE = True
try:
    from fastapi import FastAPI, HTTPException  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
except Exception:
    FASTAPI_AVAILABLE = False

    class HTTPException(RuntimeError):  # 最小占位实现，用于缺少可选依赖时保持导入可用。
        """FastAPI 不可用时的最小异常替身，让离线工具仍能 import 本文件。"""

        def __init__(self, status_code: int = 500, detail: str = ""):
            """保存 HTTP 状态码和错误详情，行为尽量接近 FastAPI.HTTPException。"""
            super().__init__(f"HTTP {status_code}: {detail}")
            self.status_code = status_code
            self.detail = detail

    class BaseModel:  # 最小占位实现，用于缺少可选依赖时保持导入可用。
        """Pydantic 不可用时的占位基类；仅用于让类型声明不阻塞离线运行。"""
        pass

    def Field(default=None, **kwargs):  # noqa: N802
        """Pydantic Field 的占位函数，离线模式下直接返回默认值。"""
        return default


# -------------------------
# 中文说明：0) 路径 / sys.path
# -------------------------
ROOT = Path(__file__).resolve().parent
REPO = ROOT
PKG_ROOT = ROOT.parent

TSFORMER_ROOT = PKG_ROOT / "Worldmodel" / "action_decoder" / "actionhead_runtime"

if not TSFORMER_ROOT.exists():
    raise FileNotFoundError(f"找不到 TSformer repo: {TSFORMER_ROOT}")

# 把 TSformer runtime 加入 import 路径，供动作头加载使用。
sys.path.insert(0, str(TSFORMER_ROOT))

# -------------------------
# 中文说明：1) InfinityStar 动态导入（支持 INFINITY_REPO_ROOT）
# -------------------------
# 运行时选择 Worldmodel runtime 根目录，便于同一服务切换不同 runtime 副本。
# 开源目录默认把它放在仓库根目录的 Worldmodel/runtime 下。
DEFAULT_INFINITY_REPO_ROOT = PKG_ROOT / "Worldmodel" / "runtime"

# 这些全局符号由 `_import_infinity_modules()` 动态填充。
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
    """解析 runtime 版 InfinityStar 代码根目录，优先使用 INFINITY_REPO_ROOT。"""
    p = os.environ.get("INFINITY_REPO_ROOT", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return DEFAULT_INFINITY_REPO_ROOT


def _import_infinity_modules(repo_root: Path) -> None:
    """
    从指定 `repo_root` 动态导入 InfinityStar runtime 模块。

    这样做的原因是：服务端可能通过 `INFINITY_REPO_ROOT` 切换不同 Worldmodel runtime；
    所有 Infinity 相关全局符号必须在调用本函数后才可使用。
    """
    global InfinityStreamingSession, SelfCorrection, get_dynamic_resolution_meta
    global _make_infinity_args, load_tokenizer, load_transformer, load_visual_tokenizer, infinity_transform, infinity_save_video, infinity_gen_one_example

    if InfinityStreamingSession is not None:
        return
    if not repo_root.exists():
        raise FileNotFoundError(f"找不到 InfinityStar repo: {repo_root}")
    # 把本次选择的 runtime 放到 import 搜索路径最前面，避免导入到其它副本。
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
# 中文说明：2) TSformer(P2P) imports
# -------------------------
# 旧版 P2P 模型只给本文件里的离线工具使用。
# HTTP 服务默认走 Stage2 latent2action 时不需要它，所以在 `_load_tsformer_p2p()` 里懒加载，
# 避免启动服务时强依赖 fvcore 等旧实验依赖。


# -------------------------
# 3) 配置：环境变量默认值
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

# 默认配置文件位置；可用 INFINITY_SERVER_CONFIG 覆盖。
DEFAULT_SERVER_CONFIG_JSON = str((ROOT / "config.json").resolve())


def _obs_points(pred_num_frames: int, step: int) -> List[int]:
    """
    生成闭环 segment 边界。

    闭环 segment 边界公式：
    - 公式：points = [1, 1+step, 1+2*step, ..., num_frames]
    例如 `pred_num_frames=49, step=16` 时返回 `[1,17,33,49]`：
    第 1 帧是预热观测，之后每 16 帧为一个动作执行片段。
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
    """世界模型推理配置：checkpoint、帧数、segment step、采样参数和闭环 cache 策略。"""

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

    # 闭环/rolling-tail 控制项，需要和 batch_closed_loop_streaming_infer_routes.py 保持一致。
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
        """返回 RGB 帧时间线上的 segment 边界，例如 `[1,17,33,49]`。"""
        return _obs_points(pred_num_frames=int(self.num_frames), step=int(self.step))

    def pt_total(self) -> int:
        """返回整条 RGB 序列对应的 latent 时间长度，默认 video VAE 时间压缩率为 4。"""
        # 帧号 f 映射到 latent 下标：latent_index(f) = (f - 1)//temporal_compress_rate + 1。
        # 本 repo 通常 temporal_compress_rate=4，所以总帧数 num_frames 对应下面这个 pt。
        return (int(self.num_frames) - 1) // 4 + 1


@dataclass
class TSformerConfig:
    """旧 TSformer(P2P) 动作头配置，主要保留给兼容路径和离线自测。"""

    ckpt: str = DEFAULT_TS_CKPT
    stats: str = DEFAULT_TS_STATS


@dataclass
class ServerConfig:
    """服务端总配置：世界模型、旧动作头和 InfinityStar runtime 根目录。"""

    infinity: InfinityConfig = field(default_factory=InfinityConfig)
    tsformer: TSformerConfig = field(default_factory=TSformerConfig)
    infinity_repo_root: Path = field(default_factory=_get_infinity_repo_root)


_SRV_CFG: Optional[ServerConfig] = None


def _load_server_config_from_json(path: str) -> ServerConfig:
    """从 config.json 读取服务配置，并兼容顶层即 infinity 配置的旧格式。"""
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
    """懒加载服务配置；环境变量可覆盖 checkpoint/stats 路径。"""
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

    # 向后兼容：如果配置文件未填写 ckpt 路径，允许环境变量覆盖。
    env_inf_ckpt = os.environ.get("INFINITY_CKPT", "").strip()
    if env_inf_ckpt and not cfg.infinity.ckpt:
        cfg.infinity.ckpt = env_inf_ckpt
    env_ts_ckpt = os.environ.get("TS_P2P_CKPT", "").strip()
    if env_ts_ckpt:
        cfg.tsformer.ckpt = env_ts_ckpt
    env_ts_stats = os.environ.get("TS_P2P_STATS", "").strip()
    if env_ts_stats:
        cfg.tsformer.stats = env_ts_stats

    _SRV_CFG = cfg
    return cfg


# -------------------------
# 中文说明：4) Utilities
# -------------------------
_DATA_URL_SPLIT_RE = re.compile(r"^data:image/[^;]+;base64,", flags=re.IGNORECASE)


def _load_image_from_base64(s: str) -> Image.Image:
    """把客户端上传的 base64/data URL 图像解码成 RGB PIL Image。"""
    if not isinstance(s, str) or not s.strip():
        raise ValueError("图像字符串为空")
    b64 = _DATA_URL_SPLIT_RE.sub("", s.strip())
    raw = base64.b64decode(b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _sorted_image_paths(images_dir: str) -> List[str]:
    """按文件名排序列出目录中的常见图片文件，用于自测或离线评估。"""
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    names = [n for n in os.listdir(images_dir) if n.lower().endswith(exts)]
    names.sort()
    return [os.path.join(images_dir, n) for n in names]


def _to_cm_deg(deltas_m_rad: torch.Tensor) -> torch.Tensor:
    """
    输入 `deltas` 形状：`[..., 6] = [dx,dy,dz,droll,dyaw,dpitch]`，
    单位是 `(m, rad)`；输出会转换成 `(cm, deg)`。

    单位换算公式：
    - 公式：meters -> cm: *100（前 3 维平移，索引 0:3）
    - 公式：radians -> degrees: *180/pi（后 3 维角度，索引 3:6）
    """
    out = deltas_m_rad.clone()
    out[..., 0:3] = out[..., 0:3] * 100.0
    out[..., 3:6] = out[..., 3:6] * (180.0 / math.pi)
    return out


def _prompt_with_duration(prompt: str, *, num_frames: int, fps: int, append_tag: bool = True) -> str:
    """按训练约定可选地给 prompt 添加 `<<<t=...s>>>` 时长标签。"""
    if not append_tag:
        return prompt
    dur_s = (int(num_frames) - 1) // max(1, int(fps))
    return f"<<<t={dur_s}s>>>{prompt}"


# -------------------------
# 5) 模型持有区：权重只加载一次，后续请求复用同一份对象。
# -------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.bfloat16 if (_DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16 if _DEVICE == "cuda" else torch.float32

_infinity_args = None
_infinity_session_template: Optional[InfinityStreamingSession] = None
_infinity_self_correction: Optional[SelfCorrection] = None

# Stage-2 latent2action：VAE decoder 中间特征 -> Adapter token -> TimesFormer 滑窗动作。
# 这是当前 `tsformer_latent` 默认路径，替代旧的“TSformer(P2P) 5 个 latent -> 4 个动作”逻辑。
DEFAULT_STAGE2_LATENT2ACTION_CKPT = os.environ.get(
    "STAGE2_LATENT2ACTION_CKPT",
    str((ROOT / "checkpoints" / "stage2_latent2action_combined.pt").resolve()),
).strip()
STAGE2_REPO_ROOT = (PKG_ROOT / "Worldmodel" / "action_decoder" / "src").resolve()
_S2_WINDOW_SIZE = 4
_S2_W_GRID = 40  # 与 Stage2 latent2action 的训练/推理配置保持一致。

_s2_tsformer: Optional[torch.nn.Module] = None
_s2_adapter: Optional[torch.nn.Module] = None
_s2_vae: Optional[torch.nn.Module] = None
_s2_label_stats: Optional[Dict[str, torch.Tensor]] = None  # mean/std 放到当前设备上。
_s2_ckpt_path: Optional[str] = None

_ts_model: Optional[torch.nn.Module] = None
_ts_mean: Optional[torch.Tensor] = None
_ts_std: Optional[torch.Tensor] = None

# ActionHead reference-video 可选模式：
# - 输入：4 帧 RGB 窗口，stride=1，并把重叠窗口平均成逐帧 delta；
# - 输出：逐帧 6D delta，再转换成 API 使用的 cm/deg。
_ah_vit_cls = None  # type: ignore
_ah_model: Optional[torch.nn.Module] = None
_ah_stats: Optional[Dict[str, "np.ndarray"]] = None  # type: ignore[name-defined]
_ah_preprocess = None  # type: ignore
_AH_KITTI_MEAN = [0.34721234, 0.36705238, 0.36066107]
_AH_KITTI_STD = [0.30737526, 0.31515116, 0.32020183]
_AH_TARGET_H = 192
_AH_TARGET_W = 640

DEFAULT_ACTIONHEAD_REPO_ROOT = PKG_ROOT / "Worldmodel" / "action_decoder" / "actionhead_runtime"


def _get_actionhead_repo_root() -> Path:
    """解析参考视频动作头代码根目录，优先使用 ACTIONHEAD_REPO_ROOT。"""
    p = os.environ.get("ACTIONHEAD_REPO_ROOT", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return DEFAULT_ACTIONHEAD_REPO_ROOT


def _import_actionhead_modules(repo_root: Path) -> None:
    """
    为 reference-video 动作头导入 TimesFormer VisionTransformer。

    注意：这里故意不导入任何 `datasets.*` 模块，因为 latent TSformer 代码库和
    actionhead runtime 代码库都有顶层 `datasets` 包，提前导入会造成包名冲突。
    """
    global _ah_vit_cls, _ah_preprocess
    if _ah_vit_cls is not None and _ah_preprocess is not None:
        return
    if np is None:
        raise RuntimeError("actionhead 模式需要 numpy")
    if not repo_root.exists():
        raise FileNotFoundError(f"找不到 ActionHead repo: {repo_root}")
    if str(repo_root) not in sys.path:
        # 追加到 sys.path 末尾，尽量减少导入遮蔽。
        sys.path.append(str(repo_root))
    try:
        from torchvision import transforms as T  # type: ignore
    except Exception as e:
        raise RuntimeError(f"actionhead 模式需要 torchvision：{e}")
    from timesformer.models.vit import VisionTransformer  # type: ignore

    _ah_vit_cls = VisionTransformer
    # 中文说明：对齐 predict_reference_videos_batch copy.py 的预处理。
    # 代码/形状说明：ToPILImage -> Resize((H,W)) -> ToTensor -> Normalize（不 crop）
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
    从环境变量读取默认动作头模式。

    数据流位置：
    server 启动初始化模型时会调用这里，决定是否必须提前加载 TSformer(P2P)；
    如果默认走 Stage-2 latent2action，则旧 P2P 头可以延后或跳过加载。
    """
    return os.environ.get("ACTION_HEAD_MODE", "").strip().lower()


def _use_actionhead_ref_mode_by_default() -> bool:
    """判断默认动作头模式是否需要启动时预加载 reference-video actionhead。"""
    mode = _default_action_head_mode()
    return mode in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead")


def _load_actionhead_stats(run_config_path: str) -> Dict[str, "np.ndarray"]:  # type: ignore[name-defined]
    """读取 reference-video actionhead 训练时保存的动作标签均值/方差。"""
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
    """初始化旧的 RGB/reference-video TimesFormer 动作头，按需懒加载到全局变量。"""
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

# 并发控制：单进程测试阶段只允许一个 GPU pipeline 进入。
try:
    import asyncio

    _LOCK: "asyncio.Lock" = asyncio.Lock()
except Exception:
    _LOCK = None  # type: ignore


def _safe_torch_load_any(path: str) -> object:
    """
    兼容 PyTorch 2.6 `torch.load` 的 `weights_only=True` 默认行为。

    背景：
    combined checkpoint 里可能包含 numpy 对象，`weights_only=True` 会加载失败。
    本函数优先尝试安全的 `weights_only=True`，必要时才回退到 `weights_only=False`。
    这个回退只应对可信 checkpoint 使用。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _is_safetensors_shard_dir(path: str | Path) -> bool:
    """判断目录是否包含 safetensors 权重或 index，可用于加载 HF 发布格式。"""
    path = Path(path)
    if not path.is_dir():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.safetensors.index.json"))


def _nested_dict_get(obj: object, dotted_path: str) -> object | None:
    """从嵌套 dict 中按 dotted path 取 state_dict，如 `trainer.gpt_fsdp`。"""
    current = obj
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _load_state_dict_from_safetensors_dir(path: str | Path, device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    """读取单目录 sharded safetensors，并补回 metadata 中的 alias tensor。"""
    from safetensors import safe_open

    path = Path(path).expanduser().resolve()
    target_device = str(device)
    state_dict: Dict[str, torch.Tensor] = {}
    alias_map: Dict[str, str] = {}

    def _merge_alias_metadata(metadata: object) -> None:
        """解析 safetensors/index metadata 中的 alias_key -> canonical_key 映射。"""
        if not isinstance(metadata, dict):
            return
        for key, value in metadata.items():
            if key == "format":
                continue
            if isinstance(key, str) and isinstance(value, str):
                alias_map[key] = value

    index_files = sorted(path.glob("*.safetensors.index.json"))
    if index_files:
        index_data = json.loads(index_files[0].read_text())
        _merge_alias_metadata(index_data.get("metadata"))
        shard_names = list(dict.fromkeys(index_data.get("weight_map", {}).values()))
        for shard_name in shard_names:
            shard_path = path / shard_name
            with safe_open(str(shard_path), framework="pt", device=target_device) as handle:
                _merge_alias_metadata(handle.metadata())
                for key in handle.keys():
                    state_dict[key] = handle.get_tensor(key)
    else:
        safetensors_files = sorted(path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"在 {path} 中找不到 .safetensors 文件")
        for shard_path in safetensors_files:
            with safe_open(str(shard_path), framework="pt", device=target_device) as handle:
                _merge_alias_metadata(handle.metadata())
                for key in handle.keys():
                    state_dict[key] = handle.get_tensor(key)

    for alias_key, canonical_key in alias_map.items():
        if alias_key not in state_dict and canonical_key in state_dict:
            state_dict[alias_key] = state_dict[canonical_key]
    return state_dict


def _load_model_weights_from_path(
    model: torch.nn.Module,
    *,
    path: str | Path,
    label: str,
    preferred_state_dict_paths: Tuple[str, ...] = (),
    strict: bool = False,
) -> None:
    """
    统一加载普通 .pth、嵌套 checkpoint 和 sharded safetensors 目录。

    训练导出的 GPT/VAE 可能是 `trainer.*` 嵌套结构，也可能是 HF 风格分片目录；
    这个函数把这些格式差异收敛到一次 `model.load_state_dict()`。
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到 {label} 权重: {path}")

    if _is_safetensors_shard_dir(path):
        print(f"[{label}] 正在从 {path} 加载分片权重")
        state_dict = _load_state_dict_from_safetensors_dir(path, device="cpu")
        result = model.load_state_dict(state_dict, strict=strict)
        if result is not None:
            missing, unexpected = result
            print(f"[{label}] 分片权重以 strict={strict} 加载：missing={len(missing)} unexpected={len(unexpected)}")
        return

    loaded = _safe_torch_load_any(str(path))
    state_dict = None
    for dotted_path in preferred_state_dict_paths:
        candidate = _nested_dict_get(loaded, dotted_path)
        if isinstance(candidate, dict) and len(candidate) > 0:
            state_dict = candidate
            break
    if state_dict is None and isinstance(loaded, dict):
        state_dict = loaded
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise ValueError(f"[{label}] 无法在 {path} 中定位 state_dict")

    result = model.load_state_dict(state_dict, strict=strict)
    if result is not None:
        missing, unexpected = result
        print(f"[{label}] 以 strict={strict} 调用 load_state_dict：missing={len(missing)} unexpected={len(unexpected)}")


def _purge_sysmodules(pkg: str) -> None:
    """从 `sys.modules` 删除指定包及其子模块，避免导入到错误的仓库副本。"""
    for k in list(sys.modules.keys()):
        if k == pkg or k.startswith(pkg + "."):
            try:
                del sys.modules[k]
            except Exception:
                pass


def _ensure_infinity_repo_on_syspath() -> Path:
    """
    确保当前选择的 InfinityStar 仓库可以作为 Python package root 被导入。

    数据流位置：
    Stage-2 动作头需要调用 `infinity.models...` 里的 VAE 构造函数；如果这里
    没有先把 repo root 插入 `sys.path`，后面的 `_build_stage2_infinity_vae_from_ckpt()`
    会导入失败，整个 `latent -> action` 分支也无法启动。
    """
    repo_root = _get_infinity_repo_root()
    if not repo_root.exists():
        raise FileNotFoundError(f"找不到 InfinityStar repo: {repo_root}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _build_stage2_infinity_vae_from_ckpt(ckpt: Dict[str, object]) -> torch.nn.Module:
    """
    构建 Stage-2 latent2action 使用的 InfinityStar VAE。

    输入流入：
    `ckpt` 来自动作头 combined checkpoint，里面可能带有 `args`、VAE 路径、
    VAE 类型和可选 `vae_state_dict`。

    输出流出：
    返回冻结后的 VAE decoder。后续 `_stage2_decode_tokens_tnd()` 会把世界模型
    预测的 `summed_codes` 送进这个 VAE，截取 decoder 中间 feature，再交给
    Adapter/TimesFormer 预测 16 个动作。
    """
    _ensure_infinity_repo_on_syspath()

    ckpt_args = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
    vae_path = (
        str(ckpt.get("infinitystar_vae_path", "")).strip()
        or str(ckpt_args.get("infinitystar_vae_path", "")).strip()
        or str((DEFAULT_INFINITY_REPO_ROOT / "checkpoint" / "infinitystar_videovae.pth").resolve())
    )
    vae_type = int(ckpt.get("infinitystar_vae_type", ckpt_args.get("infinitystar_vae_type", 64)))

    # 从 InfinityStar runtime 导入 VAE 构建函数。
    # 有些 InfinityStar 分支和特定 torch 版本耦合较紧，例如引用 torch._dynamo 的内部异常名；
    # 如果这里导入失败，并且 streaming VAE 已经加载，则退回复用 streaming VAE。
    try:
        from types import SimpleNamespace

        from infinity.models.videovae.models.load_vae_bsq_wan_absorb_patchify import (  # type: ignore
            video_vae_model,
        )
    except Exception as e:
        if _infinity_session_template is not None and getattr(_infinity_session_template, "vae", None) is not None:
            print(f"[Stage2] 警告：从运行时代码库导入/构建 InfinityStar VAE 失败，回退复用已加载的 streaming VAE：{e}")
            vae = _infinity_session_template.vae  # type: ignore[assignment]
            vae.eval()
            for p in vae.parameters():
                p.requires_grad_(False)
            # 尽力加载 checkpoint 中的 vae_state_dict；失败不阻塞兜底路径。
            vae_sd = ckpt.get("vae_state_dict")
            if isinstance(vae_sd, dict) and len(vae_sd) > 0:
                try:
                    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
                    if missing or unexpected:
                        print(f"[Stage2] streaming VAE 以 strict=False 加载：missing={len(missing)} unexpected={len(unexpected)}")
                except Exception:
                    pass
            return vae
        raise RuntimeError(f"导入 Stage2 InfinityStar VAE 构建器失败: {e}")

    global_args = SimpleNamespace(
        semantic_scale_dim=int(ckpt_args.get("semantic_scale_dim", 16)),
        detail_scale_dim=int(ckpt_args.get("detail_scale_dim", 64)),
        use_learnable_dim_proj=int(ckpt_args.get("use_learnable_dim_proj", 0)),
        detail_scale_min_tokens=int(ckpt_args.get("detail_scale_min_tokens", 350)),
        use_feat_proj=int(ckpt_args.get("use_feat_proj", 2)),
        semantic_scales=int(ckpt_args.get("semantic_scales", 11)),
    )

    device = torch.device("cuda" if _DEVICE == "cuda" else "cpu")
    vae = video_vae_model(
        vqgan_ckpt=str(vae_path),
        schedule_mode="dynamic",
        codebook_dim=int(vae_type),
        global_args=global_args,
        test_mode=True,
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # 如果 Stage-2 checkpoint 内包含 VAE 权重，则加载它；否则保留 VAE 构造时的权重。
    vae_sd = ckpt.get("vae_state_dict")
    if isinstance(vae_sd, dict) and len(vae_sd) > 0:
        missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
        if missing or unexpected:
            print(f"[Stage2] VAE 以 strict=False 加载：missing={len(missing)} unexpected={len(unexpected)}")
    return vae


def _ensure_stage2_imports() -> Tuple[object, object]:
    """
    确保从 `STAGE2_REPO_ROOT` 导入 Stage-2 TimesFormer 和 Adapter。

    关键背景：仓库里有多个 TSformer 副本，都可能定义顶层 `timesformer` 和 `models` 包。
    如果 sys.path 中已有旧副本，Python 会复用错误模块。因此这里会清理冲突模块，强制导入
    含有 `forward_features_from_patch_tokens` 的 Stage-2 版本。
    """
    if not STAGE2_REPO_ROOT.exists():
        raise FileNotFoundError(f"找不到 Stage2 TSformer repo: {STAGE2_REPO_ROOT}")
    if str(STAGE2_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(STAGE2_REPO_ROOT))

    # 如果 `timesformer` / `models` 已经从其它副本导入，先清掉再从 Stage-2 根目录解析。
    tm = sys.modules.get("timesformer")
    if tm is not None:
        f = str(getattr(tm, "__file__", "") or "")
        if f and str(STAGE2_REPO_ROOT) not in f:
            _purge_sysmodules("timesformer")
    mm = sys.modules.get("models")
    if mm is not None:
        f = str(getattr(mm, "__file__", "") or "")
        if f and str(STAGE2_REPO_ROOT) not in f:
            _purge_sysmodules("models")

    # 只导入必要符号，避免导入 datasets.* 引发包名冲突。
    from timesformer.models.vit import VisionTransformer  # type: ignore
    from models.vae96_to_tsformer_adapter import Vae96ToTSformerEmbedAdapter  # type: ignore

    if not hasattr(VisionTransformer, "forward_features_from_patch_tokens"):
        raise RuntimeError(
            "导入的 VisionTransformer 缺少 forward_features_from_patch_tokens；"
            "可能是另一个 TSformer 副本遮蔽了 Stage2 repo。"
            f"STAGE2_REPO_ROOT={STAGE2_REPO_ROOT}"
        )
    return VisionTransformer, Vae96ToTSformerEmbedAdapter


def _init_stage2_latent2action_models(*, ckpt_path: str) -> None:
    """
    初始化当前默认的 Stage-2 latent-to-action 动作头。

    加载内容包括 TimesFormer、VAE-to-TSFormer Adapter、独立 Stage-2 VAE 和可选 label_stats。
    它与 streaming world-model VAE 分开持有，避免不同训练配置的 VAE 结构混用。
    """
    global _s2_tsformer, _s2_adapter, _s2_vae, _s2_label_stats, _s2_ckpt_path
    if _s2_tsformer is not None and _s2_adapter is not None and _s2_ckpt_path == str(ckpt_path):
        return

    ckpt_path = os.path.abspath(str(ckpt_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 Stage2 checkpoint: {ckpt_path}")

    VisionTransformer, Vae96ToTSformerEmbedAdapter = _ensure_stage2_imports()
    import torch.nn as nn
    from functools import partial

    ckpt = _safe_torch_load_any(ckpt_path)
    if not isinstance(ckpt, dict):
        raise ValueError("Stage2 checkpoint 必须是 dict（组合 checkpoint）")

    ts_sd = ckpt.get("model_state_dict") or ckpt.get("tsformer_state_dict")
    ad_sd = ckpt.get("adapter_state_dict") or ckpt.get("state_dict")
    if not isinstance(ts_sd, dict) or not isinstance(ad_sd, dict):
        raise ValueError("Stage2 checkpoint 缺少 model_state_dict/adapter_state_dict（或支持的别名）")

    device = torch.device("cuda" if _DEVICE == "cuda" else "cpu")
    tsformer = VisionTransformer(
        img_size=(192, 640),
        num_classes=18,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        num_frames=int(_S2_WINDOW_SIZE),
        attention_type="divided_space_time",
    ).to(device).eval()
    adapter = Vae96ToTSformerEmbedAdapter().to(device).eval()

    missing, unexpected = tsformer.load_state_dict(ts_sd, strict=False)
    if missing or unexpected:
        print(f"[Stage2] TSFormer 以 strict=False 加载：missing={len(missing)} unexpected={len(unexpected)}")
    missing, unexpected = adapter.load_state_dict(ad_sd, strict=False)
    if missing or unexpected:
        print(f"[Stage2] Adapter 以 strict=False 加载：missing={len(missing)} unexpected={len(unexpected)}")

    # 构建/加载 Stage-2 VAE。默认不复用 streaming VAE，因为两者训练配置可能不同。
    vae = _build_stage2_infinity_vae_from_ckpt(ckpt)

    # 可选：读取 label stats，把动作头归一化输出反变换回 rad / meter。
    label_stats: Optional[Dict[str, torch.Tensor]] = None
    ls = ckpt.get("label_stats")
    if isinstance(ls, dict) and all(k in ls for k in ("mean_angles", "std_angles", "mean_t", "std_t")):
        try:
            label_stats = {
                "mean_angles": torch.as_tensor(ls["mean_angles"], dtype=torch.float32, device=device).reshape(3),
                "std_angles": torch.as_tensor(ls["std_angles"], dtype=torch.float32, device=device).reshape(3),
                "mean_t": torch.as_tensor(ls["mean_t"], dtype=torch.float32, device=device).reshape(3),
                "std_t": torch.as_tensor(ls["std_t"], dtype=torch.float32, device=device).reshape(3),
            }
            src = ckpt.get("label_stats_source") or "checkpoint"
            print(f"[Stage2] label_stats 已从 {src} 加载")
        except Exception as e:
            print(f"[Stage2] 解析 label_stats 失败: {e}")
            label_stats = None

    _s2_tsformer = tsformer
    _s2_adapter = adapter
    _s2_vae = vae
    _s2_label_stats = label_stats
    _s2_ckpt_path = ckpt_path


def _stage2_patchify_to_z64_BCTHW(summed_codes_BCTHW: torch.Tensor) -> torch.Tensor:
    """
    把 InfinityStar 的 `summed_codes` 转成 Stage-2 VAE decode 需要的 patchified `z_ext`。

    期望形状：
    - 标准输入是 `[1,64,T_lat,16,16]`，即已经 patchify 后的 64 通道 latent。
    - 有些 InfinityStar 变体输出未 patchify 的 `[1,16,T_lat,H,W]`，此时用
      `pixel_unshuffle(factor=2)` 把 2x2 空间块折到通道维，得到 64 通道。

    重点注释：
    `summed_codes` 是世界模型预测出的 latent segment。Stage-2 动作解码器训练时使用的是
    patchified `z_ext` 形态，因此这里只做张量布局转换，不改变“世界转移”的语义。

    为什么可能有 16/64 两种通道：
    - 未 patchify latent：`C=16`，空间网格较大，例如 `[1,16,T,32,32]`；
    - patchified latent：`C=64`，把每个 `2x2` 空间块折进通道维，例如 `[1,64,T,16,16]`。

    `pixel_unshuffle(2)` 的直觉：
    它不会做卷积、插值或学习变换，只是重新排列元素：
    `[C=16, H, W] -> [C=16*4, H/2, W/2]`。
    所以这一步是“布局整理”，不是重新生成 latent。
    """
    if summed_codes_BCTHW.ndim != 5 or int(summed_codes_BCTHW.shape[0]) != 1:
        raise ValueError(f"期望 summed_codes 形状为 [1,C,T,H,W]，实际 {tuple(summed_codes_BCTHW.shape)}")
    c = int(summed_codes_BCTHW.shape[1])
    if c == 64:
        # 已经是 Stage-2 VAE 训练时使用的 patchified 布局，直接规范成 contiguous。
        return summed_codes_BCTHW.contiguous()
    if c != 16:
        raise ValueError(f"Stage2 不支持 summed_codes 通道数 C={c}（需要 16 或 64）")
    # PyTorch 的 pixel_unshuffle 只接受 4D `[N,C,H,W]`，所以先把时间维 T 合并到 batch。
    # [B,C,T,H,W] -> [B,T,C,H,W]
    x = summed_codes_BCTHW.permute(0, 2, 1, 3, 4).contiguous()
    b, t, c0, h, w = x.shape
    if int(h) % 2 != 0 or int(w) % 2 != 0:
        raise ValueError(f"空间尺寸为奇数，无法 pixel_unshuffle: H,W={(int(h), int(w))}")
    x2 = x.view(int(b) * int(t), int(c0), int(h), int(w))
    # 每个 2x2 空间块进入通道维：16 * 2 * 2 = 64。
    x2 = torch.nn.functional.pixel_unshuffle(x2, 2)  # (B*T, C*4, H/2, W/2) => (B*T,64,*,*)
    x2 = x2.view(int(b), int(t), int(x2.shape[1]), int(x2.shape[2]), int(x2.shape[3]))
    out = x2.permute(0, 2, 1, 3, 4).contiguous()  # [B,64,T,H/2,W/2]
    return out


def _stage2_decode_tokens_tnd(*, vae: torch.nn.Module, adapter: torch.nn.Module, z64_BCTHW: torch.Tensor) -> torch.Tensor:
    """
    通过 VAE decoder 解码 `z_ext`，hook 最后一个 up_block feature 并映射成 TSformer patch tokens。

    返回形状：
    `tokens_tnd = (T_frames, N_patches, D)`。其中 `T_frames` 是解码后的帧数，
    `N_patches` 是每帧空间 patch 数，`D` 是 TimesFormer token 维度。

    数据流对齐：
    这里复用 Stage-2 latent2action 训练时的 “VAE feature -> Adapter -> patch token”
    路径，保证在线推理和训练时的动作头输入一致。

    中文导读：
    这里不要把 `vae.decode()` 理解成“生成给人看的 RGB 视频”。代码注册了 decoder
    最后一层 up_block hook，只截取中间特征，再通过 adapter 翻译成 TimesFormer patch tokens。
    动作头消费的是这些 token，而不是最终图像像素。
    
    形状流：
    `z64_BCTHW [1,64,T_lat,H,W]`
      -> VAE decoder 最后 up_block feature `hs [1,96,T_frames,H_feat,W_feat]`
      -> Adapter 输出 `tok [1*T_frames,N,D]`
      -> reshape 成 `tokens_tnd [T_frames,N,D]`。
    """
    if z64_BCTHW.ndim != 5 or int(z64_BCTHW.shape[0]) != 1 or int(z64_BCTHW.shape[1]) != 64:
        raise ValueError(f"期望 z_ext 形状为 (1,64,T_lat,H,W)，实际 {tuple(z64_BCTHW.shape)}")

    # 确保输入 tensor 的设备和 dtype 与 VAE 一致。
    try:
        vae_device = next(iter(vae.parameters())).device
        vae_dtype = next(iter(vae.parameters())).dtype
    except Exception:
        vae_device = torch.device("cuda" if _DEVICE == "cuda" else "cpu")
        vae_dtype = torch.float32

    z = z64_BCTHW.to(vae_device, dtype=vae_dtype, non_blocking=(vae_device.type == "cuda"))

    tokens_slices: List[torch.Tensor] = []

    def hook(_module, _inp, out):
        """VAE decoder forward hook：截取 up_block feature 并转换成 TimesFormer tokens。"""
        # out 是 decoder 某个 up_block 的中间特征，不是 RGB。训练 Stage2 时动作头正是看这类
        # feature，因此在线推理也必须从同一个位置取特征，不能改成直接看最终像素。
        hs = out[0] if isinstance(out, (tuple, list)) else out  # 代码/形状说明：(B,96,t_slice,H,W)
        if not isinstance(hs, torch.Tensor) or hs.ndim != 5:
            raise RuntimeError("VAE decoder hook 输出不是 5D Tensor")
        bh = int(hs.shape[0])
        if bh != 1:
            raise RuntimeError(f"VAE hook 中 batch 维度不符合预期: hs={tuple(hs.shape)}")
        t_slice = int(hs.shape[2])
        # Adapter 把 VAE 的 5D 卷积特征压成 TimesFormer 可接收的 patch tokens。
        tok, _t2, _w2 = adapter(hs)  # 代码/形状说明：(B*t_slice, N, D)
        tok = tok.view(bh, t_slice, int(tok.shape[1]), int(tok.shape[2])).contiguous()  # 代码/形状说明：(1,t_slice,N,D)
        tokens_slices.append(tok[0])  # 代码/形状说明：(t_slice,N,D)

    # 在 decoder 最后一个 up_block 上注册 hook，用来截取中间特征。
    # finally 中必须 remove，避免下一次请求重复注册 hook 导致 tokens 被重复收集。
    try:
        handle = vae.decoder.up_blocks[-1].register_forward_hook(hook)  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"注册 VAE decoder hook 失败: {e}")

    try:
        with torch.no_grad():
            if vae_device.type == "cuda":
                use_amp = vae_dtype in (torch.float16, torch.bfloat16)
                with torch.cuda.amp.autocast(enabled=bool(use_amp), dtype=(vae_dtype if use_amp else torch.float16)):
                    try:
                        _ = vae.decode(z, return_dict=False)[0]  # type: ignore[call-arg]
                    except Exception:
                        _ = vae.decode(z)  # type: ignore[misc]
            else:
                try:
                    _ = vae.decode(z, return_dict=False)[0]  # type: ignore[call-arg]
                except Exception:
                    _ = vae.decode(z)  # type: ignore[misc]
    finally:
        try:
            handle.remove()
        except Exception:
            pass

    if len(tokens_slices) == 0:
        raise RuntimeError("VAE decoder hook 未捕获到 tokens")
    tokens_tnd = torch.cat(tokens_slices, dim=0).contiguous()  # (T,N,D)
    return tokens_tnd


def _gather_window_tokens(tokens_tnd: torch.Tensor, starts: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    按滑窗起点从整段 token 序列中收集 TimesFormer 输入。

    输入输出形状：
    - `tokens_tnd`: `(T,N,D)`，整段视频的逐帧 patch token。
    - `starts`: `(K,)`，K 个窗口的起始帧下标。
    - 返回 `patch_tokens`: `(K*window_size, N, D)`，把 K 个窗口按 batch 维摊平。

    新手提示：
    这个函数只做索引 gather，不做模型推理。真正的动作预测发生在
    `_stage2_predict_16_actions_for_segment_cm_deg()` 调用 TimesFormer 之后。

    为什么返回 `(K*window_size,N,D)`：
    TimesFormer 的 `forward_features_from_patch_tokens()` 需要把所有窗口当作一个 batch：
    - K 是窗口数量；
    - 每个窗口有 `window_size=4` 帧；
    - 所以时间维先摊平成 `K*4`，调用时再通过 `B=K, T=4` 告诉模型如何还原窗口结构。

    例子：
    如果 `starts=[0,1,2]` 且 `window_size=4`，实际取到的窗口是
    `[0,1,2,3]`、`[1,2,3,4]`、`[2,3,4,5]`。这些窗口互相重叠，所以后面同一帧的
    多个 delta 预测需要平均。
    """
    if tokens_tnd.ndim != 3:
        raise ValueError(f"tokens_tnd 必须是 (T,N,D)，实际为 {tuple(tokens_tnd.shape)}")
    t, n, d = tokens_tnd.shape
    k = int(starts.shape[0])
    # 为了用一次 gather 完成批量索引，先把每帧的 `(N,D)` 展平成一行。
    flat = tokens_tnd.view(int(t), int(n) * int(d))
    t_idx = torch.arange(int(window_size), device=starts.device, dtype=torch.long).view(1, int(window_size))
    idx = starts.view(k, 1) + t_idx  # 代码/形状说明：(K,window_size)
    # idx2 的每一行对应要取出的一个时间帧；expand 到 `N*D` 后才能从 flat 中取整帧 token。
    idx2 = idx.view(k * int(window_size), 1).expand(k * int(window_size), int(n) * int(d))
    g = flat.gather(0, idx2).view(k * int(window_size), int(n), int(d))
    return g.contiguous()


def _stage2_deltas_to_actions_cm_deg(deltas_T6: torch.Tensor) -> List[List[float]]:
    """
    将动作头输出的 `deltas_T6` 转成 HTTP API 对外使用的动作格式。

    输入布局：
    `deltas_T6` 每行是 `[dz,dy,dx, tx,ty,tz]`，前三个角度单位是 rad，
    后三个平移单位是 meter；第 0 行通常是起点占位，不对应可执行动作。

    输出布局：
    每个可执行 step 返回 `[dx_cm,dy_cm,dz_cm, droll_deg,dyaw_deg,dpitch_deg]`。
    - 公式：meter -> cm: *100（平移维度 tx,ty,tz）
    - 公式：rad -> deg: *180/pi（角度维度 dz,dy,dx）

    为什么从第 1 行开始：
    `deltas_T6[0]` 表示窗口/clip 起点，没有“从上一帧到当前帧”的可执行动作。
    真正可发给客户端的是从 frame 0 到 frame 1、frame 1 到 frame 2 ... 的增量，
    因此循环从 `i=1` 开始，返回长度是 `T-1`。

    坐标轴换序：
    训练内部角度布局是 `[dz,dy,dx]`，但 API 对外约定是 `[roll(x), yaw(z), pitch(y)]`；
    训练内部平移布局是 `[tx,ty,tz]`，API 对外放在前三个位置 `[dx,dy,dz]`，单位改为 cm。
    """
    if deltas_T6.ndim != 2 or int(deltas_T6.shape[1]) != 6:
        raise ValueError(f"deltas 必须是 (T,6)，实际为 {tuple(deltas_T6.shape)}")
    out: List[List[float]] = []
    t = int(deltas_T6.shape[0])
    for i in range(1, t):
        dz, dy, dx = [float(x) for x in deltas_T6[i, 0:3]]
        tx, ty, tz = [float(x) for x in deltas_T6[i, 3:6]]
        # 单位转换给初学者看：meters -> cm 是乘 100；rad -> deg 是乘 180/pi。
        # 顺序转换给调用方看：内部 [dz,dy,dx,tx,ty,tz] -> API [dx,dy,dz,roll,yaw,pitch]。
        out.append(
            [
                tx * 100.0,
                ty * 100.0,
                tz * 100.0,
                dx * (180.0 / math.pi),
                dz * (180.0 / math.pi),
                dy * (180.0 / math.pi),
            ]
        )
    return out


def _stage2_predict_16_actions_for_segment_cm_deg(
    *,
    st: "TrajectoryState",
    infer_res: "SegmentInferResult",
    stride: int = 1,
) -> List[List[float]]:
    """
    Stage-2 latent2action 的单段 16 动作预测主路径。

    数据流：
    1. 从 `infer_res.summed_codes` 得到整段预测 horizon 的 `z_ext`。
    2. VAE decoder + Adapter 生成整段 `tokens_tnd`。
    3. 对当前 segment 只取左侧上下文 `[ctx_start .. clip_end]`，不看右侧未来帧。
    4. 用 `window=4` 的 TimesFormer 滑窗推理，并对重叠帧做平均。
    5. 切出本 segment 对应的 16 个动作并转成 cm/deg。

    关键公式（小白必看）：
    - 公式：meters -> cm: *100，radians -> degrees: *180/pi
      （把动作头内部使用的 (m, rad) 单位换算成 API 对外约定的 (cm, deg)，
      在 `_stage2_deltas_to_actions_cm_deg()` 中完成换算）
    - 公式：4 帧滑窗聚合 delta[t] = sum(predictions covering t) / count[t]，且 delta[0]=0
      （每帧 t 的位姿增量来自所有覆盖该帧的窗口预测的平均值，
      首帧没有“上一帧到当前帧”的转移，所以 delta[0] 强制为 0 占位）

    对齐规则：
    这与 `actionhead_ref_vit` 的“只在左侧 pad 3 帧”规则一致，避免用未来帧泄漏。

    中文导读：
    `obs_len -> next_obs_len` 是本轮闭环要执行的 16 帧动作区间。函数会额外取左侧最多
    3 帧上下文，是因为 TimesFormer 的动作头用 4 帧滑窗预测相邻帧之间的动作增量。
    输出单位会转换成 API 对外约定的 `[dx_cm, dy_cm, dz_cm, droll_deg, dyaw_deg, dpitch_deg]`。

    具体例子：
    - seg0：`obs_len=1, next_obs_len=17`，需要输出 16 个动作；
    - `clip_abs_start=2, clip_abs_end=17` 表示动作到达的目标帧是 2..17；
    - `ctx_start_abs=max(1, clip_abs_start-3)=1`，所以送入动作头的 token 覆盖帧 1..17；
    - TimesFormer 按 4 帧窗口看：`[1,2,3,4]`、`[2,3,4,5]` ...； 这表示动作头每次看连续 4 帧 token，然后预测这 4 帧里的 3 个相邻动作 delta。
    - 每个窗口输出 3 个相邻帧 delta，重叠位置会有多个预测，后面用平均合成逐帧 delta；
    - 最后只切出当前 segment 的 16 个动作，左侧上下文产生的辅助 delta 不会返回给客户端。

    窗口数量 K = T_sub - window_size + 1 = 17 - 4 + 1 = 14，代码里不是循环调用 TimesFormer 14 次，而是把 14 个窗口打包成一个 batch
    frame4 delta 来自：
    [1,2,3,4] 的第 3 个输出
    [2,3,4,5] 的第 2 个输出
    [3,4,5,6] 的第 1 个输出

    最后聚合成逐帧 delta：

    frame1: 0，占位
    frame2..frame17: 16 个动作

    为什么要求 full-horizon：
    下面的切片使用绝对帧号 `1..num_frames`。如果世界模型只预测 tail-window，本地 token
    下标会和绝对帧号错位；当前 Stage2 latent2action 路径为了简单可靠，直接要求
    `infer_num_frames == total_num_frames`。
    """
    if _s2_tsformer is None or _s2_adapter is None or _s2_vae is None:
        raise RuntimeError("Stage2 模型尚未初始化")

    # 强制使用 full-horizon 推理，保证绝对帧下标稳定。
    if int(infer_res.infer_num_frames) != int(infer_res.total_num_frames):
        raise ValueError(
            "stage2 tsformer_latent 要求 infer_num_frames == total_num_frames。"
            "该模式下请关闭 rolling_tail_infer/tail_window。"
        )

    obs_len = int(infer_res.obs_len)
    next_obs_len = int(infer_res.next_obs_len)
    if int(next_obs_len - obs_len) != 16:
        raise ValueError(f"stage2 tsformer_latent 期望 16 帧 segment，实际 obs_len={obs_len} next_obs_len={next_obs_len}")
    # 当前 segment 的动作是从 obs_len 这一帧开始，依次到达 obs_len+1 .. next_obs_len。
    # 所以真正需要返回的目标帧范围是 `[clip_abs_start, clip_abs_end]`。
    clip_abs_start = int(obs_len) + 1
    clip_abs_end = int(next_obs_len)
    # TimesFormer 4 帧窗口在 clip 起点处需要最多 3 帧左侧上下文。
    # seg0 没有更早帧，所以从 1 开始；seg1 例如 obs_len=17，则从 15 开始。
    ctx_start_abs = max(1, int(clip_abs_start) - 3)  # == max(1, obs_len-2)

    # 把 summed_codes 转成 patchified `z_ext`，再经 VAE hook/Adapter 解码成 token。
    summed = infer_res.summed_codes
    z64 = _stage2_patchify_to_z64_BCTHW(summed).detach()
    tokens_tnd = _stage2_decode_tokens_tnd(vae=_s2_vae, adapter=_s2_adapter, z64_BCTHW=z64)  # (T,N,D)

    t_full = int(tokens_tnd.shape[0])
    if int(clip_abs_end) > int(t_full):
        raise ValueError(f"tokens 时间长度不足: T={t_full}，但需要 clip_abs_end={clip_abs_end}")

    # 切出 `[ctx_start .. clip_end]` 闭区间；绝对帧号是 1-based，tensor 下标是 0-based。
    # 例如 ctx_start_abs=15、clip_abs_end=33，则 python slice 是 `[14:33]`。
    s0 = int(ctx_start_abs) - 1
    e0 = int(clip_abs_end)  # 代码/形状说明：python slice end (exclusive) => abs_end-1 + 1
    tokens_sub = tokens_tnd[s0:e0].contiguous()
    t_sub = int(tokens_sub.shape[0])
    if t_sub < int(_S2_WINDOW_SIZE):
        return []

    device = next(iter(_s2_tsformer.parameters())).device  # type: ignore[union-attr]
    tokens_sub = tokens_sub.to(device, dtype=torch.float32, non_blocking=(device.type == "cuda"))

    # 在 tokens_sub 上构造滑动窗口。stride=1 时，每相邻一帧都作为窗口起点；
    # stride>1 会减少窗口数量，但也会减少重叠平均的覆盖密度。
    starts = torch.arange(0, int(t_sub) - int(_S2_WINDOW_SIZE) + 1, max(1, int(stride)), device=device, dtype=torch.long)
    if int(starts.numel()) <= 0:
        return []
    patch_tokens = _gather_window_tokens(tokens_sub, starts=starts, window_size=int(_S2_WINDOW_SIZE))  # (K*W, N, D)
    k = int(starts.shape[0])

    with torch.no_grad():
        if device.type == "cuda":
            try:
                m_dtype = next(iter(_s2_tsformer.parameters())).dtype  # type: ignore[union-attr]
            except Exception:
                m_dtype = torch.float16
            use_amp = m_dtype in (torch.float16, torch.bfloat16)
            with torch.cuda.amp.autocast(enabled=bool(use_amp), dtype=(m_dtype if use_amp else torch.float16)):
                feat = _s2_tsformer.forward_features_from_patch_tokens(patch_tokens, B=k, T=int(_S2_WINDOW_SIZE), W=int(_S2_W_GRID))  # type: ignore[union-attr]
                pred = _s2_tsformer.head(feat)  # type: ignore[union-attr]
        else:
            feat = _s2_tsformer.forward_features_from_patch_tokens(patch_tokens, B=k, T=int(_S2_WINDOW_SIZE), W=int(_S2_W_GRID))  # type: ignore[union-attr]
            pred = _s2_tsformer.head(feat)  # type: ignore[union-attr]

    pred_f = pred.detach().float()  # (K,18)
    # 每个 4 帧窗口只预测后 3 个相邻帧转移：
    # window [t,t+1,t+2,t+3] -> delta(t+1), delta(t+2), delta(t+3)。
    window_deltas = pred_f.view(k, 3, 6)  # 代码/形状说明：(K,3,6) normalized or (rad,m)
    if isinstance(_s2_label_stats, dict):
        ma = _s2_label_stats["mean_angles"].view(1, 1, 3)
        sa = _s2_label_stats["std_angles"].view(1, 1, 3)
        mt = _s2_label_stats["mean_t"].view(1, 1, 3)
        stt = _s2_label_stats["std_t"].view(1, 1, 3)
        # Stage2 每个 delta 的布局是 `[dz,dy,dx, tx,ty,tz]`：前 3 维角度，后 3 维平移。
        window_deltas[:, :, 0:3] = window_deltas[:, :, 0:3] * sa + ma
        window_deltas[:, :, 3:6] = window_deltas[:, :, 3:6] * stt + mt

    # 聚合成逐帧 delta `(t_sub,6)`，其中 `delta[0]=0`。
    # 公式：delta[t] = sum(predictions covering t) / count[t]，且 delta[0]=0
    #
    # 重叠平均例子：
    # frame 4 的 delta 可能来自窗口 [1,2,3,4] 的第 3 个输出，
    # 也可能来自窗口 [2,3,4,5] 的第 2 个输出、窗口 [3,4,5,6] 的第 1 个输出。
    # 这些预测都指向同一个”到达 frame 4 的动作”，所以先累加到 acc，再用 cnt 平均。
    acc = torch.zeros((t_sub, 6), device=device, dtype=torch.float32)
    cnt = torch.zeros((t_sub,), device=device, dtype=torch.int32)

    # offs=[1,2,3]，因为窗口第 0 帧是起点，不产生动作；动作落在后 3 帧上。
    offs = torch.arange(1, int(_S2_WINDOW_SIZE), device=device, dtype=torch.long).view(1, -1)  # (1,3)
    t_idx = starts.view(-1, 1) + offs  # (K,3)
    mask = (t_idx >= 0) & (t_idx < int(t_sub))
    if bool(mask.any()):
        t_flat = t_idx[mask].view(-1)
        v_flat = window_deltas[mask].view(-1, 6)
        acc.scatter_add_(0, t_flat.view(-1, 1).expand(-1, 6), v_flat)
        cnt.scatter_add_(0, t_flat, torch.ones_like(t_flat, dtype=torch.int32))

    deltas = torch.zeros((t_sub, 6), device=device, dtype=torch.float32)
    m = cnt > 0
    if bool(m.any()):
        deltas[m] = acc[m] / cnt[m].to(torch.float32).view(-1, 1)

    actions_all = _stage2_deltas_to_actions_cm_deg(deltas.detach().cpu())  # len=t_sub-1

    # 精确切出当前 clip 的 16 个动作。
    # actions_all 是相对 tokens_sub 的 `T_sub-1` 个动作；start_idx 把绝对 obs_len
    # 换成 tokens_sub 内部下标，跳过左侧上下文动作。
    start_idx = int(obs_len) - int(ctx_start_abs)
    end_idx = int(start_idx) + (int(clip_abs_end) - int(obs_len))
    out = actions_all[int(start_idx) : int(end_idx)]
    need = int(clip_abs_end) - int(obs_len)
    if len(out) != need:
        raise ValueError(
            f"stage2 actions 长度不匹配：实际 {len(out)}，需要 {need} "
            f"(ctx={ctx_start_abs}, obs={obs_len}, end={clip_abs_end})"
        )
    return out


def _load_tsformer_p2p(
    *,
    ckpt_path: str,
    stats_path: str,
    device: str,
) -> Tuple[torch.nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    加载旧版 latent P2P TSformer 动作头。

    当前默认路径已经改为 Stage-2 latent2action；这个函数保留给旧 checkpoint、自测和
    offline precomputed summed_codes 评估。
    """
    try:
        from pretrain_latent_p2p import build_p2p_model  # type: ignore
    except Exception as e:
        raise RuntimeError(f"导入 legacy TSformer(P2P) 失败（请安装 fvcore 等依赖）: {e}")
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
        # 保持 strict=False：本仓库有多个模型变体，adapter 层不匹配通常不影响旧路径自测。
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
        print("[TSformer] 未找到 stats；将输出归一化后的 deltas")
    return model, mean_t, std_t


def _init_models(
    *,
    cfg: ServerConfig,
) -> None:
    """
    初始化常驻的 InfinityStar 世界模型和 per-session template。

    动作头不在这里强制加载，而是按请求的 `action_head_mode` 懒加载；这样服务端可先启动，
    并避免未使用的动作头占 GPU 显存。
    """
    global _infinity_args, _infinity_session_template, _infinity_self_correction

    # 这里只初始化 InfinityStar 权重和 session 模板。
    # 动作头（stage2 latent2action / actionhead_ref_vit）按请求模式懒加载，避免无谓占显存。
    if _infinity_session_template is not None:
        return

    print("[Service] 正在初始化模型...")
    print(f"[Service] 运行设备：device={_DEVICE} dtype={_DTYPE}")

    # 选择 InfinityStar 代码根目录，并把它的模块加入当前进程。
    _import_infinity_modules(Path(cfg.infinity_repo_root))

    def _resolve_path(p: str) -> str:
        """把配置里的相对路径解释为 infer/ 目录下的绝对路径。"""
        if not p:
            return p
        if os.path.isabs(p):
            return p
        return str((REPO / p).resolve())

    cfg.infinity.ckpt = _resolve_path(cfg.infinity.ckpt)
    cfg.tsformer.ckpt = _resolve_path(cfg.tsformer.ckpt)
    cfg.tsformer.stats = _resolve_path(cfg.tsformer.stats)

    if not cfg.infinity.ckpt:
        raise ValueError("InfinityStar checkpoint 路径为空（请在 config.json 或 INFINITY_CKPT 环境变量中设置）")

    # InfinityStar：构造参数并只加载一次模型权重。
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

    # 关键：`infinity_elegant` schedule 依赖 `args.frames_inner_clip` 计算
    # `scale_pack_info.frame_ss/frame_ee`。如果它和 schedule 家族不匹配
    # （例如 clip4frames 与 clip20frames 混用），`freqs_frames[:, frame_ss:frame_ee]`
    # 可能变成空切片，进而让 get_visual_rope_embeds() 在 size-0 tensor 上失败。
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
    vae_model_path = str(getattr(a, "vae_model_path", "") or "").strip()
    if vae_model_path:
        _load_model_weights_from_path(
            vae,
            path=vae_model_path,
            label="InfinityVAE",
            preferred_state_dict_paths=("trainer.vae_local", "vae_state_dict", "vae"),
            strict=False,
        )
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

    print("[Service] 模型初始化完成。")


# -------------------------
# 6) 单条轨迹状态
# -------------------------
@dataclass
class TrajectoryState:
    """
    单个 session 的闭环状态。

    中文导读：
    一个 `TrajectoryState` 对应一个 `session_id` 的在线导航轨迹。这里保存的不是一次性完整路线，
    而是“已经收到的真实帧 + 当前文本条件 + 可复用 KV cache + 上一段 latent 边界”。
    这使服务端能按 `1, step, step, ...` 的节奏分段推进，并在每轮动作后用真实观测纠偏。
    """

    session_id: str
    prompt_raw: str
    negative_prompt: str = ""
    created_at: float = field(default_factory=lambda: time.time())

    # 已收到并转换到 `[-1,1]`、目标尺寸 `(tgt_h,tgt_w)` 的真实帧；每帧是 CPU `[3,H,W]` tensor。
    frames_cpu: List[torch.Tensor] = field(default_factory=list)

    # 每条轨迹一个轻量 Infinity session wrapper；真实 cache 在模型里，这里保存导出的副本便于跨请求恢复。
    stream: Optional[InfinityStreamingSession] = None
    kv_cache: Optional[Any] = None

    # 第一帧 i2v 对齐所需的辅助对象，可为空。
    gt_ls_Bl_first: Optional[Any] = None

    # 闭环推理辅助状态。
    dyn_res: Optional[Any] = None
    h_sel: Optional[str] = None
    firstframe_prepared: bool = False

    # TSformer latent 记忆：跨 segment 保留 1 个边界 latent。
    last_latent_1: Optional[torch.Tensor] = None  # 代码/形状说明：[1,16,1,H,W] on CPU (float16/float32)
    latent_dir: Optional[str] = None  # 磁盘 cache 目录："<session_id>_infinity_latnet"

    # 输入变换使用的目标空间尺寸，只在 session 初期确定一次。
    tgt_h: Optional[int] = None
    tgt_w: Optional[int] = None
    h_div_w_template: float = float(DEFAULT_H_DIV_W)

    # 已输出 segment 的账本。
    last_emitted_segment: int = -1

    def num_frames(self) -> int:
        """当前 session 已收到的真实观测帧数量。"""
        return len(self.frames_cpu)


_TRAJ: Dict[str, TrajectoryState] = {}
_SESSION_ALIAS: Dict[str, str] = {}


def _make_run_session_id(external_session_id: str) -> str:
    """为 reset_session=True 的请求生成内部唯一 session，避免覆盖旧 latent cache。"""
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    # 加纳秒后缀，避免同一秒内重复启动导致目录名冲突。
    suffix = str(time.time_ns() % 1_000_000_000).rjust(9, "0")
    return f"{external_session_id}__{ts}_{suffix}"


def _get_or_create_traj(session_id: str, prompt: str, negative_prompt: str) -> TrajectoryState:
    """
    获取或创建同一个客户端 session 对应的轨迹状态。

    `TrajectoryState` 保存真实帧前缀、文本 prompt、KV cache、latent cache 目录和已输出 segment。
    后续同一 session 可以省略 prompt，只上传新增真实观测帧。
    """
    cfg = _get_server_config()
    if session_id in _TRAJ:
        st = _TRAJ[session_id]
        # 后续请求允许客户端省略 prompt。
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
    # latent cache 目录（可选，但默认启用）：
    # 这些文件不是协议必需项，主要用于离线排查某个 segment 的 latent/action 是否异常。
    root = os.environ.get("INFINITY_LATENT_CACHE_ROOT", "").strip()
    if not root:
        root = str((ROOT / "cache").resolve())
    try:
        os.makedirs(root, exist_ok=True)
        st.latent_dir = os.path.join(root, f"{session_id}_infinity_latnet")
        os.makedirs(st.latent_dir, exist_ok=True)
        # 尽力恢复：如果磁盘上已有 last_latent.pt，就作为跨段边界 latent 使用。
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
    """
    为某条轨迹创建轻量 streaming session。

    模型权重和 tokenizer 与全局 template 共享；每条轨迹只拥有自己的 prompt、KV cache 和
    首帧条件。
    """
    assert _infinity_session_template is not None
    assert _infinity_args is not None
    cfg = _get_server_config()

    if st.stream is not None:
        return

    # 创建每条轨迹自己的轻量 wrapper；模型、VAE、文本编码器都和全局模板共享。
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
    """把 TrajectoryState 中保存的 KV cache 导入当前 streaming session。"""
    assert st.stream is not None
    if st.kv_cache is None:
        return
    # 先重置当前 block 的 cache 容器，再导入这条轨迹上次保存的 cache。
    for blk in st.stream.infinity.unregistered_blocks:
        blk.attn.kv_caching(True, reset=True)
    st.stream.infinity.import_kv_cache(st.kv_cache, overwrite=True)


def _prepare_firstframe_condition_if_needed(st: TrajectoryState) -> None:
    """
    对齐 `batch_closed_loop_streaming_infer_routes.py`：
    - Step0 使用第一帧 gt_leak 注入，并且故意不把 obs1 写入 gt_obs cache。
    - 每条轨迹只预计算一次 gt_ls_Bl_first 与 dyn_res/h_sel。

    中文导读：
    第一帧在 i2v/closed-loop 里既是视觉起点，也是后续世界模型预测的条件。
    这里预先算好动态分辨率元信息和 first-frame latent condition；但不把第 1 帧写进
    `gt_obs` cache，避免和 streaming session 的 t0/text cache 语义混在一起。
    """
    cfg = _get_server_config()
    assert st.stream is not None
    assert _infinity_args is not None
    assert _infinity_self_correction is not None
    assert get_dynamic_resolution_meta is not None

    if st.firstframe_prepared:
        return
    if st.num_frames() <= 0:
        raise ValueError("尚未收到任何帧")

    dyn_res, _ = get_dynamic_resolution_meta(_infinity_args.dynamic_scale_schedule, _infinity_args.video_frames)  # type: ignore[misc]
    st.dyn_res = dyn_res

    # 从动态分辨率表里选择最接近当前宽高比模板的 key。
    try:
        import numpy as np  # 中文说明：local import; numpy is already used by InfinityStar tools

        h_keys = list(dyn_res.keys())
        h_vals = np.array([float(k) for k in h_keys], dtype=np.float64)
        st.h_sel = h_keys[int(np.argmin(np.abs(h_vals - float(st.h_div_w_template))))]
    except Exception:
        # 兜底：表结构异常时直接取第一个 key。
        st.h_sel = list(dyn_res.keys())[0]

    # 把第一帧编码成 gt_ls_Bl_first，保证严格 i2v 对齐。
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
    """用真实观测前缀 `[1..n_frames]` 覆盖 gt_obs cache，当前只支持 B=1。"""
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
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    执行一次闭环 step 的 InfinityStar 推理，返回 `summed_codes [1,16,pt,H,W]`。

    它复刻 `batch_closed_loop_streaming_infer_routes.py` 的控制流，但默认跳过 VAE decode。

    中文导读：
    这是“先预测世界怎样变”的核心调用。它不直接输出动作，而是让 InfinityStar 先根据
    `prompt + 已经确认的真实观测帧` 预测一段 latent 视频轨迹，也就是后面动作头要看的
    `summed_codes`。

    参数怎么理解：
    - `step_i`：当前第几个 segment。`0` 表示从第一帧出发预测第一段，`1` 表示从第 17
      帧附近继续预测下一段，依此类推。`step_i==0` 走首帧 i2v 条件；`step_i>0`
      走真实前缀注入逻辑。
    - `obs_len`：当前已经可以信任的真实 RGB 帧数量。注意它是“帧数”，不是 latent 下标。
    - `infer_num_frames`：本次让世界模型预测的窗口长度。默认是整条 81/49 帧；开启
      tail-window 后，后几段只看尾部窗口，减少长上下文压力。
    - `injection`：决定真实历史怎么注入世界模型。`gt_obs` 只写真实观测 cache；
      `official_leak`/`hybrid_leak_gtobs` 会额外把真实前缀编码成 `gt_ls_Bl` 做 leak 注入。

    数据流顺序：
    1. 准备本 step 的真实历史条件：首段用第一帧条件，后续段按 `injection` 选择
       `gt_obs cache` 或 `gt_leak`。
    2. 根据 `infer_num_frames` 构建动态分辨率 schedule，并生成和每个 scale 对齐的
       `tau_list`。
    3. 调官方 `gen_one_example(..., return_summed_code_only=True)`，得到 latent 级输出
       `summed_codes`。
    4. 如果需要调试视频或参考视频动作头，再把 `summed_codes` 解码成 BGR 视频；默认
       Stage2 latent2action 不需要这一步。

    关键公式：
    - 公式：pt_obs = (obs_len - 1) // temporal_compress_rate + 1
      （把"真实 RGB 前缀长度"映射到 VAE 压缩后的 latent 时间步。默认压缩率为 4。
      例如 obs_len=17 -> pt_obs=(17-1)//4+1=5，表示 17 帧 RGB 对应 5 个 latent 时间步。）
    """
    cfg = _get_server_config()
    assert st.stream is not None
    assert _infinity_args is not None

    # 确保当前 session 使用正确的宽高比模板。
    st.stream.h_div_w_template = float(st.h_div_w_template)
    st.stream.correction_clear_pred()

    # `gt_leak=-1` 表示本次不做 gt_leak 注入；只有首帧或指定 leak 模式时才会改成 >=0。
    gt_leak = -1
    # `gt_ls_Bl` 是 InfinityStar 官方路径中的真实 latent 条件，只有 leak 模式会传入。
    gt_ls_Bl = None

    if int(step_i) == 0:
        # 首段只依赖第一帧 i2v condition。这里故意不把第一帧再写入 gt_obs cache，
        # 避免“首帧条件”和“真实观测 cache”重复注入导致语义偏移。
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
            # 编码连续真实前缀 `[1..obs_len]`，并按 latent 时间自动计算 leak 深度。
            #
            # 小白容易混淆的点：
            # - `prefix` 是 RGB 帧前缀，形状先是 `[T,3,H,W]`；
            # - `prefix_obs` 转成 VAE/InfinityStar 要的 `[1,3,T,H,W]`；
            # - `video_encode()` 输出的 `gt_ls_Bl_prefix` 是 latent/token 条件，不是像素。
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
                # Hybrid 模式：同时写入 gt_obs cache，帮助后段 segment 稳定。
                # 这一步写入的是“真实前缀”，不是模型预测视频，因此后续 clear_pred_cache 不会删掉。
                st.stream.compute_kv_cache_gt(prefix_obs)

            # 帧到 latent 的压缩索引：latent_index(frame f) = (f - 1)//temporal_compress_rate + 1；
            # 这里用真实前缀长度 obs_len 算出该前缀覆盖到第几个 latent。
            pt_obs = (int(obs_len) - 1) // int(getattr(_infinity_args, "temporal_compress_rate", 4)) + 1
            pt2sched = st.dyn_res[st.h_sel][_infinity_args.pn]["pt2scale_schedule"]
            # `pt2sched[pt_obs]` 返回当前 latent 前缀在多 scale schedule 中覆盖了多少个 scale。
            # leak_auto 就是应该把多少层/多少段真实 token 作为 gt_leak 条件喂给模型。
            leak_auto = len(pt2sched[int(pt_obs)])
            gt_leak = int(leak_auto)
            gt_ls_Bl = gt_ls_Bl_prefix
        else:
            # gt_obs 模式：把真实前缀写入 cache，不再额外做 leak 注入。
            _update_gt_obs_cache_to(st, obs_len)

    # schedule 决定本次自回归会生成多少个 latent 时间步、每个 scale 的空间大小、
    # 以及 RoPE/context_info 怎么切。它必须和 `infer_num_frames` 一致。
    sched = st.stream.build_schedule_for_num_frames(int(infer_num_frames))
    tau_list = [float(cfg.infinity.tau_image)] * int(sched.tower_split_index) + [float(cfg.infinity.tau_video)] * (
        len(sched.scale_schedule) - int(sched.tower_split_index)
    )

    # 对齐 `tools/infer_v2v_segments_49f_clip16.py`：使用 gen_one_example() 风格封装，
    # 由它在每次调用内统一处理 cfg/tau 列表和 prompt 编码。
    try:
        if infinity_gen_one_example is None:
            raise RuntimeError("尚未导入 InfinityStar gen_one_example")
        assert _infinity_args is not None

        # prompt_infer 是实际喂给文本编码器的最终 prompt。这里会按配置追加视频时长标签，
        # 避免训练/推理时 “同一句文本但期望帧数不同” 的语义不一致。
        prompt_infer = _prompt_with_duration(
            st.prompt_raw,
            num_frames=int(cfg.infinity.num_frames),
            fps=int(cfg.infinity.fps),
            append_tag=bool(getattr(_infinity_args, "append_duration2caption", 0)),
        )

        with torch.no_grad():
            if _DEVICE == "cuda":
                with torch.cuda.amp.autocast(enabled=True, dtype=next(iter(st.stream.infinity.parameters())).dtype):
                    # return_summed_code_only=True 是关键：这里拿 latent，不拿最终 RGB。
                    # 后续 Stage2 动作头会直接消费 latent，省掉一次昂贵的视频 decode。
                    summed_codes = infinity_gen_one_example(  # type: ignore[misc]
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
                    )
            else:
                # CPU 路径主要用于调试/离线导入；参数保持和 CUDA 路径一致，避免两边语义分叉。
                summed_codes = infinity_gen_one_example(  # type: ignore[misc]
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
                )

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
                print(f"[pred->video] 解码/保存已跳过: {e}")
    except Exception:
        # 打印详细调试信息到服务端日志；FastAPI 会把异常包装成 HTTP 500 detail。
        print("[InfinityStar] infer_chunk 失败，正在打印调试信息...")
        print(traceback.format_exc())
        try:
            blk0 = st.stream.infinity.unregistered_blocks[0]
            ck = getattr(blk0.attn, "cached_k", {})
            cv = getattr(blk0.attn, "cached_v", {})
            keys = list(ck.keys())
            print(f"[InfinityStar] cached_k keys（第一个 block）: {keys}")
            for k in keys[:10]:
                vk = ck.get(k, None)
                vv = cv.get(k, None)
                sk = tuple(vk.shape) if isinstance(vk, torch.Tensor) else type(vk).__name__
                sv = tuple(vv.shape) if isinstance(vv, torch.Tensor) else type(vv).__name__
                print(f"  - key={k!r} k={sk} v={sv}")
        except Exception:
            print("[InfinityStar] 调试信息打印失败")
        raise

    st.stream.correction_clear_pred()
    return summed_codes, pred_vid


def _save_latent_tensor(st: TrajectoryState, name: str, t: torch.Tensor) -> None:
    """把中间 latent 以 float16 CPU tensor 存到 session cache 目录，失败时静默跳过。"""
    if not st.latent_dir:
        return
    try:
        p = os.path.join(st.latent_dir, name)
        # 存成 float16 CPU tensor，减少磁盘占用。
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
    把 latent clip 解码成视频，并保存到同一个 latent 目录。

    - seg0：5 个 latent -> 17 帧，保留第一帧；
    - seg>0：先解码 latent5，再丢弃第一个边界帧，得到 16 帧新画面。
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
        # Infinity.summed_codes2images 返回 uint8 BGR；tools.run_infinity.save_video 期望 BGR，
        # 并在写文件时内部转换到 RGB。
        out_path = os.path.join(st.latent_dir, name)
        cfg = _get_server_config()
        infinity_save_video(clip_np, fps=int(cfg.infinity.fps), save_filepath=out_path, force_all_keyframes=True)
    except Exception as e:
        print(f"[latent->video] 跳过 {name}: {e}")


def _save_pred_video(
    st: TrajectoryState,
    name: str,
    pred_video_BTHWC: Any,
) -> None:
    """把 Infinity 解码出的预测视频保存到 latent_dir；文件名必须唯一以避免覆盖。"""
    if not st.latent_dir or infinity_save_video is None:
        return
    try:
        vid = pred_video_BTHWC
        if isinstance(vid, torch.Tensor):
            vid = vid.detach().cpu().numpy()
        # 期望形状为 `[B,T,H,W,3]` 或 `[T,H,W,3]`。
        if getattr(vid, "ndim", 0) == 5:
            vid = vid[0]
        if getattr(vid, "ndim", 0) != 4 or int(vid.shape[-1]) != 3:
            return
        # Infinity.summed_codes2images 返回 BGR；tools.run_infinity.save_video 也期望 BGR。
        out_path = os.path.join(st.latent_dir, name)
        cfg = _get_server_config()
        infinity_save_video(vid, fps=int(cfg.infinity.fps), save_filepath=out_path, force_all_keyframes=True)
    except Exception as e:
        print(f"[pred->video] 跳过 {name}: {e}")


def _slice_abs_latents_from_summed_codes(
    summed_codes: torch.Tensor,
    *,
    abs_lat_start: int,
    abs_lat_end: int,
    infer_num_frames: int,
    total_num_frames: int,
) -> torch.Tensor:
    """
    `summed_codes` 是 `[1,16,pt_local,H,W]`，可能来自 full-horizon
    （`infer_num_frames==total_num_frames`），也可能来自尾部对齐的 tail-window。
    `abs_lat_start/end` 是整条视频 latent 时间线上的 1-based 绝对下标。
    返回 `[1,16,T,H,W]`，其中 `T == abs_lat_end-abs_lat_start+1`。

    中文说明：
    世界模型有两种推理窗口：
    - full-horizon：local latent 下标和整条轨迹的绝对 latent 下标一致；
    - tail-window：只预测尾部窗口，local 下标需要减去窗口起点。
    这个函数把“绝对 latent 时间线”映射回当前 `summed_codes` 的本地切片。
    """
    if abs_lat_end < abs_lat_start:
        raise ValueError(f"abs_lat 范围非法: {abs_lat_start}..{abs_lat_end}")
    t_local = int(summed_codes.shape[2])

    local_start = int(abs_lat_start)
    local_end = int(abs_lat_end)
    if int(infer_num_frames) != int(total_num_frames):
        window_start_abs = int(total_num_frames) - int(infer_num_frames) + 1  # 1-based 绝对帧号
        # tail-window 从某个绝对帧开始；先用 latent_index(f) = (f - 1)//4 + 1
        # 找到这个窗口在整条 latent 时间线上的起点，再把绝对 latent 下标平移成本地切片下标。
        abs_lat_start_window = (int(window_start_abs) - 1) // 4 + 1
        local_start = int(abs_lat_start) - int(abs_lat_start_window) + 1
        local_end = int(abs_lat_end) - int(abs_lat_start_window) + 1

    s0 = max(1, int(local_start))
    e0 = min(int(local_end), int(t_local))
    if e0 < s0:
        raise ValueError(
            f"latent 切片越界：abs [{abs_lat_start}..{abs_lat_end}] -> "
            f"本地 [{local_start}..{local_end}]，t_local={t_local}"
        )
    out = summed_codes[:, :, (s0 - 1) : e0].contiguous()
    # 如果请求切片有一部分不在当前窗口内，先按错误处理，避免静默返回错位 latent。
    if int(out.shape[2]) != int(abs_lat_end - abs_lat_start + 1):
        raise ValueError(
            f"latent 切片长度不匹配：需要 {abs_lat_end-abs_lat_start+1}，实际 {out.shape[2]} "
            f"(abs [{abs_lat_start}..{abs_lat_end}] -> 本地 [{local_start}..{local_end}]，infer_num_frames={infer_num_frames})"
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
    `pred_video_BTHWC` 是 `[1,T,H,W,3]` 或 `[T,H,W,3]` uint8(BGR)，
    其中 `T==infer_num_frames`，可能是 full-horizon 或 tail-window。
    `abs_frame_start/end` 是整条视频像素帧时间线上的 1-based 绝对下标。
    返回 `[Tslice,H,W,3]` uint8(BGR)。
    """
    if np is None:
        raise RuntimeError("actionhead 模式需要 numpy")
    if abs_frame_end < abs_frame_start:
        raise ValueError(f"abs_frame 范围非法: {abs_frame_start}..{abs_frame_end}")

    vid = pred_video_BTHWC
    if isinstance(vid, torch.Tensor):
        vid = vid.detach().cpu().numpy()
    if getattr(vid, "ndim", 0) == 5:
        vid = vid[0]
    if getattr(vid, "ndim", 0) != 4 or int(vid.shape[-1]) != 3:
        raise ValueError(f"pred_video 形状非法: {getattr(vid,'shape',None)}")
    t_local = int(vid.shape[0])

    if int(infer_num_frames) == int(total_num_frames):
        local_start = int(abs_frame_start) - 1
        local_end = int(abs_frame_end) - 1
    else:
        window_start_abs = int(total_num_frames) - int(infer_num_frames) + 1  # 1-based 绝对帧号
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
    输入 `fr_3hw` 是 `[-1,1]` 范围的 RGB `[3,H,W]` tensor。
    返回 uint8 BGR `[H,W,3]`，用于 OpenCV/视频保存链路。
    """
    if np is None:
        raise RuntimeError("需要 numpy")
    x = fr_3hw.detach().to("cpu", dtype=torch.float32).clamp(-1.0, 1.0)
    x = (x + 1.0) * 0.5 * 255.0
    x = x.round().clamp(0.0, 255.0).to(torch.uint8)
    rgb = x.permute(1, 2, 0).contiguous().numpy()  # HWC RGB
    return rgb[..., ::-1].copy()  # BGR


@dataclass
class SegmentInferResult:
    """
    单个 segment 的世界模型输出包。

    - `summed_codes`: 当前推理窗口的完整 latent 输出，是 Stage-2 动作头的主要输入；
    - `latent5_input`: 旧 TSformer(P2P) 路径需要的 5 个 latent step；
    - `pred_vid_bgr`: 可选预测视频，只在参考视频动作头或调试保存时使用；
    - `obs_len/next_obs_len`: 本段动作覆盖的绝对 RGB 帧区间。
    """
    latent5_input: torch.Tensor
    summed_codes: torch.Tensor
    pred_vid_bgr: Optional[torch.Tensor]
    infer_num_frames: int
    obs_len: int
    next_obs_len: int
    total_num_frames: int


def _infer_latents_for_actions_and_advance_cache(
    st: TrajectoryState,
    *,
    segment_index: int,
    seed: int,
    advance_gt_obs_to_next: bool = True,
    need_pred_video: bool = False,
) -> SegmentInferResult:
    """
    对给定 segment i 执行以下步骤：
    - 按闭环/rolling-tail 配置运行 InfinityStar 推理；
    - seg0：切出预测 4 个旧式动作所需的 5 个 latent step；
    - seg>0：只切 4 个新 latent，再拼上保存的 `last_latent_1` 形成 5 个 latent；
    - 在真实帧已经到达时，用新暴露的前缀 `points[i+1]` 覆盖 gt_obs cache。

    返回的 `latent5_input` 形状为 `[1,16,5,H,W]`。

    中文导读：
    这个函数把闭环协议落到代码上：先预测当前 segment 的 latent，再根据是否已经收到
    `points[i+1]` 对应的真实帧决定能否推进 `gt_obs` cache。若本轮是 future emission
    （只拿到 segment 起点真实帧就提前输出动作），就不能把预测帧写成真实历史。

    为什么动作头需要 5 个 latent：
    Stage2 latent2action 的训练样本是“一个边界 latent + 后续 4 个新 latent”。它相当于
    用 5 个 latent 表示 16 帧左右的运动片段，再由 TimesFormer 滑窗预测 16 个动作。
    因此这里不能简单把 `summed_codes` 全部丢给动作头，而要切出对齐当前 segment 的
    `latent5_input`。

    segment/帧/latent 的对应关系：
    - `points = [1, 17, 33, 49, ...]` 表示每段动作的 RGB 帧边界；
    - 第 i 段动作覆盖 `points[i] -> points[i+1]` 之间的 16 个转移；
    - latent 下标用 `latent_index(frame f) = (f - 1)//4 + 1` 从 RGB 帧号换算；
    - `seg0` 直接从世界模型输出里切 `[end-4..end]` 五个 latent；
    - `seg>0` 使用上一段保存的 `last_latent_1` 作为边界，再拼当前段的 4 个新 latent。

    cache 推进规则：
    - `advance_gt_obs_to_next=True` 只在真实帧已经到达 `points[i+1]` 时使用；
    - 如果提前输出未来段动作，则 `advance_gt_obs_to_next=False`，避免把预测帧伪装成
      真实历史写入 `gt_obs` cache。

    具体例子：
    - `points = [1, 17, 33, 49]` 时，seg0 覆盖 `frame 1 -> frame 17`，输出 16 个动作；
    - 默认 VAE 时间压缩率为 4，所以 `frame 1/5/9/13/17` 对应 latent `1/2/3/4/5`；
    - 因此 seg0 的动作头输入是 latent `[1,2,3,4,5]`；
    - seg1 覆盖 `frame 17 -> frame 33`，输入应是 `[latent5, latent6, latent7, latent8, latent9]`；
    - 这里的 `latent5` 只是一帧边界 latent，不是上一整段。保留它是为了让相邻段在
      `frame 17` 这个公共边界上连续。
    """
    cfg = _get_server_config()
    points = cfg.infinity.points()
    if segment_index < 0 or segment_index >= len(points) - 1:
        raise ValueError(f"segment_index 非法: {segment_index}, points={points}")

    # `obs_len` 是当前 segment 左边界帧号，`next_obs_len` 是右边界帧号。
    # 例如 points=[1,17,33,49] 时，seg1 的 obs_len=17、next_obs_len=33。
    obs_len = int(points[segment_index])
    next_obs_len = int(points[segment_index + 1])

    # 每个 segment 可单独调整采样参数和历史注入策略。
    # 这样前几段可以更探索，后几段可以更保守；也可从某段开始切到 tail-window。
    lock_seed = bool(cfg.infinity.lock_seed_across_steps)
    local_seed = int(seed) + (0 if lock_seed else int(segment_index))
    use_late = int(segment_index) >= int(cfg.infinity.late_step_start)
    step_top_k = int(cfg.infinity.late_top_k) if use_late else int(cfg.infinity.top_k)
    step_top_p = float(cfg.infinity.late_top_p) if use_late else float(cfg.infinity.top_p)
    inj = str(cfg.infinity.late_v2v_history_injection or cfg.infinity.v2v_history_injection) if use_late else str(cfg.infinity.v2v_history_injection)

    infer_num_frames = int(cfg.infinity.num_frames)
    if (
        bool(cfg.infinity.rolling_tail_infer)
        and str(cfg.infinity.rolling_infer_mode) == "tail_window"
        and int(segment_index) >= int(cfg.infinity.tail_window_start_step)
    ):
        # tail-window 模式只让世界模型预测尾部窗口。后面切 latent 时必须把“绝对帧号”
        # 映射回这个局部窗口，所以 `_slice_abs_latents_from_summed_codes()` 会额外处理平移。
        infer_num_frames = int(cfg.infinity.tail_window_frames)

    # 第一步：让世界模型预测当前段所在窗口的 latent。此时还没有动作，只有世界状态预测。
    summed_codes, pred_vid = _infer_summed_codes_for_step(
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

    total_num_frames = int(cfg.infinity.num_frames)
    # segment 右边界是 RGB 帧 `next_obs_len`；用 latent_index(f) = (f - 1)//4 + 1
    # 转成动作头要读取的绝对 latent 右边界。
    abs_end_lat = (int(next_obs_len) - 1) // 4 + 1  # 中文标题：当前 segment 结束后的绝对 latent

    latent5_input: torch.Tensor
    if int(segment_index) == 0 or st.last_latent_1 is None:
        # seg0（或恢复失败）：提供完整 5-latent 窗口 `[end-4..end]`。
        # 对 1->17 帧而言，end_lat 通常是 5，因此切出的就是 latent 1..5。
        abs_start_lat = max(1, int(abs_end_lat) - 4)
        latent5_input = _slice_abs_latents_from_summed_codes(
            summed_codes,
            abs_lat_start=int(abs_start_lat),
            abs_lat_end=int(abs_end_lat),
            infer_num_frames=int(infer_num_frames),
            total_num_frames=int(total_num_frames),
        )
        # 如果视频太短导致不足 5 个 latent，就重复最后一个补齐。
        # 这只是容错；正常配置下 16 帧 segment 会有足够 latent。
        if int(latent5_input.shape[2]) < 5:
            rep = latent5_input[:, :, -1:].repeat(1, 1, 5 - int(latent5_input.shape[2]), 1, 1)
            latent5_input = torch.cat([latent5_input, rep], dim=2)
    else:
        # seg>0：先把上一段 RGB 边界 `prev_obs_len` 转成 latent_index(prev_obs_len)，
        # 再只取 4 个新 latent `(prev_end+1 .. cur_end)`，并拼上上一段边界 latent。
        prev_obs_len = int(points[segment_index])  # 中文标题：等于当前 obs_len
        prev_end_lat = (int(prev_obs_len) - 1) // 4 + 1
        abs_start_lat_new = int(prev_end_lat) + 1
        abs_end_lat_new = int(abs_end_lat)
        # 只切“本段新增”的 4 个 latent；第 1 个边界 latent 来自上一段保存的 last_latent_1。
        # 这样相邻 segment 的动作头输入在边界处连续，不会凭空断开。
        new4 = _slice_abs_latents_from_summed_codes(
            summed_codes,
            abs_lat_start=int(abs_start_lat_new),
            abs_lat_end=int(abs_end_lat_new),
            infer_num_frames=int(infer_num_frames),
            total_num_frames=int(total_num_frames),
        )  # 代码/形状说明：期望 [1,16,4,H,W]
        # 保持 latent 在 CPU 上，方便后续 TSformer 输入和磁盘保存，也避免和 CPU 上的
        # `st.last_latent_1` 拼接时出现 device mismatch。
        if isinstance(new4, torch.Tensor):
            new4 = new4.detach().to("cpu").contiguous()
        if int(new4.shape[2]) < 4:
            rep = new4[:, :, -1:].repeat(1, 1, 4 - int(new4.shape[2]), 1, 1)
            new4 = torch.cat([new4, rep], dim=2)
        last1 = st.last_latent_1
        if last1 is None:
            # 理论上不该发生；保留兜底以便服务不中断。
            last1 = new4[:, :, :1].clone()
        # 确保边界 latent 和新 latent 的空间尺寸一致。
        # 如果触发这里，通常是 full-horizon/tail-window 或动态分辨率配置不一致。
        if last1.shape[-2:] != new4.shape[-2:]:
            raise ValueError(f"latent 空间尺寸不匹配: last1={tuple(last1.shape)} new4={tuple(new4.shape)}")
        latent5_input = torch.cat([last1.to(new4.dtype), new4], dim=2).contiguous()

    # 统一 latent5 的存放位置；TSformer 路径会在内部从 CPU 搬到 CUDA。
    # 放 CPU 的好处是：可以保存调试文件，也避免长时间占用 GPU 显存。
    latent5_input = latent5_input.detach().to("cpu").contiguous()

    # 推进 cache：用新暴露的真实前缀 `[1..next_obs_len]` 覆盖 gt_obs cache。
    # 最后一段如果包含纯预测尾帧，调用方会关闭这里，避免把非真实帧写入 gt_obs cache。
    if bool(advance_gt_obs_to_next):
        # 清一次 pred cache，防止刚刚生成的“想象未来”混进真实前缀编码。
        st.stream.correction_clear_pred()  # type: ignore[union-attr]
        _update_gt_obs_cache_to(st, int(next_obs_len))
        # 再清一次，确保 `compute_kv_cache_gt()` 前后没有残留 pred 条目。
        st.stream.correction_clear_pred()  # type: ignore[union-attr]
        # 导出当前这条轨迹的 GT cache 快照；下一次请求恢复同一 session 时继续使用。
        st.kv_cache = st.stream.infinity.export_kv_cache()  # type: ignore[union-attr]

    # 更新跨段记忆，并把调试 latent 存到磁盘。
    # `last_latent_1` 永远保存当前 latent5 的最后一个时间步，作为下一段的边界。
    st.last_latent_1 = latent5_input[:, :, -1:].detach().to("cpu").contiguous()
    _save_latent_tensor(st, f"seg{int(segment_index):02d}_latent5_input.pt", latent5_input)
    if int(segment_index) == 0:
        _save_latent_tensor(st, "seg00_latent5.pt", latent5_input)
        _save_latent_video_clip(st, "seg00_latent5_17f.mp4", latent5_input, drop_first_frame=False)
        # 额外拆分保存边界 latent（第 1 帧）和 4 个新 latent（第 2..17 帧），方便排查。
        _save_latent_tensor(st, "seg00_first1.pt", latent5_input[:, :, 0:1].contiguous())
        _save_latent_tensor(st, "seg00_new4.pt", latent5_input[:, :, 1:].contiguous())
        _save_latent_video_clip(st, "seg00_new4_16f.mp4", latent5_input, drop_first_frame=True)
    else:
        # 非首段只额外保存 4 个新 latent，方便排查跨段拼接。
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
    )


def _tsformer_predict_actions_from_summed_codes(
    summed_codes_BCTHW: torch.Tensor,
    *,
    prefix_latents: int,
) -> torch.Tensor:
    """
    从 `summed_codes` 中取最后 4 个动作，返回形状 `(4,6)`，单位 cm/deg。
    """
    assert _ts_model is not None
    assert summed_codes_BCTHW.ndim == 5 and summed_codes_BCTHW.shape[0] == 1, f"期望 [1,C,T,H,W]，实际为 {tuple(summed_codes_BCTHW.shape)}"

    # InfinityStar 的 WAN VAE 常用 patchified codes：`(B, 4*C0, T, H/2, W/2)`。
    # TSformer adapter 训练时使用的是还原后的表示（C0=16）。
    # 如果这里看到 C=64，就反向 pixel_shuffle，变回 C=16 且空间尺寸放大 2 倍。
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
        raise ValueError(f"prefix_latents 太小: {k}")

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

    # 取最后 4 个动作；不足 4 个时重复最后一个补齐。
    if out.shape[0] >= 4:
        last4 = out[-4:]
    else:
        # 通过重复最后一个动作补齐。
        pads = [out[-1:]] * (4 - int(out.shape[0]))
        last4 = torch.cat([out] + pads, dim=0)

    return _to_cm_deg(last4).detach().cpu()


def _ah_denorm_window_preds(pred_norm: "np.ndarray", stats: Dict[str, "np.ndarray"]) -> "np.ndarray":  # type: ignore[name-defined]
    """把 actionhead 每个 4 帧窗口的标准化输出反标准化为 rad/m 的 3 个动作。"""
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
    """
    将重叠窗口的 3-step 动作预测平均成逐帧 delta。

    每个时间点可能被多个 4 帧窗口覆盖，平均后得到 `(T,6)`，其中第 0 帧 delta 为 0。
    """
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
    参考视频动作头模式（TimesFormer ViT）：
    - 输入长度 `T>=4` 的 RGB uint8 帧序列；
    - 按 stride 滑动 4 帧窗口；
    - 每个窗口预测后 3 帧 delta，再聚合成逐帧 delta；
    - 返回 `frames[1:]` 对应动作，API 顺序为 `[dx,dy,dz,droll,dyaw,dpitch]`，单位 cm/deg。
    """
    if np is None:
        raise RuntimeError("actionhead 模式需要 numpy")
    if _ah_model is None or _ah_stats is None or _ah_preprocess is None:
        raise RuntimeError("actionhead 模型尚未初始化")
    if len(frames_rgb_uint8) < 4:
        return []

    # 可选的中间 resize（调试桥接）：
    # 有些流水线会先强制 480p -> 256x256，再替换到原生 480p actionhead。
    # 注意：当前参考 actionhead checkpoint 是用 img_size=(192,640) 训练的，
    # 所以这里只是预 resize；经过 _ah_preprocess 后，模型仍接收 192x640。
    if int(pre_resize_hw) <= 0:
        env_pre = os.environ.get("ACTIONHEAD_PRE_RESIZE_HW", "").strip()
        if env_pre:
            try:
                pre_resize_hw = int(env_pre)
            except Exception:
                pre_resize_hw = 0
    # 默认不做中间预 resize，直接把 848x480 预处理到 actionhead 输入尺寸。

    # 预处理成归一化的 `(C,H,W)` tensor。
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
        # 堆叠成 `(C,T,H,W)`。
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

    # 只为 `frames[1:]` 把逐帧 delta 转成 API 动作，单位 cm/deg。
    # 这里假设模型内部 delta 格式为 `[dz, dy, dx, tx, ty, tz]`，
    # 角度单位 rad，平移单位 meter。
    # 公式：meter -> cm: *100（tx,ty,tz -> dx_cm,dy_cm,dz_cm）
    # 公式：rad -> deg: *180/pi（dz,dy,dx -> droll_deg,dyaw_deg,dpitch_deg）
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
        title="InfinityStar + TSFormer 动作推理 API",
        description="把 InfinityStar 预测出的 summed_codes（latent 世界转移）转换成动作增量（cm/deg）。默认 tsformer_latent 模式会走 Stage-2 latent-to-action：decoder 中间特征 -> adapter token -> TimesFormer 滑窗 -> 16 个动作。",
        version="0.1.0",
    )

    class PredictDeltaActionsRequest(BaseModel):
        """`/v1/predict_delta_actions` 请求体：同一 session 分多轮上传真实观测帧。"""

        session_id: str = Field(..., description="轨迹 / session 标识符；同一条路线的多次请求必须保持一致。")
        instruction: Optional[str] = Field(None, description="导航指令 / prompt；首次请求必须提供，后续如需改写 prompt 也可再次传入。")
        prompt: Optional[str] = Field(None, description="`instruction` 的兼容别名。")
        negative_prompt: Optional[str] = Field("", description="可选的负向提示词。")
        images_base64: List[str] = Field(..., description="以 base64 编码的 RGB 图像列表；首次通常只传 1 帧预热图像，后续通常每次传 16 帧真实新观测。")
        reset_session: bool = Field(
            False,
            description="若为 true，即使 session_id 相同也强制开启新的内部 run；服务端会丢弃旧内存状态，并生成新的内部 session 避免覆盖旧缓存。",
        )
        action_head_mode: str = Field(
            "tsformer_latent",
            description=(
                "动作头模式。"
                "`tsformer_latent`（默认）：Stage-2 latent-to-action，走 decoder 中间特征 -> adapter token -> TimesFormer 滑窗，每个 16 帧 segment 输出 16 个动作。"
                "`actionhead_ref_vit`：先把 Infinity 预测 latent 解码成 RGB 视频，再跑 4 帧滑窗 ViT（stride=1，并聚合重叠窗口）输出 16 个动作。"
            ),
        )
        action_head_batch_size: int = Field(8, description="`actionhead_ref_vit` 滑窗推理时的批大小。")
        action_head_stride: int = Field(1, description="`actionhead_ref_vit` 滑窗推理的步长；默认 1 表示覆盖所有相邻窗口。")
        action_head_pre_resize_hw: int = Field(
            0,
            description=(
                "`actionhead_ref_vit` 预处理前的可选中间 resize。"
                "如果 >0，会先把每一帧解码 RGB 缩放到 `(N,N)`（例如 256x256）；0 表示关闭。"
                "注意：参考 actionhead 模型后续仍会在内部再 resize 到 `(192,640)`，并且不做裁剪，以对齐 `predict_reference_videos_batch*.py`。"
            ),
        )
        allow_future_segments: bool = Field(
            False,
            description=(
                "若为 true，服务端在真实前缀到达 `points[i]` 时就允许提前发射第 i 段动作，"
                "而不是等到 `points[i+1]`。这对应更严格的闭环协议："
                "先发 1 帧 + prompt -> 得到下一段动作 -> 执行动作收集真实新帧 -> 再把真实新帧回传。"
            ),
        )
        prefix_mode: bool = Field(
            False,
            description="若为 true，`images_base64` 每次都携带完整前缀 `[1..K]`；服务端只会追加新增尾帧，避免重复写入历史帧。",
        )
        allow_future_last_segment: bool = Field(
            False,
            description="若为 true，最后一个 segment 可以在真实前缀只到达 `points[seg]` 时就提前输出，而不强制要求到达 `points[seg+1]`。这对应“最后一段由模型预测补齐”的语义。",
        )
        seed: Optional[int] = Field(
            None,
            description="可选采样基准 seed。若不传则默认使用 0；当 `lock_seed_across_steps=true` 时，同一 session 的所有 segment 都会复用这一个 seed。",
        )
        debug: bool = False

    class PredictDeltaActionsResponse(BaseModel):
        """动作预测响应：包含本次新输出的 segment、动作列表和 session 进度。"""

        actions: List[List[float]] = Field(
            ...,
            description="动作增量列表；每个动作都是 `[dx_cm,dy_cm,dz_cm,droll_deg,dyaw_deg,dpitch_deg]`。长度由 `action_head_mode` 决定，默认 `tsformer_latent` 每段输出 16 个动作。",
        )
        segment_index: int = Field(
            ...,
            description="本次输出对应哪个 segment，范围是 `0..S-1`（其中 `S=len(points)-1`）；如果为 `-1`，表示这次请求还没有新 segment 可发射。",
        )
        num_received_frames: int
        prefix_latents: int
        done: bool
        used_prompt: Optional[str] = None

    @app.get("/health")
    async def health():
        """健康检查接口，返回模型加载状态、配置帧数、segment points 和活跃 session 数。"""
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
            "ts_ckpt_loaded": _s2_tsformer is not None,
            "stage2_ckpt": _s2_ckpt_path,
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
        可选预加载：如果环境变量已经设置好，服务启动时就加载权重。

        这样即使第一个请求还没到，模型也已经驻留在进程里。
        """
        cfg = _get_server_config()
        if not cfg.infinity.ckpt:
            # 允许无 ckpt 启动服务；请求到达时如果仍没配置，会快速失败。
            print("[Service] 启动时未设置 InfinityStar ckpt，跳过启动时预加载。")
            return
        _init_models(cfg=cfg)

    @app.post("/v1/predict_delta_actions", response_model=PredictDeltaActionsResponse)
    async def predict_delta_actions(req: "PredictDeltaActionsRequest"):
        """FastAPI 入口：串行化 GPU 推理后转入 `_predict_delta_actions_impl()` 主流程。"""
        # 全局锁：保证单进程内 GPU 推理串行执行。
        if _LOCK is not None:
            async with _LOCK:
                return _predict_delta_actions_impl(req)
        return _predict_delta_actions_impl(req)

else:
    app = None  # type: ignore


def _predict_delta_actions_impl(req) -> "PredictDeltaActionsResponse":
    """
    `/v1/predict_delta_actions` 的主请求处理函数。

    中文阅读顺序：
    1. 解析/创建 `session_id` 对应的 `TrajectoryState`。
    2. 将本次上传的真实 RGB 帧 append 到 `frames_cpu`。
    3. 根据 `points=[1,17,33,49,...]` 判断下一个 segment 是否可输出动作。
    4. 调用世界模型得到 `summed_codes`，再通过动作头输出 6D 增量动作。
    5. 只有在真实帧已经到达 segment 末端时，才把真实观测写回 `gt_obs` cache。

    这个函数是 HTTP API 和模型内部状态之间的“总调度器”，可以按下面四层理解：

    第一层：session 管理
    - 外部客户端只知道 `req.session_id`；
    - 服务端可能把它映射到一个内部 run id，避免用户复用同名 session 时覆盖旧 cache；
    - `TrajectoryState` 保存这条路线的 prompt、真实帧、Infinity streaming session、
      KV cache、上一段 latent 边界和已输出到哪一段。

    第二层：真实帧入库
    - `images_base64` 先解码成 PIL RGB；
    - 再按 Infinity 动态分辨率模板 resize/normalize 到 `[-1,1]`；
    - 最后追加到 `st.frames_cpu`。这里保存的是“真实观测”，不是模型预测帧。

    第三层：segment ready 判定
    - 默认模式：只有真实帧数 `n >= points[seg+1]`，才输出第 `seg` 段动作；
    - future 模式：如果 `allow_future_segments=True`，可以在 `n >= points[seg]`
      时提前输出动作，但不能推进 `gt_obs` cache；
    - 最后一段 future 特例由 `allow_future_last_segment` 控制。

    第四层：动作头选择
    - `tsformer_latent`：默认路径，直接把 latent 喂给 Stage2 latent2action；
    - `actionhead_ref_vit`：先把 latent 解成 RGB 视频，再走参考视频动作头。

    最重要的安全规则：
    只有真实帧已经到达右边界时，才允许 `_infer_latents_for_actions_and_advance_cache()`
    把 `gt_obs` cache 推进到 `points[seg+1]`。提前发射动作时，cache 仍停留在真实前缀，
    不能把预测结果当作真实历史。
    """
    cfg = _get_server_config()
    if not cfg.infinity.ckpt:
        raise HTTPException(status_code=500, detail="必须提供 InfinityStar ckpt（在 config.json 或 INFINITY_CKPT 环境变量中设置）")
    _init_models(cfg=cfg)

    # 外部 session_id 来自客户端；内部 session_id 可能加时间戳后缀，用来区分同名多次 run。
    external_session_id = (req.session_id or "").strip()
    if not external_session_id:
        raise HTTPException(status_code=400, detail="必须提供 session_id")

    # `instruction` 和 `prompt` 是兼容字段；首次请求必须至少提供一个。
    raw_prompt = (req.instruction or "").strip() or (req.prompt or "").strip()
    allow_future_segments = bool(getattr(req, "allow_future_segments", False))
    # 自动“新 run”规则：避免复用旧内存状态造成轨迹串线。
    # 如果前端用“1 帧 + prompt/instruction”启动路线，即使外部 session_id 相同，
    # 服务端也把它视为新的内部 run。
    auto_reset_on_one_frame = os.environ.get("INFINITY_RESET_SESSION_ON_ONE_FRAME", "1").strip() in ("1", "true", "True")
    one_frame_with_prompt = bool(raw_prompt) and int(len(getattr(req, "images_base64", []) or [])) == 1
    want_reset = bool(getattr(req, "reset_session", False)) or (auto_reset_on_one_frame and one_frame_with_prompt)
    if want_reset and not raw_prompt:
        raise HTTPException(status_code=400, detail="reset_session 需要 instruction/prompt")

    if want_reset:
        old_key = _SESSION_ALIAS.get(external_session_id, external_session_id)
        try:
            if old_key in _TRAJ:
                del _TRAJ[old_key]
        except Exception:
            pass
        try:
            # 同时清理早期版本直接挂在 external_session_id 下的状态。
            if external_session_id in _TRAJ and external_session_id != old_key:
                del _TRAJ[external_session_id]
        except Exception:
            pass
        # 建立 alias：客户端继续使用原 session_id，服务端内部使用新的 run id。
        _SESSION_ALIAS[external_session_id] = _make_run_session_id(external_session_id)

    session_id = _SESSION_ALIAS.get(external_session_id, external_session_id)
    if session_id not in _TRAJ and not raw_prompt:
        raise HTTPException(status_code=400, detail="session 的第一次调用必须提供 instruction/prompt")

    # 获取/创建 TrajectoryState。后续所有状态变化都挂在 st 上，而不是直接挂在 request 上。
    st = _get_or_create_traj(session_id, raw_prompt, req.negative_prompt or "")

    # 解码并追加本次上传的图像帧。
    if not req.images_base64:
        raise HTTPException(status_code=400, detail="必须提供 images_base64")

    new_imgs: List[Image.Image] = []
    try:
        for s in req.images_base64:
            new_imgs.append(_load_image_from_base64(s))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解码 images_base64 失败: {e}")

    # 重要：
    # 很多 UAVFlow 风格数据集把原始帧存成 256x256，但推理时仍必须使用训练期固定模板
    # （例如 h_div_w_template=0.562 -> 848x480）。
    # 因此默认不会用原始帧宽高比覆盖 `st.h_div_w_template`。
    #
    # 如果确实想从第一帧宽高比自动检测模板，可通过环境变量开启：
    # 中文说明：INFINITY_AUTO_H_DIV_W_TEMPLATE=1
    if st.num_frames() == 0 and os.environ.get("INFINITY_AUTO_H_DIV_W_TEMPLATE", "0").strip() in ("1", "true", "True"):
        w, h = new_imgs[0].size
        if w > 0 and h > 0:
            st.h_div_w_template = float(h) / float(w)

    _ensure_traj_infinity_session(st)

    # 只确定一次 `(tgt_h,tgt_w)`；schedule 来自配置的 num_frames。
    if st.tgt_h is None or st.tgt_w is None:
        assert st.stream is not None
        sched = st.stream.build_schedule_for_num_frames(int(cfg.infinity.num_frames))
        st.tgt_h, st.tgt_w = int(sched.tgt_h), int(sched.tgt_w)
        # 可选目标分辨率硬检查；强制 640x640 等模板时便于早发现配置错误。
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
                # 环境变量格式错误时忽略，保持服务可启动。
                pass

    # prefix_mode 下客户端每次发送完整前缀；这里只保留新增尾帧。
    # 例：服务端已有 17 帧，本次客户端传 33 帧完整前缀，则这里只追加第 18..33 帧。
    if bool(getattr(req, "prefix_mode", False)):
        already = int(st.num_frames())
        if already > int(len(new_imgs)):
            raise HTTPException(
                status_code=400,
                detail=f"prefix_mode 要求 prefix 长度不能减少；服务端已有 {already} 帧，本次请求只有 {len(new_imgs)} 帧",
            )
        new_imgs = new_imgs[already:]

    # 把新增帧 resize/normalize 到目标尺寸和 `[-1,1]`，再存到 CPU。
    for pil in new_imgs:
        if st.num_frames() >= int(cfg.infinity.num_frames):
            break
        if infinity_transform is None:
            raise HTTPException(status_code=500, detail="尚未导入 InfinityStar modules（请检查 INFINITY_REPO_ROOT）")
        fr = infinity_transform(pil, int(st.tgt_h), int(st.tgt_w))  # type: ignore[misc]  # [3,H,W] in [-1,1]
        st.frames_cpu.append(fr.cpu())

    n = st.num_frames()
    done = n >= int(cfg.infinity.num_frames)

    points = cfg.infinity.points()
    if len(points) < 2:
        raise HTTPException(status_code=500, detail=f"配置 points 非法：{points}（num_frames={cfg.infinity.num_frames}, step={cfg.infinity.step}）")

    # 预热：第一帧用于准备 first-frame condition。
    # 默认预热请求不返回动作。
    # 如果开启 allow_future_segments，则继续向下执行，可能仅凭第一帧就输出 seg0 动作。
    if n == 1 and st.last_emitted_segment < 0:
        try:
            _prepare_firstframe_condition_if_needed(st)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"InfinityStar 预热失败: {e}")
        if not allow_future_segments:
            return PredictDeltaActionsResponse(
                actions=[],
                segment_index=-1,
                num_received_frames=n,
                prefix_latents=0,
                done=done,
                used_prompt=st.prompt_raw if req.debug else None,
            )

    # 下一段永远是“上一段已输出 + 1”。服务端不会跳段输出，避免客户端动作序列错位。
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

    # Segment 是否 ready：
    # 代码/形状说明：- points = [1, 1+step, 1+2*step, ..., num_frames]；
    # - 默认公式：ready_default = n >= points[seg+1]，
    #   即已收到帧数 n 到达当前 segment 的右边界时才输出该段动作；
    # - 默认：真实前缀必须到达 points[seg+1]（例如 49）才输出 seg2；
    # - 特例：最后一个 segment 可以在前缀到达 points[seg]（例如 33）时输出，
    #   因为 34..49 帧本来就是预测出来的。
    seg = int(next_seg)
    is_last_seg = int(seg) == (len(points) - 2)
    ready_default = n >= int(points[seg + 1])
    ready_last_future = bool(getattr(req, "allow_future_last_segment", False)) and is_last_seg and n >= int(points[seg])
    ready_future = bool(allow_future_segments) and n >= int(points[seg])
    if not (ready_default or ready_last_future or ready_future):
        # 还没到发射条件：返回空动作，让客户端继续上传真实帧。
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

    # 选择动作头模式。
    # 默认走 latent 动作头，因为它不需要把世界模型 latent 解码成视频，延迟和显存更可控。
    mode = str(getattr(req, "action_head_mode", "tsformer_latent") or "tsformer_latent").strip().lower()
    if mode in ("", "default", "tsformer_latent"):
        env_mode = os.environ.get("ACTION_HEAD_MODE", "").strip().lower()
        if env_mode:
            mode = env_mode
    use_actionhead_ref_vit = mode in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead")

    # InfinityStar 闭环推理：生成 latent，必要时解码预测视频；当条件允许时，
    # 把 gt_obs cache 推进到新暴露的真实前缀 `points[seg+1]`。
    try:
        # 只有真实帧确实到达 points[seg+1] 时才推进 GT cache。
        # 如果是提前输出 future tail，不能把非真实帧写入 GT cache。
        #
        # 读代码时重点看 `advance_gt`：
        # - True：本段右边界已有真实帧，推理后可以把真实前缀写入 gt_obs；
        # - False：本段是提前输出，推理后只能更新动作和 last_latent，不能污染真实 cache。
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
        )  # 逐段推理结果：包含 latent5、summed_codes、可选预测视频和帧边界信息。
    except Exception as e:
        print("[Service] _infer_latents_for_actions_and_advance_cache 失败。")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"InfinityStar 推理失败: {e}")

    actions: List[List[float]] = []
    if use_actionhead_ref_vit:
        # ActionHead 参考视频模式：预测视频解码 -> 4 帧滑窗 -> 聚合成逐帧动作。
        ckpt_path = os.environ.get("ACTIONHEAD_CKPT", "").strip() or os.environ.get("ACTIONHEAD_REF_CKPT", "").strip()
        run_cfg = os.environ.get("ACTIONHEAD_RUN_CONFIG", "").strip() or os.environ.get("ACTIONHEAD_REF_RUN_CONFIG", "").strip()
        if not ckpt_path or not run_cfg:
            # 向后兼容桥接：
            # 有些前端总是发送 actionhead_ref_vit，但服务端实际可能希望运行新的
            # Stage2 latent2action 路径（decoder 特征 -> adapter token -> TimesFormer）。
            # 如果配置了 Stage2 checkpoint，就回退到该路径；否则保留原始错误。
            s2_ckpt = os.environ.get("STAGE2_LATENT2ACTION_CKPT", "").strip() or (DEFAULT_STAGE2_LATENT2ACTION_CKPT or "")
            if s2_ckpt:
                try:
                    print("[Service] actionhead_ref_vit 缺少 ACTIONHEAD_CKPT/RUN_CONFIG；回退到 stage2 latent2action（16 个动作）。")
                    _init_stage2_latent2action_models(ckpt_path=s2_ckpt)
                    actions = _stage2_predict_16_actions_for_segment_cm_deg(st=st, infer_res=infer_res, stride=1)
                    # 重要：兜底路径已成功，不要继续初始化/运行 actionhead_ref_vit。
                    st.last_emitted_segment = seg
                    return PredictDeltaActionsResponse(  # type: ignore[name-defined]
                        actions=actions,
                        segment_index=seg,
                        num_received_frames=n,
                        prefix_latents=int(prefix_latents_abs),
                        done=bool(done or ((ready_last_future or (ready_future and is_last_seg)) and is_last_seg)),
                        used_prompt=st.prompt_raw if getattr(req, "debug", False) else None,
                    )
                except Exception as e:
                    print("[Service] actionhead_ref_vit 的 stage2 兜底也失败。")
                    print(traceback.format_exc())
                    raise HTTPException(
                        status_code=500,
                        detail=f"actionhead_ref_vit 需要环境变量 ACTIONHEAD_CKPT 和 ACTIONHEAD_RUN_CONFIG；stage2 兜底也失败: {e}",
                    )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="actionhead_ref_vit 需要环境变量 ACTIONHEAD_CKPT 和 ACTIONHEAD_RUN_CONFIG（或 ACTIONHEAD_REF_CKPT/ACTIONHEAD_REF_RUN_CONFIG）",
                )
        try:
            _init_actionhead_model(ckpt_path=ckpt_path, run_config_path=run_cfg)
            pred_vid = infer_res.pred_vid_bgr
            if pred_vid is None:
                # 尽力兜底；need_pred_video=True 时理论上不该走到这里。
                assert st.stream is not None
                with torch.no_grad():
                    pred_vid = st.stream.infinity.summed_codes2images(st.stream.vae, infer_res.summed_codes)
            # 重要：为了对齐 `predict_reference_videos_batch*.py` 在 clip 边界处的行为，
            # 必须在 clip 起点前最多提供 `(window_size-1)=3` 帧历史。
            # 否则 clip 开头几个 delta 覆盖窗口不足，会和离线脚本不一致。
            #
            # 对 seg i（points=[1,17,33,49]）：
            # - 输出转移 `[obs_len->obs_len+1 .. next_obs_len-1->next_obs_len]`，共 16 个动作；
            # - actionhead 输入帧取绝对帧 `[ctx_start .. next_obs_len]`，
            #   其中 `ctx_start = max(1, (obs_len+1)-3) = max(1, obs_len-2)`。
            obs_len = int(infer_res.obs_len)
            next_obs_len = int(infer_res.next_obs_len)
            clip_abs_start = int(obs_len) + 1
            clip_abs_end = int(next_obs_len)
            ctx_start_abs = max(1, int(clip_abs_start) - 3)

            frames_rgb: List["np.ndarray"] = []  # type: ignore[name-defined]
            for abs_i in range(int(ctx_start_abs), int(clip_abs_end) + 1):
                # 如果本地已有真实帧，优先用真实帧；前缀观测应与离线脚本一致。
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

            # 精确切出当前 clip 的 16 个动作。
            # frames_rgb 中的转移是连续的；动作下标对应“到达帧”位置减 1。
            start_idx = int(obs_len) - int(ctx_start_abs)
            end_idx = int(start_idx) + (int(clip_abs_end) - int(obs_len))
            actions = actions_all[int(start_idx) : int(end_idx)]
            if len(actions) != int(clip_abs_end) - int(obs_len):
                raise ValueError(
                    f"actionhead actions 长度不匹配：实际 {len(actions)}，需要 {int(clip_abs_end)-int(obs_len)}"
                )
        except HTTPException:
            raise
        except Exception as e:
            print("[Service] actionhead_ref_vit 推理失败。")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"actionhead_ref_vit 推理失败: {e}")
    else:
        # 代码/形状说明：Stage2 latent2action: decoder-features -> adapter tokens -> TimesFormer sliding windows -> 16 actions (cm/deg)
        try:
            _init_stage2_latent2action_models(ckpt_path=os.environ.get("STAGE2_LATENT2ACTION_CKPT", "").strip() or DEFAULT_STAGE2_LATENT2ACTION_CKPT)
            actions = _stage2_predict_16_actions_for_segment_cm_deg(st=st, infer_res=infer_res, stride=1)
        except Exception as e:
            print("[Service] Stage2 latent2action 推理失败。")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"stage2 latent2action 推理失败: {e}")

    st.last_emitted_segment = seg
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("未安装 FastAPI/pydantic，服务模式不可用。")
    return PredictDeltaActionsResponse(  # type: ignore[name-defined]
        actions=actions,
        segment_index=seg,
        num_received_frames=n,
        prefix_latents=int(prefix_latents_abs),
        done=bool(done or ((ready_last_future or (ready_future and is_last_seg)) and is_last_seg)),
        used_prompt=st.prompt_raw if getattr(req, "debug", False) else None,
    )


# -------------------------
# 中文说明：8) 内部 self-test（不走 HTTP）
# -------------------------
def _self_test(
    *,
    infinity_ckpt: str,
    route_dir: str,
    ts_ckpt: str,
    ts_stats: str,
    prompt_key: str = "instruction",
) -> None:
    """
    无 HTTP 的内部自测入口。

    它读取一条 route 的 meta/images，模拟客户端 `1, step, step, ...` 上传节奏，
    直接调用 `_predict_delta_actions_impl()`，用于检查权重路径、配置和动作输出长度。
    """
    route_dir = os.path.abspath(route_dir)
    meta_path = os.path.join(route_dir, "meta.json")
    images_dir = os.path.join(route_dir, "images")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到 meta.json: {meta_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"找不到 images 目录: {images_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = str(meta.get(prompt_key) or meta.get("instruction_unified") or meta.get("instruction") or meta.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"meta.json 中找不到 prompt（key={prompt_key}）")

    paths = _sorted_image_paths(images_dir)
    if not paths:
        raise FileNotFoundError("没有找到图像")

    cfg0 = _get_server_config()
    num_frames = int(cfg0.infinity.num_frames)
    step = int(cfg0.infinity.step)
    # 按配置的 num_frames 补齐或截断，保证自测可复现。
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

    # 模拟 streaming 协议：先发 1 帧，再按 `step` 分块直到 num_frames。
    chunks: List[List[str]] = []
    chunks.append(paths[:1])
    idx = 1
    while idx < num_frames:
        chunks.append(paths[idx : min(num_frames, idx + step)])
        idx += step
    for i, ch in enumerate(chunks):
        imgs = [Image.open(p).convert("RGB") for p in ch]
        # 临时转成 base64 后复用内部实现，确保自测走同一条代码路径。
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
    """只初始化 TSformer 权重和统计量，跳过 InfinityStar。"""
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
    `actions` 是 cm/deg 单位的 6D delta 列表。这里用简单累加得到相对位姿。

    返回字段：
    - 起点位姿 `start_pose`: `[0,0,0,0,0,0]`
    - `poses`: 长度等于动作数，表示每个动作后的位姿
    - `final_pose`: 最终位姿
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
    不加载 InfinityStar 权重的离线评估路径：
    - 从 route_dir 读取 meta.json 和 video_summed_codes.npy；
    - 按前 `take_first_pixel_frames` 帧切分 summed_codes，`pt = (N-1)//4 + 1`；
    - 生成 20 个动作（5 个 segment * 4 个动作）并积分成位姿；
    - 在 out_dir 下写出两个 JSON 文件。

    返回输出目录路径。
    """
    route_dir = os.path.abspath(route_dir)
    meta_path = os.path.join(route_dir, "meta.json")
    summed_path = os.path.join(route_dir, "reshape_actionhead_data", "video_summed_codes.npy")
    images_dir = os.path.join(route_dir, "images")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到 meta.json: {meta_path}")
    if not os.path.exists(summed_path):
        raise FileNotFoundError(f"找不到 video_summed_codes.npy: {summed_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"找不到 images 目录: {images_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = str(meta.get(prompt_key) or meta.get("instruction_unified") or meta.get("instruction") or meta.get("prompt") or "").strip()

    cfg0 = _get_server_config()
    if take_first_pixel_frames is None:
        take_first_pixel_frames = int(cfg0.infinity.num_frames)

    # 即使这里不运行 Infinity，也要求有 >=N 张图像，以保持 streaming 语义一致。
    img_paths = _sorted_image_paths(images_dir)
    if len(img_paths) < int(take_first_pixel_frames):
        raise ValueError(f"离线评估至少需要 {take_first_pixel_frames} 张图像，实际只有 {len(img_paths)} 张")

    import numpy as np

    z = np.load(summed_path)  # 中文说明：期望形状为 (1,16,T_lat,H,W)
    if z.ndim != 5 or z.shape[0] != 1 or z.shape[1] != 16:
        raise ValueError(f"summed_codes 形状不符合预期：{z.shape}（期望 (1,16,T,H,W)）")

    # 按前 N 个像素帧切到对应 latent 长度 `pt`：
    # latent_index(frame f) = (f - 1)//temporal_compress_rate + 1，这里 temporal_compress_rate=4。
    pt = (int(take_first_pixel_frames) - 1) // 4 + 1
    if z.shape[2] < pt:
        raise ValueError(f"summed_codes 时间长度不足: T_lat={z.shape[2]}，但需要 pt={pt}")
    z = z[:, :, :pt]

    summed_codes = torch.from_numpy(z).to(_DEVICE, dtype=torch.float32)

    _init_tsformer_only(ts_ckpt=ts_ckpt, ts_stats=ts_stats)

    # 按 points(num_frames, step) 模拟 segment 推进。
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
        # 代码/形状说明：poses 的长度等于动作数；每个点都是“执行完该动作后的绝对位姿”。
        # 起点位姿单独放在 `start_pose`，避免和第一个动作后的位姿混淆。
        **poses_info,
        "note": "poses 的长度等于动作数；每个 pose 都表示执行完对应动作后的绝对位姿。起点位姿单独保存在 start_pose。",
    }

    with open(os.path.join(out_run, "actions.json"), "w", encoding="utf-8") as f:
        json.dump(actions_json, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_run, "relative_poses.json"), "w", encoding="utf-8") as f:
        json.dump(poses_json, f, ensure_ascii=False, indent=2)

    return out_run


def main():
    """
    命令行入口。

    支持完整权重 self-test，也支持只用预计算 `video_summed_codes.npy` 跑旧 TSformer(P2P)
    离线评估；真正线上服务通常由 `uvicorn server:app` 启动。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument(
        "--offline_eval_precomputed",
        action="store_true",
        help="使用 route_dir/reshape_actionhead_data/video_summed_codes.npy 做离线评估；不需要加载 InfinityStar 权重。",
    )
    ap.add_argument("--infinity_ckpt", type=str, default=os.environ.get("INFINITY_CKPT", ""))
    ap.add_argument("--route_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "cache"), help="离线评估 JSON 输出目录。")
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
        print(f"[offline_eval_precomputed] 已写出 JSON 文件到：{out_run}")


if __name__ == "__main__":
    main()
