#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
真实 UAV-Flow 远端闭环 rollout 生成器。

阅读顺序建议：
1. `main()`：读取候选 jsonl、初始化 `grpo_server.py`、做 shard 分发与失败重试。
2. `_run_remote_sim_rollout()`：执行一次真实闭环，流程是
   `reset 返回 1 帧 -> 本地动作头预测 1 段动作 -> 模拟器执行并返回 16 帧 -> 重复 3 段`。
3. `trajectory.json` 组装：把动作、相对位姿、old logprob、trace 文件和模拟器元数据落盘，
   供 `reward_uavflow.py`/StageB replay 继续消费。

关键约定：
- 第 0 段才发送 `instruction`，后续段复用同一 `session_id` 里的语言/latent 状态；
- `/reset` 必须返回 1 帧预热观测，前两次 `/step_actions` 必须各返回 16 帧真实观测；
- `sample_logprob_segments` 记录的是 StageA rollout 时的 old logprob，不是 StageB 训练时重算的 new logprob。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import types
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

SIM_RESET_FRAMES = 1
SIM_SEGMENTS_PER_TRAJ = 3
SIM_STEP_OBS_FRAMES = 16

def _count_candidates(candidates_jsonl: str, num_shards: int, shard_id: int) -> Tuple[int, int]:
    """统计全局候选数量和当前 shard 负责的候选数量。"""
    total_global = 0
    total_shard = 0
    global_idx = -1
    with open(os.path.abspath(candidates_jsonl), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            global_idx += 1
            total_global += 1
            if (global_idx % max(1, int(num_shards))) == int(shard_id):
                total_shard += 1
    return total_global, total_shard


def _load_api_module(py_path: str):
    """动态加载 grpo_server.py，使 rollout 脚本可直接调用内部推理实现。"""
    spec = importlib.util.spec_from_file_location("rl_infinity_api_mod", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法从 {py_path} 加载 rollout API 模块")
    mod = importlib.util.module_from_spec(spec)
    # 目标模块里的 dataclass 在执行模块时会查 sys.modules[cls.__module__]，
    # 所以要先注册再 exec_module。
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _integrate_actions(actions: List[List[float]]) -> List[List[float]]:
    """把服务端返回的 6D 增量动作简单累加成相对 pose 序列。"""
    pose = [0.0] * 6
    out = []
    for a in actions:
        if len(a) != 6:
            continue
        pose = [pose[i] + float(a[i]) for i in range(6)]
        out.append(list(pose))
    return out


def _all_finite_6d(actions: List[List[float]]) -> bool:
    """校验动作列表非空、每项 6 维且所有数值有限。"""
    if not isinstance(actions, list) or len(actions) <= 0:
        return False
    for a in actions:
        if not (isinstance(a, list) and len(a) == 6):
            return False
        for v in a:
            try:
                x = float(v)
            except Exception:
                return False
            if not math.isfinite(x):
                return False
    return True


def _all_finite(xs: List[float]) -> bool:
    """校验一维数值列表全部可转 float 且有限。"""
    for v in xs:
        try:
            x = float(v)
        except Exception:
            return False
        if not math.isfinite(x):
            return False
    return True


def _require_trace_ce_ok(trace_paths: List[str]) -> None:
    """
    严格的 GRPO 安全检查：
    当 StageB 用 trace_ce 计算 new_logprob 时，StageA 也必须用同一定义记录 old_logprob
    （teacher-forcing CE/full-softmax）。如果 StageA 回退到采样时 logprob，PPO ratio/KL
    的定义会不一致，可能导致视频质量崩掉。
    """
    mode = (os.environ.get("INFINITY_STAGEA_OLD_LOGPROB_MODE", "") or "").strip().lower()
    strict = int((os.environ.get("INFINITY_STAGEA_OLD_LOGPROB_STRICT", "0") or "0").strip() or "0") == 1
    if not (mode == "trace_ce" and strict):
        return
    import torch  # 本地导入，保持脚本启动阶段更轻。

    for p in trace_paths:
        if not p or (not os.path.exists(p)):
            continue
        tr = torch.load(p, map_location="cpu")
        if tr.get("sample_logprob_trace_ce", None) is None:
            raise RuntimeError(f"trace_ce 严格模式下缺少 sample_logprob_trace_ce：{p}")


def _make_req(**kwargs):
    """构造类似 Pydantic request 的 SimpleNamespace，直接喂给 API 内部实现。"""
    obj = types.SimpleNamespace()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def _http_post_json(url: str, payload: Dict[str, Any], *, timeout_s: int) -> Dict[str, Any]:
    """向远端模拟器服务发送 JSON 请求。"""
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        str(url),
        data=raw,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"调用 {url} 时收到 HTTP {e.code}：{detail[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"请求 {url} 失败：{e}") from e
    obj = json.loads(body or "{}")
    if not isinstance(obj, dict):
        raise RuntimeError(f"{url} 返回的不是字典，而是 {type(obj)}")
    return obj


def _task_id_from_video_path(video_path: str) -> str:
    """从 video_path 的父目录名推断 UAV-Flow task_id。"""
    return os.path.basename(os.path.dirname(os.path.abspath(video_path)))


def _build_obj_info_from_task_json(task_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从任务 json 中提取 UnrealCV 目标物体放置信息。"""
    if "obj_id" not in task_json or "use_obj" not in task_json:
        return None
    if "target_pos" in task_json and isinstance(task_json["target_pos"], list) and len(task_json["target_pos"]) == 6:
        obj_pos = [float(x) for x in task_json["target_pos"][:3]]
        obj_rot = [float(x) for x in task_json["target_pos"][3:]]
    else:
        raw_pos = task_json.get("obj_pos", None)
        raw_rot = task_json.get("obj_rot", [0, 0, 0])
        if not (isinstance(raw_pos, list) and len(raw_pos) >= 3):
            return None
        obj_pos = [float(x) for x in raw_pos[:3]]
        obj_rot = [float(x) for x in (raw_rot[:3] if isinstance(raw_rot, list) else [0, 0, 0])]
    return {
        "use_obj": int(task_json["use_obj"]),
        "obj_id": int(task_json["obj_id"]),
        "obj_pos": obj_pos,
        "obj_rot": obj_rot,
    }


def _resolve_uavflow_task_meta(row: Dict[str, Any], video_path: str, task_json_root: str) -> Dict[str, Any]:
    """解析 rollout 行对应的任务 json，返回 instruction、初始位姿和物体信息。"""
    task_id = str(row.get("task_id", "") or "").strip() or _task_id_from_video_path(video_path)
    task_json_path = os.path.join(os.path.abspath(task_json_root), f"{task_id}.json")
    if not os.path.exists(task_json_path):
        raise FileNotFoundError(f"缺少 task_id={task_id} 对应的 UAV-Flow 任务 json：{task_json_path}")
    with open(task_json_path, "r", encoding="utf-8") as f:
        task_json = json.load(f)
    if not isinstance(task_json, dict):
        raise ValueError(f"任务 json 格式错误，期望 dict：{task_json_path}")
    initial_pos = task_json.get("initial_pos", None)
    if not (isinstance(initial_pos, list) and len(initial_pos) >= 6):
        raise ValueError(f"任务 json 中的 initial_pos 非法：{task_json_path}")
    instruction = str(task_json.get("instruction") or task_json.get("instruction_unified") or row.get("tarsier2_caption") or row.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"任务 json 中的 instruction 为空：{task_json_path}")
    return {
        "task_id": task_id,
        "instruction": instruction,
        "initial_pose": [float(x) for x in initial_pos[:6]],
        "obj_info": _build_obj_info_from_task_json(task_json),
        "task_json_path": task_json_path,
    }


def _sim_reset(
    *,
    base_url: str,
    session_id: str,
    traj_id: str,
    row: Dict[str, Any],
    prompt: str,
    seed: int,
    timeout_s: int,
    task_json_root: str,
) -> Dict[str, Any]:
    """调用模拟器服务 /reset，获得初始观测帧。"""
    video_path = str(row.get("video_path", "") or "")
    task_meta = _resolve_uavflow_task_meta(row, video_path, task_json_root)
    payload = {
        "session_id": str(session_id),
        "traj_id": str(traj_id),
        "prompt": str(prompt or task_meta["instruction"]),
        "seed": int(seed),
        "video_path": video_path,
        "gt_pose_json": str(row.get("gt_pose_json", "") or ""),
        "task_id": str(task_meta["task_id"]),
        "initial_pose": task_meta["initial_pose"],
        "obj_info": task_meta["obj_info"],
    }
    return _http_post_json(base_url.rstrip("/") + "/reset", payload, timeout_s=int(timeout_s))


def _sim_step_actions(
    *,
    base_url: str,
    session_id: str,
    actions: List[List[float]],
    segment_index: int,
    timeout_s: int,
) -> Dict[str, Any]:
    """调用模拟器服务 /step_actions，执行一个动作分段并取回新帧。"""
    payload = {
        "session_id": str(session_id),
        "actions": actions,
        "segment_index": int(segment_index),
    }
    return _http_post_json(base_url.rstrip("/") + "/step_actions", payload, timeout_s=int(timeout_s))


def _run_remote_sim_rollout(
    *,
    mod,
    row: Dict[str, Any],
    traj_id: str,
    session_id: str,
    prompt: str,
    seed: int,
    simulator_base_url: str,
    simulator_timeout_s: int,
    action_head_batch_size: int,
    action_head_stride: int,
    action_head_pre_resize_hw: int,
    task_json_root: str,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    使用真实远端模拟器执行 3 段闭环 rollout。

    数据流可以按下面的固定节奏理解：
    1. `/reset` 返回 1 帧初始观测，作用类似预热，让 session 建立语言条件和起始视觉状态。
    2. 第 0 段把这 1 帧送给本地 `grpo_server.py` 内部的 `_predict_delta_actions_impl()`，
       得到一段 6D 增量动作。
    3. 把该段动作发送给远端模拟器 `/step_actions`，让真实环境执行，并回传接下来 16 帧观测。
    4. 第 1/2 段继续用这 16 帧真实观测预测下一段动作，直到总共执行 3 段。

    为什么只在 `seg_i == 0` 传 `instruction`：
    - 服务端 session 已经会把文本条件缓存到本次 rollout 的内部状态；
    - 后续段真正变化的是新观测帧，而不是任务描述本身。
    """
    reset_resp = _sim_reset(
        base_url=simulator_base_url,
        session_id=session_id,
        traj_id=traj_id,
        row=row,
        prompt=prompt,
        seed=seed,
        timeout_s=int(simulator_timeout_s),
        task_json_root=task_json_root,
    )
    init_images = reset_resp.get("images_base64", None)
    if not isinstance(init_images, list) or len(init_images) != SIM_RESET_FRAMES:
        raise RuntimeError(f"模拟器 /reset 必须精确返回 {SIM_RESET_FRAMES} 帧初始观测")

    sim_world_poses: List[Any] = []
    if isinstance(reset_resp.get("world_poses", None), list):
        sim_world_poses.extend(reset_resp["world_poses"])

    responses: List[Any] = []
    seg_exec_meta: List[Dict[str, Any]] = []
    images_for_model = list(init_images)
    timings: Dict[str, float] = {}
    for seg_i in range(SIM_SEGMENTS_PER_TRAJ):
        t0 = time.perf_counter()
        req = _make_req(
            session_id=session_id,
            # 只在首段注入 instruction；后续段复用 session 内已缓存的语言条件。
            instruction=prompt if seg_i == 0 else None,
            prompt=None,
            negative_prompt="",
            # 首段是 1 帧预热观测，后续段则是模拟器刚执行完上一段动作后回传的 16 帧真实观测。
            images_base64=images_for_model,
            reset_session=(seg_i == 0),
            action_head_mode="actionhead_ref_vit",
            action_head_batch_size=int(action_head_batch_size),
            action_head_stride=int(action_head_stride),
            action_head_pre_resize_hw=int(action_head_pre_resize_hw),
            # 允许“看到这一段起点真实帧后，就直接预测下一段动作”，这是严格闭环路径的关键。
            allow_future_segments=True,
            prefix_mode=False,
            allow_future_last_segment=True,
            seed=int(seed),
            debug=False,
        )
        resp = mod._predict_delta_actions_impl(req)
        timings[f"seg{seg_i:02d}_sec"] = time.perf_counter() - t0
        responses.append(resp)

        seg_actions = getattr(resp, "actions", None)
        if not _all_finite_6d(seg_actions):
            raise RuntimeError(f"第 {seg_i} 段动作非法：存在空动作、非 6 维动作或 NaN/Inf")

        step_resp = _sim_step_actions(
            base_url=simulator_base_url,
            session_id=session_id,
            actions=[[float(v) for v in action[:6]] for action in seg_actions],
            segment_index=seg_i,
            timeout_s=int(simulator_timeout_s),
        )
        seg_exec_meta.append(
            {
                "segment_index": int(seg_i),
                "frame_indices": step_resp.get("frame_indices", []),
                "done": bool(step_resp.get("done", False)),
            }
        )
        if isinstance(step_resp.get("world_poses", None), list):
            sim_world_poses.extend(step_resp["world_poses"])
        if seg_i < (SIM_SEGMENTS_PER_TRAJ - 1):
            images_next = step_resp.get("images_base64", None)
            if not isinstance(images_next, list) or len(images_next) != SIM_STEP_OBS_FRAMES:
                raise RuntimeError(
                    f"模拟器第 {seg_i} 段执行后必须返回 {SIM_STEP_OBS_FRAMES} 帧真实观测，"
                    "供下一段动作预测使用"
                )
            images_for_model = list(images_next)

    task_meta = _resolve_uavflow_task_meta(row, str(row.get("video_path", "") or ""), task_json_root)
    return responses, {
        "sampling_backend": "remote_sim",
        "simulator_base_url": str(simulator_base_url),
        "simulator_session_id": str(session_id),
        "simulator_world_poses": sim_world_poses,
        "simulator_segment_exec": seg_exec_meta,
        "task_id": str(task_meta["task_id"]),
        "task_json_name": os.path.basename(str(task_meta["task_json_path"])),
        **timings,
    }


def _disable_api_cache_dump(mod) -> None:
    """关闭 API 内部大体积 latent/video 调试落盘，加快批量 rollout。"""
    # 关闭 API 服务端的大体积 debug/cache 写盘，节省离线 rollout 的时间和空间。
    def _noop(*_args, **_kwargs):
        """替换调试保存函数，保持 no-op 行为。"""
        return None

    if hasattr(mod, "_save_latent_tensor"):
        mod._save_latent_tensor = _noop
    if hasattr(mod, "_save_latent_video_clip"):
        mod._save_latent_video_clip = _noop
    if hasattr(mod, "_save_pred_video"):
        mod._save_pred_video = _noop
    if hasattr(mod, "infinity_save_video"):
        mod.infinity_save_video = None


def main():
    """CLI 入口：批量读取候选记录，生成真实模拟器 rollout trajectory.json。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_infinity_repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    default_package_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    default_repo_root = os.path.abspath(os.path.join(default_package_root, ".."))
    default_actionhead_repo_root = os.path.join(default_repo_root, "Worldmodel", "action_decoder", "actionhead_runtime")
    default_task_json_root = os.path.join(default_package_root, "data", "UAV-Flow-Eval", "test_jsons")

    ap = argparse.ArgumentParser(
        description=(
            "读取候选 jsonl，调用本地 grpo_server 动作预测逻辑，并通过远端 UAV-Flow "
            "模拟器执行真实闭环 rollout，最终为每个 traj_id 写出 trajectory.json。"
        )
    )
    ap.add_argument("--candidates_jsonl", type=str, required=True, help="候选 rollout 元数据 jsonl，由 generate_candidate_rollouts.py 生成。")
    ap.add_argument("--trajectory_root", type=str, required=True, help="输出根目录；每个 traj_id 会在这里生成独立子目录和 trajectory.json。")
    ap.add_argument("--api_py", type=str, required=True, help="`action_aware_grpo/grpo_server.py` 的路径；脚本会直接导入并调用其内部推理实现。")
    ap.add_argument("--infinity_server_config", type=str, required=True, help="`action_aware_grpo/config.json` 路径；用于初始化本地 grpo_server。")
    ap.add_argument("--infinity_ckpt", type=str, required=True, help="Infinity 世界模型 checkpoint 的路径。")
    ap.add_argument(
        "--infinity_repo_root",
        type=str,
        default=default_infinity_repo_root,
        help="Infinity repo 的根目录；供 grpo_server 导入 runtime/infinity 相关模块时使用。",
    )
    ap.add_argument("--actionhead_ckpt", type=str, required=True, help="StageB/action head checkpoint 的路径。")
    ap.add_argument("--actionhead_run_config", type=str, required=True, help="动作头运行配置 json 的路径。")
    ap.add_argument(
        "--actionhead_repo_root",
        type=str,
        default=default_actionhead_repo_root,
        help="actionhead runtime 代码库的根目录；供 grpo_server 导入动作头推理模块时使用。",
    )
    ap.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help=(
            "兼容保留参数。当前开源 `remote_sim` 路径并不会直接使用它；"
            f"真实闭环长度固定为 1 帧 reset + {SIM_SEGMENTS_PER_TRAJ} 段动作，前两段各回传 "
            f"{SIM_STEP_OBS_FRAMES} 帧观测。"
        ),
    )
    ap.add_argument("--action_head_batch_size", type=int, default=8, help="动作头推理批大小（batch size）；影响单段动作预测速度与显存占用。")
    ap.add_argument("--action_head_stride", type=int, default=1, help="动作头时间步采样步长；通常保持与训练/服务端默认一致。")
    ap.add_argument("--action_head_pre_resize_hw", type=int, default=0, help="动作头前处理的预缩放尺寸；0 表示沿用服务端默认值。")
    ap.add_argument("--dump_debug_cache", type=int, default=0, help="是否保留 API 侧调试缓存。1=保留 mp4/pt，0=禁用调试落盘以提高吞吐。")
    ap.add_argument("--failed_jsonl", type=str, default="", help="可选失败记录输出路径；每行保存一个失败 traj 的元数据与报错。")
    ap.add_argument("--timing_jsonl", type=str, default="", help="可选耗时记录输出路径；每行保存一个 traj 的各段耗时与总耗时。")
    ap.add_argument("--num_shards", type=int, default=1, help="并行 rollout 的总 shard 数。")
    ap.add_argument("--shard_id", type=int, default=0, help="当前进程负责的 shard 编号，范围是 [0, num_shards)。")
    ap.add_argument(
        "--max_retry",
        type=int,
        default=int(os.environ.get("INFINITY_ROLLOUT_MAX_RETRY", "3")),
        help="单条 rollout 最大重试次数；当动作/logprob/trace 出现 NaN、Inf 或缺失时会重新采样。",
    )
    ap.add_argument(
        "--retry_seed_step",
        type=int,
        default=int(os.environ.get("INFINITY_ROLLOUT_RETRY_SEED_STEP", "9973")),
        help="每次重试时对 seed 叠加的步长；用于重新采样，避免重复命中同一坏样本。",
    )
    ap.add_argument(
        "--progress_every_n",
        type=int,
        default=int(os.environ.get("STAGEA_PROGRESS_EVERY_N", "10")),
        help="每处理多少条候选打印一次进度。",
    )
    ap.add_argument(
        "--rollout_backend",
        type=str,
        default=str(os.environ.get("UAVFLOW_STAGEA_ROLLOUT_BACKEND", "remote_sim")),
        choices=["remote_sim"],
        help="rollout 后端。当前开源路径只保留 `remote_sim`：本地预测动作，远端 UAV-Flow 模拟器执行闭环。",
    )
    ap.add_argument(
        "--simulator_base_url",
        type=str,
        default=str(os.environ.get("UAVFLOW_SIMULATOR_BASE_URL", "http://127.0.0.1:8765")),
        help="远端模拟器服务的基础 URL；在 `--rollout_backend=remote_sim` 下生效。",
    )
    ap.add_argument(
        "--simulator_timeout_s",
        type=int,
        default=int(os.environ.get("UAVFLOW_SIMULATOR_TIMEOUT_S", "120")),
        help="与远端模拟器通信时的 HTTP 超时秒数。",
    )
    ap.add_argument(
        "--uavflow_task_json_root",
        type=str,
        default=str(os.environ.get("UAVFLOW_TASK_JSON_ROOT", default_task_json_root)),
        help="UAV-Flow-Eval 任务 json 目录，文件名格式为 `<task_id>.json`。",
    )
    args = ap.parse_args()

    os.environ["INFINITY_SERVER_CONFIG"] = os.path.abspath(args.infinity_server_config)
    os.environ["INFINITY_CKPT"] = os.path.abspath(args.infinity_ckpt)
    os.environ["INFINITY_REPO_ROOT"] = os.path.abspath(args.infinity_repo_root)
    os.environ["ACTIONHEAD_CKPT"] = os.path.abspath(args.actionhead_ckpt)
    os.environ["ACTIONHEAD_RUN_CONFIG"] = os.path.abspath(args.actionhead_run_config)
    os.environ["ACTIONHEAD_REPO_ROOT"] = os.path.abspath(args.actionhead_repo_root)
    os.environ["ACTION_HEAD_MODE"] = "actionhead_ref_vit"
    os.environ["INFINITY_DISABLE_P2P_LOAD"] = "1"
    if int(args.dump_debug_cache) != 1:
        os.environ["INFINITY_LATENT_CACHE_ROOT"] = os.path.abspath(args.trajectory_root)

    mod = _load_api_module(os.path.abspath(args.api_py))
    if int(args.dump_debug_cache) != 1:
        _disable_api_cache_dump(mod)
    os.makedirs(os.path.abspath(args.trajectory_root), exist_ok=True)
    failed_path = os.path.abspath(args.failed_jsonl) if str(args.failed_jsonl).strip() else ""
    failed_fp = None
    if failed_path:
        os.makedirs(os.path.dirname(failed_path), exist_ok=True)
        failed_fp = open(failed_path, "w", encoding="utf-8")
    timing_path = os.path.abspath(args.timing_jsonl) if str(args.timing_jsonl).strip() else ""
    timing_fp = None
    if timing_path:
        os.makedirs(os.path.dirname(timing_path), exist_ok=True)
        timing_fp = open(timing_path, "w", encoding="utf-8")

    # 先预热一次模型。
    cfg = mod._get_server_config()
    mod._init_models(cfg=cfg)

    n, ok = 0, 0
    failed_cnt = 0
    num_shards = max(1, int(args.num_shards))
    shard_id = int(args.shard_id)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"非法 shard_id={shard_id}，num_shards={num_shards}")
    total_global, total_shard = _count_candidates(args.candidates_jsonl, num_shards=num_shards, shard_id=shard_id)
    progress_every_n = max(1, int(args.progress_every_n))
    stage_start = time.perf_counter()
    print(
        f"[进度] shard={shard_id}/{num_shards} 分配到 {total_shard} 条，"
        f"全局共 {total_global} 条，progress_every_n={progress_every_n}"
    )
    global_idx = -1
    with open(os.path.abspath(args.candidates_jsonl), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            global_idx += 1
            if (global_idx % num_shards) != shard_id:
                continue
            n += 1
            row: Dict[str, Any] = json.loads(line)
            traj_id = str(row.get("traj_id", f"traj_{n:06d}"))
            sid = f"GRPO_{traj_id}"
            video_path = str(row.get("video_path", ""))
            prompt = str(row.get("tarsier2_caption", row.get("instruction", "")))
            seed = int(row.get("seed", 0))
            if not video_path or not prompt:
                continue
            try:
                t0_all = time.perf_counter()
                rollout_backend = str(args.rollout_backend).strip().lower()

                # 如果 actions/logprob/trace 出现非法值，就重试 rollout。
                # 这能处理模型推理产生 NaN/Inf，或 trace 偶发写盘失败的情况。
                max_retry = max(1, int(args.max_retry))
                seed_step = int(args.retry_seed_step)
                r0 = r1 = r2 = None
                dt0 = dt1 = dt2 = 0.0
                backend_meta: Dict[str, Any] = {"sampling_backend": rollout_backend}
                for attempt in range(max_retry):
                    seed_try = int(seed) + int(attempt) * int(seed_step)
                    sid_try = f"{sid}_try{attempt}"
                    try:
                        responses_try, backend_meta_try = _run_remote_sim_rollout(
                            mod=mod,
                            row=row,
                            traj_id=traj_id,
                            session_id=sid_try,
                            prompt=prompt,
                            seed=seed_try,
                            simulator_base_url=str(args.simulator_base_url),
                            simulator_timeout_s=int(args.simulator_timeout_s),
                            action_head_batch_size=int(args.action_head_batch_size),
                            action_head_stride=int(args.action_head_stride),
                            action_head_pre_resize_hw=int(args.action_head_pre_resize_hw),
                            task_json_root=str(args.uavflow_task_json_root),
                        )
                        if len(responses_try) != SIM_SEGMENTS_PER_TRAJ:
                            raise RuntimeError(
                                f"远端模拟器 rollout 返回了 {len(responses_try)} 段，"
                                f"预期应为 {SIM_SEGMENTS_PER_TRAJ} 段"
                            )
                        r0, r1, r2 = responses_try
                        backend_meta = dict(backend_meta_try)
                        dt0 = float(backend_meta.get("seg00_sec", 0.0))
                        dt1 = float(backend_meta.get("seg01_sec", 0.0))
                        dt2 = float(backend_meta.get("seg02_sec", 0.0))

                        actions_try: List[List[float]] = []
                        for rr in (r0, r1, r2):
                            aa = getattr(rr, "actions", None)
                            if isinstance(aa, list):
                                actions_try.extend(aa)

                        seg_oldlp_try: List[float] = []
                        seg_trace_try: List[str] = []
                        for rr in (r0, r1, r2):
                            try:
                                seg_oldlp_try.append(float(getattr(rr, "segment_old_logprob", 0.0) or 0.0))
                            except Exception:
                                seg_oldlp_try.append(0.0)
                            seg_trace_try.append(str(getattr(rr, "segment_trace_path", "") or ""))

                        # 基础有效性检查：actions/logprobs 必须有限，且至少有 trace 文件存在。
                        if (not _all_finite_6d(actions_try)) or (not _all_finite(seg_oldlp_try)):
                            raise RuntimeError("rollout 数值非法：存在 NaN/Inf 或非 6 维动作")
                        if not any((p and os.path.exists(p)) for p in seg_trace_try):
                            raise RuntimeError("缺少 trace 文件")
                        # 严格 trace_ce 检查：trace 文件必须包含 sample_logprob_trace_ce。
                        _require_trace_ce_ok(seg_trace_try)

                        # 本次尝试成功。
                        seed = seed_try
                        sid = sid_try
                        break
                    except Exception as e:
                        # 进入下一次重试。
                        r0 = r1 = r2 = None
                        if attempt == max_retry - 1:
                            raise RuntimeError(f"rollout 在重试 {max_retry} 次后仍失败：{e}") from e

                actions = []
                for rr in (r0, r1, r2):
                    aa = getattr(rr, "actions", None)
                    if isinstance(aa, list):
                        actions.extend(aa)
                poses = _integrate_actions(actions)
                seg_old_logprobs: List[float] = []
                seg_trace_paths: List[str] = []
                for rr in (r0, r1, r2):
                    try:
                        seg_old_logprobs.append(float(getattr(rr, "segment_old_logprob", 0.0) or 0.0))
                    except Exception:
                        seg_old_logprobs.append(0.0)
                    seg_trace_paths.append(str(getattr(rr, "segment_trace_path", "") or ""))

                out_dir = os.path.join(os.path.abspath(args.trajectory_root), traj_id)
                os.makedirs(out_dir, exist_ok=True)
                # 保持 trace_files 和 seg index (0..2) 对齐。不能只 append，
                # 否则缺失的 seg 会让索引错位，破坏 clip 对齐。
                trace_files: List[str] = [""] * SIM_SEGMENTS_PER_TRAJ
                for si, src in enumerate(seg_trace_paths):
                    if not src or (not os.path.exists(src)):
                        continue
                    dst = os.path.join(out_dir, f"seg{si:02d}_trace.pt")
                    try:
                        shutil.copy2(src, dst)
                        if 0 <= int(si) < len(trace_files):
                            trace_files[int(si)] = dst
                    except Exception:
                        continue
                # `sample_logprob_segments` 明确记录的是 StageA rollout 时的 old logprob，
                # StageB 做 replay 训练时会再基于 trace/teacher-forcing 计算 new logprob。
                payload = {
                    "traj_id": traj_id,
                    "seed": int(seed),
                    "candidate_id": int(row.get("candidate_id", 0)),
                    "video_path": video_path,
                    "prompt": prompt,
                    "num_actions": len(actions),
                    "actions": actions,
                    "relative_poses": {
                        "start_pose": [0.0] * 6,
                        "poses": poses,
                        "final_pose": (poses[-1] if poses else [0.0] * 6),
                    },
                    "sample_logprob_segments": seg_old_logprobs,
                    "sample_logprob_total": float(sum(seg_old_logprobs)),
                    "trace_files": trace_files,
                    "note": "真实闭环 rollout：本地 grpo_server/Infinity + actionhead_ref_vit 预测动作，远端模拟器执行并返回下一段真实观测。",
                    **backend_meta,
                }
                with open(os.path.join(out_dir, "trajectory.json"), "w", encoding="utf-8") as wf:
                    json.dump(payload, wf, ensure_ascii=False, indent=2)
                dt_all = time.perf_counter() - t0_all
                lp_total = float(sum(seg_old_logprobs))
                print(
                    f"[耗时] traj_id={traj_id} 总耗时={dt_all:.2f}s "
                    f"seg00={dt0:.2f}s seg01={dt1:.2f}s seg02={dt2:.2f}s "
                    f"动作数={len(actions)} old_logprob={lp_total:.3f}"
                )
                if timing_fp is not None:
                    timing_fp.write(
                        json.dumps(
                            {
                                "traj_id": traj_id,
                                "candidate_id": int(row.get("candidate_id", 0)),
                                "video_path": video_path,
                                "total_sec": dt_all,
                                "seg00_sec": dt0,
                                "seg01_sec": dt1,
                                "seg02_sec": dt2,
                                "num_actions": len(actions),
                                "old_logprob_total": lp_total,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    timing_fp.flush()
                ok += 1
            except Exception as e:
                print(f"[generate_candidate_trajectories_real] 跳过 traj_id={traj_id}：{e}")
                failed_cnt += 1
                if failed_fp is not None:
                    failed = {
                        "traj_id": traj_id,
                        "candidate_id": int(row.get("candidate_id", 0)),
                        "video_path": video_path,
                        "error": str(e),
                    }
                    failed_fp.write(json.dumps(failed, ensure_ascii=False) + "\n")
                    failed_fp.flush()
            elapsed = time.perf_counter() - stage_start
            if (n % progress_every_n == 0) or (n == total_shard):
                avg_wall = elapsed / max(1, n)
                eta_sec = max(0.0, avg_wall * max(0, total_shard - n))
                print(
                    f"[进度] shard={shard_id}/{num_shards} 已处理 {n}/{total_shard} "
                    f"成功写出={ok} 失败={failed_cnt} 平均耗时={avg_wall:.1f}s "
                    f"预计剩余={eta_sec/60.0:.1f} 分钟 全局总数={total_global}"
                )
            continue
    if failed_fp is not None:
        failed_fp.close()
    if timing_fp is not None:
        timing_fp.close()
    print(
        f"[generate_candidate_trajectories_real] shard={shard_id}/{num_shards} "
        f"处理完成：已处理={n}, 已写出={ok}, 根目录={os.path.abspath(args.trajectory_root)}"
    )


if __name__ == "__main__":
    main()
