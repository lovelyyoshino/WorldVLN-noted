#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Action-aware GRPO 工作流使用的 Windows 友好客户端脚本。

中文导读：
这个文件是 Windows 侧 rollout/调试客户端，协议与 `infer/client.py` 一致：
同一个 `session_id` 先发 1 帧预热图像，再按 `step` 上传真实新帧。当前
`tsformer_latent` Stage-2 服务端返回 16 个逐帧动作；旧 4 个宏动作输出
仍保留兼容处理。

协议：
- 一条轨迹对应一个 `session_id`，通常使用 route 文件夹名。
- 逐步上传帧：先上传 1 帧预热图像，之后每次上传 `step` 帧，直到达到 `num_frames`。

服务端输出：
- 动作增量使用 cm/deg，顺序为 [dx, dy, dz, droll, dyaw, dpitch]。
- action_head_mode=tsformer_latent：当前 Stage-2 服务端每个 segment 返回 `step` 个逐帧动作；
  旧 P2P checkpoint 可能返回 4 个宏动作，本客户端兼容两种输出。
- action_head_mode=actionhead_ref_vit：每个 segment 返回 `step` 个逐帧动作（step=16 -> 16 actions）。

客户端输出：每个 segment 写两个 JSON 文件，文件名包含 session_id。
步骤说明：1) actions json：
   - actions_server_order：服务端顺序的 Nx6，N 取决于 action_head_mode。
   - actions_client_order：部分下游工具使用的另一种 Nx6 顺序。
   - action_frames：每个动作对应的帧标识；dataset 模式为路径，unrealcv 模式为保存的文件名。
   - cumsum_*：本文件内动作的累计和。
2) poses json（绝对坐标）：
   - segment 0：points 包含起点和 N 个终点，共 1+N 个点。
   - 后续 segment：points 只包含 N 个终点。
   - pose 顺序为 [x, y, z, roll, yaw, pitch]，单位 cm/deg。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def _read_json(path: str):
    """读取 UTF-8 JSON；task、meta、summary 等文件都通过这个入口读取。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _time_id() -> str:
    """生成本次 rollout/客户端运行的时间戳 ID。"""
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _sorted_frame_paths(images_dir: str) -> List[str]:
    """按文件名排序读取离线 route 的真实观测帧。"""
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    names = [n for n in os.listdir(images_dir) if n.lower().endswith(exts)]
    names.sort()
    return [os.path.join(images_dir, n) for n in names]


def _take_with_pad(paths: List[str], n: int, pad_short_real: bool) -> List[str]:
    """通过重复最后一帧路径来 pad，行为与官方脚本一致。"""
    if len(paths) >= int(n):
        return paths[: int(n)]
    if not paths:
        raise ValueError("未找到真实帧")
    if not bool(pad_short_real):
        raise ValueError(f"需要至少 {n} 帧，实际只有 {len(paths)} 帧（可用 --pad_short_real 1 重复最后一帧补齐）")
    return paths + [paths[-1]] * (int(n) - int(len(paths)))


def _image_to_data_url_jpeg(path: str, quality: int = 90) -> str:
    """把磁盘图片编码成 JPEG data URL。"""
    return _image_to_data_url(path, codec="jpeg", quality=int(quality))


def _pil_to_data_url_jpeg(img: Image.Image, quality: int = 90) -> str:
    """把 PIL 图像编码成 JPEG data URL。"""
    return _pil_to_data_url(img, codec="jpeg", quality=int(quality))


def _image_to_data_url(path: str, *, codec: str, quality: int = 90) -> str:
    """读取图片并编码为服务端 JSON 请求中的 base64 data URL。"""
    img = Image.open(path).convert("RGB")
    return _pil_to_data_url(img, codec=str(codec), quality=int(quality))


def _pil_to_data_url(img: Image.Image, *, codec: str, quality: int = 90) -> str:
    """将 PIL RGB 图像按 JPEG/PNG 编码成 `data:image/...;base64,...`。"""
    img = img.convert("RGB")
    bio = BytesIO()
    c = str(codec).lower().strip()
    if c in ("jpg", "jpeg"):
        img.save(bio, format="JPEG", quality=int(quality))
        mime = "image/jpeg"
    elif c == "png":
        img.save(bio, format="PNG")
        mime = "image/png"
    else:
        raise ValueError(f"不支持的 image_codec: {codec}")
    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _ensure_dir(path: str) -> None:
    """创建目录；目录已存在时不报错。"""
    os.makedirs(path, exist_ok=True)


def _save_pil_jpeg(img: Image.Image, path: str, *, quality: int = 95) -> None:
    """保存仿真采集帧，方便回放每个动作对应的观测。"""
    parent = os.path.dirname(path)
    if parent:
        _ensure_dir(parent)
    img.convert("RGB").save(path, format="JPEG", quality=int(quality))


def _safe_np_image_to_pil_rgb(img_any: Any) -> Image.Image:
    """
    UnrealCV get_image(...) 经常返回 np.ndarray(H,W,3)，很多实现使用 BGR。
    这里尽量稳健地转换成 PIL RGB 图像。
    """
    if isinstance(img_any, Image.Image):
        return img_any.convert("RGB")
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("转换 UnrealCV 图像需要安装 numpy") from e
    if not isinstance(img_any, np.ndarray):
        raise TypeError(f"不支持的图像类型: {type(img_any)}")
    arr = img_any
    if arr.ndim == 3 and int(arr.shape[2]) == 3:
        try:
            import cv2  # type: ignore

            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        except Exception:
            arr = arr[:, :, ::-1]
        return Image.fromarray(arr.astype("uint8"), mode="RGB")
    if arr.ndim == 2:
        return Image.fromarray(arr.astype("uint8"), mode="L").convert("RGB")
    raise ValueError(f"不支持的 ndarray 图像形状: {getattr(arr, 'shape', None)}")


def _reorder_server_to_client(a6: List[float]) -> List[float]:
    """
    服务端顺序：[dx,dy,dz,droll,dyaw,dpitch]
    客户端顺序：[dx,dy,dz,droll,dpitch,dyaw]
    """
    if len(a6) != 6:
        raise ValueError(f"action 必须是 6D，实际长度为 {len(a6)}")
    dx, dy, dz, droll, dyaw, dpitch = [float(x) for x in a6]
    return [dx, dy, dz, droll, dpitch, dyaw]


def _cumsum_actions(actions: List[List[float]]) -> List[List[float]]:
    """逐维累加动作序列，写入日志用于检查累计位移/角度。"""
    out: List[List[float]] = []
    cur = [0.0] * 6
    for a in actions:
        cur = [cur[i] + float(a[i]) for i in range(6)]
        out.append(cur)
    return out


def _apply_action_to_pose(pose_xyz_rpy: List[float], action_dxdy_dz_droll_dyaw_dpitch: List[float]) -> List[float]:
    """
    pose: [x,y,z,roll,yaw,pitch]，单位 cm/deg。
    action: [dx,dy,dz,droll,dyaw,dpitch]，单位 cm/deg。
    简单世界坐标系积分：pose_next = pose + delta。
    """
    if len(pose_xyz_rpy) != 6:
        raise ValueError(f"pose 必须是 6D，实际长度为 {len(pose_xyz_rpy)}")
    if len(action_dxdy_dz_droll_dyaw_dpitch) != 6:
        raise ValueError(f"action 必须是 6D，实际长度为 {len(action_dxdy_dz_droll_dyaw_dpitch)}")
    return [float(pose_xyz_rpy[i]) + float(action_dxdy_dz_droll_dyaw_dpitch[i]) for i in range(6)]


def _apply_action_to_pose_with_frame(
    pose_xyz_rpy: List[float],
    action_dxdy_dz_droll_dyaw_dpitch: List[float],
    *,
    action_frame: str,
    body_apply_order: str = "yaw_first",
    integrate_roll_pitch: bool = True,
) -> List[float]:
    """
        pose: [x,y,z,roll,yaw,pitch]，单位 cm/deg。
        action: [dx,dy,dz,droll,dyaw,dpitch]，单位 cm/deg。

        - action_frame="world"：直接相加，即世界坐标系增量。
        - action_frame="body"：把 (dx,dy) 当作机体系（前进/右移），再按 yaw 旋到世界系；
          body_apply_order 决定先转向再平移、先平移再转向，还是用中点积分。

        机体系动作转世界系的核心公式：
        核心积分公式：`x += dx*cos(theta) - dy*sin(theta)`，
        公式/形状说明：`y += dx*sin(theta) + dy*cos(theta)`。
        `yaw_first` 先用 `yaw+dyaw` 当 theta；`trans_first` 用旧 yaw 当 theta；
        `midpoint` 用 `yaw + 0.5*dyaw` 当 theta，近似一边转向一边平移。

        注意：dz 直接沿 Z(up) 相加；当 pitch/roll 很小或被忽略时，这通常可接受。

    """
    if len(pose_xyz_rpy) != 6 or len(action_dxdy_dz_droll_dyaw_dpitch) != 6:
        raise ValueError("pose/action 都必须是 6D")
    x, y, z, roll, yaw, pitch = [float(v) for v in pose_xyz_rpy]
    dx, dy, dz, droll, dyaw, dpitch = [float(v) for v in action_dxdy_dz_droll_dyaw_dpitch]

    fr = str(action_frame).strip().lower()
    if fr == "world":
        x += dx
        y += dy
        z += dz
    elif fr == "body":
        # 使用 yaw 将机体系（前进/右移）方向旋到世界系（x/y）。
        # theta 的选择：yaw_first 用转向后的 yaw；trans_first 用转向前的 yaw；
        # midpoint 用 yaw+0.5*dyaw，表示在本小段内边转边走。
        import math

        order = str(body_apply_order).strip().lower()
        if order in ("yaw_first", "rotate_first", "turn_first"):
            yaw = float(yaw) + float(dyaw)
            theta = math.radians(yaw)
        elif order in ("trans_first", "translate_first", "move_first"):
            theta = math.radians(yaw)
            yaw = float(yaw) + float(dyaw)
        elif order in ("midpoint", "mid", "half"):
            theta = math.radians(float(yaw) + 0.5 * float(dyaw))
            yaw = float(yaw) + float(dyaw)
        else:
            raise ValueError(f"非法 body_apply_order={body_apply_order}，期望 yaw_first|trans_first|midpoint")
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # 二维旋转公式：body 中 dx 是前进、dy 是右移；乘上旋转矩阵后变成 world 的 x/y 增量。
        x += dx * cos_t - dy * sin_t
        y += dx * sin_t + dy * cos_t
        z += dz
    else:
        raise ValueError(f"非法 action_frame={action_frame}，期望 world|body")

    # 角度积分。
    if fr == "world":
        yaw += dyaw
    if bool(integrate_roll_pitch):
        roll += droll
        pitch += dpitch
    return [x, y, z, roll, yaw, pitch]


def _http_post_json(url: str, payload: Dict, *, timeout_s: int = 120) -> Dict:
    """向 WorldVLN 推理服务发送 JSON 请求并返回响应 dict。"""
    try:
        import requests  # type: ignore
    except Exception as e:
        raise RuntimeError("客户端 HTTP 调用需要安装 requests") from e

    r = requests.post(url, json=payload, timeout=int(timeout_s))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


@dataclass
class Route:
    """离线 dataset 模式下的一条 route 描述。"""

    route_dir: str
    route_id: str
    images_dir: str
    meta_path: str
    raw_logs_path: Optional[str]


@dataclass
class UnrealcvServiceSession:
    """service 模式维护的当前 UnrealCV 任务状态。"""

    session_id: str
    prompt: str
    current_pose: List[float]
    task_id: str = ""
    frame_index: int = 1
    max_frames: int = 49


_SERVICE_LOCK = threading.Lock()


def _discover_routes(dataset_root: str) -> List[Route]:
    """扫描 dataset_root，发现包含 images/ 与 meta.json 的 route。"""
    routes: List[Route] = []
    for name in sorted(os.listdir(dataset_root)):
        rd = os.path.join(dataset_root, name)
        if not os.path.isdir(rd):
            continue
        images_dir = os.path.join(rd, "images")
        meta_path = os.path.join(rd, "meta.json")
        if not os.path.isdir(images_dir) or not os.path.exists(meta_path):
            continue
        raw_logs = os.path.join(rd, "raw_logs.json")
        routes.append(
            Route(
                route_dir=rd,
                route_id=os.path.basename(rd.rstrip("/")),
                images_dir=images_dir,
                meta_path=meta_path,
                raw_logs_path=raw_logs if os.path.exists(raw_logs) else None,
            )
        )
    return routes


def _load_prompt(meta_path: str) -> str:
    """从 meta.json 中按兼容字段读取导航指令。"""
    meta = _read_json(meta_path)
    prompt = (meta.get("instruction") or meta.get("instruction_unified") or meta.get("prompt") or "").strip()
    return str(prompt)


def _load_start_pose_cm_deg(raw_logs_path: Optional[str]) -> List[float]:
    """读取 route 起始位姿；没有 raw_logs.json 时返回全零 pose。"""
    if not raw_logs_path or not os.path.exists(raw_logs_path):
        return [0.0] * 6
    arr = _read_json(raw_logs_path)
    if not isinstance(arr, list) or len(arr) == 0:
        return [0.0] * 6
    p0 = arr[0]
    if not (isinstance(p0, (list, tuple)) and len(p0) == 6):
        return [0.0] * 6
    return [float(x) for x in p0]  # [x,y,z,roll,yaw,pitch]，单位 cm/deg。


def _load_num_frames_step_from_config(config_json: str) -> Tuple[int, int]:
    """从服务端风格 config.json 中读取 num_frames 和 step。"""
    cfg = _read_json(config_json)
    if not isinstance(cfg, dict):
        raise ValueError(f"config json 格式错误（期望 dict）: {config_json}")
    inf = cfg.get("infinity", cfg)
    num_frames = int(inf.get("num_frames", 81))
    step = int(inf.get("step", 16))
    if num_frames <= 0 or step <= 0:
        raise ValueError(f"config 中 num_frames/step 非法: num_frames={num_frames} step={step}")
    return num_frames, step


def _load_instruction_and_initial_pose_from_task_json(task_json_path: str) -> Tuple[str, List[float]]:
    """
    读取 UAV-Flow-Eval task json（例如 test_jsons/*.json），提取：
    - instruction（或 instruction_unified）
    - initial_pos: [x,y,z,roll,yaw,pitch]，单位 cm/deg。
    """
    d = _read_json(task_json_path)
    if not isinstance(d, dict):
        raise ValueError(f"task json 格式错误（期望 dict）: {task_json_path}")
    instr = (d.get("instruction") or d.get("instruction_unified") or "").strip()
    if not instr:
        raise ValueError(f"task json 中 instruction 为空: {task_json_path}")
    initial_pos = d.get("initial_pos", None)
    if not (isinstance(initial_pos, list) and len(initial_pos) >= 6):
        raise ValueError(f"task json 中 initial_pos 非法: {task_json_path}")
    init6 = [float(x) for x in initial_pos[:6]]
    return instr, init6


def _build_obj_info_from_task_json(task_json_path: str) -> Optional[Dict[str, Any]]:
    """
    与 batch_run_act_all.py 对齐：
    - 只有 obj_id 和 use_obj 都存在时才放置目标物。
    - 如果有 target_pos，优先用 target_pos[:3]/target_pos[3:] 作为 obj_pos/obj_rot。
    - 否则回退到 obj_pos/obj_rot。
    """
    d = _read_json(task_json_path)
    if not isinstance(d, dict):
        return None
    if "obj_id" not in d or "use_obj" not in d:
        return None
    if "target_pos" in d and isinstance(d["target_pos"], list) and len(d["target_pos"]) == 6:
        obj_pos = [float(x) for x in d["target_pos"][:3]]
        obj_rot = [float(x) for x in d["target_pos"][3:]]
    else:
        raw_pos = d.get("obj_pos", None)
        raw_rot = d.get("obj_rot", [0, 0, 0])
        if not (isinstance(raw_pos, list) and len(raw_pos) >= 3):
            return None
        obj_pos = [float(x) for x in raw_pos[:3]]
        obj_rot = [float(x) for x in (raw_rot[:3] if isinstance(raw_rot, list) else [0, 0, 0])]
    return {
        "use_obj": int(d["use_obj"]),
        "obj_id": int(d["obj_id"]),
        "obj_pos": obj_pos,
        "obj_rot": obj_rot,
    }


def _init_marker_objects_if_needed(env: Any) -> None:
    """
    在场景中创建/初始化 marker objects（只做一次），对齐 batch_run_act_all.py 的初始化行为。
    """
    # 避免同一进程内跨任务重复初始化。
    if bool(getattr(env.unwrapped, "_xjc_marker_inited", False)):
        return
    try:
        time.sleep(1.0)
        env.unwrapped.unrealcv.new_obj("bp_character_C", "BP_Character_21", [0, 0, 0])
        env.unwrapped.unrealcv.set_appearance("BP_Character_21", 0)
        env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", [0, 0, 0])
        time.sleep(1.0)
        env.unwrapped.unrealcv.new_obj("BP_BaseCar_C", "BP_Character_22", [1000, 0, 0])
        env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
        env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", [0, 0, 0])
        env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
        time.sleep(1.0)
        env.unwrapped._xjc_marker_inited = True
    except Exception:
        # 对象可能已存在，或 class name 有差异；不要阻塞主控制流。
        env.unwrapped._xjc_marker_inited = True


def _create_obj_if_needed_unrealcv(env: Any, obj_info: Optional[Dict[str, Any]]) -> None:
    """
    放置任务目标物；逻辑与 batch_run_act_all.py 的 create_obj_if_needed 对齐。
    """
    if obj_info is None:
        return
    use_obj = obj_info.get("use_obj", None)
    obj_id = obj_info.get("obj_id", None)
    obj_pos = obj_info.get("obj_pos", None)
    obj_rot = obj_info.get("obj_rot", None)
    if obj_pos is None:
        return
    try:
        if int(use_obj) == 1:
            env.unwrapped.unrealcv.set_appearance("BP_Character_21", int(obj_id))
            env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos)
            env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", obj_rot if obj_rot is not None else [0, 0, 0])
            env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [0, 0, -1000])
            env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos)
        elif int(use_obj) == 2:
            env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
            env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [obj_pos[0], obj_pos[1], 0])
            env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", obj_rot if obj_rot is not None else [0, 0, 0])
            env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
            env.unwrapped.unrealcv.set_obj_location("BP_Character_21", [0, 0, -1000])
            env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [obj_pos[0], obj_pos[1], 0])
        if int(use_obj) in (1, 2):
            time.sleep(1.0)
    except Exception:
        # 不打断主控制流，避免因为场景 asset 差异导致失败。
        pass


def _setup_unrealcv_camera_follow(env: Any, *, cam_id: int = 0) -> None:
    """
    将相机绑定到 UAV 位置，近似第一人称视角。
    逻辑对应 batch_run_act_all.py 的 set_cam。
    """
    x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])  # 代码/形状说明：[roll, yaw, pitch]
    cam_loc = [x, y, z]
    cam_rot = [roll, pitch, yaw]  # UnrealCV set_cam 的旋转顺序。
    env.unwrapped.unrealcv.set_cam(int(cam_id), cam_loc, cam_rot)


def _apply_pose_unrealcv(
    env: Any,
    *,
    pose_xyz_rpy: List[float],
    yaw_offset_deg: float = -180.0,
) -> None:
    """
    将 [x,y,z,roll,yaw,pitch] 应用到模拟器，单位 cm/deg。
    对 drone 而言，gym_unrealcv 的 set_obj_rotation 不一定稳定生效，因此这里使用 set_rotation(yaw)。
    """
    if len(pose_xyz_rpy) < 6:
        raise ValueError(f"pose 必须是 6D，实际长度为 {len(pose_xyz_rpy)}")
    x, y, z, _roll, yaw, _pitch = [float(v) for v in pose_xyz_rpy[:6]]
    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], [x, y, z])
    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], float(yaw) + float(yaw_offset_deg))
    _setup_unrealcv_camera_follow(env, cam_id=0)


def _capture_unrealcv_lit_pil(env: Any, *, cam_id: int = 0) -> Image.Image:
    """抓取 UnrealCV lit 视图并统一转成 PIL RGB。"""
    _setup_unrealcv_camera_follow(env, cam_id=int(cam_id))
    img = env.unwrapped.unrealcv.get_image(int(cam_id), "lit")
    return _safe_np_image_to_pil_rgb(img)


def _angle_diff_deg(a: float, b: float) -> float:
    """计算两个角度之间的最短环形差值。"""
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)


def _wait_pose_settle(
    env: Any,
    *,
    target_pose_xyz_rpy: List[float],
    yaw_offset_deg: float,
    max_tries: int = 20,
    sleep_s: float = 0.05,
    pos_tol_cm: float = 1.0,
    yaw_tol_deg: float = 1.0,
) -> None:
    """等待 set_obj_location/set_rotation 生效，降低抓到陈旧首帧的概率。"""
    tx, ty, tz = [float(v) for v in target_pose_xyz_rpy[:3]]
    tyaw_set = float(target_pose_xyz_rpy[4]) + float(yaw_offset_deg)
    for _ in range(int(max_tries)):
        try:
            x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
            _roll, yaw_now, _pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])
            pos_ok = abs(float(x) - tx) <= pos_tol_cm and abs(float(y) - ty) <= pos_tol_cm and abs(float(z) - tz) <= pos_tol_cm
            yaw_ok = _angle_diff_deg(float(yaw_now), tyaw_set) <= yaw_tol_deg
            if pos_ok and yaw_ok:
                return
        except Exception:
            pass
        time.sleep(float(sleep_s))


def _make_unrealcv_env(
    *,
    env_id: str,
    time_dilation_value: int,
    seed: int,
    resolution_wh: Tuple[int, int],
    ue_port: int,
) -> Any:
    """创建并初始化 gym_unrealcv 环境，供本地 rollout 或 service 模式使用。"""
    try:
        import gym  # type: ignore
        import gym_unrealcv  # noqa: F401  # type: ignore
        from gym_unrealcv.envs.wrappers import configUE, time_dilation  # type: ignore
    except Exception as e:
        raise SystemExit(f"mode=service/unrealcv 需要导入 gym 和 gym_unrealcv，但导入失败: {e}") from e

    env = gym.make(str(env_id))
    if int(time_dilation_value) > 0:
        env = time_dilation.TimeDilationWrapper(env, int(time_dilation_value))
    try:
        env.unwrapped.agents_category = ["drone"]
    except Exception:
        pass
    env = configUE.ConfigUEWrapper(env, resolution=(int(resolution_wh[0]), int(resolution_wh[1])))
    try:
        env.seed(int(seed))
    except Exception:
        pass
    try:
        if int(ue_port) > 0:
            try:
                env.unwrapped.ue_binary.write_port(int(ue_port))
            except Exception:
                pass
        env.reset()
        env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
        env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)
    except Exception:
        pass
    return env


def _reset_unrealcv_env(env: Any) -> None:
    """重置仿真环境，并恢复 viewport/物理设置。"""
    try:
        env.reset()
    except Exception:
        pass
    try:
        env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
        env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)
    except Exception:
        pass


def _resolve_service_task_fields(payload: Dict[str, Any], task_json_root: str) -> Tuple[str, List[float], Optional[Dict[str, Any]], str, str]:
    """
    解析 service /reset 请求中的任务字段。

    请求可以直接带 prompt/initial_pose，也可以只给 task_id/task_json_path，由本地 task_json_root
    解析出指令、初始位姿和目标物体信息。
    """
    task_id = str(payload.get("task_id", "") or "").strip()
    task_json_path = str(payload.get("task_json_path", "") or "").strip()
    loaded_prompt = ""
    loaded_pose: Optional[List[float]] = None
    loaded_obj: Optional[Dict[str, Any]] = None

    if not task_json_path and task_id and task_json_root:
        candidate = os.path.join(os.path.abspath(task_json_root), f"{task_id}.json")
        if os.path.exists(candidate):
            task_json_path = candidate

    if task_json_path and os.path.exists(task_json_path):
        loaded_prompt, loaded_pose = _load_instruction_and_initial_pose_from_task_json(task_json_path)
        loaded_obj = _build_obj_info_from_task_json(task_json_path)
        if not task_id:
            task_id = os.path.splitext(os.path.basename(task_json_path))[0]

    prompt = str(payload.get("prompt") or payload.get("instruction") or loaded_prompt or "").strip()
    if not prompt:
        raise ValueError("必须提供 prompt 或 instruction")

    initial_pose = payload.get("initial_pose", None)
    if isinstance(initial_pose, list) and len(initial_pose) >= 6:
        init6 = [float(x) for x in initial_pose[:6]]
    elif loaded_pose is not None:
        init6 = [float(x) for x in loaded_pose[:6]]
    else:
        raise ValueError("无法读取 task json 时必须提供 initial_pose")

    obj_info = payload.get("obj_info", None)
    if isinstance(obj_info, dict):
        resolved_obj = obj_info
    else:
        resolved_obj = loaded_obj

    return prompt, init6, resolved_obj, task_id, os.path.basename(task_json_path) if task_json_path else ""


def _capture_env_data_url(env: Any, *, image_codec: str, jpeg_quality: int) -> str:
    """采集当前仿真画面并编码成 data URL，返回给远端 rollout 进程。"""
    img = _capture_unrealcv_lit_pil(env, cam_id=0)
    return _pil_to_data_url(img, codec=str(image_codec), quality=int(jpeg_quality))


def _json_response(handler: BaseHTTPRequestHandler, status_code: int, obj: Dict[str, Any]) -> None:
    """给内置 HTTP 服务写 JSON 响应。"""
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(int(status_code))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _service_reset(
    *,
    payload: Dict[str, Any],
    env: Any,
    task_json_root: str,
    yaw_offset_deg: float,
    image_codec: str,
    jpeg_quality: int,
) -> Tuple[UnrealcvServiceSession, Dict[str, Any]]:
    """处理 service /reset：重置任务、放置目标物、采集初始帧并返回 session。"""
    session_id = str(payload.get("session_id", "") or "").strip()
    if not session_id:
        raise ValueError("必须提供 session_id")
    prompt, init_pose, obj_info, task_id, task_json_name = _resolve_service_task_fields(payload, task_json_root)

    _reset_unrealcv_env(env)
    _init_marker_objects_if_needed(env)
    _create_obj_if_needed_unrealcv(env, obj_info)
    _apply_pose_unrealcv(env, pose_xyz_rpy=init_pose, yaw_offset_deg=float(yaw_offset_deg))
    _wait_pose_settle(env, target_pose_xyz_rpy=init_pose, yaw_offset_deg=float(yaw_offset_deg))
    time.sleep(1.0)

    session = UnrealcvServiceSession(
        session_id=session_id,
        prompt=prompt,
        current_pose=list(init_pose),
        task_id=str(task_id),
        frame_index=1,
        max_frames=int(payload.get("max_frames", 49) or 49),
    )
    return session, {
        "session_id": session_id,
        "task_id": str(task_id),
        "task_json_name": str(task_json_name),
        "instruction": prompt,
        "images_base64": [_capture_env_data_url(env, image_codec=image_codec, jpeg_quality=int(jpeg_quality))],
        "frame_indices": [1],
        "world_poses": [list(session.current_pose)],
        "done": False,
    }


def _service_step_actions(
    *,
    payload: Dict[str, Any],
    env: Any,
    session: UnrealcvServiceSession,
    action_frame: str,
    body_apply_order: str,
    yaw_offset_deg: float,
    image_codec: str,
    jpeg_quality: int,
) -> Dict[str, Any]:
    """处理 service /step_actions：执行一批 6D 动作并返回新观测帧。"""
    session_id = str(payload.get("session_id", "") or "").strip()
    if not session_id:
        raise ValueError("必须提供 session_id")
    if session_id != str(session.session_id):
        raise ValueError(f"活动 session 不匹配: 期望 {session.session_id}，实际 {session_id}")
    actions = payload.get("actions", None)
    if not isinstance(actions, list) or len(actions) <= 0:
        raise ValueError("actions 必须是非空 list")

    images_base64: List[str] = []
    frame_indices: List[int] = []
    world_poses: List[List[float]] = []
    for action in actions:
        if not (isinstance(action, list) and len(action) == 6):
            raise ValueError("每个 action 都必须是 6D list")
        next_pose = _apply_action_to_pose_with_frame(
            session.current_pose,
            [float(x) for x in action[:6]],
            action_frame=str(action_frame),
            body_apply_order=str(body_apply_order),
            integrate_roll_pitch=False,
        )
        session.current_pose[0] = float(next_pose[0])
        session.current_pose[1] = float(next_pose[1])
        session.current_pose[2] = float(next_pose[2])
        session.current_pose[4] = float(next_pose[4])
        _apply_pose_unrealcv(env, pose_xyz_rpy=session.current_pose, yaw_offset_deg=float(yaw_offset_deg))
        session.frame_index += 1
        frame_indices.append(int(session.frame_index))
        world_poses.append(list(session.current_pose))
        images_base64.append(_capture_env_data_url(env, image_codec=image_codec, jpeg_quality=int(jpeg_quality)))

    return {
        "session_id": session.session_id,
        "task_id": session.task_id,
        "images_base64": images_base64,
        "frame_indices": frame_indices,
        "world_poses": world_poses,
        "done": bool(int(session.frame_index) >= int(session.max_frames)),
    }


def _make_service_handler(
    *,
    env: Any,
    state: Dict[str, Optional[UnrealcvServiceSession]],
    task_json_root: str,
    yaw_offset_deg: float,
    action_frame: str,
    body_apply_order: str,
    image_codec: str,
    jpeg_quality: int,
):
    """构造绑定当前 UnrealCV env 的 HTTP handler 类。"""
    class _ServiceHandler(BaseHTTPRequestHandler):
        """内置 HTTP 服务：提供 /health、/reset、/step_actions 给远端 GRPO rollout 调用。"""

        server_version = "UAVFlowUnrealCVService/0.1"

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            """关闭默认 HTTP 访问日志，避免 rollout 日志被刷屏。"""
            return

        def do_GET(self) -> None:  # noqa: N802
            """处理 /health 请求，返回当前 service 与 session 状态。"""
            if self.path.rstrip("/") != "/health":
                _json_response(self, 404, {"error": f"未知路径: {self.path}"})
                return
            with _SERVICE_LOCK:
                session = state.get("session")
                _json_response(
                    self,
                    200,
                    {
                        "status": "ok",
                        "active_session": str(session.session_id) if session else "",
                        "frame_index": int(session.frame_index) if session else 0,
                    },
                )

        def do_POST(self) -> None:  # noqa: N802
            """处理 /reset 和 /step_actions 请求。"""
            try:
                content_len = int(self.headers.get("Content-Length", "0") or "0")
                payload = json.loads(self.rfile.read(content_len).decode("utf-8") or "{}")
                if not isinstance(payload, dict):
                    raise ValueError("payload 必须是 JSON 对象")
                with _SERVICE_LOCK:
                    if self.path.rstrip("/") == "/reset":
                        session, resp = _service_reset(
                            payload=payload,
                            env=env,
                            task_json_root=task_json_root,
                            yaw_offset_deg=float(yaw_offset_deg),
                            image_codec=str(image_codec),
                            jpeg_quality=int(jpeg_quality),
                        )
                        state["session"] = session
                        _json_response(self, 200, resp)
                        return
                    if self.path.rstrip("/") == "/step_actions":
                        session = state.get("session")
                        if session is None:
                            raise ValueError("没有活动 session；请先调用 /reset")
                        resp = _service_step_actions(
                            payload=payload,
                            env=env,
                            session=session,
                            action_frame=str(action_frame),
                            body_apply_order=str(body_apply_order),
                            yaw_offset_deg=float(yaw_offset_deg),
                            image_codec=str(image_codec),
                            jpeg_quality=int(jpeg_quality),
                        )
                        _json_response(self, 200, resp)
                        return
                _json_response(self, 404, {"error": f"未知路径: {self.path}"})
            except Exception as e:
                _json_response(self, 500, {"error": str(e)})

    return _ServiceHandler


def run_env_service(
    *,
    host: str,
    port: int,
    env: Any,
    task_json_root: str,
    yaw_offset_deg: float,
    action_frame: str,
    body_apply_order: str,
    image_codec: str,
    jpeg_quality: int,
) -> None:
    """启动模拟器侧 HTTP 服务，让远端 GRPO 进程通过网络控制本地 UnrealCV。"""
    state: Dict[str, Optional[UnrealcvServiceSession]] = {"session": None}
    handler = _make_service_handler(
        env=env,
        state=state,
        task_json_root=task_json_root,
        yaw_offset_deg=float(yaw_offset_deg),
        action_frame=str(action_frame),
        body_apply_order=str(body_apply_order),
        image_codec=str(image_codec),
        jpeg_quality=int(jpeg_quality),
    )
    server = ThreadingHTTPServer((str(host), int(port)), handler)
    print(f"[env_service] 正在监听 http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            env.close()
        except Exception:
            pass


def _split_action_to_substeps(action6: List[float], substeps: int) -> List[List[float]]:
    """把旧 4 macro action 均分成多个子步，以兼容 16 帧执行节奏。"""
    if len(action6) != 6:
        raise ValueError(f"action 必须是 6D，实际长度为 {len(action6)}")
    k = int(substeps)
    if k <= 0:
        raise ValueError(f"substeps 必须 >0，实际为 {substeps}")
    a = [float(x) for x in action6]
    return [[a[i] / float(k) for i in range(6)] for _ in range(k)]


def run_one_task_unrealcv(
    *,
    task_json_path: str,
    env: Any,
    env_id: str,
    server_base_url: str,
    out_root: str,
    run_id: str,
    num_frames: int,
    step: int,
    max_actions: int = 0,
    timeout_s: int,
    action_head_mode: str,
    action_head_batch_size: int,
    action_head_stride: int,
    action_head_pre_resize_hw: int,
    image_codec: str,
    jpeg_quality: int,
    yaw_offset_deg: float = -180.0,
    allow_future_last_segment: bool = False,
    action_frame: str = "world",
    body_apply_order: str = "yaw_first",
    save_images: bool = True,
) -> None:
    """
    在线模式（gym_unrealcv）：
    - 从 task_json 读取 instruction 和 initial_pos。
    - 抓取 256x256 lit RGB；分辨率由 ConfigUEWrapper 设置。
    - 按增量上传帧：1, step, step, ...（prefix_mode=false；历史帧由服务端按 session_id 保存）。
    - tsformer_latent：当前 Stage-2 服务端返回 step(=16) 个逐帧动作；旧 4-action 宏动作
      输出仍被接受，并会把每个宏动作拆成 4 个子动作。
    - actionhead_ref_vit：直接接收 step(=16) 个逐帧动作，逐个执行并保存帧。

    注意：n==1 时服务端可能只做预热，不发射动作。客户端会原地采集/上传 `step`
    帧来推进服务端时间线，使 seg0 可以生成。
    """
    instruction, init_pose = _load_instruction_and_initial_pose_from_task_json(task_json_path)
    obj_info = _build_obj_info_from_task_json(task_json_path)

    base_name = os.path.splitext(os.path.basename(task_json_path))[0]
    session_id = f"{base_name}__{run_id}"
    out_dir = os.path.join(os.path.abspath(out_root), f"client_run_{run_id}", base_name)
    _ensure_dir(out_dir)
    images_dir = os.path.join(out_dir, "images")
    _ensure_dir(images_dir)

    # 每个任务开始前重置 env 状态。
    try:
        env.reset()
    except Exception:
        pass
    try:
        env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
        env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)
    except Exception:
        pass

    _init_marker_objects_if_needed(env)
    _create_obj_if_needed_unrealcv(env, obj_info)
    _apply_pose_unrealcv(env, pose_xyz_rpy=init_pose, yaw_offset_deg=float(yaw_offset_deg))
    _wait_pose_settle(env, target_pose_xyz_rpy=init_pose, yaw_offset_deg=float(yaw_offset_deg))
    # 给相机额外刷新时间，降低首帧抓到陈旧视图的风险。
    time.sleep(1.0)

    summary = {
        "mode": "unrealcv",
        "task_json": os.path.abspath(task_json_path),
        "env_id": str(env_id),
        "session_id": session_id,
        "server_base_url": server_base_url,
        "endpoint": "/v1/predict_delta_actions",
        "instruction": instruction,
        "initial_pose_cm_deg_order_xyz_roll_yaw_pitch": init_pose,
        "num_frames": int(num_frames),
        "step": int(step),
        "max_actions": int(max_actions),
        "prefix_mode": False,
        "allow_future_last_segment": bool(allow_future_last_segment),
        "camera": {"cam_id": 0, "viewmode": "lit"},
        "image_source": {"type": "gym_unrealcv", "client_capture_resolution": list(getattr(env.unwrapped, "resolution", [None, None]))},
        "yaw_offset_deg_applied_in_unrealcv": float(yaw_offset_deg),
        "action_frame_for_integration": str(action_frame),
        "body_apply_order": str(body_apply_order),
        "action_head_mode": str(action_head_mode),
        "action_head_batch_size": int(action_head_batch_size),
        "action_head_stride": int(action_head_stride),
        "action_head_pre_resize_hw": int(action_head_pre_resize_hw),
        "image_codec": str(image_codec),
        "jpeg_quality": int(jpeg_quality),
        "units": {"translation": "cm", "angles": "deg"},
        "time": int(time.time()),
    }
    with open(os.path.join(out_dir, f"{base_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    def _post_frames(frames: List[Image.Image], *, include_instruction: bool) -> Dict:
        """向 WorldVLN 服务端发送本轮新增真实帧；首轮附带 instruction 并重置 session。"""
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "images_base64": [_pil_to_data_url(im, codec=str(image_codec), quality=int(jpeg_quality)) for im in frames],
            "prefix_mode": False,
            "allow_future_last_segment": bool(allow_future_last_segment),
            "allow_future_segments": True,
            "action_head_mode": str(action_head_mode),
            "action_head_batch_size": int(action_head_batch_size),
            "action_head_stride": int(action_head_stride),
            "action_head_pre_resize_hw": int(action_head_pre_resize_hw),
        }
        if include_instruction:
            payload["instruction"] = instruction
            payload["reset_session"] = True
        return _http_post_json(server_base_url.rstrip("/") + "/v1/predict_delta_actions", payload, timeout_s=int(timeout_s))

    # 第 1 帧：初始观测。
    frame_idx = 1
    actions_executed = 0
    cur_pose = init_pose[:]  # 代码/形状说明：[x,y,z,roll,yaw,pitch]
    im0 = _capture_unrealcv_lit_pil(env, cam_id=0)
    if bool(save_images):
        _save_pil_jpeg(im0, os.path.join(images_dir, f"frame_{frame_idx:04d}.jpg"), quality=95)
    resp = _post_frames([im0], include_instruction=True)

    def _write_segment_logs(
        seg: int,
        call_i: int,
        frames_in_call: int,
        resp_obj: Dict,
        actions_server: List[List[float]],
        pose_before: List[float],
        action_image_names: Optional[List[str]] = None,
    ) -> None:
        """把每个 segment 的动作、累计动作和积分位姿写成 GRPO 可回放日志。"""
        actions_client = [_reorder_server_to_client(a) for a in actions_server]
        cumsum_server = _cumsum_actions(actions_server)
        cumsum_client = _cumsum_actions(actions_client)
        actions_json = {
            "session_id": session_id,
            "task_json": os.path.abspath(task_json_path),
            "segment_index": int(seg),
            "call_index": int(call_i),
            "frames_in_call": int(frames_in_call),
            "num_received_frames": resp_obj.get("num_received_frames", None),
            "done": resp_obj.get("done", None),
            "prefix_latents": resp_obj.get("prefix_latents", None),
            "units": {"translation": "cm", "angles": "deg"},
            "action_head_mode": str(action_head_mode),
            "num_actions": int(len(actions_server)),
            "action_order_server": ["dx", "dy", "dz", "droll", "dyaw", "dpitch"],
            "action_order_client": ["dx", "dy", "dz", "droll", "dpitch", "dyaw"],
            "actions_server_order": actions_server,
            "actions_client_order": actions_client,
            "action_frames": action_image_names or [],
            "cumsum_server_order": cumsum_server,
            "cumsum_client_order": cumsum_client,
        }
        with open(os.path.join(out_dir, f"{base_name}_seg{seg:02d}_actions.json"), "w", encoding="utf-8") as f:
            json.dump(actions_json, f, ensure_ascii=False, indent=2)

        # 每个动作后的 pose 点；动作可能是 macro，也可能是逐帧动作。
        p = pose_before[:]
        pts: List[List[float]] = []
        if int(seg) == 0:
            pts.append(p[:])
        for a in actions_server:
            p = _apply_action_to_pose_with_frame(
                p,
                a,
                action_frame=str(action_frame),
                body_apply_order=str(body_apply_order),
                integrate_roll_pitch=True,
            )
            pts.append(p[:])
        poses_json = {
            "session_id": session_id,
            "task_json": os.path.abspath(task_json_path),
            "segment_index": int(seg),
            "call_index": int(call_i),
            "units": {"translation": "cm", "angles": "deg"},
            "pose_order": ["x", "y", "z", "roll", "yaw", "pitch"],
            "points": pts,
        }
        with open(os.path.join(out_dir, f"{base_name}_seg{seg:02d}_poses.json"), "w", encoding="utf-8") as f:
            json.dump(poses_json, f, ensure_ascii=False, indent=2)

    # 确定执行时预期的动作数量。
    # 当前 `tsformer_latent` Stage-2 返回逐帧动作；旧 P2P 风格路径返回 4 个宏动作。
    mode_l = str(action_head_mode).strip().lower()
    is_per_frame_mode = mode_l in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead")
    expected_actions = int(step) if bool(is_per_frame_mode) else 4

    call_i = 0
    last_upload_n = 1  # 首次上传：1 帧。
    max_actions_i = int(max_actions)
    # 停止条件：
    # - max_actions>0：执行到指定动作数后停止（严格协议：48 个动作 => 1+48 帧）。
    # - 否则：回退到 num_frames 边界。
    while True:
        if max_actions_i > 0 and int(actions_executed) >= int(max_actions_i):
            break
        if max_actions_i <= 0 and frame_idx >= int(num_frames):
            break
        call_i += 1
        actions = resp.get("actions", [])
        seg = int(resp.get("segment_index", -1))
        done = bool(resp.get("done", False))

        pending_actions: Optional[List[List[float]]] = None
        if isinstance(actions, list) and len(actions) == int(expected_actions) and seg >= 0:
            pending_actions = [[float(x) for x in a] for a in actions]
            if max_actions_i > 0:
                remain = int(max_actions_i) - int(actions_executed)
                if remain <= 0:
                    pending_actions = []
                elif 0 < remain < len(pending_actions):
                    pending_actions = pending_actions[:remain]
            _write_segment_logs(seg, call_i=call_i, frames_in_call=int(last_upload_n), resp_obj=resp, actions_server=pending_actions, pose_before=cur_pose[:])

        if done:
            break

        # 收集下一批 `step` 帧，用于增量上传。
        new_frames: List[Image.Image] = []
        if pending_actions is None:
            # 保持位置不动并补帧，直到服务端准备好发射某个 segment。
            for _ in range(int(step)):
                if max_actions_i > 0:
                    # 严格动作计数模式下，没有执行动作就不要推进帧。
                    break
                if frame_idx >= int(num_frames):
                    break
                im = _capture_unrealcv_lit_pil(env, cam_id=0)
                frame_idx += 1
                if bool(save_images):
                    _save_pil_jpeg(im, os.path.join(images_dir, f"frame_{frame_idx:04d}.jpg"), quality=95)
                new_frames.append(im)
        else:
            total_needed = int(step)
            produced = 0
            action_image_names: List[str] = []
            if bool(is_per_frame_mode):
                # 逐帧动作直接执行；通常 16 个动作 -> 16 帧。
                for a in pending_actions:
                    if produced >= total_needed:
                        break
                    if max_actions_i > 0 and int(actions_executed) >= int(max_actions_i):
                        break
                    if max_actions_i <= 0 and frame_idx >= int(num_frames):
                        break
                    next_pose = _apply_action_to_pose_with_frame(
                        cur_pose,
                        a,
                        action_frame=str(action_frame),
                        body_apply_order=str(body_apply_order),
                        integrate_roll_pitch=False,
                    )
                    cur_pose[0] = float(next_pose[0])
                    cur_pose[1] = float(next_pose[1])
                    cur_pose[2] = float(next_pose[2])
                    cur_pose[4] = float(next_pose[4])
                    _apply_pose_unrealcv(env, pose_xyz_rpy=cur_pose, yaw_offset_deg=float(yaw_offset_deg))
                    im = _capture_unrealcv_lit_pil(env, cam_id=0)
                    frame_idx += 1
                    produced += 1
                    actions_executed += 1
                    name = f"frame_{frame_idx:04d}.jpg"
                    if bool(save_images):
                        _save_pil_jpeg(im, os.path.join(images_dir, name), quality=95)
                    action_image_names.append(name)
                    new_frames.append(im)
            else:
                # 执行旧版 4 个宏动作；每个拆成 4 个子步，总共 16 帧。
                substeps_per_action = 4
                for a in pending_actions:
                    for sub in _split_action_to_substeps(a, substeps=substeps_per_action):
                        if produced >= total_needed:
                            break
                        if max_actions_i > 0 and int(actions_executed) >= int(max_actions_i):
                            break
                        if max_actions_i <= 0 and frame_idx >= int(num_frames):
                            break
                        next_pose = _apply_action_to_pose_with_frame(
                            cur_pose,
                            sub,
                            action_frame=str(action_frame),
                            body_apply_order=str(body_apply_order),
                            integrate_roll_pitch=False,
                        )
                        cur_pose[0] = float(next_pose[0])
                        cur_pose[1] = float(next_pose[1])
                        cur_pose[2] = float(next_pose[2])
                        cur_pose[4] = float(next_pose[4])
                        _apply_pose_unrealcv(env, pose_xyz_rpy=cur_pose, yaw_offset_deg=float(yaw_offset_deg))
                        im = _capture_unrealcv_lit_pil(env, cam_id=0)
                        frame_idx += 1
                        produced += 1
                        actions_executed += 1
                        name = f"frame_{frame_idx:04d}.jpg"
                        if bool(save_images):
                            _save_pil_jpeg(im, os.path.join(images_dir, name), quality=95)
                        action_image_names.append(name)
                        new_frames.append(im)
                    if produced >= total_needed:
                        break
                    if max_actions_i > 0 and int(actions_executed) >= int(max_actions_i):
                        break
                    if max_actions_i <= 0 and frame_idx >= int(num_frames):
                        break
            # 如果生成帧不足：只在 num_frames 模式下用原地采样补齐；max_actions 模式下不补。
            # max_actions 模式严格保持“每个已执行动作对应一帧”。
            if max_actions_i <= 0:
                while produced < total_needed and frame_idx < int(num_frames):
                    im = _capture_unrealcv_lit_pil(env, cam_id=0)
                    frame_idx += 1
                    produced += 1
                    if bool(save_images):
                        _save_pil_jpeg(im, os.path.join(images_dir, f"frame_{frame_idx:04d}.jpg"), quality=95)
                    new_frames.append(im)

            # 尽力把最新 segment action log 补上每个动作对应的帧名：
            # 仅当文件存在且长度匹配时才重写。
            try:
                if seg >= 0 and action_image_names and len(action_image_names) == len(pending_actions):
                    p_actions = os.path.join(out_dir, f"{base_name}_seg{seg:02d}_actions.json")
                    if os.path.exists(p_actions):
                        obj = _read_json(p_actions)
                        obj["action_frames"] = action_image_names
                        with open(p_actions, "w", encoding="utf-8") as f:
                            json.dump(obj, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        if not new_frames:
            break
        last_upload_n = int(len(new_frames))
        resp = _post_frames(new_frames, include_instruction=False)

    # 尽力更新 summary 中的最终计数器。
    try:
        p_sum = os.path.join(out_dir, f"{base_name}_summary.json")
        if os.path.exists(p_sum):
            obj = _read_json(p_sum)
            obj["final"] = {
                "frames_captured": int(frame_idx),
                "actions_executed": int(actions_executed),
            }
            with open(p_sum, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _chunks_stream(paths: List[str], *, num_frames: int, step: int) -> List[List[str]]:
    """增量流式上传切分：首轮 1 帧，之后每轮 step 帧。"""
    p = paths[: int(num_frames)]
    chunks: List[List[str]] = []
    chunks.append(p[:1])
    idx = 1
    while idx < int(num_frames):
        chunks.append(p[idx : min(int(num_frames), idx + int(step))])
        idx += int(step)
    return chunks


def _obs_points(pred_num_frames: int, step: int) -> List[int]:
    """
    与服务端相同的 points 逻辑。

    初学者公式：`points = [1, 1+step, 1+2*step, ..., num_frames]`。
    服务端默认用 `ready_default = n >= points[seg+1]` 判断某个 segment 是否可输出动作：
    已收到帧数 `n` 到达该段右边界时，该段才算 ready。
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


def _chunks_prefix(paths: List[str], *, num_frames: int, step: int) -> List[List[str]]:
    """
        Prefix mode：每次调用都发送完整 prefix，用来匹配 v2v 语义：
          第 0 次调用 `call0`: [1]
          公式/形状说明：call1: [1..17]
          公式/形状说明：call2: [1..33]
          公式/形状说明：call3: [1..49]

    """
    p = paths[: int(num_frames)]
    pts = _obs_points(int(num_frames), int(step))
    if not pts or pts[0] != 1 or pts[-1] != int(num_frames):
        raise ValueError(f"points 计算结果非法: {pts}，num_frames={num_frames} step={step}")
    return [p[: int(k)] for k in pts]


def _chunks_prefix_from_points(paths: List[str], *, points: List[int]) -> List[List[str]]:
    """按显式绝对 points 构造 prefix chunks，例如 [1,17,33]。"""
    if not points or int(points[0]) != 1:
        raise ValueError(f"points 非法: {points}")
    max_k = int(max(points))
    if len(paths) < max_k:
        raise ValueError(f"points={points} 需要至少 {max_k} 帧，实际只有 {len(paths)} 帧")
    p = paths[:max_k]
    return [p[: int(k)] for k in points]


def run_one_route(
    *,
    route: Route,
    server_base_url: str,
    out_root: str,
    save_dir_name: Optional[str],
    session_id: str,
    num_frames: int,
    step: int,
    prefix_mode: bool = False,
    allow_future_last_segment: bool = False,
    action_frame: str = "world",
    body_apply_order: str = "yaw_first",
    timeout_s: int = 120,
    image_codec: str = "jpeg",
    jpeg_quality: int = 90,
    pad_short_real: bool = False,
    action_head_mode: str = "actionhead_ref_vit",
    action_head_batch_size: int = 8,
    action_head_stride: int = 1,
    action_head_pre_resize_hw: int = 256,
) -> None:
    """离线 dataset 模式：回放已有 route 图像并保存每段动作/位姿 JSON。"""
    prompt = _load_prompt(route.meta_path)
    if not prompt:
        raise RuntimeError(f"prompt 为空: {route.meta_path}")
    start_pose = _load_start_pose_cm_deg(route.raw_logs_path)

    real_paths = _sorted_frame_paths(route.images_dir)
    real_count = int(len(real_paths))

    obs_points = _obs_points(int(num_frames), int(step))  # e.g. [1,17,33,49]
    if len(obs_points) < 2:
        raise RuntimeError(f"points 计算结果非法: {obs_points}，num_frames={num_frames} step={step}")

    # 如果允许在没有 points[-1] 真实帧时发射最后一个 segment，
    # 则 dataset 中只需要存在到 points[-2]（例如 33）的 prefix。
    send_points = obs_points
    if bool(prefix_mode) and bool(allow_future_last_segment) and int(obs_points[-1]) > int(obs_points[-2]):
        send_points = obs_points[:-1]  # 丢掉最终的 49。

    max_need = int(max(send_points))
    frame_paths = _take_with_pad(real_paths, max_need, bool(pad_short_real))

    paths_for_map: List[str] = []
    if prefix_mode:
        chunks = _chunks_prefix_from_points(frame_paths, points=[int(k) for k in send_points])
        paths_for_map = frame_paths
        # 不增加更多真实帧，触发最后一个 segment（seg02）：
        # 再发送一次最大 prefix；服务端看到没有新增帧，但会发射最后一个 seg。
        if bool(allow_future_last_segment) and int(obs_points[-1]) > int(obs_points[-2]):
            chunks.append(chunks[-1])
    else:
        # stream 模式使用增量 chunks，需要完整 num_frames 帧。
        full_paths = _take_with_pad(real_paths, int(num_frames), bool(pad_short_real))
        chunks = _chunks_stream(full_paths, num_frames=int(num_frames), step=int(step))
        paths_for_map = full_paths

    out_dir = os.path.join(out_root, str(save_dir_name or session_id))
    os.makedirs(out_dir, exist_ok=True)

    # 只保存一次输入 summary。
    summary = {
        "session_id": session_id,
        "route_id": route.route_id,
        "route_dir": route.route_dir,
        "server_base_url": server_base_url,
        "endpoint": "/v1/predict_delta_actions",
        "prompt": prompt,
        "start_pose_cm_deg_order_xyz_roll_yaw_pitch": start_pose,
        "num_frames": int(num_frames),
        "step": int(step),
        "real_frame_count": real_count,
        "pad_short_real": bool(pad_short_real),
        "allow_future_last_segment": bool(allow_future_last_segment),
        "action_frame_for_integration": str(action_frame),
        "body_apply_order": str(body_apply_order),
        "frames_sent": {"total": int(num_frames), "chunks": [len(c) for c in chunks], "prefix_mode": bool(prefix_mode)},
        "action_head_mode": str(action_head_mode),
        "action_head_batch_size": int(action_head_batch_size),
        "action_head_stride": int(action_head_stride),
        "action_head_pre_resize_hw": int(action_head_pre_resize_hw),
        "image_codec": str(image_codec),
        "jpeg_quality": int(jpeg_quality),
        "action_order_server": ["dx", "dy", "dz", "droll", "dyaw", "dpitch"],
        "pose_order": ["x", "y", "z", "roll", "yaw", "pitch"],
        "units": {"translation": "cm", "angles": "deg"},
        "time": int(time.time()),
    }
    with open(os.path.join(out_dir, f"{session_id}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cur_pose = start_pose[:]  # 绝对 pose，单位 cm/deg。

    for call_i, chunk_paths in enumerate(chunks):
        images_b64 = [_image_to_data_url(p, codec=str(image_codec), quality=int(jpeg_quality)) for p in chunk_paths]
        payload = {
            "session_id": session_id,
            "images_base64": images_b64,
            "prefix_mode": bool(prefix_mode),
            "allow_future_last_segment": bool(allow_future_last_segment),
            "action_head_mode": str(action_head_mode),
            "action_head_batch_size": int(action_head_batch_size),
            "action_head_stride": int(action_head_stride),
            "action_head_pre_resize_hw": int(action_head_pre_resize_hw),
        }
        if call_i == 0:
            payload["instruction"] = prompt

        resp = _http_post_json(server_base_url.rstrip("/") + "/v1/predict_delta_actions", payload, timeout_s=int(timeout_s))

        actions = resp.get("actions", [])
        seg = int(resp.get("segment_index", -1))
        # 预热调用（首帧）可能不返回动作，此时 segment_index=-1。
        if seg < 0:
            continue
        # 按模式校验动作数量。
        # 当前 Stage-2 `tsformer_latent` 每个帧间转移返回一个动作；
        # 为兼容旧 checkpoint/tools，仍接受旧版 4 动作宏输出。
        mode_l = str(action_head_mode).strip().lower()
        if mode_l in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead"):
            if seg < 0 or seg >= len(obs_points) - 1:
                raise RuntimeError(f"服务端返回的 segment_index 非法: seg={seg} obs_points={obs_points}")
            expected_n = int(obs_points[seg + 1]) - int(obs_points[seg])
        else:
            expected_n = 4
        if not isinstance(actions, list) or len(actions) != int(expected_n):
            raise RuntimeError(
                f"服务端在 call={call_i}, segment={seg} 返回非法 actions: "
                f"{type(actions)} len={getattr(actions, '__len__', lambda: None)()}，期望 {expected_n}"
            )

        actions_server = [[float(x) for x in a] for a in actions]
        actions_client = [_reorder_server_to_client(a) for a in actions_server]
        cumsum_server = _cumsum_actions(actions_server)
        cumsum_client = _cumsum_actions(actions_client)

        # 每个动作对应的帧映射（dataset 模式）：
        # 仅当服务端每个帧间转移返回一个动作时才有意义。
        action_frames: List[Dict[str, Any]] = []
        if mode_l in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead") and seg >= 0 and seg < len(obs_points) - 1:
            obs_len = int(obs_points[seg])
            for i in range(int(len(actions_server))):
                abs_from = int(obs_len) + int(i)
                abs_to = int(obs_len) + int(i) + 1
                img_path_upload: Optional[str] = None
                img_path_real: Optional[str] = None

                # 上传侧映射帧；--pad_short_real=1 时包含 padding/重复帧。
                if 1 <= abs_to <= len(paths_for_map):
                    img_path_upload = paths_for_map[int(abs_to) - 1]

                # 真实 dataset 帧；即使 allow_future_last_segment 模式未上传，也可能存在。
                if 1 <= abs_to <= int(real_count):
                    img_path_real = real_paths[int(abs_to) - 1]

                is_real = 1 <= abs_to <= int(real_count)
                is_padded = (not is_real) and (img_path_upload is not None)

                # 兼容旧格式的单路径：优先真实帧，否则使用可用的上传帧（可能是 padded）。
                img_path = img_path_real or img_path_upload
                action_frames.append(
                    {
                        "index": int(i),
                        "abs_from": int(abs_from),
                        "abs_to": int(abs_to),
                        "image_path": img_path,
                        "image_path_real": img_path_real,
                        "image_path_upload": img_path_upload,
                        "image_available": bool(img_path) and os.path.exists(str(img_path)),
                        "image_available_real": bool(img_path_real) and os.path.exists(str(img_path_real)),
                        "image_available_upload": bool(img_path_upload) and os.path.exists(str(img_path_upload)),
                        "is_real_frame": bool(is_real),
                        "is_padded_frame": bool(is_padded),
                    }
                )

        # 写 actions json；每个 segment 一个文件。
        actions_json = {
            "session_id": session_id,
            "route_id": route.route_id,
            "segment_index": seg,
            "call_index": call_i,
            "frames_in_call": len(chunk_paths),
            "num_received_frames": resp.get("num_received_frames", None),
            "done": resp.get("done", None),
            "prefix_latents": resp.get("prefix_latents", None),
            "units": {"translation": "cm", "angles": "deg"},
            "action_head_mode": str(action_head_mode),
            "num_actions": int(len(actions_server)),
            "action_order_server": ["dx", "dy", "dz", "droll", "dyaw", "dpitch"],
            "action_order_client": ["dx", "dy", "dz", "droll", "dpitch", "dyaw"],
            "actions_server_order": actions_server,
            "actions_client_order": actions_client,
            "action_frames": action_frames,
            "cumsum_server_order": cumsum_server,
            "cumsum_client_order": cumsum_client,
        }
        with open(os.path.join(out_dir, f"{session_id}_seg{seg:02d}_actions.json"), "w", encoding="utf-8") as f:
            json.dump(actions_json, f, ensure_ascii=False, indent=2)

        # 绝对 pose 点；按配置的 frame/order 积分。
        pose_points: List[List[float]] = []
        if seg == 0:
            pose_points.append(cur_pose[:])

        for a in actions_server:
            cur_pose = _apply_action_to_pose_with_frame(
                cur_pose,
                a,
                action_frame=str(action_frame),
                body_apply_order=str(body_apply_order),
                integrate_roll_pitch=True,
            )
            pose_points.append(cur_pose[:])

        poses_json = {
            "session_id": session_id,
            "route_id": route.route_id,
            "segment_index": seg,
            "call_index": call_i,
            "units": {"translation": "cm", "angles": "deg"},
            "pose_order": ["x", "y", "z", "roll", "yaw", "pitch"],
            "points": pose_points,  # seg0 为 1+N 个点（起点 + N），其他 segment 为 N 个点。
        }
        with open(os.path.join(out_dir, f"{session_id}_seg{seg:02d}_poses.json"), "w", encoding="utf-8") as f:
            json.dump(poses_json, f, ensure_ascii=False, indent=2)


def main():
    """命令行入口：支持 dataset、unrealcv 和模拟器侧 service 三种模式。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="dataset", choices=["dataset", "unrealcv", "service"], help="dataset：从 --dataset_root 读取离线帧；unrealcv：使用 task json 现场采集 gym_unrealcv 帧；service：暴露 /reset 和 /step_actions，供本地 StageA remote_sim 调用。")
    ap.add_argument("--dataset_root", type=str, default="", help="(mode=dataset) 数据集根目录，内部应包含 route 子目录、images/、meta.json，以及可选 raw_logs.json。")
    ap.add_argument("--task_json", type=str, default="", help="(mode=unrealcv) 单个 task json 路径，例如 ./test_jsons/2025-03-30_11-49-14.json。")
    ap.add_argument("--json_folder", type=str, default="", help="(mode=unrealcv) 包含多个 task json 的目录。")
    ap.add_argument("--json_order", type=str, default="asc", choices=["asc", "desc"], help="(mode=unrealcv) 遍历 --json_folder 时按文件名升序或降序排序。")
    ap.add_argument("--json_start", type=str, default="", help="(mode=unrealcv) 可选起始文件名下界，例如 2025-03-30_12-02-10 或带 .json 后缀。")
    ap.add_argument("--json_start_exclusive", type=int, default=0, choices=[0, 1], help="(mode=unrealcv) 设为 1 时从 --json_start 之后的文件开始，常用于断点续跑。")
    ap.add_argument("--json_end", type=str, default="", help="(mode=unrealcv) 可选结束文件名上界（包含该文件），格式同 --json_start。")
    ap.add_argument("--env_id", type=str, default="UnrealTrack-DowntownWest-ContinuousColor-v0", help="(mode=unrealcv) gym 环境 id。")
    ap.add_argument("--time_dilation", type=int, default=10, help="(mode=unrealcv) time_dilation 包装器参数，用来加速/减速仿真。")
    ap.add_argument("--seed", type=int, default=0, help="(mode=unrealcv) 随机种子。")
    ap.add_argument("--resolution", type=str, default="256,256", help="(mode=unrealcv) 采集分辨率，格式为 'W,H'，默认 256,256。")
    ap.add_argument("--ue_port", type=int, default=0, help="(mode=unrealcv) >0 时在启动 UE 前写入 unrealcv.ini 的 socket 基础端口，便于并行运行多个 UE 实例（如 9393/9394）。")
    ap.add_argument("--host", type=str, default="0.0.0.0", help="(mode=service) HTTP 服务监听主机。")
    ap.add_argument("--port", type=int, default=8765, help="(mode=service) HTTP 服务监听端口。")
    ap.add_argument("--task_json_root", type=str, default="", help="(mode=service) 模拟器机器上的 UAV-Flow-Eval test_jsons 目录，用于把 task_id 解析成 task json。")
    ap.add_argument("--yaw_offset_deg", type=float, default=-180.0, help="(mode=unrealcv) 调用 UnrealCV set_rotation 时附加的 yaw 偏移；batch_run_act_all.py 使用 -180。")
    ap.add_argument("--action_frame", type=str, default="body", choices=["world", "body"], help="积分 pose/log 时如何解释 dx/dy：world=直接相加；body=把 dx,dy 视为机体系前进/右移，并按 yaw 旋转到世界系。")
    ap.add_argument("--body_apply_order", type=str, default="yaw_first", choices=["yaw_first", "trans_first", "midpoint"], help="仅 --action_frame=body 生效。yaw_first: 先 yaw+=dyaw 再平移；trans_first: 用旧 yaw 平移再 yaw+=dyaw；midpoint: 用 yaw+0.5*dyaw 平移再更新 yaw。")
    ap.add_argument("--max_actions", type=int, default=48, help="(mode=unrealcv) >0 时执行到指定动作数后停止（每个动作对应 1 帧）。默认 48，即 1+48=49 帧；设为 0 时改用 --num_frames 边界。")
    ap.add_argument("--server_url", type=str, default="http://127.0.0.1:8002", help="推理服务基础 URL，不要包含具体端点。")
    ap.add_argument("--out_dir", type=str, default=r"E:\xjc\UAV-Flow-main1\UAV-Flow-main\UAV-Flow-Eval\cache", help="每个 session 的 JSON 输出目录。")
    ap.add_argument("--route_id", type=str, default="", help="设置后只运行该 route_id（通常是 route 文件夹名）。")
    ap.add_argument("--max_routes", type=int, default=0, help=">0 时最多处理这么多条 route。")
    ap.add_argument("--select_n", type=int, default=0, help=">0 时在过滤后确定性选择前 N 条 route。")
    ap.add_argument("--min_real_frames", type=int, default=0, help=">0 时只保留真实图片数量不小于该阈值的 route。")
    ap.add_argument("--pad_short_real", type=int, default=0, choices=[0, 1], help="设为 1 时，真实帧不足的 route 会重复最后一帧补齐到所需帧数。")
    ap.add_argument("--run_id", type=str, default="", help="可选 run id；为空时使用当前时间戳。")
    ap.add_argument("--run_subdir", type=int, default=1, choices=[0, 1], help="设为 1 时输出到 out_dir/client_run_<run_id>/，避免覆盖历史结果。")
    ap.add_argument("--session_id_mode", type=str, default="route_run", choices=["route", "route_run"], help="服务端 session_id 命名方式，用于避免覆盖服务端 latent 目录。")
    ap.add_argument("--allow_future_last_segment", type=int, default=1, choices=[0, 1], help="设为 1 时允许只用 33 帧真实 prefix 触发 seg02（34-49 由预测段补足）；需要服务端支持。")
    ap.add_argument("--dry_run", type=int, default=0, choices=[0, 1], help="设为 1 时只打印将要处理的 route/task，不实际请求服务端。")
    ap.add_argument("--image_codec", type=str, default="jpeg", choices=["jpeg", "jpg", "png"], help="上传给服务端的图像编码格式。")
    ap.add_argument("--jpeg_quality", type=int, default=90, help="仅 --image_codec=jpeg/jpg 时使用的 JPEG 质量。")
    ap.add_argument("--prefix_mode", type=int, default=0, choices=[0, 1], help="设为 1 时每次发送完整 prefix：1,1-17,1-33,...；需要服务端支持。")
    ap.add_argument("--timeout_s", type=int, default=600, help="HTTP 请求超时秒数；seg0 推理可能需要几分钟。")
    ap.add_argument("--action_head_mode", type=str, default="actionhead_ref_vit", choices=["tsformer_latent", "actionhead_ref_vit"], help="请求服务端使用的 action head 模式。")
    ap.add_argument("--action_head_batch_size", type=int, default=8, help="传给服务端 action head 的批大小。")
    ap.add_argument("--action_head_stride", type=int, default=1, help="传给服务端 action head 的帧间步长。")
    ap.add_argument("--action_head_pre_resize_hw", type=int, default=256, help="actionhead 预处理前的中间缩放尺寸，默认 256。")
    ap.add_argument("--config_json", type=str, default="", help="可选服务端风格 config.json；设置后从其中读取 infinity.num_frames/step。")
    ap.add_argument("--num_frames", type=int, default=81, help="未设置 --config_json 时使用的 num_frames 默认值。")
    ap.add_argument("--step", type=int, default=16, help="未设置 --config_json 时使用的 step 默认值。")
    args = ap.parse_args()

    if args.config_json.strip():
        num_frames, step = _load_num_frames_step_from_config(args.config_json.strip())
    else:
        num_frames, step = int(args.num_frames), int(args.step)

    run_id = (args.run_id or "").strip() or _time_id()

    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    if bool(int(args.dry_run)):
        print(f"[预演] run_id={run_id} mode={args.mode} out_root={out_root}（只预览，不执行）")
        return

    try:
        parts = [p.strip() for p in str(args.resolution).split(",")]
        res_w = int(parts[0])
        res_h = int(parts[1])
    except Exception as e:
        raise SystemExit(f"--resolution 非法: '{args.resolution}'，期望格式为 'W,H'") from e

    if str(args.mode).strip().lower() == "service":
        env = _make_unrealcv_env(
            env_id=str(args.env_id),
            time_dilation_value=int(args.time_dilation),
            seed=int(args.seed),
            resolution_wh=(int(res_w), int(res_h)),
            ue_port=int(args.ue_port),
        )
        run_env_service(
            host=str(args.host),
            port=int(args.port),
            env=env,
            task_json_root=str(args.task_json_root),
            yaw_offset_deg=float(args.yaw_offset_deg),
            action_frame=str(args.action_frame),
            body_apply_order=str(args.body_apply_order),
            image_codec=str(args.image_codec),
            jpeg_quality=int(args.jpeg_quality),
        )
        return

    if str(args.mode).strip().lower() == "dataset":
        if not str(args.dataset_root).strip():
            raise SystemExit("--mode=dataset 时必须提供 --dataset_root")
        dataset_root = os.path.abspath(args.dataset_root)
        routes = _discover_routes(dataset_root)
        if args.route_id.strip():
            routes = [r for r in routes if r.route_id == args.route_id.strip()]
        if args.min_real_frames and int(args.min_real_frames) > 0:
            keep: List[Route] = []
            for r in routes:
                try:
                    n = len(_sorted_frame_paths(r.images_dir))
                except Exception:
                    n = 0
                if int(n) >= int(args.min_real_frames):
                    keep.append(r)
            routes = keep

        # 确定性选择。
        if args.select_n and int(args.select_n) > 0:
            routes = routes[: int(args.select_n)]
        if args.max_routes and args.max_routes > 0:
            routes = routes[: int(args.max_routes)]
        if not routes:
            raise SystemExit("没有找到有效 route。")

        out_root_eff = out_root
        if bool(int(args.run_subdir)):
            out_root_eff = os.path.join(out_root_eff, f"client_run_{run_id}")
        os.makedirs(out_root_eff, exist_ok=True)

        for r in routes:
            if str(args.session_id_mode) == "route_run":
                session_id = f"{r.route_id}__{run_id}"
            else:
                session_id = r.route_id
            save_dir_name = r.route_id  # 本次 run 下稳定的目录名。
            try:
                real_n = len(_sorted_frame_paths(r.images_dir))
            except Exception:
                real_n = -1
            print(f"[运行] mode=dataset route_id={r.route_id} session_id={session_id} real_frames={real_n} images_dir={r.images_dir}，开始处理")
            try:
                run_one_route(
                    route=r,
                    server_base_url=args.server_url,
                    out_root=out_root_eff,
                    save_dir_name=save_dir_name,
                    session_id=session_id,
                    num_frames=int(num_frames),
                    step=int(step),
                    prefix_mode=bool(int(args.prefix_mode)),
                    allow_future_last_segment=bool(int(args.allow_future_last_segment)),
                    action_frame=str(args.action_frame),
                    body_apply_order=str(args.body_apply_order),
                    timeout_s=int(args.timeout_s),
                    image_codec=str(args.image_codec),
                    jpeg_quality=int(args.jpeg_quality),
                    pad_short_real=bool(int(args.pad_short_real)),
                    action_head_mode=str(args.action_head_mode),
                    action_head_batch_size=int(args.action_head_batch_size),
                    action_head_stride=int(args.action_head_stride),
                    action_head_pre_resize_hw=int(args.action_head_pre_resize_hw),
                )
            except Exception as e:
                print(f"[失败] route_id={r.route_id} err={e}，该 route 处理失败")
        return

    # 中文说明：mode=unrealcv。
    task_paths: List[str] = []
    if str(args.task_json).strip():
        task_paths = [os.path.abspath(str(args.task_json).strip())]
    elif str(args.json_folder).strip():
        jf = os.path.abspath(str(args.json_folder).strip())
        if not os.path.isdir(jf):
            raise SystemExit(f"--json_folder 不是目录: {jf}")
        names = [n for n in os.listdir(jf) if n.lower().endswith(".json")]
        names.sort(reverse=(str(args.json_order).strip().lower() == "desc"))
        # 可选文件名边界。
        def _norm_json_name(s: str) -> str:
            """把起止过滤参数标准化成 `.json` 文件名。"""
            s = str(s or "").strip()
            if not s:
                return ""
            return s if s.lower().endswith(".json") else (s + ".json")

        start_name = _norm_json_name(str(args.json_start))
        end_name = _norm_json_name(str(args.json_end))
        if start_name:
            if bool(int(args.json_start_exclusive)):
                names = [n for n in names if str(n) > str(start_name)]
            else:
                names = [n for n in names if str(n) >= str(start_name)]
        if end_name:
            names = [n for n in names if str(n) <= str(end_name)]
        task_paths = [os.path.join(jf, n) for n in names]
    else:
        raise SystemExit("--mode=unrealcv 时必须提供 --task_json 或 --json_folder")
    if not task_paths:
        raise SystemExit("没有找到 task json 文件。")

    env = _make_unrealcv_env(
        env_id=str(args.env_id),
        time_dilation_value=int(args.time_dilation),
        seed=int(args.seed),
        resolution_wh=(int(res_w), int(res_h)),
        ue_port=int(args.ue_port),
    )

    for p in task_paths:
        print(f"[运行] mode=unrealcv task={p} env_id={args.env_id} resolution={res_w}x{res_h}，开始处理")
        try:
            run_one_task_unrealcv(
                task_json_path=p,
                env=env,
                env_id=str(args.env_id),
                server_base_url=args.server_url,
                out_root=out_root,
                run_id=run_id,
                num_frames=int(num_frames),
                step=int(step),
                max_actions=int(args.max_actions),
                timeout_s=int(args.timeout_s),
                action_head_mode=str(args.action_head_mode),
                action_head_batch_size=int(args.action_head_batch_size),
                action_head_stride=int(args.action_head_stride),
                action_head_pre_resize_hw=int(args.action_head_pre_resize_hw),
                image_codec=str(args.image_codec),
                jpeg_quality=int(args.jpeg_quality),
                yaw_offset_deg=float(args.yaw_offset_deg),
                allow_future_last_segment=bool(int(args.allow_future_last_segment)),
                action_frame=str(args.action_frame),
                body_apply_order=str(args.body_apply_order),
                save_images=True,
            )
        except Exception as e:
            print(f"[失败] task={p} err={e}，该 task 处理失败")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
