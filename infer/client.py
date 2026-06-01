#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WorldVLN 在线推理服务端的客户端脚本。

小白导读：
这个文件从客户端视角展示闭环协议：同一个 `session_id` 先发 1 帧预热图像，再按 `step`
发送真实新帧。当前服务端的 `tsformer_latent` Stage-2 路径返回 16 个逐帧动作；
为了兼容旧实验，客户端也能处理 4 个宏动作并拆成子步执行。

协议说明：
- 一条轨迹对应一个 `session_id`，通常直接使用 route 文件夹名。
- 帧是增量上传的：第一次上传 1 帧预热图像，之后每次上传 `step` 帧，直到达到 `num_frames`。

服务端输出：
- 动作单位是 cm/deg，顺序为 `[dx, dy, dz, droll, dyaw, dpitch]`。
- `action_head_mode=tsformer_latent`：当前 Stage-2 服务端每段输出 `step` 个逐帧动作；
  旧 P2P checkpoint 可能只输出 4 个宏动作，本客户端保留两种兼容逻辑。
- `action_head_mode=actionhead_ref_vit`：每段输出 `step` 个逐帧动作，默认 step=16 时就是 16 个动作。

客户端输出：每个 segment 写两个 JSON 文件，文件名包含 session_id。
步骤说明：1. actions JSON：
   - `actions_server_order`: 服务端原始动作顺序，形状 Nx6；
   - `actions_client_order`: 下游部分工具使用的另一种角度顺序；
   - `action_frames`: 每个动作对应的帧标识，dataset 模式是路径，UnrealCV 模式是保存文件名；
   - `cumsum_*`: 文件内部的动作逐维累加，方便肉眼检查漂移。
2. poses JSON（绝对坐标）：
   - segment 0 会包含起点和 N 个终点，共 1+N 个点；
   - 后续 segment 只包含 N 个新终点；
   - pose 顺序是 `[x, y, z, roll, yaw, pitch]`，单位 cm/deg。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def _read_json(path: str):
    """读取 UTF-8 JSON 文件；客户端所有 meta、task、输出回写都复用这个入口。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _time_id() -> str:
    """生成本次客户端运行的时间戳 ID，避免同一路线多次运行互相覆盖。"""
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _sorted_frame_paths(images_dir: str) -> List[str]:
    """按文件名排序读取 route/images 下的真实观测帧路径。"""
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    names = [n for n in os.listdir(images_dir) if n.lower().endswith(exts)]
    names.sort()
    return [os.path.join(images_dir, n) for n in names]


def _take_with_pad(paths: List[str], n: int, pad_short_real: bool) -> List[str]:
    """截取前 n 帧；真实帧不足且允许 padding 时，重复最后一帧补齐。"""
    if len(paths) >= int(n):
        return paths[: int(n)]
    if not paths:
        raise ValueError("没有找到真实图像帧")
    if not bool(pad_short_real):
        raise ValueError(f"至少需要 {n} 帧真实图像，但只找到 {len(paths)} 帧；可用 --pad_short_real 1 重复最后一帧补齐")
    return paths + [paths[-1]] * (int(n) - int(len(paths)))


def _image_to_data_url_jpeg(path: str, quality: int = 90) -> str:
    """把磁盘图片编码成服务端请求使用的 JPEG data URL。"""
    return _image_to_data_url(path, codec="jpeg", quality=int(quality))


def _pil_to_data_url_jpeg(img: Image.Image, quality: int = 90) -> str:
    """把内存中的 PIL 图像编码成 JPEG data URL。"""
    return _pil_to_data_url(img, codec="jpeg", quality=int(quality))


def _image_to_data_url(path: str, *, codec: str, quality: int = 90) -> str:
    """读取图片文件并编码为 `data:image/...;base64,...`，用于 HTTP JSON 传输。"""
    img = Image.open(path).convert("RGB")
    return _pil_to_data_url(img, codec=str(codec), quality=int(quality))


def _pil_to_data_url(img: Image.Image, *, codec: str, quality: int = 90) -> str:
    """将 PIL RGB 图像按 JPEG/PNG 编码到 base64 data URL。"""
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
        raise ValueError(f"不支持的图像编码格式：{codec}")
    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _ensure_dir(path: str) -> None:
    """创建输出目录；重复调用不报错。"""
    os.makedirs(path, exist_ok=True)


def _save_pil_jpeg(img: Image.Image, path: str, *, quality: int = 95) -> None:
    """保存客户端采集/回放的帧，主要用于复查每个动作对应的真实图像。"""
    parent = os.path.dirname(path)
    if parent:
        _ensure_dir(parent)
    img.convert("RGB").save(path, format="JPEG", quality=int(quality))


def _safe_np_image_to_pil_rgb(img_any: Any) -> Image.Image:
    """
    把 UnrealCV 返回的图像尽量稳健地转成 PIL RGB。

    许多 `get_image(...)` 实现返回 `np.ndarray(H,W,3)`，且通道顺序常常是 BGR；
    这里统一转成 RGB，避免后续 base64 编码和保存时颜色错位。
    """
    if isinstance(img_any, Image.Image):
        return img_any.convert("RGB")
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("把 UnrealCV 图像转成 PIL RGB 需要 numpy，但当前环境不可用") from e
    if not isinstance(img_any, np.ndarray):
        raise TypeError(f"不支持的 UnrealCV 图像类型：{type(img_any)}")
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
    raise ValueError(f"不支持的 ndarray 图像形状：{getattr(arr, 'shape', None)}")


def _reorder_server_to_client(a6: List[float]) -> List[float]:
    """
    把服务端动作顺序转换成部分旧客户端工具使用的顺序。

    服务端顺序：`[dx,dy,dz,droll,dyaw,dpitch]`
    客户端兼容顺序：`[dx,dy,dz,droll,dpitch,dyaw]`
    """
    if len(a6) != 6:
        raise ValueError(f"action 必须是 6 维，收到 {len(a6)} 维")
    dx, dy, dz, droll, dyaw, dpitch = [float(x) for x in a6]
    return [dx, dy, dz, droll, dpitch, dyaw]


def _cumsum_actions(actions: List[List[float]]) -> List[List[float]]:
    """对动作序列逐维累加，输出到 JSON 方便肉眼检查累计位移/角度是否漂移。"""
    out: List[List[float]] = []
    cur = [0.0] * 6
    for a in actions:
        cur = [cur[i] + float(a[i]) for i in range(6)]
        out.append(cur)
    return out


def _apply_action_to_pose(pose_xyz_rpy: List[float], action_dxdy_dz_droll_dyaw_dpitch: List[float]) -> List[float]:
    """
    用最简单的世界坐标系加法积分动作。

    `pose`: `[x,y,z,roll,yaw,pitch]`，单位 cm/deg；
    `action`: `[dx,dy,dz,droll,dyaw,dpitch]`，单位 cm/deg；
    当前实现是 `pose_next = pose + delta`，适合离线检查和简单轨迹可视化。
    """
    if len(pose_xyz_rpy) != 6:
        raise ValueError(f"pose 必须是 6 维，收到 {len(pose_xyz_rpy)} 维")
    if len(action_dxdy_dz_droll_dyaw_dpitch) != 6:
        raise ValueError(f"action 必须是 6 维，收到 {len(action_dxdy_dz_droll_dyaw_dpitch)} 维")
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
        `pose` 是 `[x,y,z,roll,yaw,pitch]`，单位 cm/deg。
        `action` 是 `[dx,dy,dz,droll,dyaw,dpitch]`，单位 cm/deg。

        - `action_frame="world"`：直接相加，表示动作已经是世界坐标系 delta；
        - `action_frame="body"`：把 `(dx,dy)` 当作机体系前进/右移，并按 yaw 旋到世界系；
          `body_apply_order` 决定先转再移、先移再转，还是用中点积分。

        机体系动作转世界系的核心公式：
        - 公式：x += dx*cos(theta) - dy*sin(theta)
        - 公式：y += dx*sin(theta) + dy*cos(theta)
        其中 theta 由 yaw 决定，单位 radians（代码内部用 math.radians(yaw) 换算）。

        三种 body_apply_order（决定 theta 取转向前/后/中的哪一刻 yaw）：
        - `yaw_first`：先 `yaw += dyaw` 再平移，theta 用转向后的新 yaw（先转再走）；
        - `trans_first`：用旧 yaw 当 theta 平移，再 `yaw += dyaw`（先走再转）；
        - `midpoint`：theta 用 `yaw + 0.5*dyaw`（边转边走的近似中点积分）。

        注意：`dz` 直接沿世界 Z 轴累加；当 pitch/roll 很小或被忽略时，这个近似通常够用。

    """
    if len(pose_xyz_rpy) != 6 or len(action_dxdy_dz_droll_dyaw_dpitch) != 6:
        raise ValueError("pose/action 都必须是 6 维")
    x, y, z, roll, yaw, pitch = [float(v) for v in pose_xyz_rpy]
    dx, dy, dz, droll, dyaw, dpitch = [float(v) for v in action_dxdy_dz_droll_dyaw_dpitch]

    fr = str(action_frame).strip().lower()
    if fr == "world":
        x += dx
        y += dy
        z += dz
    elif fr == "body":
        # 用 yaw 把机体系的前进/右移旋转到世界系 x/y。
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
            raise ValueError(f"非法 body_apply_order={body_apply_order}，只支持 yaw_first|trans_first|midpoint")
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # 二维旋转公式：body 中 dx 是前进、dy 是右移；乘上旋转矩阵后变成 world 的 x/y 增量。
        x += dx * cos_t - dy * sin_t
        y += dx * sin_t + dy * cos_t
        z += dz
    else:
        raise ValueError(f"非法 action_frame={action_frame}，只支持 world|body")

    # 角度相关变量。
    if fr == "world":
        yaw += dyaw
    if bool(integrate_roll_pitch):
        roll += droll
        pitch += dpitch
    return [x, y, z, roll, yaw, pitch]


def _http_post_json(url: str, payload: Dict, *, timeout_s: int = 120) -> Dict:
    """向 FastAPI 服务端发送一次 `/v1/predict_delta_actions` 请求并返回 JSON。"""
    try:
        import requests  # type: ignore
    except Exception as e:
        raise RuntimeError("客户端发送 HTTP 请求需要 requests，但当前环境不可用") from e

    r = requests.post(url, json=payload, timeout=int(timeout_s))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


@dataclass
class Route:
    """离线 dataset 模式下的一条 route：图像目录、meta 指令和可选起始位姿日志。"""
    route_dir: str
    route_id: str
    images_dir: str
    meta_path: str
    raw_logs_path: Optional[str]


def _discover_routes(dataset_root: str) -> List[Route]:
    """扫描 dataset_root 下所有包含 `images/` 和 `meta.json` 的 route 目录。"""
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
    """从 meta.json 中按兼容字段顺序读取导航指令。"""
    meta = _read_json(meta_path)
    prompt = (meta.get("instruction") or meta.get("instruction_unified") or meta.get("prompt") or "").strip()
    return str(prompt)


def _load_start_pose_cm_deg(raw_logs_path: Optional[str]) -> List[float]:
    """从 raw_logs.json 读取第 0 帧绝对位姿；缺失时返回全零位姿。"""
    if not raw_logs_path or not os.path.exists(raw_logs_path):
        return [0.0] * 6
    arr = _read_json(raw_logs_path)
    if not isinstance(arr, list) or len(arr) == 0:
        return [0.0] * 6
    p0 = arr[0]
    if not (isinstance(p0, (list, tuple)) and len(p0) == 6):
        return [0.0] * 6
    return [float(x) for x in p0]  # 代码/形状说明：[x,y,z,roll,yaw,pitch] in cm/deg


def _load_num_frames_step_from_config(config_json: str) -> Tuple[int, int]:
    """从服务端 config.json 中读取 `infinity.num_frames` 和 `infinity.step`。"""
    cfg = _read_json(config_json)
    if not isinstance(cfg, dict):
        raise ValueError(f"config json 格式不合法，期望顶层是 dict：{config_json}")
    inf = cfg.get("infinity", cfg)
    num_frames = int(inf.get("num_frames", 81))
    step = int(inf.get("step", 16))
    if num_frames <= 0 or step <= 0:
        raise ValueError(f"config 中 num_frames/step 不合法：num_frames={num_frames} step={step}")
    return num_frames, step


def _load_instruction_and_initial_pose_from_task_json(task_json_path: str) -> Tuple[str, List[float]]:
    """
    读取 UAV-Flow-Eval 任务 JSON（例如 `test_jsons/*.json`）并取出在线执行所需字段。

    输出：
    - `instruction` 或 `instruction_unified`：传给 server/world model 的导航语言指令。
    - `initial_pos`: `[x,y,z,roll,yaw,pitch]`，单位是 cm/deg，用来初始化 UnrealCV 位置。

    数据流位置：
    `run_one_task_unrealcv()` 会先调用这里拿到任务文本和初始位姿，然后才开始抓首帧、
    上传 server，并把 server 返回的相对动作积分成下一帧位置。
    """
    d = _read_json(task_json_path)
    if not isinstance(d, dict):
        raise ValueError(f"task json 格式不合法，期望顶层是 dict：{task_json_path}")
    instr = (d.get("instruction") or d.get("instruction_unified") or "").strip()
    if not instr:
        raise ValueError(f"task json 中 instruction 为空：{task_json_path}")
    initial_pos = d.get("initial_pos", None)
    if not (isinstance(initial_pos, list) and len(initial_pos) >= 6):
        raise ValueError(f"task json 中 initial_pos 不合法，至少需要 6 维：{task_json_path}")
    init6 = [float(x) for x in initial_pos[:6]]
    return instr, init6


def _build_obj_info_from_task_json(task_json_path: str) -> Optional[Dict[str, Any]]:
    """
    从任务 JSON 构造 UnrealCV 目标物体放置信息，对齐 `batch_run_act_all.py`。

    读取规则：
    - 只有同时存在 `obj_id` 和 `use_obj` 时才放置物体；
    - 优先使用 `target_pos[:3] / target_pos[3:]` 作为 `obj_pos / obj_rot`；
    - 如果没有 `target_pos`，再回退到 `obj_pos / obj_rot`。

    数据流位置：
    返回值会传给 `_create_obj_if_needed_unrealcv()`，在每条任务开始前把目标物体放进场景。
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
    在场景中创建/初始化一次 marker objects，对齐 `batch_run_act_all.py` 的初始化行为。
    """
    # 避免同一进程跑多个任务时重复初始化。
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
        # 对象可能已存在，或不同地图里类名不同；不要因此中断主控制流程。
        env.unwrapped._xjc_marker_inited = True


def _create_obj_if_needed_unrealcv(env: Any, obj_info: Optional[Dict[str, Any]]) -> None:
    """
    按任务配置放置目标物体，逻辑对齐 `batch_run_act_all.py` 的 `create_obj_if_needed`。
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
        # 不因场景资产差异中断主控制流程。
        pass


def _setup_unrealcv_camera_follow(env: Any, *, cam_id: int = 0) -> None:
    """
    把相机绑定到 UAV 当前位置，近似第一人称视角。

    该逻辑对齐 `batch_run_act_all.py` 的 `set_cam`。
    """
    x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])  # 代码/形状说明：[roll, yaw, pitch]
    cam_loc = [x, y, z]
    cam_rot = [roll, pitch, yaw]  # 中文说明：UnrealCV `set_cam` 使用 `[roll,pitch,yaw]` 旋转顺序
    env.unwrapped.unrealcv.set_cam(int(cam_id), cam_loc, cam_rot)


def _apply_pose_unrealcv(
    env: Any,
    *,
    pose_xyz_rpy: List[float],
    yaw_offset_deg: float = -180.0,
) -> None:
    """
    把 `[x,y,z,roll,yaw,pitch]` 应用到仿真器，单位 cm/deg。

    对无人机来说，gym_unrealcv 的 `set_obj_rotation` 不一定稳定生效，
    因此这里用 `set_rotation(yaw)` 控制朝向。
    """
    if len(pose_xyz_rpy) < 6:
        raise ValueError(f"pose 必须至少是 6 维，收到 {len(pose_xyz_rpy)} 维")
    x, y, z, _roll, yaw, _pitch = [float(v) for v in pose_xyz_rpy[:6]]
    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], [x, y, z])
    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], float(yaw) + float(yaw_offset_deg))
    _setup_unrealcv_camera_follow(env, cam_id=0)


def _capture_unrealcv_lit_pil(env: Any, *, cam_id: int = 0) -> Image.Image:
    """刷新跟随相机并抓取 UnrealCV lit 视图，统一转成 PIL RGB。"""
    _setup_unrealcv_camera_follow(env, cam_id=int(cam_id))
    img = env.unwrapped.unrealcv.get_image(int(cam_id), "lit")
    return _safe_np_image_to_pil_rgb(img)


def _angle_diff_deg(a: float, b: float) -> float:
    """计算两个角度的最短环形差值，单位 degree。"""
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
    """等待 set_obj_location/set_rotation 生效，降低第一帧抓到旧画面的概率。"""
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


def _split_action_to_substeps(action6: List[float], substeps: int) -> List[List[float]]:
    """把旧 4 macro action 均分成多个子步，以兼容 16 帧闭环执行节奏。"""
    if len(action6) != 6:
        raise ValueError(f"action 必须是 6 维，收到 {len(action6)} 维")
    k = int(substeps)
    if k <= 0:
        raise ValueError(f"substeps 必须大于 0，收到 {substeps}")
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
    在线模式（`gym_unrealcv`）下跑单条任务。

    主数据流：
    1. 从 `task_json` 读取 `instruction` 和 `initial_pos`。
    2. 初始化 UnrealCV 场景、目标物体和首帧相机。
    3. 抓取 256x256 lit RGB 帧（分辨率由 `ConfigUEWrapper` 控制）。
    4. 按增量方式上传帧：首帧 1 张，之后每轮追加 `step` 张；`prefix_mode=false`
       时历史帧由 server 侧按 `session_id` 缓存。
    5. `tsformer_latent` 模式下，Stage-2 server 当前返回 `step=16` 个逐帧动作；
       旧版 4 个 macro action 仍兼容，会被拆成 16 个子动作。
    6. `actionhead_ref_vit` 模式下，直接接收 16 个逐帧动作，逐个执行并保存新帧。

    首轮特殊情况：
    当 `n==1` 时 server 可能只 warm up，不返回动作。client 会原地抓取并上传
    `step` 张帧，把 server 时间线推进到可以生成第 0 段动作的位置。
    """
    instruction, init_pose = _load_instruction_and_initial_pose_from_task_json(task_json_path)
    obj_info = _build_obj_info_from_task_json(task_json_path)

    base_name = os.path.splitext(os.path.basename(task_json_path))[0]
    session_id = f"{base_name}__{run_id}"
    out_dir = os.path.join(os.path.abspath(out_root), f"client_run_{run_id}", base_name)
    _ensure_dir(out_dir)
    images_dir = os.path.join(out_dir, "images")
    _ensure_dir(images_dir)

    # 每个任务开始前重置环境状态。
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
    # 给相机额外刷新时间，降低第一帧抓到旧视角的风险。
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
        """
        发送本轮新增真实帧；首轮附带 instruction 并重置同名 session。

        HTTP 调用契约（服务端 `/v1/predict_delta_actions` 期望的字段）：
        - `session_id`：同一条轨迹必须保持一致，服务端用它定位 TrajectoryState；
        - `images_base64`：本轮新增的真实观测帧（base64 data URL），首轮通常 1 帧、
          后续每轮 `step` 帧；服务端按 `points` 边界增量缓存这些帧；
        - `prefix_mode=False`：表示走增量协议，服务端不会把已有帧再次解码；
        - `allow_future_*`：在帧数仅到达 `points[seg]` 时允许提前输出 segment 动作；
        - `action_head_*`：传给服务端选择的动作头模式 / 滑窗参数；
        - `instruction`/`reset_session`：仅首轮需要，强制服务端按新轨迹建立内部 session。

        返回的 JSON 字段：
        - `actions`: 本轮动作列表，长度由动作头模式决定（默认 16 个）；
        - `segment_index`: 本轮发射的 segment（-1 表示尚未 ready）；
        - `num_received_frames` / `done` / `prefix_latents`：闭环进度指示。
        """
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

    # 第 1 帧是初始观测。
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
        """把服务端返回的动作、累计动作和客户端积分位姿写成每段 JSON。"""
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

        # 每执行一个动作后记录一个 pose 点；动作可能是旧版宏动作，也可能是当前逐帧动作。
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

    # 判断服务端本轮应该返回几个动作。
    # 当前 `tsformer_latent` Stage-2 返回逐帧动作；旧 P2P 风格路径返回 4 个宏动作。
    mode_l = str(action_head_mode).strip().lower()
    is_per_frame_mode = mode_l in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead")
    expected_actions = int(step) if bool(is_per_frame_mode) else 4

    call_i = 0
    last_upload_n = 1  # 第一次上传只有 1 帧预热图像。
    max_actions_i = int(max_actions)
    # 停止条件：
    # - max_actions>0：执行到指定动作数就停，严格协议下 48 个动作对应 1+48 帧；
    # - 否则按 num_frames 上限停止。
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

        # 收集下一轮要上传的 `step` 帧真实观测。
        new_frames: List[Image.Image] = []
        if pending_actions is None:
            # 服务端还没返回动作时，保持位置并继续采集帧，直到满足 segment 发射条件。
            for _ in range(int(step)):
                if max_actions_i > 0:
                    # 严格动作数模式下，不能在没有执行动作的情况下推进帧。
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
                # 逐帧动作可直接执行：通常 16 个动作生成 16 帧。
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
                # 兼容旧式 4 个宏动作：每个动作拆成 4 个子步，共 16 帧。
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
            # 如果生成帧数不足，只在 num_frames 模式下通过原地采样补帧；
            # max_actions 模式严格保持“每执行一个动作对应一帧”，不额外补。
            if max_actions_i <= 0:
                while produced < total_needed and frame_idx < int(num_frames):
                    im = _capture_unrealcv_lit_pil(env, cam_id=0)
                    frame_idx += 1
                    produced += 1
                    if bool(save_images):
                        _save_pil_jpeg(im, os.path.join(images_dir, f"frame_{frame_idx:04d}.jpg"), quality=95)
                    new_frames.append(im)

            # 尽力把每个动作对应的帧名写回最新 segment action 日志；
            # 只有文件存在且长度匹配时才重写。
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

    # 尽力更新 summary 文件里的最终计数。
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
    """增量流式上传切分：第 1 次 1 帧，之后每次 `step` 帧。"""
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
    和服务端一致的 segment 边界逻辑。

    闭环 segment 边界公式：
    - 公式：points = [1, 1+step, 1+2*step, ..., num_frames]
    例如 `pred_num_frames=49, step=16` 时返回 `[1,17,33,49]`：
    第 1 帧是预热观测，之后每 16 帧（一个 step）作为一个新 segment 的右边界。

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
    prefix 模式：每次发送完整观测前缀，以匹配 v2v 的语义。

    例子：
    - 第 0 次调用 `call0`: `[1]`
    - 第 1 次调用 `call1`: `[1..17]`
    - 第 2 次调用 `call2`: `[1..33]`
    - 第 3 次调用 `call3`: `[1..49]`
    """
    p = paths[: int(num_frames)]
    pts = _obs_points(int(num_frames), int(step))
    if not pts or pts[0] != 1 or pts[-1] != int(num_frames):
        raise ValueError(f"根据 num_frames={num_frames}, step={step} 计算出的 points 不合法：{pts}")
    return [p[: int(k)] for k in pts]


def _chunks_prefix_from_points(paths: List[str], *, points: List[int]) -> List[List[str]]:
    """按显式绝对帧边界构造 prefix chunks，例如 `[1,17,33]`。"""
    if not points or int(points[0]) != 1:
        raise ValueError(f"points 不合法，必须从 1 开始：{points}")
    max_k = int(max(points))
    if len(paths) < max_k:
        raise ValueError(f"points={points} 至少需要 {max_k} 帧，但只收到 {len(paths)} 帧")
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
    """
    离线 dataset 模式：从 route/images 读取真实帧，按服务端闭环协议分段上传。

    这个函数不会执行动作，只把已有真实帧作为“执行后观测”回放给服务端，并把每个 segment
    的 actions/poses JSON 写到输出目录。它适合用来调试服务协议、动作长度和单位转换。
    """
    prompt = _load_prompt(route.meta_path)
    if not prompt:
        raise RuntimeError(f"meta 中 prompt/instruction 为空：{route.meta_path}")
    start_pose = _load_start_pose_cm_deg(route.raw_logs_path)

    real_paths = _sorted_frame_paths(route.images_dir)
    real_count = int(len(real_paths))

    obs_points = _obs_points(int(num_frames), int(step))  # e.g. [1,17,33,49]
    if len(obs_points) < 2:
        raise RuntimeError(f"根据 num_frames={num_frames}, step={step} 计算出的 obs_points 不合法：{obs_points}")

    # 如果允许在没有 points[-1] 真实帧时输出最后一段，
    # 数据集中只需要存在到 points[-2]（例如 33）的真实前缀。
    send_points = obs_points
    if bool(prefix_mode) and bool(allow_future_last_segment) and int(obs_points[-1]) > int(obs_points[-2]):
        send_points = obs_points[:-1]  # 丢弃最后 49 帧。

    max_need = int(max(send_points))
    frame_paths = _take_with_pad(real_paths, max_need, bool(pad_short_real))

    paths_for_map: List[str] = []
    if prefix_mode:
        chunks = _chunks_prefix_from_points(frame_paths, points=[int(k) for k in send_points])
        paths_for_map = frame_paths
        # 不增加真实帧也触发最后一个 segment：
        # 再发送一次最长 prefix，服务端看到“没有新帧”，但会发射最后一段预测动作。
        if bool(allow_future_last_segment) and int(obs_points[-1]) > int(obs_points[-2]):
            chunks.append(chunks[-1])
    else:
        # stream 模式使用增量 chunks，因此需要完整 num_frames 帧。
        full_paths = _take_with_pad(real_paths, int(num_frames), bool(pad_short_real))
        chunks = _chunks_stream(full_paths, num_frames=int(num_frames), step=int(step))
        paths_for_map = full_paths

    out_dir = os.path.join(out_root, str(save_dir_name or session_id))
    os.makedirs(out_dir, exist_ok=True)

    # 输入摘要只保存一次，便于回看本次离线回放的配置。
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

    cur_pose = start_pose[:]  # 绝对位姿单位为 cm/deg。

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
        # 预热调用只有第一帧，可能返回空动作且 segment_index=-1。
        if seg < 0:
            continue
        # 按动作头模式检查返回动作数量。
        # 当前 Stage-2 `tsformer_latent` 每个相邻帧 transition 返回一个动作；
        # 旧 checkpoint/工具可能仍返回 4 个宏动作，因此这里继续兼容。
        mode_l = str(action_head_mode).strip().lower()
        if mode_l in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead"):
            if seg < 0 or seg >= len(obs_points) - 1:
                raise RuntimeError(f"服务端返回的 segment_index 不合法：seg={seg} obs_points={obs_points}")
            expected_n = int(obs_points[seg + 1]) - int(obs_points[seg])
        else:
            expected_n = 4
        if not isinstance(actions, list) or len(actions) != int(expected_n):
            raise RuntimeError(
                f"服务端返回的 actions 不合法：call={call_i}, segment={seg}, "
                f"type={type(actions)} len={getattr(actions, '__len__', lambda: None)()}"
            )

        actions_server = [[float(x) for x in a] for a in actions]
        actions_client = [_reorder_server_to_client(a) for a in actions_server]
        cumsum_server = _cumsum_actions(actions_server)
        cumsum_client = _cumsum_actions(actions_client)

        # dataset 模式下记录“每个动作对应哪一帧”。
        # 只有服务端返回逐帧 transition 动作时，这个映射才真正有意义。
        action_frames: List[Dict[str, Any]] = []
        if mode_l in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead") and seg >= 0 and seg < len(obs_points) - 1:
            obs_len = int(obs_points[seg])
            for i in range(int(len(actions_server))):
                abs_from = int(obs_len) + int(i)
                abs_to = int(obs_len) + int(i) + 1
                img_path_upload: Optional[str] = None
                img_path_real: Optional[str] = None

                # 上传给服务端的帧；开启 --pad_short_real=1 时可能是重复补齐帧。
                if 1 <= abs_to <= len(paths_for_map):
                    img_path_upload = paths_for_map[int(abs_to) - 1]

                # 数据集中的真实帧；allow_future_last_segment 模式下可能存在但没有上传。
                if 1 <= abs_to <= int(real_count):
                    img_path_real = real_paths[int(abs_to) - 1]

                is_real = 1 <= abs_to <= int(real_count)
                is_padded = (not is_real) and (img_path_upload is not None)

                # 兼容旧字段：优先写真实帧路径，否则写上传用的补齐帧路径。
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

        # 写出每个 segment 对应的 actions JSON。
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

        # 中文说明：这里保存的是每执行一个动作后的绝对位姿点；
        # 积分时使用当前配置的 `action_frame` 和 `body_apply_order`。
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
            "points": pose_points,  # 中文说明：seg0 写 1+N 个点（起点+N个终点），后续段只写 N 个新终点
        }
        with open(os.path.join(out_dir, f"{session_id}_seg{seg:02d}_poses.json"), "w", encoding="utf-8") as f:
            json.dump(poses_json, f, ensure_ascii=False, indent=2)


def main():
    """
    命令行入口，支持两种模式：
    - `dataset`: 回放数据集已有帧，验证服务端协议和 JSON 输出；
    - `unrealcv`: 连接 gym_unrealcv，按服务端动作真实推进仿真并采集下一帧。

    推荐阅读方式：
    1. 先看 `dataset` 模式，因为它不依赖仿真环境，最适合理解 HTTP 协议和 JSON 落盘格式；
    2. 再看 `unrealcv` 模式，理解“动作真正执行后，下一帧真实观测如何回传给服务端”。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="dataset", choices=["dataset", "unrealcv"], help="运行模式。dataset：从 dataset_root 回放真实帧；unrealcv：从 gym_unrealcv 抓取真实帧并在线执行动作。")
    ap.add_argument("--dataset_root", type=str, default="", help="(mode=dataset) 数据集根目录，内部应包含多个 route 文件夹，每个 route 至少有 images/ 和 meta.json，可选 raw_logs.json。")
    ap.add_argument("--task_json", type=str, default="", help="(mode=unrealcv) 单个任务 json 路径，例如 ./test_jsons/2025-03-30_11-49-14.json。")
    ap.add_argument("--json_folder", type=str, default="", help="(mode=unrealcv) 包含多个任务 json 的目录。")
    ap.add_argument("--json_order", type=str, default="asc", choices=["asc", "desc"], help="(mode=unrealcv) 遍历 --json_folder 时按文件名升序还是降序。")
    ap.add_argument("--json_start", type=str, default="", help="(mode=unrealcv) 对 --json_folder 的可选起始文件名下界，例如 2025-03-30_12-02-10 或带 .json 后缀的完整名。")
    ap.add_argument("--json_start_exclusive", type=int, default=0, choices=[0, 1], help="(mode=unrealcv) 若为 1，则严格从 --json_start 之后开始，适合断点续跑。")
    ap.add_argument("--json_end", type=str, default="", help="(mode=unrealcv) 对 --json_folder 的可选终止文件名上界（包含该文件）。格式与 --json_start 相同。")
    ap.add_argument("--env_id", type=str, default="UnrealTrack-DowntownWest-ContinuousColor-v0", help="(mode=unrealcv) gym 环境 id。")
    ap.add_argument("--time_dilation", type=int, default=10, help="(mode=unrealcv) 时间膨胀包装器参数；值越大通常仿真推进越慢、更稳。")
    ap.add_argument("--seed", type=int, default=0, help="(mode=unrealcv) 随机种子。")
    ap.add_argument("--resolution", type=str, default="256,256", help="(mode=unrealcv) 抓图分辨率，格式为 'W,H'，默认 256,256。")
    ap.add_argument("--ue_port", type=int, default=0, help="(mode=unrealcv) 若 >0，则在启动 UE 前写入 unrealcv.ini 的 socket 基准端口，适合多实例并行。")
    ap.add_argument("--yaw_offset_deg", type=float, default=-180.0, help="(mode=unrealcv) 调用 UnrealCV set_rotation 时附加的 yaw 偏移角。历史脚本 batch_run_act_all.py 默认使用 -180。")
    ap.add_argument("--action_frame", type=str, default="body", choices=["world", "body"], help="积分位姿时如何解释 dx/dy：world 表示直接在世界系相加；body 表示把 dx/dy 当作机体系前进/右移，再按 yaw 旋到世界系。")
    ap.add_argument("--body_apply_order", type=str, default="yaw_first", choices=["yaw_first", "trans_first", "midpoint"], help="仅对 --action_frame=body 生效。yaw_first：先 yaw+=dyaw 再平移；trans_first：先按旧 yaw 平移再更新 yaw；midpoint：按 yaw+0.5*dyaw 近似边转边走。")
    ap.add_argument("--max_actions", type=int, default=48, help="(mode=unrealcv) 若 >0，则执行到指定动作数就停止。默认 48，对应 1 张首帧 + 48 个动作 = 49 帧总长度。设为 0 时改用 --num_frames 限制。")
    ap.add_argument("--server_url", type=str, default="http://127.0.0.1:8002", help="服务端根 URL，不要带尾部 endpoint。")
    ap.add_argument("--out_dir", type=str, default=r"E:\xjc\UAV-Flow-main1\UAV-Flow-main\UAV-Flow-Eval\cache", help="每条 session 的 JSON 输出根目录。")
    ap.add_argument("--route_id", type=str, default="", help="若设置，只运行这个 route id（即文件夹名）。")
    ap.add_argument("--max_routes", type=int, default=0, help="若 >0，限制最多处理多少条 route。")
    ap.add_argument("--select_n", type=int, default=0, help="若 >0，在筛选完成后只保留前 N 条 route，顺序确定。")
    ap.add_argument("--min_real_frames", type=int, default=0, help="若 >0，只保留真实图像帧数不少于该阈值的 route。")
    ap.add_argument("--pad_short_real", type=int, default=0, choices=[0, 1], help="若为 1，当 route 真实帧不足时重复最后一帧补齐。")
    ap.add_argument("--run_id", type=str, default="", help="可选运行 id；留空时自动使用当前时间戳。")
    ap.add_argument("--run_subdir", type=int, default=1, choices=[0, 1], help="若为 1，则把本次输出写到 out_dir/client_run_<run_id>/ 下，避免覆盖旧结果。")
    ap.add_argument("--session_id_mode", type=str, default="route_run", choices=["route", "route_run"], help="构造 server session_id 的方式；route_run 更安全，可避免覆盖服务端 latent cache 目录。")
    ap.add_argument("--allow_future_last_segment", type=int, default=1, choices=[0, 1], help="若为 1，则允许只凭 33 张真实前缀帧发射最后一段 seg02（34-49 由模型预测），需要服务端支持。")
    ap.add_argument("--dry_run", type=int, default=0, choices=[0, 1], help="若为 1，只打印本次选择到的任务/route 信息，不真正执行。")
    ap.add_argument("--image_codec", type=str, default="jpeg", choices=["jpeg", "jpg", "png"], help="上传给服务端时使用的图像编码格式。")
    ap.add_argument("--jpeg_quality", type=int, default=90, help="仅在 --image_codec=jpeg/jpg 时生效。")
    ap.add_argument("--prefix_mode", type=int, default=0, choices=[0, 1], help="若为 1，每次都发送完整前缀：1、1-17、1-33……；否则只发送增量新帧。需要服务端支持。")
    ap.add_argument("--timeout_s", type=int, default=600, help="HTTP 请求超时时间（秒）；seg0 首次推理可能较慢。")
    ap.add_argument("--action_head_mode", type=str, default="actionhead_ref_vit", choices=["tsformer_latent", "actionhead_ref_vit"], help="请求服务端使用哪种动作头模式。")
    ap.add_argument("--action_head_batch_size", type=int, default=8, help="请求 `actionhead_ref_vit` 时使用的滑窗批大小。")
    ap.add_argument("--action_head_stride", type=int, default=1, help="请求 `actionhead_ref_vit` 时使用的滑窗步长。")
    ap.add_argument("--action_head_pre_resize_hw", type=int, default=256, help="送入 actionhead 预处理前的中间 resize 尺寸；默认 256。")
    ap.add_argument("--config_json", type=str, default="", help="可选：服务端风格的 config.json；若设置，则从中读取 infinity.num_frames 和 infinity.step。")
    ap.add_argument("--num_frames", type=int, default=81, help="当 --config_json 未设置时使用的回退 num_frames。")
    ap.add_argument("--step", type=int, default=16, help="当 --config_json 未设置时使用的回退 step。")
    args = ap.parse_args()

    if args.config_json.strip():
        num_frames, step = _load_num_frames_step_from_config(args.config_json.strip())
    else:
        num_frames, step = int(args.num_frames), int(args.step)

    run_id = (args.run_id or "").strip() or _time_id()

    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    if bool(int(args.dry_run)):
        print(f"[预演] run_id={run_id} mode={args.mode} 输出根目录={out_root}")
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

        # 确定性选择，保证复现实验结果。
        if args.select_n and int(args.select_n) > 0:
            routes = routes[: int(args.select_n)]
        if args.max_routes and args.max_routes > 0:
            routes = routes[: int(args.max_routes)]
        if not routes:
            raise SystemExit("没有找到可用 route，请检查 dataset_root、route_id、min_real_frames 等过滤条件")

        out_root_eff = out_root
        if bool(int(args.run_subdir)):
            out_root_eff = os.path.join(out_root_eff, f"client_run_{run_id}")
        os.makedirs(out_root_eff, exist_ok=True)

        for r in routes:
            if str(args.session_id_mode) == "route_run":
                session_id = f"{r.route_id}__{run_id}"
            else:
                session_id = r.route_id
            save_dir_name = r.route_id  # 当前运行下使用稳定的文件夹名。
            try:
                real_n = len(_sorted_frame_paths(r.images_dir))
            except Exception:
                real_n = -1
            print(f"[运行] mode=dataset route_id={r.route_id} session_id={session_id} 真实帧数={real_n} images_dir={r.images_dir}")
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
                print(f"[失败] route_id={r.route_id} 错误={e}")
        return

    # mode=unrealcv：从任务 JSON 或目录中收集待运行任务。
    task_paths: List[str] = []
    if str(args.task_json).strip():
        task_paths = [os.path.abspath(str(args.task_json).strip())]
    elif str(args.json_folder).strip():
        jf = os.path.abspath(str(args.json_folder).strip())
        if not os.path.isdir(jf):
            raise SystemExit(f"--json_folder 不是目录：{jf}")
        names = [n for n in os.listdir(jf) if n.lower().endswith(".json")]
        names.sort(reverse=(str(args.json_order).strip().lower() == "desc"))
        # 可选文件名边界筛选。
        def _norm_json_name(s: str) -> str:
            """把起止过滤参数统一成 `.json` 文件名，便于字符串边界筛选。"""
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
        raise SystemExit("没有找到 task json 文件")

    # 解析分辨率字符串 `"W,H"`。
    try:
        parts = [p.strip() for p in str(args.resolution).split(",")]
        res_w = int(parts[0])
        res_h = int(parts[1])
    except Exception as e:
        raise SystemExit(f"--resolution 格式不合法：{args.resolution!r}，期望格式为 'W,H'") from e

    # 懒加载 gym_unrealcv 相关依赖，避免离线 dataset 模式强依赖仿真环境。
    try:
        import gym  # type: ignore
        import gym_unrealcv  # noqa: F401  # type: ignore
        from gym_unrealcv.envs.wrappers import configUE, time_dilation  # type: ignore
    except Exception as e:
        raise SystemExit(f"mode=unrealcv 需要 gym 和 gym_unrealcv，但导入失败：{e}")

    env = gym.make(str(args.env_id))
    if int(args.time_dilation) > 0:
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    try:
        env.unwrapped.agents_category = ["drone"]
    except Exception:
        pass
    env = configUE.ConfigUEWrapper(env, resolution=(int(res_w), int(res_h)))
    try:
        env.seed(int(args.seed))
    except Exception:
        pass
    try:
        # 可选：为并行 UE 实例固定 UnrealCV 端口起点。
        if int(args.ue_port) > 0:
            try:
                env.unwrapped.ue_binary.write_port(int(args.ue_port))
            except Exception:
                # 尽力设置；如果失败，launcher 仍会按自己的逻辑自动递增端口。
                pass
        env.reset()
        env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
        env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)
    except Exception:
        pass

    for p in task_paths:
        print(f"[运行] mode=unrealcv task={p} env_id={args.env_id} 分辨率={res_w}x{res_h}")
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
            print(f"[失败] task={p} 错误={e}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
