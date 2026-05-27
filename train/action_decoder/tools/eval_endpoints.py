#!/usr/bin/env python3
"""
评估 UAV-Flow 风格 route 目录中的 latent-to-action 预测结果。

中文导读：
这个脚本是 Stage-2 动作头的离线评估器。它不重新跑模型，只读取
`predict_pose.py` 或兼容脚本写出的 `pred_actions.json` / `pred_path.json`，
和 GT route 日志比较“终点位置误差 + 终点姿态误差”。

阅读主线：
1. `_discover_routes()` 从预测目录里找所有已经生成预测的 route。
2. `_eval_one()` 对单条 route 读取 GT 和预测，并算 endpoint 指标。
3. `integrate_actions_to_traj()` 说明 `pred_actions.json` 的相对动作如何还原成轨迹。
4. `main()` 汇总合格率、分布图和 3D 轨迹叠加图。

期望的 GT route 目录结构：
  路径格式：<gt_root>/<route>/
    preprocessed_logs.json   # 优先使用，(T,6) [x,y,z,roll,yaw,pitch]
    raw_logs.json            # 可选 fallback，布局相同

期望的预测 route 目录结构：
  路径格式：<pred_root>/<route>/
    pred_actions.json        # 优先使用，包含 actions6 = [dz,dy,dx,tx,ty,tz]
    pred_path.json           # fallback，包含 poses = [roll,yaw,pitch,x,y,z]

单位约定：
  - 预测动作始终按 radians + meters 解释。
  - GT 角度默认是 degrees（--angles_in_degrees）。
  - GT 平移会先除以 --translation_divisor 再比较。
    UAV-Flow 数据通常使用 1.0。

输出：
  路径格式：<out_root>/summary.txt
  路径格式：<out_root>/images/*.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_ROOT = os.path.abspath(os.path.join(_TOOLS_DIR, ".."))
_OPEN_ROOT = os.path.abspath(os.path.join(_TRAIN_ROOT, "..", ".."))
_ARCH_ROOT = os.path.join(_OPEN_ROOT, "Worldmodel", "action_decoder", "src")
if _ARCH_ROOT not in sys.path:
    sys.path.insert(0, _ARCH_ROOT)

from datasets.utils import euler_to_rotation, rotation_to_euler  # noqa: E402


def _read_json(path: str):
    """读取 UTF-8 JSON；GT 日志和预测结果都通过这里进入评估流程。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mkdir(path: str) -> None:
    """创建输出目录或图片目录。"""
    os.makedirs(path, exist_ok=True)


def _write_text(path: str, text: str) -> None:
    """写 summary.txt，并自动创建父目录。"""
    _mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _unwrap_angles_rad(rpy: np.ndarray) -> np.ndarray:
    """按 roll/yaw/pitch 三列分别 unwrap，避免角度跨 +/-pi 时 RMSE/终点误差跳变。"""
    out = np.empty_like(rpy, dtype=np.float32)
    for i in range(3):
        out[:, i] = np.unwrap(rpy[:, i])
    return out


def _R_from_rpy_zyx(roll: float, yaw: float, pitch: float) -> np.ndarray:
    """构造 R = Rz(yaw) * Ry(pitch) * Rx(roll)。"""
    return np.asarray(euler_to_rotation(z=yaw, y=pitch, x=roll, isRadian=True, seq="zyx"), dtype=np.float32)


def _rpy_from_R_zyx(R: np.ndarray) -> np.ndarray:
    """把旋转矩阵转回 `[roll,yaw,pitch]`，内部工具返回的是 `[yaw,pitch,roll]`。"""
    zyx = rotation_to_euler(R, seq="zyx")  # 代码/形状说明：[yaw, pitch, roll]
    yaw, pitch, roll = float(zyx[0]), float(zyx[1]), float(zyx[2])
    return np.asarray([roll, yaw, pitch], dtype=np.float32)


def _geodesic_angle_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    计算 SO(3) geodesic 旋转误差，单位 degree。

    初学者公式：先算相对旋转 `R_rel = R_gt^T @ R_pred`，再用
    `theta = acos((trace(R_rel) - 1) / 2) * 180/pi` 得到两个姿态之间的最短旋转角。
    """
    R_rel = (R_gt.T @ R_pred).astype(np.float32)
    c = (float(np.trace(R_rel)) - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    return float(math.acos(c) * 180.0 / math.pi)


def _yaw_error_deg(pred_yaw_rad: float, gt_yaw_rad: float) -> float:
    """
    单独计算 yaw 的环形误差，便于定位航向角偏差。

    公式：`dy = (pred_yaw - gt_yaw)` 转成 degree 后绕 360 取模，
    最终误差取 `min(|dy|, 360 - |dy|)`，所以 359 度和 1 度只差 2 度。
    """
    dy = (float(pred_yaw_rad - gt_yaw_rad) * 180.0 / math.pi) % 360.0
    if dy > 180.0:
        dy = 360.0 - dy
    return float(abs(dy))


def integrate_actions_to_traj(
    actions6: np.ndarray,
    *,
    start_xyz_m: np.ndarray,
    start_rpy_rad: np.ndarray,
) -> np.ndarray:
    """
    actions6: (T-1,6)，每行是 [dz,dy,dx,tx,ty,tz]。
      - dz/dy/dx 是上一帧坐标系下的相对 ZYX Euler 旋转，单位 radians。
      - tx/ty/tz 是上一帧坐标系下的平移，单位 meters。
    返回：(T,6) 绝对轨迹 [roll,yaw,pitch,x,y,z]，单位 radians + meters。

    中文说明：
    这是评估 `pred_actions.json` 时最重要的函数。动作不是世界坐标系下的绝对位姿，
    而是“上一帧坐标系中的相对旋转和平移”。因此每一步要用上一帧旋转矩阵 R 把平移
    转到世界坐标，再把相对旋转右乘到当前姿态上。

    初学者公式：
    - `p_i = p_{i-1} + R_{i-1} @ [tx,ty,tz]`；
    - `R_i = R_{i-1} @ R_rel`，其中 `R_rel = Rz(dz) @ Ry(dy) @ Rx(dx)`。
    """
    actions6 = np.asarray(actions6, dtype=np.float32)
    T = int(actions6.shape[0]) + 1
    traj = np.zeros((T, 6), dtype=np.float32)

    roll0, yaw0, pitch0 = [float(x) for x in start_rpy_rad.reshape(3)]
    R = _R_from_rpy_zyx(roll0, yaw0, pitch0)
    p = start_xyz_m.astype(np.float32).reshape(3).copy()
    traj[0, 0:3] = np.asarray([roll0, yaw0, pitch0], dtype=np.float32)
    traj[0, 3:6] = p

    for i in range(1, T):
        dz, dy, dx = [float(x) for x in actions6[i - 1, 0:3]]
        t_rel = actions6[i - 1, 3:6].astype(np.float32)
        R_rel = _R_from_rpy_zyx(roll=dx, yaw=dz, pitch=dy)
        p = p + (R @ t_rel)
        R = (R @ R_rel).astype(np.float32)
        traj[i, 0:3] = _rpy_from_R_zyx(R)
        traj[i, 3:6] = p

    traj[:, 0:3] = _unwrap_angles_rad(traj[:, 0:3])
    return traj


def _load_gt_traj(gt_dir: str, *, pose_file: str, translation_divisor: float, angles_in_degrees: bool) -> np.ndarray:
    """
    读取 GT 轨迹并统一成 `[roll,yaw,pitch,x,y,z]`、rad + meter。

    UAV-Flow 原始/预处理日志常见布局是 `[x,y,z,roll,yaw,pitch]`，角度默认是 degree；
    这里把单位和列顺序转成评估内部格式，避免后面重复做转换。
    """
    candidates = [os.path.join(gt_dir, pose_file)]
    if pose_file != "preprocessed_logs.json":
        candidates.append(os.path.join(gt_dir, "preprocessed_logs.json"))
    candidates.append(os.path.join(gt_dir, "raw_logs.json"))

    path = ""
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if not path:
        raise FileNotFoundError(f"在 {gt_dir} 下找不到 GT pose json；已尝试 {candidates}")

    arr = np.asarray(_read_json(path), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"GT 轨迹形状不合法：{arr.shape}，文件={path}")
    xyz_m = arr[:, 0:3].astype(np.float32) / float(translation_divisor)
    rpy = arr[:, 3:6].astype(np.float32)
    if bool(angles_in_degrees):
        rpy = rpy * (np.pi / 180.0)
    rpy = _unwrap_angles_rad(rpy)
    return np.concatenate([rpy, xyz_m], axis=1).astype(np.float32)  # [rpy,xyz]


def _load_pred_actions(path: str) -> np.ndarray:
    """读取相对动作预测。优先格式是 `actions6=[dz,dy,dx,tx,ty,tz]`。"""
    obj = _read_json(path)
    acts = obj.get("actions6")
    if not isinstance(acts, list):
        raise ValueError("pred_actions.json 缺少 actions6 字段")
    a = np.asarray(acts, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] < 6:
        raise ValueError(f"actions6 形状不合法：{a.shape}，文件={path}")
    return a[:, :6]


def _load_pred_path(path: str) -> np.ndarray:
    """读取已经积分好的预测轨迹；在没有 pred_actions.json 时作为兜底输入。"""
    obj = _read_json(path)
    poses = obj.get("poses")
    if not isinstance(poses, list):
        raise ValueError("pred_path.json 缺少 poses 字段")
    p = np.asarray(poses, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] < 6:
        raise ValueError(f"poses 形状不合法：{p.shape}，文件={path}")
    return p[:, :6].astype(np.float32)  # 代码/形状说明：[roll,yaw,pitch,x,y,z], rad+m


def _discover_routes(pred_root: str) -> List[str]:
    """递归发现含预测 JSON 的 route 目录，返回相对 pred_root 的 route 名。"""
    routes: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(pred_root):
        if "pred_actions.json" in filenames or "pred_path.json" in filenames:
            rel = os.path.relpath(dirpath, pred_root)
            if rel != ".":
                routes.append(rel.replace(os.sep, "/"))
    routes.sort()
    return routes


@dataclass
class RouteEval:
    """单条 route 的端点评估结果。qualified 表示同时通过位置和姿态阈值。"""
    route: str
    T_use: int
    dist_m: float
    dist_cm: float
    ang_deg: float
    yaw_err_deg: float
    qualified: bool


def _eval_one(
    route: str,
    pred_dir: str,
    gt_dir: str,
    *,
    pose_file: str,
    translation_divisor: float,
    angles_in_degrees: bool,
    dist_thr_m: float,
    ang_thr_deg: float,
) -> RouteEval:
    """
    评估单条 route。

    如果有 `pred_actions.json`，先用 GT 第 0 帧作为起点积分相对动作；
    如果只有 `pred_path.json`，则直接比较其中的绝对轨迹。最终指标只看最后一帧：
    - `dist_m`: 终点欧氏距离；
    - `ang_deg`: SO(3) geodesic 姿态误差；
    - `yaw_err_deg`: 单独的 yaw 误差，方便排查航向偏差。
    """
    gt_traj = _load_gt_traj(
        gt_dir,
        pose_file=pose_file,
        translation_divisor=translation_divisor,
        angles_in_degrees=angles_in_degrees,
    )
    gt_rpy = gt_traj[:, 0:3].astype(np.float32)
    gt_xyz = gt_traj[:, 3:6].astype(np.float32)

    pred_actions_json = os.path.join(pred_dir, "pred_actions.json")
    pred_path_json = os.path.join(pred_dir, "pred_path.json")
    if os.path.exists(pred_actions_json):
        actions6 = _load_pred_actions(pred_actions_json)
        T_use = min(int(gt_traj.shape[0]), int(actions6.shape[0]) + 1)
        pred_traj = integrate_actions_to_traj(
            actions6[: max(0, T_use - 1)],
            start_xyz_m=gt_xyz[0],
            start_rpy_rad=gt_rpy[0],
        )
    elif os.path.exists(pred_path_json):
        pred_traj = _load_pred_path(pred_path_json)
        T_use = min(int(gt_traj.shape[0]), int(pred_traj.shape[0]))
        pred_traj = pred_traj[:T_use]
    else:
        raise FileNotFoundError(f"在 {pred_dir} 下既找不到 pred_actions.json，也找不到 pred_path.json")

    gt_rpy = gt_rpy[:T_use]
    gt_xyz = gt_xyz[:T_use]
    pred_rpy = pred_traj[:T_use, 0:3].astype(np.float32)
    pred_xyz = pred_traj[:T_use, 3:6].astype(np.float32)

    # 终点平移误差：只看最后一帧，dist_m = ||pred_xyz[-1] - gt_xyz[-1]||_2。
    dp = (pred_xyz[-1] - gt_xyz[-1]).astype(np.float32)
    dist_m = float(np.linalg.norm(dp))
    R_pred = _R_from_rpy_zyx(float(pred_rpy[-1, 0]), float(pred_rpy[-1, 1]), float(pred_rpy[-1, 2]))
    R_gt = _R_from_rpy_zyx(float(gt_rpy[-1, 0]), float(gt_rpy[-1, 1]), float(gt_rpy[-1, 2]))
    # 终点姿态误差：ang_deg 是 SO(3) 最短旋转角；yaw_err_deg 只看航向角的环形差值。
    ang_deg = _geodesic_angle_deg(R_pred, R_gt)
    yaw_err_deg = _yaw_error_deg(float(pred_rpy[-1, 1]), float(gt_rpy[-1, 1]))

    qualified = dist_m <= float(dist_thr_m) and ang_deg <= float(ang_thr_deg)
    return RouteEval(
        route=route,
        T_use=int(T_use),
        dist_m=dist_m,
        dist_cm=dist_m * 100.0,
        ang_deg=ang_deg,
        yaw_err_deg=yaw_err_deg,
        qualified=bool(qualified),
    )


def _plot_distribution(vals: List[float], *, title: str, xlabel: str, out_path: str, bins: int = 50) -> None:
    """画单个指标的直方图，输出到 images/，用于快速判断误差分布长尾。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = np.asarray(vals, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    _mkdir(os.path.dirname(out_path))
    plt.figure(figsize=(10, 6))
    plt.hist(v, bins=int(bins), color="skyblue", edgecolor="black", alpha=0.9)
    plt.axvline(float(np.mean(v)), color="red", linestyle="--", linewidth=1.5, label=f"均值: {np.mean(v):.2f}")
    plt.axvline(float(np.median(v)), color="green", linestyle="--", linewidth=1.5, label=f"中位数: {np.median(v):.2f}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("数量")
    plt.grid(True, alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_overlay(routes: List[str], *, pred_root: str, gt_root: str, pose_file: str, translation_divisor: float, angles_in_degrees: bool, out_path: str, max_routes: int) -> None:
    """把多条 GT/预测 3D 轨迹叠加到一张图里，主要用于检查整体尺度和方向是否错位。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _mkdir(os.path.dirname(out_path))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    n = 0
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for r in routes[: int(max_routes)]:
        try:
            gt = _load_gt_traj(
                os.path.join(gt_root, r),
                pose_file=pose_file,
                translation_divisor=translation_divisor,
                angles_in_degrees=angles_in_degrees,
            )
            pred_path = os.path.join(pred_root, r, "pred_path.json")
            if not os.path.exists(pred_path):
                continue
            pred = _load_pred_path(pred_path)
        except Exception:
            continue
        gt_xyz = gt[:, 3:6].astype(np.float32)
        pred_xyz = pred[:, 3:6].astype(np.float32)
        T = min(int(gt_xyz.shape[0]), int(pred_xyz.shape[0]))
        if T <= 1:
            continue
        gt_xyz = gt_xyz[:T]
        pred_xyz = pred_xyz[:T]
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color="#1f77b4", alpha=0.15, linewidth=1.0)
        ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], color="#d62728", alpha=0.15, linewidth=1.0)
        mins = np.minimum(mins, np.min(gt_xyz, axis=0))
        mins = np.minimum(mins, np.min(pred_xyz, axis=0))
        maxs = np.maximum(maxs, np.max(gt_xyz, axis=0))
        maxs = np.maximum(maxs, np.max(pred_xyz, axis=0))
        n += 1

    ax.set_title(f"UAV-Flow 轨迹叠加图（GT 蓝 / Pred 红），n={n}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    if np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs)) and np.all(maxs > mins):
        center = (mins + maxs) / 2.0
        span = float(np.max(maxs - mins))
        lo = center - span / 2.0
        hi = center + span / 2.0
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    """CLI 入口：发现预测 route、逐条评估、写 summary 和误差分布图。"""
    ap = argparse.ArgumentParser(
        description=(
            "评估 UAV-Flow 风格 route 目录中的 latent-to-action 预测结果。"
            "优先读取 pred_actions.json 并从 GT 第 0 帧重新积分轨迹，最后统计终点位置/姿态误差。"
        )
    )
    ap.add_argument("--pred_root", type=str, default="./outputs/stage2_latent2action", help="预测结果根目录；其下每个 route 子目录通常包含 pred_actions.json 或 pred_path.json。")
    ap.add_argument("--gt_root", type=str, default="./data/uavflow", help="真值（GT）route 根目录。")
    ap.add_argument("--out_root", type=str, default="./outputs/eval_uavflow", help="评估输出目录，会写 summary.txt 和 images/*.png。")
    ap.add_argument("--gt_pose_file", type=str, default="preprocessed_logs.json", help="每条真值（GT）route 内优先查找的 pose json 文件名。")
    ap.add_argument("--translation_divisor", type=float, default=1.0, help="比较前先把 GT xyz 除以该值；用于兼容不同数据集的平移单位。")
    ap.add_argument("--angles_in_degrees", action="store_true", default=True, help="真值（GT）日志中的 roll/yaw/pitch 是否按 degree 解释。")
    ap.add_argument("--dist_thr_m", type=float, default=3.0, help="端点位置误差合格阈值，单位米。")
    ap.add_argument("--ang_thr_deg", type=float, default=10.0, help="端点姿态 geodesic 误差合格阈值，单位度。")
    ap.add_argument("--expect_routes", type=int, default=0, help="期望评估的 route 数量；仅用于 summary 提示，0 表示不写这个提示。")
    ap.add_argument("--overlay_max_routes", type=int, default=300, help="3D 轨迹叠加图里最多绘制多少条 route。")
    ap.add_argument("--tqdm", action="store_true", default=True, help="若本机安装了 tqdm，则显示评估进度条。")
    args = ap.parse_args()

    # 评估根目录全部转绝对路径，summary 里记录完整位置，方便复现实验。
    pred_root = os.path.abspath(str(args.pred_root))
    gt_root = os.path.abspath(str(args.gt_root))
    out_root = os.path.abspath(str(args.out_root))
    img_root = os.path.join(out_root, "images")
    _mkdir(img_root)

    routes = _discover_routes(pred_root)
    results: List[RouteEval] = []
    skipped: List[Tuple[str, str]] = []
    # 单条 route 失败只记 skipped，不中断整批评估；这样坏样本不会掩盖其他路线的统计。
    iterator = tqdm(routes, desc="评估 routes", dynamic_ncols=True) if bool(args.tqdm) and tqdm is not None else routes
    for r in iterator:
        try:
            results.append(
                _eval_one(
                    r,
                    os.path.join(pred_root, r),
                    os.path.join(gt_root, r),
                    pose_file=str(args.gt_pose_file),
                    translation_divisor=float(args.translation_divisor),
                    angles_in_degrees=bool(args.angles_in_degrees),
                    dist_thr_m=float(args.dist_thr_m),
                    ang_thr_deg=float(args.ang_thr_deg),
                )
            )
        except Exception as e:
            skipped.append((r, str(e)))

    ok = len(results)
    qual = sum(1 for x in results if x.qualified)
    qual_rate = float(qual) / float(max(1, ok))
    dist_cm = [x.dist_cm for x in results]
    ang_deg = [x.ang_deg for x in results]
    yaw_err_deg = [x.yaw_err_deg for x in results]
    results_sorted = sorted(results, key=lambda x: (x.dist_m, x.ang_deg))

    # summary.txt 是主要评估产物：先给全局统计，再按误差从小到大列出每条 route。
    expected = f"期望约 {int(args.expect_routes)} 条" if int(args.expect_routes) > 0 else "未指定期望 route 数"
    lines: List[str] = []
    lines.append("数据集=uav-flow")
    lines.append(f"预测目录={pred_root}")
    lines.append(f"GT目录={gt_root}")
    lines.append(f"GT pose 文件名={args.gt_pose_file}")
    lines.append(f"translation_divisor={float(args.translation_divisor)}")
    lines.append(f"成功评估 route 数={ok}（{expected}），跳过={len(skipped)}")
    lines.append(f"合格率（dist<={args.dist_thr_m}m 且 ang<={args.ang_thr_deg}deg）= {qual}/{ok} = {qual_rate*100.0:.2f}%")
    if ok > 0:
        lines.append(f"终点距离误差(cm): mean={np.mean(dist_cm):.2f} p50={np.percentile(dist_cm,50):.2f} p90={np.percentile(dist_cm,90):.2f} max={np.max(dist_cm):.2f}")
        lines.append(f"终点姿态误差(deg): mean={np.mean(ang_deg):.2f} p50={np.percentile(ang_deg,50):.2f} p90={np.percentile(ang_deg,90):.2f} max={np.max(ang_deg):.2f}")
        lines.append(f"终点航向误差(deg): mean={np.mean(yaw_err_deg):.2f} p50={np.percentile(yaw_err_deg,50):.2f} p90={np.percentile(yaw_err_deg,90):.2f} max={np.max(yaw_err_deg):.2f}")
    lines.append("")
    lines.append("逐 route 端点误差（先按距离、再按姿态误差排序）：")
    lines.append("route\tT_use\tdist_cm\tang_deg(geo)\tyaw_err_deg\tqualified")
    for x in results_sorted:
        lines.append(f"{x.route}\t{x.T_use}\t{x.dist_cm:.2f}\t{x.ang_deg:.2f}\t{x.yaw_err_deg:.2f}\t{int(x.qualified)}")
    if skipped:
        lines.append("")
        lines.append("跳过的 routes：")
        for r, e in skipped[:100]:
            lines.append(f"{r}\t{e}")
        if len(skipped) > 100:
            lines.append(f"...（其余 {len(skipped) - 100} 条省略）")
    _write_text(os.path.join(out_root, "summary.txt"), "\n".join(lines) + "\n")

    if ok > 0:
        _plot_distribution(dist_cm, title="终点距离误差分布", xlabel="距离误差 (cm)", out_path=os.path.join(img_root, "distance_error_distribution.png"))
        _plot_distribution(ang_deg, title="终点姿态误差分布", xlabel="姿态误差 (deg)", out_path=os.path.join(img_root, "rotation_error_distribution.png"))
        _plot_distribution(yaw_err_deg, title="终点航向误差分布", xlabel="航向误差 (deg)", out_path=os.path.join(img_root, "yaw_error_distribution.png"))
        _plot_overlay(
            routes,
            pred_root=pred_root,
            gt_root=gt_root,
            pose_file=str(args.gt_pose_file),
            translation_divisor=float(args.translation_divisor),
            angles_in_degrees=bool(args.angles_in_degrees),
            out_path=os.path.join(img_root, "trajectories_3d_overlay.png"),
            max_routes=int(args.overlay_max_routes),
        )

    print(f"评估完成。summary={os.path.join(out_root, 'summary.txt')} images_dir={img_root}")


if __name__ == "__main__":
    main()
