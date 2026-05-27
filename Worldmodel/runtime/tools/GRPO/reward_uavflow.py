#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WorldVLN Action-aware GRPO 的奖励构建脚本。

中文导读：
这个脚本把在线 rollout 产生的 `trajectory.json` 转成 GRPO replay 需要的奖励字段。
它不是只看动作是否像专家，而是同时记录三类信号：
- 动作/轨迹奖励（action/trajectory reward）：预测动作轨迹和专家轨迹的几何一致性；
- 任务奖励（task reward）：当前 clip 或整条轨迹终点是否更接近任务目标；
- 参考/CE 奖励（reference/CE reward）：用旧策略或参考分布的 logprob 约束更新幅度，避免偏离监督阶段学到的世界模型先验。

读代码时从 `main()` 的 clip 模式开始看，最终组合点在 `grpo_reward =
公式/形状说明：lambda_act * r_act + lambda_task * r_task + lambda_ce * r_ce`。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List

import numpy as np


def _yaw_wrap_deg(d: np.ndarray) -> np.ndarray:
    """把 yaw 差值 wrap 到 (-180, 180]，避免跨 360 度时误差虚高。"""
    return (d + 180.0) % 360.0 - 180.0


def _angles_wrap_deg(d: np.ndarray) -> np.ndarray:
    """把角度归一到 (-180, 180]。"""
    return (d + 180.0) % 360.0 - 180.0


def _load_gt_poses(path: str) -> List[List[float]]:
    """读取 GT pose 日志，兼容 list pose 和 dict commanded/observed 两种格式。"""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        return [[float(v) for v in row[:6]] for row in obj]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        out: List[List[float]] = []
        for row in obj:
            c = row.get("commanded", row.get("observed", {})) if isinstance(row, dict) else {}
            out.append(
                [
                    float(c.get("x", 0.0)),
                    float(c.get("y", 0.0)),
                    float(c.get("z", 0.0)),
                    float(c.get("roll", 0.0)),
                    float(c.get("yaw", 0.0)),
                    float(c.get("pitch", 0.0)),
                ]
            )
        return out
    return []


def _to_relative_cmdeg(poses: List[List[float]]) -> List[List[float]]:
    """
    以第一帧 pose 为锚点，把 pose 列表转成相对轨迹（cm/deg）。
    这样即使输入 json 存的是绝对/world pose，成功阈值检查也更稳。
    """
    if not poses:
        return poses
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        return poses
    base = arr[0:1, :6].copy()
    out = arr[:, :6].copy()
    out[:, 0:3] = out[:, 0:3] - base[:, 0:3]
    # 角度也转成相对差值，并处理跨 360 度的 wrap。
    out[:, 3:6] = _angles_wrap_deg(out[:, 3:6] - base[:, 3:6])
    return [[float(v) for v in row.tolist()] for row in out]


def _to_m_rad(arr_deg_cm: np.ndarray) -> np.ndarray:
    """
        输入/输出布局：[x, y, z, roll, yaw, pitch]
        列表项说明：- xyz: cm -> m
        列表项说明：- ryp: deg -> rad

    """
    out = arr_deg_cm.astype(np.float64).copy()
    out[:, 0:3] = out[:, 0:3] / 100.0
    out[:, 3:6] = out[:, 3:6] * (math.pi / 180.0)
    return out


def _yaw_wrap_rad(d: np.ndarray) -> np.ndarray:
    """把 radian yaw 差值 wrap 到 (-pi, pi]。"""
    return np.arctan2(np.sin(d), np.cos(d))


def _clip_mse(pred_poses: List[List[float]], gt_poses: List[List[float]]) -> Dict[str, float]:
    """计算一个 clip 内预测轨迹和专家轨迹的 MSE 奖励诊断项。"""
    n = min(len(pred_poses), len(gt_poses))
    if n <= 0:
        return {
            "mse_all6_mrad": 0.0,
            "mse_xyz_m2": 0.0,
            "mse_yaw_rad2": 0.0,
            "mse_xyz_cm2": 0.0,
            "mse_yaw_deg2": 0.0,
        }
    p_cmdeg = np.asarray(pred_poses[:n], dtype=np.float64)
    g_cmdeg = np.asarray(gt_poses[:n], dtype=np.float64)
    p = _to_m_rad(p_cmdeg)
    g = _to_m_rad(g_cmdeg)
    yaw_diff_rad = _yaw_wrap_rad(p[:, 4] - g[:, 4])
    mse_all6_mrad = float(np.mean((p - g) ** 2))
    mse_xyz_m2 = float(np.mean((p[:, :3] - g[:, :3]) ** 2))
    mse_yaw_rad2 = float(np.mean(yaw_diff_rad**2))
    # 同时保留 cm/deg 诊断值，方便人工检查。
    yaw_diff_deg = _yaw_wrap_deg(p_cmdeg[:, 4] - g_cmdeg[:, 4])
    mse_xyz_cm2 = float(np.mean((p_cmdeg[:, :3] - g_cmdeg[:, :3]) ** 2))
    mse_yaw_deg2 = float(np.mean(yaw_diff_deg**2))
    return {
        "mse_all6_mrad": mse_all6_mrad,
        "mse_xyz_m2": mse_xyz_m2,
        "mse_yaw_rad2": mse_yaw_rad2,
        "mse_xyz_cm2": mse_xyz_cm2,
        "mse_yaw_deg2": mse_yaw_deg2,
    }


def _reward_from_mse(mse: Dict[str, float], alpha_xyz: float, alpha_yaw: float, alpha_all6: float) -> float:
    """
    使用单位统一到 m/rad 的 loss，再用倒数映射，避免 reward 下溢塌缩。
    val >= 0；reward 位于 (0, 1]。
    """
    val = (
        alpha_xyz * mse["mse_xyz_m2"]
        + alpha_yaw * mse["mse_yaw_rad2"]
        + alpha_all6 * mse["mse_all6_mrad"]
    )
    return float(1.0 / (1.0 + max(0.0, val)))


def _act_mse_scalar(mse: Dict[str, float], alpha_xyz: float, alpha_yaw: float, alpha_all6: float) -> float:
    """用于组内相对动作奖励的标量 loss 指标（>=0，越低越好）。"""
    return float(
        alpha_xyz * float(mse.get("mse_xyz_m2", 0.0))
        + alpha_yaw * float(mse.get("mse_yaw_rad2", 0.0))
        + alpha_all6 * float(mse.get("mse_all6_mrad", 0.0))
    )


def _zscore_exp_reward(
    xs: np.ndarray,
    *,
    eps: float,
    zmax: float,
) -> Dict[str, np.ndarray]:
    """
        GRPO 风格的组内相对塑形：
        列表项说明：- z = (x - mean) / max(std, eps)
        列表项说明：- z_tilde = clip(max(0, z), 0, zmax)
        - r = exp(-z_tilde)，位于 (0, 1]
        x 越低越好；x 低于均值时 z<0，因此 r=1。

    """
    xs = xs.astype(np.float64)
    mu = float(np.mean(xs)) if xs.size > 0 else 0.0
    sig = float(np.sqrt(np.mean((xs - mu) ** 2))) if xs.size > 0 else 0.0
    denom = max(float(sig), float(eps))
    z = (xs - mu) / denom
    z_tilde = np.maximum(0.0, z)
    if zmax and zmax > 0:
        z_tilde = np.minimum(z_tilde, float(zmax))
    r = np.exp(-z_tilde)
    return {
        "mu": np.asarray([mu], dtype=np.float64),
        "sigma": np.asarray([sig], dtype=np.float64),
        "z": z,
        "z_tilde": z_tilde,
        "r": r,
    }


def _minstd_exp_reward(
    xs: np.ndarray,
    *,
    eps: float,
    zmax: float,
) -> Dict[str, np.ndarray]:
    """
        另一种组内相对塑形（非负，经过 exp 映射后 <=1）：
        列表项说明：- xmin = min(x)
        列表项说明：- denom = max(std(x), eps)
        列表项说明：- xminus = (x - xmin) / denom   (>=0)
        列表项说明：- xminus_tilde = clip(xminus, 0, zmax)
        - r = exp(-xminus_tilde)，位于 (0, 1]

        说明：
        - x 越低越好；组内最好的样本得到 r=1。
        - xminus 天然非负，因此不再需要 max(0, z) 截断。

    """
    xs = xs.astype(np.float64)
    if xs.size <= 0:
        xmin = 0.0
        mu = 0.0
        sig = 0.0
        denom = max(float(eps), 1.0)
        xminus = xs
    else:
        xmin = float(np.min(xs))
        mu = float(np.mean(xs))
        sig = float(np.sqrt(np.mean((xs - mu) ** 2)))
        denom = max(float(sig), float(eps))
        xminus = (xs - xmin) / denom
    xminus_tilde = np.maximum(0.0, xminus)
    if zmax and zmax > 0:
        xminus_tilde = np.minimum(xminus_tilde, float(zmax))
    r = np.exp(-xminus_tilde)
    return {
        "xmin": np.asarray([xmin], dtype=np.float64),
        "mu": np.asarray([mu], dtype=np.float64),
        "sigma": np.asarray([sig], dtype=np.float64),
        "xminus": xminus,
        "xminus_tilde": xminus_tilde,
        "r": r,
    }


def _loo_adv(r: np.ndarray) -> np.ndarray:
    """
        留一法优势值（LOO）。

        公式给初学者看就是：
          公式/形状说明：adv_i = r_i - mean(r_j, j != i)
        也就是“当前候选奖励 - 同组其他候选的平均奖励”。

    """
    r = r.astype(np.float64)
    k = int(r.size)
    if k <= 1:
        return np.zeros_like(r, dtype=np.float64)
    ssum = float(np.sum(r))
    out = np.zeros_like(r, dtype=np.float64)
    for i in range(k):
        others_mean = (ssum - float(r[i])) / max(1.0, float(k - 1))
        out[i] = float(r[i]) - others_mean
    return out


def _rank01_average_ties(vals: np.ndarray) -> np.ndarray:
    """
    把同组候选 reward 转成 [0,1] rank，平分 ties。

    公式：rank01_i = average_rank_i / max(1, K-1)。
    K 是同组候选数；分数越大，rank01 越接近 1。
    """
    vals = vals.astype(np.float64)
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


def _centered_rank_adv(vals: np.ndarray) -> np.ndarray:
    """把 [0,1] rank 转成 [-1,1]：adv_rank_i = 2 * (rank01_i - 0.5)。"""
    rank01 = _rank01_average_ties(vals)
    return (rank01 - 0.5) * 2.0


def _print_final_weight_summary(rows: List[Dict[str, Any]], out_path: str) -> None:
    """打印最终 advantage 分布，快速检查成功样本是否被错误赋负权。"""
    total = len(rows)
    if total <= 0:
        print(f"[reward_uavflow][summary] 输出为空：{out_path}")
        return
    adv = np.asarray([float(r.get("grpo_adv_final", 0.0)) for r in rows], dtype=np.float64)
    success_raw = np.asarray([float(r.get("grpo_reward_task_success_raw", 0.0)) for r in rows], dtype=np.float64)
    task_raw = np.asarray([float(r.get("grpo_reward_task_raw", 0.0)) for r in rows], dtype=np.float64)
    pos_frac = float(np.mean(adv > 0.0))
    zero_frac = float(np.mean(adv == 0.0))
    neg_frac = float(np.mean(adv < 0.0))
    success_negative = int(np.sum((success_raw > 0.0) & (adv < 0.0)))
    high_task_negative = int(np.sum((task_raw >= 0.8) & (adv < 0.0)))
    print(
        "[reward_uavflow][summary] "
        f"行数 rows={total} 正权重比例 pos_frac={pos_frac:.6f} "
        f"零权重比例 zero_frac={zero_frac:.6f} 负权重比例 neg_frac={neg_frac:.6f} "
        f"成功样本负权重数 success_negative={success_negative} "
        f"高 task 分负权重数 high_task_negative={high_task_negative} 输出 out={out_path}"
    )


def _task_success_from_final(
    pred_poses: List[List[float]],
    gt_poses: List[List[float]],
    pos_thresh_m: float,
    yaw_thresh_deg: float,
) -> float:
    """按最终位置误差和 yaw 误差阈值判断整条轨迹是否成功。"""
    if not pred_poses or not gt_poses:
        return 0.0
    p = np.asarray(pred_poses[-1], dtype=np.float64)  # cm/deg
    g = np.asarray(gt_poses[-1], dtype=np.float64)    # cm/deg
    pos_err_m = float(np.linalg.norm((p[:3] - g[:3]) / 100.0))
    yaw_err_deg = float(abs(((p[4] - g[4] + 180.0) % 360.0) - 180.0))
    return 1.0 if (pos_err_m <= float(pos_thresh_m) and yaw_err_deg <= float(yaw_thresh_deg)) else 0.0


def _dense_task_reward_from_final(
    pred_poses: List[List[float]],
    gt_poses: List[List[float]],
    pos_scale_m: float,
    yaw_scale_deg: float,
    pos_weight: float,
    yaw_weight: float,
) -> Dict[str, float]:
    """
        与最终 xyz + yaw 评估对齐的密集终点任务奖励：
          公式/形状说明：cost = w_pos * (pos_err_m / pos_scale_m)^2 + w_yaw * (yaw_err_deg / yaw_scale_deg)^2
          公式/形状说明：reward = 1 / (1 + cost)

        输入/输出单位说明：
        - `pred_poses` / `gt_poses` 里的 pose 仍按本项目轨迹约定使用 `cm/deg`；
        - 计算 reward 时，位置误差先转成 `m`，yaw 误差保留 `deg`；
        - 返回字典里的 `pos_err_m`、`yaw_err_deg` 是便于人工检查的诊断量。

    """
    if not pred_poses or not gt_poses:
        return {
            "pos_err_m": 0.0,
            "yaw_err_deg": 180.0,
            "cost": float(max(0.0, pos_weight) * 1e6 + max(0.0, yaw_weight) * 1e6),
            "reward": 0.0,
        }
    p = np.asarray(pred_poses[-1], dtype=np.float64)  # cm/deg
    g = np.asarray(gt_poses[-1], dtype=np.float64)  # cm/deg
    pos_err_m = float(np.linalg.norm((p[:3] - g[:3]) / 100.0))
    yaw_err_deg = float(abs(((p[4] - g[4] + 180.0) % 360.0) - 180.0))
    pos_scale = max(float(pos_scale_m), 1e-6)
    yaw_scale = max(float(yaw_scale_deg), 1e-6)
    cost = float(
        max(0.0, float(pos_weight)) * (pos_err_m / pos_scale) ** 2
        + max(0.0, float(yaw_weight)) * (yaw_err_deg / yaw_scale) ** 2
    )
    return {
        "pos_err_m": pos_err_m,
        "yaw_err_deg": yaw_err_deg,
        "cost": cost,
        "reward": float(1.0 / (1.0 + max(0.0, cost))),
    }


def _compose_task_reward_raw(
    task_mode: str,
    dense_raw: float,
    success_raw: float,
    enable_success_bonus: int,
    task_dense_weight: float,
    task_success_weight: float,
) -> float:
    """
    合成密集终点进度奖励和可选成功加分（success bonus）。

    中文导读：
    密集奖励让模型即使没完全成功也能获得“离目标更近”的信号；成功加分保留
    明确阈值判断。GRPO 里二者合成后还会在组内做 LOO/排序塑形。

    模式区别：
    - `raw_dense`：使用密集终点奖励，并按配置决定是否叠加成功加分；
    - `raw_succ`：只看是否成功，适合把目标严格离散化的实验；
    - `loo` / `dense_loo`：这个函数先产出“原始 task reward”，稍后在组内再做 LOO。
    """
    mode = str(task_mode or "raw_dense").strip().lower()
    if mode == "raw_succ":
        return float(success_raw)
    if int(enable_success_bonus) != 1:
        return float(dense_raw)
    return float(
        max(0.0, float(task_dense_weight)) * float(dense_raw)
        + max(0.0, float(task_success_weight)) * float(success_raw)
    )


def _split_into_clips(poses: List[List[float]], clip_len: int, num_clips: int) -> List[List[List[float]]]:
    """
    把一条完整 49 帧轨迹的相对 pose 拆成多个 clip 片段。

    典型布局：
    - 49 帧 => 48 个相对 delta/pose，对齐到帧 1..48
    - 3 个 clip => 每个 16 个 pose：[0:16], [16:32], [32:48]

    注意不要把“pose 序列索引”和“帧窗口索引”混为一谈：
    - pose 序列这里按动作步来切，例如前 16 个 pose；
    - 真正喂给 world model 的观测窗口通常是 17 帧，对应 `[0..16]` 这样的 4n+1 帧边界。
    """
    out: List[List[List[float]]] = []
    L = int(max(0, len(poses)))
    clip_len_i = int(max(1, clip_len))
    n_i = int(max(1, num_clips))
    for ci in range(n_i):
        st = ci * clip_len_i
        ed = min(L, st + clip_len_i)
        out.append(poses[st:ed] if (st < ed) else [])
    return out


def _reward_act_with_clip_decay(
    pred_poses: List[List[float]],
    gt_poses: List[List[float]],
    clip_len: int,
    num_clips: int,
    clip_alpha: float,
    alpha_xyz: float,
    alpha_yaw: float,
    alpha_all6: float,
) -> Dict[str, Any]:
    """
        对每个 clip 的 reward 做时间衰减后求和，得到动作级奖励：
          公式/形状说明：r_act = r0 * 1 + r1 * alpha + r2 * alpha^2

        设计直觉：
        越靠前的 clip 越接近“这段观测一开始模型就该决定什么动作”，因此默认给更高权重；
        越靠后的 clip 更像长时滚动后的结果，噪声和误差累积更大，所以用 `clip_alpha`
        逐段衰减，避免尾段完全主导训练信号。

    """
    pred_clips = _split_into_clips(pred_poses, clip_len=clip_len, num_clips=num_clips)
    gt_clips = _split_into_clips(gt_poses, clip_len=clip_len, num_clips=num_clips)
    r_list: List[float] = []
    mse_list: List[Dict[str, float]] = []
    for ci in range(int(num_clips)):
        mse = _clip_mse(pred_clips[ci], gt_clips[ci])
        r = _reward_from_mse(mse=mse, alpha_xyz=alpha_xyz, alpha_yaw=alpha_yaw, alpha_all6=alpha_all6)
        r_list.append(float(r))
        mse_list.append(mse)
    clip_alpha_f = float(clip_alpha)
    w_list = [float(clip_alpha_f**ci) for ci in range(int(num_clips))]
    r_total = float(sum(r_list[ci] * w_list[ci] for ci in range(int(num_clips))))
    return {
        "reward_act": r_total,
        "reward_act_clips": r_list,
        "reward_act_weights": w_list,
        "mse_clips": mse_list,
    }


def main():
    """
    把 rollout 轨迹转成带奖励字段的 replay row。

    中文导读：
    Stage A rollout 会先生成候选轨迹和 logprob/trace 信息；这个脚本把每条 rollout
    变成 Stage B 可训练的 replay metadata。推荐阅读 `output_mode="clip"` 分支，
    因为它把一条 49 帧轨迹拆成 3 个 16 动作片段，更贴近在线闭环的 segment 协议。

    输入 replay row 的最小要求：
    - 需要能定位到 `trajectory.json`：通常依赖 `traj_id` 或 `id`；
    - 需要 `gt_pose_json` 指向专家/真值轨迹；
    - 若要启用 CE/reference 奖励，`trajectory.json` 里还需要旧策略 logprob。
    """
    ap = argparse.ArgumentParser(
        description=(
            "把 rollout 轨迹整理成 Action-aware GRPO 所需的 replay metadata。"
            "初学者建议优先阅读 clip 模式，因为它最贴近 49 帧 -> 3 段 16 动作的训练协议。"
        )
    )
    ap.add_argument("--replay_jsonl", type=str, required=True, help="输入 replay jsonl 路径；每行对应一个 rollout 候选的基础元数据。")
    ap.add_argument("--trajectory_json_dir", type=str, required=True, help="存放各条 `trajectory.json` 的目录；脚本会按 `traj_id` 或 `id` 去拼接子目录。")
    ap.add_argument("--output_jsonl", type=str, required=True, help="输出 jsonl 路径；会写入带奖励字段的 replay rows。")
    ap.add_argument(
        "--output_mode",
        type=str,
        default="clip",
        choices=["clip", "traj"],
        help="`clip`：把每条 rollout 展开成 `num_clips` 个片段样本（推荐）；`traj`：每条 rollout 只保留 1 个轨迹级样本。",
    )
    # 在 m/rad 域里：
    # - xyz 项单位是 m^2
    # - yaw 项单位是 rad^2
    ap.add_argument("--alpha_xyz", type=float, default=1.0, help="动作奖励里 xyz 位置误差项的权重；误差在内部按 m^2 计。")
    ap.add_argument("--alpha_yaw", type=float, default=1.0, help="动作奖励里 yaw 误差项的权重；误差在内部按 rad^2 计。")
    ap.add_argument("--alpha_all6", type=float, default=0.2, help="动作奖励里 6 自由度整体 MSE 项的权重，用于补充 roll/pitch 等整体几何约束。")
    # 接近 GRPO 的 reward shaping：组内相对 z-score -> exp 映射。
    ap.add_argument(
        "--act_reward_mode",
        type=str,
        default="zscore_exp",
        choices=["inv1p", "zscore_exp", "minstd_exp"],
        help=(
            "`inv1p`：逐样本直接用 `1/(1+cost)`；"
            "`zscore_exp`：先做组内 z-score，再映射成 `exp(-max(0,z))`；"
            "`minstd_exp`：先算 `(x-min)/std`，再映射成 `exp(-x)`，结果非负且不超过 1，之后再做 LOO。"
        ),
    )
    ap.add_argument("--zscore_eps", type=float, default=1e-6, help="组内标准差的下界，避免同组样本几乎一致时除零。")
    ap.add_argument("--zscore_zmax", type=float, default=10.0, help="标准化后的截断上限，防止极端差样本把奖励压得过小。")
    ap.add_argument("--enable_ce_reward", type=int, default=1, help="是否启用基于旧策略 logprob/NLL 的 reference/CE 奖励项。")
    ap.add_argument("--lambda_act", type=float, default=1.0, help="最终总奖励里动作奖励 `r_act` 的系数。")
    ap.add_argument("--lambda_task", type=float, default=1.0, help="最终总奖励里任务奖励 `r_task` 的系数。")
    ap.add_argument("--lambda_ce", type=float, default=0.3, help="最终总奖励里 reference/CE 奖励 `r_ce` 的系数。")
    # clip 级 action reward 的时间衰减。
    ap.add_argument("--clip_len", type=int, default=16, help="每个 clip 覆盖多少个动作步；默认 16，对应常见的 17 帧窗口。")
    ap.add_argument("--num_clips", type=int, default=3, help="一条 rollout 最多拆成多少个 clip；默认配置下 49 帧通常对应 3 段。")
    ap.add_argument("--clip_alpha", type=float, default=0.9, help="后续 clip 的时间衰减系数；越靠后的片段权重越小。")
    # 成功阈值：
    # - clip_* 用于 clip 终点诊断
    # - task_* 用于对齐成功的加分（success-aligned bonus）/ 轨迹成功
    ap.add_argument("--clip_task_pos_thresh_m", type=float, default=0.6, help="clip 终点判成功时允许的位置误差阈值，单位米。")
    ap.add_argument("--clip_task_yaw_thresh_deg", type=float, default=2.5, help="clip 终点判成功时允许的 yaw 误差阈值，单位度。")
    ap.add_argument("--task_pos_thresh_m", type=float, default=3.0, help="整条轨迹判成功时允许的位置误差阈值，单位米。")
    ap.add_argument("--task_yaw_thresh_deg", type=float, default=10.0, help="整条轨迹判成功时允许的 yaw 误差阈值，单位度。")
    # dense 终点任务奖励的控制参数：
    # 代码/形状说明：reward = 1 / (1 + w_pos * (pos_err_m / pos_scale_m)^2 + w_yaw * (yaw_err_deg / yaw_scale_deg)^2 )
    ap.add_argument("--task_pos_scale_m", type=float, default=2.0, help="密集任务奖励里位置误差的归一化尺度，单位米。")
    ap.add_argument("--task_yaw_scale_deg", type=float, default=10.0, help="密集任务奖励里 yaw 误差的归一化尺度，单位度。")
    ap.add_argument("--task_pos_weight", type=float, default=1.0, help="密集任务奖励中位置误差项的权重。")
    ap.add_argument("--task_yaw_weight", type=float, default=1.0, help="密集任务奖励中 yaw 误差项的权重。")
    ap.add_argument("--task_enable_success_bonus", type=int, default=1, help="是否在密集奖励之外叠加成功加分。")
    ap.add_argument("--task_dense_weight", type=float, default=0.85, help="合成原始 task reward 时，密集奖励部分的系数。")
    ap.add_argument("--task_success_weight", type=float, default=0.15, help="合成原始 task reward 时，成功加分部分的系数。")
    # task reward 模式：
    # - raw_dense：直接使用密集奖励 + 可选成功加分
    # - raw_succ：直接使用只看成功的 task reward
    # - dense_loo/loo：在组内对原始 task reward 做 LOO
    ap.add_argument(
        "--task_reward_mode",
        type=str,
        default="raw_dense",
        choices=["raw_succ", "loo", "raw_dense", "dense_loo"],
        help="任务奖励模式：`raw_dense` 用密集奖励，`raw_succ` 只看成功，`loo`/`dense_loo` 会在组内再做留一法相对化。",
    )
    ap.add_argument("--require_old_logprob", type=int, default=1, help="严格模式：若缺旧策略 logprob，是否直接报错。")
    ap.add_argument("--require_all_trajectories", type=int, default=1, help="严格模式：若缺少 `trajectory.json`，是否直接报错。")
    args = ap.parse_args()

    in_path = os.path.abspath(args.replay_jsonl)
    out_path = os.path.abspath(args.output_jsonl)
    traj_root = os.path.abspath(args.trajectory_json_dir)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    n, ok = 0, 0
    missing_traj = 0
    missing_oldlp = 0
    mode = str(args.output_mode or "clip").strip().lower()
    if mode == "traj":
        # 轨迹级模式（legacy）：每条 rollout 只生成 1 个样本。
        for idx, meta in enumerate(rows):
            n += 1
            traj_id = str(meta.get("traj_id", meta.get("id", idx + 1)))
            traj_path = os.path.join(traj_root, str(traj_id), "trajectory.json")
            try:
                if not os.path.exists(traj_path):
                    missing_traj += 1
                    continue
                with open(traj_path, "r", encoding="utf-8") as f:
                    tj = json.load(f)
                pred = tj.get("relative_poses", {}).get("poses", [])
                try:
                    if "sample_logprob_total" not in tj:
                        missing_oldlp += 1
                    meta["grpo_old_logprob"] = float(tj.get("sample_logprob_total", meta.get("grpo_old_logprob", 0.0)))
                except Exception:
                    missing_oldlp += 1
                    pass
                if isinstance(tj.get("trace_files", None), list):
                    meta["grpo_trace_files"] = tj.get("trace_files", [])
                gt_path = meta.get("gt_pose_json", "")
                if not gt_path or not os.path.exists(gt_path):
                    continue
                gt = _load_gt_poses(gt_path)
                gt_use = gt[1:] if isinstance(gt, list) and len(gt) > 1 else []
                act_pack = _reward_act_with_clip_decay(
                    pred_poses=pred,
                    gt_poses=gt_use,
                    clip_len=int(args.clip_len),
                    num_clips=int(args.num_clips),
                    clip_alpha=float(args.clip_alpha),
                    alpha_xyz=float(args.alpha_xyz),
                    alpha_yaw=float(args.alpha_yaw),
                    alpha_all6=float(args.alpha_all6),
                )
                succ = _task_success_from_final(
                    pred_poses=pred,
                    gt_poses=gt_use,
                    pos_thresh_m=float(args.task_pos_thresh_m),
                    yaw_thresh_deg=float(args.task_yaw_thresh_deg),
                )
                dense_task = _dense_task_reward_from_final(
                    pred_poses=pred,
                    gt_poses=gt_use,
                    pos_scale_m=float(args.task_pos_scale_m),
                    yaw_scale_deg=float(args.task_yaw_scale_deg),
                    pos_weight=float(args.task_pos_weight),
                    yaw_weight=float(args.task_yaw_weight),
                )
                task_mode = str(args.task_reward_mode or "raw_dense").strip().lower()
                dense_raw = float(dense_task["reward"])
                success_raw = float(succ)
                task_raw = _compose_task_reward_raw(
                    task_mode=task_mode,
                    dense_raw=dense_raw,
                    success_raw=success_raw,
                    enable_success_bonus=int(args.task_enable_success_bonus),
                    task_dense_weight=float(args.task_dense_weight),
                    task_success_weight=float(args.task_success_weight),
                )
                meta["grpo_reward_act"] = float(act_pack["reward_act"])
                meta["grpo_reward_act_clips"] = act_pack["reward_act_clips"]
                meta["grpo_reward_act_weights"] = act_pack["reward_act_weights"]
                meta["grpo_succ"] = float(succ)
                meta["grpo_task_final_pos_err_m"] = float(dense_task["pos_err_m"])
                meta["grpo_task_final_yaw_err_deg"] = float(dense_task["yaw_err_deg"])
                meta["grpo_task_final_cost"] = float(dense_task["cost"])
                meta["grpo_reward_task_dense_raw"] = float(dense_raw)
                meta["grpo_reward_task_success_raw"] = float(success_raw)
                meta["grpo_reward_task_raw"] = float(task_raw)
                meta["grpo_mse"] = _clip_mse(pred, gt_use)
                meta["grpo_mse_clips"] = act_pack["mse_clips"]
                ok += 1
            except Exception:
                continue

        # 轨迹级 task reward 模式：raw_succ，或在原始组内做 LOO。
        groups: Dict[str, List[int]] = {}
        for i, meta in enumerate(rows):
            gid = str(meta.get("grpo_group_id", ""))
            if not gid:
                gid = f"__single_{i}"
            groups.setdefault(gid, []).append(i)
        task_mode = str(args.task_reward_mode or "raw_dense").strip().lower()
        if task_mode in ("loo", "dense_loo"):
            for _, inds in groups.items():
                if len(inds) <= 1:
                    for i in inds:
                        rows[i]["grpo_reward_task"] = 0.0
                    continue
                raws = np.asarray([float(rows[i].get("grpo_reward_task_raw", 0.0)) for i in inds], dtype=np.float64)
                advs = _loo_adv(raws)
                for off, i in enumerate(inds):
                    rows[i]["grpo_reward_task"] = float(advs[off])
        else:
            for meta in rows:
                meta["grpo_reward_task"] = float(meta.get("grpo_reward_task_raw", 0.0))
        for meta in rows:
            ra = float(meta.get("grpo_reward_act", meta.get("grpo_reward", 0.0)))
            rt = float(meta.get("grpo_reward_task", 0.0))
            meta["grpo_reward"] = float(ra + rt)
        for _, inds in groups.items():
            scores = np.asarray([float(rows[i].get("grpo_reward", 0.0)) for i in inds], dtype=np.float64)
            advs = _centered_rank_adv(scores)
            rank01 = _rank01_average_ties(scores)
            for off, i in enumerate(inds):
                rows[i]["grpo_score_final"] = float(scores[off])
                rows[i]["grpo_rank_final"] = float(rank01[off])
                rows[i]["grpo_adv_final"] = float(advs[off])

        out_rows = rows
    else:
        # clip 级模式（推荐）：把每条 rollout 展开成 num_clips 个 clip 样本。
        #
        # 中文导读：
        # 49 帧观测通常对应 48 个动作增量。这里默认拆成 3 个 clip：
        # - clip1：帧/动作 0..16
        # - clip2：帧/动作 16..32
        # - clip3：帧/动作 32..48
        # 每个 clip 单独成为 GRPO 训练样本，但仍保留整条轨迹成功率作为诊断/门控信号。
        out_rows: List[Dict[str, Any]] = []
        clip_len = int(args.clip_len)
        num_clips = int(args.num_clips)
        clip_alpha = float(args.clip_alpha)
        task_id_key = "grpo_group_id"

        for idx, meta in enumerate(rows):
            n += 1
            base_traj_id = str(meta.get("traj_id", meta.get("id", idx + 1)))
            traj_path = os.path.join(traj_root, str(base_traj_id), "trajectory.json")
            try:
                if not os.path.exists(traj_path):
                    missing_traj += 1
                    continue
                with open(traj_path, "r", encoding="utf-8") as f:
                    tj = json.load(f)
                pred_all = tj.get("relative_poses", {}).get("poses", [])
                seg_lp = tj.get("sample_logprob_segments", None)
                seg_tr = tj.get("trace_files", None)
                if not isinstance(seg_lp, list) or len(seg_lp) < num_clips:
                    missing_oldlp += 1
                    seg_lp = [tj.get("sample_logprob_total", meta.get("grpo_old_logprob", 0.0))] * num_clips
                if not isinstance(seg_tr, list) or len(seg_tr) < num_clips:
                    seg_tr = [[] for _ in range(num_clips)]
                gt_path = meta.get("gt_pose_json", "")
                if not gt_path or not os.path.exists(gt_path):
                    continue
                gt = _load_gt_poses(gt_path)
                gt_rel = _to_relative_cmdeg(gt) if isinstance(gt, list) else []
                gt_use_all = gt_rel[1:] if isinstance(gt_rel, list) and len(gt_rel) > 1 else []

                gid_base = str(meta.get(task_id_key, "")) or f"task_{idx:06d}"

                # 使用轨迹级阈值计算整条轨迹是否成功，仅作为诊断。
                succ_traj = 0.0
                traj_dense_task = {
                    "pos_err_m": 0.0,
                    "yaw_err_deg": 180.0,
                    "cost": 1e6,
                    "reward": 0.0,
                }
                traj_task_raw = 0.0
                try:
                    if len(pred_all) > 0 and len(gt_use_all) > 0:
                        succ_traj = _task_success_from_final(
                            pred_poses=[pred_all[-1]],
                            gt_poses=[gt_use_all[-1]],
                            pos_thresh_m=float(args.task_pos_thresh_m),
                            yaw_thresh_deg=float(args.task_yaw_thresh_deg),
                        )
                        traj_dense_task = _dense_task_reward_from_final(
                            pred_poses=[pred_all[-1]],
                            gt_poses=[gt_use_all[-1]],
                            pos_scale_m=float(args.task_pos_scale_m),
                            yaw_scale_deg=float(args.task_yaw_scale_deg),
                            pos_weight=float(args.task_pos_weight),
                            yaw_weight=float(args.task_yaw_weight),
                        )
                        traj_task_raw = _compose_task_reward_raw(
                            task_mode=str(args.task_reward_mode or "raw_dense").strip().lower(),
                            dense_raw=float(traj_dense_task["reward"]),
                            success_raw=float(succ_traj),
                            enable_success_bonus=int(args.task_enable_success_bonus),
                            task_dense_weight=float(args.task_dense_weight),
                            task_success_weight=float(args.task_success_weight),
                        )
                except Exception:
                    succ_traj = 0.0
                    traj_dense_task = {
                        "pos_err_m": 0.0,
                        "yaw_err_deg": 180.0,
                        "cost": 1e6,
                        "reward": 0.0,
                    }
                    traj_task_raw = 0.0

                for clip_pos in range(1, num_clips + 1):
                    st = (clip_pos - 1) * clip_len
                    ed = st + clip_len
                    pred_seg = pred_all[st:ed]
                    gt_seg = gt_use_all[st:ed]
                    # 动作奖励先衡量预测轨迹和专家轨迹的几何一致性。
                    # 这里先算成类似 MSE 的标量 cost，下面再转成组内相对优势值，
                    # 让同一任务里的候选彼此竞争。
                    mse = _clip_mse(pred_seg, gt_seg)
                    act_mse_scalar = _act_mse_scalar(
                        mse=mse,
                        alpha_xyz=float(args.alpha_xyz),
                        alpha_yaw=float(args.alpha_yaw),
                        alpha_all6=float(args.alpha_all6),
                    )
                    r_act_inv = _reward_from_mse(
                        mse=mse,
                        alpha_xyz=float(args.alpha_xyz),
                        alpha_yaw=float(args.alpha_yaw),
                        alpha_all6=float(args.alpha_all6),
                    )
                    w = float(clip_alpha ** (clip_pos - 1))
                    # r_act 会在拿到组内统计量后再最终确定（z-score-exp 模式）。
                    r_act = float(r_act_inv * w)

                    # 密集奖励和成功加分都对齐当前 clip 终点。
                    # 整条轨迹的 success 只单独保留作诊断。
                    succ = 0.0
                    dense_task = {
                        "pos_err_m": 0.0,
                        "yaw_err_deg": 180.0,
                        "cost": 1e6,
                        "reward": 0.0,
                    }
                    if len(pred_all) >= ed and len(gt_use_all) >= ed:
                        succ = _task_success_from_final(
                            pred_poses=[pred_all[ed - 1]],
                            gt_poses=[gt_use_all[ed - 1]],
                            pos_thresh_m=float(args.clip_task_pos_thresh_m),
                            yaw_thresh_deg=float(args.clip_task_yaw_thresh_deg),
                        )
                        dense_task = _dense_task_reward_from_final(
                            pred_poses=[pred_all[ed - 1]],
                            gt_poses=[gt_use_all[ed - 1]],
                            pos_scale_m=float(args.task_pos_scale_m),
                            yaw_scale_deg=float(args.task_yaw_scale_deg),
                            pos_weight=float(args.task_pos_weight),
                            yaw_weight=float(args.task_yaw_weight),
                        )

                    # 使用 17 帧窗口来满足 4n+1 规则：[0..16], [16..32], [32..48]。
                    begin_frame = (clip_pos - 1) * clip_len
                    end_frame = begin_frame + clip_len
                    clip_meta = dict(meta)
                    # frame id 保持为 17 帧窗口边界 [begin, end]。
                    # 这样满足 world model 使用的 video-latent 4n+1 约定。
                    clip_meta["begin_frame_id"] = int(begin_frame)
                    clip_meta["end_frame_id"] = int(end_frame)
                    clip_meta["traj_id"] = f"{base_traj_id}_c{clip_pos}"
                    clip_meta["grpo_group_id"] = f"{gid_base}_clip{clip_pos}"
                    clip_meta["grpo_clip_id"] = int(clip_pos)
                    clip_meta["grpo_traj_group_id"] = str(gid_base)
                    clip_meta["grpo_old_logprob"] = float(seg_lp[clip_pos - 1])
                    clip_meta["grpo_trace_files"] = [seg_tr[clip_pos - 1]] if seg_tr[clip_pos - 1] else []
                    clip_meta["grpo_act_mse_scalar"] = float(act_mse_scalar)
                    clip_meta["grpo_reward_act_inv_raw"] = float(r_act_inv)
                    clip_meta["grpo_reward_act"] = float(r_act)
                    clip_meta["grpo_succ"] = float(succ)
                    clip_meta["grpo_succ_traj"] = float(succ_traj)
                    clip_meta["grpo_traj_final_pos_err_m"] = float(traj_dense_task["pos_err_m"])
                    clip_meta["grpo_traj_final_yaw_err_deg"] = float(traj_dense_task["yaw_err_deg"])
                    clip_meta["grpo_traj_final_cost"] = float(traj_dense_task["cost"])
                    clip_meta["grpo_reward_task_traj_dense_raw"] = float(traj_dense_task["reward"])
                    clip_meta["grpo_reward_task_traj_success_raw"] = float(succ_traj)
                    clip_meta["grpo_reward_task_traj_raw"] = float(traj_task_raw)
                    clip_meta["grpo_task_final_pos_err_m"] = float(dense_task["pos_err_m"])
                    clip_meta["grpo_task_final_yaw_err_deg"] = float(dense_task["yaw_err_deg"])
                    clip_meta["grpo_task_final_cost"] = float(dense_task["cost"])
                    clip_meta["grpo_mse"] = mse
                    clip_meta["grpo_ce_nll"] = float(max(0.0, -float(clip_meta.get("grpo_old_logprob", 0.0))))
                    clip_meta["grpo_reward_ce_raw"] = 0.0
                    clip_meta["grpo_reward_ce_adv"] = 0.0
                    dense_raw = float(dense_task["reward"])
                    success_raw = float(succ)
                    task_raw = _compose_task_reward_raw(
                        task_mode=str(args.task_reward_mode or "raw_dense").strip().lower(),
                        dense_raw=dense_raw,
                        success_raw=success_raw,
                        enable_success_bonus=int(args.task_enable_success_bonus),
                        task_dense_weight=float(args.task_dense_weight),
                        task_success_weight=float(args.task_success_weight),
                    )
                    clip_meta["grpo_reward_task_dense_raw"] = float(dense_raw)
                    clip_meta["grpo_reward_task_success_raw"] = float(success_raw)
                    clip_meta["grpo_reward_task_raw"] = float(task_raw)
                    clip_meta["grpo_reward_task"] = 0.0
                    clip_meta["grpo_reward"] = float(r_act)  # LOO 之后再加 task 项。
                    clip_meta["grpo_reward_decay"] = float(w)
                    out_rows.append(clip_meta)
                    ok += 1
            except Exception:
                continue

        # 在 clip group 粒度为 task reward 计算 group LOO baseline。
        #
        # 中文导读：
        # `grpo_group_id` 把同一个任务/clip 的 K 个候选放在一起。LOO baseline
        # 让奖励表达“这个候选相对同组其他候选是否更好”，而不是只看绝对分数。
        groups: Dict[str, List[int]] = {}
        for i, meta in enumerate(out_rows):
            gid = str(meta.get("grpo_group_id", "")) or f"__single_{i}"
            groups.setdefault(gid, []).append(i)

        task_mode = str(args.task_reward_mode or "raw_dense").strip().lower()
        for _, inds in groups.items():
            task_raws = np.asarray([float(out_rows[i].get("grpo_reward_task_raw", 0.0)) for i in inds], dtype=np.float64)
            if task_mode in ("loo", "dense_loo"):
                task_vals = _loo_adv(task_raws) if len(inds) > 1 else np.zeros_like(task_raws, dtype=np.float64)
            else:
                task_vals = task_raws
            for off, i in enumerate(inds):
                w = float(out_rows[i].get("grpo_reward_decay", 1.0))
                out_rows[i]["grpo_reward_task_adv"] = float(task_vals[off] if task_mode in ("loo", "dense_loo") else 0.0)
                out_rows[i]["grpo_reward_task"] = float(task_vals[off] * w)
                out_rows[i]["grpo_reward"] = float(out_rows[i].get("grpo_reward_act", 0.0) + out_rows[i].get("grpo_reward_task", 0.0))

        # act reward shaping 和可选 CE reward shaping 都是组内相对的（接近 GRPO）。
        #
        # 中文导读：
        # - act reward：基于动作/轨迹误差，误差越小越好；
        # - task reward：基于当前 clip 终点或成功阈值；
        # - CE/ref reward：基于旧策略 logprob/NLL 的约束项，减少策略更新后偏离参考分布。
        act_mode = str(args.act_reward_mode or "zscore_exp").strip().lower()
        do_ce = int(args.enable_ce_reward) == 1
        eps = float(args.zscore_eps)
        zmax = float(args.zscore_zmax)
        for _, inds in groups.items():
            idx = np.asarray(inds, dtype=np.int64)
            # ---- act：基于 MSE 标量（越低越好）----
            act_x = np.asarray([float(out_rows[i].get("grpo_act_mse_scalar", 0.0)) for i in idx], dtype=np.float64)
            if act_mode == "zscore_exp":
                pack = _zscore_exp_reward(act_x, eps=eps, zmax=zmax)
                r_nodecay = pack["r"]
                a_nodecay = _loo_adv(r_nodecay)
                for j, i in enumerate(idx.tolist()):
                    w = float(out_rows[i].get("grpo_reward_decay", 1.0))
                    out_rows[i]["grpo_reward_act_raw"] = float(r_nodecay[j])
                    out_rows[i]["grpo_reward_act_adv"] = float(a_nodecay[j] * w)
                    # 使用 LOO advantage 作为 act reward（更接近 GRPO）。
                    out_rows[i]["grpo_reward_act"] = float(a_nodecay[j] * w)
                    out_rows[i]["grpo_reward_act_z"] = float(pack["z"][j])
                    out_rows[i]["grpo_reward_act_mu"] = float(pack["mu"][0])
                    out_rows[i]["grpo_reward_act_sigma"] = float(pack["sigma"][0])
            elif act_mode == "minstd_exp":
                pack = _minstd_exp_reward(act_x, eps=eps, zmax=zmax)
                r_nodecay = pack["r"]
                a_nodecay = _loo_adv(r_nodecay)
                for j, i in enumerate(idx.tolist()):
                    w = float(out_rows[i].get("grpo_reward_decay", 1.0))
                    out_rows[i]["grpo_reward_act_raw"] = float(r_nodecay[j])
                    out_rows[i]["grpo_reward_act_adv"] = float(a_nodecay[j] * w)
                    out_rows[i]["grpo_reward_act"] = float(a_nodecay[j] * w)
                    out_rows[i]["grpo_reward_act_xminus"] = float(pack["xminus"][j])
                    out_rows[i]["grpo_reward_act_xmin"] = float(pack["xmin"][0])
                    out_rows[i]["grpo_reward_act_mu"] = float(pack["mu"][0])
                    out_rows[i]["grpo_reward_act_sigma"] = float(pack["sigma"][0])
            else:
                # inv1p legacy：保留已经写入的 r_act_inv_raw * decay。
                for i in idx.tolist():
                    out_rows[i]["grpo_reward_act_raw"] = float(out_rows[i].get("grpo_reward_act_inv_raw", 0.0))
                    out_rows[i]["grpo_reward_act_adv"] = 0.0

            # ---- ce：基于 old logprob 得到的 NLL（越低越好）----
            if do_ce:
                ce_x = np.asarray([float(out_rows[i].get("grpo_ce_nll", 0.0)) for i in idx], dtype=np.float64)
                # act_mode 为 minstd_exp 时，让 CE shaping 和 act shaping 保持一致。
                pack_ce = _minstd_exp_reward(ce_x, eps=eps, zmax=zmax) if act_mode == "minstd_exp" else _zscore_exp_reward(ce_x, eps=eps, zmax=zmax)
                r_ce_raw_nodecay = pack_ce["r"]
                a_ce_nodecay = _loo_adv(r_ce_raw_nodecay)
                for j, i in enumerate(idx.tolist()):
                    w = float(out_rows[i].get("grpo_reward_decay", 1.0))
                    out_rows[i]["grpo_reward_ce_raw"] = float(r_ce_raw_nodecay[j] * w)
                    out_rows[i]["grpo_reward_ce_adv"] = float(a_ce_nodecay[j] * w)
                    if act_mode == "minstd_exp":
                        out_rows[i]["grpo_reward_ce_xminus"] = float(pack_ce["xminus"][j])
                        out_rows[i]["grpo_reward_ce_xmin"] = float(pack_ce["xmin"][0])
                    else:
                        out_rows[i]["grpo_reward_ce_z"] = float(pack_ce["z"][j])
                    out_rows[i]["grpo_reward_ce_mu"] = float(pack_ce["mu"][0])
                    out_rows[i]["grpo_reward_ce_sigma"] = float(pack_ce["sigma"][0])

            # ---- 最终组合 reward ----
            for i in idx.tolist():
                r_act = float(out_rows[i].get("grpo_reward_act", 0.0))
                r_task = float(out_rows[i].get("grpo_reward_task", 0.0))
                r_ce = float(out_rows[i].get("grpo_reward_ce_adv", 0.0)) if do_ce else 0.0
                # 中文导读：
                # 这里对应 Action-aware GRPO 的片段奖励组合。
                # 核心公式是加权和：
                # 代码/形状说明：reward = lambda_act*r_act + lambda_task*r_task + lambda_ce*r_ce
                # `r_act` 约束动作几何一致性，`r_task` 关注任务进度，
                # `r_ce` 近似 reference reward，防止策略更新把世界模型先验拉坏。
                out_rows[i]["grpo_reward"] = float(float(args.lambda_act) * r_act + float(args.lambda_task) * r_task + float(args.lambda_ce) * r_ce)
                # KL anchor 使用的 reference logprob；离线阶段默认用 behavior/old logprob。
                if "grpo_ref_logprob" not in out_rows[i]:
                    out_rows[i]["grpo_ref_logprob"] = float(out_rows[i].get("grpo_old_logprob", 0.0))
            scores = np.asarray([float(out_rows[i].get("grpo_reward", 0.0)) for i in idx], dtype=np.float64)
            score_pos = np.clip(scores, 0.0, None)
            traj_gate = np.asarray(
                [
                    1.0
                    if float(out_rows[i].get("grpo_succ_traj", 0.0)) > 0.0
                    else np.clip(float(out_rows[i].get("grpo_reward_task_traj_raw", 0.0)), 0.0, 1.0)
                    for i in idx.tolist()
                ],
                dtype=np.float64,
            )
            final_w = np.clip(score_pos * traj_gate, 0.0, 1.0)
            rank01 = _rank01_average_ties(scores)
            for j, i in enumerate(idx.tolist()):
                out_rows[i]["grpo_score_final"] = float(scores[j])
                out_rows[i]["grpo_score_final_raw"] = float(scores[j])
                out_rows[i]["grpo_score_final_pos"] = float(score_pos[j])
                out_rows[i]["grpo_traj_gate"] = float(traj_gate[j])
                out_rows[i]["grpo_rank_final"] = float(rank01[j])
                out_rows[i]["grpo_adv_final"] = float(final_w[j])

    _print_final_weight_summary(out_rows, out_path)
    with open(out_path, "w", encoding="utf-8") as fout:
        for meta in out_rows:
            fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

    if int(args.require_all_trajectories) == 1 and int(missing_traj) > 0:
        raise RuntimeError(f"严格模式：缺少 trajectory.json，共 {missing_traj} 行")
    if int(args.require_old_logprob) == 1 and int(missing_oldlp) > 0:
        raise RuntimeError(f"严格模式：缺少 sample_logprob_total，共 {missing_oldlp} 行")
    print(f"[reward_uavflow] 已处理 processed={n}, 已更新 updated={ok}, 输出 out={out_path}")


if __name__ == "__main__":
    main()
