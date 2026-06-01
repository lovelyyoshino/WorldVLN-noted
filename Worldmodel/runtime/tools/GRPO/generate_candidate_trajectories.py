#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合成候选轨迹生成器（bootstrap 用）。

中文导读：
和 `generate_candidate_trajectories_real.py` 不同，这个脚本不需要真实模拟器，也不调用
世界模型。它只读取 GT pose 序列，按 `seed + candidate_id` 加扰动，伪造 K 条候选轨迹。

它处于 GRPO 数据准备链的“可选 bootstrap”位置：
    candidates jsonl
        -> 本脚本（每条候选只是 GT + 噪声）
        -> reward_uavflow.py（依然能跑出 r_act / r_task）
        -> 训练流水线

主要用途：
- 在没有真实闭环 rollout 之前先跑通整条 GRPO 训练流水线；
- 当真实模拟器不可用时，提供一个 deterministic 的回退候选源；
- candidate_id 越大，注入的噪声越强，便于观察 reward 分布是否随之恶化。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List

import numpy as np


def _load_gt_poses(path: str) -> List[List[float]]:
    """读取 GT pose，兼容 list pose 和 pose_log dict 格式。"""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # 情况 A：已经是 [[x,y,z,roll,yaw,pitch], ...]。
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        return [[float(v) for v in row[:6]] for row in obj]
    # 情况 B：pose_log 格式 [{commanded:{...}}, ...]。
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
    raise ValueError(f"不支持的 GT pose 格式：{path}")


def _to_relative_poses(poses_abs: List[List[float]]) -> List[List[float]]:
    """把绝对 pose 序列转成相对第 0 帧的 pose 序列。"""
    if len(poses_abs) <= 1:
        return []
    base = poses_abs[0]
    rel = []
    for p in poses_abs[1:]:
        rel.append([float(p[i] - base[i]) for i in range(6)])
    return rel


def _jitter_rel_poses(rel: List[List[float]], seed: int, candidate_id: int, pos_std: float, yaw_std_deg: float) -> List[List[float]]:
    """给相对轨迹加受 seed/candidate_id 控制的扰动，用于自举候选（bootstrap）。"""
    # 为 GRPO 自举（bootstrap）构造多样化的合成候选：
    # candidate_id=0 尽量贴近 GT，id 越大扰动越强。
    rng = np.random.default_rng(int(seed) + int(candidate_id) * 1000003)
    scale = 1.0 + 0.5 * float(candidate_id)
    out = []
    for p in rel:
        q = list(p)
        q[0] += float(rng.normal(0.0, pos_std * scale))
        q[1] += float(rng.normal(0.0, pos_std * scale))
        q[2] += float(rng.normal(0.0, pos_std * scale))
        q[4] += float(rng.normal(0.0, yaw_std_deg * scale))
        q[4] = float((q[4] + 180.0) % 360.0 - 180.0)
        out.append(q)
    return out


def main():
    """CLI 入口：从 GT 轨迹生成带扰动的候选 trajectory.json。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_jsonl", type=str, required=True)
    ap.add_argument("--trajectory_root", type=str, required=True)
    ap.add_argument("--pos_noise_std", type=float, default=0.02, help="位置噪声标准差，单位为米")
    ap.add_argument("--yaw_noise_std_deg", type=float, default=2.0, help="yaw 噪声标准差，单位为度")
    args = ap.parse_args()

    cand_path = os.path.abspath(args.candidates_jsonl)
    traj_root = os.path.abspath(args.trajectory_root)
    os.makedirs(traj_root, exist_ok=True)

    n, ok = 0, 0
    with open(cand_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            row: Dict[str, Any] = json.loads(line)
            traj_id = str(row.get("traj_id", f"traj_{n:06d}"))
            gt_path = str(row.get("gt_pose_json", row.get("coordinates_path", "")))
            if not gt_path or not os.path.exists(gt_path):
                continue
            try:
                gt_abs = _load_gt_poses(gt_path)
                rel = _to_relative_poses(gt_abs)
                cand_id = int(row.get("candidate_id", 0))
                seed = int(row.get("seed", 0))
                pred_rel = _jitter_rel_poses(
                    rel,
                    seed=seed,
                    candidate_id=cand_id,
                    pos_std=float(args.pos_noise_std),
                    yaw_std_deg=float(args.yaw_noise_std_deg),
                )
                out_dir = os.path.join(traj_root, traj_id)
                os.makedirs(out_dir, exist_ok=True)
                payload = {
                    "traj_id": traj_id,
                    "seed": seed,
                    "candidate_id": cand_id,
                    "relative_poses": {
                        "start_pose": [0.0] * 6,
                        "poses": pred_rel,
                        "final_pose": (pred_rel[-1] if pred_rel else [0.0] * 6),
                    },
                    "note": "从 GT 加 seed 控制扰动生成的合成候选轨迹。",
                }
                with open(os.path.join(out_dir, "trajectory.json"), "w", encoding="utf-8") as wf:
                    json.dump(payload, wf, ensure_ascii=False, indent=2)
                ok += 1
            except Exception:
                continue
    print(f"[generate_candidate_trajectories] 已处理 processed={n}, 已写入 wrote={ok}, 根目录 root={traj_root}")


if __name__ == "__main__":
    main()
