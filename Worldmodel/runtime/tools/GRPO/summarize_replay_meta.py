#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import numpy as np


def _iter_jsonl(path: str):
    """逐行读取 jsonl replay meta。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _resolve_inputs(input_jsonl: str, replay_meta_dir: str) -> List[str]:
    """解析单个 input_jsonl 或 replay_meta_dir 下的 part_*.jsonl。"""
    if input_jsonl:
        paths = [os.path.abspath(input_jsonl)]
    elif replay_meta_dir:
        paths = sorted(glob.glob(os.path.join(os.path.abspath(replay_meta_dir), "part_*.jsonl")))
    else:
        raise ValueError("必须提供 --input_jsonl 或 --replay_meta_dir 其中一个")
    paths = [path for path in paths if os.path.isfile(path)]
    if not paths:
        raise FileNotFoundError("没有找到有效的 replay 输入文件")
    return paths


def _safe_mean(values: List[float]) -> float:
    """空列表返回 0 的 mean，避免 summary 阶段异常。"""
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> None:
    """CLI 入口：汇总 replay meta 的 reward/advantage 分布并执行安全检查。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, default="")
    ap.add_argument("--replay_meta_dir", type=str, default="")
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--fail_on_negative_adv", type=int, default=1)
    ap.add_argument("--fail_on_success_negative", type=int, default=1)
    args = ap.parse_args()

    paths = _resolve_inputs(args.input_jsonl, args.replay_meta_dir)
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(list(_iter_jsonl(path)))

    if not rows:
        raise RuntimeError("replay rows 为空")

    group_size: Dict[str, int] = {}
    for row in rows:
        gid = str(row.get("grpo_group_id", ""))
        group_size[gid] = group_size.get(gid, 0) + 1

    adv = np.asarray([float(row.get("grpo_adv_final", row.get("grpo_weight", 0.0))) for row in rows], dtype=np.float64)
    succ = np.asarray([float(row.get("grpo_reward_task_success_raw", 0.0)) for row in rows], dtype=np.float64)
    task_raw = np.asarray([float(row.get("grpo_reward_task_raw", row.get("grpo_reward_task", 0.0))) for row in rows], dtype=np.float64)
    succ_diag = np.asarray([float(row.get("grpo_succ_traj", row.get("grpo_succ", 0.0))) for row in rows], dtype=np.float64)
    pos_err = [float(row.get("grpo_task_final_pos_err_m", 0.0)) for row in rows]
    yaw_err = [float(row.get("grpo_task_final_yaw_err_deg", 0.0)) for row in rows]

    success_negative_count = int(np.sum((succ > 0.0) & (adv < 0.0)))
    high_task_negative_count = int(np.sum((task_raw >= 0.8) & (adv < 0.0)))

    summary = {
        "rows": int(len(rows)),
        "groups": int(len(group_size)),
        "group_size_set": sorted({int(v) for v in group_size.values()}),
        "succ_frac": float(np.mean(succ_diag > 0.0)),
        "reward_task_raw_mean": float(np.mean(task_raw)) if task_raw.size > 0 else 0.0,
        "pos_err_m_mean": _safe_mean(pos_err),
        "yaw_err_deg_mean": _safe_mean(yaw_err),
        "weight_pos_frac": float(np.mean(adv > 0.0)),
        "weight_zero_frac": float(np.mean(adv == 0.0)),
        "weight_neg_frac": float(np.mean(adv < 0.0)),
        "success_negative_count": success_negative_count,
        "high_task_negative_count": high_task_negative_count,
        "weight_min": float(np.min(adv)) if adv.size > 0 else 0.0,
        "weight_max": float(np.max(adv)) if adv.size > 0 else 0.0,
        "weight_mean": float(np.mean(adv)) if adv.size > 0 else 0.0,
        "inputs": paths,
    }

    out_path = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "[summarize_replay_meta] "
        f"行数 rows={summary['rows']} 组数 groups={summary['groups']} "
        f"weight_pos_frac={summary['weight_pos_frac']:.6f} weight_zero_frac={summary['weight_zero_frac']:.6f} "
        f"weight_neg_frac={summary['weight_neg_frac']:.6f} success_negative_count={success_negative_count} "
        f"high_task_negative_count={high_task_negative_count} 输出 out={out_path}"
    )

    if int(args.fail_on_negative_adv) == 1 and float(summary["weight_neg_frac"]) > 0.0:
        raise SystemExit("replay meta 中检测到负 advantage")
    if int(args.fail_on_success_negative) == 1 and success_negative_count > 0:
        raise SystemExit("replay meta 中检测到成功样本却带有负 advantage")
    if int(args.fail_on_success_negative) == 1 and high_task_negative_count > 0:
        raise SystemExit("replay meta 中检测到高 task 分样本却带有负 advantage")


if __name__ == "__main__":
    main()
