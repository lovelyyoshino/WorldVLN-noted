#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO replay 训练前的最终归一化脚本。

中文导读：
这个脚本处于 GRPO 数据管线的“最后一公里”：

    rollout cache  ->  reward_uavflow.py（每行带 r_act / r_task / r_ce 等原始字段）
                  ->  本脚本（统一写入 grpo_weight / grpo_score / grpo_gate）
                  ->  trainer 直接消费 jsonl

它不会再去重新计算几何或任务奖励，而是依据 `--mode` 把已有的 reward 字段“翻译”
成训练阶段实际使用的三元组：
- `grpo_weight`：实际进入 loss 的样本权重；
- `grpo_score`：用于排序、记录的标量分数；
- `grpo_gate`：是否参与策略梯度的二值/软门控。

四种模式的应用场景：
- `precomputed_adv`：reward_uavflow.py 已经写入 `grpo_adv_final`，直接拿来当权重；
- `raw_reward`：不做门控/排序，按原始 reward 训练，调试用最多；
- `gate_mean`：组内均值以上才参与训练，简单的 hard gate；
- `rank_gate`：保留旧版 rank+decay 训练协议，便于复现历史实验。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np


def _rank01_average_ties(vals: np.ndarray) -> np.ndarray:
    """把同组 reward 转成 [0,1] rank，ties 使用平均 rank。"""
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


def main():
    """CLI 入口：从 reward 字段生成 grpo_weight/grpo_score/grpo_gate。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--lambda_act", type=float, default=1.0)
    ap.add_argument("--lambda_task", type=float, default=1.0)
    ap.add_argument("--lambda_ce", type=float, default=0.0)
    ap.add_argument("--alpha_decay", type=float, default=0.9)
    ap.add_argument(
        "--mode",
        type=str,
        default="precomputed_adv",
        choices=["precomputed_adv", "raw_reward", "gate_mean", "rank_gate"],
        help="如何从 rewards 推导 grpo_weight/grpo_score/grpo_gate（保留旧版 rank_gate 以兼容历史数据）。",
    )
    args = ap.parse_args()

    in_path = os.path.abspath(args.input_jsonl)
    out_path = os.path.abspath(args.output_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        gid = str(r.get("grpo_group_id", ""))
        if not gid:
            gid = f"__single_{i}"
        groups.setdefault(gid, []).append(i)

    for _, inds in groups.items():
        idx = np.asarray(inds, dtype=np.int64)
        r_act = np.asarray([float(rows[i].get("grpo_reward_act", rows[i].get("grpo_reward", 0.0))) for i in idx], dtype=np.float64)
        r_task = np.asarray([float(rows[i].get("grpo_reward_task", 0.0)) for i in idx], dtype=np.float64)
        # 优先使用已有 CE 优势值（advantage，更接近 GRPO）；否则退回原始 CE。
        r_ce = np.asarray(
            [
                float(
                    rows[i].get(
                        "grpo_reward_ce_adv",
                        rows[i].get("grpo_reward_ce_raw", rows[i].get("grpo_reward_ce", 0.0)),
                    )
                )
                for i in idx
            ],
            dtype=np.float64,
        )
        clip_ids = np.asarray([int(rows[i].get("grpo_clip_id", 1)) for i in idx], dtype=np.int64)
        r = float(args.lambda_act) * r_act + float(args.lambda_task) * r_task + float(args.lambda_ce) * r_ce
        mode = str(args.mode or "raw_reward").strip().lower()
        has_precomputed_adv = all("grpo_adv_final" in rows[i] for i in idx.tolist())
        adv_pre = np.asarray([float(rows[i].get("grpo_adv_final", 0.0)) for i in idx], dtype=np.float64)
        if mode == "precomputed_adv" and has_precomputed_adv:
            s = r.copy()
            w = adv_pre.copy()
            m = (w > 0).astype(np.float64)
        elif mode == "gate_mean":
            mu = float(np.mean(r)) if r.size > 0 else 0.0
            m = (r >= mu).astype(np.float64)
            s = r.copy()
            w = m * r
        elif mode == "rank_gate":
            s_act = _rank01_average_ties(r_act)
            s_task = _rank01_average_ties(r_task)
            s_ce = _rank01_average_ties(r_ce)
            s = float(args.lambda_act) * s_act + float(args.lambda_task) * s_task + float(args.lambda_ce) * s_ce
            mu = float(np.mean(r)) if r.size > 0 else 0.0
            m = (r >= mu).astype(np.float64)
            w = (np.power(float(args.alpha_decay), np.maximum(0, clip_ids - 1))) * m * s
        else:
            # raw_reward：不做门控（gating），也不做排序（ranking）。
            m = np.ones_like(r, dtype=np.float64)
            s = r.copy()
            w = r.copy()
        for j, i in enumerate(idx.tolist()):
            rows[i]["grpo_weight"] = float(w[j])
            rows[i]["grpo_score"] = float(s[j])
            rows[i]["grpo_gate"] = float(m[j])

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build_replay_dataset] 已写入 {len(rows)} 行 -> {out_path}")


if __name__ == "__main__":
    main()
