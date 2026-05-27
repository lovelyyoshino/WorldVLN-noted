#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把任务级 jsonl 扩展成 K 候选 rollout 元数据 jsonl。

中文导读：
`build_rollout_tasks.py` 产出的还是“每个 clip 一条任务”；这个脚本再往前走一步，
把每条任务复制成 K 个候选，给每个候选分配稳定 seed、`candidate_id` 和 `traj_id`，
以便后续 rollout 生成器把每个候选的视频/轨迹结果写回固定目录。

输出新增的关键字段：
- `candidate_id`: 同一任务内第几个候选，范围通常是 `0..K-1`
- `seed`: 该候选的采样随机种子
- `traj_id`: 默认形如 `000123_k02`，下游常用它去定位 `trajectory.json`

要点：
- `grpo_group_id` 保持不变，表示“这些候选属于同一任务组”；
- `traj_id` 必须唯一，表示“这一条具体候选的 rollout 目录名”。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def _candidate_seed(seed_base: int, task_idx: int, cand_idx: int, task_seed_stride: int, candidate_seed_stride: int) -> int:
    """
        为同一任务的不同候选生成稳定但分散的采样 seed。

        公式：
          公式/形状说明：seed = seed_base + task_idx*task_seed_stride + cand_idx*candidate_seed_stride
        task_idx 负责把不同任务隔开，cand_idx 负责把同一任务的 K 个候选隔开。

    """
    # 保持 group id 稳定，同时用较大的 stride 拉开候选 seed；
    # 这样同一任务的 K 个候选会探索不同采样路径，而不是只差相邻 seed。
    return int(seed_base + task_idx * task_seed_stride + cand_idx * candidate_seed_stride)


def main():
    """
    CLI 入口：把每条任务扩展成 K 个候选 rollout 记录。

    推荐先对一小份 task jsonl 运行，观察输出里的：
    `grpo_group_id -> candidate_id -> seed -> traj_id`
    四个字段是如何共同标识“同组不同候选”的。
    """
    ap = argparse.ArgumentParser(
        description=(
            "把任务级 GRPO task jsonl 扩展成 K 候选 rollout 元数据。"
            "输出结果常作为后续视频生成/trajectory 收集脚本的输入。"
        )
    )
    ap.add_argument("--task_jsonl", type=str, required=True, help="输入任务级 jsonl；通常由 build_rollout_tasks.py 生成。")
    ap.add_argument("--output_jsonl", type=str, required=True, help="输出候选级 jsonl；每条任务会扩展成 K 行。")
    ap.add_argument("--k", type=int, default=4, help="每条任务展开成多少个候选 rollout。")
    ap.add_argument("--seed_base", type=int, default=20260320, help="全局 seed 基值；最终候选 seed 会在此基础上按任务和候选编号偏移。")
    ap.add_argument("--task_seed_stride", type=int, default=1000003, help="不同任务之间的 seed 间隔；值大一些可减少不同任务 seed 过近。")
    ap.add_argument("--candidate_seed_stride", type=int, default=65537, help="同一任务内不同候选之间的 seed 间隔。")
    args = ap.parse_args()

    task_path = os.path.abspath(args.task_jsonl)
    out_path = os.path.abspath(args.output_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tasks: List[Dict[str, Any]] = []
    with open(task_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    rows: List[Dict[str, Any]] = []
    for ti, t in enumerate(tasks):
        gid = str(t.get("grpo_group_id", f"task_{ti}"))
        clip_id = int(t.get("grpo_clip_id", 1))
        for ki in range(int(args.k)):
            r = dict(t)
            r["grpo_group_id"] = gid  # 同组候选保持同一个 group id，后续 reward/advantage 会按组比较。
            r["grpo_clip_id"] = int(clip_id)
            r["candidate_id"] = int(ki)  # 组内第几个候选。
            r["seed"] = _candidate_seed(
                seed_base=int(args.seed_base),
                task_idx=int(ti),
                cand_idx=int(ki),
                task_seed_stride=int(args.task_seed_stride),
                candidate_seed_stride=int(args.candidate_seed_stride),
            )
            # traj_id 默认同时编码“第几个任务 + 第几个候选”，下游常用它作为 rollout 目录名。
            r["traj_id"] = f"{ti:06d}_k{ki:02d}"
            r["grpo_old_logprob"] = float(r.get("grpo_old_logprob", 0.0))
            rows.append(r)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[generate_candidate_rollouts] 任务数 tasks={len(tasks)} k={args.k} 输出行数 rows={len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
