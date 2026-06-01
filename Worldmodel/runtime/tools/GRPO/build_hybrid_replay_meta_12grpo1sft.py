#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
12 GRPO + 1 SFT 混合 replay 元数据构建器。

中文导读：
这是 GRPO 数据准备链的“最后整理”阶段，把 reward_uavflow.py 输出的 part_*.jsonl
重新整理成训练阶段直接消费的混合 replay：

    rollout cache
        -> reward_uavflow.py（输出 part_*.jsonl，每条 row 带 grpo_clip_id / candidate_id）
        -> 本脚本：按 base task 重组，每个任务凑齐 K * num_clips 条 GRPO 候选，
            可选追加 1 条 SFT anchor，然后再分片成 num_parts 个 part_*.jsonl
        -> trainer 直接读取整理后的 part_*.jsonl

每个 group 的常见布局：
- `num_clips=3`、`k=4` -> 12 条 GRPO 候选；
- 默认 `add_sft_anchor=1` 时，再加 1 条 `hybrid_role="sft"` 的 anchor row；
- 这样每个 group 大小固定为 13，下游训练循环不需要处理可变长度。

`pad_to_equal_groups=1` 是为了适配多卡训练：让每个 shard 的 group 数量一致，
避免按最短 shard 截断时丢失独有任务。
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Item:
    """一条候选 replay row 的索引信息；保留给后续扩展和类型提示。"""

    base: str
    clip_id: int
    cand_id: int
    obj: dict


def _traj_base(traj_id: str) -> str:
    """从 `000000_k00_c1` 这类候选轨迹 ID 中取基础任务 ID。"""
    # 预期格式：000000_k00_c1
    s = str(traj_id)
    if "_k" in s:
        return s.split("_k", 1)[0]
    return s


def _read_all_parts(inp_dir: str) -> List[dict]:
    """读取 replay_meta_dir 下所有 part_*.jsonl 并合并成 row 列表。"""
    parts = []
    for name in sorted(os.listdir(inp_dir)):
        if not (name.endswith(".jsonl") and name.startswith("part_")):
            continue
        parts.append(os.path.join(inp_dir, name))
    if not parts:
        raise FileNotFoundError(f"在 replay_meta_dir 下没有找到 part_*.jsonl：{inp_dir}")
    rows: List[dict] = []
    for p in parts:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _index_rows(rows: List[dict]) -> Dict[str, Dict[Tuple[int, int], dict]]:
    """按 base task、clip_id、candidate_id 建索引，方便检查每组 K 候选是否完整。"""
    # 索引结构：base -> (clip_id, cand_id) -> obj
    out: Dict[str, Dict[Tuple[int, int], dict]] = defaultdict(dict)
    for obj in rows:
        traj_id = str(obj.get("traj_id", "") or "")
        if not traj_id:
            continue
        base = _traj_base(traj_id)
        clip_id = int(obj.get("grpo_clip_id", 0) or 0)
        cand_id = int(obj.get("candidate_id", -1) if obj.get("candidate_id", None) is not None else -1)
        if cand_id < 0:
            # 回退：从 _kXX 解析 candidate_id。
            if "_k" in traj_id:
                try:
                    cand_id = int(traj_id.split("_k", 1)[1].split("_", 1)[0])
                except Exception:
                    cand_id = -1
        if clip_id <= 0 or cand_id < 0:
            continue
        out[base][(clip_id, cand_id)] = obj
    return out


def _make_anchor(example_obj: dict, *, base: str, anchor_begin: int, anchor_end: int, fps: int) -> dict:
    """基于同任务样本构造一个 SFT anchor row，和 GRPO 候选混合训练。"""
    a = dict(example_obj)
    a["begin_frame_id"] = int(anchor_begin)
    a["end_frame_id"] = int(anchor_end)
    a["fps"] = int(fps)
    a["traj_id"] = f"{base}_sft"
    a["hybrid_role"] = "sft"
    # 清空 GRPO 专用字段，避免 SFT anchor 误用这些值。
    a["grpo_reward"] = 0.0
    a["grpo_old_logprob"] = 0.0
    a["grpo_ref_logprob"] = 0.0
    a["grpo_trace_files"] = []
    a["grpo_group_id"] = str(a.get("grpo_group_id", base))
    a["grpo_clip_id"] = 0
    return a


def main():
    """CLI 入口：把每个任务的 12 条 GRPO clip 候选加 1 条 SFT anchor 后重新分片。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_replay_meta_dir", required=True)
    ap.add_argument("--output_replay_meta_dir", required=True)
    ap.add_argument("--num_parts", type=int, default=8)
    ap.add_argument("--k", type=int, default=4, help="每个 clip 的候选数量")
    ap.add_argument("--num_clips", type=int, default=3)
    ap.add_argument("--add_sft_anchor", type=int, default=1)
    ap.add_argument("--anchor_begin", type=int, default=0)
    ap.add_argument("--anchor_end", type=int, default=48)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--pad_to_equal_groups", type=int, default=1, help="复制靠前的 groups，让各个 shard 的长度相同")
    args = ap.parse_args()

    rows = _read_all_parts(args.input_replay_meta_dir)
    idx = _index_rows(rows)
    bases = sorted(idx.keys())
    if not bases:
        raise RuntimeError("没有找到有效的 traj_id/grpo_clip_id/candidate_id 行")

    groups: List[List[dict]] = []
    missing = 0
    for base in bases:
        m = idx[base]
        ok = True
        lst: List[dict] = []
        for c in range(1, int(args.num_clips) + 1):
            for k in range(int(args.k)):
                obj = m.get((c, k), None)
                if obj is None:
                    ok = False
                    break
                o2 = dict(obj)
                o2["hybrid_role"] = "grpo"
                lst.append(o2)
            if not ok:
                break
        if not ok:
            missing += 1
            continue
        if int(args.add_sft_anchor) == 1:
            anchor = _make_anchor(lst[0], base=base, anchor_begin=args.anchor_begin, anchor_end=args.anchor_end, fps=args.fps)
            lst.append(anchor)
        groups.append(lst)

    if not groups:
        raise RuntimeError("没有构造出完整 groups（请检查 k/num_clips）")

    os.makedirs(args.output_replay_meta_dir, exist_ok=True)

    # 按 group 序号对 num_parts 取模来分片。
    shards: List[List[List[dict]]] = [[] for _ in range(int(args.num_parts))]
    for gi, g in enumerate(groups):
        shards[gi % int(args.num_parts)].append(g)

    # 可选补齐每个 shard 的 group 数，避免多卡按最短迭代数训练时丢掉独有 group。
    lens = [len(s) for s in shards]
    max_groups = max(lens)
    if int(args.pad_to_equal_groups) == 1 and max_groups > 0:
        # shard 为空时，从一个非空 shard 里选补齐来源（很少发生）。
        pad_pool: List[List[dict]] = []
        for s in shards:
            if len(s) > 0:
                pad_pool = s
                break
        for pi in range(int(args.num_parts)):
            while len(shards[pi]) < max_groups:
                # 从开头稳定复制，保证结果可复现。
                if len(shards[pi]) > 0:
                    shards[pi].append(shards[pi][0])
                elif pad_pool:
                    shards[pi].append(pad_pool[len(shards[pi]) % len(pad_pool)])
                else:
                    break

    # 写出 part 文件：保持 group-major 顺序；组内先 12 条 GRPO，再 1 条 SFT。
    total_out = 0
    for pi in range(int(args.num_parts)):
        out_p = os.path.join(args.output_replay_meta_dir, f"part_{pi:02d}.jsonl")
        with open(out_p, "w", encoding="utf-8") as f:
            for g in shards[pi]:
                for obj in g:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    total_out += 1

    print(
        json.dumps(
            {
                "input_dir": args.input_replay_meta_dir,
                "output_dir": args.output_replay_meta_dir,
                "bases_total": len(bases),
                "groups_complete": len(groups),
                "groups_missing": int(missing),
                "num_parts": int(args.num_parts),
                "groups_per_part": [len(s) for s in shards],
                "rows_out": int(total_out),
                "rows_per_group": len(groups[0]) if groups else 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
