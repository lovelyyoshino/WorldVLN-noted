#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把来源不统一的任务 JSON 归一化成 GRPO rollout task jsonl。

中文导读：
这个脚本处在 GRPO 数据准备链的第一段。上游可能给你的是 list、dict、`items`、
`data` 等不同格式，字段名也可能不统一；这里的目标是统一成下游 rollout/视频数据集
都能直接读取的 jsonl，每行描述一个“待生成的 clip 任务”。

输出行的关键字段：
- `video_path`, `begin_frame_id`, `end_frame_id`, `fps`
- `tarsier2_caption`
- `grpo_reward`, `grpo_reward_act`, `grpo_reward_task`
- `grpo_old_logprob`
- `grpo_group_id`, `grpo_clip_id`
- 可选 `gt_pose_json`，供 reward 计算或评估使用

阅读顺序建议：
1. 先看 `_read_any_json()`，理解输入 JSON 能兼容哪些外层包装；
2. 再看 `_norm_item()`，理解每个标准字段从哪些候选 key 回退得到；
3. 最后看 `main()`，理解哪些行会被跳过、输出路径怎么写出。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List


def _read_any_json(path: str) -> List[Dict[str, Any]]:
    """读取 list/dict/items/data 多种任务 JSON 格式，统一返回 dict 列表。"""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        if isinstance(obj.get("items"), list):
            return [x for x in obj["items"] if isinstance(x, dict)]
        if isinstance(obj.get("data"), list):
            return [x for x in obj["data"] if isinstance(x, dict)]
        return [obj]
    raise ValueError(f"暂不支持的 JSON 外层格式：{type(obj)}")


def _pick(d: Dict[str, Any], keys: Iterable[str], default=None):
    """按候选 key 顺序取第一个存在且非 None 的字段。"""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _norm_item(x: Dict[str, Any], default_fps: int, fallback_group_id: str) -> Dict[str, Any]:
    """
    把不同来源任务字段归一化为 rollout jsonl 的标准字段。

    字段映射优先级：
    - `video_path`: `video_path -> source_video -> video -> path`
    - `caption`: `tarsier2_caption -> instruction_unified -> instruction -> caption -> prompt`
    - `begin/end`: `begin_frame_id/start_frame/start` 与 `end_frame_id/end_frame/end`
    - `gt_pose_json`: `preprocessed_logs_json -> gt_pose_json -> pose_json -> coordinates_path`
    - `grpo_group_id`: `grpo_group_id -> group_id`，若都没有则退回 `fallback_group_id`

    这样做的原因是：
    不同数据准备脚本对“视频路径、指令、GT pose、group id”的命名并不统一；
    这里集中做回退，避免下游每个脚本都各写一套兼容逻辑。
    """
    video_path = _pick(x, ("video_path", "source_video", "video", "path"), "")
    caption = _pick(x, ("tarsier2_caption", "instruction_unified", "instruction", "caption", "prompt"), "")
    begin = int(_pick(x, ("begin_frame_id", "start_frame", "start"), 0))
    end = int(_pick(x, ("end_frame_id", "end_frame", "end"), max(begin, begin + 48)))
    fps = int(_pick(x, ("fps", "video_fps"), default_fps))
    gt_pose_json = _pick(x, ("preprocessed_logs_json", "gt_pose_json", "pose_json", "coordinates_path"), "")
    group_id = _pick(x, ("grpo_group_id", "group_id"), "")
    clip_id = int(_pick(x, ("grpo_clip_id", "clip_id"), 1))
    if not group_id:
        group_id = str(fallback_group_id)
    out = {
        "video_path": str(video_path),
        "begin_frame_id": int(begin),
        "end_frame_id": int(end),
        "fps": int(fps),
        "tarsier2_caption": str(caption),
        "grpo_reward": float(_pick(x, ("grpo_reward", "reward"), 0.0)),
        "grpo_reward_act": float(_pick(x, ("grpo_reward_act", "reward_act", "grpo_reward", "reward"), 0.0)),
        "grpo_reward_task": float(_pick(x, ("grpo_reward_task", "reward_task"), 0.0)),
        "grpo_old_logprob": float(_pick(x, ("grpo_old_logprob", "old_logprob"), 0.0)),
        "grpo_group_id": str(group_id),
        "grpo_clip_id": int(clip_id),
    }
    if gt_pose_json:
        out["gt_pose_json"] = str(gt_pose_json)
    return out


def main():
    """
    CLI 入口：把源任务 JSON 转成 GRPO rollout task jsonl。

    注意：
    - 缺少核心字段 `video_path` 或 `tarsier2_caption` 的样本会被跳过；
    - 归一化异常的样本也会被跳过；
    - 因此“脚本成功执行”不代表“所有输入样本都保留了”，要关注最后的统计输出。
    """
    ap = argparse.ArgumentParser(
        description=(
            "把来源不统一的任务 JSON 归一化成 GRPO rollout task jsonl。"
            "输出每一行都对应一个可供下游 rollout/视频数据集直接读取的 clip 任务。"
        )
    )
    ap.add_argument("--input_json", type=str, required=True, help="输入任务 JSON 路径；外层可为 list、dict、items 或 data。")
    ap.add_argument("--output_jsonl", type=str, required=True, help="输出 jsonl 路径；每行一个归一化后的 rollout task。")
    ap.add_argument("--default_fps", type=int, default=16, help="当输入样本没有 `fps/video_fps` 字段时使用的默认帧率。")
    args = ap.parse_args()

    items = _read_any_json(os.path.abspath(args.input_json))
    lines: List[Dict[str, Any]] = []
    skipped_missing_core = 0
    skipped_exception = 0
    for i, x in enumerate(items):
        try:
            y = _norm_item(x, args.default_fps, fallback_group_id=f"task_{i:06d}")
            if not y["video_path"] or not y["tarsier2_caption"]:
                skipped_missing_core += 1
                continue
            lines.append(y)
        except Exception:
            skipped_exception += 1
            continue

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    with open(os.path.abspath(args.output_jsonl), "w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        "[build_rollout_tasks] "
        f"输入条数={len(items)} 保留={len(lines)} "
        f"缺少核心字段跳过={skipped_missing_core} 异常跳过={skipped_exception} "
        f"输出={os.path.abspath(args.output_jsonl)}"
    )


if __name__ == "__main__":
    main()
