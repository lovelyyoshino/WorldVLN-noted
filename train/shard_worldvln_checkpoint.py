#!/usr/bin/env python3
"""
把 WorldVLN 训练产生的单文件 .pth checkpoint 转成分片 safetensors，并支持再加载回原结构。

中文导读：
训练阶段保存的 checkpoint 通常是一个很大的 Python dict，其中 `trainer.vae_local`
和 `trainer.gpt_fsdp` 才是真正需要发布/推理加载的 tensor state_dict。这个脚本做三件事：

1. inspect：查看 .pth 顶层结构，确认有哪些 trainer 子权重。
2. export：把指定 dotted path 的 state_dict 展平成 `group.key`，保存为 safetensors 分片。
3. load：读取分片和 manifest，再按 dotted path 还原成近似原始 checkpoint dict。

注意：这里导出的不是 Hugging Face `AutoModel` 可直接加载的标准 transformers 模型，
而是方便大权重传输和复原的 WorldVLN 自定义 checkpoint 格式。
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import save_torch_state_dict
from safetensors.torch import load_file as load_safetensors_file


DEFAULT_GROUPS = ("trainer.vae_local", "trainer.gpt_fsdp")
MANIFEST_NAME = "export_manifest.json"


def is_tensor_state_dict(obj: Any) -> bool:
    """判断一个对象是否是纯 tensor state_dict。非 tensor 元数据不能直接写进 safetensors。"""
    return isinstance(obj, Mapping) and len(obj) > 0 and all(isinstance(v, torch.Tensor) for v in obj.values())


def to_jsonable(obj: Any) -> Any:
    """把 checkpoint 中的 Path、dict、list 等元数据递归转成 JSON 可写对象。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return repr(obj)


def nested_get(obj: Mapping[str, Any], dotted_path: str) -> Any:
    """按 `trainer.gpt_fsdp` 这样的 dotted path 从嵌套 checkpoint dict 里取值。"""
    current: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"缺少 key 路径：{dotted_path}")
        current = current[part]
    return current


def nested_set(obj: dict[str, Any], dotted_path: str, value: Any) -> None:
    """`load` 子命令使用：把展平读取的 group state_dict 放回嵌套路径。"""
    parts = dotted_path.split(".")
    current = obj
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def load_torch_checkpoint(path: Path) -> Any:
    """以 CPU/mmap 方式加载原始 .pth checkpoint；旧 PyTorch 自动降级。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def build_manifest(checkpoint: Mapping[str, Any], source_path: Path, groups: list[str]) -> dict[str, Any]:
    """
    记录导出的来源、分组和少量训练元数据。

    safetensors 文件只保存 tensor；epoch、args、arch 等 Python 元数据需要放到 manifest，
    否则后续无法判断这批分片来自哪个训练 run。
    """
    trainer = checkpoint.get("trainer", {})
    manifest = {
        "format": "worldvln_sharded_safetensors_v1",
        "source_checkpoint": str(source_path),
        "exported_groups": groups,
        "top_level_metadata": {
            "args": to_jsonable(checkpoint.get("args")),
            "arch": to_jsonable(checkpoint.get("arch")),
            "epoch": to_jsonable(checkpoint.get("epoch")),
            "iter": to_jsonable(checkpoint.get("iter")),
            "acc_str": to_jsonable(checkpoint.get("acc_str")),
            "g_it": to_jsonable(checkpoint.get("g_it")),
        },
        "trainer_metadata": {
            "config": to_jsonable(trainer.get("config")),
        },
    }
    return manifest


def flatten_export_groups(checkpoint: Mapping[str, Any], groups: list[str]) -> OrderedDict[str, torch.Tensor]:
    """
    把多个嵌套 state_dict 合并成一个扁平 OrderedDict。

    例如 `trainer.gpt_fsdp.blocks.0...` 和 `trainer.vae_local.encoder...`
    会保留 group 前缀，避免不同子模块里相同 key 发生冲突。
    """
    merged_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for group_path in groups:
        state_dict = nested_get(checkpoint, group_path)
        if not is_tensor_state_dict(state_dict):
            raise ValueError(f"{group_path} 不是纯 tensor state_dict")
        prefix = f"{group_path}."
        for key, tensor in state_dict.items():
            merged_key = f"{prefix}{key}"
            if merged_key in merged_state:
                raise KeyError(f"合并后的 key 重复：{merged_key}")
            merged_state[merged_key] = tensor.detach().contiguous()
    return merged_state


def autodetect_index_file(checkpoint_dir: Path) -> Path:
    """在分片目录中定位唯一的 safetensors index json。"""
    matches = sorted(checkpoint_dir.glob("*.safetensors.index.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"期望在 {checkpoint_dir} 中找到且只找到一个 '*.safetensors.index.json'，实际找到 {len(matches)} 个"
        )
    return matches[0]


def device_string(device: str | torch.device) -> str:
    """把 torch.device 或字符串统一成 safetensors loader 接受的 device 字符串。"""
    if isinstance(device, torch.device):
        return str(device)
    return device


def load_flat_sharded_state(checkpoint_dir: Path, device: str | torch.device = "cpu") -> OrderedDict[str, torch.Tensor]:
    """
    读取 export 生成的 safetensors 分片。

    `save_torch_state_dict` 可能生成 index json，也可能在权重较小时只生成单个 safetensors；
    这里同时兼容两种布局，并补回 index metadata 中记录的 alias tensor。
    """
    merged_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    target_device = device_string(device)

    index_files = sorted(checkpoint_dir.glob("*.safetensors.index.json"))
    if index_files:
        index_path = autodetect_index_file(checkpoint_dir)
        index_data = json.loads(index_path.read_text())
        shard_names = list(dict.fromkeys(index_data["weight_map"].values()))
        for shard_name in shard_names:
            shard_path = checkpoint_dir / shard_name
            shard_state = load_safetensors_file(str(shard_path), device=target_device)
            merged_state.update(shard_state)

        # 说明：`save_torch_state_dict` 在多个 key 共享同一份 tensor 内存时，
        # 可能只存储一份并把别名写进 index metadata，形式为 {"alias_key": "canonical_key"}。
        # 这里把这些别名补回 merged_state，保证后续按原始 key 查询时不会缺失。
        for alias_key, canonical_key in index_data.get("metadata", {}).items():
            if (
                isinstance(alias_key, str)
                and isinstance(canonical_key, str)
                and alias_key not in merged_state
                and canonical_key in merged_state
            ):
                merged_state[alias_key] = merged_state[canonical_key]
        return merged_state

    safetensor_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if len(safetensor_files) != 1:
        raise FileNotFoundError(
            f"期望在 {checkpoint_dir} 中找到一个独立的 '*.safetensors' 文件，实际找到 {len(safetensor_files)} 个"
        )
    merged_state.update(load_safetensors_file(str(safetensor_files[0]), device=target_device))
    return merged_state


def load_sharded_worldvln_checkpoint(checkpoint_dir: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """
    将分片 safetensors + manifest 还原为 WorldVLN 风格 checkpoint dict。

    这个函数主要用于离线验证、迁移或重新打包；训练/推理真正加载模型时，通常会只取
    `checkpoint["trainer"]["gpt_fsdp"]` 和 `checkpoint["trainer"]["vae_local"]`。
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少 manifest 文件：{manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    exported_groups = manifest.get("exported_groups", [])
    flat_state = load_flat_sharded_state(checkpoint_dir, device=device)

    checkpoint: dict[str, Any] = dict(manifest.get("top_level_metadata", {}))
    trainer_metadata = manifest.get("trainer_metadata", {})
    if trainer_metadata:
        checkpoint["trainer"] = dict(trainer_metadata)

    for group_path in exported_groups:
        prefix = f"{group_path}."
        group_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, tensor in flat_state.items():
            if key.startswith(prefix):
                group_state[key[len(prefix):]] = tensor
        if not group_state:
            raise KeyError(f"没有找到 group={group_path} 对应的 tensor")
        nested_set(checkpoint, group_path, group_state)

    return checkpoint


def export_checkpoint(
    input_path: Path,
    output_dir: Path,
    prefix: str,
    max_shard_size: str,
    groups: list[str],
) -> None:
    """执行 export 子命令：加载 .pth、展平指定 group、写 safetensors 和 manifest。"""
    print(f"[1/4] 加载 checkpoint：{input_path}")
    checkpoint = load_torch_checkpoint(input_path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"期望 checkpoint 是 dict/Mapping，实际收到 {type(checkpoint)}")

    print(f"[2/4] 收集导出分组：{', '.join(groups)}")
    merged_state = flatten_export_groups(checkpoint, groups)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] 把 safetensors 分片写入：{output_dir}")
    save_torch_state_dict(
        merged_state,
        save_directory=output_dir,
        filename_pattern=f"{prefix}{{suffix}}.safetensors",
        force_contiguous=True,
        max_shard_size=max_shard_size,
        metadata={"source_checkpoint": input_path.name},
        safe_serialization=True,
    )

    manifest = build_manifest(checkpoint, input_path, groups)
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print("[4/4] 完成。")
    print(f"        manifest：{output_dir / MANIFEST_NAME}")
    for path in sorted(output_dir.glob(f"{prefix}*.safetensors*")):
        print(f"        文件：{path.name}")


def inspect_checkpoint(input_path: Path) -> None:
    """打印 checkpoint 顶层 key 和 trainer 子 key，先用它确认 `--groups` 应该填什么。"""
    checkpoint = load_torch_checkpoint(input_path)
    print(f"checkpoint 类型={type(checkpoint)}")
    if not isinstance(checkpoint, Mapping):
        return

    print(f"顶层 keys={list(checkpoint.keys())}")
    trainer = checkpoint.get("trainer")
    if isinstance(trainer, Mapping):
        print(f"trainer 子 keys={list(trainer.keys())}")
        for key, value in trainer.items():
            if is_tensor_state_dict(value):
                print(f"trainer.{key}: tensor_state_dict，包含 {len(value)} 个 tensor")
            else:
                print(f"trainer.{key}: {type(value)}")


def cmd_export(args: argparse.Namespace) -> None:
    """CLI export 子命令入口。"""
    export_checkpoint(
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
        max_shard_size=args.max_shard_size,
        groups=list(args.groups),
    )


def cmd_load(args: argparse.Namespace) -> None:
    """CLI load 子命令入口：读取分片并可选重新打包为 .pth。"""
    checkpoint = load_sharded_worldvln_checkpoint(args.checkpoint_dir, device=args.device)

    print("已加载分片 checkpoint。")
    print(f"顶层 keys={list(checkpoint.keys())}")
    trainer = checkpoint.get("trainer", {})
    if isinstance(trainer, Mapping):
        for key, value in trainer.items():
            if is_tensor_state_dict(value):
                print(f"trainer.{key}: tensor_state_dict，包含 {len(value)} 个 tensor")
            else:
                print(f"trainer.{key}: {type(value)}")

    if args.save_pth:
        save_path = Path(args.save_pth)
        torch.save(checkpoint, save_path)
        print(f"重新打包后的 checkpoint 已保存到：{save_path}")


def cmd_inspect(args: argparse.Namespace) -> None:
    """CLI inspect 子命令入口。"""
    inspect_checkpoint(Path(args.input_path))


def build_parser() -> argparse.ArgumentParser:
    """构建 inspect/export/load 三个子命令的 argparse parser。"""
    # 三个子命令对应“看结构 -> 导出 -> 验证加载”的完整迁移流程。
    parser = argparse.ArgumentParser(
        description="把 WorldVLN 风格的单文件 .pth checkpoint 导出成分片 safetensors，并支持再次加载回原始层级结构。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="查看 checkpoint 顶层和 trainer 子结构")
    inspect_parser.add_argument("input_path", help="原始 .pth checkpoint 路径")
    inspect_parser.set_defaults(func=cmd_inspect)

    export_parser = subparsers.add_parser("export", help="把选定权重组导出成 safetensors 分片")
    export_parser.add_argument("input_path", help="原始 .pth checkpoint 路径")
    export_parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录；会写入 safetensors 分片、index json 和 manifest",
    )
    export_parser.add_argument("--prefix", default="WorldVLN_backbone", help="分片文件名前缀")
    export_parser.add_argument("--max-shard-size", default="10GB", help="单个分片的最大大小，例如 10GB")
    export_parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="checkpoint 内部要导出的 tensor state_dict 组的 dotted key 路径",
    )
    export_parser.set_defaults(func=cmd_export)

    load_parser = subparsers.add_parser("load", help="把分片导出结果重新加载回一个 checkpoint dict")
    load_parser.add_argument("checkpoint_dir", help="包含分片 safetensors 导出结果的目录")
    load_parser.add_argument("--device", default="cpu", help="把 tensor 加载到哪个设备上")
    load_parser.add_argument("--save-pth", default="", help="可选：把重新组装后的 checkpoint 再打包成 .pth 的保存路径")
    load_parser.set_defaults(func=cmd_load)

    return parser


def main() -> None:
    """命令行入口：解析子命令并分发到对应处理函数。"""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
