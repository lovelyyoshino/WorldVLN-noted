#!/usr/bin/env python3
"""
把 WorldVLN 训练 checkpoint 导出成适合上传 Hugging Face Hub 的目录布局。

中文导读：
`shard_worldvln_checkpoint.py` 导出的是“一个 WorldVLN checkpoint 的分片版本”；
这个脚本更面向发布：它把 `trainer.gpt_fsdp` 和 `trainer.vae_local` 分别放进
`gpt/`、`vae/` 两个标准 safetensors 子目录，并额外生成：

- `README.md`：模型卡，说明这是 custom-code 权重；
- `load_weights.py`：加载 gpt/vae 两套 state_dict 的辅助函数；
- `export_manifest.json`：来源 checkpoint、训练元数据和分片清单；
- `.gitattributes`：让 safetensors 走 Git LFS。

因此，这个目录可以上传到 HF，但仍需要本项目代码先构建模型实例，再调用 helper 加载权重。
"""
from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open


TOP_LEVEL_META_KEYS = ("arch", "epoch", "iter", "acc_str", "g_it")
TRAINER_GROUPS = {
    "gpt": "trainer.gpt_fsdp",
    "vae": "trainer.vae_local",
}
MANIFEST_NAME = "export_manifest.json"


def parse_args() -> argparse.Namespace:
    """解析导出参数；默认只需要 input_path 和 --output-dir。"""
    parser = argparse.ArgumentParser(
        description="把 WorldVLN 训练 checkpoint 导出成适合上传 Hugging Face 的分片仓库目录布局。"
    )
    parser.add_argument("input_path", help="原始 WorldVLN `.pth` checkpoint 路径")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="要生成的目标仓库目录；最终可以把整个目录上传到 Hugging Face。",
    )
    parser.add_argument(
        "--max-shard-size",
        default="10GB",
        help="每个导出 safetensors 分片的最大大小，例如 10GB",
    )
    parser.add_argument(
        "--model-name",
        default="WorldVLN 主干模型",
        help="写入生成 README.md 的模型显示名",
    )
    parser.add_argument(
        "--license",
        default="other",
        help="Hugging Face 模型卡中的 license 字段，例如 mit / apache-2.0 / other",
    )
    parser.add_argument(
        "--repo-id",
        default="",
        help="可选：在生成 README 中提到的 Hugging Face repo id，例如 org/name",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出目录已存在，则先删除再重建",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="跳过导出后的逐 tensor 校验",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    """以 CPU/mmap 方式加载大 .pth；老 PyTorch 没有 mmap 参数时自动降级。"""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"期望 checkpoint 是 dict/Mapping，实际收到 {type(checkpoint)}")
    return checkpoint


def nested_get(obj: Mapping[str, Any], dotted_path: str) -> Any:
    """按 dotted path 从嵌套 checkpoint 里取子 state_dict。"""
    current: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"缺少 key 路径：{dotted_path}")
        current = current[part]
    return current


def is_tensor_state_dict(obj: Any) -> bool:
    """判断对象是否为纯 tensor state_dict，便于导出为 safetensors。"""
    return isinstance(obj, Mapping) and len(obj) > 0 and all(isinstance(v, torch.Tensor) for v in obj.values())


def to_jsonable(obj: Any) -> Any:
    """把 checkpoint 元数据转成 JSON 可写结构。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return repr(obj)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """准备目标 repo 目录。默认拒绝覆盖非空目录，避免误删已有发布产物。"""
    if output_dir.exists():
        if not overwrite:
            existing = list(output_dir.iterdir())
            if existing:
                raise FileExistsError(
                    f"输出目录已存在且非空：{output_dir}。"
                    "如需覆盖，请显式传入 --overwrite。"
                )
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def export_group(
    state_dict: Mapping[str, torch.Tensor],
    output_dir: Path,
    max_shard_size: str,
) -> dict[str, Any]:
    """
    将单个 tensor state_dict 写成 Hugging Face 兼容的 sharded safetensors 目录。

    GPT 和 VAE 分开导出，是为了让推理代码按各自模型实例分别 load_state_dict，
    避免一个混合大 dict 里还要手动拆 prefix。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    export_state = dict(state_dict)
    save_torch_state_dict(
        export_state,
        save_directory=output_dir,
        filename_pattern="model{suffix}.safetensors",
        force_contiguous=True,
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )

    files = sorted(p.name for p in output_dir.iterdir())
    shard_files = [name for name in files if name.endswith(".safetensors")]
    index_files = [name for name in files if name.endswith(".safetensors.index.json")]
    return {
        "tensor_count": len(state_dict),
        "shard_files": shard_files,
        "index_file": index_files[0] if index_files else "",
    }


def extract_alias_map_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    """从 safetensors metadata 中提取别名 key 到 canonical key 的映射。"""
    alias_map: dict[str, str] = {}
    if not isinstance(metadata, Mapping):
        return alias_map
    for key, value in metadata.items():
        if key == "format":
            continue
        if isinstance(key, str) and isinstance(value, str):
            alias_map[key] = value
    return alias_map


def iter_exported_tensors(export_dir: Path):
    """遍历一个分片目录里所有导出的 tensor key，用于逐 tensor 校验。"""
    for file_path in sorted(export_dir.glob("*.safetensors")):
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


def collect_export_metadata(export_dir: Path) -> tuple[set[str], dict[str, str]]:
    """收集分片目录中实际写出的 key，以及 safetensors metadata 记录的 alias 映射。"""
    exported_keys: set[str] = set()
    alias_map: dict[str, str] = {}

    index_files = sorted(export_dir.glob("*.safetensors.index.json"))
    if index_files:
        index_data = json.loads(index_files[0].read_text())
        exported_keys.update(index_data.get("weight_map", {}).keys())
        alias_map.update(extract_alias_map_from_metadata(index_data.get("metadata")))

    for file_path in sorted(export_dir.glob("*.safetensors")):
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            exported_keys.update(handle.keys())
            alias_map.update(extract_alias_map_from_metadata(handle.metadata()))

    return exported_keys, alias_map


def validate_group_export(name: str, original_state: Mapping[str, torch.Tensor], export_dir: Path) -> None:
    """
    导出后逐 key 校验 shape、dtype 和 tensor 内容。

    这个校验比较重，但对大权重发布很重要：只检查文件存在不能证明分片没有丢 key 或写坏。
    """
    print(f"[校验] 正在检查 {name}: {export_dir}")
    exported_keys, alias_map = collect_export_metadata(export_dir)
    expected_keys = set(original_state.keys())
    semantic_keys = set(exported_keys)

    for alias_key, canonical_key in alias_map.items():
        if canonical_key in expected_keys:
            semantic_keys.add(alias_key)

    missing = sorted(expected_keys - semantic_keys)
    extra = sorted(semantic_keys - expected_keys)
    if missing or extra:
        raise RuntimeError(
            f"{name} 导出的 key 不匹配：missing={missing[:10]} extra={extra[:10]}"
        )

    for key, exported_tensor in iter_exported_tensors(export_dir):
        original_tensor = original_state[key]
        if exported_tensor.shape != original_tensor.shape:
            raise RuntimeError(
                f"{name}:{key} 形状不匹配：exported={tuple(exported_tensor.shape)} "
                f"original={tuple(original_tensor.shape)}"
            )
        if exported_tensor.dtype != original_tensor.dtype:
            raise RuntimeError(
                f"{name}:{key} dtype 不匹配：exported={exported_tensor.dtype} original={original_tensor.dtype}"
            )
        if not torch.equal(exported_tensor, original_tensor):
            raise RuntimeError(f"{name}:{key} 导出后 tensor 内容不一致")

    for alias_key, canonical_key in alias_map.items():
        if alias_key in original_state and canonical_key in original_state:
            if not torch.equal(original_state[alias_key], original_state[canonical_key]):
                raise RuntimeError(
                    f"{name}：alias 映射 {alias_key} -> {canonical_key} 在原始 state_dict 中并不相等"
                )


def build_manifest(
    checkpoint: Mapping[str, Any],
    input_path: Path,
    output_dir: Path,
    model_name: str,
    license_name: str,
    repo_id: str,
    max_shard_size: str,
    exports: dict[str, Any],
) -> dict[str, Any]:
    """生成发布目录的 provenance：原始 checkpoint、训练元数据、每个子目录的分片信息。"""
    trainer = checkpoint.get("trainer", {})
    top_level_metadata = {key: to_jsonable(checkpoint.get(key)) for key in TOP_LEVEL_META_KEYS}
    top_level_metadata["args"] = to_jsonable(checkpoint.get("args"))
    trainer_metadata = {"config": to_jsonable(trainer.get("config"))}

    return {
        "format": "worldvln_hf_repo_v1",
        "model_name": model_name,
        "license": license_name,
        "repo_id": repo_id,
        "source_checkpoint": str(input_path),
        "output_dir": str(output_dir),
        "max_shard_size": max_shard_size,
        "top_level_metadata": top_level_metadata,
        "trainer_metadata": trainer_metadata,
        "exports": exports,
    }


def build_readme(manifest: Mapping[str, Any]) -> str:
    """生成 HF 模型卡；强调这是 custom-code 权重，不能直接 AutoModel 加载。"""
    model_name = manifest["model_name"]
    license_name = manifest["license"]
    repo_id = manifest.get("repo_id", "")
    source_checkpoint = manifest["source_checkpoint"]
    exports = manifest["exports"]
    arch = manifest["top_level_metadata"].get("arch", "")
    epoch = manifest["top_level_metadata"].get("epoch", "")
    iter_idx = manifest["top_level_metadata"].get("iter", "")
    g_it = manifest["top_level_metadata"].get("g_it", "")
    repo_line = f"- 建议使用的 Hugging Face repo id：`{repo_id}`\n" if repo_id else ""

    return textwrap.dedent(
        f"""\
        ---
        license: {license_name}
        library_name: pytorch
        tags:
        - custom-code
        - visual-navigation
        - worldvln
        - safetensors
        ---

        # {model_name}

        这个仓库目录由 WorldVLN 训练 checkpoint 导出，已经整理成 Hugging Face Hub 友好的布局。
        你可以把整个目录作为 Hugging Face model repo 的根目录上传。

        ## 包含的权重

        - `gpt/`：`trainer.gpt_fsdp` 的标准 sharded `safetensors` 导出结果
        - `vae/`：`trainer.vae_local` 的标准 sharded `safetensors` 导出结果
        - `load_weights.py`：直接加载这两个子目录的辅助函数
        - `{MANIFEST_NAME}`：导出来源、训练元数据和分片清单

        ## 来源 checkpoint

        - 原始 checkpoint：`{source_checkpoint}`
        - 模型结构：`{arch}`
        - Epoch：`{epoch}`
        - Iter：`{iter_idx}`
        - 全局步数：`{g_it}`
        {repo_line}
        ## 文件布局

        - `gpt/model.safetensors.index.json`
        - `gpt/model-00001-of-xxxxx.safetensors`
        - `vae/model.safetensors.index.json`
        - `vae/model-00001-of-xxxxx.safetensors`

        GPT 分片数量：`{len(exports["gpt"]["shard_files"])}`

        VAE 分片数量：`{len(exports["vae"]["shard_files"])}`

        ## 直接加载

        这个导出结果故意拆成两个模型目录，而不是继续保留一个混合训练 checkpoint。
        请先用本项目代码构建 GPT 模型和 VAE 模型实例，再分别加载权重。

        ```python
        from load_weights import load_worldvln_models

        load_worldvln_models(
            repo_dir=".",
            gpt_model=infinity_model,
            vae_model=vae_model,
            strict=False,
            device="cpu",
        )
        ```

        也可以只读取原始 state dict：

        ```python
        from load_weights import load_worldvln_state_dicts

        bundle = load_worldvln_state_dicts(".", device="cpu")
        gpt_state_dict = bundle["gpt"]
        vae_state_dict = bundle["vae"]
        ```

        ## 注意事项

        - 这是 custom-code 模型导出，不是可以直接用 `transformers.AutoModel.from_pretrained(...)` 加载的通用 repo。
        - 权重使用标准 sharded `safetensors` 格式，不需要手动拼接文件。
        - 在本项目里推理时，GPT 加载器指向 `gpt/`，VAE 加载器指向 `vae/`。
        """
    )


def build_helper_py() -> str:
    """
    生成随 repo 一起发布的加载辅助脚本。

    helper 不负责构造模型结构，只负责从 `gpt/`、`vae/` 读取 safetensors state_dict，
    然后把它们加载到调用方已经实例化好的 WorldVLN GPT/VAE 模型上。
    """
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        from __future__ import annotations

        import json
        from pathlib import Path
        from typing import Any

        import torch
        from safetensors import safe_open


        MANIFEST_NAME = "{MANIFEST_NAME}"


        def _alias_map_from_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
            alias_map: dict[str, str] = {{}}
            if not isinstance(metadata, dict):
                return alias_map
            for key, value in metadata.items():
                if key == "format":
                    continue
                if isinstance(key, str) and isinstance(value, str):
                    alias_map[key] = value
            return alias_map


        def load_sharded_state_dict(folder: str | Path, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
            folder = Path(folder)
            target_device = str(device)
            state_dict: dict[str, torch.Tensor] = {{}}
            alias_map: dict[str, str] = {{}}

            index_files = sorted(folder.glob("*.safetensors.index.json"))
            if index_files:
                index_data = json.loads(index_files[0].read_text())
                shard_names = list(dict.fromkeys(index_data["weight_map"].values()))
                alias_map.update(_alias_map_from_metadata(index_data.get("metadata")))
                for shard_name in shard_names:
                    shard_path = folder / shard_name
                    with safe_open(str(shard_path), framework="pt", device=target_device) as handle:
                        alias_map.update(_alias_map_from_metadata(handle.metadata()))
                        for key in handle.keys():
                            state_dict[key] = handle.get_tensor(key)
            else:
                safetensor_files = sorted(folder.glob("*.safetensors"))
                if len(safetensor_files) != 1:
                    raise FileNotFoundError(
                        f"{{folder}} 中期望有一个独立 .safetensors 文件或一个 index json，实际找到 {{len(safetensor_files)}} 个 .safetensors 文件"
                    )
                with safe_open(str(safetensor_files[0]), framework="pt", device=target_device) as handle:
                    alias_map.update(_alias_map_from_metadata(handle.metadata()))
                    for key in handle.keys():
                        state_dict[key] = handle.get_tensor(key)

            for alias_key, canonical_key in alias_map.items():
                if alias_key not in state_dict and canonical_key in state_dict:
                    state_dict[alias_key] = state_dict[canonical_key]
            return state_dict


        def load_worldvln_state_dicts(repo_dir: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
            repo_dir = Path(repo_dir)
            manifest_path = repo_dir / MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {{}}
            return {{
                "manifest": manifest,
                "gpt": load_sharded_state_dict(repo_dir / "gpt", device=device),
                "vae": load_sharded_state_dict(repo_dir / "vae", device=device),
            }}


        def load_worldvln_models(
            repo_dir: str | Path,
            *,
            gpt_model: torch.nn.Module | None = None,
            vae_model: torch.nn.Module | None = None,
            strict: bool = False,
            device: str | torch.device = "cpu",
        ) -> dict[str, Any]:
            bundle = load_worldvln_state_dicts(repo_dir, device=device)

            if gpt_model is not None:
                gpt_result = gpt_model.load_state_dict(bundle["gpt"], strict=strict)
            else:
                gpt_result = None

            if vae_model is not None:
                vae_result = vae_model.load_state_dict(bundle["vae"], strict=strict)
            else:
                vae_result = None

            bundle["gpt_load_result"] = gpt_result
            bundle["vae_load_result"] = vae_result
            return bundle
        """
    )


def write_support_files(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    """写 manifest、README、load_weights.py 和 Git LFS 配置。"""
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "README.md").write_text(build_readme(manifest), encoding="utf-8")
    (output_dir / "load_weights.py").write_text(build_helper_py(), encoding="utf-8")
    (output_dir / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.bin filter=lfs diff=lfs merge=lfs -text\n"
        "*.pt filter=lfs diff=lfs merge=lfs -text\n"
        "*.pth filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )


def main() -> None:
    """导出主流程：加载 checkpoint -> 拆 GPT/VAE -> 写分片 -> 写支持文件 -> 可选逐 tensor 校验。"""
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入 checkpoint：{input_path}")

    prepare_output_dir(output_dir, overwrite=args.overwrite)

    print(f"[1/6] 加载 checkpoint：{input_path}")
    checkpoint = load_checkpoint(input_path)

    trainer = checkpoint.get("trainer")
    if not isinstance(trainer, Mapping):
        raise KeyError("checkpoint 中不包含 `trainer` 映射")

    exports: dict[str, Any] = {}
    original_states: dict[str, Mapping[str, torch.Tensor]] = {}

    print("[2/6] 收集导出分组")
    for name, dotted_path in TRAINER_GROUPS.items():
        # TRAINER_GROUPS 是发布边界：只发布推理所需的 GPT/VAE 权重，不把 optimizer 等训练态写进 repo。
        state_dict = nested_get(checkpoint, dotted_path)
        if not is_tensor_state_dict(state_dict):
            raise ValueError(f"{dotted_path} 不是只包含 tensor 的 state_dict")
        original_states[name] = dict(state_dict)

    print(f"[3/6] 导出 GPT 分片 -> {output_dir / 'gpt'}")
    exports["gpt"] = export_group(original_states["gpt"], output_dir / "gpt", args.max_shard_size)
    exports["gpt"]["source_key"] = TRAINER_GROUPS["gpt"]

    print(f"[4/6] 导出 VAE 分片 -> {output_dir / 'vae'}")
    exports["vae"] = export_group(original_states["vae"], output_dir / "vae", args.max_shard_size)
    exports["vae"]["source_key"] = TRAINER_GROUPS["vae"]

    manifest = build_manifest(
        checkpoint=checkpoint,
        input_path=input_path,
        output_dir=output_dir,
        model_name=args.model_name,
        license_name=args.license,
        repo_id=args.repo_id,
        max_shard_size=args.max_shard_size,
        exports=exports,
    )

    print("[5/6] 写入仓库辅助文件")
    write_support_files(output_dir, manifest)

    if args.skip_validation:
        print("[6/6] 跳过校验")
    else:
        print("[6/6] 校验导出结果")
        validate_group_export("gpt", original_states["gpt"], output_dir / "gpt")
        validate_group_export("vae", original_states["vae"], output_dir / "vae")
        print("[validate] 全部检查通过")

    print("\n导出完成。")
    print(f"HF 仓库目录：{output_dir}")
    print(f"GPT 目录：{output_dir / 'gpt'}")
    print(f"VAE 目录：{output_dir / 'vae'}")
    print(f"模型卡：{output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
