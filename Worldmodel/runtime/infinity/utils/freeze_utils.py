from typing import Dict, List, Tuple


def _resolve_block_container(model) -> Tuple[object, str]:
    """中文说明：`_resolve_block_container` 实现StageB 局部冻结工具中的 `_resolve_block_container` 步骤，供训练、推理或调试流程复用。

    新手提示：阅读时关注 requires_grad 如何按模块名切换，以及冻结摘要如何帮助确认训练范围。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    if hasattr(model, 'block_chunks') and len(model.block_chunks) > 0:
        return model.block_chunks, 'block_chunks'
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        return model.blocks, 'blocks'
    raise AttributeError('partial freeze 需要模型包含 block_chunks 或 blocks')


def summarize_parameter_status(model, sample_limit: int = 12) -> Dict[str, object]:
    """中文说明：`summarize_parameter_status` 实现StageB 局部冻结工具中的 `summarize_parameter_status` 步骤，供训练、推理或调试流程复用。

    新手提示：阅读时关注 requires_grad 如何按模块名切换，以及冻结摘要如何帮助确认训练范围。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    summary = {
        'total_params': 0,
        'trainable_params': 0,
        'frozen_params': 0,
        'total_param_tensors': 0,
        'trainable_param_tensors': 0,
        'frozen_param_tensors': 0,
        'trainable_samples': [],
        'frozen_samples': [],
    }

    for name, para in model.named_parameters():
        summary['total_params'] += para.numel()
        summary['total_param_tensors'] += 1
        if para.requires_grad:
            summary['trainable_params'] += para.numel()
            summary['trainable_param_tensors'] += 1
            if len(summary['trainable_samples']) < sample_limit:
                summary['trainable_samples'].append(name)
        else:
            summary['frozen_params'] += para.numel()
            summary['frozen_param_tensors'] += 1
            if len(summary['frozen_samples']) < sample_limit:
                summary['frozen_samples'].append(name)

    return summary


def _print_partial_freeze_summary(summary: Dict[str, object]) -> None:
    """中文说明：`_print_partial_freeze_summary` 实现StageB 局部冻结工具中的 `_print_partial_freeze_summary` 步骤，供训练、推理或调试流程复用。

    新手提示：阅读时关注 requires_grad 如何按模块名切换，以及冻结摘要如何帮助确认训练范围。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    total_params = summary['total_params']
    trainable_params = summary['trainable_params']
    frozen_params = summary['frozen_params']
    trainable_ratio = (trainable_params / total_params) if total_params else 0.0
    frozen_ratio = (frozen_params / total_params) if total_params else 0.0

    print(
        '[partial-freeze] '
        f"启用={summary['enabled']} mode={summary['mode']} "
        f"freeze_chunk_prefix={summary['freeze_chunk_prefix']} "
        f"total_chunks={summary['total_chunks']} "
        f"可训练 chunks={summary['trainable_chunk_indices']}",
        flush=True,
    )
    print(
        '[partial-freeze] '
        f"总参数={total_params / 1e9:.4f}B 可训练={trainable_params / 1e9:.4f}B ({trainable_ratio:.2%}) "
        f"冻结={frozen_params / 1e9:.4f}B ({frozen_ratio:.2%})",
        flush=True,
    )
    print(
        '[partial-freeze] '
        f"可训练 tensors={summary['trainable_param_tensors']} 冻结 tensors={summary['frozen_param_tensors']}",
        flush=True,
    )
    print(f"[partial-freeze] 可训练样例={summary['trainable_samples']}", flush=True)
    print(f"[partial-freeze] 冻结样例={summary['frozen_samples']}", flush=True)


def apply_stageb_partial_freeze(model, freeze_chunk_prefix: int, print_summary: bool = True) -> Dict[str, object]:
    """中文说明：`apply_stageb_partial_freeze` 实现StageB 局部冻结工具中的 `apply_stageb_partial_freeze` 步骤，供训练、推理或调试流程复用。

    新手提示：阅读时关注 requires_grad 如何按模块名切换，以及冻结摘要如何帮助确认训练范围。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    block_container, block_container_name = _resolve_block_container(model)
    total_chunks = len(block_container)

    if freeze_chunk_prefix < 0:
        raise ValueError(f'freeze_chunk_prefix 必须 >= 0，实际为 {freeze_chunk_prefix}')
    if total_chunks <= 0:
        raise ValueError('partial freeze 至少需要一个 transformer block container')
    if freeze_chunk_prefix >= total_chunks:
        raise ValueError(
            f'freeze_chunk_prefix={freeze_chunk_prefix} 必须小于 total_chunks={total_chunks}，'
            '以便至少保留一个可训练 chunk'
        )

    if freeze_chunk_prefix == 0:
        summary = summarize_parameter_status(model)
        summary.update({
            'enabled': False,
            'mode': 'full-train',
            'freeze_chunk_prefix': 0,
            'total_chunks': total_chunks,
            'block_container_name': block_container_name,
            'trainable_chunk_indices': list(range(total_chunks)),
        })
        if print_summary:
            _print_partial_freeze_summary(summary)
        return summary

    model.requires_grad_(False)

    for chunk_idx in range(freeze_chunk_prefix, total_chunks):
        block_container[chunk_idx].requires_grad_(True)

    for module_name in ('norm_hidden_sates', 'head', 'semantic_head2'):
        if hasattr(model, module_name):
            getattr(model, module_name).requires_grad_(True)

    summary = summarize_parameter_status(model)
    summary.update({
        'enabled': True,
        'mode': 'chunk-prefix',
        'freeze_chunk_prefix': freeze_chunk_prefix,
        'total_chunks': total_chunks,
        'block_container_name': block_container_name,
        'trainable_chunk_indices': list(range(freeze_chunk_prefix, total_chunks)),
    })
    if print_summary:
        _print_partial_freeze_summary(summary)
    return summary
