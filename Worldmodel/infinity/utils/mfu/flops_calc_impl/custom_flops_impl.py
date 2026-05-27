# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch

def custom_rmsnorm_forward_hook(module, args, kwargs, output):
    """中文说明：`custom_rmsnorm_forward_hook` 实现FLOPs 计算公式实现中的 `custom_rmsnorm_forward_hook` 步骤，供训练、推理或调试流程复用。

    新手提示：常见公式是矩阵乘 FLOPs≈2*M*N*K，卷积 FLOPs≈2*out_elements*kernel_mul*in_channels/groups。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    if module.training and not torch.is_grad_enabled():
        return

    flops = 0
    hidden_states = args[0]
    if len(hidden_states.shape) == 2:
            # 变分辨率 NaViT 模式
            bsz = 1
            seq_len = hidden_states.shape[0]
    else:
            bsz = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]

    flops = bsz * seq_len * (2 * getattr(module, "hidden_size") + 1) * 2
    module.__flops__ += int(flops) * (3 if module.training else 1)

def custom_goku_attention_forward_hook(module, args, kwargs, output):
    """中文说明：`custom_goku_attention_forward_hook` 实现FLOPs 计算公式实现中的 `custom_goku_attention_forward_hook` 步骤，供训练、推理或调试流程复用。

    新手提示：常见公式是矩阵乘 FLOPs≈2*M*N*K，卷积 FLOPs≈2*out_elements*kernel_mul*in_channels/groups。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    if module.training and not torch.is_grad_enabled():
        return

    flops = 0
    inputs_q = kwargs["inputs_q"]
    inputs_kv = kwargs["inputs_kv"] if kwargs["inputs_kv"] is not None else inputs_q

    if len(inputs_q.shape) == 2:
        # 变分辨率 NaViT 模式
        q_bsz = kv_bsz = 1
        q_len = inputs_q.shape[0]
        kv_len = inputs_kv.shape[0]

        cu_seqlens_q = kwargs["cu_seqlens_q"].to(torch.int64).cpu().numpy()
        cu_seqlens_k = kwargs["cu_seqlens_k"].to(torch.int64).cpu().numpy()

        attn_seq_coef = 0
        for i in range(len(cu_seqlens_q) - 1):
            seqlen_q = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
            seqlen_k = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
            attn_seq_coef += seqlen_q * seqlen_k
    else:
        q_bsz = inputs_q.shape[0]
        q_len = inputs_q.shape[1]
        kv_bsz = inputs_kv.shape[0]
        kv_len = inputs_kv.shape[1]
        attn_seq_coef = q_len * kv_len

    sp_size = getattr(module, "sequence_parallel_size", 1) or 1
    num_heads = getattr(module, "num_heads")
    head_dim = getattr(module, "head_dim")

    flops = q_bsz * num_heads * attn_seq_coef * head_dim * 2 * 2 // sp_size

    module.__flops__ += int(flops) * (3 if module.training else 1)

def custom_flex_attention_forward_hook(module, args, kwargs, output):
    """中文说明：`custom_flex_attention_forward_hook` 实现FLOPs 计算公式实现中的 `custom_flex_attention_forward_hook` 步骤，供训练、推理或调试流程复用。

    新手提示：常见公式是矩阵乘 FLOPs≈2*M*N*K，卷积 FLOPs≈2*out_elements*kernel_mul*in_channels/groups。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    if module.training and not torch.is_grad_enabled():
        return

    flops = 0

    q = args[0]
    k = args[1]

    q_bs, q_head, q_len ,q_dim = q.shape
    kv_bs, kv_head, kv_len ,kv_dim = k.shape

    block_mask = getattr(module, "block_mask")
    density = 1
    if block_mask:
        # 参考：https://gist.github.com/Chillee/2e270fc5413dbbce58c779f8c4eac66c
        density = (100 - block_mask.sparsity())/100

    flops = density * q_bs * q_head * q_dim * q_len * kv_len * 2 * 2

    module.__flops__ += int(flops) * (3 if module.training else 1)


CUSTOM_HOOK_MAPPING = {}
CUSTOM_NAME_MAPPING = {}

try:
    from infinity.models.flex_attn import FlexAttn
    CUSTOM_HOOK_MAPPING[FlexAttn] = custom_flex_attention_forward_hook
except:
    print(f"[WARN] 无法导入自定义模块：FlexAttn")
