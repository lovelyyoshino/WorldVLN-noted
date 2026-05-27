# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch
from collections import OrderedDict
import numpy as np

Tensor = torch.Tensor

def _prod(dims):
    """中文说明：`_prod` 实现FLOPs 计算公式实现中的 `_prod` 步骤，供训练、推理或调试流程复用。

    新手提示：常见公式是矩阵乘 FLOPs≈2*M*N*K，卷积 FLOPs≈2*out_elements*kernel_mul*in_channels/groups。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    p = 1
    for v in dims:
        p *= v
    return p

def matmul_flops_compute(input, other, *, out=None):
    """
    统计 matmul 操作的 FLOPs。
    """
    macs = _prod(input.shape) * other.shape[-1]
    return 2 * macs, macs


def addmm_flops_compute(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    统计 addmm 操作的 FLOPs。
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(input.shape), macs


def einsum_flops_compute(equation, *operands):
    """
    统计 einsum 操作的 FLOPs。
    """
    equation = equation.replace(" ", "")
    input_shapes = [o.shape for o in operands]

    # 重新映射公式，使不同字母表示的同一公式可以归一化比较
    # 归一化后表示形式保持一致。
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
    for line in optim.split("\n"):
        if "optimized flop" in line.lower():
            flop = int(float(line.split(":")[-1]))
            return flop, 0
    raise NotImplementedError("暂不支持这种 einsum 操作。")


def einops_einsum_flops_compute(*args):
    """
    统计 einops.einsum 操作的 FLOPs。
    """
    *operands, equation = args
    input_shapes = [o.shape for o in operands]

    # 重新映射公式，使不同字母表示的同一公式可以归一化比较
    # 归一化后表示形式保持一致。
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
    for line in optim.split("\n"):
        if "optimized flop" in line.lower():
            flop = int(float(line.split(":")[-1]))
            return flop, 0

    raise NotImplementedError("暂不支持这种 einops.einsum 操作。")


def tensor_addmm_flops_compute(self, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    统计 tensor.addmm 操作的 FLOPs。
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(self.shape), macs


def mul_flops_compute(input, other, *, out=None):
    """中文说明：`mul_flops_compute` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常见公式是矩阵乘 FLOPs≈2*M*N*K，卷积 FLOPs≈2*out_elements*kernel_mul*in_channels/groups。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    return elementwise_flops_compute(input, other)


def add_flops_compute(input, other, *, alpha=1, out=None):
    """中文说明：`add_flops_compute` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常见公式是矩阵乘 FLOPs≈2*M*N*K，卷积 FLOPs≈2*out_elements*kernel_mul*in_channels/groups。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    return elementwise_flops_compute(input, other)


def elementwise_flops_compute(input, other):
    """中文说明：`elementwise_flops_compute` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常见公式是矩阵乘 FLOPs≈2*M*N*K，卷积 FLOPs≈2*out_elements*kernel_mul*in_channels/groups。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return _prod(other.shape), 0
        else:
            return 1, 0
    elif not torch.is_tensor(other):
        return _prod(input.shape), 0
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)

        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = _prod(final_shape)
        return flops, 0
