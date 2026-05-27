# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

# 来源：https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/profiling/flops_profiler

# 来源：DeepSpeed 团队
import os
import time
import torch
import torch.nn.functional as F
import logging
from functools import partial
import einops

from .flops_calc_impl.func_flops_impl import *
from .flops_calc_impl.nn_flops_impl import *
from .flops_calc_impl.tensor_flops_impl import *
from .flops_calc_impl.custom_flops_impl import *

logger = logging.getLogger(__name__)

old_functions = {}

DEFAULT_PRECISION = 2

class FlopsProfiler(object):
    """中文说明：统计 PyTorch 模型中每个模块的延迟、估算 FLOPs 和参数量。

        FLOPs profiler 会统计 PyTorch 模型 forward 过程，并把每个模块的延迟、FLOPs 和参数量挂到模型图上，帮助定位瓶颈。它还能按用户指定的深度和 top-k 输出累计延迟、FLOPs、参数量最高的模块。统计结果按每个输入 batch 计算。
        DeepSpeed FLOPs profiler 既可以配合 DeepSpeed runtime 使用，也可以作为独立工具使用。
        使用 DeepSpeed 训练时，可以在 `deepspeed_config` 中配置 FLOPs profiler，无需改用户训练代码。

        如果作为独立工具使用，导入 `flops_profiler` 包并调用对应 API 即可。

        下面是典型训练流程中的使用示例：

            说明：.. code-block:: python

                公式/形状说明：model = Model()
                公式/形状说明：prof = FlopsProfiler(model)

                公式/形状说明：for step, batch in enumerate(data_loader):
                    公式/形状说明：if step == profile_step:
                        公式/形状说明：prof.start_profile()

                    公式/形状说明：loss = model(batch)

                    公式/形状说明：if step == profile_step:
                        公式/形状说明：flops = prof.get_total_flops()
                        公式/形状说明：prof.end_profile()

                    公式/形状说明：loss.backward()
                    公式/形状说明：optimizer.step()

        如果要在推理阶段统计已训练模型，可使用 `get_model_profile` API。

        参数：
            object (torch.nn.Module)：需要统计的 PyTorch 模型。

    """

    def __init__(self):
        """中文说明：`__init__` 初始化运行时 FLOPs profiler需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
        关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        self.models = []
        self.started = False
        self.func_patched = False
        self.module_flop_count = []
        self.detail_flops = ""

    def append(self, model):
        """中文说明：`append` 实现运行时 FLOPs profiler中的 `append` 步骤，供训练、推理或调试流程复用。

        新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
        关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        self.models.append(model)

    def start_profile(self, ignore_list=None):
        """中文说明：开始性能统计。

        递归给所有模块添加统计属性，并 monkey patch 需要统计的 `torch.nn.functional` 函数。

        参数：
            ignore_list (list, optional)：统计时要忽略的模块列表，默认 None。
        """
        self.ignore_list = ignore_list
        self.reset_profile()
        _patch_functionals(self.module_flop_count)
        _patch_tensor_methods(self.module_flop_count)
        _patch_miscellaneous_operations(self.module_flop_count)

        def register_module_hooks(module, ignore_list):
            """中文说明：`register_module_hooks` 实现运行时 FLOPs profiler中的 `register_module_hooks` 步骤，供训练、推理或调试流程复用。

            新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
            关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
            """
            if ignore_list and type(module) in ignore_list:
                return

            # 如果直接计算某个模块 module 的 FLOPs
            if type(module) in MODULE_HOOK_MAPPING:
                if not hasattr(module, "__flops_handle__"):
                    module.__flops_handle__ = module.register_forward_hook(MODULE_HOOK_MAPPING[type(module)])
                return

            if type(module) in CUSTOM_HOOK_MAPPING:
                if not hasattr(module, "__flops_handle__"):
                    module.__flops_handle__ = module.register_forward_hook(CUSTOM_HOOK_MAPPING[type(module)], with_kwargs=True)
                return

            # 如果计算模块 module 内 torch.nn.functional 调用的 FLOPs
            def pre_hook(module, input):
                """中文说明：`pre_hook` 实现运行时 FLOPs profiler中的 `pre_hook` 步骤，供训练、推理或调试流程复用。

                新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
                关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
                """
                self.module_flop_count.append([])

            if not hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__ = module.register_forward_pre_hook(pre_hook)

            def post_hook(module, input, output):
                """中文说明：`post_hook` 实现运行时 FLOPs profiler中的 `post_hook` 步骤，供训练、推理或调试流程复用。

                新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
                关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
                """
                if self.module_flop_count:

                    if torch.is_grad_enabled():
                        module.__flops__ += sum([elem[1] for elem in self.module_flop_count[-1]]) * (3 if module.training else 1)

                    self.module_flop_count.pop()

            if not hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__ = module.register_forward_hook(post_hook)


        for model in self.models:
            model.apply(partial(register_module_hooks, ignore_list=ignore_list))

        self.started = True
        self.func_patched = True
        logger.info("FLOPs 性能统计器已开始")

    def stop_profile(self):
        """中文说明：停止性能统计。

        把所有被 patch 的 `torch.nn.functional` 函数恢复成原始实现。
        """
        self.module_flop_count.clear()
        if self.started and self.func_patched:
            _reload_functionals()
            _reload_tensor_methods()
            _reload_miscellaneous_operations()
            self.func_patched = False

        def remove_profile_attrs(module):
            """中文说明：`remove_profile_attrs` 实现运行时 FLOPs profiler中的 `remove_profile_attrs` 步骤，供训练、推理或调试流程复用。

            新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
            关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
            """
            if hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__.remove()
                del module.__pre_hook_handle__
            if hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__.remove()
                del module.__post_hook_handle__
            if hasattr(module, "__flops_handle__"):
                module.__flops_handle__.remove()
                del module.__flops_handle__

        for model in self.models:
            model.apply(remove_profile_attrs)

    def reset_profile(self):
        """中文说明：重置性能统计状态。

        添加或重置统计用的额外属性。
        """
        self.module_flop_count.clear()
        def add_or_reset_attrs(module):
            """中文说明：`add_or_reset_attrs` 实现运行时 FLOPs profiler中的 `add_or_reset_attrs` 步骤，供训练、推理或调试流程复用。

            新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
            关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
            """
            module.__flops__ = 0

        for model in self.models:
            model.apply(add_or_reset_attrs)

    def end_profile(self):
        """中文说明：结束性能统计并恢复被 patch 的函数。

        递归移除所有模块上的统计属性和 hook 句柄。
        """
        if not self.started:
            return
        self.stop_profile()
        self.started = False
        self.module_flop_count.clear()

        def remove_profile_attrs(module):
            """中文说明：`remove_profile_attrs` 实现运行时 FLOPs profiler中的 `remove_profile_attrs` 步骤，供训练、推理或调试流程复用。

            新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
            关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
            """
            if hasattr(module, "__flops__"):
                del module.__flops__

        for model in self.models:
            model.apply(remove_profile_attrs)
        logger.info("FLOPs 性能统计器已结束")

    def get_total_flops(self):
        """中文说明：返回模型总 FLOPs。

        返回：
            模型前向传播中的乘加操作数量。
        """
        total_flops = 0
        self.detail_flops = ""
        for model in self.models:
            flops, log = get_module_flops(model, prefix="")
            total_flops += flops
            self.detail_flops += log
        return total_flops, self.detail_flops

def wrapFunc(func, funcFlopCompute, module_flop_count):
    """中文说明：`wrapFunc` 实现运行时 FLOPs profiler中的 `wrapFunc` 步骤，供训练、推理或调试流程复用。

    新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    oldFunc = func
    name = func.__str__
    old_functions[name] = oldFunc

    @torch.compiler.disable()
    def newFunc(*args, **kwds):
        """中文说明：`newFunc` 实现运行时 FLOPs profiler中的 `newFunc` 步骤，供训练、推理或调试流程复用。

        新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
        关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        flops, macs = funcFlopCompute(*args, **kwds)
        if module_flop_count:
            module_flop_count[-1].append((name, flops, func.__name__))
        return oldFunc(*args, **kwds)

    newFunc.__str__ = func.__str__

    return newFunc


def _patch_functionals(module_flop_count):
    # 全连接层
    """中文说明：`_patch_functionals` 实现运行时 FLOPs profiler中的 `_patch_functionals` 步骤，供训练、推理或调试流程复用。

    新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    F.linear = wrapFunc(F.linear, linear_flops_compute, module_flop_count)

    # 卷积层
    F.conv1d = wrapFunc(F.conv1d, conv_flops_compute, module_flop_count)
    F.conv2d = wrapFunc(F.conv2d, conv_flops_compute, module_flop_count)
    F.conv3d = wrapFunc(F.conv3d, conv_flops_compute, module_flop_count)

    # 转置卷积
    F.conv_transpose1d = wrapFunc(F.conv_transpose1d, conv_trans_flops_compute, module_flop_count)
    F.conv_transpose2d = wrapFunc(F.conv_transpose2d, conv_trans_flops_compute, module_flop_count)
    F.conv_transpose3d = wrapFunc(F.conv_transpose3d, conv_trans_flops_compute, module_flop_count)

    # 激活函数
    F.relu = wrapFunc(F.relu, relu_flops_compute, module_flop_count)
    F.prelu = wrapFunc(F.prelu, prelu_flops_compute, module_flop_count)
    F.elu = wrapFunc(F.elu, elu_flops_compute, module_flop_count)
    F.leaky_relu = wrapFunc(F.leaky_relu, leaky_relu_flops_compute, module_flop_count)
    F.relu6 = wrapFunc(F.relu6, relu6_flops_compute, module_flop_count)
    if hasattr(F, "silu"):
        F.silu = wrapFunc(F.silu, silu_flops_compute, module_flop_count)
    F.gelu = wrapFunc(F.gelu, gelu_flops_compute, module_flop_count)

    # 归一化层
    F.batch_norm = wrapFunc(F.batch_norm, batch_norm_flops_compute, module_flop_count)
    F.layer_norm = wrapFunc(F.layer_norm, layer_norm_flops_compute, module_flop_count)
    F.instance_norm = wrapFunc(F.instance_norm, instance_norm_flops_compute, module_flop_count)
    F.group_norm = wrapFunc(F.group_norm, group_norm_flops_compute, module_flop_count)

    # 池化层
    F.avg_pool1d = wrapFunc(F.avg_pool1d, pool_flops_compute, module_flop_count)
    F.avg_pool2d = wrapFunc(F.avg_pool2d, pool_flops_compute, module_flop_count)
    F.avg_pool3d = wrapFunc(F.avg_pool3d, pool_flops_compute, module_flop_count)
    F.max_pool1d = wrapFunc(F.max_pool1d, pool_flops_compute, module_flop_count)
    F.max_pool2d = wrapFunc(F.max_pool2d, pool_flops_compute, module_flop_count)
    F.max_pool3d = wrapFunc(F.max_pool3d, pool_flops_compute, module_flop_count)
    F.adaptive_avg_pool1d = wrapFunc(F.adaptive_avg_pool1d, pool_flops_compute, module_flop_count)
    F.adaptive_avg_pool2d = wrapFunc(F.adaptive_avg_pool2d, pool_flops_compute, module_flop_count)
    F.adaptive_avg_pool3d = wrapFunc(F.adaptive_avg_pool3d, pool_flops_compute, module_flop_count)
    F.adaptive_max_pool1d = wrapFunc(F.adaptive_max_pool1d, pool_flops_compute, module_flop_count)
    F.adaptive_max_pool2d = wrapFunc(F.adaptive_max_pool2d, pool_flops_compute, module_flop_count)
    F.adaptive_max_pool3d = wrapFunc(F.adaptive_max_pool3d, pool_flops_compute, module_flop_count)

    # 上采样
    F.upsample = wrapFunc(F.upsample, upsample_flops_compute, module_flop_count)
    F.interpolate = wrapFunc(F.interpolate, upsample_flops_compute, module_flop_count)

    # 归一化指数运算 softmax
    F.softmax = wrapFunc(F.softmax, softmax_flops_compute, module_flop_count)

    # 词表/码本查表 embedding
    F.embedding = wrapFunc(F.embedding, embedding_flops_compute, module_flop_count)

    # 注意力：scaled_dot_product_attention 在 torch 2.0+ 中加入
    F.scaled_dot_product_attention = wrapFunc(F.scaled_dot_product_attention, attn_flops_compute, module_flop_count)

def _patch_tensor_methods(module_flop_count):
    """中文说明：`_patch_tensor_methods` 实现运行时 FLOPs profiler中的 `_patch_tensor_methods` 步骤，供训练、推理或调试流程复用。

    新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    torch.matmul = wrapFunc(torch.matmul, matmul_flops_compute, module_flop_count)
    torch.Tensor.matmul = wrapFunc(torch.Tensor.matmul, matmul_flops_compute, module_flop_count)
    torch.Tensor.__matmul__ = wrapFunc(torch.Tensor.__matmul__, matmul_flops_compute, module_flop_count)
    torch.mm = wrapFunc(torch.mm, matmul_flops_compute, module_flop_count)
    torch.Tensor.mm = wrapFunc(torch.Tensor.mm, matmul_flops_compute, module_flop_count)
    torch.bmm = wrapFunc(torch.bmm, matmul_flops_compute, module_flop_count)
    torch.Tensor.bmm = wrapFunc(torch.Tensor.bmm, matmul_flops_compute, module_flop_count)

    torch.addmm = wrapFunc(torch.addmm, addmm_flops_compute, module_flop_count)
    torch.Tensor.addmm = wrapFunc(torch.Tensor.addmm, tensor_addmm_flops_compute, module_flop_count)

    torch.mul = wrapFunc(torch.mul, mul_flops_compute, module_flop_count)
    torch.Tensor.mul = wrapFunc(torch.Tensor.mul, mul_flops_compute, module_flop_count)

    torch.add = wrapFunc(torch.add, add_flops_compute, module_flop_count)
    torch.Tensor.add = wrapFunc(torch.Tensor.add, add_flops_compute, module_flop_count)

    torch.einsum = wrapFunc(torch.einsum, einsum_flops_compute, module_flop_count)

    torch.baddbmm = wrapFunc(torch.baddbmm, tensor_addmm_flops_compute, module_flop_count)


def _patch_miscellaneous_operations(module_flop_count):
    """中文说明：`_patch_miscellaneous_operations` 实现运行时 FLOPs profiler中的 `_patch_miscellaneous_operations` 步骤，供训练、推理或调试流程复用。

    新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    einops.einsum = wrapFunc(einops.einsum, einops_einsum_flops_compute, module_flop_count)


def _reload_functionals():
    # PyTorch 的 torch.nn.functional 不支持 importlib.reload()
    """中文说明：`_reload_functionals` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    F.linear = old_functions[F.linear.__str__]
    F.conv1d = old_functions[F.conv1d.__str__]
    F.conv2d = old_functions[F.conv2d.__str__]
    F.conv3d = old_functions[F.conv3d.__str__]
    F.conv_transpose1d = old_functions[F.conv_transpose1d.__str__]
    F.conv_transpose2d = old_functions[F.conv_transpose2d.__str__]
    F.conv_transpose3d = old_functions[F.conv_transpose3d.__str__]
    F.relu = old_functions[F.relu.__str__]
    F.prelu = old_functions[F.prelu.__str__]
    F.elu = old_functions[F.elu.__str__]
    F.leaky_relu = old_functions[F.leaky_relu.__str__]
    F.relu6 = old_functions[F.relu6.__str__]
    if hasattr(F, "silu"):
        F.silu = old_functions[F.silu.__str__]
    F.gelu = old_functions[F.gelu.__str__]
    F.batch_norm = old_functions[F.batch_norm.__str__]
    F.layer_norm = old_functions[F.layer_norm.__str__]
    F.instance_norm = old_functions[F.instance_norm.__str__]
    F.group_norm = old_functions[F.group_norm.__str__]
    F.avg_pool1d = old_functions[F.avg_pool1d.__str__]
    F.avg_pool2d = old_functions[F.avg_pool2d.__str__]
    F.avg_pool3d = old_functions[F.avg_pool3d.__str__]
    F.max_pool1d = old_functions[F.max_pool1d.__str__]
    F.max_pool2d = old_functions[F.max_pool2d.__str__]
    F.max_pool3d = old_functions[F.max_pool3d.__str__]
    F.adaptive_avg_pool1d = old_functions[F.adaptive_avg_pool1d.__str__]
    F.adaptive_avg_pool2d = old_functions[F.adaptive_avg_pool2d.__str__]
    F.adaptive_avg_pool3d = old_functions[F.adaptive_avg_pool3d.__str__]
    F.adaptive_max_pool1d = old_functions[F.adaptive_max_pool1d.__str__]
    F.adaptive_max_pool2d = old_functions[F.adaptive_max_pool2d.__str__]
    F.adaptive_max_pool3d = old_functions[F.adaptive_max_pool3d.__str__]
    F.upsample = old_functions[F.upsample.__str__]
    F.interpolate = old_functions[F.interpolate.__str__]
    F.softmax = old_functions[F.softmax.__str__]
    F.embedding = old_functions[F.embedding.__str__]


def _reload_tensor_methods():
    """中文说明：`_reload_tensor_methods` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    torch.matmul = old_functions[torch.matmul.__str__]
    torch.Tensor.matmul = old_functions[torch.Tensor.matmul.__str__]
    torch.mm = old_functions[torch.mm.__str__]
    torch.Tensor.mm = old_functions[torch.Tensor.mm.__str__]
    torch.bmm = old_functions[torch.matmul.__str__]
    torch.Tensor.bmm = old_functions[torch.Tensor.bmm.__str__]
    torch.addmm = old_functions[torch.addmm.__str__]
    torch.Tensor.addmm = old_functions[torch.Tensor.addmm.__str__]
    torch.mul = old_functions[torch.mul.__str__]
    torch.Tensor.mul = old_functions[torch.Tensor.mul.__str__]
    torch.add = old_functions[torch.add.__str__]
    torch.Tensor.add = old_functions[torch.Tensor.add.__str__]

    torch.einsum = old_functions[torch.einsum.__str__]

    torch.baddbmm = old_functions[torch.baddbmm.__str__]


def _reload_miscellaneous_operations():
    """中文说明：`_reload_miscellaneous_operations` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    einops.einsum = old_functions[einops.einsum.__str__]

# 不能用 self.model.modules() 遍历所有子模块
# 因为 modules() 对重复模块只返回一次
def get_module_flops(module, prefix=""):
    """中文说明：`get_module_flops` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：它通过 monkey patch 和 module hook 统计算子 FLOPs，阅读时注意 patch/reload 必须成对。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    sum = module.__flops__
    log = ""

    if os.getenv("RANK","0") == "0":
        log = f"| {prefix}{module.__class__} 的 FLOPs = {sum/1e12:.5f} T\n"


    for child in module.children():
        flop,clog = get_module_flops(child, prefix=prefix+"    ")
        sum += flop
        log += clog

    return sum, log
