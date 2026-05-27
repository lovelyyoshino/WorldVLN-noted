# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import os
import math
from datetime import datetime
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.modules.conv import _ConvNd
from torch.utils.checkpoint import TorchDispatchMode


def get_device_tflops():
    """中文说明：`get_device_tflops` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    peak_tflops = -1
    arch = torch.cuda.get_device_capability()
    if arch[0] == 8 and arch[1] == 0:  # A100/A800
        peak_tflops = 312
    elif arch[0] == 9 and arch[1] == 0:  # H100/H800
        peak_tflops = 989
    else:
        print(f"未知设备算力上限：device capability={arch[0]}.{arch[1]}，无法使用默认 TFLOPS")
    return peak_tflops


class NullCtx(TorchDispatchMode):

    """中文说明：`NullCtx` 封装VideoVAE FLOPs/MFU 统计工具中的状态和子模块。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """中文说明：`__torch_dispatch__` 实现VideoVAE FLOPs/MFU 统计工具中的 `__torch_dispatch__` 步骤，供训练、推理或调试流程复用。

        新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
        关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


class DisableMfu(NullCtx):
    """中文说明：`DisableMfu` 封装VideoVAE FLOPs/MFU 统计工具中的状态和子模块。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    def __enter__(self):
        """中文说明：`__enter__` 实现上下文管理协议，用于在进入/退出代码块时开启或恢复VideoVAE FLOPs/MFU 统计工具状态。

        新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
        关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        super().__enter__()
        self.old_flop_enable = Flops.enable
        Flops.enable = False

    def __exit__(self, *args, **kwargs):
        """中文说明：`__exit__` 实现上下文管理协议，用于在进入/退出代码块时开启或恢复VideoVAE FLOPs/MFU 统计工具状态。

        新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
        关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        Flops.enable = self.old_flop_enable
        super().__exit__(*args, **kwargs)


def context_fn():
    """中文说明：`context_fn` 实现VideoVAE FLOPs/MFU 统计工具中的 `context_fn` 步骤，供训练、推理或调试流程复用。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    return NullCtx(), DisableMfu()


class CustomFlops(ABC):
    """
        用于函数级 FLOPs 统计。
        1. 在 `CustomFlops` 上下文中运行目标函数。
        步骤说明：2. implement the hook `flops`
        说明：to support register_forward_hook

    """
    @abstractmethod
    def flops(self, args, kwargs, output) -> dict:
        """中文说明：`flops` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

        新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
        关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        pass


def conv_flops_func(module, args, kwargs, output):
    """中文说明：`conv_flops_func` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    return 2 * math.prod(module.kernel_size) * module.in_channels * output.numel()


def linear_flops_func(module, args, kwargs, output):
    """中文说明：`linear_flops_func` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    return 2 * module.in_features * output.numel()


def layernorm_flops_func(module, args, kwargs, output):
    """中文说明：`layernorm_flops_func` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    return 4 * output.numel()


def groupnorm_flops_func(module, args, kwargs, output):
    """中文说明：`groupnorm_flops_func` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    return 2 * output.numel()


def syncbatchnorm_flops_func(module, args, kwargs, output):
    """中文说明：`syncbatchnorm_flops_func` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    return 2 * output.numel()


basic_flops_func = {
    _ConvNd: conv_flops_func,
    nn.Linear: linear_flops_func,
    nn.LayerNorm: layernorm_flops_func,
    nn.GroupNorm: groupnorm_flops_func,
    nn.SyncBatchNorm: syncbatchnorm_flops_func,
}

@torch._dynamo.disable()
def calculate_flops(module, args, kwargs, output):
    """中文说明：`calculate_flops` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    flops = 0
    flops_dict = {}
    if isinstance(module, CustomFlops):
        flops_dict = module.flops(args, kwargs, output)
    else:
        flops_func = basic_flops_func[module._base_m]
        flops_dict = {module.__class__.__name__: flops_func(module, args, kwargs, output)}

    for module_class, module_flops in flops_dict.items():
        if module_class not in Flops.module_flops_dict:
            Flops.module_flops_dict[module_class] = module_flops * (3 if module.training else 1)
        else:
            Flops.module_flops_dict[module_class] += module_flops * (3 if module.training else 1)

    flops = sum(list(flops_dict.values()))
    Flops.flops += flops * (3 if module.training else 1)


class Flops:
    """中文说明：`Flops` 封装VideoVAE FLOPs/MFU 统计工具中的状态和子模块。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    handlers = []
    flops = 0
    enable = True
    module_flops_dict = {}

    @staticmethod
    def reset():
        """中文说明：`reset` 清空VideoVAE FLOPs/MFU 统计工具的累计统计或缓存，避免不同 step/实验互相污染。

        新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
        关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        tmp = Flops.flops
        Flops.flops = 0
        Flops.module_flops_dict = {}
        return tmp

    @staticmethod
    def _hook(module, args, kwargs, output):
        """中文说明：`_hook` 实现VideoVAE FLOPs/MFU 统计工具中的 `_hook` 步骤，供训练、推理或调试流程复用。

        新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
        关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        if not Flops.enable:
            return

        if module.training and not torch.is_grad_enabled():
            # 激活检查点模式
            return
        calculate_flops(module, args, kwargs, output)

    @staticmethod
    def _dfs_register_hooks(parent_name: str, cur_m: nn.Module):
        """中文说明：`_dfs_register_hooks` 实现VideoVAE FLOPs/MFU 统计工具中的 `_dfs_register_hooks` 步骤，供训练、推理或调试流程复用。

        新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
        关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        for name, m in cur_m.named_children():
            # 自定义 hooks
            if isinstance(m, CustomFlops):
                assert isinstance(m, nn.Module)
                Flops.handlers.append(
                    m.register_forward_hook(Flops._hook, with_kwargs=True)
                )
                continue
            # 内置 hooks
            is_registered = False
            for base_m, flops_func in basic_flops_func.items():
                if isinstance(m, base_m):
                    m._base_m = base_m
                    Flops.handlers.append(
                        m.register_forward_hook(Flops._hook, with_kwargs=True)
                    )
                    is_registered = True
                    break
            if not is_registered:
                Flops._dfs_register_hooks(parent_name + "." + name, m)

    @staticmethod
    def unwrap(self):
        """中文说明：`unwrap` 实现VideoVAE FLOPs/MFU 统计工具中的 `unwrap` 步骤，供训练、推理或调试流程复用。

        新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
        关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        for hdl in Flops.handlers:
            hdl.remove()



def register_mfu_hook(model):
    """中文说明：`register_mfu_hook` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    Flops._dfs_register_hooks("root", model)


def get_tflops():
    """中文说明：`get_tflops` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    return Flops.flops / 1e12


def get_tflops_dict(record_iters=1):
    """中文说明：`get_tflops_dict` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    tflops_dict = {module: round(flops / record_iters/ 1e12, 3) for module, flops in Flops.module_flops_dict.items()}
    return tflops_dict


def get_mfu(iter_time):
    # 计算 MFU
    """中文说明：`get_mfu` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常用公式是 TFLOPS = FLOPs / step_time / 1e12，MFU = measured_TFLOPS / device_peak_TFLOPS。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    ideal_TFLOPS = get_device_tflops()
    achieve_TFLOPs = Flops.reset() / 1e12
    mfu = achieve_TFLOPs / iter_time / ideal_TFLOPS
    return mfu
