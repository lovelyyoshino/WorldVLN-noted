# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import os
import logging

import torch

logger = logging.getLogger(__name__)
RANK = int(os.environ["RANK"]) if "RANK" in os.environ else 0

class NanDetector:
    """
    检测 forward/backward 中第一次出现的 NaN 或 Inf，并连同模块名一起记录。
    """

    def __init__(self, model, forward=True, backward=True):
        """中文说明：`__init__` 初始化NaN/Inf 检测 hook 工具需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：它通过 forward/backward hook 定位异常张量，调试训练发散时优先看这里。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.bhooks = []
        self.fhooks = []
        self.forward = forward
        self.backward = backward
        self.named_parameters = list(model.named_parameters())
        self.reset()

        for name, mod in model.named_modules():
            mod.__module_name = name
            self.add_hooks(mod)

    def __enter__(self):
        """中文说明：`__enter__` 实现上下文管理协议，用于在进入/退出代码块时开启或恢复NaN/Inf 检测 hook 工具状态。

        新手提示：它通过 forward/backward hook 定位异常张量，调试训练发散时优先看这里。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # 输出模型所有梯度范数，方便定位异常层
        """中文说明：`__exit__` 实现上下文管理协议，用于在进入/退出代码块时开启或恢复NaN/Inf 检测 hook 工具状态。

        新手提示：它通过 forward/backward hook 定位异常张量，调试训练发散时优先看这里。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        norm = {}
        gradients = {}
        for name, param in self.named_parameters:
            if param.grad is not None:
                grad_norm = torch.norm(param.grad.data, p=2, dtype=torch.float32)
                norm[name] = grad_norm.item()
                if torch.isnan(grad_norm).any() or torch.isinf(grad_norm).any():
                    gradients[name] = param.grad.data
        if len(gradients) > 0:
            logger.info("检测到 NaN/Inf 梯度范数，正在输出各层范数和异常梯度。")
            logger.info(f"梯度范数: {norm}")
            logger.info(f"异常梯度: {gradients}")

        self.close()

    def add_hooks(self, module):
        """中文说明：`add_hooks` 实现NaN/Inf 检测 hook 工具中的 `add_hooks` 步骤，供训练、推理或调试流程复用。

        新手提示：它通过 forward/backward hook 定位异常张量，调试训练发散时优先看这里。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if self.forward:
            self.fhooks.append(module.register_forward_hook(self.fhook_fn))
        if self.backward:
            self.bhooks.append(module.register_backward_hook(self.bhook_fn))

    def reset(self):
        """中文说明：`reset` 清空NaN/Inf 检测 hook 工具的累计统计或缓存，避免不同 step/实验互相污染。

        新手提示：它通过 forward/backward hook 定位异常张量，调试训练发散时优先看这里。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.has_printed_f = False
        self.has_printed_b = False

    def _detect(self, tensor, name, backward):
        """中文说明：`_detect` 实现NaN/Inf 检测 hook 工具中的 `_detect` 步骤，供训练、推理或调试流程复用。

        新手提示：它通过 forward/backward hook 定位异常张量，调试训练发散时优先看这里。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        err = None
        if (
            torch.is_floating_point(tensor)
            # 单值张量（例如 loss）能提供的信息有限
            and tensor.numel() >= 2
        ):
            with torch.no_grad():
                if torch.isnan(tensor).any():
                    err = "NaN"
                elif torch.isinf(tensor).any():
                    err = "Inf"
        if err is not None:
            stage = "反向传播 backward" if backward else "前向传播 forward"
            err = f"在 {name} 的输出中检测到 {err}，shape: {tensor.shape}，阶段: {stage}"
        return err

    def _apply(self, module, inp, x, backward):
        """中文说明：`_apply` 实现NaN/Inf 检测 hook 工具中的 `_apply` 步骤，供训练、推理或调试流程复用。

        新手提示：它通过 forward/backward hook 定位异常张量，调试训练发散时优先看这里。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if torch.is_tensor(x):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            err = self._detect(x, module.__module_name, backward)
            if err is not None:
                if torch.is_tensor(inp) and not backward:
                    err += (
                        f"；输入最大值: {inp.max().item()}，输入最小值: {inp.min().item()}"
                    )
                has_printed_attr = "has_printed_b" if backward else "has_printed_f"
                logger.warning(f"rank-{RANK}，错误信息 err_info: {err}")
                setattr(self, has_printed_attr, True)
        elif isinstance(x, dict):
            for v in x.values():
                self._apply(module, inp, v, backward)
        elif isinstance(x, list) or isinstance(x, tuple):
            for v in x:
                self._apply(module, inp, v, backward)

    def fhook_fn(self, module, inp, output):
        """中文说明：`fhook_fn` 实现NaN/Inf 检测 hook 工具中的 `fhook_fn` 步骤，供训练、推理或调试流程复用。

        新手提示：它通过 forward/backward hook 定位异常张量，调试训练发散时优先看这里。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if not self.has_printed_f:
            self._apply(module, inp, output, backward=False)

    def bhook_fn(self, module, inp, output):
        """中文说明：`bhook_fn` 实现NaN/Inf 检测 hook 工具中的 `bhook_fn` 步骤，供训练、推理或调试流程复用。

        新手提示：它通过 forward/backward hook 定位异常张量，调试训练发散时优先看这里。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if not self.has_printed_b:
            self._apply(module, inp, output, backward=True)

    def close(self):
        """中文说明：`close` 释放NaN/Inf 检测 hook 工具持有的文件句柄、视频句柄或 hook 资源。

        新手提示：它通过 forward/backward hook 定位异常张量，调试训练发散时优先看这里。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        for hook in self.fhooks + self.bhooks:
            hook.remove()
