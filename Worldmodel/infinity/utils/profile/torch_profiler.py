# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import logging
import os
from contextlib import contextmanager, nullcontext
from datetime import datetime

import torch
import torch.distributed as dist
from torch.profiler import record_function as torch_record_function


class _TraceHandler:
    """中文说明：`_TraceHandler` 封装Python/Torch profiler 包装中的状态和子模块。

    新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    def __init__(self, save_path="/tmp/trace.json", logger=None, rank=None):
        """中文说明：`__init__` 初始化Python/Torch profiler 包装需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger(__name__)

        self.logger.info(f"trace 导出路径：{save_path}")
        self.save_path = save_path + ".json.gz"
        self.rank = rank

    def __call__(self, prof):
        """中文说明：`__call__` 实现Python/Torch profiler 包装中的 `__call__` 步骤，供训练、推理或调试流程复用。

        新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if self.logger is not None:
            self.logger.info(f"将 trace 导出到 {self.save_path}")
        prof.export_chrome_trace(self.save_path)

class torch_profiler:
    """
        用法：

        代码示例：```python
        说明：import pnp

        公式/形状说明：pnp.torch_profiler.setup(output_folder="./", wait_steps=30)

        公式/形状说明：for step in range(100):
            公式/形状说明：pnp.torch_profiler.step()
            ...

            公式/形状说明：with pnp.troch_profiler.mark("fwd"):
                公式/形状说明：model.forward()

            ...

            公式/形状说明：with pnp.torch_profiler.mark("bwd"):
                公式/形状说明：loss.backward()

        ```


    """
    _TP = None
    mark = nullcontext

    @staticmethod
    def step():
        """中文说明：`step` 实现Python/Torch profiler 包装中的 `step` 步骤，供训练、推理或调试流程复用。

        新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if torch_profiler._TP is None:
            return

        torch_profiler._TP.step()

    @staticmethod
    @property
    def mark():
        """中文说明：`mark` 实现Python/Torch profiler 包装中的 `mark` 步骤，供训练、推理或调试流程复用。

        新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return torch_profiler.mark

    @staticmethod
    def setup(enabled=True, output_folder="./", file_prefix="", wait_steps=30):
        """
                enabled 为 False 时，profiler 不执行任何统计。
                output_folder：保存 profiler trace 的目录。
                公式/形状说明：wait_steps: 在训练循环中等待 wait_steps 步后开始 profiling
                说明：file_prefix: 自定义 trace 文件名前缀

        """
        if enabled:
           if not os.path.exists(output_folder):
               os.makedirs(output_folder, exist_ok=True)

           torch_profiler._TP = torch.profiler.profile(
               activities=[
                   torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA,
               ],
               schedule=torch.profiler.schedule(
                   wait=wait_steps,
                   warmup=3,
                   active=5,
                   repeat=0,
               ),
               with_stack=True,
               record_shapes=True,
               profile_memory=False,
               on_trace_ready=_TraceHandler(
                   f"{output_folder}/{file_prefix}world_size-{dist.get_world_size()}-rank{dist.get_rank()}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                   None,
                   dist.get_rank(),
               ),
           )
           torch_profiler._TP.start()
           torch_profiler.mark = torch_record_function
