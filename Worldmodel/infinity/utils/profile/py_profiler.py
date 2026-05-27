# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import logging
import pstats
import cProfile
import contextlib

def _colored(st, color, background=False):
    """中文说明：`_colored` 实现Python/Torch profiler 包装中的 `_colored` 步骤，供训练、推理或调试流程复用。

    新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st

def _format_fcn(fcn):
    """中文说明：`_format_fcn` 实现Python/Torch profiler 包装中的 `_format_fcn` 步骤，供训练、推理或调试流程复用。

    新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    return f"{fcn[0]}:{fcn[1]}:{fcn[2]}"

class py_profiler(contextlib.ContextDecorator):
    """中文说明：`py_profiler` 封装Python/Torch profiler 包装中的状态和子模块。

    新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    def __init__(self, enabled=True, sort='cumtime', fn=None, ts=1):
        """中文说明：`__init__` 初始化Python/Torch profiler 包装需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.enabled, self.sort, self.fn, self.time_scale = enabled, sort, fn, 1e3/ts
    def __enter__(self):
        """中文说明：`__enter__` 实现上下文管理协议，用于在进入/退出代码块时开启或恢复Python/Torch profiler 包装状态。

        新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.pr = cProfile.Profile()
        if self.enabled:
            self.pr.enable()
    def __exit__(self, *exc):
        """中文说明：`__exit__` 实现上下文管理协议，用于在进入/退出代码块时开启或恢复Python/Torch profiler 包装状态。

        新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if self.enabled:
            self.pr.disable()
            if self.fn:
                self.pr.dump_stats(self.fn)
            stats = pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort)
            for fcn in stats.fcn_list[0:int(len(stats.fcn_list))]:
                (_primitive_calls, num_calls, tottime, cumtime, callers) = stats.stats[fcn]
                scallers = sorted(callers.items(), key=lambda x: -x[1][2])
                print(f"调用次数:{num_calls:8d}  自身耗时:{tottime*self.time_scale:7.2f}ms  累计耗时:{cumtime*self.time_scale:7.2f}ms", _colored(_format_fcn(fcn).ljust(50), "yellow"))


if __name__ == "__main__":
    def fn():
        """中文说明：`fn` 实现Python/Torch profiler 包装中的 `fn` 步骤，供训练、推理或调试流程复用。

        新手提示：这些上下文管理器只收集性能数据，不应改变模型数值结果。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        s = 0
        for i in range(10000000):
            s += i
        return s

    with py_profiler():
        fn()
