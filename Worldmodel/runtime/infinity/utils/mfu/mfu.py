# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import time
import torch
import torch.distributed as dist
from contextlib import contextmanager, nullcontext
from functools import wraps
from .flops_profiler import FlopsProfiler
from .flops_calc_impl.custom_flops_impl import CUSTOM_HOOK_MAPPING, CUSTOM_NAME_MAPPING

class _MFU:
    """中文说明：`_MFU` 封装MFU 汇总工具中的状态和子模块。

    新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
    关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    def __init__(self, calibration_steps = 5, repeat_after_steps = -1):
        """
        `calibration_steps=-1` 表示每次都做校准，额外开销很小。
        `repeat_after_steps=-1` 表示永不重复统计。
        """
        self.profs = []
        self.iter_time = None
        self.is_during_calibration = False
        self.calibration_steps = calibration_steps
        self.repeat_after_steps = repeat_after_steps
        self.steps = 0
        self.flops = []
        self.detail_flops = ""
        self.ideal_TFLOPS = self._get_device_tflops()
        self.ignore_list=[]
        self.prof = FlopsProfiler()

    def append(self, model):
        """中文说明：`append` 实现MFU 汇总工具中的 `append` 步骤，供训练、推理或调试流程复用。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        self.prof.append(model)

    def step(self, iter_time):
        """中文说明：`step` 实现MFU 汇总工具中的 `step` 步骤，供训练、推理或调试流程复用。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        self.steps += 1
        self.iter_time = iter_time

        if self.calibration_steps < 0 or self.steps <= self.calibration_steps:
            self.is_during_calibration = True
            flop = 0

            try:
                flop, log = self.prof.get_total_flops()
            except Exception as e:
                print(f"[WARN]: get_total_flops 失败：{e}")

            self.detail_flops = log
            self.flops.append(flop)
            self.reset()

            if self.steps == self.calibration_steps:
                self.is_during_calibration = False
                self.clear()

        if self.calibration_steps > 0 and self.repeat_after_steps > 0:
            if self.steps >= self.calibration_steps + self.repeat_after_steps:
                self.flops.clear()
                self.steps = 0
                self.start()


    def stop(self):
       """中文说明：`stop` 实现MFU 汇总工具中的 `stop` 步骤，供训练、推理或调试流程复用。

       新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
       关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
       """
       self.prof.stop_profile()

    def reset(self):
       """中文说明：`reset` 清空MFU 汇总工具的累计统计或缓存，避免不同 step/实验互相污染。

       新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
       关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
       """
       self.prof.reset_profile()

    def clear(self):
        """中文说明：`clear` 清空MFU 汇总工具的累计统计或缓存，避免不同 step/实验互相污染。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        self.prof.end_profile()

    def start(self):
        """中文说明：`start` 实现MFU 汇总工具中的 `start` 步骤，供训练、推理或调试流程复用。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        self.prof.start_profile(self.ignore_list)

    def get_flops_detail_info(self):
        """中文说明：`get_flops_detail_info` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        return self.detail_flops

    def get_mfu(self):
        """中文说明：`get_mfu` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        mfu = -1
        if self.iter_time is not None and len(self.flops) > 0:
            avg_flop = sum(self.flops) / len(self.flops)
            avg_Tflops = avg_flop / 1e12
            mfu = avg_Tflops / self.iter_time / self.ideal_TFLOPS
            if not isinstance(mfu, float):
                print(f"[WARN]: MFU 计算异常，{type(mfu)=}。")
                mfu = -1

        return mfu

    def _get_device_tflops(self):
        """中文说明：`_get_device_tflops` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        peak_tflops = -1
        arch = torch.cuda.get_device_capability()
        if arch[0] == 8 and arch[1] == 0:  # A100/A800
            peak_tflops = 312  # 中文说明：fp16，不使用稀疏加速
        elif arch[0] == 9 and arch[1] == 0:  # H100/H800
            peak_tflops = 989  # 中文说明：fp16，不使用稀疏加速
        else:
            print(f"未知的 GPU 设备能力 {arch[0]}.{arch[1]}，无法给出默认 TFLOPS")
        return peak_tflops



class mfutool:
    """中文说明：`mfutool` 封装MFU 汇总工具中的状态和子模块。

    新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
    关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    _mfu = None
    _last_time = None
    _iter_time = None

    @staticmethod
    def setup(calibration_steps = 5, repeat_after_steps = -1):
        """
        `calibration_steps=-1` 表示每次都做校准，额外开销很小。
        `repeat_after_steps=-1` 表示永不重复统计。
        """
        if mfutool._mfu is None:
            mfutool._mfu = _MFU(calibration_steps = calibration_steps, repeat_after_steps = repeat_after_steps)

    @staticmethod
    def add(model):
        """中文说明：`add` 实现MFU 汇总工具中的 `add` 步骤，供训练、推理或调试流程复用。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        if mfutool._mfu is None:
            mfutool._mfu = _MFU()
        mfutool._mfu.append(model)

    @staticmethod
    def enable():
        """中文说明：`enable` 实现MFU 汇总工具中的 `enable` 步骤，供训练、推理或调试流程复用。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        if mfutool._mfu is not None:
            mfutool._mfu.start()

    @staticmethod
    def disable():
        """中文说明：`disable` 实现MFU 汇总工具中的 `disable` 步骤，供训练、推理或调试流程复用。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        if mfutool._mfu is not None:
            mfutool._mfu.stop()

    @staticmethod
    def step():
        """中文说明：`step` 实现MFU 汇总工具中的 `step` 步骤，供训练、推理或调试流程复用。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        if mfutool._mfu is not None:
            if mfutool._last_time is not None:
                mfutool._iter_time = time.time() - mfutool._last_time
                mfutool._mfu.step(mfutool._iter_time)
            mfutool._last_time = time.time()

    @staticmethod
    def iter_time():
        """中文说明：`iter_time` 实现MFU 汇总工具中的 `iter_time` 步骤，供训练、推理或调试流程复用。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        return mfutool._iter_time

    @staticmethod
    def get_mfu():
        """中文说明：`get_mfu` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        if mfutool._mfu is not None:
            return mfutool._mfu.get_mfu()

    @staticmethod
    def get_flops_detail_info():
        """中文说明：`get_flops_detail_info` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        if mfutool._mfu is not None:
            return mfutool._mfu.get_flops_detail_info()

    @staticmethod
    def register_custom(name, func):
        """中文说明：`register_custom` 实现MFU 汇总工具中的 `register_custom` 步骤，供训练、推理或调试流程复用。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        if name not in CUSTOM_NAME_MAPPING:
            print(f"[WARN] 找不到 {name}，请先用 @mfutool.custom_flops 装饰对应的模块类")
            return
        CUSTOM_HOOK_MAPPING[CUSTOM_NAME_MAPPING[name]] = func

    @staticmethod
    def custom_flops(cls, name):
        """中文说明：`custom_flops` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

        新手提示：MFU = 实测吞吐 TFLOPS / 设备峰值 TFLOPS，用来估算模型利用率。
        关键公式：TFLOPS = FLOPs / 秒数 / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
        """
        CUSTOM_NAME_MAPPING[name] = cls
        return cls
