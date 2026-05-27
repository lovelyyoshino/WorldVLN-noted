# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import datetime
import functools
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Iterator, List, Tuple

import numpy as np
import pytz
import torch
import torch.distributed as tdist
import torch.nn.functional as F

import infinity.utils.dist as dist

os_system = functools.partial(subprocess.call, shell=True)
def echo(info):
    """中文说明：`echo` 实现Infinity 通用日志、指标和位置编码工具中的 `echo` 步骤，供训练、推理或调试流程复用。

    新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    os_system(f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"')
def os_system_get_stdout(cmd):
    """中文说明：`os_system_get_stdout` 实现Infinity 通用日志、指标和位置编码工具中的 `os_system_get_stdout` 步骤，供训练、推理或调试流程复用。

    新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
def os_system_get_stdout_stderr(cmd):
    """中文说明：`os_system_get_stdout_stderr` 实现Infinity 通用日志、指标和位置编码工具中的 `os_system_get_stdout_stderr` 步骤，供训练、推理或调试流程复用。

    新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    cnt = 0
    while True:
        try:
            sp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        except subprocess.TimeoutExpired:
            cnt += 1
            print(f'[fetch free_port file] 超时次数={cnt}')
        else:
            return sp.stdout.decode('utf-8'), sp.stderr.decode('utf-8')


def is_pow2n(x):
    """中文说明：`is_pow2n` 实现Infinity 通用日志、指标和位置编码工具中的 `is_pow2n` 步骤，供训练、推理或调试流程复用。

    新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    return x > 0 and (x & (x - 1) == 0)


def time_str(fmt='[%m-%d %H:%M:%S]'):
    """中文说明：`time_str` 实现Infinity 通用日志、指标和位置编码工具中的 `time_str` 步骤，供训练、推理或调试流程复用。

    新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(fmt)


class DistLogger(object):
    """中文说明：`DistLogger` 封装Infinity 通用日志、指标和位置编码工具中的状态和子模块。

    新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    def __init__(self, lg):
        """中文说明：`__init__` 初始化Infinity 通用日志、指标和位置编码工具需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self._lg = lg

    @staticmethod
    def do_nothing(*args, **kwargs):
        """中文说明：`do_nothing` 实现Infinity 通用日志、指标和位置编码工具中的 `do_nothing` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        pass

    def __getattr__(self, attr: str):
        """中文说明：`__getattr__` 实现Infinity 通用日志、指标和位置编码工具中的 `__getattr__` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return getattr(self._lg, attr) if self._lg is not None else DistLogger.do_nothing

class TensorboardLogger(object):
    """中文说明：`TensorboardLogger` 封装Infinity 通用日志、指标和位置编码工具中的状态和子模块。

    新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    def __init__(self, log_dir, filename_suffix):
        """中文说明：`__init__` 初始化Infinity 通用日志、指标和位置编码工具需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        try: import tensorflow_io as tfio
        except: pass
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
        self.step = 0

    def set_step(self, step=None):
        """中文说明：`set_step` 实现Infinity 通用日志、指标和位置编码工具中的 `set_step` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def loggable(self):
        """中文说明：`loggable` 实现Infinity 通用日志、指标和位置编码工具中的 `loggable` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return self.step == 0 or (self.step + 1) % 500 == 0

    def update(self, head='scalar', step=None, **kwargs):
        """中文说明：`update` 实现Infinity 通用日志、指标和位置编码工具中的 `update` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if step is None:
            step = self.step
            if not self.loggable(): return
        for k, v in kwargs.items():
            if v is None: continue
            if hasattr(v, 'item'): v = v.item()
            self.writer.add_scalar(f'{head}/{k}', v, step)

    def log_tensor_as_distri(self, tag, tensor1d, step=None):
        """中文说明：`log_tensor_as_distri` 实现Infinity 通用日志、指标和位置编码工具中的 `log_tensor_as_distri` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if step is None:
            step = self.step
            if not self.loggable(): return
        try:
            self.writer.add_histogram(tag=tag, values=tensor1d, global_step=step)
        except Exception as e:
            print(f'[log_tensor_as_distri 写入 writer.add_histogram 失败]: {e}')

    def log_image(self, tag, img_chw, step=None):
        """中文说明：`log_image` 实现Infinity 通用日志、指标和位置编码工具中的 `log_image` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if step is None:
            step = self.step
            if not self.loggable(): return
        self.writer.add_image(tag, img_chw, step, dataformats='CHW')

    def flush(self):
        """中文说明：`flush` 实现Infinity 通用日志、指标和位置编码工具中的 `flush` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.writer.flush()

    def close(self):
        """中文说明：`close` 释放Infinity 通用日志、指标和位置编码工具持有的文件句柄、视频句柄或 hook 资源。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.writer.close()


class TouchingDaemonDontForgetToStartMe(threading.Thread):
    """中文说明：`TouchingDaemonDontForgetToStartMe` 封装Infinity 通用日志、指标和位置编码工具中的状态和子模块。

    新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    def __init__(self, files: List[str], sleep_secs: int, verbose=False):
        """中文说明：`__init__` 初始化Infinity 通用日志、指标和位置编码工具需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        super().__init__(daemon=True)
        self.files = tuple(files)
        self.sleep_secs = sleep_secs
        self.is_finished = False
        self.verbose = verbose

        f_back = sys._getframe().f_back
        file_desc = f'{f_back.f_code.co_filename:24s}'[-24:]
        self.print_prefix = f' ({file_desc}, line{f_back.f_lineno:-4d}) @daemon@ '

    def finishing(self):
        """中文说明：`finishing` 实现Infinity 通用日志、指标和位置编码工具中的 `finishing` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.is_finished = True

    def run(self) -> None:
        """中文说明：`run` 实现Infinity 通用日志、指标和位置编码工具中的 `run` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        kw = {}
        if tdist.is_initialized(): kw['clean'] = True

        stt = time.time()
        if self.verbose: print(f'{time_str()}{self.print_prefix}[TouchingDaemon tid={threading.get_native_id()}] 开始每 {self.sleep_secs}s touch {self.files} ...', **kw)
        while not self.is_finished:
            for f in self.files:
                if os.path.exists(f):
                    try:
                        os.utime(f)
                        fp = open(f, 'a')
                        fp.close()
                    except: pass
            time.sleep(self.sleep_secs)

        if self.verbose: print(f'{time_str()}{self.print_prefix}[TouchingDaemon tid={threading.get_native_id()}] touch 结束，耗时 {time.time()-stt:.1f}s，文件 {self.files}，间隔 {self.sleep_secs}s。', **kw)


class SmoothedValue(object):
    """中文说明：跟踪一系列数值，并提供滑动窗口内的平滑统计。
    窗口平均值或全局序列平均值。
    """

    def __init__(self, window_size=30, fmt=None):
        """中文说明：`__init__` 初始化Infinity 通用日志、指标和位置编码工具需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """中文说明：`update` 实现Infinity 通用日志、指标和位置编码工具中的 `update` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        警告：这里不会同步 deque 本身。
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """中文说明：`median` 实现Infinity 通用日志、指标和位置编码工具中的 `median` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return np.median(self.deque) if len(self.deque) else 0

    @property
    def avg(self):
        """中文说明：`avg` 实现Infinity 通用日志、指标和位置编码工具中的 `avg` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return sum(self.deque) / (len(self.deque) or 1)

    @property
    def global_avg(self):
        """中文说明：`global_avg` 实现Infinity 通用日志、指标和位置编码工具中的 `global_avg` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return self.total / (self.count or 1)

    @property
    def max(self):
        """中文说明：`max` 实现Infinity 通用日志、指标和位置编码工具中的 `max` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return max(self.deque) if len(self.deque) else 0

    @property
    def value(self):
        """中文说明：`value` 实现Infinity 通用日志、指标和位置编码工具中的 `value` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return self.deque[-1] if len(self.deque) else 0

    def time_preds(self, counts) -> Tuple[float, str, str]:
        """中文说明：`time_preds` 实现Infinity 通用日志、指标和位置编码工具中的 `time_preds` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        remain_secs = counts * self.median
        return remain_secs, str(datetime.timedelta(seconds=round(remain_secs))), time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remain_secs))

    def __str__(self):
        """中文说明：`__str__` 实现Infinity 通用日志、指标和位置编码工具中的 `__str__` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger(object):
    """中文说明：`MetricLogger` 封装Infinity 通用日志、指标和位置编码工具中的状态和子模块。

    新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    def __init__(self):
        """中文说明：`__init__` 初始化Infinity 通用日志、指标和位置编码工具需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.meters = defaultdict(SmoothedValue)
        self.iter_end_t = time.time()
        self.log_iters = set()
        self.log_every_iter = False

    def update(self, **kwargs):
        # 可选过滤：if it != 0 and it not in self.log_iters: return
        """中文说明：`update` 实现Infinity 通用日志、指标和位置编码工具中的 `update` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        for k, v in kwargs.items():
            if v is None: continue
            if hasattr(v, 'item'): v = v.item()
            # 可选断言：assert isinstance(v, (float, int)), type(v)
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """中文说明：`__getattr__` 实现Infinity 通用日志、指标和位置编码工具中的 `__getattr__` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """中文说明：`__str__` 实现Infinity 通用日志、指标和位置编码工具中的 `__str__` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        loss_str = []
        for name, meter in self.meters.items():
            if len(meter.deque):
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return '  '.join(loss_str)

    def synchronize_between_processes(self):
        """中文说明：`synchronize_between_processes` 实现Infinity 通用日志、指标和位置编码工具中的 `synchronize_between_processes` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """中文说明：`add_meter` 实现Infinity 通用日志、指标和位置编码工具中的 `add_meter` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.meters[name] = meter

    def log_every(self, start_it, max_iters, itrt, log_freq, log_every_iter=False, header='', args=None):    # 中文说明：also solve logging & skipping iterations before start_it
        """中文说明：`log_every` 实现Infinity 通用日志、指标和位置编码工具中的 `log_every` 步骤，供训练、推理或调试流程复用。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        start_it = start_it % max_iters
        self.log_iters = set(range(start_it, max_iters, log_freq))
        self.log_iters.add(start_it)
        self.log_iters.add(max_iters-1)
        self.log_iters.add(max_iters)
        self.log_every_iter = log_every_iter
        self.iter_end_t = time.time()
        self.iter_time = SmoothedValue(fmt='{value:.4f}')
        self.data_time = SmoothedValue(fmt='{value:.3f}')
        header_fmt = header + ':  [{0:' + str(len(str(max_iters))) + 'd}/{1}]'

        start_time = time.time()
        if isinstance(itrt, Iterator) and not hasattr(itrt, 'preload') and not hasattr(itrt, 'set_epoch'): # 中文说明：this
            for it in range(start_it, max_iters):
                obj = next(itrt)
                if it < start_it: continue
                if args is not None and args.twoclip_alternatingtraining:  # 中文说明：2 clips alternating training
                    T = obj['raw_features_bcthw'][0].shape[1]
                    while (it % 2 == 0 and T > 21) or (it % 2 > 0 and T <= 21):
                        obj = next(itrt)
                        T = obj['raw_features_bcthw'][0].shape[1]
                self.data_time.update(time.time() - self.iter_end_t)
                yield it, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if self.log_every_iter or it in self.log_iters:
                    eta_seconds = self.iter_time.avg * (max_iters - it)
                    print(f'{header_fmt.format(it, max_iters)}  预计剩余: {str(datetime.timedelta(seconds=int(eta_seconds)))}  {str(self)}  单步耗时: {self.iter_time.value:.3f}s  数据耗时: {self.data_time.value*1e3:.1f}ms', flush=True)
                self.iter_end_t = time.time()
        else:
            if isinstance(itrt, int): itrt = range(itrt)
            for it, obj in enumerate(itrt):
                if it < start_it:
                    self.iter_end_t = time.time()
                    continue
                self.data_time.update(time.time() - self.iter_end_t)
                yield it, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if self.log_every_iter or it in self.log_iters:
                    eta_seconds = self.iter_time.avg * (max_iters - it)
                    print(f'{header_fmt.format(it, max_iters)}  预计剩余: {str(datetime.timedelta(seconds=int(eta_seconds)))}  {str(self)}  单步耗时: {self.iter_time.value:.3f}s  数据耗时: {self.data_time.value*1e3:.1f}ms', flush=True)
                self.iter_end_t = time.time()
        cost = time.time() - start_time
        cost_str = str(datetime.timedelta(seconds=int(cost)))
        print(f'{header}   本轮耗时:      {cost_str}   ({cost / (max_iters-start_it):.3f} s / it)', flush=True)


class NullDDP(torch.nn.Module):
    """中文说明：`NullDDP` 封装Infinity 通用日志、指标和位置编码工具中的状态和子模块。

    新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    def __init__(self, module, *args, **kwargs):
        """中文说明：`__init__` 初始化Infinity 通用日志、指标和位置编码工具需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        """中文说明：`forward` 执行Infinity 通用日志、指标和位置编码工具的前向计算；重点核对输入输出张量 shape 是否与调用方约定一致。

        新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return self.module(*args, **kwargs)


def build_2d_sincos_position_embedding(h, w, embed_dim, temperature=10000., sc=0, verbose=True):    # 代码/形状说明：(1, hw**2, embed_dim)
    # 模型 DiT 设置：sc=0
    # 模型 DETR 经验设置：sc=2?
    """中文说明：`build_2d_sincos_position_embedding` 实现Infinity 通用日志、指标和位置编码工具中的 `build_2d_sincos_position_embedding` 步骤，供训练、推理或调试流程复用。

    新手提示：这里是训练脚手架合集，重点看 MetricLogger、TensorBoard、2D sin/cos 位置编码。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid([grid_w, grid_h], indexing='ij')
    if sc == 0:
        scale = 1
    elif sc == 1:
        scale = math.pi * 2 / w
    else:
        scale = 1 / w
    grid_w = scale * grid_w.reshape(h*w, 1) # 代码/形状说明：scale * [0, 0, 0, 1, 1, 1, 2, 2, 2]
    grid_h = scale * grid_h.reshape(h*w, 1) # 代码/形状说明：scale * [0, 1, 2, 0, 1, 2, 0, 1, 2]

    assert embed_dim % 4 == 0, f'2D sin-cos 位置 embedding 要求 embed_dim={embed_dim} 能被 4 整除'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = (-math.log(temperature) * omega).exp()
    # 公式：omega == (1/T) ** (arange(pos_dim) / pos_dim)，只依赖通道维 C
    out_w = grid_w * omega.view(1, pos_dim) # 代码/形状说明：out_w: scale * [0*ome, 0*ome, 0*ome, 1*ome, 1*ome, 1*ome, 2*ome, 2*ome, 2*ome]
    out_h = grid_h * omega.view(1, pos_dim) # 代码/形状说明：out_h: scale * [0*ome, 1*ome, 2*ome, 0*ome, 1*ome, 2*ome, 0*ome, 1*ome, 2*ome]
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    if verbose: print(f'[build_2d_sincos_position_embedding @ {hw} x {hw}] scale_type={sc}, temperature={temperature:g}, shape={pos_emb.shape}')
    return pos_emb  # 代码/形状说明：(1, hw**2, embed_dim)


if __name__ == '__main__':
    pass
