# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import math
import os
import signal
import sys
import time
from typing import List, Optional, Tuple, Union

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# 可选调试：from memory_profiler import profile

import infinity.utils.dist as dist

class NullCtx:
    """中文说明：`NullCtx` 封装混合精度优化器封装中的状态和子模块。

    新手提示：阅读时按 backward -> unscale/clip -> optimizer.step -> scaler.update 的顺序看。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    def __enter__(self):
        """中文说明：`__enter__` 实现上下文管理协议，用于在进入/退出代码块时开启或恢复混合精度优化器封装状态。

        新手提示：阅读时按 backward -> unscale/clip -> optimizer.step -> scaler.update 的顺序看。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """中文说明：`__exit__` 实现上下文管理协议，用于在进入/退出代码块时开启或恢复混合精度优化器封装状态。

        新手提示：阅读时按 backward -> unscale/clip -> optimizer.step -> scaler.update 的顺序看。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        pass


class AmpOptimizer:
    """中文说明：`AmpOptimizer` 封装混合精度优化器封装中的状态和子模块。

    新手提示：阅读时按 backward -> unscale/clip -> optimizer.step -> scaler.update 的顺序看。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    def __init__(
        self,
        model_name_3letters: str, mixed_precision: int,
        optimizer: torch.optim.Optimizer, model_maybe_fsdp: Union[torch.nn.Module, FSDP],
        r_accu: float, grad_clip: float, zero: int,
    ):
        """中文说明：`__init__` 初始化混合精度优化器封装需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：阅读时按 backward -> unscale/clip -> optimizer.step -> scaler.update 的顺序看。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        self.enable_amp = mixed_precision > 0
        self.zero = zero
        if self.enable_amp:
            self.using_fp16_rather_bf16 = mixed_precision != 2
            self.max_sc = float(mixed_precision if mixed_precision > 128 else 32768)

            self.amp_ctx = torch.autocast('cuda', enabled=True, dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16, cache_enabled=self.zero == 0)    # 中文说明：todo: cache_enabled=False
            if self.using_fp16_rather_bf16:
                self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11, growth_interval=1000)
            else:
                self.scaler = None
        else:
            self.using_fp16_rather_bf16 = True
            self.amp_ctx = NullCtx()
            self.scaler = None

        t = torch.zeros(dist.get_world_size())
        t[dist.get_rank()] = float(self.enable_amp)
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'enable_amp: {t}'

        t = torch.zeros(dist.get_world_size())
        t[dist.get_rank()] = float(self.using_fp16_rather_bf16)
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'using_fp16_rather_bf16: {t}'

        self.model_name_3letters = model_name_3letters
        self.optimizer, self.model_maybe_fsdp = optimizer, model_maybe_fsdp
        self.r_accu = r_accu

        self.paras = self.names = ...    # 中文说明：todo: solve EMA-related codes

        self.grad_clip, self.grad_clip_we = grad_clip, 0    # 中文说明：todo: disable wclip
        if self.grad_clip > 100:
            self.grad_clip %= 100
            self.per_param = True
        else:
            self.per_param = False
        self.per_param = False          # 中文说明：todo: disable wclip

        self.early_clipping = grad_clip > 0 and not hasattr(optimizer, 'global_grad_norm')
        self.late_clipping = grad_clip > 0 and hasattr(optimizer, 'global_grad_norm')   # 中文说明：deepspeed's optimizer

        self.fp = None
        self.last_orig_norm: torch.Tensor = torch.tensor(0.1)


    # 中文说明：@profile(precision=4, stream=open('amp_sc.log', 'w+'))
    def backward_clip_step(
        self, ep: int, it: int, g_it: int, stepping: bool, loss: torch.Tensor, clip_decay_ratio=1, stable=False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        # 反向传播
        """中文说明：`backward_clip_step` 读取、采样或保存视频帧序列；重点看 fps、帧号范围和输出维度。

        新手提示：阅读时按 backward -> unscale/clip -> optimizer.step -> scaler.update 的顺序看。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        loss = loss.mul(self.r_accu)   # 代码/形状说明：r_accu == 1.0 / n_gradient_accumulation
        orig_norm = scaler_sc = None
        # 调试条件：if self.fp is not None:
        # 代码/形状说明：if g_it % 20 == 0: self.fp.seek(0); self.fp.truncate(0)
        if self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=False, create_graph=False)  # 中文说明：retain_graph=retain_graph, create_graph=create_graph
        else:
            loss.backward(retain_graph=False, create_graph=False)
        # 调试条件：if self.fp is not None: self.fp.write(f'[backward_clip_step:131] [it{it}, g_it{g_it}] after backward\n'); self.fp.flush()

        # 先裁剪梯度，再执行优化器 step
        if stepping:
            if self.scaler is not None: self.scaler.unscale_(self.optimizer)    # 中文说明：unscale 后才能拿到真实梯度用于裁剪和统计。
            # 调试条件：if self.fp is not None: self.fp.write(f'[backward_clip_step:137] [it{it}, g_it{g_it}] after scaler.unscale_\n'); self.fp.flush()

            skipped, orig_norm = 0, self.last_orig_norm
            # 调试保护：try:
            if self.fp is not None:
                if g_it % 10 == 0: self.fp.seek(0); self.fp.truncate(0)
                self.fp.write(f'<ep{ep} it{it} {g_it}>\n'); self.fp.flush()
            if self.early_clipping:
                c = self.grad_clip * clip_decay_ratio
                if self.zero:
                    orig_norm: Optional[torch.Tensor] = self.model_maybe_fsdp.clip_grad_norm_(c)
                else:
                    orig_norm: Optional[torch.Tensor] = torch.nn.utils.clip_grad_norm_(self.model_maybe_fsdp.parameters(), c)

            # 调试条件：if self.fp is not None: self.fp.write(f'[backward_clip_step:175] [it{it}, g_it{g_it}] before opt step\n'); self.fp.flush()
            if self.scaler is not None:
                self.scaler: torch.cuda.amp.GradScaler
                if self.zero:
                    # 调用 step 前同步 found_inf_per_device，确保只有部分 rank 出现 inf 时其他 rank 也能知道
                    # 否则保存 FSDP 优化器状态时会因不同 rank 的 step 不一致而触发 AssertionError
                    for optimizer_state in self.scaler._per_optimizer_states.values():
                        for t in optimizer_state['found_inf_per_device'].values():
                            dist.allreduce(t)   # 中文说明：ideally, each rank only has one single t; so no need to use async allreduce

                self.scaler.step(self.optimizer)
                scaler_sc: Optional[float] = self.scaler.get_scale()
                if scaler_sc > self.max_sc: # 中文说明：fp16 will overflow when >65536, so multiply 32768 could be dangerous
                    # 调试输出：print(f'[fp16 scaling] too large loss scale {scaler_sc}! (clip to {self.max_sc:g})')
                    self.scaler.update(new_scale=self.max_sc)
                else:
                    self.scaler.update()
                try:
                    scaler_sc = float(math.log2(scaler_sc))
                except Exception as e:
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    time.sleep(1)
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    raise e
            else:
                self.optimizer.step()

            if self.late_clipping:
                orig_norm: Optional[torch.Tensor] = self.optimizer.global_grad_norm
            self.last_orig_norm = orig_norm
            # 这里不调用 zero_grad，因为后续还要记录这些梯度
        return orig_norm, scaler_sc

    def state_dict(self):
        """中文说明：`state_dict` 保存或恢复混合精度优化器封装的可序列化状态，保证断点续训和推理加载一致。

        新手提示：阅读时按 backward -> unscale/clip -> optimizer.step -> scaler.update 的顺序看。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return {
            'optimizer': self.optimizer.state_dict()
        } if self.scaler is None else {
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_state_dict(self, state, strict=True):
        """中文说明：`load_state_dict` 保存或恢复混合精度优化器封装的可序列化状态，保证断点续训和推理加载一致。

        新手提示：阅读时按 backward -> unscale/clip -> optimizer.step -> scaler.update 的顺序看。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if self.scaler is not None:
            try: self.scaler.load_state_dict(state['scaler'])
            except Exception as e: print(f'[fp16 load_state_dict 错误] {e}')
        self.optimizer.load_state_dict(state['optimizer'])
