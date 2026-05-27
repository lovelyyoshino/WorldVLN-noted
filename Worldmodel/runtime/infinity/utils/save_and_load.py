# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import gc
import os
import os.path as osp
import subprocess
import time
import re
from typing import List, Optional, Tuple

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import glob
import shutil
from infinity.utils import arg_util
import infinity.utils.dist as dist


def glob_with_epoch_iter(pattern, recursive=False):
    """中文说明：`glob_with_epoch_iter` 实现checkpoint 保存、恢复和合并工具中的 `glob_with_epoch_iter` 步骤，供训练、推理或调试流程复用。

    新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
    关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
    """
    def extract_ep_iter(filename):
        """中文说明：`extract_ep_iter` 实现checkpoint 保存、恢复和合并工具中的 `extract_ep_iter` 步骤，供训练、推理或调试流程复用。

        新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
        关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
        """
        match = re.search(r'ep(\d+)-iter(\d+)', filename)
        if match:
            ep = int(match.group(1))
            iter_idx = int(match.group(2))
            return ep, iter_idx
        return 0, 0
    return sorted(glob.glob(pattern, recursive=recursive), key=lambda x: extract_ep_iter(os.path.basename(x)), reverse=True)


def glob_with_global_step(pattern, recursive=False):
    """中文说明：`glob_with_global_step` 实现checkpoint 保存、恢复和合并工具中的 `glob_with_global_step` 步骤，供训练、推理或调试流程复用。

    新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
    关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
    """
    def extract_ep_iter(filename):
        """中文说明：`extract_ep_iter` 实现checkpoint 保存、恢复和合并工具中的 `extract_ep_iter` 步骤，供训练、推理或调试流程复用。

        新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
        关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
        """
        match = re.search(r'global_step_(\d+)', filename)
        if match:
            iter_idx = int(match.group(1))
            return iter_idx
        return 0
    return sorted(glob.glob(pattern, recursive=recursive), key=lambda x: extract_ep_iter(os.path.basename(x)), reverse=True)


class CKPTSaver(object):
    """中文说明：`CKPTSaver` 封装checkpoint 保存、恢复和合并工具中的状态和子模块。

    新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
    关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
    """
    def __init__(self, is_master: bool, eval_milestone: List[Tuple[float, float]]):
        """中文说明：`__init__` 初始化checkpoint 保存、恢复和合并工具需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
        关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
        """
        self.is_master = is_master
        self.time_stamp = torch.tensor([time.time() - 1e5, time.time()], device=dist.get_device())
        self.sp_also: subprocess.Popen = None
        self.sp_best: subprocess.Popen = None
        self.sp_backup: subprocess.Popen = None
        self.acc_str, self.eval_milestone = '[no acc str]', eval_milestone

    def sav(
        self, args: arg_util.Args, g_it: int, next_ep: int, next_it: int, trainer,
        acc_str: Optional[str] = None, eval_milestone: Optional[List[Tuple[float, float]]] = None,
        also_save_to: str = None, best_save_to: str = None,
    ):
        """中文说明：`sav` 实现checkpoint 保存、恢复和合并工具中的 `sav` 步骤，供训练、推理或调试流程复用。

        新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
        关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
        """
        fname = f'global_step_{g_it}.pth'
        local_out_ckpt = os.path.join(args.bed, fname)
        trainer_state = trainer.state_dict()
        stt = time.time()
        if self.is_master:
            torch.save({
                'args':         args.state_dict(),
                'arch':         args.model,
                'epoch':        next_ep,
                'iter':         next_it,
                'trainer':      trainer_state,
                'acc_str':      self.acc_str,
                'g_it':         g_it,
            }, local_out_ckpt)
        cost = time.time() - stt
        print(f'Checkpoint 保存耗时: {cost:.2f}s', flush=True)
        print(f'Checkpoint 保存到: {local_out_ckpt}', flush=True)

        del trainer_state
        gc.collect(),
        torch.cuda.empty_cache()
        dist.barrier()


def auto_resume(args: arg_util.Args, pattern='*.pth') -> Tuple[List[str], int, int, str, List[Tuple[float, float]], dict, dict]:
    """中文说明：`auto_resume` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
    关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
    """
    info = []
    resume = ''
    if args.auto_resume:
        all_ckpt = glob_with_global_step(os.path.join(args.bed, pattern))
        if len(all_ckpt) == 0:
            info.append(f'[auto_resume] 未找到 ckpt @ {pattern}')
            info.append(f'[auto_resume quit] 退出自动恢复')
        else:
            resume = all_ckpt[0]
            info.append(f'[auto_resume] 自动从 @ {resume} 加载 ...')
    else:
        info.append(f'[auto_resume] 已禁用')
        info.append(f'[auto_resume quit] 退出自动恢复')

    if len(resume) == 0:
        return info, 0, 0, '[no acc str]', [], {}, {}

    print(f'自动恢复 checkpoint: {resume}')
    ckpt = torch.load(resume, map_location='cpu')

    dist.barrier()
    ep, it, g_it = ckpt['epoch'], ckpt['iter'], ckpt['g_it']
    eval_milestone = ckpt.get('milestones', [])
    info.append(f'[auto_resume success] 从 ep{ep}, it{it} 恢复，eval_milestone: {eval_milestone}')
    return info, ep, g_it, ckpt.get('acc_str', '[no acc str]'), eval_milestone, ckpt['trainer'], ckpt['args']

def omnistore_auto_resume(args: arg_util.Args, pattern='ckpt*.pth'):
    """中文说明：`omnistore_auto_resume` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
    关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
    """
    info = []
    resume = ''
    if args.auto_resume:
        for dd in (args.local_out_path, args.bed):
            all_ckpt = glob_with_global_step(os.path.join(dd, pattern))
            if len(all_ckpt): break
        if len(all_ckpt) == 0:
            info.append(f'[auto_resume] 未找到 ckpt @ {pattern}')
            info.append(f'[auto_resume quit] 退出自动恢复')
        else:
            resume = all_ckpt[0]
            info.append(f'[auto_resume] 自动从 @ {resume} 加载 ...')
    else:
        info.append(f'[auto_resume] 已禁用')
        info.append(f'[auto_resume quit] 退出自动恢复')

    return resume, info


class omnistoreCheckpoint(object):
    """中文说明：`omnistoreCheckpoint` 封装checkpoint 保存、恢复和合并工具中的状态和子模块。

    新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
    关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
    """
    def __init__(self, eval_milestone: List[Tuple[float, float]]):
        """中文说明：`__init__` 初始化checkpoint 保存、恢复和合并工具需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
        关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
        """
        self.time_stamp = torch.tensor([time.time() - 1e5, time.time()], device=dist.get_device())
        self.sp_also: subprocess.Popen = None
        self.sp_best: subprocess.Popen = None
        self.sp_backup: subprocess.Popen = None
        self.acc_str, self.eval_milestone = '[no acc str]', eval_milestone

    def sav(
        self, args: arg_util.Args, global_it: int, next_ep: int, next_it: int, fsdp_object: FSDP, optimizer_object: torch.optim.Optimizer,
        acc_str: Optional[str] = None, eval_milestone: Optional[List[Tuple[float, float]]] = None,
    ):
        """中文说明：`sav` 实现checkpoint 保存、恢复和合并工具中的 `sav` 步骤，供训练、推理或调试流程复用。

        新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
        关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
        """
        if acc_str is not None: self.acc_str = acc_str
        if eval_milestone is not None: self.eval_milestone = eval_milestone

        stt = time.time()

        checkpoint_state = {
            # 代码/形状说明：'model': {
                # 代码/形状说明：'main_model': fsdp_object,
                # 代码/形状说明：'ema_model': ema_fsdp_object,
            # },
            'model': fsdp_object,
            # 代码/形状说明：'optimizer': optimizer_object,
            'extra_state': {}
        }

        from omnistore import FSDPCheckpointer
        print(f"{FSDPCheckpointer=}")

        FSDPCheckpointer.save(
            path=args.bed,
            checkpoint_state=checkpoint_state,
            global_steps=global_it,
            async_fast_checkpoint=True,
            save_flatten_model_optimizer=True,
        )
        if dist.is_master():
            torch.save({
                'args': args.state_dict(),
                'next_ep': next_ep,
                'next_it': next_it,
                'global_it': global_it,
                'acc_str': self.acc_str,
                'milestones': self.eval_milestone,
            }, os.path.join(args.bed, 'meta.pth'))

            if self.sp_backup is not None:
                self.sp_backup.wait(timeout=300); self.sp_backup.kill(); self.sp_backup.communicate()
            self.time_stamp[0] = time.time()
            def auto_sync(source_filename, target_filename):
                """中文说明：`auto_sync` 实现checkpoint 保存、恢复和合并工具中的 `auto_sync` 步骤，供训练、推理或调试流程复用。

                新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
                关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
                """
                cmd = f'cp -r {source_filename} {target_filename}'
                self.sp_backup = subprocess.Popen(cmd, shell=True, bufsize=-1)
                print(f'[Saver] 自动同步命令: {cmd}', flush=True)
            local_files = glob.glob(f"{args.local_out_path}/*.txt")
            for filename in local_files:
                basename = os.path.basename(filename)
                target_filename = f'{args.bed}/{basename}'
                auto_sync(filename, target_filename)
            cost = time.time() - stt
        print(f'[CKPTSaver][rank00][omnistore: {FSDPCheckpointer is not None}] 耗时={time.time()-stt:.2f}s, ckpt 已保存到 global_step_{global_it}', flush=True)

        dist.barrier()
        del checkpoint_state

    def load(self, ckpt_path, fsdp_object, optimizer_object):
        """中文说明：`load` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

        新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
        关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
        """
        from omnistore import FSDPCheckpointer
        checkpoint_state = {
            'model': fsdp_object,
            # 代码/形状说明：'optimizer': optimizer_object,
            'extra_state': {}
        }
        FSDPCheckpointer.load(
            ckpt_path,
            checkpoint_state,
            load_flatten_model_optimizer=True,
        )
        global_it = -1
        meta_path = os.path.join(os.path.dirname(ckpt_path), 'meta.pth')
        if os.path.exists(meta_path):
            train_meta = torch.load(meta_path)
            args_state, next_ep, next_it, acc_str, milestones = train_meta['args'], train_meta['next_ep'], train_meta['next_it'], train_meta['acc_str'], train_meta['milestones']
            global_it = train_meta.get('global_it', -1)
        else:
            args_state, next_ep, next_it, acc_str, milestones = {}, 0, 0, '', []
        return args_state, next_ep, next_it, global_it, acc_str, milestones

def merge_ckpt(omnistore_ckpt_path, output_path, fsdp_save_flatten_model, save=False):
    """中文说明：`merge_ckpt` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：阅读时先看文件名如何编码 epoch/iter/global_step，再看 auto_resume 选择最新权重。
    关键关系：恢复优先级由文件名中的 epoch/iter/global_step 排序决定。
    """
    print(f'正在把 omnistore ckpt 合并为 torch-format ckpt')
    start = time.time()
    from omnistore.utilities.ckpt_format_tool import omnistore_ckpt_to_pytorch_ckpt
    state_dict = omnistore_ckpt_to_pytorch_ckpt(
        save_path=omnistore_ckpt_path,
        output_path=output_path,
        framework="fsdp",
        model_only=True,
        return_dict=True,
        fsdp_save_flatten_model=fsdp_save_flatten_model,
    )
    print(f"ckpt 合并完成，耗时 {time.time() - start:.2f} 秒")
    state_dict_model = state_dict["model"]
    if '.cfg_uncond' in state_dict_model:
        state_dict_model['cfg_uncond'] = state_dict_model['.cfg_uncond']
        del state_dict_model['.cfg_uncond']
    if '.pos_start' in state_dict_model:
        state_dict_model['pos_start'] = state_dict_model['.pos_start']
        del state_dict_model['.pos_start']
    if '.sos_token' in state_dict_model:
        state_dict_model['sos_token'] = state_dict_model['.sos_token']
        del state_dict_model['.sos_token']
    if 'semantic_head.weight' in state_dict_model:
        print(f'[rush_resume] 将 semantic_head 权重重命名为 semantic_head2')
        state_dict_model['semantic_head2.weight'] = state_dict_model['semantic_head.weight']
        state_dict_model['semantic_head2.bias'] = state_dict_model['semantic_head.bias']
        del state_dict_model['semantic_head.weight']
        del state_dict_model['semantic_head.bias']
    if save:
        save_file = os.path.join(output_path, "slim-model.pt")
        print(f'保存到 {save_file}')
        torch.save(state_dict_model, save_file)
    return state_dict_model
