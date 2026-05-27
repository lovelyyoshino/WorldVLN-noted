# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

# 来源：https://github.com/FoundationVision/LlamaGen/blob/main/utils/distributed.py
import os
import sys
import glob
import torch
import subprocess
import torch.distributed as dist
import datetime
import logging

from infinity.models.videovae.utils.misc import rank_zero_only, COLOR_BLUE, COLOR_RESET

from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from infinity.models.videovae.models.cvivit_vqgan import CViViT_Decoder, CViViT_Encoder


def setup_for_distributed(is_master, logging_dir=""):
    """
    该函数在非 master 进程上关闭普通打印，避免多卡重复输出。
    并把 stdout 重定向到 `log_out.txt`，把 stderr 重定向到 `log_err.txt`。
    """
    import builtins as __builtin__

    class Logger(logging.StreamHandler):
        """中文说明：`Logger` 封装VideoVAE 分布式初始化与 loss 规约工具中的状态和子模块。

        新手提示：重点看 rank/world size、FSDP 包装和 loss 平均，避免把单卡行为误读成多卡行为。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        def __init__(self, stream, file):
            """中文说明：`__init__` 初始化VideoVAE 分布式初始化与 loss 规约工具需要的配置、缓存或子模块，不直接执行训练/推理主循环。

            新手提示：重点看 rank/world size、FSDP 包装和 loss 平均，避免把单卡行为误读成多卡行为。
            阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
            """
            super().__init__(stream)
            self.file = file

        def emit(self, record):
            """中文说明：`emit` 实现VideoVAE 分布式初始化与 loss 规约工具中的 `emit` 步骤，供训练、推理或调试流程复用。

            新手提示：重点看 rank/world size、FSDP 包装和 loss 平均，避免把单卡行为误读成多卡行为。
            阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
            """
            try:
                msg = self.format(record)
                stream = self.stream
                fs = "%s\n"

                # 写入原始输出流并刷新
                stream.write(fs % msg)
                stream.flush()

                # 写入日志文件并刷新
                self.file.write(fs % msg)
                self.file.flush()
            except Exception as e:
                self.handleError(record)

        def isatty(self):
            # 模拟文件对象常见的 isatty 方法
            """中文说明：`isatty` 实现VideoVAE 分布式初始化与 loss 规约工具中的 `isatty` 步骤，供训练、推理或调试流程复用。

            新手提示：重点看 rank/world size、FSDP 包装和 loss 平均，避免把单卡行为误读成多卡行为。
            阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
            """
            return self.stream.isatty()

    # 只让 rank 0 打印
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        """中文说明：`print` 实现VideoVAE 分布式初始化与 loss 规约工具中的 `print` 步骤，供训练、推理或调试流程复用。

        新手提示：重点看 rank/world size、FSDP 包装和 loss 平均，避免把单卡行为误读成多卡行为。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

    if is_master:
        os.makedirs(logging_dir, exist_ok=True)
        existing_logs = glob.glob(os.path.join(logging_dir, 'log_out_*.txt'))
        log_numbers = [int(log.split('.txt')[0].split('_')[-1]) for log in existing_logs]
        next_log_number = max(log_numbers) + 1 if log_numbers else 1

        log_out_path = os.path.join(logging_dir, f'log_out_{next_log_number}.txt')
        log_err_path = os.path.join(logging_dir, f'log_err_{next_log_number}.txt')

        logger_stdout = Logger(sys.stdout, open(log_out_path, 'w'))
        logger_stderr = Logger(sys.stderr, open(log_err_path, 'w'))
        logging.basicConfig(level=logging.DEBUG, handlers=[logger_stdout, logger_stderr])

        print(f"{COLOR_BLUE}stdout 将写入 {log_out_path}{COLOR_RESET}")
        print(f"{COLOR_BLUE}stderr 将写入 {log_err_path}{COLOR_RESET}")

def init_distributed_mode(args, timeout_minutes=15):
    """中文说明：`init_distributed_mode` 实现VideoVAE 分布式初始化与 loss 规约工具中的 `init_distributed_mode` 步骤，供训练、推理或调试流程复用。

    新手提示：重点看 rank/world size、FSDP 包装和 loss 平均，避免把单卡行为误读成多卡行为。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('未使用 distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed 初始化 (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank,
                                         timeout=datetime.timedelta(seconds=timeout_minutes * 60)
                                         )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0, args.default_root_dir)

def _FSDP(model: torch.nn.Module, device, zero) -> FSDP:
    """中文说明：`_FSDP` 实现VideoVAE 分布式初始化与 loss 规约工具中的 `_FSDP` 步骤，供训练、推理或调试流程复用。

    新手提示：重点看 rank/world size、FSDP 包装和 loss 平均，避免把单卡行为误读成多卡行为。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy([CViViT_Encoder, CViViT_Decoder]),
        device_id=device,
        sharding_strategy={1:ShardingStrategy.HYBRID_SHARD, 2:ShardingStrategy.SHARD_GRAD_OP, 3:ShardingStrategy.FULL_SHARD}.get(zero),
        mixed_precision=MixedPrecision(
            param_dtype=torch.float,
            reduce_dtype=torch.float,
            buffer_dtype=torch.float,
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def reduce_losses(loss_dict, dst=0):
    """中文说明：`reduce_losses` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点看 rank/world size、FSDP 包装和 loss 平均，避免把单卡行为误读成多卡行为。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    loss_names = list(loss_dict.keys())
    loss_tensor = torch.stack([loss_dict[name] for name in loss_names])

    dist.reduce(loss_tensor, dst=dst, op=dist.ReduceOp.SUM)
    # 只在目标 rank 上平均 loss 值
    if dist.get_rank() == dst:
        loss_tensor /= dist.get_world_size()
        averaged_losses = {name: loss_tensor[i].item() for i, name in enumerate(loss_names)}
    else:
        averaged_losses = {name: None for name in loss_names}

    return averaged_losses

@rank_zero_only
def average_losses(loss_dict_list):
    """中文说明：`average_losses` 实现VideoVAE 分布式初始化与 loss 规约工具中的 `average_losses` 步骤，供训练、推理或调试流程复用。

    新手提示：重点看 rank/world size、FSDP 包装和 loss 平均，避免把单卡行为误读成多卡行为。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    sum_dict = {}
    count_dict = {}
    for loss_dict in loss_dict_list:
        for key, value in loss_dict.items():
            if key in sum_dict:
                sum_dict[key] += value
                count_dict[key] += 1
            else:
                sum_dict[key] = value
                count_dict[key] = 1

    avg_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
    return avg_dict
