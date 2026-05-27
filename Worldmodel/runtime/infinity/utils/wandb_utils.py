# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math


def is_main_process():
    """中文说明：`is_main_process` 实现Weights & Biases 日志工具中的 `is_main_process` 步骤，供训练、推理或调试流程复用。

    新手提示：只在主进程初始化/记录，避免多卡重复写同一个 run。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    """中文说明：`namespace_to_dict` 实现Weights & Biases 日志工具中的 `namespace_to_dict` 步骤，供训练、推理或调试流程复用。

    新手提示：只在主进程初始化/记录，避免多卡重复写同一个 run。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # 参考：https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    """中文说明：`generate_run_id` 实现Weights & Biases 日志工具中的 `generate_run_id` 步骤，供训练、推理或调试流程复用。

    新手提示：只在主进程初始化/记录，避免多卡重复写同一个 run。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, entity, exp_name, project_name):
    """中文说明：`initialize` 实现Weights & Biases 日志工具中的 `initialize` 步骤，供训练、推理或调试流程复用。

    新手提示：只在主进程初始化/记录，避免多卡重复写同一个 run。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    config_dict = namespace_to_dict(args)
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
    )


def log(stats, step=None):
    """中文说明：`log` 实现Weights & Biases 日志工具中的 `log` 步骤，供训练、推理或调试流程复用。

    新手提示：只在主进程初始化/记录，避免多卡重复写同一个 run。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(name, sample, step=None):
    """中文说明：`log_image` 实现Weights & Biases 日志工具中的 `log_image` 步骤，供训练、推理或调试流程复用。

    新手提示：只在主进程初始化/记录，避免多卡重复写同一个 run。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"{name}": wandb.Image(sample), "train_step": step})


def array2grid(x):
    """中文说明：`array2grid` 实现Weights & Biases 日志工具中的 `array2grid` 步骤，供训练、推理或调试流程复用。

    新手提示：只在主进程初始化/记录，避免多卡重复写同一个 run。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x