# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""多进程辅助函数。"""

import torch


def run(
    local_rank,
    num_proc,
    func,
    init_method,
    shard_id,
    num_shards,
    backend,
    cfg,
    output_queue=None,
):
    """
    在子进程中运行目标函数。

    参数：
        local_rank (int): 当前进程在本机上的 rank。
        num_proc (int): 每台机器启动的进程数。
        func (function): 每个进程需要执行的函数。
        init_method (string): 分布式初始化方法。
            TCP 初始化要求所有进程都能访问同一个网络地址和端口；
            共享文件系统初始化要求所有机器都能看到同一个共享路径，且 URL 需以
            `file://` 开头并指向一个不存在的文件。
        shard_id (int): 当前机器的 rank。
        num_shards (int): 分布式训练的机器总数。
        backend (string): 分布式后端，可选 `nccl`、`gloo`、`mpi`，
            细节见 https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): 配置对象，细节见 `slowfast/config/defaults.py`。
        output_queue (queue): 可选，用于从 master 进程回传结果。
    """
    # 初始化进程组。
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e

    torch.cuda.set_device(local_rank)
    ret = func(cfg)
    if output_queue is not None and local_rank == 0:
        output_queue.put(ret)
