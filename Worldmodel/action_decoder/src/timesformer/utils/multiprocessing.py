# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""多进程启动辅助函数。"""

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
    """在子进程中运行给定函数，是多进程启动的包装入口。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    # 初始化分布式进程组。
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
