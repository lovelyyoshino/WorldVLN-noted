# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch.distributed as dist
from torch.distributed import ReduceOp


class AsyncOpList(object):
    """中文说明：`AsyncOpList` 封装可求导分布式通信封装中的状态和子模块。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    def __init__(self, ops):
        """中文说明：`__init__` 初始化可求导分布式通信封装需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        self.ops = ops

    def wait(self):
        """中文说明：`wait` 实现可求导分布式通信封装中的 `wait` 步骤，供训练、推理或调试流程复用。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        for op in self.ops:
            op.wait()

    def is_completed(self):
        """中文说明：`is_completed` 实现可求导分布式通信封装中的 `is_completed` 步骤，供训练、推理或调试流程复用。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        for op in self.ops:
            if not op.is_completed():
                return False
        return True


def reduce_scatter(tensor,
                   tensor_list,
                   op=ReduceOp.SUM,
                   group=dist.group.WORLD,
                   async_op=False):
    """中文说明：`reduce_scatter` 执行分布式通信包装；阅读时把张量切分维度、`rank` 方向和 `backward` 的反向通信配对检查。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    ranks = dist.get_process_group_ranks(group)
    rank = dist.get_rank(group)
    if tensor is None:
        tensor = tensor_list[rank]
    if tensor.dim() == 0:
        tensor = tensor.view(-1)
    tensor[:] = tensor_list[rank]
    ops = []
    for i in range(dist.get_world_size(group)):
        if i == rank:
            tmp = dist.reduce(tensor.contiguous(), ranks[i], op, group, async_op=True)
        else:
            tmp = dist.reduce(tensor_list[i].contiguous(), ranks[i], op, group, async_op=True)
        ops.append(tmp)

    oplist = AsyncOpList(ops)
    if async_op:
        return oplist
    else:
        oplist.wait()
