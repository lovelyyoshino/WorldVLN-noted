# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import infinity.models.videovae.utils.diffdist.modules as mods
import torch.distributed as dist


def consume_variable(tensor_to_consume, tensors_to_return, set_ones_grad=True):
    """中文说明：`consume_variable` 实现可求导分布式通信封装中的 `consume_variable` 步骤，供训练、推理或调试流程复用。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    return mods.ConsumeVariable(set_ones_grad)(tensor_to_consume,
                                               *tensors_to_return)


def send(tensor, dst, group=dist.group.WORLD, tag=0):
    """中文说明：`send` 执行分布式通信包装；阅读时把张量切分维度、`rank` 方向和 `backward` 的反向通信配对检查。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    return mods.Send(dst, group, tag)(tensor)


def recv(tensor,
         src=None,
         group=dist.group.WORLD,
         tag=0,
         next_backprop=None,
         inplace=True):
    """中文说明：`recv` 执行分布式通信包装；阅读时把张量切分维度、`rank` 方向和 `backward` 的反向通信配对检查。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    return mods.Recv(src, group, tag, next_backprop, inplace)(tensor)


def broadcast(tensor,
              src,
              group=dist.group.WORLD,
              next_backprop=None,
              inplace=True):
    """中文说明：`broadcast` 执行分布式通信包装；阅读时把张量切分维度、`rank` 方向和 `backward` 的反向通信配对检查。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    return mods.Broadcast(src, group, next_backprop, inplace)(tensor)


def gather(tensor,
           gather_list=None,
           dst=None,
           group=dist.group.WORLD,
           next_backprop=None,
           inplace=True):
    """中文说明：`gather` 执行分布式通信包装；阅读时把张量切分维度、`rank` 方向和 `backward` 的反向通信配对检查。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    return mods.Gather(dst, group, next_backprop, inplace)(tensor, gather_list)


def scatter(tensor,
            scatter_list=None,
            src=None,
            group=dist.group.WORLD,
            next_backprop=None,
            inplace=True):
    """中文说明：`scatter` 执行分布式通信包装；阅读时把张量切分维度、`rank` 方向和 `backward` 的反向通信配对检查。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    return mods.Scatter(src, group, next_backprop, inplace)(tensor,
                                                            scatter_list)


def all_gather(gather_list,
               tensor,
               group=dist.group.WORLD,
               next_backprop=None,
               inplace=True):
    """中文说明：`all_gather` 执行分布式通信包装；阅读时把张量切分维度、`rank` 方向和 `backward` 的反向通信配对检查。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    return mods.AllGather(group, next_backprop, inplace)(gather_list, tensor)
