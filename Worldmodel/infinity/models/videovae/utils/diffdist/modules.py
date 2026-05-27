# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch.nn as nn
import torch.distributed as dist
import infinity.models.videovae.utils.diffdist.functions as funcs


class ConsumeVariable(nn.Module):
    """中文说明：`ConsumeVariable` 封装可求导分布式通信封装中的状态和子模块。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    def __init__(self, set_ones_grad=True):
        """
        如果 `set_ones_grad=True`，则 `tensor_to_consume` 对应梯度会被置为全 1。
        在反向传播时会设为 1，否则设为 0。
        """
        super(ConsumeVariable, self).__init__()
        self.set_ones_grad = set_ones_grad

    def forward(self, tensor_to_consume, *tensors_to_return):
        """中文说明：`forward` 执行可求导分布式通信封装的前向计算；重点核对输入输出张量 `shape` 是否与调用方约定一致。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        tensors_to_return = funcs.ConsumeVariableFunc.apply(
            tensor_to_consume, self.set_ones_grad, *tensors_to_return)
        return tensors_to_return


class Send(nn.Module):
    """中文说明：`Send` 封装可求导分布式通信封装中的状态和子模块。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    def __init__(self, dst, group=dist.group.WORLD, tag=0):
        """中文说明：`__init__` 初始化可求导分布式通信封装需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        super(Send, self).__init__()
        self.dst = dst
        self.group = group
        self.tag = tag

    def forward(self, tensor):
        """中文说明：`forward` 执行可求导分布式通信封装的前向计算；重点核对输入输出张量 `shape` 是否与调用方约定一致。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        return funcs.SendFunc.apply(tensor, self.dst, self.group, self.tag)


class Recv(nn.Module):
    """中文说明：`Recv` 封装可求导分布式通信封装中的状态和子模块。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    def __init__(self,
                 src=None,
                 group=dist.group.WORLD,
                 tag=0,
                 next_backprop=None,
                 inplace=True):
        """中文说明：`__init__` 初始化可求导分布式通信封装需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        super(Recv, self).__init__()
        self.next_backprop = next_backprop
        self.src = src
        self.group = group
        self.tag = tag
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor):
        """中文说明：`forward` 执行可求导分布式通信封装的前向计算；重点核对输入输出张量 `shape` 是否与调用方约定一致。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        tensor, sender = funcs.RecvFunc.apply(tensor, self.src, self.group,
                                              self.tag, self.inplace)
        return tensor, sender.item()


class Broadcast(nn.Module):
    """中文说明：`Broadcast` 封装可求导分布式通信封装中的状态和子模块。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    def __init__(self,
                 src,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        """中文说明：`__init__` 初始化可求导分布式通信封装需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        super(Broadcast, self).__init__()
        self.src = src
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor):
        """中文说明：`forward` 执行可求导分布式通信封装的前向计算；重点核对输入输出张量 `shape` 是否与调用方约定一致。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        return funcs.BroadcastFunc.apply(tensor, self.src, self.group,
                                         self.inplace)


class Gather(nn.Module):
    """中文说明：`Gather` 封装可求导分布式通信封装中的状态和子模块。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    def __init__(self,
                 dst=None,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        """中文说明：`__init__` 初始化可求导分布式通信封装需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        super(Gather, self).__init__()
        self.dst = dst
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor, gather_list=None):
        """中文说明：`forward` 执行可求导分布式通信封装的前向计算；重点核对输入输出张量 `shape` 是否与调用方约定一致。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        if dist.get_rank(self.group) == self.dst:
            return list(
                funcs.GatherFunc.apply(tensor, self.dst, self.group,
                                       self.inplace, *gather_list))
        else:
            return funcs.GatherFunc.apply(tensor, self.dst, self.group,
                                          self.inplace, None)


class Scatter(nn.Module):
    """中文说明：`Scatter` 封装可求导分布式通信封装中的状态和子模块。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    def __init__(self,
                 src=None,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        """中文说明：`__init__` 初始化可求导分布式通信封装需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        super(Scatter, self).__init__()
        self.src = src
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor, scatter_list=None):
        """中文说明：`forward` 执行可求导分布式通信封装的前向计算；重点核对输入输出张量 `shape` 是否与调用方约定一致。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        if dist.get_rank(self.group) == self.src:
            return funcs.ScatterFunc.apply(tensor, self.src, self.group,
                                           self.inplace, *scatter_list)
        else:
            return funcs.ScatterFunc.apply(tensor, self.src, self.group,
                                           self.inplace, None)


class AllGather(nn.Module):
    """中文说明：`AllGather` 封装可求导分布式通信封装中的状态和子模块。

    新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
    关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
    """
    def __init__(self,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        """中文说明：`__init__` 初始化可求导分布式通信封装需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        super(AllGather, self).__init__()
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, gather_list, tensor):
        """中文说明：`forward` 执行可求导分布式通信封装的前向计算；重点核对输入输出张量 `shape` 是否与调用方约定一致。

        新手提示：这些包装把 `send`/`recv`/`gather`/`scatter` 接入自动求导，阅读时要把 `forward` 的通信和 `backward` 的梯度回传成对看。
        关键关系：`forward` 负责数据分发或聚合，`backward` 通常执行互逆通信来传回梯度。
        """
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        return list(
            funcs.AllGatherFunc.apply(tensor, self.group, self.inplace,
                                      *gather_list))
