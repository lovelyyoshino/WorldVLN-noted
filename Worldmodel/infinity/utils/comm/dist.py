# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import torch
import torch.distributed as dist


# ====================
# 全互换通信（All-to-All）
# ====================
def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    """中文说明：`_all_to_all` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """中文说明：执行 all-to-all 通信。

    参数：
        input_：输入矩阵。
        process_group：通信进程组。
        scatter_dim：执行 scatter 的维度。
        gather_dim：执行 gather 的维度。
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        """中文说明：`forward` 执行Infinity 序列并行通信算子的前向计算；重点核对输入输出张量 shape 是否与调用方约定一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input_, ctx.world_size, process_group, scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    """中文说明：`all_to_all` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)


def _gather(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    gather_dim: int,
):
    """中文说明：`_gather` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    if gather_list is None:
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    dist.gather(input_, gather_list, group=group, gather_dim=gather_dim)
    return gather_list


# ====================
# 聚合与拆分通信（Gather-Split）
# ====================


def _split(input_, pg: dist.ProcessGroup, dim=-1):
    # 只有一个 rank 参与时跳过通信
    """中文说明：`_split` 实现Infinity 序列并行通信算子中的 `_split` 步骤，供训练、推理或调试流程复用。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    # 沿最后一维拆分。
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"要切分的维度大小 ({dim_size}) 不是 world size ({world_size}) 的倍数，"
        f"无法均匀切分 tensor"
    )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_, pg: dist.ProcessGroup, dim=-1):
    # 只有一个 rank 参与时跳过通信
    """中文说明：`_gather` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)
    dist.get_rank(pg)

    if world_size == 1:
        return input_

    # 聚合 all_gather 结果
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    assert input_.device.type == "cuda"
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # 拼接
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """中文说明：从模型并行区域 gather 输入并拼接。

    参数：
        input_：输入矩阵。.
        process_group：并行通信进程组。
        dim：通信作用的维度。
    """

    @staticmethod
    def symbolic(graph, input_):
        """中文说明：`symbolic` 实现Infinity 序列并行通信算子中的 `symbolic` 步骤，供训练、推理或调试流程复用。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return _gather(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        """中文说明：`forward` 执行Infinity 序列并行通信算子的前向计算；重点核对输入输出张量 shape 是否与调用方约定一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _gather(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)

        return _split(grad_output, ctx.mode, ctx.dim), None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    拆分输入，并只保留当前 rank 对应的 chunk。

    参数：
        input_：输入矩阵。.
        process_group：并行通信进程组。
        dim：通信作用的维度。
    """

    @staticmethod
    def symbolic(graph, input_):
        """中文说明：`symbolic` 实现Infinity 序列并行通信算子中的 `symbolic` 步骤，供训练、推理或调试流程复用。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return _split(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        """中文说明：`forward` 执行Infinity 序列并行通信算子的前向计算；重点核对输入输出张量 shape 是否与调用方约定一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _split(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim), None, None, None


def split_forward_gather_backward(input_, process_group, dim, grad_scale=1.0):
    """中文说明：`split_forward_gather_backward` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale)


def gather_forward_split_backward(input_, process_group, dim, grad_scale=None):
    """中文说明：`gather_forward_split_backward` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale)
