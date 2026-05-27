# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup

if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        """中文说明：`noop_custom_op_wrapper` 实现Infinity 序列并行通信算子中的 `noop_custom_op_wrapper` 步骤，供训练、推理或调试流程复用。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        def wrap(func):
            """中文说明：`wrap` 实现Infinity 序列并行通信算子中的 `wrap` 步骤，供训练、推理或调试流程复用。

            新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
            关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
            """
            return func
        if fn is None:
            return wrap
        return fn
    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        """中文说明：`noop_register_fake_wrapper` 实现Infinity 序列并行通信算子中的 `noop_register_fake_wrapper` 步骤，供训练、推理或调试流程复用。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        def wrap(func):
            """中文说明：`wrap` 实现Infinity 序列并行通信算子中的 `wrap` 步骤，供训练、推理或调试流程复用。

            新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
            关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
            """
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


__sp_comm_group__ = None

def set_sp_comm_group(group=None):
    """中文说明：`set_sp_comm_group` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    global __sp_comm_group__
    assert __sp_comm_group__ is None and group is not None
    __sp_comm_group__ = group

def get_sp_comm_group():
    """中文说明：`get_sp_comm_group` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    global __sp_comm_group__
    assert __sp_comm_group__ is not None
    return __sp_comm_group__


# ======================================================
# 模型
# ======================================================


def model_sharding(model: torch.nn.Module):
    """中文说明：`model_sharding` 实现Infinity 序列并行通信算子中的 `model_sharding` 步骤，供训练、推理或调试流程复用。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for _, param in model.named_parameters():
        padding_size = (world_size - param.numel() % world_size) % world_size
        if padding_size > 0:
            padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
        else:
            padding_param = param.data.view(-1)
        splited_params = padding_param.split(padding_param.numel() // world_size)
        splited_params = splited_params[global_rank]
        param.data = splited_params


# ======================================================
# 聚合 AllGather 与切分规约 ReduceScatter
# ======================================================


class AsyncAllGatherForTwo(torch.autograd.Function):
    """中文说明：`AsyncAllGatherForTwo` 封装Infinity 序列并行通信算子中的状态和子模块。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        weight: Tensor,
        bias: Tensor,
        sp_rank: int,
        sp_size: int,
        group: Optional[ProcessGroup] = None,
    ) -> Tuple[Tensor, Any]:
        """
        返回：
            outputs：输出张量。
            handle：当 overlap=True 时返回异步 Work 句柄。
        """
        from torch.distributed._functional_collectives import all_gather_tensor

        ctx.group = group
        ctx.sp_rank = sp_rank
        ctx.sp_size = sp_size

        # 聚合 all_gather 输入
        all_inputs = all_gather_tensor(inputs.unsqueeze(0), 0, group)
        # 计算本地 qkv
        local_qkv = F.linear(inputs, weight, bias).unsqueeze(0)

        # 远端计算
        remote_inputs = all_inputs[1 - sp_rank].view(list(local_qkv.shape[:-1]) + [-1])
        # 计算远端 qkv
        remote_qkv = F.linear(remote_inputs, weight, bias)

        # 拼接本地与远端 qkv
        if sp_rank == 0:
            qkv = torch.cat([local_qkv, remote_qkv], dim=0)
        else:
            qkv = torch.cat([remote_qkv, local_qkv], dim=0)
        qkv = rearrange(qkv, "sp b n c -> b (sp n) c")

        ctx.save_for_backward(inputs, weight, remote_inputs)
        return qkv

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        from torch.distributed._functional_collectives import reduce_scatter_tensor

        group = ctx.group
        sp_rank = ctx.sp_rank
        sp_size = ctx.sp_size
        inputs, weight, remote_inputs = ctx.saved_tensors

        # 拆分 qkv 梯度
        qkv_grad = grad_outputs[0]
        qkv_grad = rearrange(qkv_grad, "b (sp n) c -> sp b n c", sp=sp_size)
        qkv_grad = torch.chunk(qkv_grad, 2, dim=0)
        if sp_rank == 0:
            local_qkv_grad, remote_qkv_grad = qkv_grad
        else:
            remote_qkv_grad, local_qkv_grad = qkv_grad

        # 计算远端梯度
        remote_inputs_grad = torch.matmul(remote_qkv_grad, weight).squeeze(0)
        weight_grad = torch.matmul(remote_qkv_grad.transpose(-1, -2), remote_inputs).squeeze(0).sum(0)
        bias_grad = remote_qkv_grad.squeeze(0).sum(0).sum(0)

        # 启动异步 reduce_scatter
        remote_inputs_grad_zero = torch.zeros_like(remote_inputs_grad)
        if sp_rank == 0:
            remote_inputs_grad = torch.cat([remote_inputs_grad_zero, remote_inputs_grad], dim=0)
        else:
            remote_inputs_grad = torch.cat([remote_inputs_grad, remote_inputs_grad_zero], dim=0)
        remote_inputs_grad = reduce_scatter_tensor(remote_inputs_grad, "sum", 0, group)

        # 计算本地梯度并等待 reduce_scatter
        local_input_grad = torch.matmul(local_qkv_grad, weight).squeeze(0)
        weight_grad += torch.matmul(local_qkv_grad.transpose(-1, -2), inputs).squeeze(0).sum(0)
        bias_grad += local_qkv_grad.squeeze(0).sum(0).sum(0)

        # 合并远端与本地梯度
        inputs_grad = remote_inputs_grad + local_input_grad
        return inputs_grad, weight_grad, bias_grad, None, None, None


class AllGather(torch.autograd.Function):
    """中文说明：`AllGather` 封装Infinity 序列并行通信算子中的状态和子模块。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: Optional[ProcessGroup] = None,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        返回：
            outputs：输出张量。
            handle：当 overlap=True 时返回异步 Work 句柄。
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group

        comm_size = dist.get_world_size(group)
        if comm_size == 1:
            return inputs.unsqueeze(0), None

        buffer_shape = (comm_size,) + inputs.shape
        outputs = torch.empty(buffer_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(outputs, comm_size, dim=0))
        if not overlap:
            dist.all_gather(buffer_list, inputs, group=group)
            return outputs, None
        else:
            handle = dist.all_gather(buffer_list, inputs, group=group, async_op=True)
            return outputs, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return (
            ReduceScatter.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


class ReduceScatter(torch.autograd.Function):
    """中文说明：`ReduceScatter` 封装Infinity 序列并行通信算子中的状态和子模块。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: ProcessGroup,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        返回：
            outputs：输出张量。
            handle：当 overlap=True 时返回异步 Work 句柄。
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group

        comm_size = dist.get_world_size(group)
        if comm_size == 1:
            return inputs.squeeze(0), None

        if not inputs.is_contiguous():
            inputs = inputs.contiguous()

        output_shape = inputs.shape[1:]
        outputs = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(inputs, comm_size, dim=0))
        if not overlap:
            dist.reduce_scatter(outputs, buffer_list, group=group)
            return outputs, None
        else:
            handle = dist.reduce_scatter(outputs, buffer_list, group=group, async_op=True)
            return outputs, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        # 待办：支持异步 backward
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return (
            AllGather.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


# ======================================================
# 全互换通信（All-to-All）
# ======================================================


@_torch_custom_op_wrapper("distributed::_all_to_all_func", mutates_args=(), device_types="cuda")
def _all_to_all_func(input_: torch.Tensor, world_size: int = 1, scatter_dim: int = 0, gather_dim: int = 0) -> torch.Tensor:
    """中文说明：`_all_to_all_func` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    group = get_sp_comm_group()
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


@_torch_register_fake_wrapper("distributed::_all_to_all_func")
def _all_to_all_func_fake(input_: torch.Tensor, world_size: int = 1, scatter_dim: int = 0, gather_dim: int = 0) -> torch.Tensor:
    """中文说明：`_all_to_all_func_fake` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    inp_shape = list(input_.shape)
    group = get_sp_comm_group()
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    inp_shape[gather_dim] = inp_shape[gather_dim] * world_size
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // world_size
    outputs = torch.empty(torch.Size(inp_shape), dtype=input_.dtype, device=input_.device, layout=input_.layout)
    return outputs


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
        world_size = dist.get_world_size(process_group)

        return _wrapper_all_to_all_func(input_, world_size, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return (return_grad, None, None, None)


def all_to_all_comm(input_, process_group=None, scatter_dim=2, gather_dim=1):
    """中文说明：`all_to_all_comm` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)


# ======================================================
# 序列维 gather 与 split
# ======================================================


def _split_sequence_func(inputs, pg: dist.ProcessGroup, dim=-1):
    """中文说明：`_split_sequence_func` 实现Infinity 序列并行通信算子中的 `_split_sequence_func` 步骤，供训练、推理或调试流程复用。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return inputs

    # 沿最后一维拆分。
    rank = dist.get_rank(pg)
    dim_size = inputs.size(dim)
    assert dim_size % world_size == 0, (
        f"要切分的维度大小 ({dim_size}) 不是 world size ({world_size}) 的倍数，"
        f"无法均匀切分 tensor"
    )

    outputs = torch.split(inputs, dim_size // world_size, dim=dim)[rank]
    return outputs


@_torch_custom_op_wrapper("distributed::_gather_sequence_func", mutates_args=(), device_types="cuda")
def _gather_sequence_func(inputs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """中文说明：`_gather_sequence_func` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    pg = get_sp_comm_group()
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return inputs

    # 聚合 all_gather 结果
    inputs = inputs.contiguous()
    outputs = [torch.empty_like(inputs) for _ in range(world_size)]
    dist.all_gather(outputs, inputs, group=pg)

    # 拼接
    outputs = torch.cat(outputs, dim=dim)
    return outputs


@_torch_register_fake_wrapper("distributed::_gather_sequence_func")
def _gather_sequence_func_fake(inputs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """中文说明：`_gather_sequence_func_fake` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    inp_shape = list(inputs.shape)
    pg = get_sp_comm_group()
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return inputs

    inp_shape[dim] = inp_shape[dim] * world_size
    outputs = torch.empty(torch.Size(inp_shape), dtype=inputs.dtype, device=inputs.device, layout=inputs.layout)
    return outputs


if torch.__version__ >= "2.4.0":
    _wrapper_all_to_all_func = torch.ops.distributed._all_to_all_func
    _wrapper_gather_sequence_func = torch.ops.distributed._gather_sequence_func
else:
    _wrapper_all_to_all_func = _all_to_all_func
    _wrapper_gather_sequence_func = _gather_sequence_func


class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    聚合输入序列。

    参数：
        input_：输入矩阵。.
        process_group：进程组。
        dim：通信作用的维度。
    """

    @staticmethod
    def symbolic(graph, input_):
        """中文说明：`symbolic` 实现Infinity 序列并行通信算子中的 `symbolic` 步骤，供训练、推理或调试流程复用。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return _wrapper_gather_sequence_func(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        """中文说明：`forward` 执行Infinity 序列并行通信算子的前向计算；重点核对输入输出张量 shape 是否与调用方约定一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _wrapper_gather_sequence_func(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)

        return _split_sequence_func(grad_output, ctx.process_group, ctx.dim), None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    拆分序列。

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
        return _split_sequence_func(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        """中文说明：`forward` 执行Infinity 序列并行通信算子的前向计算；重点核对输入输出张量 shape 是否与调用方约定一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _split_sequence_func(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)
        return _wrapper_gather_sequence_func(grad_output, ctx.dim), None, None, None


def split_sequence(input_, process_group, dim, grad_scale=1.0):
    """中文说明：`split_sequence` 实现Infinity 序列并行通信算子中的 `split_sequence` 步骤，供训练、推理或调试流程复用。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale)


def gather_sequence(input_, process_group, dim, grad_scale=None):
    """中文说明：`gather_sequence` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale)
