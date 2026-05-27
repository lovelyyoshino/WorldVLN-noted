# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch.nn as nn

def rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    """中文说明：`rnn_flops` 计算或汇总 FLOPs/TFLOPS/MFU；公式通常是 FLOPs 除以耗时和 1e12，再与设备峰值相除。

    新手提示：常见公式是矩阵乘 FLOPs≈2*M*N*K，卷积 FLOPs≈2*out_elements*kernel_mul*in_channels/groups。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    gates_size = w_ih.shape[0]
    # 输入 ih 状态与内部状态的矩阵乘
    flops += 2 * w_ih.shape[0] * w_ih.shape[1] - gates_size
    # 隐状态 hh 与内部状态的矩阵乘
    flops += 2 * w_hh.shape[0] * w_hh.shape[1] - gates_size
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # 累加两个操作的 FLOPs
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # 门控 r 的 Hadamard 逐元素乘
        flops += rnn_module.hidden_size
        # 累加两个状态分支的操作量
        flops += rnn_module.hidden_size * 3
        # 最后两次 Hadamard 乘法和加法
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # 累加两个状态分支的操作量
        flops += rnn_module.hidden_size * 4
        # 单元状态 C 的两次 Hadamard 乘法和加法
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # 最后的 Hadamard 逐元素乘
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def rnn_forward_hook(rnn_module, input, output):
    """中文说明：`rnn_forward_hook` 实现FLOPs 计算公式实现中的 `rnn_forward_hook` 步骤，供训练、推理或调试流程复用。

    新手提示：常见公式是矩阵乘 FLOPs≈2*M*N*K，卷积 FLOPs≈2*out_elements*kernel_mul*in_channels/groups。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    flops = 0
    # 输入是一个元组，包含待处理序列以及可选隐藏状态
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__("weight_ih_l" + str(i))
        w_hh = rnn_module.__getattr__("weight_hh_l" + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__("bias_ih_l" + str(i))
            b_hh = rnn_module.__getattr__("bias_hh_l" + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def rnn_cell_forward_hook(rnn_cell_module, input, output):
    """中文说明：`rnn_cell_forward_hook` 实现FLOPs 计算公式实现中的 `rnn_cell_forward_hook` 步骤，供训练、推理或调试流程复用。

    新手提示：常见公式是矩阵乘 FLOPs≈2*M*N*K，卷积 FLOPs≈2*out_elements*kernel_mul*in_channels/groups。
    关键公式：TFLOPS = FLOPs / seconds / 1e12，MFU = TFLOPS / device_peak_TFLOPS。
    """
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__("weight_ih")
    w_hh = rnn_cell_module.__getattr__("weight_hh")
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__("bias_ih")
        b_hh = rnn_cell_module.__getattr__("bias_hh")
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {
    # 循环神经网络 RNN
    nn.RNN: rnn_forward_hook,
    nn.GRU: rnn_forward_hook,
    nn.LSTM: rnn_forward_hook,
    nn.RNNCell: rnn_cell_forward_hook,
    nn.LSTMCell: rnn_cell_forward_hook,
    nn.GRUCell: rnn_cell_forward_hook,
}
