# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch

def swish(x):
    """Swish/SiLU 激活函数，逐元素计算 `x * sigmoid(x)`。

    这里额外兼容列表输入和分时片回退路径，原因是某些 5D 张量在大分辨率下
    可能需要按时间维逐片计算以降低峰值显存。
    """
    if type(x) == list:
        for i in range(len(x)):
            x[i] = swish(x[i])
        return x
    try:
        return x*torch.sigmoid(x)
    except:
        for _i in range(x.shape[2]):
            x[:,:,_i:_i+1,:,:] = x[:,:,_i:_i+1,:,:]*torch.sigmoid(x[:,:,_i:_i+1,:,:])
        return x
