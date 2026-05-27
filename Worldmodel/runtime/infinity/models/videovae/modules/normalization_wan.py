# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_norm(norm_type):
    """根据名称返回 WAN 版本归一化层类型。"""
    if norm_type == "spatial-group":
        return SpatialGroupNorm
    elif norm_type == "rms":
        return RMS_norm
    elif norm_type == "group":
        return nn.GroupNorm
    else:
        raise NotImplementedError

class RMS_norm(nn.Module):
    """WAN 版本的轻量 RMS 归一化层。"""

    def __init__(self, num_channels, channel_first=True, bias=False, **kwargs):
        """构造按通道均方根缩放的轻量归一化层。"""
        super().__init__()
        broadcastable_dims = (1, 1, 1)
        shape = (num_channels, *broadcastable_dims)

        self.channel_first = channel_first
        self.scale = num_channels**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        """执行 RMS 归一化并恢复可学习缩放。"""
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias

class SpatialGroupNorm(nn.GroupNorm):
    """适配 5D 视频张量的 WAN GroupNorm。"""
    def __init__(self, *args, **kwargs):
        """构造适配 5D 视频张量的 GroupNorm。"""
        super(SpatialGroupNorm, self).__init__(*args, **kwargs)

    def shard_norm(self, x):
        """按样本分片归一化，作为显存不足时的回退实现。"""
        dtype = x.dtype
        x = x.to(torch.float32)
        with torch.amp.autocast("cuda", torch.float32):
            for _i in range(x.shape[0]):
                x[_i:_i+1,...] = super(SpatialGroupNorm, self).forward(x[_i:_i+1,...])
        x = x.to(dtype=dtype)
        return x

    def forward(self, x):
        """把 `(B, C, T, H, W)` 展平成逐帧 2D 特征图后再做 GroupNorm。"""
        dtype = x.dtype
        x = x.to(torch.float32)
        assert x.ndim == 5
        T = x.shape[2]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        try:
            x = super(SpatialGroupNorm, self).forward(x)
        except:
            x = self.shard_norm(x) # 显存不足时改为逐样本归一化。
        x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
        x = x.to(dtype=dtype)
        return x

class Normalize(nn.Module):
    """WAN 路径使用的统一归一化封装。"""
    def __init__(self, in_channels, norm_type, norm_axis="spatial"):
        """统一封装 group/batch/no 三类归一化策略。"""
        super().__init__()
        self.norm_axis = norm_axis
        assert norm_type in ['group', 'batch', "no"], f"norm_type 只支持 group/batch/no，实际为 {norm_type}"
        if norm_type == 'group':
            if in_channels % 32 == 0:
                self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
            elif in_channels % 24 == 0:
                self.norm = nn.GroupNorm(num_groups=24, num_channels=in_channels, eps=1e-6, affine=True)
            else:
                raise NotImplementedError(f"group norm 要求 in_channels 能被 32 或 24 整除，实际为 {in_channels}")
        elif norm_type == 'batch':
            self.norm = nn.SyncBatchNorm(in_channels, track_running_stats=False) # 若为 True，训练中可能触发 inplace grad 报错。
        elif norm_type == 'no':
            self.norm = nn.Identity()

    def _norm(self, x):
        """执行底层归一化，并在异常时回退到 CPU。"""
        try:
            x = self.norm(x)
        except:
            device = x.device
            self.norm_cpu = self.norm.cpu()
            x = self.norm_cpu(x.cpu().pin_memory()).to(device=device)
        return x

    def shard_norm(self, x):
        """按样本分批归一化，降低峰值显存。"""
        dtype = x.dtype
        x = x.to(torch.float32)
        with torch.amp.autocast("cuda", torch.float32):
            for _i in range(x.shape[0]):
                x[_i:_i+1,...] = self.norm(x[_i:_i+1,...])
        x = x.to(dtype=dtype)
        return x

    def forward(self, x):
        """按空间轴或时空整体执行归一化。"""
        if self.norm_axis == "spatial":
            if type(x) == list:
                for i in range(len(x)):
                    x[i] = self.norm(x[i])
                return x
            if x.ndim == 4:
                try:
                    x = self.norm(x)
                except:
                    x = self.shard_norm(x)
            else:
                B, C, T, H, W = x.shape
                x = rearrange(x, "B C T H W -> (B T) C H W")
                # 代码/形状说明：x = self.shard_norm(x)
                try:
                    x = self.norm(x)
                except:
                    x = self.shard_norm(x)
                x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
        elif self.norm_axis == "spatial-temporal":
            x = self._norm(x)
        else:
            raise NotImplementedError
        return x

def l2norm(t):
    """沿最后一维执行 L2 归一化。"""
    return F.normalize(t, dim=-1)

class LayerNorm(nn.Module):
    """最后一维 LayerNorm 封装。"""
    def __init__(self, dim):
        """构造最后一维 LayerNorm 所需的可学习参数。"""
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        """对最后一维执行 LayerNorm。"""
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 代码/形状说明：https://github.com/huggingface/transformers/blob/2f12e408225b1ebceb0d2f701ce419d46678dc31/src/transformers/models/llama/modeling_llama.py#L76
class RMSNorm(nn.Module):
    """Llama 风格 RMSNorm 封装。"""
    def __init__(self, hidden_size, eps=1e-6):
        """构造 Llama 风格 RMSNorm，语义上等价于 T5LayerNorm。"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, sp_slice=None):
        """执行 RMS 归一化，并支持只对切片通道应用权重。"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if sp_slice is None:
            return (self.weight * hidden_states).to(input_dtype)
        else:
            return (self.weight[sp_slice] * hidden_states).to(input_dtype)  # 代码/形状说明：torch.float32 * torchbfloat16 in DDP will cast to torch.float32
