# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import math
from operator import truediv
import torch
import torch.nn.functional as F
from torch import nn, einsum
from beartype import beartype
from typing import Tuple

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from infinity.models.videovae.modules.drop_path import DropPath

from fairscale.nn import checkpoint_wrapper
from torch.nn.attention import SDPBackend, sdpa_kernel
from infinity.models.videovae.utils.misc import is_dtype_16
from infinity.models.videovae.modules.normalization import l2norm, LayerNorm, RMSNorm


def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    """在 token 序列上执行分组最大池化。

    输入形状为 `(B, N, C)`。函数把序列维按 `stride` 分组后取最大值，
    可以理解为“在 token 轴上的 max-pool”。
    """
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values


def exists(val):
    """判断值是否不为 `None`。"""
    return val is not None


def default(val, d):
    """若 `val` 存在则返回它，否则返回默认值 `d`。"""
    return val if exists(val) else d


def leaky_relu(p=0.1):
    """返回位置偏置网络使用的激活层。

    当前实现退化为恒等映射，保留这个函数是为了与原始结构接口兼容。
    """
    return nn.Identity()


def precompute_freqs_cis_2d(dim: int, end: int, H, W, theta: float = 10000.0, scale=1.0, use_cls=False):
    """预计算 2D RoPE 复数相位。

    这里把平面位置 `(x, y)` 映射成复平面旋转因子，后续会作用到 query/key 上。
    对初学者来说，可以把它理解为“把二维位置信息编码进向量旋转角度”。
    """
    assert  H * W == end
    flat_patch_pos = torch.arange(0 if not use_cls else -1, end) # N 表示 end。
    x_pos = flat_patch_pos % H # N
    y_pos = flat_patch_pos // H # N
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(end if not use_cls else end + 1, -1)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """把 RoPE 缓存重排成可与 query/key 广播相乘的形状。"""
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if freqs_cis.shape[-1] == x.shape[-1]:
        shape = [1 if i == 2 or i == 0 else d for i, d in enumerate(x.shape)]  # 1, N, 1, Hc/2
    else:
        shape = [d if i != 0 else 1 for i, d in enumerate(x.shape)] # 1, N, H, Hc/2
        # B, N, Hc/2
    return freqs_cis.view(*shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 query/key 应用旋转位置编码。

    最后一维会按两两成对解释为复数实部和虚部，旋转后再摊平成原张量。
    这正是 RoPE 的核心：`[x1, x2]` 被当作一个二维向量进行相位旋转。
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Pooling(nn.Module):
    """把 token 序列按 2x2 邻域降采样。"""
    def __init__(self, pool_type, dim):
        """根据类型选择平均池化、最大池化或线性池化。"""
        super().__init__()
        if pool_type == "a":
            self.pool = nn.AvgPool2d(kernel_size=2)

        elif pool_type == "m":
            self.pool = nn.MaxPool2d(kernel_size=2)

        elif pool_type == "l":
            self.pool = nn.Linear(4 * dim, dim)

        else:
            raise NotImplementedError

        self.pool_type = pool_type

    def forward(self, x):
        """把 `(B, N, C)` token 序列压缩为更短的序列。"""
        B, N, C= x.shape
        if self.pool_type in ["a", "m"]:
            H, W = int(math.sqrt(N)), int(math.sqrt(N))
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = self.pool(x)
            x = x.view(B, C, -1).transpose(1, 2).contiguous()

        else:
            x = x.view(B, N//4, -1)
            x = self.pool(x)

        return x


class Up(nn.Module):
    """把 token 序列按 2 倍空间尺度上采样。"""
    def __init__(self, up_type, dim):
        """根据类型选择最近邻或带线性投影的上采样。"""
        super().__init__()
        if up_type == "n":
            self.up = nn.Upsample(scale_factor=2, mode='nearest')

        elif up_type == "r":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                Rearrange('b c h w -> b (h w) c'),
                nn.Linear(dim, dim)
            )

        else:
            raise NotImplementedError

        self.up_type = up_type

    def forward(self, x):
        """把 `(B, N, C)` token 序列恢复到更高空间分辨率。"""
        B, N, C= x.shape
        if self.up_type == "n":
            H, W = int(math.sqrt(N)), int(math.sqrt(N))
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = self.up(x)
            x = x.view(B, C, -1).transpose(1, 2).contiguous()

        else:
            # 代码/形状说明：x = self.up(x) # B, N, 4c
            # 代码/形状说明：x = x.view(B, N * 4, -1)
            H, W = int(math.sqrt(N)), int(math.sqrt(N))
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # B, C, H, W
            x = self.up(x) # B, (2H 2W), C

        return x


class GEGLU(nn.Module):
    """GEGLU 门控前馈单元。

    若输入按最后一维均分为 `x` 与 `gate`，则输出为 `x * GELU(gate)`。
    相比普通 MLP，GEGLU 通过门控分支提升了非线性表达能力。
    """
    def forward(self, x):
        """执行 `x * GELU(gate)` 门控。"""
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.):
    """构造 Transformer 前馈网络。

    内部宽度使用 `int(mult * 2 / 3 * dim)`，与 GEGLU 配合后整体参数量
    接近常见 `4 * dim` 的 FFN，但门控形式更强。
    """
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )

def window_partition(x, window_size):
    """把特征图切分成不重叠窗口。

    输入为 `(B, H, W, C)`，输出为 `(num_windows * B, ws, ws, C)`。
    这一步常用于窗口注意力，避免在整幅图上做全局注意力。
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """把窗口张量拼回完整特征图。"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" 基于窗口的多头自注意力（W-MSA）模块，带相对位置偏置。
    同时支持 shifted window 和非 shifted window。

    参数：
        dim (int)：输入通道数。
        window_size (tuple[int])：窗口的高和宽。
        num_heads (int)：注意力头数量。
        qkv_bias (bool, optional)：为 query/key/value 添加可学习 bias。默认 True。
        qk_scale (float | None, optional)：如提供则覆盖默认缩放 `head_dim ** -0.5`。
        attn_drop (float, optional)：注意力权重的 dropout 比例。默认 0.0。
        proj_drop (float, optional)：输出投影的 dropout 比例。默认 0.0。
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """构造窗口注意力层，并初始化相对位置偏置表。"""
        super().__init__()
        self.dim = dim
        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        self.norm = LayerNorm(dim)
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 定义相对位置偏置参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # 计算窗口内每对 token 的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 平移到从 0 开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        参数：
            x：输入特征，形状为 `(num_windows*B, N, C)`。
            mask：`0/-inf` 掩码，形状为 `(num_windows, Wh*Ww, Wh*Ww)`，也可以为 None。
        """
        B_, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        x = self.norm(x)

        x = x.view(B_, H, W, -1)
        # 切分窗口
        x_windows = window_partition(x, self.window_size[0])  # 代码/形状说明：nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # 代码/形状说明：nW*B, window_size*window_size, C

        BW, NW = x_windows.shape[:2]

        qkv = self.qkv(x_windows).reshape(BW, NW, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 中文说明：make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x_windows = (attn @ v).transpose(1, 2).reshape(BW, NW, C)
        x_windows = self.proj(x_windows)
        x_windows = self.proj_drop(x_windows)

        x = window_reverse(x_windows, self.window_size[0], H, W)  # B H' W' C
        x = x.view(B_, H * W, C)

        return x




class PEG(nn.Module):
    """位置生成模块（PEG）。

    它使用 depth-wise 3D 卷积在局部邻域内注入位置感知能力。与显式位置编码不同，
    PEG 把位置信息作为卷积归纳偏置直接加回特征。
    """
    def __init__(self, dim, causal=False):
        """构造因果或非因果的 3D depth-wise 位置卷积。"""
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    @beartype
    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        """把位置卷积结果加到 token 或视频特征上。"""
        needs_shape = x.ndim == 3
        assert not (needs_shape and not exists(shape))

        orig_shape = x.shape
        if needs_shape:
            x = x.reshape(*shape, -1)

        x = rearrange(x, 'b ... d -> b d ...')

        frame_padding = (2, 0) if self.causal else (1, 1)

        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.)
        x = self.dsconv(x)

        x = rearrange(x, 'b d ... -> b ... d')

        if needs_shape:
            x = rearrange(x, 'b ... d -> b (...) d')

        return x.reshape(orig_shape)

class Attention(nn.Module):
    """多头注意力模块。

    既可做自注意力，也可在传入 `context` 时做交叉注意力；并支持 RoPE、
    ALiBi 风格位置偏置、query 下采样和 q/k 归一化。
    """
    def __init__(
        self,
        dim,
        dim_context=None,
        dim_head=64,
        heads=8,
        causal=False,
        norm_context=False,
        dropout=0.,
        spatial_pos="rel",
        mlp_block=False,
        qk_norm=None
    ):
        """构造视频/图像 Transformer 中使用的注意力层。"""
        super().__init__()
        self.heads = heads
        self.causal = causal
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.spatial_pos = spatial_pos
        self.freqs_cis = None

        self.p_dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(
            dim_context) if norm_context else nn.Identity()

        self.qk_norm = qk_norm
        if qk_norm == "l2norm":
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))
        elif qk_norm == "rmsnorm":
            self.q_norm = RMSNorm(dim_head)
            self.k_norm = RMSNorm(dim_head)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)
        self.dim = inner_dim

        # 中文标题：mlp branch
        self.mlp_block = mlp_block
        if mlp_block:
            self.mlp_in = nn.Linear(dim, inner_dim)
            self.mlp_gelu = nn.GELU()
            self.mlp_out = nn.Linear(dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        mask=None,
        context=None,
        is_spatial=True,
        q_stride=1,
        rope_cache=None,
        upcast_attention=None
    ):
        """执行注意力前向传播。

        主要步骤是：
        1. 线性映射得到 `q/k/v`；
        2. 视需要注入 RoPE；
        3. 调用 `scaled_dot_product_attention`；
        4. 再经过输出投影。
        """
        batch, device, dtype = x.shape[0], x.device, x.dtype

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)
        N = x.shape[1]

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b n h d', h=self.heads), (q, k, v))

        if self.spatial_pos == "rope" and is_spatial and rope_cache != None:
            q, k = apply_rotary_emb(q, k, freqs_cis=rope_cache)

        q, k, v = map(lambda t: rearrange(
            t, 'b n h d -> b h n d', h=self.heads), (q, k, v))

        B, H, _, D = q.shape
        if q_stride > 1:
            # 在 query 序列上做分组 max-pool，以降低注意力计算量。
            q = (
                q.view(B, H, q_stride, -1, D)
                .max(dim=2)
                .values
            )

        if self.qk_norm == "l2norm":
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale
        elif self.qk_norm == "rmsnorm":
            q = self.q_norm(q)
            k = self.k_norm(k)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if q.shape[-2] == 1 and k.shape[-2] == 1 and v.shape[-2] == 1:
            dummy_op = torch.sum(q) * 0 + torch.sum(k) * 0 # 中文说明：加入空操作，确保 q 和 k 仍参与计算图。
            out = v + dummy_op
        else:
            q = q.to(torch.float32) if "q" in upcast_attention else q
            k = k.to(torch.float32) if "k" in upcast_attention else k
            v = v.to(torch.float32) if "v" in upcast_attention else v
            if is_dtype_16(q) or is_dtype_16(k) or is_dtype_16(v):
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p_dropout, is_causal=self.causal)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p_dropout, is_causal=self.causal)

        out = rearrange(out, 'b h n d -> b n (h d)')

        # 中文标题：mlp_block branch
        if self.mlp_block:
            mlp_x = self.mlp_in(x)
            mlp_x = self.mlp_gelu(mlp_x)
            mlp_out = self.mlp_out(mlp_x)
            out = out + mlp_out

        return self.to_out(out)


class AlibiPositionalBias(nn.Module):
    """ALiBi 位置偏置。

    ALiBi 通过与相对距离线性相关的 bias 代替显式位置嵌入，优点是长度外推更稳定。
    """
    def __init__(self, heads):
        """为每个注意力头预计算不同斜率。"""
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    def get_bias(self, i, j, device):
        """构造长度为 `i x j` 的相对距离偏置矩阵。"""
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') -
                          rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        """生成 ALiBi 论文使用的 head-wise 斜率序列。"""
        def get_slopes_power_of_2(n):
            """在 head 数为 2 的幂时生成基础斜率。"""
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, sim):
        """返回与当前注意力矩阵大小匹配的 ALiBi 偏置。"""
        h, i, j, device = *sim.shape[-3:], sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent=False)

        return self.bias


class ContinuousPositionBias(nn.Module):
    """连续位置偏置网络。

    该模块把相对坐标送入一个小型 MLP，输出每个注意力头的 bias，
    思路来自 `https://arxiv.org/abs/2111.09883`。
    """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims=2,  # 中文说明：2 for images, 3 for video
        layers=2,
        log_dist=True,
        cache_rel_pos=False
    ):
        """构造从相对坐标到位置偏置的 MLP。"""
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(
            nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent=False)

    def forward(self, *dimensions, device=torch.device('cpu')):
        """根据目标空间尺寸生成连续位置偏置。"""

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing='ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - \
                rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent=False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, 'i j h -> h i j')

class Transformer(nn.Module):
    """视频/图像用 Transformer 堆叠。

    每层可选 PEG、自注意力、交叉注意力、前馈网络和 DropPath，
    并支持用 `q_strides` 在注意力阶段执行 token 下采样。
    """
    def __init__(
        self,
        dim,
        *,
        depth,
        block,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        ff_mult=4,
        peg=False,
        peg_causal=False,
        has_cross_attn=False,
        attn_dropout=0.,
        ff_dropout=0.,
        window_size=4,
        spatial_pos="rel",
        mlp_block=False,
        upcast_attention=None,
        qk_norm=None,
        drop_path=0.
    ):
        """按 `block` 配置构造多层 Transformer。"""
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.upcast_attention = upcast_attention
        assert len(block) == depth
        self.layers = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        for i in range(depth):
            if block[i] == 't':
                self.layers.append(nn.ModuleList([
                    PEG(dim=dim, causal=peg_causal) if peg else None,
                    Attention(dim=dim, dim_head=dim_head, heads=heads,
                            causal=causal, dropout=attn_dropout, spatial_pos=spatial_pos, mlp_block=mlp_block, qk_norm=qk_norm),
                    Attention(dim=dim, dim_head=dim_head, dim_context=dim_context, heads=heads, causal=False,
                            dropout=attn_dropout, mlp_block=mlp_block, qk_norm=qk_norm) if has_cross_attn else None,
                    FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
                ]))

            # 代码/形状说明：elif block[i] == 'w':
            # 代码/形状说明：self.layers.append(nn.ModuleList([
            # 中文说明：None,
            # 中文说明：WindowAttention(dim=dim, window_size=window_size, num_heads=heads, attn_drop=attn_dropout),
            # 中文说明：None,
            # 中文说明：FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            #     ]))

            # 中文说明：不同 pooling 方法的输入约定为 B, N, C。
            # 代码/形状说明：elif block[i] in ['a', 'm', 'l']:
            # 代码/形状说明：self.layers.append(nn.ModuleList([
            # 中文说明：None,
            # 中文说明：Pooling(block[i], dim),
            # 中文说明：None,
            # 中文说明：FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            #     ]))

            # 代码/形状说明：elif block[i] in ['n', 'r']:
            # 代码/形状说明：self.layers.append(nn.ModuleList([
            # 中文说明：None,
            # 中文说明：Up(block[i], dim),
            # 中文说明：None,
            # 中文说明：FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            #     ]))

            else:
                raise NotImplementedError

        self.block = block
        self.norm_out = nn.LayerNorm(dim)

    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        context=None,
        self_attn_mask=None,
        cross_attn_context_mask=None,
        q_strides=None,
        is_spatial=True
    ):
        """逐层执行 PEG、注意力、前馈与残差连接。"""
        if q_strides is None:
            q_strides = '1' * len(self.layers)

        for blk, q_stride, (peg, self_attn, cross_attn, ff, drop_path) in zip(self.block, q_strides, self.layers):
            if exists(peg):
                with torch.amp.autocast("cuda", enabled=False):
                    x = peg(x, shape=video_shape) + x

            if isinstance(self_attn, Attention):
                H, W = video_shape[2], video_shape[3]
                if x.shape[-2] == H * W:
                    rope_cache = precompute_freqs_cis_2d(self.dim_head, x.shape[1], H, W).to(x.device)
                elif x.shape[-2] == 1 or is_spatial == False:
                    rope_cache = None
                else:
                    raise NotImplementedError
                x = drop_path(self_attn(
                    x, mask=self_attn_mask,
                    q_stride=int(q_stride), is_spatial=is_spatial,
                    rope_cache=rope_cache, upcast_attention=self.upcast_attention
                )) + do_pool(x, int(q_stride))

            elif isinstance(self_attn, WindowAttention):
                x = drop_path(self_attn(x)) + x
            else:
                x = self_attn(x)

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context,
                               mask=cross_attn_context_mask) + x

            x = ff(x) + x

            # 处理下采样/上采样后的 video_shape 变化。
            if blk in ['a', 'm', 'l']:
                video_shape = (video_shape[0], video_shape[1], video_shape[2]//2, video_shape[3]//2) # 中文说明：video_shape: B, T, H, W

            elif blk in ['n', 'r']:
                video_shape = (video_shape[0], video_shape[1], int(video_shape[2]*2), int(video_shape[3]*2))


            if q_stride != '1':
                down_ratio = int(math.sqrt(int(q_stride)))
                video_shape = (video_shape[0], video_shape[1], video_shape[2]//down_ratio, video_shape[3]//down_ratio)

        return self.norm_out(x)
