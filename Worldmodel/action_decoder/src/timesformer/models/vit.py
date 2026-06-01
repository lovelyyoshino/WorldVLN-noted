# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# 修改过的模型定义。

"""TimesFormer Vision Transformer 主体。

中文导读：
    本文件实现 WorldVLN 动作解码器使用的 TimeSformer 主干。它在标准 ViT 的基础上把
    每个 Transformer block 拆成 “divided space-time attention”：先对每个空间位置内
    沿时间 T 做 self-attention，再对每个时间步内沿 N 个空间 token 做 self-attention，
    最后通过 MLP 残差更新 cls + patch token。

    Divided space-time attention 公式：
        公式：先做时间 self-attention（每个 spatial 位置内沿 T），再做空间 self-attention
        （每个时间步内沿 N）。
        公式：x_t = x + DropPath( temporal_attn( norm(x) ) )         # 时间分支
        公式：x_s = x_t + DropPath( spatial_attn( norm(x_t) ) )      # 空间分支
        公式：x_o = x_s + DropPath( mlp( norm(x_s) ) )               # MLP 残差

    DropPath 公式：
        公式：训练时 mask ~ Bernoulli(keep_prob)，
              x_out = x * mask / keep_prob；评估时直接返回 x。

    本文件提供三种入口：
        - ``forward(x)``：完整前向，输入原始视频 ``(B,C,T,H,W)``，输出 logits。
        - ``forward_features(x)``：完整前向，但只返回 cls embedding，未过 head。
        - ``forward_features_from_patch_tokens(tokens, B, T, W)``：跳过 PatchEmbed，
          直接接收外部 patch token（典型用法见 ``Vae96ToTSformerEmbedAdapter``）。
"""

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

from timesformer.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timesformer.models.helpers import load_pretrained
from timesformer.models.vit_utils import DropPath, to_2tuple, trunc_normal_

from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat

def _cfg(url='', **kwargs):
    """
    生成 Vision Transformer 默认配置字典，并允许调用方覆盖字段。
    """
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class Mlp(nn.Module):
    """
    Transformer block 中的两层前馈网络。

    结构为 ``Linear(in -> hidden) -> GELU -> Dropout -> Linear(hidden -> out) -> Dropout``，
    残差连接在外层 ``Block`` 中实现。
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        初始化 MLP 的两个 Linear 层、激活函数和 dropout。

        参数：
            in_features (int): 输入维度 D。
            hidden_features (int): 中间维度，默认与 ``in_features`` 一致；
                TimesFormer 默认 ``4 * D``。
            out_features (int): 输出维度，默认与 ``in_features`` 一致。
            act_layer (Callable): 激活函数，默认 ``nn.GELU``。
            drop (float): dropout 概率。
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        执行 Linear、激活、dropout、Linear、dropout 的前向计算。

        参数：
            x (Tensor): 形状 ``(B, N, D)`` 的 token 序列。
        返回：
            Tensor: 与输入同形状的 token。
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """
    多头自注意力模块，可选择是否在内部生成 q、k、v。

    公式：q, k, v = split( Linear_qkv(x) )，每个形状为 ``(B, heads, N, D/heads)``。
    公式：attn = softmax( q @ k^T * scale )，再做 attn dropout。
    公式：out = (attn @ v).reshape(B, N, D)，再过 ``proj``、``proj_drop``。

    其中 ``scale = (D/heads)^(-0.5)``，对应论文中 ``1/sqrt(d_k)``。
    ``with_qkv=False`` 时假设外部已生成 q=k=v=x（divided space-time 不会用到这条路径）。
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        """
        初始化多头注意力的 qkv 投影、输出投影和 dropout。

        参数：
            dim (int): token 维度 D。
            num_heads (int): 多头数。
            qkv_bias (bool): qkv Linear 是否带 bias。
            qk_scale (float|None): q-k 点积缩放因子；为 None 时使用 ``head_dim**-0.5``。
            attn_drop (float): attention 矩阵 dropout 概率。
            proj_drop (float): 输出投影 dropout 概率。
            with_qkv (bool): 是否在模块内部生成 q/k/v；False 时假设输入已为 q=k=v。
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        """
        对输入 token 序列执行多头自注意力。

        参数：
            x (Tensor): 形状 ``(B, N, D)``。divided space-time 中 ``B`` 会被外层
                重排成 ``(B*H*W)`` 或 ``(B*T)`` 来分别承担时间/空间注意力。
        返回：
            Tensor: 与输入同形状 ``(B, N, D)``。
        """
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        # 公式：attn = softmax( q @ k^T * scale )，scale = head_dim^(-0.5)。
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 公式：out = (attn @ v).reshape(B, N, D)。
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """
    TimeSformer 的 Transformer block，支持空间、时间或联合注意力。

    Divided space-time attention 公式：
        公式：先做时间 self-attention（每个 spatial 位置内沿 T），再做空间 self-attention
        （每个时间步内沿 N）。
        公式：x_t = x + DropPath( temporal_fc( temporal_attn( norm(x) ) ) )
        公式：x_s = x_t + DropPath( spatial_attn( norm(x_t with cls) ) )
        公式：x_o = x_s + DropPath( mlp( norm(x_s) ) )

    ``space_only`` / ``joint_space_time`` 退化为标准 ViT block，cls token 与 patch
    token 一起做完整 self-attention。
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        """
        初始化注意力、MLP、归一化和可选的分离时空注意力分支。

        参数：
            dim (int): token 维度 D。
            num_heads (int): 多头数。
            mlp_ratio (float): MLP hidden_dim 相对 dim 的倍率。
            qkv_bias / qk_scale / drop / attn_drop: 透传给 :class:`Attention`。
            drop_path (float): 残差 DropPath 概率。
            act_layer / norm_layer: 激活与归一化层类型。
            attention_type (str): ``'space_only'`` / ``'joint_space_time'`` /
                ``'divided_space_time'``。WorldVLN 标准为最后一种。
        """
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## 时间注意力参数。
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        # 中文说明：# drop path。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        """
        根据 ``attention_type`` 对视频 token 执行一个 Transformer block。

        参数：
            x (Tensor): 形状 ``(B, 1+N*T, D)`` 的 token 序列，第 0 列是全局 cls token，
                后续 ``N*T`` 列按 ``(h, w, t)`` 的 raster 顺序排列空间-时间 token。
            B (int): batch size（视频数量）。
            T (int): 时间帧数。
            W (int): 空间 patch 网格宽度（H_patch 由 ``num_spatial_tokens // W`` 推得）。
        返回：
            Tensor: 与输入同形状的更新后 token。
        """
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## 时间注意力。
            # 公式：把 (b, h*w*t, m) 重排成 (b*h*w, t, m)，每个 spatial 位置内沿 t 做 self-attention。
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            # 公式：x_t = x_patch + DropPath( temporal_fc( temporal_attn( norm(x_patch) ) ) )。
            xt = x[:,1:,:] + res_temporal

            ## 空间注意力。
            # 把 cls token 复制 T 份，每一帧拼一个 cls，便于在每个时间步内沿 N 做 self-attention。
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            # 公式：把 (b, h*w*t, m) 重排成 (b*t, h*w, m)，每个时间步内沿 N=h*w 做 self-attention。
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### 处理 CLS token。
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## 对每一帧的 CLS token 取平均。
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            # 中文说明：# MLP。
            # 公式：x = [cls; x_t] + [cls_avg; spatial_attn(...)]，再 + DropPath( mlp( norm(x) ) )。
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class PatchEmbed(nn.Module):
    """
    将视频帧图像切成 patch，并投影成 token embedding。

    形状公式：
        公式：(B, C, T, H, W) -> (B*T, C, H, W)                  # 时间维并入 batch
        公式：Conv2d(kernel=patch_size, stride=patch_size) ->     # 切 patch + 线性投影
              (B*T, embed_dim, H/P, W/P)
        公式：flatten(2).transpose(1, 2) -> (B*T, N, embed_dim)   # N = (H/P)*(W/P)
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        """
        初始化 patch 大小、patch 数量和 Conv2d 投影层。

        参数：
            img_size (Tuple[int,int]): ``(H, W)``，TimesFormer-WorldVLN 用 (192,640)。
            patch_size (int): patch 边长，stride 与之相同。
            in_chans (int): 输入通道数；RGB 帧为 3。
            embed_dim (int): 输出 token 维度 D。
        """
        super().__init__()
        # 如果需要支持标量 img_size，可以用 to_2tuple 转成二维大小。
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        将 ``(B, C, T, H, W)`` 视频输入转成每帧 patch token。

        参数：
            x (Tensor): ``(B, C, T, H, W)`` 视频。
        返回：
            Tuple[Tensor, int, int]: ``(tokens, T, W_patch)``，``tokens`` 形状
            ``(B*T, N, embed_dim)``，``W_patch = W / patch_size``。
        """
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class VisionTransformer(nn.Module):
    """
    TimeSformer 使用的 Vision Transformer 主体网络。

    主要组件：
        ``patch_embed``  : :class:`PatchEmbed`，把视频帧切成 ``(B*T, N, D)`` token。
        ``cls_token``    : ``(1, 1, D)`` 全局分类 token。
        ``pos_embed``    : ``(1, 1+num_patches, D)``，按训练分辨率初始化；推理时若网格
                           不匹配会做最近邻 resize。
        ``time_embed``   : ``(1, num_frames, D)``，仅在 ``divided_space_time`` /
                           ``joint_space_time`` 下使用，按帧附加。
        ``blocks``       : ``depth`` 层 :class:`Block`，可配置 divided / joint / space-only。
        ``norm``         : 输出前的 LayerNorm。
        ``head``         : Linear 分类头，把 cls embedding 投影到 ``num_classes`` 维。

    本类提供三个前向入口：
        - :meth:`forward` ：完整视频 -> logits，训练时主入口。
        - :meth:`forward_features` ：完整视频 -> cls embedding（未过 head）。
        - :meth:`forward_features_from_patch_tokens` ：跳过 PatchEmbed，从外部
          patch token 出发跑后续 pipeline，供 ``Vae96ToTSformerEmbedAdapter`` 使用。

    Divided space-time attention 公式：
        公式：先做时间 self-attention（每个 spatial 位置内沿 T），再做空间 self-attention
        （每个时间步内沿 N）。详见 :class:`Block` docstring。
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.):
        """
        初始化 patch embedding、位置/时间 embedding、Transformer blocks 和分类头。

        参数：
            img_size (Tuple[int,int]): ``(H, W)`` 输入分辨率，WorldVLN 用 (192,640)。
            patch_size (int): patch 边长，标准 16。
            in_chans (int): 输入通道；RGB 为 3。
            num_classes (int): 分类/回归输出维度。WorldVLN 用 18 = 6D delta * 3 帧。
            embed_dim (int): token 维度 D，标准 384（也支持 192/768/1024）。
            depth (int): Transformer block 层数。
            num_heads (int): 多头注意力头数。
            mlp_ratio (float): MLP hidden 相对 D 的倍率，标准 4。
            qkv_bias / qk_scale / drop_rate / attn_drop_rate / drop_path_rate:
                透传给 :class:`Attention` 与 :class:`Block`，``drop_path_rate``
                会按 stochastic depth 等差衰减分配到各层。
            num_frames (int): 训练 ``time_embed`` 的帧数；推理时帧数不一致会自动 resize。
            attention_type (str): 见 :class:`Block`。
            dropout (float): 顶层 dropout（很少使用）。
        """
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # 中文说明：为了和其他模型保持一致而保留 num_features。
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## 位置 embedding。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## 注意力 blocks。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth 衰减规则。
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # 分类头。
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # 时间注意力权重初始化。
        # 保留的上游调试/兼容代码：if self.attention_type == 'divided_space_time':
        #     i = 0
        # 保留的上游调试/兼容代码：for m in self.blocks.modules():
        # 中文说明：m_str = str(m)
        # 保留的上游调试/兼容代码：if 'Block' in m_str:
        #             if i > 0:
        # 中文说明：nn.init.constant_(m.temporal_fc.weight, 0)
        # 中文说明：nn.init.constant_(m.temporal_fc.bias, 0)
        #             i += 1

    def _init_weights(self, m):
        """
        初始化 Linear 和 LayerNorm 层的权重。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        返回不参与 weight decay 的参数名集合。
        """
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        """
        返回当前分类头模块。
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        """
        重置分类类别数，并按新类别数重建分类头。
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """
        从原始视频张量提取 CLS token 特征，尚不经过分类头。

        与 :meth:`forward_features_from_patch_tokens` 的区别：本方法接收 RGB 视频
        ``(B, C, T, H, W)``，内部用 :class:`PatchEmbed` 把图像切成 patch；
        ``forward_features_from_patch_tokens`` 则跳过 PatchEmbed，从外部提供的
        patch token 出发，专为 Adapter 链路设计。

        参数：
            x (Tensor): ``(B, C, T, H, W)`` 视频。
        返回：
            Tensor: ``(B, D)`` 的 cls embedding。
        """
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## 推理时如果输入尺寸与训练位置 embedding 不匹配，则调整位置 embedding 大小。
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## 时间 embedding。
        if self.attention_type != 'space_only':
            # 注意：
            # x 当前形状为 (B*T, 1+N, D)（经过 PatchEmbed + cls + pos 后）。
            # 每个视频（batch item）只需要一个 cls token，而不是 (B*T) 的前 B 个元素。
            # 每个视频第一帧的正确索引是：0, T, 2T, ...
            cls_tokens = x[0::T, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## 如果帧数 T 与 time embedding 长度不匹配，则调整 time embedding 大小。
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## 注意力 blocks。
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### space-only baseline 的预测处理。
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # 对每一帧的预测取平均。

        x = self.norm(x)
        return x[:, 0]

    def forward_features_from_patch_tokens(self, patch_tokens, B: int, T: int, W: int):
        """
                从预先计算好的 patch tokens 提取 TimeSformer 特征，跳过 PatchEmbed。

                与 :meth:`forward_features` 的区别：本方法不再调用 ``PatchEmbed``，调用方
                需要自行把视频/隐空间特征压成 patch token。这是 WorldVLN
                ``Vae96ToTSformerEmbedAdapter`` 的入口：先用 Adapter 把
                ``(B, 96, T, 256, 256)`` -> ``(B*T, 480, 384)`` 的 token，再调用本方法
                跑完整的 divided space-time 注意力。

                参数：
                    patch_tokens: 形状为 (B*T, N, D) 的 Tensor，其中：
                      - N = (H/P)*(W/P)，表示每帧的空间 patch 数。
                      说明：- D = embed_dim。
                    B: batch size，也就是视频数量。
                    T: 每个视频的帧数。
                    W: patch 网格宽度 (W/P)，供 block 和 pos_embed resize 使用。

                返回：
                    形状为 (B, D) 的 Tensor，表示 transformer blocks 后的 cls embedding。

        """
        x = patch_tokens
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B*T,1,D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B*T,1+N,D)

        # 位置 embedding，resize 逻辑与 forward_features 相同。
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)  # (1,1,D)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)  # (1,D,P*P)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode="nearest")
            new_pos_embed = new_pos_embed.flatten(2).transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # 时间 embedding，逻辑相同，但会正确选择 cls token。
        if self.attention_type != "space_only":
            cls_tokens = x[0::T, 0, :].unsqueeze(1)  # (B,1,D)
            x = x[:, 1:]  # (B*T, N, D)
            x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T)
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode="nearest").transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, "(b n) t m -> b (n t) m", b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N*T, D)

        for blk in self.blocks:
            x = blk(x, B, T, W)

        if self.attention_type == "space_only":
            x = rearrange(x, "(b t) n m -> b t n m", b=B, t=T)
            x = torch.mean(x, 1)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        """
        提取视频特征并送入分类头，得到类别 logits。

        相当于 ``head(forward_features(x))``，是训练/评估时的主入口。
        如果训练流水线先经过 ``Vae96ToTSformerEmbedAdapter``，请改用
        :meth:`forward_features_from_patch_tokens` + 自定义 head。

        参数：
            x (Tensor): ``(B, C, T, H, W)`` 视频。
        返回：
            Tensor: ``(B, num_classes)``；WorldVLN 中 ``num_classes=18`` 表示
            3 帧 6D delta 动作的拼接。
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _conv_filter(state_dict, patch_size=16):
    """将手工 patchify + linear proj 的 patch embedding 权重转换成卷积权重。"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@MODEL_REGISTRY.register()
class vit_base_patch16_224(nn.Module):
    """
    注册到项目模型表中的 ViT-Base/16 TimeSformer 包装类。
    """

    def __init__(self, cfg, **kwargs):
        """
        根据 cfg 创建 ViT-Base/16 模型，并按配置加载预训练权重。
        """
        super(vit_base_patch16_224, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        """
        将输入视频转发给内部 VisionTransformer。
        """
        x = self.model(x)
        return x

@MODEL_REGISTRY.register()
class TimeSformer(nn.Module):
    """
    直接构建 TimeSformer 的便捷包装类。
    """

    def __init__(self, img_size=(224, 224), patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        """
        创建指定 patch 大小、帧数和注意力类型的 TimeSformer。
        """
        super(TimeSformer, self).__init__()
        self.pretrained=True
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
    def forward(self, x):
        """
        将输入视频转发给内部 VisionTransformer。
        """
        x = self.model(x)
        return x
