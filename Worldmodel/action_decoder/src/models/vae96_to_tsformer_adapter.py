"""VAE decoder 96 通道特征 -> TimesFormer patch token 的适配层。

中文导读：
    本模块定义 ``Vae96ToTSformerEmbedAdapter``，它是 WorldVLN 动作解码器中最关键
    的“桥”。世界模型先把视频压缩成 latent 再用 InfinityStar VAE decoder 还原；
    我们不取最终 RGB，而是 hook ``up_block_3`` 拿到 96 通道时空特征
    ``(B, 96, T, 256, 256)``。Adapter 负责把它压成 TimesFormer ``PatchEmbed``
    输出兼容的 patch token，让动作头可以从“预测到的隐空间世界”里直接回归 6D 动作。

    关键形状（见 ``forward`` 中详细注释）：
        公式：(B, 96, T, 256, 256) -> (B*T, 96, 256, 256)              # 时间维并入 batch
        公式：(B*T, 96, 256, 256)  -> (B*T, 96, 192, 640)              # bilinear resize 到 TimesFormer 训练分辨率
        公式：(B*T, 96, 192, 640)  -> (B*T, 384, 12, 40)               # conv_a + 16x16 patch 卷积
        公式：(B*T, 384, 12, 40)   -> (B*T, 480, 384)                  # flatten patch 网格
        其中 N = H_patch * W_patch = 12 * 40 = 480，D = embed_dim = 384。

------------------------------------------------------------------
为什么是“VAE -> TimesFormer”而不是“VAE 被 TimesFormer 替换”？
------------------------------------------------------------------
新读者最常踩的坑是把这两件事看成同一件，其实它们做的事完全不同：

    | 模块             | 做什么                                       |
    | ---------------- | -------------------------------------------- |
    | InfinityStar     | 在脑子里“想象”未来世界的剧本（预测 latent）  |
    | VAE decoder      | 把脑子里想的世界还原成 RGB / 中间特征        |
    | TimesFormer      | 看“世界在怎么动”，从时序特征回归 6D 动作     |

也就是说：**VAE 看得懂世界长什么样，TimesFormer 看得懂世界在怎么动**。
动作头需要的不是“好看的图”，而是“运动信号”，所以路线是分工，不是替换。

为什么不直接 latent -> VAE 全部解码 -> RGB -> TimesFormer.patch_embed？
答案是“浪费”：

    路径 A（朴素）：latent -> VAE 完整 decoder -> RGB -> TSformer.patch_embed -> token
                                                ^ 又慢又占显存       ^ 又把像素压回 patch
    路径 B（本项目）：latent -> VAE decoder 中途特征 -> Adapter -> token
                                            ^ 在 up_block_3 处直接拦截

  - 省一半算力：VAE decoder 后半段主要是补 RGB 细节（光照、纹理），
    对动作预测几乎没贡献；走完整解码再压回 patch 是纯浪费。
  - 特征更干净：中间层 96 通道还保留时空结构和语义，没有像素噪声。
  - 复用预训练：通过 Stage A 蒸馏，让 Adapter 输出的 token 在统计意义上对齐
    “真实 RGB 喂给 TimesFormer.patch_embed 得到的 teacher token”，原本在
    真实视频上学会动作回归的 TSformer-VO 知识可以直接迁移过来。

那为什么中间还要塞一个 Adapter？因为形状和分布都对不上：

    VAE 中间特征 ``(B, 96, T, 24, 20)``  -> 96 通道、自己的统计分布
    TSformer 期望 ``(B*T, 480, 384)``    -> 480 个 patch token，每个 384 维

Adapter 同时干两件事：
  1. **形状变换**：把空间网格 ``H_patch * W_patch = 480`` 个位置展平，
     96 通道经 ``conv_a + patch`` 投影到 384 维（公式见下方各步）。
  2. **分布对齐**：Stage A 蒸馏让 student token 与 teacher token 在
     ``cosine + mse + mean + std`` 四项 loss 下尽量重合，这样
     TimesFormer 接到的 token 不论来自“真实 RGB 还是预测 latent”，
     看上去都像同一种东西，原本的运动先验可以直接复用。

一句话总结：
    **VAE 没有被替换，被替换的是 VAE 的“RGB 重建后半段 + TSformer 的 patch_embed”。**
    Adapter 在 VAE decoder 完整解码完成之前“截胡”96 通道特征，
    一次性省掉「补细节 + 压回 patch」两步，又借蒸馏复用了 TSformer-VO 的动作能力。

代码佐证：
  - 截胡（VAE forward hook）：``infer/server.py::_stage2_predict_16_actions_for_segment_cm_deg``
    与 ``train/action_decoder/tools/train_stageB_ddp.py::_decode_tokens_full_T``
  - 形状变换（本文件）：``Vae96ToTSformerEmbedAdapter.forward``
  - 分布对齐（蒸馏 loss）：``train/action_decoder/tools/train_stageA_ddp.py::compute_distill_loss``
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_to_192x640(x: torch.Tensor) -> torch.Tensor:
    """把 VAE decoder 特征调整到 TimesFormer 训练使用的 192x640 输入网格。

    参数：
        x (torch.Tensor): 形状 ``(B*T, 96, H, W)`` 的 2D 特征。
    返回：
        torch.Tensor: 形状 ``(B*T, 96, 192, 640)`` 的 bilinear 上/下采样结果。
    """
    return F.interpolate(x, size=(192, 640), mode="bilinear", align_corners=False)


class Vae96ToTSformerEmbedAdapter(nn.Module):
    """
        把 InfinityStar decoder 的 ``up_block_3`` 特征 ``(B,96,T,256,256)`` 映射成
        TSformer patch tokens。

        输出 token 与 TSformer ``PatchEmbed`` 的输出保持一致：
          - 输出 ``patch_tokens``：形状 ``(B*T, N=12*40=480, D=384)``
          - 返回 ``(patch_tokens, T, W_grid=40)``，可直接喂给
            :py:meth:`timesformer.models.vit.VisionTransformer.forward_features_from_patch_tokens`。

        形状公式（patch + proj 两步）：
          公式：第 1 步 patch 化：``(B,96,T,H,W) -> (B*T,96,H,W) -> resize -> (B*T,96,192,640)``。
          公式：第 2 步 proj 投影：``(B*T,96,192,640) --conv_a--> (B*T,128,192,640)
                --patch(16x16,stride=16)--> (B*T, D=384, H_patch=12, W_patch=40)``。
          公式：H_patch = 192 / patch_size，W_patch = 640 / patch_size，N = H_patch * W_patch；
          每个 token 对应 resize 后图像网格上的一个 patch。
          公式：flatten + transpose：``(B*T, D, 12, 40) -> (B*T, 480, 384)``。

        中文导读：
        这是动作解码器中最关键的“翻译层”。世界模型给出的 latent 先经过 VAE decoder，
        但代码不会使用最终 RGB 输出，而是在 decoder 中间层取 96 通道时空特征。Adapter
        把这些 VAE 特征重排成 TimesFormer 能直接消费的 patch tokens，使后续动作头可以
        从预测到的隐空间世界转移中恢复 6D 动作。

        参数：
            embed_dim (int): TimesFormer 主干维度 D，WorldVLN 标准为 384。
            patch_size (int): 空间 patch 边长，标准为 16。
            use_skip (bool): 是否额外加一条 1x1 + avg_pool 的“低级特征跳连”，默认 False。
                打开后会用 ``h + 0.1 * skip(x)`` 的方式注入未经 conv_a 的原始 96 通道信息。
    """

    def __init__(self, embed_dim: int = 384, patch_size: int = 16, use_skip: bool = False):
        """构造 Adapter：96 通道 VAE 特征 -> 384 维、16x16 patch token。

        模块组成：
            ``conv_a``  : 96 -> 128 的 3x3 conv + GroupNorm(32) + SiLU，做特征精炼。
            ``patch``   : 128 -> embed_dim 的 ``patch_size`` x ``patch_size`` 卷积，
                          stride=patch_size，相当于 TimesFormer 内部 ``PatchEmbed.proj``。
            ``skip``    : 可选的 1x1 残差通道，把原始 96 通道直接投影到 embed_dim。
            ``out_norm``: LayerNorm，让 token 分布与 TimesFormer pos_embed 兼容。
        """
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.use_skip = bool(use_skip)

        self.conv_a = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
        )

        self.patch = nn.Conv2d(128, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.skip = nn.Conv2d(96, self.embed_dim, kernel_size=1, padding=0, bias=False) if self.use_skip else None
        self.out_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, f96_up3: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        将一段 VAE decoder 特征片段转成 TimesFormer patch tokens。

        中文导读：
        输入的时间维 ``T`` 保留下来，但 batch 和时间会临时合并成 ``B*T`` 做 2D patch 化；
        输出 token 再交给 TimesFormer 的 ``forward_features_from_patch_tokens()``。

        token 网格含义：192x640 调整尺寸后用 16x16 patch 时，网格是 12x40，
        所以 ``N = 12 * 40 = 480`` 个空间 token。

        参数：
            f96_up3 (torch.Tensor): 形状 ``(B, 96, T, H, W)`` 的 VAE decoder 中间特征。
                通常 ``H=W=256``，但本函数只要求 ``ndim=5`` 且 ``C=96``，会自动 resize。
        返回：
            Tuple[torch.Tensor, int, int]: ``(tokens, T, grid_w)``。
                - tokens: ``(B*T, 480, 384)`` 的 patch token，已经 LayerNorm。
                - T: 原始时间帧数（保持不变）。
                - grid_w: patch 网格宽度（标准为 40），供 TimesFormer 的 pos_embed
                  resize 与 divided space-time block 推断 H_patch 使用。
        """
        if f96_up3.ndim != 5:
            raise ValueError(f"期望 f96_up3 形状为 (B,96,T,H,W)，实际收到 {tuple(f96_up3.shape)}")
        B, C, T, H, W = f96_up3.shape
        if int(C) != 96:
            raise ValueError(f"期望通道数 channel=96，实际 C={C}")

        # 公式：先把时间维并入 batch：每一帧单独做 2D patch，
        # (B,96,T,H,W) --permute(0,2,1,3,4)--> (B,T,96,H,W) --reshape--> (B*T,96,H,W)。
        x = f96_up3.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (BT,96,H,W)
        x = _resize_to_192x640(x)  # (BT,96,192,640)

        # 第 1 步（patch 前的特征精炼）：96 -> 128，保持空间分辨率不变。
        h = self.conv_a(x)  # (BT,128,192,640)
        # 第 2 步（patch 卷积）：128 -> 384，stride=patch_size 后空间被压成 (12,40)。
        # 公式：H_patch = 192 / patch_size = 12，W_patch = 640 / patch_size = 40，
        # N = H_patch * W_patch = 480。
        h = self.patch(h)  # (BT,384,12,40)

        if self.skip is not None:
            # 可选跳连：把原始 96 通道直接 1x1 投影到 embed_dim，再用 avg_pool 对齐
            # patch 网格，最后以 0.1 的小权重加回主路径。
            s = self.skip(x)  # (BT,384,192,640)
            s = F.avg_pool2d(s, kernel_size=self.patch_size, stride=self.patch_size)  # (BT,384,12,40)
            xg = h + 0.1 * s
        else:
            xg = h

        # 公式：flatten(2) 把 (12,40) 网格拉平成 N=480，再 transpose 得到 (B*T, N, D)。
        tokens = xg.flatten(2).transpose(1, 2).contiguous()  # (BT,480,384)
        tokens = self.out_norm(tokens)
        grid_w = int(xg.shape[-1])
        return tokens, int(T), grid_w
