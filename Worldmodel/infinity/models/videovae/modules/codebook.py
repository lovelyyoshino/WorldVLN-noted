# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

from enum import unique
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from infinity.models.videovae.utils.misc import shift_dim

class Codebook(nn.Module):
    """单尺度 EMA 码本。"""
    def __init__(self, n_codes, embedding_dim, no_random_restart=False, restart_thres=1.0, usage_sigma=0.99, fp32_quant=False):
        """构造 EMA 更新的离散码本。"""
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())
        self.register_buffer('codebook_usage', torch.zeros(n_codes))

        self.call_cnt = 0
        self.usage_sigma = usage_sigma

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

        self.fp32_quant = fp32_quant

    def _tile(self, x):
        """在样本数不足时重复并加少量噪声，满足码本初始化需要。"""
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        """用当前 batch 的潜变量初始化码本向量。"""
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))


    def calculate_batch_codebook_usage_percentage(self, batch_encoding_indices):
        """统计一个 batch 内每个码字的使用占比。"""
        # 先把整批离散索引压成一维，便于做全局计数。
        all_indices = batch_encoding_indices.flatten()

        # 统计总 token 数，后续把计数换成百分比。
        total_indices = all_indices.numel()

        # 预先分配每个码字的使用率槽位。
        codebook_usage_percentage = torch.zeros(self.n_codes, device=all_indices.device)

        # 统计每个离散索引出现了多少次。
        unique_indices, counts = torch.unique(all_indices, return_counts=True)
        # 把次数换成占比。
        percentages = (counts.float() / total_indices)

        # 把非零占比写回对应码字位置。
        codebook_usage_percentage[unique_indices.long()] = percentages

        return codebook_usage_percentage



    def forward(self, z):
        """执行单尺度向量量化并返回码本统计信息。"""
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2) # 把 `(B, C, T, H, W)` 展平为 token 列表。

        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ self.embeddings.t() \
                    + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True) # 这里对应平方欧氏距离展开式。

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs) # 代码/形状说明：[bthw, ncode]
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:]) # 代码/形状说明：[b, t, h, w, ncode]

        embeddings = F.embedding(encoding_indices, self.embeddings) # [b, t, h, w, c]
        embeddings = shift_dim(embeddings, -1, 1) # [b, c, t, h, w]

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # 使用 EMA 更新码本中心，避免每步都通过梯度直接改 embedding。
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1) >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        try:
            usage = self.calculate_batch_codebook_usage_percentage(encoding_indices)
        except:
            usage = torch.zeros(self.n_codes, device=encoding_indices.device)


        # 代码/形状说明：print(usage.shape, torch.zeros(self.n_codes).shape)

        if self.call_cnt == 0:
            self.codebook_usage.data = usage
        else:
            self.codebook_usage.data = self.usage_sigma * self.codebook_usage.data + (1 - self.usage_sigma) * usage

        self.call_cnt += 1
        # 代码/形状说明：avg_distribution = self.codebook_usage.data.sum() / self.n_codes
        avg_usage = (self.codebook_usage.data > (1/self.n_codes)).sum() / self.n_codes

        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity, avg_usage=avg_usage, batch_usage=usage)

    def dictionary_lookup(self, encodings):
        """根据离散索引查回连续码本向量。"""
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings


# 中文标题：多尺度 Codebook
from typing import List, Optional, Tuple, Sequence, Union


class ResConvAfterUpsample(nn.Conv3d):
    """上采样后的量化残差卷积。"""
    def __init__(self, embed_dim, quant_resi):
        """构造量化残差卷积头。"""
        ks = 3 if quant_resi < 0 else 1
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BCthw):
        """把卷积残差按比例混入上采样后的量化特征。"""
        return h_BCthw.mul(1-self.resi_ratio) + super().forward(h_BCthw).mul_(self.resi_ratio)


class SharedResConvAfterUpsample(nn.Module):
    """所有尺度共享同一个残差卷积头的包装器。"""
    def __init__(self, qresi: ResConvAfterUpsample):
        """包装单个共享残差卷积头。"""
        super().__init__()
        self.qresi: ResConvAfterUpsample = qresi

    def __getitem__(self, _) -> ResConvAfterUpsample:
        """忽略索引，始终返回同一个残差卷积头。"""
        return self.qresi


class ResConvAfterUpsampleList(nn.Module):
    """按尺度位置选择残差卷积头的容器。"""
    def __init__(self, qresi_ls: nn.ModuleList):
        """按比例位置选择不同的残差卷积头。"""
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)

    def __getitem__(self, at_from_0_to_1: float) -> ResConvAfterUpsample:
        """根据 `0~1` 的尺度位置返回最近的残差头。"""
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        """打印每个残差头对应的尺度刻度。"""
        return f'ticks={self.ticks}'


class ResConvAfterUpsampleModuleList(nn.ModuleList):
    """`ModuleList` 形式的尺度残差卷积头容器。"""
    def __init__(self, qresi: List):
        """`ModuleList` 版尺度残差头容器。"""
        super().__init__(qresi)
        # 代码/形状说明：self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)

    def __getitem__(self, at_from_0_to_1: float) -> ResConvAfterUpsample:
        """根据尺度位置选择最近的模块。"""
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        """打印每个模块对应的尺度刻度。"""
        return f'ticks={self.ticks}'

class MultiScaleCodebook(nn.Module):
    """多尺度码本。

    它不是只在单一分辨率上找最近邻，而是在多个 `(T, H, W)` 尺度上逐步量化残差。
    可以把它理解成“从粗到细”的离散重建：前面尺度先抓全局结构，后面尺度再补细节。
    """
    def __init__(self, n_codes,
                embedding_dim, no_random_restart=False,
                restart_thres=1.0, usage_sigma=0.99, fp32_quant=False,
                quant_resi = -0.5, share_quant_resi = 4, default_qresi_counts = 10,
                t_patch_nums = (1, 1, 2, 2, 2, 4, 4, 4, 4, 4),
                v_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            ):
        """构造多尺度量化器及其量化残差卷积头。"""
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())
        self.register_buffer('codebook_usage', torch.zeros(n_codes))

        self.call_cnt = 0
        self.usage_sigma = usage_sigma

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

        self.fp32_quant = fp32_quant

        # `quant_resi` 控制每个尺度在上采样后是否再走一层残差卷积做细修正。

        self.t_patch_nums = t_patch_nums
        self.v_patch_nums = v_patch_nums
        self.quant_resi_ratio = quant_resi

        if share_quant_resi == 1:   # 中文说明：args.qsr
            self.quant_resi = SharedResConvAfterUpsample(ResConvAfterUpsample(embedding_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        elif share_quant_resi == 0:
            self.quant_resi = ResConvAfterUpsampleModuleList([(ResConvAfterUpsample(embedding_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        else:
            self.quant_resi = ResConvAfterUpsampleList(nn.ModuleList([(ResConvAfterUpsample(embedding_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))

        self.z_interplote_down = 'area'
        self.z_interplote_up = 'trilinear'



    def _tile(self, x):
        """在初始化阶段补足码本需要的样本数。"""
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        """使用当前 batch 的潜变量初始化多尺度码本。"""
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))


    def calculate_batch_codebook_usage_percentage(self, batch_encoding_indices):
        """统计一个 batch 在当前尺度下的码字使用占比。"""
        all_indices = batch_encoding_indices.flatten()

        total_indices = all_indices.numel()

        codebook_usage_percentage = torch.zeros(self.n_codes, device=all_indices.device)

        unique_indices, counts = torch.unique(all_indices, return_counts=True)
        percentages = (counts.float() / total_indices)

        codebook_usage_percentage[unique_indices.long()] = percentages

        return codebook_usage_percentage



    def forward(self, z):
        """执行多尺度残差量化。

        第 `si` 个尺度会先取出当前残差 `rest_z`，把它缩放到较粗尺度后做最近邻量化，
        再上采样回原尺度并累加到 `accu_h`。因此 `accu_h` 可以理解为“到当前尺度为止的
        累积重建”，其与原始 `z` 的差值就是下一尺度要继续编码的残差。
        """
        if self._need_init and self.training:
            self._init_embeddings(z)

        # 保留 `(T, H, W)` 结构做多尺度插值；仅在最近邻搜索时临时展平。
        B, C, T, H, W = z.shape

        z_no_grad = z.detach()
        accu_h = torch.zeros_like(z_no_grad)


        if self.training:
            all_flat_inputs, all_encode_onehot = [], []

        commitment_loss = 0.0
        scale_num = len(self.v_patch_nums)
        ms_encoding_indices = []


        with torch.cuda.amp.autocast(enabled=False):

            for si, (tpn, pn) in enumerate(zip(self.t_patch_nums, self.v_patch_nums)):
                tpn = min(tpn, T)

                # 当前尺度要处理的是“原特征 - 已累计重建”的残差。
                rest_z = z_no_grad - accu_h.data

                if si != scale_num - 1:  # 非最后一层时，先缩到当前尺度。
                    rest_z = F.interpolate(rest_z, size=(tpn, pn, pn), mode=self.z_interplote_down)

                z_NC =  rest_z.permute(0, 2, 3, 4, 1).reshape(-1, C)

                # 计算当前尺度残差与所有码字的距离。
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embeddings.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embeddings.t(), alpha=-2, beta=1)

                # 选取最近邻码字，得到离散索引。
                encoding_indices = torch.argmin(d_no_grad, dim=1)
                encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(z_NC) # 代码/形状说明：[bthw, ncode]
                encoding_indices = encoding_indices.view(rest_z.shape[0], *rest_z.shape[2:]) # 代码/形状说明：[b, t, h, w, ncode]

                ms_encoding_indices.append(encoding_indices)

                # 再把离散索引查回连续 embedding，得到当前尺度重建。
                h_BTHWC = F.embedding(encoding_indices, self.embeddings)    # [b, t, h, w, c]
                h_BCTHW = h_BTHWC.permute(0, 4, 1, 2, 3).contiguous()    # [b, c, t, h, w]

                # 上采样回原始 latent 尺度，并可选做一层量化残差卷积修正。
                h_BCTHW = F.interpolate(h_BCTHW, size=(T, H, W), mode=self.z_interplote_up).contiguous()

                quant_head = si / max(1, (scale_num - 1))
                h_BCTHW = self.quant_resi[quant_head](h_BCTHW)

                # 累积到当前的整体重建中，供下一尺度继续编码残差。
                accu_h = accu_h + h_BCTHW

                commitment_loss += 0.25 * F.mse_loss(accu_h, z.detach())  # 0.25 对应常见 VQ-VAE beta 系数。

                if self.training:
                    all_flat_inputs.append(z_NC)
                    all_encode_onehot.append(encode_onehot)

        if self.training:

            encode_onehot = torch.cat(all_encode_onehot, dim=0)
            flat_inputs = torch.cat(all_flat_inputs, dim=0)

            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1) >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        commitment_loss *= 1.0 / scale_num
        embeddings_st = (accu_h - z_no_grad).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        try:
            usage = self.calculate_batch_codebook_usage_percentage(encoding_indices)
        except:
            usage = torch.zeros(self.n_codes, device=encoding_indices.device)


        # 代码/形状说明：print(usage.shape, torch.zeros(self.n_codes).shape)

        if self.call_cnt == 0:
            self.codebook_usage.data = usage
        else:
            self.codebook_usage.data = self.usage_sigma * self.codebook_usage.data + (1 - self.usage_sigma) * usage

        self.call_cnt += 1
        # 代码/形状说明：avg_distribution = self.codebook_usage.data.sum() / self.n_codes
        avg_usage = (self.codebook_usage.data > (1/self.n_codes)).sum() / self.n_codes

        # 调试输出：print(f"训练中：{embeddings_st.size()=}, {encoding_indices.size()=}")
        # for idx, en_idx in enumerate(ms_encoding_indices):
        # 代码/形状说明：print(f"{idx=}, {en_idx.size()=}", flush=True)

        return dict(embeddings=embeddings_st, encodings=ms_encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity, avg_usage=avg_usage, batch_usage=usage)

    def dictionary_lookup(self, encodings):
        """根据离散索引反查码本向量。"""
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings
