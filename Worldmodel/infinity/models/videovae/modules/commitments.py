# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DiagonalGaussianDistribution(object):
    """VAE 常用的对角高斯分布封装。"""
    def __init__(self, parameters, deterministic=False):
        """把 VAE 输出解释为对角高斯分布参数。

        输入会沿通道维拆成均值 `mean` 和对数方差 `logvar`。
        这种参数化允许使用重参数化技巧：
        `z = mean + exp(0.5 * logvar) * eps`，其中 `eps ~ N(0, I)`。
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        """按重参数化技巧采样潜变量。"""
        x = self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device)
        return x

    def kl(self, other=None, reduction="sum"):
        """计算 KL 散度。

        `other is None` 时，对应 `KL(q(z|x) || N(0, I))`；
        否则计算两个对角高斯之间的 KL。
        """
        if reduction == "sum":
            reduction_op = torch.sum
        elif reduction == "mean":
            reduction_op = torch.mean
        if self.mean.ndim == 4:
            dims = [1,2,3]
        else:
            dims = [1,2,3,4]
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * reduction_op(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=dims)
            else:
                return 0.5 * reduction_op(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=dims)

    def nll(self, sample, dims=[1,2,3]):
        """计算样本在该高斯分布下的负对数似然。"""
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        """返回分布众数；对高斯分布而言就是均值。"""
        return self.mean



def normal_kl(mean1, logvar1, mean2, logvar2):
    """计算两个高斯分布之间的 KL 散度。

    该实现允许标量与张量混合输入，广播规则与 PyTorch 一致。
    源实现来自 guided-diffusion。
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "至少一个参数必须是 Tensor"

    # 先把标量方差项显式转成张量；广播虽能处理加减乘除，但不能直接替代 `torch.exp`。
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

class VectorQuantizer(nn.Module):
    """基础向量量化器。"""
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        """构造基础向量量化器。

        核心思路是把连续潜变量映射到最近的离散码字，再用 straight-through estimator
        保持反向传播可达。
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))


    def forward(self, z):
        """执行最近邻量化并返回量化损失。

        距离公式使用平方欧氏距离展开式：
        `||z - e||^2 = ||z||^2 + ||e||^2 - 2 z^T e`。
        """
        # 先把通道移到最后，再展平成二维矩阵以便和码本逐向量比较。
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # 最近邻搜索使用平方欧氏距离展开式，避免显式构造 `(z - e)`。

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # 码本损失推动 embedding 靠近编码器输出，commitment 损失推动编码器承诺到选中的码字。
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2)
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # Straight-through estimator：前向使用量化值，反向让梯度穿过原始 z。
        z_q = z + (z_q - z).detach()

        # 恢复为输入期望的 `(B, C, H, W)` 布局。
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        """根据离散索引反查码本向量，并按目标形状重排。"""
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # 变形回原始输入形状。
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q

def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    """计算码本熵正则。

    该项以 `sample_entropy - avg_entropy` 的形式工作：
    既限制单样本分布不要过于尖锐，也鼓励 batch 平均分布更均匀，从而提升码本利用率。
    """
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("不支持 Entropy loss {}".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss
