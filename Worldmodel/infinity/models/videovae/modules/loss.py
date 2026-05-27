# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

def hinge_d_loss(logits_real, logits_fake):
    """计算判别器的 hinge loss。

        公式为：
        公式/形状说明：`L_real = mean(max(0, 1 - D(x)))`
        公式/形状说明：`L_fake = mean(max(0, 1 + D(G(z))))`
        公式/形状说明：`L_D = 0.5 * (L_real + L_fake)`。

    """
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    """计算基于 `softplus` 的原始 GAN 判别器损失。"""
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def get_disc_loss(disc_loss_type):
    """根据名称返回判别器损失函数。"""
    if disc_loss_type == 'vanilla':
        disc_loss = vanilla_d_loss
    elif disc_loss_type == 'hinge':
        disc_loss = hinge_d_loss
    return disc_loss

def adopt_weight(global_step, threshold=0, value=0., warmup=0):
    """按训练步数启用或预热某个损失权重。"""
    if global_step < threshold or threshold < 0:
        weight = value
    else:
        weight = 1
        if global_step - threshold < warmup:
            weight = min((global_step - threshold) / warmup, 1)
    return weight

def gradient_penalty(discriminator, real_data, fake_data, device):
    """计算 WGAN-GP 风格的梯度惩罚 `E[(||∇D||_2 - 1)^2]`。"""
    alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand_as(real_data)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    d_interpolates = discriminator(interpolates)
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class InfoNCELoss(nn.Module):
    """InfoNCE 对比损失。

    对一对增强视图 `(features, features_prime)`，正样本来自同一原始样本，
    负样本来自 batch 内其他样本。相似度除以温度 `tau` 后进入交叉熵。
    """
    def __init__(self, temperature: float = 0.07):
        """设置温度系数，温度越小越强调最相近样本的差异。"""
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, features_prime: torch.Tensor) -> torch.Tensor:
        """计算两组特征之间的双向对比损失。"""
        batch_size = features.shape[0]

        # 先做 L2 归一化，使点积直接对应余弦相似度。
        features = F.normalize(features, dim=1)
        features_prime = F.normalize(features_prime, dim=1)

        # 拼接两份视图，形成 2B x 2B 的相似度矩阵。
        combined_features = torch.cat([features, features_prime], dim=0)

        # 每个元素表示一对样本的温度缩放相似度。
        similarity_matrix = torch.matmul(combined_features, combined_features.T) / self.temperature

        # 屏蔽主对角线，避免样本与自身配对。
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(features.device)
        similarity_matrix.masked_fill_(mask, float('-inf'))

        # 同一原样本的两份视图在另一半 batch 中处于相同下标。
        labels = torch.arange(batch_size).to(features.device)
        labels = torch.cat([labels, labels], dim=0)

        # 取出互为正样本的两个子矩阵作为交叉熵 logits。
        positives_logits = torch.cat([similarity_matrix[:batch_size, batch_size:], similarity_matrix[batch_size:, :batch_size]], dim=0)

        # 代码/形状说明：labels 形如 [0, 1, 2, ..., batch_size-1, 0, 1, 2, ..., batch_size-1]。
        loss = F.cross_entropy(positives_logits, labels)

        return loss
