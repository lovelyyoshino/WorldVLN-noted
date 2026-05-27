# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import torch.nn as nn


def init_weights(model: nn.Module, conv_std_or_gain: float = 0.02, other_std: float = 0.02):
    """
        参数 model：待初始化权重的模型。
        参数 conv_std_or_gain：控制每个卷积层 `m` 的初始化方式。
            公式/形状说明：> 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
            路径格式：< 0: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
        说明：:param other_std: 每个 linear layer 或 embedding layer 的初始化方式。
            公式/形状说明：使用 nn.init.trunc_normal_(m.weight.data, std=other_std)

    """
    skip = abs(conv_std_or_gain) > 10
    if skip: return
    print(f'[init_weights] {type(model).__name__} 使用 {"std" if conv_std_or_gain > 0 else "gain"}={abs(conv_std_or_gain):g}')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight.data, std=other_std)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, std=other_std)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain) if conv_std_or_gain > 0 else nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)   # 代码/形状说明：todo: StyleSwin: (..., gain=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
            if m.weight is not None:
                nn.init.constant_(m.weight.data, 1.)
