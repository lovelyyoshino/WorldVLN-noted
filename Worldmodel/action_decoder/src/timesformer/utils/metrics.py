# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""检测或分类指标计算工具。"""

import torch
import numpy as np

def topks_correct(preds, labels, ks):
    """计算 top-k 预测中命中真实标签的样本数。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    assert preds.size(0) == labels.size(
        0
    ), "预测和标签的 batch 维度必须一致"
    # 为每个样本找到 top max_k 预测。
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # 形状/映射说明：(batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # 形状/映射说明：(batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # 若第 j 个样本的第 i 个 top 预测正确，则 (i, j) = 1。
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # 计算每个 k 下 top-k 命中的数量。
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """计算不同 k 下的 top-k 错误率。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """计算不同 k 下的 top-k 准确率。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]

def multitask_topks_correct(preds, labels, ks=(1,)):
    """多任务分类场景下分别计算各任务的 top-k 命中数。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    max_k = int(np.max(ks))
    task_count = len(preds)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda()
    for output, label in zip(preds, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # 中文说明：Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    multitask_topks_correct = [
        torch.ge(all_correct[:k].float().sum(0), task_count).float().sum(0) for k in ks
    ]

    return multitask_topks_correct
