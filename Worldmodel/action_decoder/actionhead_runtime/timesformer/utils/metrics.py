# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""指标计算函数。"""

import torch
import numpy as np

def topks_correct(preds, labels, ks):
    """
    根据预测、标签和 top-k 列表，计算每个 top-k 下的正确预测数。

    参数：
        preds (array): 预测数组，维度为 batchsize 的 `N x ClassNum`。
        labels (array): 标签数组，维度为 batchsize 的 `N`。
        ks (list): top-k 列表，例如 `ks = [1, 5]` 表示 top-1 和 top-5。

    返回：
        topks_correct (list): 正确预测数列表，其中第 `i` 项对应
            `top-ks[i]` 的正确预测数量。
    """
    assert preds.size(0) == labels.size(
        0
    ), "预测和标签的 batch 维度必须一致"
    # 为每个样本取前 max_k 个预测结果。
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # 形状/映射说明：(batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # 形状/映射说明：(batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # 若第 j 个样本的第 i 个 top 预测正确，则 (i, j) = 1。
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # 统计每个 k 对应的 top-k 正确预测数。
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    计算每个 k 对应的 top-k 错误率。

    参数：
        preds (array): 预测数组，维度为 `N`。
        labels (array): 标签数组，维度为 `N`。
        ks (list): 需要计算的 top-k 列表。
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    计算每个 k 对应的 top-k accuracy。

    参数：
        preds (array): 预测数组，维度为 `N`。
        labels (array): 标签数组，维度为 `N`。
        ks (list): 需要计算的 top-k 列表。
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]

def multitask_topks_correct(preds, labels, ks=(1,)):
    """
    计算多任务场景下的 top-k 正确数。

    参数：
        preds: `tuple(torch.FloatTensor)`，每个张量形状均为
            `[batch_size, class_count]`，不同任务的 `class_count` 可不同。
        labels: `tuple(torch.LongTensor)`，每个张量形状均为 `[batch_size]`。
        ks: `tuple(int)`，指定需要计算的 top-k。

    返回：
        tuple(float): 与 `ks` 等长，对应各个 `accuracy@k` 的正确样本数。
    """
    max_k = int(np.max(ks))
    task_count = len(preds)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda()
    for output, label in zip(preds, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # 交换 batch_size 和 class 维度，因为 `.view` 不适用于非连续张量。
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    multitask_topks_correct = [
        torch.ge(all_correct[:k].float().sum(0), task_count).float().sum(0) for k in ks
    ]

    return multitask_topks_correct
