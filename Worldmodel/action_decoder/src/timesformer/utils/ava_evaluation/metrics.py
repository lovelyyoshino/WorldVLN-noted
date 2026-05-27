# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""检测或分类指标计算工具。"""
from __future__ import division
import numpy as np


def compute_precision_recall(scores, labels, num_gt):
    """根据分数和真阳性/假阳性标记计算 precision/recall 曲线。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if (
        not isinstance(labels, np.ndarray)
        or labels.dtype != np.bool
        or len(labels.shape) != 1
    ):
        raise ValueError("labels 必须是一维 bool numpy array。")

    if not isinstance(scores, np.ndarray) or len(scores.shape) != 1:
        raise ValueError("scores 必须是一维 numpy array。")

    if num_gt < np.sum(labels):
        raise ValueError(
            "真阳性的数量不能超过 num_gt。"
        )

    if len(scores) != len(labels):
        raise ValueError("scores 和 labels 的长度必须一致。")

    if num_gt == 0:
        return None, None

    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    labels = labels.astype(int)
    true_positive_labels = labels[sorted_indices]
    false_positive_labels = 1 - true_positive_labels
    cum_true_positives = np.cumsum(true_positive_labels)
    cum_false_positives = np.cumsum(false_positive_labels)
    precision = cum_true_positives.astype(float) / (
        cum_true_positives + cum_false_positives
    )
    recall = cum_true_positives.astype(float) / num_gt
    return precision, recall


def compute_average_precision(precision, recall):
    """按照 VOC 定义对 precision/recall 曲线积分得到 AP。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if precision is None:
        if recall is not None:
            raise ValueError("如果 precision 为 None，recall 也必须为 None。")
        return np.NAN

    if not isinstance(precision, np.ndarray) or not isinstance(
        recall, np.ndarray
    ):
        raise ValueError("precision 和 recall 必须是 numpy array。")
    if precision.dtype != np.float or recall.dtype != np.float:
        raise ValueError("输入必须是 float numpy array。")
    if len(precision) != len(recall):
        raise ValueError("precision 和 recall 的长度必须一致。")
    if not precision.size:
        return 0.0
    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError("precision 必须位于 [0, 1] 范围内。")
    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError("recall 必须位于 [0, 1] 范围内。")
    if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
        raise ValueError("recall 必须是非递减数组。")

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # 预处理 precision，使其成为非递减数组。
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices]
    )
    return average_precision


def compute_cor_loc(
    num_gt_imgs_per_class, num_images_correctly_detected_per_class
):
    """计算 CorLoc 指标，用于弱监督检测场景。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    # 没有 GT 样本的类别可能出现除零。
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            num_gt_imgs_per_class == 0,
            np.nan,
            num_images_correctly_detected_per_class / num_gt_imgs_per_class,
        )
