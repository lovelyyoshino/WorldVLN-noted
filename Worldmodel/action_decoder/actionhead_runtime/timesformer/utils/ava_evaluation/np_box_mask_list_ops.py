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

"""带掩码框列表的集合运算工具。"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import numpy as np

from . import np_box_list_ops, np_box_mask_list, np_mask_ops


def box_list_to_box_mask_list(boxlist):
    """把包含掩码字段的 BoxList 转换为带掩码框列表。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if not boxlist.has_field("masks"):
        raise ValueError("boxlist 不包含掩码字段。")
    box_mask_list = np_box_mask_list.BoxMaskList(
        box_data=boxlist.get(), mask_data=boxlist.get_field("masks")
    )
    extra_fields = boxlist.get_extra_fields()
    for key in extra_fields:
        if key != "masks":
            box_mask_list.data[key] = boxlist.get_field(key)
    return box_mask_list


def area(box_mask_list):
    """计算框或掩码的面积。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    return np_mask_ops.area(box_mask_list.get_masks())


def intersection(box_mask_list1, box_mask_list2):
    """计算两组框或掩码的两两交集面积。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    return np_mask_ops.intersection(
        box_mask_list1.get_masks(), box_mask_list2.get_masks()
    )


def iou(box_mask_list1, box_mask_list2):
    """计算两组框或掩码的交并比。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    return np_mask_ops.iou(
        box_mask_list1.get_masks(), box_mask_list2.get_masks()
    )


def ioa(box_mask_list1, box_mask_list2):
    """计算两组框或掩码的交集面积占比。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    return np_mask_ops.ioa(
        box_mask_list1.get_masks(), box_mask_list2.get_masks()
    )


def gather(box_mask_list, indices, fields=None):
    """按索引收集框、掩码及附加字段。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if fields is not None:
        if "masks" not in fields:
            fields.append("masks")
    return box_list_to_box_mask_list(
        np_box_list_ops.gather(
            boxlist=box_mask_list, indices=indices, fields=fields
        )
    )


def sort_by_field(
    box_mask_list, field, order=np_box_list_ops.SortOrder.DESCEND
):
    """按指定标量字段对框集合排序。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    return box_list_to_box_mask_list(
        np_box_list_ops.sort_by_field(
            boxlist=box_mask_list, field=field, order=order
        )
    )


def non_max_suppression(
    box_mask_list,
    max_output_size=10000,
    iou_threshold=1.0,
    score_threshold=-10.0,
):
    """执行非极大值抑制，去掉高度重叠的重复检测。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if not box_mask_list.has_field("scores"):
        raise ValueError("字段 scores 不存在。")
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError("IOU threshold 必须位于 [0, 1]。")
    if max_output_size < 0:
        raise ValueError("max_output_size 必须大于等于 0。")

    box_mask_list = filter_scores_greater_than(box_mask_list, score_threshold)
    if box_mask_list.num_boxes() == 0:
        return box_mask_list

    box_mask_list = sort_by_field(box_mask_list, "scores")

    # 如果禁用 NMS，就跳过后续计算。
    if iou_threshold == 1.0:
        if box_mask_list.num_boxes() > max_output_size:
            selected_indices = np.arange(max_output_size)
            return gather(box_mask_list, selected_indices)
        else:
            return box_mask_list

    masks = box_mask_list.get_masks()
    num_masks = box_mask_list.num_boxes()

    # is_index_valid 只会对仍然有效的候选掩码保持 True。
    is_index_valid = np.full(num_masks, 1, dtype=bool)
    selected_indices = []
    num_output = 0
    for i in range(num_masks):
        if num_output < max_output_size:
            if is_index_valid[i]:
                num_output += 1
                selected_indices.append(i)
                is_index_valid[i] = False
                valid_indices = np.where(is_index_valid)[0]
                if valid_indices.size == 0:
                    break

                intersect_over_union = np_mask_ops.iou(
                    np.expand_dims(masks[i], axis=0), masks[valid_indices]
                )
                intersect_over_union = np.squeeze(intersect_over_union, axis=0)
                is_index_valid[valid_indices] = np.logical_and(
                    is_index_valid[valid_indices],
                    intersect_over_union <= iou_threshold,
                )
    return gather(box_mask_list, np.array(selected_indices))


def multi_class_non_max_suppression(
    box_mask_list, score_thresh, iou_thresh, max_output_size
):
    """按类别分别执行 NMS，再合并多类别检测结果。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if not 0 <= iou_thresh <= 1.0:
        raise ValueError("thresh 必须位于 0 到 1 之间。")
    if not isinstance(box_mask_list, np_box_mask_list.BoxMaskList):
        raise ValueError("box_mask_list 必须是 BoxMaskList。")
    if not box_mask_list.has_field("scores"):
        raise ValueError("输入 box_mask_list 必须包含 'scores' 字段。")
    scores = box_mask_list.get_field("scores")
    if len(scores.shape) == 1:
        scores = np.reshape(scores, [-1, 1])
    elif len(scores.shape) == 2:
        if scores.shape[1] is None:
            raise ValueError(
                "scores 字段必须静态定义第二维。"
            )
    else:
        raise ValueError("scores 字段必须是 1 维或 2 维。")

    num_boxes = box_mask_list.num_boxes()
    num_scores = scores.shape[0]
    num_classes = scores.shape[1]

    if num_boxes != num_scores:
        raise ValueError("scores 字段长度错误：实际长度与期望长度不一致。")

    selected_boxes_list = []
    for class_idx in range(num_classes):
        box_mask_list_and_class_scores = np_box_mask_list.BoxMaskList(
            box_data=box_mask_list.get(), mask_data=box_mask_list.get_masks()
        )
        class_scores = np.reshape(scores[0:num_scores, class_idx], [-1])
        box_mask_list_and_class_scores.add_field("scores", class_scores)
        box_mask_list_filt = filter_scores_greater_than(
            box_mask_list_and_class_scores, score_thresh
        )
        nms_result = non_max_suppression(
            box_mask_list_filt,
            max_output_size=max_output_size,
            iou_threshold=iou_thresh,
            score_threshold=score_thresh,
        )
        nms_result.add_field(
            "classes", np.zeros_like(nms_result.get_field("scores")) + class_idx
        )
        selected_boxes_list.append(nms_result)
    selected_boxes = np_box_list_ops.concatenate(selected_boxes_list)
    sorted_boxes = np_box_list_ops.sort_by_field(selected_boxes, "scores")
    return box_list_to_box_mask_list(boxlist=sorted_boxes)


def prune_non_overlapping_masks(box_mask_list1, box_mask_list2, minoverlap=0.0):
    """移除与参考掩码重叠不足的候选项。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    intersection_over_area = ioa(
        box_mask_list2, box_mask_list1
    )  # 形状/映射说明：[M, N] tensor
    intersection_over_area = np.amax(
        intersection_over_area, axis=0
    )  # 形状/映射说明：[N] tensor
    keep_bool = np.greater_equal(intersection_over_area, np.array(minoverlap))
    keep_inds = np.nonzero(keep_bool)[0]
    new_box_mask_list1 = gather(box_mask_list1, keep_inds)
    return new_box_mask_list1


def concatenate(box_mask_lists, fields=None):
    """把多个 BoxList/BoxMaskList 沿样本维拼接。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if fields is not None:
        if "masks" not in fields:
            fields.append("masks")
    return box_list_to_box_mask_list(
        np_box_list_ops.concatenate(boxlists=box_mask_lists, fields=fields)
    )


def filter_scores_greater_than(box_mask_list, thresh):
    """只保留分数大于阈值的检测框或掩码。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if not isinstance(box_mask_list, np_box_mask_list.BoxMaskList):
        raise ValueError("box_mask_list 必须是 BoxMaskList。")
    if not box_mask_list.has_field("scores"):
        raise ValueError("输入 box_mask_list 必须包含 'scores' 字段。")
    scores = box_mask_list.get_field("scores")
    if len(scores.shape) > 2:
        raise ValueError("scores 必须是 1 维或 2 维。")
    if len(scores.shape) == 2 and scores.shape[1] != 1:
        raise ValueError(
            "scores 应为 1 维，或 shape 与 [None, 1] 一致。"
        )
    high_score_indices = np.reshape(
        np.where(np.greater(scores, thresh)), [-1]
    ).astype(np.int32)
    return gather(box_mask_list, high_score_indices)
