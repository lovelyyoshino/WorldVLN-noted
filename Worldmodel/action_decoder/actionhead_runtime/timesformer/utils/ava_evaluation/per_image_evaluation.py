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
"""单图目标检测评估逻辑。"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import numpy as np

from . import (
    np_box_list,
    np_box_list_ops,
    np_box_mask_list,
    np_box_mask_list_ops,
)


class PerImageEvaluation(object):
    """单张图像级别的目标检测评估器。 小白可以先看 `__init__` 保存了哪些字段，再看其他方法如何读取这些字段。

    数据流提示：类属性通常在初始化时写入，后续方法通过这些属性完成评估、采样或状态转换。
    """

    def __init__(self, num_groundtruth_classes, matching_iou_threshold=0.5):
        """初始化当前对象，保存后续方法会复用的配置、字段或评估状态。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
        """
        self.matching_iou_threshold = matching_iou_threshold
        self.num_groundtruth_classes = num_groundtruth_classes

    def compute_object_detection_metrics(
        self,
        detected_boxes,
        detected_scores,
        detected_class_labels,
        groundtruth_boxes,
        groundtruth_class_labels,
        groundtruth_is_difficult_list,
        groundtruth_is_group_of_list,
        detected_masks=None,
        groundtruth_masks=None,
    ):
        """对单张图像计算检测结果的真阳性、假阳性和忽略标记。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
        """
        (
            detected_boxes,
            detected_scores,
            detected_class_labels,
            detected_masks,
        ) = self._remove_invalid_boxes(
            detected_boxes,
            detected_scores,
            detected_class_labels,
            detected_masks,
        )
        scores, tp_fp_labels = self._compute_tp_fp(
            detected_boxes=detected_boxes,
            detected_scores=detected_scores,
            detected_class_labels=detected_class_labels,
            groundtruth_boxes=groundtruth_boxes,
            groundtruth_class_labels=groundtruth_class_labels,
            groundtruth_is_difficult_list=groundtruth_is_difficult_list,
            groundtruth_is_group_of_list=groundtruth_is_group_of_list,
            detected_masks=detected_masks,
            groundtruth_masks=groundtruth_masks,
        )

        return scores, tp_fp_labels

    def _compute_tp_fp(
        self,
        detected_boxes,
        detected_scores,
        detected_class_labels,
        groundtruth_boxes,
        groundtruth_class_labels,
        groundtruth_is_difficult_list,
        groundtruth_is_group_of_list,
        detected_masks=None,
        groundtruth_masks=None,
    ):
        """按类别为检测框分配真阳性或假阳性。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
        """
        if detected_masks is not None and groundtruth_masks is None:
            raise ValueError(
                "检测掩码已提供，但真实标注掩码未提供。"
            )
        if detected_masks is None and groundtruth_masks is not None:
            raise ValueError(
                "真实标注掩码已提供，但检测掩码未提供。"
            )

        result_scores = []
        result_tp_fp_labels = []
        for i in range(self.num_groundtruth_classes):
            groundtruth_is_difficult_list_at_ith_class = groundtruth_is_difficult_list[
                groundtruth_class_labels == i
            ]
            groundtruth_is_group_of_list_at_ith_class = groundtruth_is_group_of_list[
                groundtruth_class_labels == i
            ]
            (
                gt_boxes_at_ith_class,
                gt_masks_at_ith_class,
                detected_boxes_at_ith_class,
                detected_scores_at_ith_class,
                detected_masks_at_ith_class,
            ) = self._get_ith_class_arrays(
                detected_boxes,
                detected_scores,
                detected_masks,
                detected_class_labels,
                groundtruth_boxes,
                groundtruth_masks,
                groundtruth_class_labels,
                i,
            )
            scores, tp_fp_labels = self._compute_tp_fp_for_single_class(
                detected_boxes=detected_boxes_at_ith_class,
                detected_scores=detected_scores_at_ith_class,
                groundtruth_boxes=gt_boxes_at_ith_class,
                groundtruth_is_difficult_list=groundtruth_is_difficult_list_at_ith_class,
                groundtruth_is_group_of_list=groundtruth_is_group_of_list_at_ith_class,
                detected_masks=detected_masks_at_ith_class,
                groundtruth_masks=gt_masks_at_ith_class,
            )
            result_scores.append(scores)
            result_tp_fp_labels.append(tp_fp_labels)
        return result_scores, result_tp_fp_labels

    def _get_overlaps_and_scores_box_mode(
        self,
        detected_boxes,
        detected_scores,
        groundtruth_boxes,
        groundtruth_is_group_of_list,
    ):
        """在框模式下计算预测框与真实框的重叠度和分数。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
        """
        detected_boxlist = np_box_list.BoxList(detected_boxes)
        detected_boxlist.add_field("scores", detected_scores)
        gt_non_group_of_boxlist = np_box_list.BoxList(
            groundtruth_boxes[~groundtruth_is_group_of_list]
        )
        iou = np_box_list_ops.iou(detected_boxlist, gt_non_group_of_boxlist)
        scores = detected_boxlist.get_field("scores")
        num_boxes = detected_boxlist.num_boxes()
        return iou, None, scores, num_boxes

    def _compute_tp_fp_for_single_class(
        self,
        detected_boxes,
        detected_scores,
        groundtruth_boxes,
        groundtruth_is_difficult_list,
        groundtruth_is_group_of_list,
        detected_masks=None,
        groundtruth_masks=None,
    ):
        """对单个类别计算检测框的 TP/FP 归属。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
        """
        if detected_boxes.size == 0:
            return np.array([], dtype=float), np.array([], dtype=bool)

        (
            iou,
            _,
            scores,
            num_detected_boxes,
        ) = self._get_overlaps_and_scores_box_mode(
            detected_boxes=detected_boxes,
            detected_scores=detected_scores,
            groundtruth_boxes=groundtruth_boxes,
            groundtruth_is_group_of_list=groundtruth_is_group_of_list,
        )

        if groundtruth_boxes.size == 0:
            return scores, np.zeros(num_detected_boxes, dtype=bool)

        tp_fp_labels = np.zeros(num_detected_boxes, dtype=bool)
        is_matched_to_difficult_box = np.zeros(num_detected_boxes, dtype=bool)
        is_matched_to_group_of_box = np.zeros(num_detected_boxes, dtype=bool)

        # 评估分两阶段完成：
        # 1. 所有检测结果先与非 group-of 框匹配，用来确定真阳性；
        #    匹配到 difficult 框的检测结果会被忽略。
        # 2. 判定为假阳性的检测结果再与 group-of 框匹配，匹配成功时也会被忽略。

        # 对非 group-of 框执行 TP/FP 评估。
        if iou.shape[1] > 0:
            groundtruth_nongroup_of_is_difficult_list = groundtruth_is_difficult_list[
                ~groundtruth_is_group_of_list
            ]
            max_overlap_gt_ids = np.argmax(iou, axis=1)
            is_gt_box_detected = np.zeros(iou.shape[1], dtype=bool)
            for i in range(num_detected_boxes):
                gt_id = max_overlap_gt_ids[i]
                if iou[i, gt_id] >= self.matching_iou_threshold:
                    if not groundtruth_nongroup_of_is_difficult_list[gt_id]:
                        if not is_gt_box_detected[gt_id]:
                            tp_fp_labels[i] = True
                            is_gt_box_detected[gt_id] = True
                    else:
                        is_matched_to_difficult_box[i] = True

        return (
            scores[~is_matched_to_difficult_box & ~is_matched_to_group_of_box],
            tp_fp_labels[
                ~is_matched_to_difficult_box & ~is_matched_to_group_of_box
            ],
        )

    def _get_ith_class_arrays(
        self,
        detected_boxes,
        detected_scores,
        detected_masks,
        detected_class_labels,
        groundtruth_boxes,
        groundtruth_masks,
        groundtruth_class_labels,
        class_index,
    ):
        """取出指定类别对应的检测框、分数和真实框数组。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
        """
        selected_groundtruth = groundtruth_class_labels == class_index
        gt_boxes_at_ith_class = groundtruth_boxes[selected_groundtruth]
        if groundtruth_masks is not None:
            gt_masks_at_ith_class = groundtruth_masks[selected_groundtruth]
        else:
            gt_masks_at_ith_class = None
        selected_detections = detected_class_labels == class_index
        detected_boxes_at_ith_class = detected_boxes[selected_detections]
        detected_scores_at_ith_class = detected_scores[selected_detections]
        if detected_masks is not None:
            detected_masks_at_ith_class = detected_masks[selected_detections]
        else:
            detected_masks_at_ith_class = None
        return (
            gt_boxes_at_ith_class,
            gt_masks_at_ith_class,
            detected_boxes_at_ith_class,
            detected_scores_at_ith_class,
            detected_masks_at_ith_class,
        )

    def _remove_invalid_boxes(
        self,
        detected_boxes,
        detected_scores,
        detected_class_labels,
        detected_masks=None,
    ):
        """过滤无效边界框，避免面积或坐标异常影响评估。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
        """
        valid_indices = np.logical_and(
            detected_boxes[:, 0] < detected_boxes[:, 2],
            detected_boxes[:, 1] < detected_boxes[:, 3],
        )
        detected_boxes = detected_boxes[valid_indices]
        detected_scores = detected_scores[valid_indices]
        detected_class_labels = detected_class_labels[valid_indices]
        if detected_masks is not None:
            detected_masks = detected_masks[valid_indices]
        return [
            detected_boxes,
            detected_scores,
            detected_class_labels,
            detected_masks,
        ]
