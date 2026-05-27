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
"""目标检测评估模块。

ObjectDetectionEvaluation 用来管理目标检测数据集的真实标注信息，并根据
提供的检测结果计算常用指标，例如 Precision、Recall、CorLoc。
它支持以下操作：
1) 按顺序加入图像的真实标注信息。
2) 按顺序加入图像的检测结果。
3) 基于已经插入的检测结果评估检测指标。
4) 将评估结果写入 pickle 文件，便于后续处理或可视化。

注意：本模块操作的是 numpy 边界框数组和框列表。
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import collections
import logging
import numpy as np
from abc import ABCMeta, abstractmethod

from . import label_map_util, metrics, per_image_evaluation, standard_fields


class DetectionEvaluator(object):
    """目标检测评估类的接口。

      评估器使用示例：
      ------------------------------
      说明：evaluator = DetectionEvaluator(categories)

      # 图像 1 的检测结果和真实标注。
      说明：evaluator.add_single_groundtruth_image_info(...)
      说明：evaluator.add_single_detected_image_info(...)

      # 图像 2 的检测结果和真实标注。
      说明：evaluator.add_single_groundtruth_image_info(...)
      说明：evaluator.add_single_detected_image_info(...)

      说明：metrics_dict = evaluator.evaluate()

    """

    __metaclass__ = ABCMeta

    def __init__(self, categories):
        """保存类别定义，供后续评估时生成每类指标。

            参数：
              categories: dict 列表，每个 dict 包含以下键：
                'id': 必需，唯一标识该类别的整数 id。
                'name': 必需，类别名称字符串，例如 'cat'、'dog'。

        """
        self._categories = categories

    @abstractmethod
    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """加入单张图像的真实标注信息，用于后续评估。

            参数：
              image_id: 图像的唯一字符串或整数标识。
              groundtruth_dict: 评估所需的真实标注 numpy array 字典。

        """
        pass

    @abstractmethod
    def add_single_detected_image_info(self, image_id, detections_dict):
        """加入单张图像的检测结果信息，用于后续评估。

            参数：
              image_id: 图像的唯一字符串或整数标识。
              detections_dict: 评估所需的检测结果 numpy array 字典。

        """
        pass

    @abstractmethod
    def evaluate(self):
        """评估检测结果，并返回一个指标字典。"""
        pass

    @abstractmethod
    def clear(self):
        """清空内部状态，为一次新的评估做准备。"""
        pass


class ObjectDetectionEvaluator(DetectionEvaluator):
    """用于评估目标检测结果的类。"""

    def __init__(
        self,
        categories,
        matching_iou_threshold=0.5,
        evaluate_corlocs=False,
        metric_prefix=None,
        use_weighted_mean_ap=False,
        evaluate_masks=False,
    ):
        """初始化检测评估器及其内部统计对象。

            参数：
              categories: dict 列表，每个 dict 包含以下键：
                'id': 必需，唯一标识该类别的整数 id。
                'name': 必需，类别名称字符串，例如 'cat'、'dog'。
              matching_iou_threshold: 匹配真实标注框和检测框时使用的
                IOU 阈值。
              evaluate_corlocs: 可选，是否返回 corloc 分数。
              metric_prefix: 可选，指标名称前缀；为 None 时不使用前缀。
              use_weighted_mean_ap: 可选，是否直接基于所有类别的 scores 和 tp_fp_labels
                计算平均精度均值。
              evaluate_masks: 如果为 False，基于框评估；如果为 True，改为评估掩码。

            异常：
              ValueError: 如果类别 id 不是从 1 开始编号。

        """
        super(ObjectDetectionEvaluator, self).__init__(categories)
        self._num_classes = max([cat["id"] for cat in categories])
        if min(cat["id"] for cat in categories) < 1:
            raise ValueError("类别应从 1 开始编号。")
        self._matching_iou_threshold = matching_iou_threshold
        self._use_weighted_mean_ap = use_weighted_mean_ap
        self._label_id_offset = 1
        self._evaluate_masks = evaluate_masks
        self._evaluation = ObjectDetectionEvaluation(
            num_groundtruth_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset,
        )
        self._image_ids = set([])
        self._evaluate_corlocs = evaluate_corlocs
        self._metric_prefix = (metric_prefix + "_") if metric_prefix else ""

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """加入单张图像的真实标注信息，用于评估。

            参数：
              image_id: 图像的唯一字符串或整数标识。
              groundtruth_dict: 包含以下内容的字典：
                说明：standard_fields.InputDataFields.groundtruth_boxes: float32 numpy array
                  shape 为 [num_boxes, 4]，包含 `num_boxes` 个真实标注框，
                  格式为绝对图像坐标下的 [ymin, xmin, ymax, xmax]。
                说明：standard_fields.InputDataFields.groundtruth_classes: integer numpy array
                  shape 为 [num_boxes]，包含从 1 开始编号的框类别。
                说明：standard_fields.InputDataFields.groundtruth_difficult: Optional length
                  M 的 numpy boolean array，表示真实标注框是否为 difficult
                  实例。该字段是可选的，用于支持没有 difficult 框的情况。
                说明：standard_fields.InputDataFields.groundtruth_instance_masks: Optional
                  shape 为 [num_boxes, height, width] 的 numpy array，取值在 {0, 1}。

            异常：
              ValueError: 如果同一图像重复加入真实标注；如果评估掩码但字典中
                没有实例掩码，也会报错。

        """
        if image_id in self._image_ids:
            raise ValueError("id 为 {} 的图像已添加。".format(image_id))

        groundtruth_classes = (
            groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_classes
            ]
            - self._label_id_offset
        )
        # 如果 key 不在 groundtruth_dict 中，或数组为空（除非该图像确实没有
        # 真实标注），则优先使用字典中的值，否则填入 None。
        if standard_fields.InputDataFields.groundtruth_difficult in groundtruth_dict.keys() and (
            groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_difficult
            ].size
            or not groundtruth_classes.size
        ):
            groundtruth_difficult = groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_difficult
            ]
        else:
            groundtruth_difficult = None
            if not len(self._image_ids) % 1000:
                logging.warn(
                    "图像 %s 未指定真实标注 difficult 标记",
                    image_id,
                )
        groundtruth_masks = None
        if self._evaluate_masks:
            if (
                standard_fields.InputDataFields.groundtruth_instance_masks
                not in groundtruth_dict
            ):
                raise ValueError(
                    "真实标注字典中没有实例掩码。"
                )
            groundtruth_masks = groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_instance_masks
            ]
        self._evaluation.add_single_ground_truth_image_info(
            image_key=image_id,
            groundtruth_boxes=groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_boxes
            ],
            groundtruth_class_labels=groundtruth_classes,
            groundtruth_is_difficult_list=groundtruth_difficult,
            groundtruth_masks=groundtruth_masks,
        )
        self._image_ids.update([image_id])

    def add_single_detected_image_info(self, image_id, detections_dict):
        """加入单张图像的检测结果信息，用于评估。

            参数：
              image_id: 图像的唯一字符串或整数标识。
              detections_dict: 包含以下内容的字典：
                说明：standard_fields.DetectionResultFields.detection_boxes: float32 numpy
                  array，shape 为 [num_boxes, 4]，包含 `num_boxes` 个检测框，
                  格式为绝对图像坐标下的 [ymin, xmin, ymax, xmax]。
                说明：standard_fields.DetectionResultFields.detection_scores: float32 numpy
                  array，shape 为 [num_boxes]，包含每个框的检测分数。
                说明：standard_fields.DetectionResultFields.detection_classes: integer numpy
                  array，shape 为 [num_boxes]，包含从 1 开始编号的检测类别。
                说明：standard_fields.DetectionResultFields.detection_masks: uint8 numpy
                  array，shape 为 [num_boxes, height, width]，包含 `num_boxes` 个取值在
                  0 到 1 之间的掩码。

            异常：
              ValueError: 如果需要评估掩码但 detections 字典中没有检测掩码。

        """
        detection_classes = (
            detections_dict[
                standard_fields.DetectionResultFields.detection_classes
            ]
            - self._label_id_offset
        )
        detection_masks = None
        if self._evaluate_masks:
            if (
                standard_fields.DetectionResultFields.detection_masks
                not in detections_dict
            ):
                raise ValueError(
                    "检测结果字典中没有检测掩码。"
                )
            detection_masks = detections_dict[
                standard_fields.DetectionResultFields.detection_masks
            ]
        self._evaluation.add_single_detected_image_info(
            image_key=image_id,
            detected_boxes=detections_dict[
                standard_fields.DetectionResultFields.detection_boxes
            ],
            detected_scores=detections_dict[
                standard_fields.DetectionResultFields.detection_scores
            ],
            detected_class_labels=detection_classes,
            detected_masks=detection_masks,
        )

    def evaluate(self):
        """计算评估结果。

            返回：
              包含以下字段的指标字典：

              说明：1. summary_metrics:
                'Precision/mAP@<matching_iou_threshold>IOU': 指定 IOU 阈值下的
                说明：平均精度均值。

              2. per_category_ap: 每个类别的结果，key 形式为
                说明：'PerformanceByCategory/mAP@<matching_iou_threshold>IOU/category'.

        """
        (
            per_class_ap,
            mean_ap,
            _,
            _,
            per_class_corloc,
            mean_corloc,
        ) = self._evaluation.evaluate()
        pascal_metrics = {
            self._metric_prefix
            + "Precision/mAP@{}IOU".format(
                self._matching_iou_threshold
            ): mean_ap
        }
        if self._evaluate_corlocs:
            pascal_metrics[
                self._metric_prefix
                + "Precision/meanCorLoc@{}IOU".format(
                    self._matching_iou_threshold
                )
            ] = mean_corloc
        category_index = label_map_util.create_category_index(self._categories)
        for idx in range(per_class_ap.size):
            if idx + self._label_id_offset in category_index:
                display_name = (
                    self._metric_prefix
                    + "PerformanceByCategory/AP@{}IOU/{}".format(
                        self._matching_iou_threshold,
                        category_index[idx + self._label_id_offset]["name"],
                    )
                )
                pascal_metrics[display_name] = per_class_ap[idx]

                # 可选地加入 CorLoc 类别指标。
                if self._evaluate_corlocs:
                    display_name = (
                        self._metric_prefix
                        + "PerformanceByCategory/CorLoc@{}IOU/{}".format(
                            self._matching_iou_threshold,
                            category_index[idx + self._label_id_offset]["name"],
                        )
                    )
                    pascal_metrics[display_name] = per_class_corloc[idx]

        return pascal_metrics

    def clear(self):
        """清空内部状态，为一次新的评估做准备。"""
        self._evaluation = ObjectDetectionEvaluation(
            num_groundtruth_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset,
        )
        self._image_ids.clear()


class PascalDetectionEvaluator(ObjectDetectionEvaluator):
    """使用 PASCAL 指标评估检测结果的类。"""

    def __init__(self, categories, matching_iou_threshold=0.5):
        """创建使用 PASCAL Boxes 指标的检测评估器。"""
        super(PascalDetectionEvaluator, self).__init__(
            categories,
            matching_iou_threshold=matching_iou_threshold,
            evaluate_corlocs=False,
            metric_prefix="PascalBoxes",
            use_weighted_mean_ap=False,
        )


class WeightedPascalDetectionEvaluator(ObjectDetectionEvaluator):
    """使用加权 PASCAL 指标评估检测结果的类。

  加权 PASCAL 指标会把所有类别的 scores 和 tp_fp_labels 放在一起计算
  平均精度，并将它作为平均精度均值。相比之下，PASCAL 指标
  会把每个类别的平均精度再取平均。

  这个定义很接近“按类别频率加权的每类平均精度均值”。但由于
  平均精度不是 scores 和 tp_fp_labels 的线性函数，两者通常并不相同。
  """

    def __init__(self, categories, matching_iou_threshold=0.5):
        """创建使用加权 PASCAL Boxes 指标的检测评估器。"""
        super(WeightedPascalDetectionEvaluator, self).__init__(
            categories,
            matching_iou_threshold=matching_iou_threshold,
            evaluate_corlocs=False,
            metric_prefix="WeightedPascalBoxes",
            use_weighted_mean_ap=True,
        )


class PascalInstanceSegmentationEvaluator(ObjectDetectionEvaluator):
    """使用 PASCAL 指标评估实例掩码的类。"""

    def __init__(self, categories, matching_iou_threshold=0.5):
        """创建使用 PASCAL Masks 指标的实例分割评估器。"""
        super(PascalInstanceSegmentationEvaluator, self).__init__(
            categories,
            matching_iou_threshold=matching_iou_threshold,
            evaluate_corlocs=False,
            metric_prefix="PascalMasks",
            use_weighted_mean_ap=False,
            evaluate_masks=True,
        )


class WeightedPascalInstanceSegmentationEvaluator(ObjectDetectionEvaluator):
    """使用加权 PASCAL 指标评估实例掩码的类。

  加权 PASCAL 指标会把所有类别的 scores 和 tp_fp_labels 放在一起计算
  平均精度，并将它作为平均精度均值。相比之下，PASCAL 指标
  会把每个类别的平均精度再取平均。

  这个定义很接近“按类别频率加权的每类平均精度均值”。但由于
  平均精度不是 scores 和 tp_fp_labels 的线性函数，两者通常并不相同。
  """

    def __init__(self, categories, matching_iou_threshold=0.5):
        """创建使用加权 PASCAL Masks 指标的实例分割评估器。"""
        super(WeightedPascalInstanceSegmentationEvaluator, self).__init__(
            categories,
            matching_iou_threshold=matching_iou_threshold,
            evaluate_corlocs=False,
            metric_prefix="WeightedPascalMasks",
            use_weighted_mean_ap=True,
            evaluate_masks=True,
        )


class OpenImagesDetectionEvaluator(ObjectDetectionEvaluator):
    """使用 Open Images V2 指标评估检测结果的类。

    Open Images V2 引入了 group_of 类型的边界框，该指标会正确处理这些框。
  """

    def __init__(
        self, categories, matching_iou_threshold=0.5, evaluate_corlocs=False
    ):
        """初始化 Open Images V2 检测评估器。

            参数：
              categories: dict 列表，每个 dict 包含以下键：
                'id': 必需，唯一标识该类别的整数 id。
                'name': 必需，类别名称字符串，例如 'cat'、'dog'。
              matching_iou_threshold: 匹配真实标注框和检测框时使用的
                IOU 阈值。
              evaluate_corlocs: 如果为 True，额外评估并返回 CorLoc。

        """
        super(OpenImagesDetectionEvaluator, self).__init__(
            categories,
            matching_iou_threshold,
            evaluate_corlocs,
            metric_prefix="OpenImagesV2",
        )

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """加入单张图像的真实标注信息，用于 Open Images 评估。

            参数：
              image_id: 图像的唯一字符串或整数标识。
              groundtruth_dict: 包含以下内容的字典：
                说明：standard_fields.InputDataFields.groundtruth_boxes: float32 numpy array
                  shape 为 [num_boxes, 4]，包含 `num_boxes` 个真实标注框，
                  格式为绝对图像坐标下的 [ymin, xmin, ymax, xmax]。
                说明：standard_fields.InputDataFields.groundtruth_classes: integer numpy array
                  shape 为 [num_boxes]，包含从 1 开始编号的框类别。
                说明：standard_fields.InputDataFields.groundtruth_group_of: Optional length
                  M 的 numpy boolean array，表示真实标注框是否包含一组实例。

            异常：
              ValueError: 如果同一图像重复加入真实标注。

        """
        if image_id in self._image_ids:
            raise ValueError("id 为 {} 的图像已添加。".format(image_id))

        groundtruth_classes = (
            groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_classes
            ]
            - self._label_id_offset
        )
        # 如果 key 不在 groundtruth_dict 中，或数组为空（除非该图像确实没有
        # 真实标注），则优先使用字典中的值，否则填入 None。
        if standard_fields.InputDataFields.groundtruth_group_of in groundtruth_dict.keys() and (
            groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_group_of
            ].size
            or not groundtruth_classes.size
        ):
            groundtruth_group_of = groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_group_of
            ]
        else:
            groundtruth_group_of = None
            if not len(self._image_ids) % 1000:
                logging.warn(
                    "图像 %s 未指定真实标注 group_of 标记",
                    image_id,
                )
        self._evaluation.add_single_ground_truth_image_info(
            image_id,
            groundtruth_dict[standard_fields.InputDataFields.groundtruth_boxes],
            groundtruth_classes,
            groundtruth_is_difficult_list=None,
            groundtruth_is_group_of_list=groundtruth_group_of,
        )
        self._image_ids.update([image_id])


ObjectDetectionEvalMetrics = collections.namedtuple(
    "ObjectDetectionEvalMetrics",
    [
        "average_precisions",
        "mean_ap",
        "precisions",
        "recalls",
        "corlocs",
        "mean_corloc",
    ],
)


class ObjectDetectionEvaluation(object):
    """PASCAL 目标检测指标的内部实现。"""

    def __init__(
        self,
        num_groundtruth_classes,
        matching_iou_threshold=0.5,
        nms_iou_threshold=1.0,
        nms_max_output_boxes=10000,
        use_weighted_mean_ap=False,
        label_id_offset=0,
    ):
        """创建内部评估状态，并准备累计每张图像的统计量。"""
        if num_groundtruth_classes < 1:
            raise ValueError(
                "评估至少需要 1 个真实标注类别。"
            )

        self.per_image_eval = per_image_evaluation.PerImageEvaluation(
            num_groundtruth_classes=num_groundtruth_classes,
            matching_iou_threshold=matching_iou_threshold,
        )
        self.num_class = num_groundtruth_classes
        self.use_weighted_mean_ap = use_weighted_mean_ap
        self.label_id_offset = label_id_offset

        self.groundtruth_boxes = {}
        self.groundtruth_class_labels = {}
        self.groundtruth_masks = {}
        self.groundtruth_is_difficult_list = {}
        self.groundtruth_is_group_of_list = {}
        self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=int)
        self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)

        self._initialize_detections()

    def _initialize_detections(self):
        """初始化或重置所有检测结果相关的累计统计容器。"""
        self.detection_keys = set()
        self.scores_per_class = [[] for _ in range(self.num_class)]
        self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
        self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
        self.average_precision_per_class = np.empty(self.num_class, dtype=float)
        self.average_precision_per_class.fill(np.nan)
        self.precisions_per_class = []
        self.recalls_per_class = []
        self.corloc_per_class = np.ones(self.num_class, dtype=float)

    def clear_detections(self):
        """只清空检测结果，保留已经加入的真实标注信息。"""
        self._initialize_detections()

    def add_single_ground_truth_image_info(
        self,
        image_key,
        groundtruth_boxes,
        groundtruth_class_labels,
        groundtruth_is_difficult_list=None,
        groundtruth_is_group_of_list=None,
        groundtruth_masks=None,
    ):
        """加入单张图像的真实标注信息，用于评估。

            参数：
              image_key: 图像的唯一字符串或整数标识。
              groundtruth_boxes: float32 numpy array，shape 为 [num_boxes, 4]，
                包含 `num_boxes` 个真实标注框，格式为绝对图像坐标下的
                说明：[ymin, xmin, ymax, xmax]。
              groundtruth_class_labels: integer numpy array，shape 为 [num_boxes]，
                包含从 0 开始编号的框类别。
              groundtruth_is_difficult_list: 长度为 M 的 numpy boolean array，表示
                真实标注框是否为 difficult 实例。为支持没有 difficult 框的
                情况，默认设为 None。
              groundtruth_is_group_of_list: 长度为 M 的 numpy boolean array，表示
                  真实标注框是否为 group-of 框。为支持没有 group-of 框的情况，
                  默认设为 None。
              groundtruth_masks: uint8 numpy array，shape 为 [num_boxes, height, width]，
                包含 `num_boxes` 个真实标注掩码，掩码取值范围为 0 到 1。

        """
        if image_key in self.groundtruth_boxes:
            logging.warn(
                "图像 %s 已经加入真实标注数据库。",
                image_key,
            )
            return

        self.groundtruth_boxes[image_key] = groundtruth_boxes
        self.groundtruth_class_labels[image_key] = groundtruth_class_labels
        self.groundtruth_masks[image_key] = groundtruth_masks
        if groundtruth_is_difficult_list is None:
            num_boxes = groundtruth_boxes.shape[0]
            groundtruth_is_difficult_list = np.zeros(num_boxes, dtype=bool)
        self.groundtruth_is_difficult_list[
            image_key
        ] = groundtruth_is_difficult_list.astype(dtype=bool)
        if groundtruth_is_group_of_list is None:
            num_boxes = groundtruth_boxes.shape[0]
            groundtruth_is_group_of_list = np.zeros(num_boxes, dtype=bool)
        self.groundtruth_is_group_of_list[
            image_key
        ] = groundtruth_is_group_of_list.astype(dtype=bool)

        self._update_ground_truth_statistics(
            groundtruth_class_labels,
            groundtruth_is_difficult_list.astype(dtype=bool),
            groundtruth_is_group_of_list.astype(dtype=bool),
        )

    def add_single_detected_image_info(
        self,
        image_key,
        detected_boxes,
        detected_scores,
        detected_class_labels,
        detected_masks=None,
    ):
        """加入单张图像的检测结果信息，用于评估。

            参数：
              image_key: 图像的唯一字符串或整数标识。
              detected_boxes: float32 numpy array，shape 为 [num_boxes, 4]，
                包含 `num_boxes` 个检测框，格式为绝对图像坐标下的
                说明：[ymin, xmin, ymax, xmax]。
              detected_scores: float32 numpy array，shape 为 [num_boxes]，包含每个框的
                说明：检测分数。
              detected_class_labels: integer numpy array，shape 为 [num_boxes]，包含从 0
                开始编号的检测类别。
              detected_masks: np.uint8 numpy array，shape 为 [num_boxes, height, width]，
                包含 `num_boxes` 个检测掩码，取值范围为 0 到 1。

            异常：
              ValueError: 如果边界框、分数和类别标签的数量不一致。

        """
        if len(detected_boxes) != len(detected_scores) or len(
            detected_boxes
        ) != len(detected_class_labels):
            raise ValueError(
                "detected_boxes、detected_scores 和 "
                "detected_class_labels 的长度必须一致。实际为"
                "[%d, %d, %d]" % len(detected_boxes),
                len(detected_scores),
                len(detected_class_labels),
            )

        if image_key in self.detection_keys:
            logging.warn(
                "图像 %s 已经加入检测结果数据库",
                image_key,
            )
            return

        self.detection_keys.add(image_key)
        if image_key in self.groundtruth_boxes:
            groundtruth_boxes = self.groundtruth_boxes[image_key]
            groundtruth_class_labels = self.groundtruth_class_labels[image_key]
            # 这里弹出掩码而不是只查询，因为不希望把所有掩码都留在内存中，
            # 否则可能导致内存溢出。
            groundtruth_masks = self.groundtruth_masks.pop(image_key)
            groundtruth_is_difficult_list = self.groundtruth_is_difficult_list[
                image_key
            ]
            groundtruth_is_group_of_list = self.groundtruth_is_group_of_list[
                image_key
            ]
        else:
            groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
            groundtruth_class_labels = np.array([], dtype=int)
            if detected_masks is None:
                groundtruth_masks = None
            else:
                groundtruth_masks = np.empty(shape=[0, 1, 1], dtype=float)
            groundtruth_is_difficult_list = np.array([], dtype=bool)
            groundtruth_is_group_of_list = np.array([], dtype=bool)
        (
            scores,
            tp_fp_labels,
        ) = self.per_image_eval.compute_object_detection_metrics(
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

        for i in range(self.num_class):
            if scores[i].shape[0] > 0:
                self.scores_per_class[i].append(scores[i])
                self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])

    def _update_ground_truth_statistics(
        self,
        groundtruth_class_labels,
        groundtruth_is_difficult_list,
        groundtruth_is_group_of_list,
    ):
        """更新真实标注统计信息。

            1. 统计真实标注实例数量时会忽略 difficult 框，这与 Pascal VOC
            devkit 的做法一致。
            2. 计算 CorLoc 相关统计时，会把 difficult 框当作普通框处理。

            参数：
              groundtruth_class_labels: 长度为 M 的 integer numpy array，表示真实标注
                  中 M 个目标实例的类别标签。
              groundtruth_is_difficult_list: 长度为 M 的 boolean numpy array，表示
                  真实标注框是否为 difficult 实例。
              groundtruth_is_group_of_list: 长度为 M 的 boolean numpy array，表示
                  真实标注框是否为 group-of 框。

        """
        for class_index in range(self.num_class):
            num_gt_instances = np.sum(
                groundtruth_class_labels[
                    ~groundtruth_is_difficult_list
                    & ~groundtruth_is_group_of_list
                ]
                == class_index
            )
            self.num_gt_instances_per_class[class_index] += num_gt_instances
            if np.any(groundtruth_class_labels == class_index):
                self.num_gt_imgs_per_class[class_index] += 1

    def evaluate(self):
        """计算当前累计数据上的评估结果。

            返回：
              包含以下字段的 named tuple：
                average_precision: float numpy array，每个类别的平均精度。
                mean_ap: 所有类别的平均精度均值，float scalar。
                precisions: precision 列表，每个 precision 是 float numpy array。
                recalls: recall 列表，每个 recall 是 float numpy array。
                说明：corloc: numpy float array。
                mean_corloc: 每个类别 CorLoc 分数的均值，float scalar。

        """
        if (self.num_gt_instances_per_class == 0).any():
            logging.info(
                "以下类别没有真实标注样本：%s",
                np.squeeze(np.argwhere(self.num_gt_instances_per_class == 0))
                + self.label_id_offset,
            )

        if self.use_weighted_mean_ap:
            all_scores = np.array([], dtype=float)
            all_tp_fp_labels = np.array([], dtype=bool)

        for class_index in range(self.num_class):
            if self.num_gt_instances_per_class[class_index] == 0:
                continue
            if not self.scores_per_class[class_index]:
                scores = np.array([], dtype=float)
                tp_fp_labels = np.array([], dtype=bool)
            else:
                scores = np.concatenate(self.scores_per_class[class_index])
                tp_fp_labels = np.concatenate(
                    self.tp_fp_labels_per_class[class_index]
                )
            if self.use_weighted_mean_ap:
                all_scores = np.append(all_scores, scores)
                all_tp_fp_labels = np.append(all_tp_fp_labels, tp_fp_labels)
            precision, recall = metrics.compute_precision_recall(
                scores,
                tp_fp_labels,
                self.num_gt_instances_per_class[class_index],
            )
            self.precisions_per_class.append(precision)
            self.recalls_per_class.append(recall)
            average_precision = metrics.compute_average_precision(
                precision, recall
            )
            self.average_precision_per_class[class_index] = average_precision

        self.corloc_per_class = metrics.compute_cor_loc(
            self.num_gt_imgs_per_class,
            self.num_images_correctly_detected_per_class,
        )

        if self.use_weighted_mean_ap:
            num_gt_instances = np.sum(self.num_gt_instances_per_class)
            precision, recall = metrics.compute_precision_recall(
                all_scores, all_tp_fp_labels, num_gt_instances
            )
            mean_ap = metrics.compute_average_precision(precision, recall)
        else:
            mean_ap = np.nanmean(self.average_precision_per_class)
        mean_corloc = np.nanmean(self.corloc_per_class)
        return ObjectDetectionEvalMetrics(
            self.average_precision_per_class,
            mean_ap,
            self.precisions_per_class,
            self.recalls_per_class,
            self.corloc_per_class,
            mean_corloc,
        )
