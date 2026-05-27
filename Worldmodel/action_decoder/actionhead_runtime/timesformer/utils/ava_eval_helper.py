# Copyright (c) Facebook, Inc. and its affiliates.
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
##############################################################################
#
# 中文说明：Based on:
# --------------------------------------------------------
# 中文说明：ActivityNet
# Copyright (c) 2015 ActivityNet
# Licensed under The MIT License
# [see https://github.com/activitynet/ActivityNet/blob/master/LICENSE for details]
# --------------------------------------------------------

"""AVA 行为检测评估辅助函数。"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import csv
import logging
import numpy as np
import pprint
import time
from collections import defaultdict
from fvcore.common.file_io import PathManager
import timesformer.utils.distributed as du

from timesformer.utils.ava_evaluation import (
    object_detection_evaluation,
    standard_fields,
)

logger = logging.getLogger(__name__)


def make_image_key(video_id, timestamp):
    """把 video_id 与时间戳拼成 AVA 评估使用的唯一图像键。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    return "%s,%04d" % (video_id, int(timestamp))


def read_csv(csv_file, class_whitelist=None, load_score=False):
    """读取 AVA 格式 CSV，解析检测框、类别和分数。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    with PathManager.open(csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            assert len(row) in [7, 8], "列数错误：" + row
            image_key = make_image_key(row[0], row[1])
            x1, y1, x2, y2 = [float(n) for n in row[2:6]]
            action_id = int(row[6])
            if class_whitelist and action_id not in class_whitelist:
                continue
            score = 1.0
            if load_score:
                score = float(row[7])
            boxes[image_key].append([y1, x1, y2, x2])
            labels[image_key].append(action_id)
            scores[image_key].append(score)
    return boxes, labels, scores


def read_exclusions(exclusions_file):
    """读取 AVA 排除列表，得到不参与评估的时间戳集合。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    excluded = set()
    if exclusions_file:
        with PathManager.open(exclusions_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                assert len(row) == 2, "期望只有 2 列，实际为：" + row
                excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """读取 AVA label map，得到类别 id 与类别名称。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """

    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""
    with PathManager.open(labelmap_file, "r") as f:
        for line in f:
            if line.startswith("  name:"):
                name = line.split('"')[1]
            elif line.startswith("  id:") or line.startswith("  label_id:"):
                class_id = int(line.strip().split(" ")[-1])
                labelmap.append({"id": class_id, "name": name})
                class_ids.add(class_id)
    return labelmap, class_ids


def evaluate_ava_from_files(labelmap, groundtruth, detections, exclusions):
    """从标注文件和预测文件加载数据并运行 AVA 评估。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """

    categories, class_whitelist = read_labelmap(labelmap)
    excluded_keys = read_exclusions(exclusions)
    groundtruth = read_csv(groundtruth, class_whitelist, load_score=False)
    detections = read_csv(detections, class_whitelist, load_score=True)
    run_evaluation(categories, groundtruth, detections, excluded_keys)


def evaluate_ava(
    preds,
    original_boxes,
    metadata,
    excluded_keys,
    class_whitelist,
    categories,
    groundtruth=None,
    video_idx_to_name=None,
    name="latest",
):
    """直接使用 numpy 数组运行 AVA 检测评估。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """

    eval_start = time.time()

    detections = get_ava_eval_data(
        preds,
        original_boxes,
        metadata,
        class_whitelist,
        video_idx_to_name=video_idx_to_name,
    )

    logger.info("正在评估 %d 个唯一 GT 帧。" % len(groundtruth[0]))
    logger.info("正在评估 %d 个唯一检测结果帧" % len(detections[0]))

    write_results(detections, "detections_%s.csv" % name)
    write_results(groundtruth, "groundtruth_%s.csv" % name)

    results = run_evaluation(categories, groundtruth, detections, excluded_keys)

    logger.info("AVA 评估完成，耗时 %f 秒。" % (time.time() - eval_start))
    return results["PascalBoxes_Precision/mAP@0.5IOU"]


def run_evaluation(
    categories, groundtruth, detections, excluded_keys, verbose=True
):
    """AVA 评估主流程，负责组织真实标注、预测结果和评估器。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """

    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories
    )

    boxes, labels, _ = groundtruth

    gt_keys = []
    pred_keys = []

    for image_key in boxes:
        if image_key in excluded_keys:
            logging.info(
                (
                    "在真实标注中发现被排除的时间戳：%s。"
                    "该项将被忽略。"
                ),
                image_key,
            )
            continue
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key,
            {
                standard_fields.InputDataFields.groundtruth_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.InputDataFields.groundtruth_classes: np.array(
                    labels[image_key], dtype=int
                ),
                standard_fields.InputDataFields.groundtruth_difficult: np.zeros(
                    len(boxes[image_key]), dtype=bool
                ),
            },
        )

        gt_keys.append(image_key)

    boxes, labels, scores = detections

    for image_key in boxes:
        if image_key in excluded_keys:
            logging.info(
                (
                    "在检测结果中发现被排除的时间戳：%s。"
                    "该项将被忽略。"
                ),
                image_key,
            )
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key,
            {
                standard_fields.DetectionResultFields.detection_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.DetectionResultFields.detection_classes: np.array(
                    labels[image_key], dtype=int
                ),
                standard_fields.DetectionResultFields.detection_scores: np.array(
                    scores[image_key], dtype=float
                ),
            },
        )

        pred_keys.append(image_key)

    metrics = pascal_evaluator.evaluate()

    if du.is_master_proc():
        pprint.pprint(metrics, indent=2)
    return metrics


def get_ava_eval_data(
    scores,
    boxes,
    metadata,
    class_whitelist,
    verbose=False,
    video_idx_to_name=None,
):
    """把项目内部检测输出转换为官方 AVA 评估器需要的数据结构。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """

    out_scores = defaultdict(list)
    out_labels = defaultdict(list)
    out_boxes = defaultdict(list)
    count = 0
    for i in range(scores.shape[0]):
        video_idx = int(np.round(metadata[i][0]))
        sec = int(np.round(metadata[i][1]))

        video = video_idx_to_name[video_idx]

        key = video + "," + "%04d" % (sec)
        batch_box = boxes[i].tolist()
        # 第一列是 batch 索引。
        batch_box = [batch_box[j] for j in [0, 2, 1, 4, 3]]

        one_scores = scores[i].tolist()
        for cls_idx, score in enumerate(one_scores):
            if cls_idx + 1 in class_whitelist:
                out_scores[key].append(score)
                out_labels[key].append(cls_idx + 1)
                out_boxes[key].append(batch_box[1:])
                count += 1

    return out_boxes, out_labels, out_scores


def write_results(detections, filename):
    """把预测结果写成官方 AVA 提交/评估格式。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    start = time.time()

    boxes, labels, scores = detections
    with PathManager.open(filename, "w") as f:
        for key in boxes.keys():
            for box, label, score in zip(boxes[key], labels[key], scores[key]):
                f.write(
                    "%s,%.03f,%.03f,%.03f,%.03f,%d,%.04f\n"
                    % (key, box[1], box[0], box[3], box[2], label, score)
                )

    logger.info("AVA 结果已写入 %s" % filename)
    logger.info("\t耗时 %d 秒。" % (time.time() - start))
