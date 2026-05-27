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

"""面向 Numpy BoxList 的边界框列表操作。

支持的典型框操作包括：
  * 面积：计算边界框面积
  * IOU：计算两两交并比分数
"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import numpy as np

from . import np_box_list, np_box_ops


class SortOrder(object):
    """排序顺序的枚举类。

      属性：
        ascend: 升序。
        descend: 降序。

    """

    ASCEND = 1
    DESCEND = 2


def area(boxlist):
    """计算 BoxList 中每个边界框的面积。

      参数：
        boxlist: 保存 N 个框的 BoxList。

      返回：
        shape 为 [N*1] 的 numpy array，表示每个框的面积。

    """
    y_min, x_min, y_max, x_max = boxlist.get_coordinates()
    return (y_max - y_min) * (x_max - x_min)


def intersection(boxlist1, boxlist2):
    """计算两个 BoxList 之间两两边界框的交集面积。

      参数：
        boxlist1: 保存 N 个框的 BoxList。
        boxlist2: 保存 M 个框的 BoxList。

      返回：
        shape 为 [N*M] 的 numpy array，表示两两交集面积。

    """
    return np_box_ops.intersection(boxlist1.get(), boxlist2.get())


def iou(boxlist1, boxlist2):
    """计算两个边界框集合之间两两交并比。

      参数：
        boxlist1: 保存 N 个框的 BoxList。
        boxlist2: 保存 M 个框的 BoxList。

      返回：
        shape 为 [N, M] 的 numpy array，表示两两 iou 分数。

    """
    return np_box_ops.iou(boxlist1.get(), boxlist2.get())


def ioa(boxlist1, boxlist2):
    """计算两个边界框集合之间两两交集面积占比。

      两个框 box1 和 box2 的交集面积占比 (ioa) 定义为它们的交集面积
      除以 box2 的面积。注意 ioa 不是对称的，即 IOA(box1, box2) !=
      说明：IOA(box2, box1)。

      参数：
        boxlist1: 保存 N 个框的 BoxList。
        boxlist2: 保存 M 个框的 BoxList。

      返回：
        shape 为 [N, M] 的 numpy array，表示两两 ioa 分数。

    """
    return np_box_ops.ioa(boxlist1.get(), boxlist2.get())


def gather(boxlist, indices, fields=None):
    """按照 indices 从 BoxList 中取出部分框，并返回新的 BoxList。

      默认情况下，gather 会返回输入索引对应的框，同时带上 boxlist 中保存的所有
      附加字段（沿第一维索引）。也可以通过 fields 只选择部分字段。

      参数：
        boxlist: 保存 N 个框的 BoxList。
        indices: dtype 为 int_ 的 1-D numpy array。
        fields: 可选字段列表。如果为 None（默认），会收集所有字段；传空列表时
            只收集框坐标。

      返回：
        subboxlist: 与 indices 指定子集对应的新 BoxList。

      异常：
        ValueError: 如果指定字段不在 boxlist 中，或 indices 类型不是 int_。

    """
    if indices.size:
        if np.amax(indices) >= boxlist.num_boxes() or np.amin(indices) < 0:
            raise ValueError("indices 超出有效范围。")
    subboxlist = np_box_list.BoxList(boxlist.get()[indices, :])
    if fields is None:
        fields = boxlist.get_extra_fields()
    for field in fields:
        extra_field_data = boxlist.get_field(field)
        subboxlist.add_field(field, extra_field_data[indices, ...])
    return subboxlist


def sort_by_field(boxlist, field, order=SortOrder.DESCEND):
    """根据某个标量字段对框及其关联字段排序。

      常见用法是按照 scores 从高到低重新排列检测框。

      参数：
        boxlist: 保存 N 个框的 BoxList。
        field: 用来排序并重排 BoxList 的字段。
        order: 可选，'descend' 或 'ascend'。默认是 descend。

      返回：
        sorted_boxlist: 按指定顺序排序后的 BoxList。

      异常：
        ValueError: 如果指定字段不存在，或字段不是一维。
        ValueError: 如果 order 既不是 descend 也不是 ascend。

    """
    if not boxlist.has_field(field):
        raise ValueError("字段 " + field + " 不存在。")
    if len(boxlist.get_field(field).shape) != 1:
        raise ValueError("字段 " + field + " 必须是一维。")
    if order != SortOrder.DESCEND and order != SortOrder.ASCEND:
        raise ValueError("排序顺序非法。")

    field_to_sort = boxlist.get_field(field)
    sorted_indices = np.argsort(field_to_sort)
    if order == SortOrder.DESCEND:
        sorted_indices = sorted_indices[::-1]
    return gather(boxlist, sorted_indices)


def non_max_suppression(
    boxlist, max_output_size=10000, iou_threshold=1.0, score_threshold=-10.0
):
    """非极大值抑制（Non maximum suppression, NMS）。

      这个操作会贪心地选择一部分检测框，并删除那些与已选框有较高 IOU
      (intersection over union, 大于阈值) 重叠的框。每轮都会从当前可选框中
      选择分数最高的检测框。

      参数：
        boxlist: 保存 N 个框的 BoxList，必须包含表示检测分数的 'scores' 字段。
          这里假设所有分数属于同一个类别。
        max_output_size: 最多保留的框数量。
        iou_threshold: intersection over union 阈值。
        score_threshold: 最小分数阈值，低于该值的框会被移除。默认值 -10 很低，
                         基本会放过所有框，除非用户显式设置其他阈值。

      返回：
        保存 M 个框的 BoxList，其中 M <= max_output_size。

      异常：
        ValueError: 如果 'scores' 字段不存在。
        ValueError: 如果阈值不在 [0, 1]。
        ValueError: 如果 max_output_size < 0。

    """
    if not boxlist.has_field("scores"):
        raise ValueError("字段 scores 不存在。")
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError("IOU threshold 必须位于 [0, 1]。")
    if max_output_size < 0:
        raise ValueError("max_output_size 必须大于等于 0。")

    boxlist = filter_scores_greater_than(boxlist, score_threshold)
    if boxlist.num_boxes() == 0:
        return boxlist

    boxlist = sort_by_field(boxlist, "scores")

    # 如果 NMS 被关闭，就避免继续做多余计算。
    if iou_threshold == 1.0:
        if boxlist.num_boxes() > max_output_size:
            selected_indices = np.arange(max_output_size)
            return gather(boxlist, selected_indices)
        else:
            return boxlist

    boxes = boxlist.get()
    num_boxes = boxlist.num_boxes()
    # is_index_valid 只会对仍然有效的候选框保持 True。
    is_index_valid = np.full(num_boxes, 1, dtype=bool)
    selected_indices = []
    num_output = 0
    for i in range(num_boxes):
        if num_output < max_output_size:
            if is_index_valid[i]:
                num_output += 1
                selected_indices.append(i)
                is_index_valid[i] = False
                valid_indices = np.where(is_index_valid)[0]
                if valid_indices.size == 0:
                    break

                intersect_over_union = np_box_ops.iou(
                    np.expand_dims(boxes[i, :], axis=0), boxes[valid_indices, :]
                )
                intersect_over_union = np.squeeze(intersect_over_union, axis=0)
                is_index_valid[valid_indices] = np.logical_and(
                    is_index_valid[valid_indices],
                    intersect_over_union <= iou_threshold,
                )
    return gather(boxlist, np.array(selected_indices))


def multi_class_non_max_suppression(
    boxlist, score_thresh, iou_thresh, max_output_size
):
    """多类别版本的非极大值抑制。

      这个操作会贪心地选择一部分检测框，并删除那些与已选框有较高 IOU
      (intersection over union, 大于阈值) 重叠的框。它会对输入 box_list 的
      scores 字段中提供的每个类别独立运行，并在 NMS 前先删掉低于分数阈值的框。

      参数：
        boxlist: 保存 N 个框的 BoxList，必须包含表示检测分数的 'scores' 字段。
          scores 可以是一维（单类别），也可以是二维；二维时假设 shape 为
          [num_boxes, num_classes]。还假设 rank 和 scores.shape[1] 都是已知的
          （也就是类别数固定）。
        score_thresh: 分数标量阈值，低分框会被移除。
        iou_thresh: IOU 标量阈值，与已选框 IOU 过高的框会被移除。
        max_output_size: 每个类别最多保留的框数量。

      返回：
        保存 M 个框的 BoxList，其中一维 scores 字段表示每个框的分数并按降序排列，
          一维 classes 字段表示每个框的类别标签。

      异常：
        ValueError: 如果 iou_thresh 不在 [0, 1]，或输入 boxlist 没有合法的
          scores 字段。

    """
    if not 0 <= iou_thresh <= 1.0:
        raise ValueError("thresh 必须位于 0 到 1 之间。")
    if not isinstance(boxlist, np_box_list.BoxList):
        raise ValueError("boxlist 必须是 BoxList。")
    if not boxlist.has_field("scores"):
        raise ValueError("输入 boxlist 必须包含 'scores' 字段。")
    scores = boxlist.get_field("scores")
    if len(scores.shape) == 1:
        scores = np.reshape(scores, [-1, 1])
    elif len(scores.shape) == 2:
        if scores.shape[1] is None:
            raise ValueError(
                "scores 字段必须静态定义第二维。"
            )
    else:
        raise ValueError("scores 字段必须是 1 维或 2 维。")
    num_boxes = boxlist.num_boxes()
    num_scores = scores.shape[0]
    num_classes = scores.shape[1]

    if num_boxes != num_scores:
        raise ValueError("scores 字段长度错误：实际长度与期望长度不一致。")

    selected_boxes_list = []
    for class_idx in range(num_classes):
        boxlist_and_class_scores = np_box_list.BoxList(boxlist.get())
        class_scores = np.reshape(scores[0:num_scores, class_idx], [-1])
        boxlist_and_class_scores.add_field("scores", class_scores)
        boxlist_filt = filter_scores_greater_than(
            boxlist_and_class_scores, score_thresh
        )
        nms_result = non_max_suppression(
            boxlist_filt,
            max_output_size=max_output_size,
            iou_threshold=iou_thresh,
            score_threshold=score_thresh,
        )
        nms_result.add_field(
            "classes", np.zeros_like(nms_result.get_field("scores")) + class_idx
        )
        selected_boxes_list.append(nms_result)
    selected_boxes = concatenate(selected_boxes_list)
    sorted_boxes = sort_by_field(selected_boxes, "scores")
    return sorted_boxes


def scale(boxlist, y_scale, x_scale):
    """分别在 y 和 x 维度上缩放边界框坐标。

      参数：
        boxlist: 保存 N 个框的 BoxList。
        y_scale: float，y 方向缩放比例。
        x_scale: float，x 方向缩放比例。

      返回：
        boxlist: 保存 N 个缩放后框的 BoxList。

    """
    y_min, x_min, y_max, x_max = np.array_split(boxlist.get(), 4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = np_box_list.BoxList(
        np.hstack([y_min, x_min, y_max, x_max])
    )

    fields = boxlist.get_extra_fields()
    for field in fields:
        extra_field_data = boxlist.get_field(field)
        scaled_boxlist.add_field(field, extra_field_data)

    return scaled_boxlist


def clip_to_window(boxlist, window):
    """将边界框裁剪到指定 window 内。

      该操作把输入边界框（用角点坐标表示）限制在 window 范围内，并过滤掉与
      window 完全没有重叠的框。

      参数：
        boxlist: 保存 M_in 个框的 BoxList。
        window: shape 为 [4] 的 numpy array，表示用于裁剪的
                说明：[y_min, x_min, y_max, x_max] window。

      返回：
        保存 M_out 个框的 BoxList，其中 M_out <= M_in。

    """
    y_min, x_min, y_max, x_max = np.array_split(boxlist.get(), 4, axis=1)
    win_y_min = window[0]
    win_x_min = window[1]
    win_y_max = window[2]
    win_x_max = window[3]
    y_min_clipped = np.fmax(np.fmin(y_min, win_y_max), win_y_min)
    y_max_clipped = np.fmax(np.fmin(y_max, win_y_max), win_y_min)
    x_min_clipped = np.fmax(np.fmin(x_min, win_x_max), win_x_min)
    x_max_clipped = np.fmax(np.fmin(x_max, win_x_max), win_x_min)
    clipped = np_box_list.BoxList(
        np.hstack([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped])
    )
    clipped = _copy_extra_fields(clipped, boxlist)
    areas = area(clipped)
    nonzero_area_indices = np.reshape(
        np.nonzero(np.greater(areas, 0.0)), [-1]
    ).astype(np.int32)
    return gather(clipped, nonzero_area_indices)


def prune_non_overlapping_boxes(boxlist1, boxlist2, minoverlap=0.0):
    """删除 boxlist1 中与 boxlist2 重叠小于阈值的框。

      对 boxlist1 中的每个框，我们要求它至少与 boxlist2 中一个框的 IOA 大于
      minoverlap；否则就移除该框。

      参数：
        boxlist1: 保存 N 个框的 BoxList。
        boxlist2: 保存 M 个框的 BoxList。
        minoverlap: 认为两个框有重叠所需的最小重叠值。

      返回：
        裁剪后的 boxlist，大小为 [N', 4]。

    """
    intersection_over_area = ioa(boxlist2, boxlist1)  # 形状/映射说明：[M, N] tensor
    intersection_over_area = np.amax(
        intersection_over_area, axis=0
    )  # 形状/映射说明：[N] tensor
    keep_bool = np.greater_equal(intersection_over_area, np.array(minoverlap))
    keep_inds = np.nonzero(keep_bool)[0]
    new_boxlist1 = gather(boxlist1, keep_inds)
    return new_boxlist1


def prune_outside_window(boxlist, window):
    """删除落在给定 window 外部的边界框。

      只要边界框有一部分落在给定 window 外，这个函数就会删除它。另见
      ClipToWindow：它只删除完全在 window 外的框，并会裁剪部分越界的框。

      参数：
        boxlist: 保存 M_in 个框的 BoxList。
        window: size 为 4 的 numpy array，表示 window 的
                说明：[ymin, xmin, ymax, xmax]。

      返回：
        pruned_corners: shape 为 [M_out, 4] 的 tensor，其中 M_out <= M_in。
        valid_indices: shape 为 [M_out] 的 tensor，表示输入 tensor 中有效框的索引。

    """

    y_min, x_min, y_max, x_max = np.array_split(boxlist.get(), 4, axis=1)
    win_y_min = window[0]
    win_x_min = window[1]
    win_y_max = window[2]
    win_x_max = window[3]
    coordinate_violations = np.hstack(
        [
            np.less(y_min, win_y_min),
            np.less(x_min, win_x_min),
            np.greater(y_max, win_y_max),
            np.greater(x_max, win_x_max),
        ]
    )
    valid_indices = np.reshape(
        np.where(np.logical_not(np.max(coordinate_violations, axis=1))), [-1]
    )
    return gather(boxlist, valid_indices), valid_indices


def concatenate(boxlists, fields=None):
    """拼接多个 BoxList。

      该操作把多个输入 BoxList 拼成一个更大的 BoxList。只要字段 tensor 除第一维外
      shape 相同，也会同时拼接这些字段。

      参数：
        boxlists: BoxList 对象列表。
        fields: 可选字段列表，表示也要拼接的字段。默认会包含列表中第一个
          BoxList 的所有字段。

      返回：
        一个 BoxList，其框数量等于 sum([boxlist.num_boxes() for boxlist in BoxList])。

      异常：
        ValueError: 如果 boxlists 不合法（不是列表、为空、或包含非 BoxList 对象），
          或请求的字段没有同时存在于所有 boxlists 中。

    """
    if not isinstance(boxlists, list):
        raise ValueError("boxlists 必须是 list。")
    if not boxlists:
        raise ValueError("boxlists 不能为空。")
    for boxlist in boxlists:
        if not isinstance(boxlist, np_box_list.BoxList):
            raise ValueError(
                "boxlists 中所有元素都必须是 BoxList 对象。"
            )
    concatenated = np_box_list.BoxList(
        np.vstack([boxlist.get() for boxlist in boxlists])
    )
    if fields is None:
        fields = boxlists[0].get_extra_fields()
    for field in fields:
        first_field_shape = boxlists[0].get_field(field).shape
        first_field_shape = first_field_shape[1:]
        for boxlist in boxlists:
            if not boxlist.has_field(field):
                raise ValueError("boxlist 必须包含所有请求字段。")
            field_shape = boxlist.get_field(field).shape
            field_shape = field_shape[1:]
            if field_shape != first_field_shape:
                raise ValueError(
                    "除第 0 维外，字段 %s 在所有 boxlists 中的 shape 必须一致。" % field
                )
        concatenated_field = np.concatenate(
            [boxlist.get_field(field) for boxlist in boxlists], axis=0
        )
        concatenated.add_field(field, concatenated_field)
    return concatenated


def filter_scores_greater_than(boxlist, thresh):
    """只保留分数高于给定阈值的框。

      该操作会保留对应 scores 大于输入阈值的框集合。

      参数：
        boxlist: 保存 N 个框的 BoxList，必须包含表示检测分数的 'scores' 字段。
        thresh: 标量阈值。

      返回：
        保存 M 个框的 BoxList，其中 M <= N。

      异常：
        ValueError: 如果 boxlist 不是 BoxList 对象，或它没有 scores 字段。

    """
    if not isinstance(boxlist, np_box_list.BoxList):
        raise ValueError("boxlist 必须是 BoxList。")
    if not boxlist.has_field("scores"):
        raise ValueError("输入 boxlist 必须包含 'scores' 字段。")
    scores = boxlist.get_field("scores")
    if len(scores.shape) > 2:
        raise ValueError("scores 必须是 1 维或 2 维。")
    if len(scores.shape) == 2 and scores.shape[1] != 1:
        raise ValueError(
            "scores 应为 1 维，或 shape 与 [None, 1] 一致。"
        )
    high_score_indices = np.reshape(
        np.where(np.greater(scores, thresh)), [-1]
    ).astype(np.int32)
    return gather(boxlist, high_score_indices)


def change_coordinate_frame(boxlist, window):
    """把 boxlist 的坐标系转换为相对于 window 的坐标系。

      给定 [ymin, xmin, ymax, xmax] 形式的 window，把 boxlist 中的边界框坐标
      转成相对于该 window 的坐标（例如最小角映射到 (0,0)，最大角映射到 (1,1)）。

      一个常见场景是数据增强：已知真实标注框 (boxlist)，并希望把图像随机
      裁剪到某个 window。此时需要把每个真实标注框的坐标转换到新 window。

      参数：
        boxlist: 保存 N 个框的 BoxList 对象。
        window: size 为 4 的 1-D numpy array。

      返回：
        返回包含 N 个框的 BoxList 对象。

    """
    win_height = window[2] - window[0]
    win_width = window[3] - window[1]
    boxlist_new = scale(
        np_box_list.BoxList(
            boxlist.get() - [window[0], window[1], window[0], window[1]]
        ),
        1.0 / win_height,
        1.0 / win_width,
    )
    _copy_extra_fields(boxlist_new, boxlist)

    return boxlist_new


def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):
    """把 boxlist_to_copy_from 的附加字段复制到 boxlist_to_copy_to。

      参数：
        boxlist_to_copy_to: 接收附加字段的 BoxList。
        boxlist_to_copy_from: 提供附加字段的 BoxList。

      返回：
        带有附加字段的 boxlist_to_copy_to。

    """
    for field in boxlist_to_copy_from.get_extra_fields():
        boxlist_to_copy_to.add_field(
            field, boxlist_to_copy_from.get_field(field)
        )
    return boxlist_to_copy_to


def _update_valid_indices_by_removing_high_iou_boxes(
    selected_indices, is_index_valid, intersect_over_union, threshold
):
    """根据已选框的最大 IOU，把重叠过高的候选框标记为无效。"""
    max_iou = np.max(intersect_over_union[:, selected_indices], axis=1)
    return np.logical_and(is_index_valid, max_iou <= threshold)
