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
"""检测 label map 读取与类别索引工具。"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import logging

# 保留的上游调试/兼容代码：from google.protobuf import text_format
# 保留的上游调试/兼容代码：from google3.third_party.tensorflow_models.object_detection.protos import string_int_label_map_pb2


def _validate_label_map(label_map):
    """检查 label map 条目是否合法，尤其是类别 id 和名称。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    for item in label_map.item:
        if item.id < 1:
            raise ValueError("标签映射 id 应 >= 1。")


def create_category_index(categories):
    """把类别列表转换成按类别 id 索引的字典。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    category_index = {}
    for cat in categories:
        category_index[cat["id"]] = cat
    return category_index


def get_max_label_map_index(label_map):
    """读取 label map 中最大的类别 id。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    return max([item.id for item in label_map.item])


def convert_label_map_to_categories(
    label_map, max_num_classes, use_display_name=True
):
    """把 label map proto 转换成 COCO/检测评估兼容的类别列表。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories.append(
                {
                    "id": class_id + label_id_offset,
                    "name": "category_{}".format(class_id + label_id_offset),
                }
            )
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info(
                "忽略条目 %d，因为它超出了请求的标签范围。",
                item.id,
            )
            continue
        if use_display_name and item.HasField("display_name"):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({"id": item.id, "name": name})
    return categories


def load_labelmap(path):
    """从文件读取 label map proto。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    with open(path, "r") as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    _validate_label_map(label_map)
    return label_map


def get_label_map_dict(label_map_path, use_display_name=False):
    """读取 label map 并生成类别名称到 id 的映射。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    label_map = load_labelmap(label_map_path)
    label_map_dict = {}
    for item in label_map.item:
        if use_display_name:
            label_map_dict[item.display_name] = item.id
        else:
            label_map_dict[item.name] = item.id
    return label_map_dict


def create_category_index_from_labelmap(label_map_path):
    """从 label map 文件直接创建类别索引。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    label_map = load_labelmap(label_map_path)
    max_num_classes = max(item.id for item in label_map.item)
    categories = convert_label_map_to_categories(label_map, max_num_classes)
    return create_category_index(categories)


def create_class_agnostic_category_index():
    """创建类别无关评估时使用的单一目标类别索引。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    return {1: {"id": 1, "name": "object"}}
