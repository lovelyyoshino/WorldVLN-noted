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

"""numpy 掩码数组的面积、交集和重叠度工具。"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import numpy as np

EPSILON = 1e-7


def area(masks):
    """计算框或掩码的面积。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if masks.dtype != np.uint8:
        raise ValueError("掩码数组的类型必须是 np.uint8。")
    return np.sum(masks, axis=(1, 2), dtype=np.float32)


def intersection(masks1, masks2):
    """计算两组框或掩码的两两交集面积。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError("两个掩码数组的类型都必须是 np.uint8。")
    n = masks1.shape[0]
    m = masks2.shape[0]
    answer = np.zeros([n, m], dtype=np.float32)
    for i in np.arange(n):
        for j in np.arange(m):
            answer[i, j] = np.sum(
                np.minimum(masks1[i], masks2[j]), dtype=np.float32
            )
    return answer


def iou(masks1, masks2):
    """计算两组框或掩码的交并比。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError("两个掩码数组的类型都必须是 np.uint8。")
    intersect = intersection(masks1, masks2)
    area1 = area(masks1)
    area2 = area(masks2)
    union = (
        np.expand_dims(area1, axis=1)
        + np.expand_dims(area2, axis=0)
        - intersect
    )
    return intersect / np.maximum(union, EPSILON)


def ioa(masks1, masks2):
    """计算两组框或掩码的交集面积占比。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError("两个掩码数组的类型都必须是 np.uint8。")
    intersect = intersection(masks1, masks2)
    areas = np.expand_dims(area(masks2), axis=0)
    return intersect / (areas + EPSILON)
