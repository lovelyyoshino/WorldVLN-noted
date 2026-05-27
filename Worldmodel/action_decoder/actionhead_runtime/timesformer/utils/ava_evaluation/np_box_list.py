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

"""基于 numpy 的 BoxList 类和辅助函数。"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import numpy as np


class BoxList(object):
    """边界框集合。

  BoxList 用 numpy array 表示一组边界框，每一行是一个框，
  格式为 [y_min, x_min, y_max, x_max]。这里默认同一个 BoxList 中的
  所有框都来自同一张图像。

  用户也可以额外添加相关字段，例如目标置信度或分类分数。
  """

    def __init__(self, data):
        """构造一个边界框集合。

            参数：
              data: shape 为 [N, 4] 的 numpy array，表示框坐标。

            异常：
              ValueError: 如果边界框数据不是 numpy array。
              ValueError: 如果边界框数据维度不合法。

        """
        if not isinstance(data, np.ndarray):
            raise ValueError("data 必须是 numpy array。")
        if len(data.shape) != 2 or data.shape[1] != 4:
            raise ValueError("边界框数据维度不合法，应为 [N, 4]。")
        if data.dtype != np.float32 and data.dtype != np.float64:
            raise ValueError(
                "边界框数据类型不合法：必须是 float。"
            )
        if not self._is_valid_boxes(data):
            raise ValueError(
                "边界框数据坐标不合法。data 必须是形如 "
                "N*[y_min, x_min, y_max, x_max] 的 numpy array。"
            )
        self.data = {"boxes": data}

    def num_boxes(self):
        """返回集合中保存的边界框数量。"""
        return self.data["boxes"].shape[0]

    def get_extra_fields(self):
        """返回所有非边界框的附加字段名。"""
        return [k for k in self.data.keys() if k != "boxes"]

    def has_field(self, field):
        """判断指定字段是否已经存在于当前 BoxList 中。"""
        return field in self.data

    def add_field(self, field, field_data):
        """向指定字段添加数据。

            参数：
              field: 字符串，表示要访问或保存的相关字段名。
              field_data: shape 为 [N, ...] 的 numpy array，表示该字段对应的数据。

            异常：
              ValueError: 如果字段已存在，或字段数据第一维与框数量不一致。

        """
        if self.has_field(field):
            raise ValueError("字段 " + field + " 已存在。")
        if len(field_data.shape) < 1 or field_data.shape[0] != self.num_boxes():
            raise ValueError("字段数据维度不合法，第一维必须等于框数量。")
        self.data[field] = field_data

    def get(self):
        """便捷地获取边界框坐标。

            返回：
              shape 为 [N, 4] 的 numpy array，表示框的四个角坐标。

        """
        return self.get_field("boxes")

    def get_field(self, field):
        """读取边界框集合中指定字段的数据。

            参数：
              field: 字符串，表示要访问的相关字段名。

            返回：
              numpy array，表示该字段关联的数据。

            异常：
              ValueError: 如果字段不存在。

        """
        if not self.has_field(field):
            raise ValueError("字段 {} 不存在。".format(field))
        return self.data[field]

    def get_coordinates(self):
        """获取所有框的四个角坐标。

            返回：
             由 4 个 1-D numpy array 组成的列表：[y_min, x_min, y_max, x_max]。

        """
        box_coordinates = self.get()
        y_min = box_coordinates[:, 0]
        x_min = box_coordinates[:, 1]
        y_max = box_coordinates[:, 2]
        x_max = box_coordinates[:, 3]
        return [y_min, x_min, y_max, x_max]

    def _is_valid_boxes(self, data):
        """检查数据是否满足 N*[ymin, xmin, ymax, xmax] 的格式。

            参数：
              data: shape 为 [N, 4] 的 numpy array，表示框坐标。

            返回：
              bool，表示每个框是否满足 ymax >= ymin 且 xmax >= xmin。

        """
        if data.shape[0] > 0:
            for i in range(data.shape[0]):
                if data[i, 0] > data[i, 2] or data[i, 1] > data[i, 3]:
                    return False
        return True
