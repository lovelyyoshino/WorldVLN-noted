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

"""带掩码的 numpy BoxList 数据结构。"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import numpy as np

from . import np_box_list


class BoxMaskList(np_box_list.BoxList):
    """带掩码字段的 BoxList 包装类，用于同时管理框和像素级掩码。 小白可以先看 `__init__` 保存了哪些字段，再看其他方法如何读取这些字段。

    数据流提示：类属性通常在初始化时写入，后续方法通过这些属性完成评估、采样或状态转换。
    """

    def __init__(self, box_data, mask_data):
        """初始化当前对象，保存后续方法会复用的配置、字段或评估状态。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
        """
        super(BoxMaskList, self).__init__(box_data)
        if not isinstance(mask_data, np.ndarray):
            raise ValueError("掩码数据必须是 numpy array。")
        if len(mask_data.shape) != 3:
            raise ValueError("掩码数据维度不合法，应为 [N, H, W]。")
        if mask_data.dtype != np.uint8:
            raise ValueError(
                "掩码数据类型不合法：必须是 uint8。"
            )
        if mask_data.shape[0] != box_data.shape[0]:
            raise ValueError(
                "边界框和掩码的数量必须一致。"
            )
        self.data["masks"] = mask_data

    def get_masks(self):
        """`get_masks` 是 TimeSformer/PySlowFast 兼容工具函数。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 DataLoader、模型、评估器或 checkpoint 流程。
        """
        return self.get_field("masks")
