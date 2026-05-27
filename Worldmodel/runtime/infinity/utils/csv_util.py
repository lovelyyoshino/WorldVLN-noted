# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import csv

def write_dicts2csv_file(input_dict_list, csv_filename):
    """中文说明：`write_dicts2csv_file` 实现CSV 读写工具中的 `write_dicts2csv_file` 步骤，供训练、推理或调试流程复用。

    新手提示：用于把实验指标或任务列表落盘，重点看字段名是否与调用方一致。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    os.makedirs(osp.dirname(csv_filename), exist_ok=True)
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = input_dict_list[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(input_dict_list)
    print(f'"{csv_filename}" 已写入。')

def load_csv_as_dicts(csv_filename):
    """中文说明：`load_csv_as_dicts` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：用于把实验指标或任务列表落盘，重点看字段名是否与调用方一致。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    with open(csv_filename, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)
