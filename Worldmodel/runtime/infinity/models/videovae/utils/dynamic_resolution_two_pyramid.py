# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import math

import numpy as np

video_frames = 97
vae_stride = 16
compressed_frames = video_frames // 4 + 1

def append_dummy_t(ratio2hws):
    """中文说明：`append_dummy_t` 实现VideoVAE 动态分辨率表工具中的 `append_dummy_t` 步骤，供训练、推理或调试流程复用。

    新手提示：核心关系是 ratio -> (T,H,W) 或 (H,W)，再由 patch size 推出 token/latent 数量。
    关键关系：分辨率表把比例映射为 H/W/T，再换算为 patch 或 latent token 数。
    """
    for key in ratio2hws:
        for i in range(len(ratio2hws[key])):
            h, w = ratio2hws[key][i]
            ratio2hws[key][i] = (1, h, w)
    return ratio2hws

def get_full_ratio2hws(ratio2hws, total_pixels2scales):
    """中文说明：`get_full_ratio2hws` 实现VideoVAE 动态分辨率表工具中的 `get_full_ratio2hws` 步骤，供训练、推理或调试流程复用。

    新手提示：核心关系是 ratio -> (T,H,W) 或 (H,W)，再由 patch size 推出 token/latent 数量。
    关键关系：分辨率表把比例映射为 H/W/T，再换算为 patch 或 latent token 数。
    """
    full_ratio2hws = {}
    for ratio, hws in ratio2hws.items():
        real_ratio = hws[-1][1] / hws[-1][2]
        full_ratio2hws[int(real_ratio*1000)/1000] = hws
        if ratio != 1.000:
            full_ratio2hws[int(1/real_ratio*1000)/1000] = [(item[0], item[2], item[1]) for item in hws]

    dynamic_resolution_h_w = {}
    for ratio in full_ratio2hws:
        dynamic_resolution_h_w[ratio] = {}
        for _, scales_num in total_pixels2scales.items():
            h, w = full_ratio2hws[ratio][scales_num-1][1], full_ratio2hws[ratio][scales_num-1][2]
            # 像素尺寸 = (h * vae_stride, w * vae_stride)
            scales = full_ratio2hws[ratio][:scales_num]
            dynamic_resolution_h_w[ratio][(h, w)] = scales
    return dynamic_resolution_h_w

# 比例到尺寸表 ratio2hws 示例 = {
#     1.000: [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(10,10),(12,12),(16,16),(24,24),(32,32),(48,48),(60,60),(64,64)],
#     1.250: [(1,1),(2,2),(3,3),(4,3),(5,4),(6,5),(7,5),(8,6),(10,8),(15,12),(20,16),(30,24),(35,28),(45,36),(66,52),(70,56)],
#     1.333: [(1,1),(2,2),(3,2),(4,3),(5,4),(6,5),(7,5),(8,6),(12,9),(16,12),(20,15),(28,21),(36,27),(48,36),(68,50),(72,54)],
#     1.500: [(1,1),(2,2),(3,2),(4,3),(5,3),(6,4),(7,4),(8,6),(12,8),(15,10),(21,14),(33,22),(39,26),(48,32),(72,48),(78,52)],
#     1.750: [(1,1),(2,2),(3,3),(4,3),(5,3),(6,4),(7,4),(8,5),(12,7),(14,8),(21,12),(32,18),(42,24),(54,30),(80,45),(84,48)],
#     2.000: [(1,1),(2,2),(3,2),(4,2),(5,3),(6,3),(7,4),(8,4),(12,6),(16,8),(22,11),(38,19),(46,23),(60,30),(82,41),(90,45)],
#     2.500: [(1,1),(2,2),(3,2),(4,2),(5,2),(7,3),(8,3),(10,4),(15,6),(20,8),(25,10),(40,16),(50,20),(65,26),(90,36),(100,40)],
#     3.000: [(1,1),(2,2),(3,2),(4,2),(5,2),(6,2),(8,3),(9,3),(15,5),(21,7),(27,9),(45,15),(54,18),(72,24),(96,32),(111,37)],
# }
# 像素数到尺度表 total_pixels2scales 示例 = {
#     '0.06M': 11,
#     '0.15M': 13,
#     '0.25M': 13,
#     '0.40M': 14,
#     '0.90M': 15,
#     '1M': 16,
# }

def get_ratio2hws_video_v2():
    """中文说明：`get_ratio2hws_video_v2` 读取、采样或保存视频帧序列；重点看 fps、帧号范围和输出维度。

    新手提示：核心关系是 ratio -> (T,H,W) 或 (H,W)，再由 patch size 推出 token/latent 数量。
    关键关系：分辨率表把比例映射为 H/W/T，再换算为 patch 或 latent token 数。
    """
    ratio2hws_video_common_v2 = {}
    for h_div_w in [1, 100/116, 3/4, 2/3, 9/16, 1/2, 2/5, 1/3]:
        scale_schedule = []
        # 48*48 is 480p, 60*60 is 720p
        # 示例遍历：for scale in list(range(1,1+16)) + [20, 24, 30, 40]:
        for scale in [1,2,3,4,5,6,7,8,10,12,16] + [24, 32, 40, 48, 60]:
            area = scale * scale
            pw_float = math.sqrt(area / h_div_w)
            ph_float = pw_float * h_div_w
            ph, pw = int(np.round(ph_float)), int(np.round(pw_float))
            scale_schedule.append((ph, pw))
        ratio2hws_video_common_v2[h_div_w] = scale_schedule
    total_pixels2scales = {
        '0.06M': 11,
        '0.15M': 13,
        '0.40M': 14,
        '0.60M': 15,
        '0.90M': 16,
    }
    return ratio2hws_video_common_v2, total_pixels2scales

ratio2hws, total_pixels2scales = get_ratio2hws_video_v2()
ratio2hws = append_dummy_t(ratio2hws)
dynamic_resolution_h_w = get_full_ratio2hws(ratio2hws, total_pixels2scales)
dynamic_resolution_thw = {}
for ratio in dynamic_resolution_h_w:
    for (h, w) in dynamic_resolution_h_w[ratio]:
        image_scale_schedule = dynamic_resolution_h_w[ratio][(h, w)]
        spatial_time_schedule = []
        spatial_time_schedule.extend(image_scale_schedule)
        firstframe_scalecnt = len(image_scale_schedule)
        # 示例条件：if compressed_frames > 1:
        # 代码/形状说明：scale_schedule = dynamic_resolution_h_w[ratio][pn]['scales']
        # 代码/形状说明：# predefined_t = np.linspace(1, compressed_frames - 1, len(scale_schedule))
        # 代码/形状说明：predefined_t = np.linspace(1, compressed_frames - 1, total_pixels2scales['0.06M']-1).tolist() + [compressed_frames - 1] * (len(scale_schedule)-total_pixels2scales['0.06M']+1)
        # 代码/形状说明：spatial_time_schedule.extend([(min(int(np.round(predefined_t[i])), compressed_frames - 1), h, w) for i, (_, h, w) in enumerate(scale_schedule)])
        dynamic_resolution_thw[(h, w)] = {}
        dynamic_resolution_thw[(h, w)]['scales'] = spatial_time_schedule
        dynamic_resolution_thw[(h, w)]['tower_split_index'] = firstframe_scalecnt

# 调试输出：print(dynamic_resolution_thw)

if __name__ == '__main__':
    ratio2hws_video_common_v2, total_pixels2scales = get_ratio2hws_video_v2()
    for h_div_w in ratio2hws_video_common_v2:
        print(h_div_w, ratio2hws_video_common_v2[h_div_w][10])
