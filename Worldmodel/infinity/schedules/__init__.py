# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

def get_encode_decode_func(dynamic_scale_schedule):
    """根据动态尺度调度名称，返回对应的视频编解码和辅助函数。"""
    # 这里做调度分发：不同 schedule 名称会导入不同实现，调用方拿到统一接口。
    if dynamic_scale_schedule == 'infinity_star_interact':
        from infinity.schedules.infinity_star_interact import video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info
    elif 'infinity_elegant' in dynamic_scale_schedule:
        from infinity.schedules.infinity_elegant import video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info
    else:
        raise NotImplementedError(f'dynamic_scale_schedule 暂未实现：{dynamic_scale_schedule}')
    return video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info
