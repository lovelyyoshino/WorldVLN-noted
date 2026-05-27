# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import av


def get_video_container(path_to_vid, multi_thread_decode=False, backend="pyav"):
    """
    给定视频路径，返回对应的视频容器。
    参数：
        path_to_vid (str): 视频路径。
        multi_thread_decode (bool): 若为 True，则启用多线程解码。
        backend (str): 解码后端，可选 `pyav` 和 `torchvision`，
            默认 `pyav`。
    返回：
        container (container): 视频容器。
    """
    if backend == "torchvision":
        with open(path_to_vid, "rb") as fp:
            container = fp.read()
        return container
    elif backend == "pyav":
        # 中文说明：try:
        container = av.open(path_to_vid)
        if multi_thread_decode:
            # 为解码启用多线程。
            container.streams.video[0].thread_type = "AUTO"
        # 保留的上游调试/兼容代码：except:
        # 中文说明：container = None
        return container
    else:
        raise NotImplementedError("未知解码后端 {}".format(backend))
