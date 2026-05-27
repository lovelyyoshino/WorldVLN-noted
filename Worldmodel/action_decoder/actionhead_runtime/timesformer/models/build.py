# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""模型构建相关函数。"""

import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
视频模型注册表。

注册后的对象会以 `obj(cfg)` 的形式被调用。
调用结果应返回一个 `torch.nn.Module` 对象。
"""


def build_model(cfg, gpu_id=None):
    """
        根据配置创建视频模型，并在需要时移动到指定 GPU。

        参数：
            cfg (configs): 包含 backbone 构建超参数的配置。
            gpu_id (Optional[int]): 指定构建模型使用的 GPU 编号。

    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "请求使用的 GPU 数量超过当前可用数量"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "CUDA 不可用。若要在 CPU 上运行，请设置 `NUM_GPUS: 0`。"

    # 构建模型主体。
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # 使用当前进程已经绑定的 GPU。
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # 将模型移动到当前 GPU 设备。
        model = model.cuda(device=cur_device)


    # 多 GPU 时使用多进程分布式数据并行。
    if cfg.NUM_GPUS > 1:
        # 让模型副本在当前设备上运行。
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    return model
