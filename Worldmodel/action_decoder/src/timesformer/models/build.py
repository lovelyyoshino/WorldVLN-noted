# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""模型构建相关函数。

中文导读：
    本模块从上游 SlowFast 项目移植而来，提供两个东西：
        - ``MODEL_REGISTRY``：``Registry`` 实例，``vit.py`` / ``video_model_builder.py``
          中的模型类用 ``@MODEL_REGISTRY.register()`` 装饰器注册到名字 -> 类的映射。
        - ``build_model(cfg)``：根据 ``cfg.MODEL.MODEL_NAME`` 在注册表中查到类并实例化，
          可选地把模型移到当前 GPU 并包装 ``DistributedDataParallel``。

    在没有 ``fvcore`` 时使用本文件内置的最小 ``Registry`` 兼容实现，保证只需要
    ``VisionTransformer`` 时也能成功 import。WorldVLN 训练流水线一般直接调用
    ``Worldmodel/action_decoder/src/build_model.build_model(args, model_params)``，
    并不会走这里的注册表路径，但 ``vit_base_patch16_224`` 等装饰器仍依赖此文件。
"""

import torch

try:
    from fvcore.common.registry import Registry  # type: ignore
except ModuleNotFoundError:
    class Registry:  # 最小 fallback，满足 MODEL_REGISTRY.register/get 即可。
        """在没有 fvcore 时使用的轻量注册表。"""

        def __init__(self, name: str):
            """记录注册表名称，并创建名字到对象的映射。"""
            self._name = str(name)
            self._obj_map = {}

        def register(self):
            """返回一个装饰器，用于把类或函数按 ``__name__`` 注册进来。"""

            def _decorator(obj):
                """把传入对象保存到注册表，并原样返回该对象。"""
                self._obj_map[obj.__name__] = obj
                return obj

            return _decorator

        def get(self, name: str):
            """按名称取出已注册对象；名称不存在时抛出 ``KeyError``。"""
            if name not in self._obj_map:
                raise KeyError(f"{name} 未注册到 {self._name}")
            return self._obj_map[name]

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
