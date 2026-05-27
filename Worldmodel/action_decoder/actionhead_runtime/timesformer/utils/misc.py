# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import logging
import math
import numpy as np
import os
from datetime import datetime
import psutil
import torch
from fvcore.common.file_io import PathManager
from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count
from matplotlib import pyplot as plt
from torch import nn

import timesformer.utils.logging as logging
import timesformer.utils.multiprocessing as mpu
from timesformer.datasets.utils import pack_pathway_output
from timesformer.models.batchnorm_helper import SubBatchNorm3d

logger = logging.get_logger(__name__)


def check_nan_losses(loss):
    """检查 loss 是否为 NaN，及时中断异常训练。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    if math.isnan(loss):
        raise RuntimeError("错误：检测到 NaN loss {}".format(datetime.now()))


def params_count(model, ignore_bn=False):
    """统计模型参数量。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count


def gpu_mem_usage():
    """读取当前 GPU 显存占用。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """读取当前进程或系统的 CPU 内存占用。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def _get_model_analysis_input(cfg, use_train_input):
    """构造模型分析用的虚拟输入张量。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    rgb_dimension = 3
    if use_train_input:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_FRAMES,
            cfg.DATA.TRAIN_CROP_SIZE,
            cfg.DATA.TRAIN_CROP_SIZE,
        )
    else:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_FRAMES,
            cfg.DATA.TEST_CROP_SIZE,
            cfg.DATA.TEST_CROP_SIZE,
        )
    if not cfg.MODEL.ARCH in ['resformer', 'vit']:
       model_inputs = pack_pathway_output(cfg, input_tensors)
       for i in range(len(model_inputs)):
           model_inputs[i] = model_inputs[i].unsqueeze(0)
           if cfg.NUM_GPUS:
               model_inputs[i] = model_inputs[i].cuda(non_blocking=True)

    else:
       model_inputs = input_tensors.cuda(non_blocking=True).unsqueeze(0)

    # 检测模式下按一个 proposal 统计 FLOPs。
    if cfg.DETECTION.ENABLE:
        bbox = torch.tensor([[0, 0, 1.0, 0, 1.0]])
        if cfg.NUM_GPUS:
            bbox = bbox.cuda()
        inputs = (model_inputs, bbox)
    else:
        inputs = (model_inputs,)
    return inputs


def get_model_stats(model, cfg, mode, use_train_input):
    """计算模型 FLOPs、激活量等统计信息。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    assert mode in [
        "flop",
        "activation",
    ], "'{}' 不支持用于 model 分析".format(mode)
    if mode == "flop":
        model_stats_fun = flop_count
    elif mode == "activation":
        model_stats_fun = activation_count

    # 模型分析前切换到 eval 模式。
    # eval 模式可以避免 SyncBatchNorm 卡住。
    model_mode = model.training
    model.eval()
    inputs = _get_model_analysis_input(cfg, use_train_input)
    count_dict, *_ = model_stats_fun(model, inputs)
    count = sum(count_dict.values())
    model.train(model_mode)
    return count


def log_model_info(model, cfg, use_train_input=True):
    """记录模型结构、参数量、显存和计算量信息。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    logger.info("模型：\n{}".format(model))
    logger.info("参数量：{:,}".format(params_count(model)))
    logger.info("GPU 显存：{:,} MB".format(gpu_mem_usage()))
    logger.info(
        "FLOPs：{:,} G".format(
            get_model_stats(model, cfg, "flop", use_train_input)
        )
    )
    logger.info(
        "激活量：{:,} M".format(
            get_model_stats(model, cfg, "activation", use_train_input)
        )
    )
    logger.info("nvidia-smi")
    os.system("nvidia-smi")


def is_eval_epoch(cfg, cur_epoch, multigrid_schedule):
    """判断当前 epoch 是否需要运行验证。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True
    if multigrid_schedule is not None:
        prev_epoch = 0
        for s in multigrid_schedule:
            if cur_epoch < s[-1]:
                period = max(
                    (s[-1] - prev_epoch) // cfg.MULTIGRID.EVAL_FREQ + 1, 1
                )
                return (s[-1] - 1 - cur_epoch) % period == 0
            prev_epoch = s[-1]

    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0


def plot_input(tensor, bboxes=(), texts=(), path="./tmp_vis.png"):
    """把输入视频帧和可选检测框保存成可视化图片。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    tensor = tensor.float()
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    f, ax = plt.subplots(nrows=1, ncols=tensor.shape[0], figsize=(50, 20))
    for i in range(tensor.shape[0]):
        ax[i].axis("off")
        ax[i].imshow(tensor[i].permute(1, 2, 0))
        # 中文说明：ax[1][0].axis('off')
        if bboxes is not None and len(bboxes) > i:
            for box in bboxes[i]:
                x1, y1, x2, y2 = box
                ax[i].vlines(x1, y1, y2, colors="g", linestyles="solid")
                ax[i].vlines(x2, y1, y2, colors="g", linestyles="solid")
                ax[i].hlines(y1, x1, x2, colors="g", linestyles="solid")
                ax[i].hlines(y2, x1, x2, colors="g", linestyles="solid")

        if texts is not None and len(texts) > i:
            ax[i].text(0, 0, texts[i])
    f.savefig(path)


def frozen_bn_stats(model):
    """把 BatchNorm 层切到 eval 模式，冻结统计量。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()


def aggregate_sub_bn_stats(module):
    """递归聚合 Sub-BN 的统计量。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    count = 0
    for child in module.children():
        if isinstance(child, SubBatchNorm3d):
            child.aggregate_stats()
            count += 1
        else:
            count += aggregate_sub_bn_stats(child)
    return count


def launch_job(cfg, init_method, func, daemon=False):
    """按配置启动单卡或多卡训练/评估任务。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg=cfg)


def get_class_names(path, parent_path=None, subset_path=None):
    """读取类别名文件，并可返回父子类别映射。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    try:
        with PathManager.open(path, "r") as f:
            class2idx = json.load(f)
    except Exception as err:
        print("从 {} 加载文件失败，错误为 {}".format(path, err))
        return

    max_key = max(class2idx.values())
    class_names = [None] * (max_key + 1)

    for k, i in class2idx.items():
        class_names[i] = k

    class_parent = None
    if parent_path is not None and parent_path != "":
        try:
            with PathManager.open(parent_path, "r") as f:
                d_parent = json.load(f)
        except EnvironmentError as err:
            print(
                "从 {} 加载文件失败，错误为 {}".format(
                    parent_path, err
                )
            )
            return
        class_parent = {}
        for parent, children in d_parent.items():
            indices = [
                class2idx[c] for c in children if class2idx.get(c) is not None
            ]
            class_parent[parent] = indices

    subset_ids = None
    if subset_path is not None and subset_path != "":
        try:
            with PathManager.open(subset_path, "r") as f:
                subset = f.read().split("\n")
                subset_ids = [
                    class2idx[name]
                    for name in subset
                    if class2idx.get(name) is not None
                ]
        except EnvironmentError as err:
            print(
                "从 {} 加载文件失败，错误为 {}".format(
                    subset_path, err
                )
            )
            return

    return class_names, class_parent, subset_ids
