# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""checkpoint 保存、加载和参数名转换工具。"""

import copy
import numpy as np
import os
import pickle
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager

import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
from timesformer.utils.c2_model_loading import get_name_convert_func
import torch.nn.functional as F

logger = logging.get_logger(__name__)


def make_checkpoint_dir(path_to_job):
    """创建当前训练任务的 checkpoints 目录。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints")
    # 只让主进程创建 checkpoint 目录。
    if du.is_master_proc() and not PathManager.exists(checkpoint_dir):
        try:
            PathManager.mkdirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def get_checkpoint_dir(path_to_job):
    """返回当前训练任务保存 checkpoint 的目录。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    return os.path.join(path_to_job, "checkpoints")


def get_path_to_checkpoint(path_to_job, epoch):
    """根据 epoch 生成 checkpoint文件路径。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job), name)


def get_last_checkpoint(path_to_job):
    """查找 checkpoint 目录中最新的 checkpoint文件。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """

    d = get_checkpoint_dir(path_to_job)
    names = PathManager.ls(d) if PathManager.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "在 '{}' 中没有找到 checkpoint。".format(d)
    # 按 epoch 对 checkpoint 排序。
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint(path_to_job):
    """判断目录中是否已经存在 checkpoint。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    d = get_checkpoint_dir(path_to_job)
    files = PathManager.ls(d) if PathManager.exists(d) else []
    return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cfg, cur_epoch, multigrid_schedule=None):
    """判断当前 epoch 是否应该保存 checkpoint。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
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

    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0


def save_checkpoint(path_to_job, model, optimizer, epoch, cfg):
    """保存模型、优化器、epoch 和配置到 checkpoint。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    # 只允许主进程保存 checkpoint。
    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        return
    # 确保 checkpoint 目录存在。
    PathManager.mkdirs(get_checkpoint_dir(path_to_job))
    # 多卡训练时去掉 DDP 外层包装。
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    normalized_sd = sub_to_normal_bn(sd)

    # 记录训练状态。
    checkpoint = {
        "epoch": epoch,
        "model_state": normalized_sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # 写出 checkpoint文件。
    path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1)
    with PathManager.open(path_to_checkpoint, "wb") as f:
        torch.save(checkpoint, f)
    return path_to_checkpoint


def inflate_weight(state_dict_2d, state_dict_3d):
    """把 2D 卷积权重扩展成 3D 卷积权重，用于加载预训练模型。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    state_dict_inflated = OrderedDict()
    # 保留的上游调试/兼容代码：print(state_dict_2d.keys())
    # 保留的上游调试/兼容代码：print('----')
    # 保留的上游调试/兼容代码：print(state_dict_3d.keys())
    for k, v2d in state_dict_2d.items():
        assert k in state_dict_3d.keys()
        v3d = state_dict_3d[k]
        # 把 2D 卷积权重扩展为 3D 卷积权重。
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            logger.info(
                "对 {} 做 inflation：{} -> {}: {}".format(k, v2d.shape, k, v3d.shape)
            )
            # 维度需要匹配。
            try:
               assert v2d.shape[-2:] == v3d.shape[-2:]
               assert v2d.shape[:2] == v3d.shape[:2]
               v3d = (
                   v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
               )
            except: # 中文说明：这里兼容输入通道数不完全一致的 checkpoint。
               temp = (
                   v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
               )
               v3d = torch.zeros(v3d.shape)
               v3d[:,:v2d.shape[1],:,:,:] = temp
            ####################

        elif v2d.shape == v3d.shape:
            v3d = v2d
        else:
            logger.info(
                "unexpected 形状不匹配 {}: {} -|> {}: {}".format(
                    k, v2d.shape, k, v3d.shape
                )
            )
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated


def load_checkpoint(
    path_to_checkpoint,
    model,
    data_parallel=True,
    optimizer=None,
    inflation=False,
    convert_from_caffe2=False,
    epoch_reset=False,
    clear_name_pattern=(),
):
    """从checkpoint文件恢复模型，并可选择恢复优化器状态。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    assert PathManager.exists(
        path_to_checkpoint
    ), "没有找到 checkpoint '{}'".format(path_to_checkpoint)
    logger.info("正在从 {} 加载网络权重。".format(path_to_checkpoint))

    # 多卡训练时，模型外面可能包着 DDP，这里先取出真正的模型。
    try:
      ms = model.module if data_parallel else model
    except:
      ms = model

    if convert_from_caffe2:
        with PathManager.open(path_to_checkpoint, "rb") as f:
            caffe2_checkpoint = pickle.load(f, encoding="latin1")
        state_dict = OrderedDict()
        name_convert_func = get_name_convert_func()
        for key in caffe2_checkpoint["blobs"].keys():
            converted_key = name_convert_func(key)
            converted_key = c2_normal_to_sub_bn(converted_key, ms.state_dict())
            if converted_key in ms.state_dict():
                c2_blob_shape = caffe2_checkpoint["blobs"][key].shape
                model_blob_shape = ms.state_dict()[converted_key].shape

                # 如果维度数不同，就补 1 维；常见于把 linear 参数转成 conv 参数。
                if len(c2_blob_shape) < len(model_blob_shape):
                    c2_blob_shape += (1,) * (
                        len(model_blob_shape) - len(c2_blob_shape)
                    )
                    caffe2_checkpoint["blobs"][key] = np.reshape(
                        caffe2_checkpoint["blobs"][key], c2_blob_shape
                    )
                # 把 BN stats 复制到 Sub-BN 对应的通道上。
                if (
                    len(model_blob_shape) == 1
                    and len(c2_blob_shape) == 1
                    and model_blob_shape[0] > c2_blob_shape[0]
                    and model_blob_shape[0] % c2_blob_shape[0] == 0
                ):
                    caffe2_checkpoint["blobs"][key] = np.concatenate(
                        [caffe2_checkpoint["blobs"][key]]
                        * (model_blob_shape[0] // c2_blob_shape[0])
                    )
                    c2_blob_shape = caffe2_checkpoint["blobs"][key].shape

                if c2_blob_shape == tuple(model_blob_shape):
                    state_dict[converted_key] = torch.tensor(
                        caffe2_checkpoint["blobs"][key]
                    ).clone()
                    logger.info(
                        "{}: {} => {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
                else:
                    logger.warn(
                        "!! {}: {} 和 {}: {} 的形状对不上".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
            else:
                if not any(
                    prefix in key for prefix in ["momentum", "lr", "model_iter"]
                ):
                    logger.warn(
                        "!! {} 无法转换，转换后得到 {}".format(
                            key, converted_key
                        )
                    )
        diff = set(ms.state_dict()) - set(state_dict)
        diff = {d for d in diff if "num_batches_tracked" not in d}
        if len(diff) > 0:
            logger.warn("这些权重没有被加载 {}".format(diff))
        ms.load_state_dict(state_dict, strict=False)
        epoch = -1
    else:
        # 先把 checkpoint 加载到 CPU，避免 GPU 显存突然升高。
        with PathManager.open(path_to_checkpoint, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        try:
# 保留的上游调试/兼容代码：if True:
            model_state_dict_3d = (
                model.module.state_dict() if data_parallel else model.state_dict()
            )
            checkpoint["model_state"] = normal_to_sub_bn(
                checkpoint["model_state"], model_state_dict_3d
            )
        except:

            model_state_dict_3d = model.state_dict()
            checkpoint["model_state"] = normal_to_sub_bn(
                checkpoint["model_state"], model_state_dict_3d
            )

# 保留的上游调试/兼容代码：except: ####### checkpoint from DEIT
# 保留的上游调试/兼容代码：print(checkpoint.keys())
# 中文说明：model_state_dict_3d = model.state_dict()
# 中文说明：checkpoint["model_state"] = normal_to_sub_bn(
# 中文说明：#                checkpoint["model"], model_state_dict_3d
# 中文说明：checkpoint, model_state_dict_3d
#            )
# 中文说明：keys = checkpoint['model_state'].keys()
# 中文说明：checkpoint['new_model_state'] = {}
# 保留的上游调试/兼容代码：for key in keys:
# 中文说明：new_key = 'model.'+key
# 中文说明：checkpoint['new_model_state'][new_key] = checkpoint['model_state'][key]
# 中文说明：checkpoint['model_state'] = checkpoint['new_model_state']
# 中文说明：del checkpoint['new_model_state']
#
#        ############

        if inflation:
            # 尝试对模型权重做 inflation。
            inflated_model_dict = inflate_weight(
                checkpoint["model_state"], model_state_dict_3d
            )
            ms.load_state_dict(inflated_model_dict, strict=False)
        else:
            if clear_name_pattern:
                for item in clear_name_pattern:
                    model_state_dict_new = OrderedDict()
                    for k in checkpoint["model_state"]:
                        if item in k:
                            k_re = k.replace(item, "")
                            model_state_dict_new[k_re] = checkpoint[
                                "model_state"
                            ][k]
                            logger.info("重命名权重键：{} -> {}".format(k, k_re))
                        else:
                            model_state_dict_new[k] = checkpoint["model_state"][
                                k
                            ]
                    checkpoint["model_state"] = model_state_dict_new

            pre_train_dict = checkpoint["model_state"]
            model_dict = ms.state_dict()
            ############
            if 'model.time_embed' in pre_train_dict:
                k = 'model.time_embed'
                v = pre_train_dict[k]
                v = v[0,:,:].unsqueeze(0).transpose(1,2)
                new_v = F.interpolate(v, size=(model_dict[k].size(1)), mode='nearest')
                pre_train_dict[k] = new_v.transpose(1,2)
            ###################

            # 只挑出和当前模型同名、同形状的 pre-trained 权重。
            pre_train_dict_match = {
                k: v
                for k, v in pre_train_dict.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }
# 保留的上游调试/兼容代码：print(pre_train_dict.keys())
# 保留的上游调试/兼容代码：print('-------------')
# 保留的上游调试/兼容代码：print(model_dict.keys())
# 保留的上游调试/兼容代码：print(pre_train_dict_match)
# 保留的上游调试/兼容代码：print(xy)
            # 这些层在 pre-trained 权重里没有可直接加载的匹配项。
            not_load_layers = [
                k
                for k in model_dict.keys()
                if k not in pre_train_dict_match.keys()
            ]
            # 记录没有从 pre-trained 权重中加载到的层，方便排查。
            if not_load_layers:
                for k in not_load_layers:
                    logger.info("网络权重 {} 没有加载。".format(k))
            # 加载已经匹配上的 pre-trained 权重。
            ms.load_state_dict(pre_train_dict_match, strict=False)
            epoch = -1

            # 如果是继续训练，就恢复 optimizer 状态；微调时通常不会这样做。
        if "epoch" in checkpoint.keys() and not epoch_reset:
            epoch = checkpoint["epoch"]
            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            epoch = -1
    return epoch


def sub_to_normal_bn(sd):
    """把 Sub-BN 参数转换成普通 BN 参数名。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    new_sd = copy.deepcopy(sd)
    modifications = [
        ("bn.bn.running_mean", "bn.running_mean"),
        ("bn.bn.running_var", "bn.running_var"),
        ("bn.split_bn.num_batches_tracked", "bn.num_batches_tracked"),
    ]
    to_remove = ["bn.bn.", ".split_bn."]
    for key in sd:
        for before, after in modifications:
            if key.endswith(before):
                new_key = key.split(before)[0] + after
                new_sd[new_key] = new_sd.pop(key)

        for rm in to_remove:
            if rm in key and key in new_sd:
                del new_sd[key]

    for key in new_sd:
        if key.endswith("bn.weight") or key.endswith("bn.bias"):
            if len(new_sd[key].size()) == 4:
                assert all(d == 1 for d in new_sd[key].size()[1:])
                new_sd[key] = new_sd[key][:, 0, 0, 0]

    return new_sd


def c2_normal_to_sub_bn(key, model_keys):
    """把 Caffe2 普通 BN 参数转换成 Sub-BN 参数名。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    if "bn.running_" in key:
        if key in model_keys:
            return key

        new_key = key.replace("bn.running_", "bn.split_bn.running_")
        if new_key in model_keys:
            return new_key
    else:
        return key


def normal_to_sub_bn(checkpoint_sd, model_sd):
    """把普通 BN 参数转换成 Sub-BN 参数名。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    for key in model_sd:
        if key not in checkpoint_sd:
            if "bn.split_bn." in key:
                load_key = key.replace("bn.split_bn.", "bn.")
                bn_key = key.replace("bn.split_bn.", "bn.bn.")
                checkpoint_sd[key] = checkpoint_sd.pop(load_key)
                checkpoint_sd[bn_key] = checkpoint_sd[key]

    for key in model_sd:
        if key in checkpoint_sd:
            model_blob_shape = model_sd[key].shape
            c2_blob_shape = checkpoint_sd[key].shape

            if (
                len(model_blob_shape) == 1
                and len(c2_blob_shape) == 1
                and model_blob_shape[0] > c2_blob_shape[0]
                and model_blob_shape[0] % c2_blob_shape[0] == 0
            ):
                before_shape = checkpoint_sd[key].shape
                checkpoint_sd[key] = torch.cat(
                    [checkpoint_sd[key]]
                    * (model_blob_shape[0] // c2_blob_shape[0])
                )
                logger.info(
                    "{} {} -> {}".format(
                        key, before_shape, checkpoint_sd[key].shape
                    )
                )
    return checkpoint_sd


def load_test_checkpoint(cfg, model):
    """测试阶段加载 checkpoint。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    # 测试时优先加载用户指定的 checkpoint。
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        # 如果当前 checkpoint 目录没有合适文件，就使用
        # TEST.CHECKPOINT_FILE_PATH 指向的 checkpoint 进行测试。
        load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    elif has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # 如果 TEST.CHECKPOINT_FILE_PATH 和当前 checkpoint 目录都没有可用文件，
        # 就尝试使用 TRAIN.CHECKPOINT_FILE_PATH 指向的 checkpoint 进行测试。
        load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        logger.info(
            "没有可用的 checkpoint 加载方式；将使用随机初始化，仅用于调试。"
        )


def load_train_checkpoint(cfg, model, optimizer):
    """训练阶段加载或恢复 checkpoint。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
    """
    if cfg.TRAIN.AUTO_RESUME and has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        logger.info("从最新 checkpoint 加载：{}。".format(last_checkpoint))
        checkpoint_epoch = load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("从指定 checkpoint文件加载。")
        checkpoint_epoch = load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    return start_epoch
