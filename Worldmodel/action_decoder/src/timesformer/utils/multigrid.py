# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""multigrid 训练日程生成与更新工具。"""

import numpy as np

import timesformer.utils.logging as logging

logger = logging.get_logger(__name__)


class MultigridSchedule(object):
    """管理 multigrid 训练日程，按 epoch 动态调整分辨率和帧数。 小白可以先看 `__init__` 保存了哪些字段，再看其他方法如何读取这些字段。

    数据流提示：类属性通常在初始化时写入，后续方法通过这些属性完成评估、采样或状态转换。
    """

    def init_multigrid(self, cfg):
        """根据 multigrid 配置初始化长短周期训练参数。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
        """
        self.schedule = None
        # 训练过程中可能会修改 cfg.TRAIN.BATCH_SIZE、cfg.DATA.NUM_FRAMES 和
        # cfg.DATA.TRAIN_CROP_SIZE，因此先把原始值保存到 cfg 中作为全局基准。
        cfg.MULTIGRID.DEFAULT_B = cfg.TRAIN.BATCH_SIZE
        cfg.MULTIGRID.DEFAULT_T = cfg.DATA.NUM_FRAMES
        cfg.MULTIGRID.DEFAULT_S = cfg.DATA.TRAIN_CROP_SIZE

        if cfg.MULTIGRID.LONG_CYCLE:
            self.schedule = self.get_long_cycle_schedule(cfg)
            cfg.SOLVER.STEPS = [0] + [s[-1] for s in self.schedule]
            # 微调阶段。
            cfg.SOLVER.STEPS[-1] = (
                cfg.SOLVER.STEPS[-2] + cfg.SOLVER.STEPS[-1]
            ) // 2
            cfg.SOLVER.LRS = [
                cfg.SOLVER.GAMMA ** s[0] * s[1][0] for s in self.schedule
            ]
            # 微调阶段。
            cfg.SOLVER.LRS = cfg.SOLVER.LRS[:-1] + [
                cfg.SOLVER.LRS[-2],
                cfg.SOLVER.LRS[-1],
            ]

            cfg.SOLVER.MAX_EPOCH = self.schedule[-1][-1]

        elif cfg.MULTIGRID.SHORT_CYCLE:
            cfg.SOLVER.STEPS = [
                int(s * cfg.MULTIGRID.EPOCH_FACTOR) for s in cfg.SOLVER.STEPS
            ]
            cfg.SOLVER.MAX_EPOCH = int(
                cfg.SOLVER.MAX_EPOCH * cfg.MULTIGRID.EPOCH_FACTOR
            )
        return cfg

    def update_long_cycle(self, cfg, cur_epoch):
        """在 epoch 前检查并更新 long cycle 的训练形状。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
        """
        base_b, base_t, base_s = get_current_long_cycle_shape(
            self.schedule, cur_epoch
        )
        if base_s != cfg.DATA.TRAIN_CROP_SIZE or base_t != cfg.DATA.NUM_FRAMES:

            cfg.DATA.NUM_FRAMES = base_t
            cfg.DATA.TRAIN_CROP_SIZE = base_s
            cfg.TRAIN.BATCH_SIZE = base_b * cfg.MULTIGRID.DEFAULT_B

            bs_factor = (
                float(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
                / cfg.MULTIGRID.BN_BASE_SIZE
            )

            if bs_factor < 1:
                cfg.BN.NORM_TYPE = "sync_batchnorm"
                cfg.BN.NUM_SYNC_DEVICES = int(1.0 / bs_factor)
            elif bs_factor > 1:
                cfg.BN.NORM_TYPE = "sub_batchnorm"
                cfg.BN.NUM_SPLITS = int(bs_factor)
            else:
                cfg.BN.NORM_TYPE = "batchnorm"

            cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = cfg.DATA.SAMPLING_RATE * (
                cfg.MULTIGRID.DEFAULT_T // cfg.DATA.NUM_FRAMES
            )
            logger.info("Long cycle 更新：")
            logger.info("\tBN.NORM_TYPE: {}".format(cfg.BN.NORM_TYPE))
            if cfg.BN.NORM_TYPE == "sync_batchnorm":
                logger.info(
                    "\tBN.NUM_SYNC_DEVICES: {}".format(cfg.BN.NUM_SYNC_DEVICES)
                )
            elif cfg.BN.NORM_TYPE == "sub_batchnorm":
                logger.info("\tBN.NUM_SPLITS: {}".format(cfg.BN.NUM_SPLITS))
            logger.info("\tTRAIN.BATCH_SIZE: {}".format(cfg.TRAIN.BATCH_SIZE))
            logger.info(
                "\tDATA.NUM_FRAMES x LONG_CYCLE_SAMPLING_RATE: {}x{}".format(
                    cfg.DATA.NUM_FRAMES, cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE
                )
            )
            logger.info(
                "\tDATA.TRAIN_CROP_SIZE: {}".format(cfg.DATA.TRAIN_CROP_SIZE)
            )
            return cfg, True
        else:
            return cfg, False

    def get_long_cycle_schedule(self, cfg):
        """根据 multigrid 超参数生成 long cycle 日程。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

        数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
        """

        steps = cfg.SOLVER.STEPS

        default_size = float(
            cfg.DATA.NUM_FRAMES * cfg.DATA.TRAIN_CROP_SIZE ** 2
        )
        default_iters = steps[-1]

        # 获取每个 long cycle shape 对应的形状和平均 batch size。
        avg_bs = []
        all_shapes = []
        for t_factor, s_factor in cfg.MULTIGRID.LONG_CYCLE_FACTORS:
            base_t = int(round(cfg.DATA.NUM_FRAMES * t_factor))
            base_s = int(round(cfg.DATA.TRAIN_CROP_SIZE * s_factor))
            if cfg.MULTIGRID.SHORT_CYCLE:
                shapes = [
                    [
                        base_t,
                        cfg.MULTIGRID.DEFAULT_S
                        * cfg.MULTIGRID.SHORT_CYCLE_FACTORS[0],
                    ],
                    [
                        base_t,
                        cfg.MULTIGRID.DEFAULT_S
                        * cfg.MULTIGRID.SHORT_CYCLE_FACTORS[1],
                    ],
                    [base_t, base_s],
                ]
            else:
                shapes = [[base_t, base_s]]

            # (T, S) -> (B, T, S)
            shapes = [
                [int(round(default_size / (s[0] * s[1] * s[1]))), s[0], s[1]]
                for s in shapes
            ]
            avg_bs.append(np.mean([s[0] for s in shapes]))
            all_shapes.append(shapes)

        # 无论 cfg.MULTIGRID.EPOCH_FACTOR 如何都先生成日程。
        total_iters = 0
        schedule = []
        for step_index in range(len(steps) - 1):
            step_epochs = steps[step_index + 1] - steps[step_index]

            for long_cycle_index, shapes in enumerate(all_shapes):
                cur_epochs = (
                    step_epochs * avg_bs[long_cycle_index] / sum(avg_bs)
                )

                cur_iters = cur_epochs / avg_bs[long_cycle_index]
                total_iters += cur_iters
                schedule.append((step_index, shapes[-1], cur_epochs))

        iter_saving = default_iters / total_iters

        final_step_epochs = cfg.SOLVER.MAX_EPOCH - steps[-1]

        # 令 fine-tuning 阶段与其余训练阶段具有相同的迭代节省比例。
        ft_epochs = final_step_epochs / iter_saving * avg_bs[-1]

        schedule.append((step_index + 1, all_shapes[-1][2], ft_epochs))

        # 根据目标 cfg.MULTIGRID.EPOCH_FACTOR 生成最终日程。
        x = (
            cfg.SOLVER.MAX_EPOCH
            * cfg.MULTIGRID.EPOCH_FACTOR
            / sum(s[-1] for s in schedule)
        )

        final_schedule = []
        total_epochs = 0
        for s in schedule:
            epochs = s[2] * x
            total_epochs += epochs
            final_schedule.append((s[0], s[1], int(round(total_epochs))))
        print_schedule(final_schedule)
        return final_schedule


def print_schedule(schedule):
    """打印 multigrid 日程，便于核对训练形状变化。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    logger.info("Long cycle 索引\t基础形状\tEpoch 数")
    for s in schedule:
        logger.info("{}\t{}\t{}".format(s[0], s[1], s[2]))


def get_current_long_cycle_shape(schedule, epoch):
    """根据当前 epoch 从日程中取出 long cycle 形状。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    for s in schedule:
        if epoch < s[-1]:
            return s[1]
    return schedule[-1][1]
