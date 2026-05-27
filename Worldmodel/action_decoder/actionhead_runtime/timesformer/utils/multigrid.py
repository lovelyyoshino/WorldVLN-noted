# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""multigrid 训练辅助函数。"""

import numpy as np

import timesformer.utils.logging as logging

logger = logging.get_logger(__name__)


class MultigridSchedule(object):
    """
    定义 multigrid 训练日程，并据此更新配置。
    """

    def init_multigrid(self, cfg):
        """
                根据 multigrid 设置更新配置。

                参数：
                    cfg (configs): 包含训练和 multigrid 超参数的配置，细节见
                        说明：`slowfast/config/defaults.py`。

                返回：
                    cfg (configs): 更新后的配置。

        """
        self.schedule = None
        # 训练过程中可能会改动这些字段，因此先保存默认值，供全局复用。
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
        """
                每个 epoch 开始前检查 long cycle 的 shape 是否需要切换，若需要则同步更新配置。

                参数：
                    cfg (configs): 包含训练和 multigrid 超参数的配置，细节见
                        说明：`slowfast/config/defaults.py`。
                    cur_epoch (int): 当前 epoch 索引。

                返回：
                    cfg (configs): 更新后的配置。
                    changed (bool): 当前 epoch 是否发生了 long cycle shape 切换。

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
        """
                根据 multigrid 超参数构建 long cycle 训练日程。

                参数：
                    cfg (configs): 包含训练和 multigrid 超参数的配置，细节见
                        说明：`slowfast/config/defaults.py`。

                返回：
                    schedule (list): long cycle 的 base shape 列表及其对应训练 epoch。

        """

        steps = cfg.SOLVER.STEPS

        default_size = float(
            cfg.DATA.NUM_FRAMES * cfg.DATA.TRAIN_CROP_SIZE ** 2
        )
        default_iters = steps[-1]

        # 计算每个 long cycle shape 对应的形状和平均 batch size。
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

        # 先构造不考虑 cfg.MULTIGRID.EPOCH_FACTOR 的基础日程。
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

        # 令微调阶段与前面训练阶段具有相同的迭代节省比例。
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
    """
    记录训练日程。
    """
    logger.info("Long cycle 索引\t基础形状\tEpoch 数")
    for s in schedule:
        logger.info("{}\t{}\t{}".format(s[0], s[1], s[2]))


def get_current_long_cycle_shape(schedule, epoch):
    """
        根据日程和 epoch 索引返回当前的 long cycle base shape。

        参数：
            schedule (configs): 包含训练和 multigrid 超参数的配置，细节见
                说明：`slowfast/config/defaults.py`。
            cur_epoch (int): 当前 epoch 索引。

        返回：
            shapes (list): 描述 long cycle base shape 的列表：
                [相对默认值的 batch size，帧数，空间尺寸]。

    """
    for s in schedule:
        if epoch < s[-1]:
            return s[1]
    return schedule[-1][1]
