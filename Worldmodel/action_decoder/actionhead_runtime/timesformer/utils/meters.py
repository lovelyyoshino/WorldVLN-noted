# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""训练、验证和测试过程中的统计计量器。"""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import timesformer.utils.logging as logging
import timesformer.utils.metrics as metrics
import timesformer.utils.misc as misc

logger = logging.get_logger(__name__)


class TestMeter(object):
    """
    在测试阶段执行 multi-view ensemble。

    每个带唯一索引的视频会采样多个 clip，这些 clip 的预测会聚合成该视频的
    最终预测。准确率会使用给定的 ground truth labels 计算。
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
                构造用于保存预测和标签的 tensor。

                期望每个视频得到 num_clips 个预测，并在 num_videos 个视频上计算指标。

                参数：
                    num_videos (int): 待测试视频数量。
                    num_clips (int): 每个视频采样的 clip 数量，用于聚合视频最终预测。
                    num_cls (int): 每个预测的类别数。
                    overall_iters (int): 测试阶段的总迭代次数。
                    multi_label (bool): 如果为 True，使用 mAP 作为指标。
                    ensemble_method (str): ensemble 方法，可选 "sum" 或 "max"。

        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        # 初始化 tensor。
        self.video_preds = torch.zeros((num_videos, num_cls))
        if multi_label:
            self.video_preds -= 1e10

        self.video_labels = (
            torch.zeros((num_videos, num_cls))
            if multi_label
            else torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # 重置指标。
        self.reset()

    def reset(self):
        """
        重置测试指标缓存。
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        if self.multi_label:
            self.video_preds -= 1e10
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
                收集当前 batch 的预测，并即时执行 ensemble 累加。

                参数：
                    preds (tensor): 当前 batch 的预测，维度为 N x C，其中 N 是 batch size，
                        C 是通道数（num_cls）。
                    labels (tensor): 当前 batch 对应的标签，维度为 N。
                    clip_ids (tensor): 当前 batch 的 clip 索引，维度为 N。

        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "不支持的 ensemble_method：{}".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
                记录当前测试迭代的统计信息。

                参数：
                    cur_iter (int): 当前测试迭代编号。

        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        开始记录一次迭代的耗时。
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        停止记录一次迭代的耗时。
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        """结束数据加载计时，并开始网络前向计时。"""
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        """
        计算并记录最终 ensemble 后的指标。

        ks (tuple): topk_accuracies 使用的 top-k 列表。例如 ks = (1, 5)
            对应 top-1 和 top-5 accuracy。
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip 数量 {} ~= 预期 clip 数 {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}
        if self.multi_label:
            map = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            self.stats["map"] = map
        else:
            num_topks_correct = metrics.topks_correct(
                self.video_preds, self.video_labels, ks
            )
            topks = [
                (x / self.video_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]

            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
        logging.log_json_stats(self.stats)


class ScalarMeter(object):
    """
    用 deque 跟踪一串标量值的计量器。

    它会使用给定窗口大小保存近期数值，支持计算窗口内的中位数、平均值，也支持
    计算全局平均值。
    """

    def __init__(self, window_size):
        """
                参数：
                    window_size (int): deque 的最大长度。

        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        清空 deque 和累计统计量。
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        向 deque 中加入一个新的标量值。
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        计算当前窗口内数值的中位数。
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        计算当前窗口内数值的平均值。
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        计算从上次 reset 以来的全局平均值。
        """
        return self.total / self.count


class TrainMeter(object):
    """
    统计训练阶段的 loss、error、学习率和耗时等信息。
    """

    def __init__(self, epoch_iters, cfg):
        """
                参数：
                    epoch_iters (int): 一个 epoch 中的总迭代次数。
                    cfg (CfgNode): 配置对象。

        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # 当前小批量的错误率（按窗口平滑）。
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # 分类错误样本数量。
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD

    def reset(self):
        """
        重置训练计量器中的累计统计量。
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def iter_tic(self):
        """
        开始记录一次训练迭代的耗时。
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        停止记录一次训练迭代的耗时。
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        """结束数据加载计时，并开始网络前向/反向计时。"""
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size, stats={}):
        """
                更新当前训练统计信息。

                参数：
                    top1_err (float): top1 错误率。
                    top5_err (float): top5 错误率。
                    loss (float): loss 值。
                    lr (float): 学习率。
                    mb_size (int): 小批量大小。

        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        if not self._cfg.DATA.MULTI_LABEL:
            # 当前小批量统计。
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            # 累计统计。
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
                记录当前训练迭代的统计信息。

                参数：
                    cur_epoch (int): 当前 epoch 编号。
                    cur_iter (int): 当前迭代编号。

        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
                记录当前 epoch 的训练统计信息。

                参数：
                    cur_epoch (int): 当前 epoch 编号。

        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            avg_loss = self.loss_total / self.num_samples
            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["loss"] = avg_loss
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        logging.log_json_stats(stats)


class ValMeter(object):
    """
    统计验证阶段的 error、mAP 和耗时等信息。
    """

    def __init__(self, max_iter, cfg):
        """
                参数：
                    max_iter (int): 当前 epoch 的最大迭代次数。
                    cfg (CfgNode): 配置对象。

        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # 当前小批量的错误率（按窗口平滑）。
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # 全验证集上的最小错误率。
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # 分类错误样本数量。
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD

    def reset(self):
        """
        重置验证计量器中的累计统计量。
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def iter_tic(self):
        """
        开始记录一次验证迭代的耗时。
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        停止记录一次验证迭代的耗时。
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        """结束数据加载计时，并开始网络前向计时。"""
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, mb_size, stats={}):
        """
                更新当前验证统计信息。

                参数：
                    top1_err (float): top1 错误率。
                    top5_err (float): top5 错误率。
                    mb_size (int): 小批量大小。

        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size


    def update_predictions(self, preds, labels):
        """
                缓存当前批量的预测和标签，供多标签 mAP 计算使用。

                参数：
                    preds (tensor): 模型输出预测。
                    labels (tensor): 标签。

        """
        # 待办：将 update_prediction 与 update_stats 合并。
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
                记录当前验证迭代的统计信息。

                参数：
                    cur_epoch (int): 当前 epoch 编号。
                    cur_iter (int): 当前迭代编号。

        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats[key].get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
                记录当前 epoch 的验证统计信息。

                参数：
                    cur_epoch (int): 当前 epoch 编号。

        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            stats["map"] = get_map(
                torch.cat(self.all_preds).cpu().numpy(),
                torch.cat(self.all_labels).cpu().numpy(),
            )
        else:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            self.min_top1_err = min(self.min_top1_err, top1_err)
            self.min_top5_err = min(self.min_top5_err, top5_err)

            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["min_top1_err"] = self.min_top1_err
            stats["min_top5_err"] = self.min_top5_err

        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples

        logging.log_json_stats(stats)


def get_map(preds, labels):
    """
        计算多标签任务的 mAP。

        参数：
            说明：preds (numpy tensor): num_examples x num_classes。
            说明：labels (numpy tensor): num_examples x num_classes。

        返回：
            mean_ap (int): 最终 mAP 分数。

    """

    logger.info("正在为 {} 个样本计算 mAP".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "平均精度计算需要批量中包含足够数量的样本，"
            "当前样本不满足要求。"
        )

    mean_ap = np.mean(aps)
    return mean_ap
