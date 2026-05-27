# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
from pprint import pformat
from typing import Optional, Tuple, Union
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch.distributed as tdist

import infinity.utils.dist as dist
from infinity.models import Infinity
from infinity.models.ema import update_ema
from infinity.models.self_correction import SelfCorrection
from infinity.utils import arg_util, misc, wandb_utils
from infinity.utils.amp_opt import AmpOptimizer
from infinity.schedules import get_encode_decode_func
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

import queue
import threading

def save_token():
    """后台线程：异步把新提取的 VAE token 保存到磁盘缓存。"""
    while True:
        try:
            raw_features, feature_cache_files4images = save_token_queue.get()
            for i in range(len(feature_cache_files4images)):
                if not osp.exists(feature_cache_files4images[i]):
                    os.makedirs(osp.dirname(feature_cache_files4images[i]), exist_ok=True)
                    torch.save(raw_features[i], feature_cache_files4images[i])
                    print(f'VAE token cache 已保存到 {feature_cache_files4images[i]}')
                else:
                    print(f'{feature_cache_files4images[i]} 已存在，跳过保存')
        except Exception as e:
            print(f"保存 VAE token cache 失败: {e}")
        finally:
            save_token_queue.task_done()

save_token_queue = queue.Queue()
saver = threading.Thread(target=save_token, daemon=True)
saver.start()

class InfinityTrainer(object):
    """Infinity 模型的 SFT 训练器，负责前向、损失重加权、反向和保存恢复。"""
    def __init__(
        self,
        device,
        raw_scale_schedule: Tuple[int, ...],
        vae_local,
        gpt_wo_ddp: Infinity, gpt: DDP,
        gpt_opt: AmpOptimizer,
        label_smooth: float,
        zero=0,
        vae_type=True,
        reweight_loss_by_scale=0,
        gpt_wo_ddp_ema=None,
        gpt_ema=None,
        use_fsdp_model_ema=False,
        other_args=None,
    ):
        """保存训练依赖对象，并初始化损失函数与多尺度元信息。"""
        super(InfinityTrainer, self).__init__()

        self.zero = zero
        self.vae_type = vae_type

        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local = gpt, vae_local
        self.dynamic_scale_schedule = other_args.dynamic_scale_schedule
        self.steps_per_frame = other_args.steps_per_frame
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(other_args.dynamic_scale_schedule, other_args.video_frames)
        self.gpt_opt: AmpOptimizer = gpt_opt
        self.gpt_wo_ddp: Union[Infinity, torch._dynamo.eval_frame.OptimizedModule] = gpt_wo_ddp  # 中文说明：可能已经经过 torch.compile 包装
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema
        self.self_correction = SelfCorrection(self.vae_local, other_args)
        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.batch_size, self.seq_len = 0, 0
        self.reweight_loss_by_scale = reweight_loss_by_scale
        print(f'self.reweight_loss_by_scale: {self.reweight_loss_by_scale}（0 表示按 token 数加权，非 0 表示按尺度体积差重加权）')
        video_encode, _, _, _ = get_encode_decode_func(other_args.dynamic_scale_schedule)
        self.video_encode = video_encode

        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        del gpt_uncompiled.rng
        gpt_uncompiled.rng = torch.Generator(device=device)
        del gpt_uncompiled

        self.label_smooth = label_smooth

        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
        self.loss_weight = {0:{}, 1:{}}

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.generator = np.random.default_rng(0)

    def train_step(
        self, epoch: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float, metric_lg: misc.MetricLogger,
        raw_features_bcthw: FTen, feature_cache_files4images: list, media: str,
        inp_B3HW: FTen, text_cond_tuple: Union[ITen, FTen], args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """执行一次训练 step，并返回梯度范数与 AMP 缩放信息。"""
        device = args.device
        B = len(inp_B3HW) + len(raw_features_bcthw)

        if media == 'images':
            is_image_batch = 1
        else:
            is_image_batch = 0
        # 前向阶段：先把图像编码成 VAE latent，再做 packed transformer 前向。
        with self.gpt_opt.amp_ctx:
            with torch.amp.autocast('cuda', enabled=False):
                raw_features_list = []
                if len(inp_B3HW):
                    with torch.no_grad():
                        for inp_ind, inp in enumerate(inp_B3HW):
                            raw_features_, _, _ = self.vae_local.encode_for_raw_features(inp.unsqueeze(0), scale_schedule=None, slice=args.use_slice)
                            raw_features_list.append(raw_features_)
                            if args.use_vae_token_cache and args.save_vae_token_cache and (not osp.exists(feature_cache_files4images[inp_ind])):
                                os.makedirs(osp.dirname(feature_cache_files4images[inp_ind]), exist_ok=True)
                                save_token_queue.put((raw_features_.cpu().data, [feature_cache_files4images[inp_ind]]))
                if len(raw_features_bcthw):
                    raw_features_bcthw = [item.unsqueeze(0) for item in raw_features_bcthw]
                    raw_features_list = raw_features_list + raw_features_bcthw

            full_pts_this_batch = [item.shape[-3] for item in raw_features_list]
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
            x_BLC, x_BLC_mask, gt_BLC, pred_all_bit_indices, visual_rope_cache, sequece_packing_scales, super_scale_lengths, super_querysid_super_refsid, other_info_by_scale = self.video_encode(
                vae=self.vae_local,
                inp_B3HW=None,
                vae_features=raw_features_list,
                self_correction=self.self_correction,
                args=args,
                device=device,
                rope2d_freqs_grid=self.gpt.rope2d_freqs_grid,
                dynamic_resolution_h_w=self.dynamic_resolution_h_w,
                text_lens=lens,
                tokens_remain=args.train_max_token_len,
            )

            loss, acc_bit, valid_sequence_ratio = self.gpt(
                text_cond_tuple,
                x_BLC,
                gt_BL=gt_BLC,
                is_image_batch=is_image_batch,
                visual_rope_cache=visual_rope_cache,
                sequece_packing_scales=sequece_packing_scales,
                super_scale_lengths=super_scale_lengths,
                super_querysid_super_refsid=super_querysid_super_refsid,
                other_info_by_scale=other_info_by_scale,
            ) # 中文说明：loss & acc_bit: [seq_len]

            # 多尺度损失重加权：更小的 patch 数量可按体积比放大权重。
            # 代码/形状说明：import pdb; pdb.set_trace()
            acc_pt2scale_acc = {}
            acc_pt2scale_acc_counter = {}
            for full_pt, scale_schedule in self.dynamic_resolution_h_w[self.h_div_w_templates[0]][args.pn]['pt2scale_schedule'].items():
                acc_pt2scale_acc[full_pt] = [[] for _ in range(len(scale_schedule))]
                acc_pt2scale_acc_counter[full_pt] = [0 for _ in range(len(scale_schedule))]

            flatten_L_list, flatten_acc_bit_list, flatten_weight_list = [], [], []
            ptr = 0
            global_scale_ind = 0
            for sample_ind, item in enumerate(sequece_packing_scales):
                full_pt = full_pts_this_batch[sample_ind]
                for si, (pt, ph, pw) in enumerate(item):
                    mul_pt_ph_pw = pt * ph * pw
                    start, end = ptr, ptr+mul_pt_ph_pw
                    ptr = end
                    if x_BLC_mask is None:
                        loss_this_scale = loss[start:end].mean()
                        acc_this_scale = acc_bit[start:end].mean()
                    else:
                        pred_elem_num = x_BLC_mask[start:end].sum()
                        assert pred_elem_num > 0
                        loss_this_scale = loss[start:end].sum() / pred_elem_num
                        acc_this_scale = acc_bit[start:end].sum() / pred_elem_num
                    real_si = other_info_by_scale[global_scale_ind]['real_si']
                    volume_times = np.array(other_info_by_scale[global_scale_ind]['largest_scale']).prod() / mul_pt_ph_pw
                    acc_pt2scale_acc[full_pt][real_si].append(acc_this_scale)
                    acc_pt2scale_acc_counter[full_pt][real_si] += 1
                    if self.reweight_loss_by_scale == 0:
                        weight = 1 * mul_pt_ph_pw
                    else:
                        reweight_value = min(args.max_reweight_value, np.power(volume_times, 1/(1+self.reweight_loss_by_scale)))
                        weight = reweight_value * mul_pt_ph_pw
                    flatten_weight_list.append(weight)
                    flatten_L_list.append(loss_this_scale)
                    flatten_acc_bit_list.append(acc_this_scale)
                    global_scale_ind += 1
            flatten_weight_list = torch.tensor(flatten_weight_list, dtype=loss.dtype, device=loss.device)
            flatten_weight_list = flatten_weight_list / flatten_weight_list.sum()
            final_loss = (torch.stack(flatten_L_list) * flatten_weight_list).sum()
            final_acc_bit = (torch.stack(flatten_acc_bit_list) * flatten_weight_list).sum()

        # 反向阶段：统一由 AmpOptimizer 负责 mixed precision / grad clip / step。
        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(ep=epoch, it=it, g_it=g_it, stepping=stepping, loss=final_loss, clip_decay_ratio=clip_decay_ratio)

        # EMA 更新公式仍是 `ema = decay * ema + (1 - decay) * param`。
        if args.use_fsdp_model_ema and (args.model_ema_decay < 1):
            update_ema(self.gpt_ema, self.gpt)

        # 只有真的发生 optimizer.step 时才清空梯度。
        if stepping:
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)

        # 指标聚合：先按尺度求和，再跨卡 all_reduce。
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
            def sum_dict(acc_pt2scale_acc):
                """把按尺度收集的 tensor 列表压成逐尺度总和。"""
                for full_pt in acc_pt2scale_acc:
                    for si in range(len(acc_pt2scale_acc[full_pt])):
                        acc_pt2scale_acc[full_pt][si] = torch.tensor(acc_pt2scale_acc[full_pt][si]).sum()
                return acc_pt2scale_acc

            def dict2list(acc_pt2scale_acc):
                """把嵌套字典拍平成列表，方便组成一个 all_reduce 张量。"""
                flatten_acc_pt2scale_acc = []
                for key, val in acc_pt2scale_acc.items():
                    flatten_acc_pt2scale_acc.extend(val)
                return flatten_acc_pt2scale_acc

            def list2dict(acc_pt2scale_acc, flatten_acc_pt2scale_acc):
                """把 all_reduce 后的扁平列表按原结构写回字典。"""
                ptr = 0
                for key in acc_pt2scale_acc:
                    for ind in range(len(acc_pt2scale_acc[key])):
                        acc_pt2scale_acc[key][ind] = flatten_acc_pt2scale_acc[ptr]
                        ptr += 1
                return acc_pt2scale_acc

            acc_pt2scale_acc = sum_dict(acc_pt2scale_acc)
            flatten_acc_pt2scale_acc = dict2list(acc_pt2scale_acc)
            flatten_acc_pt2scale_acc_counter = dict2list(acc_pt2scale_acc_counter)

            train_loss = final_loss.item()
            train_acc = final_acc_bit.item()
            metrics = torch.tensor(flatten_acc_pt2scale_acc + flatten_acc_pt2scale_acc_counter + [grad_norm_t.item(), train_loss, train_acc, is_image_batch, valid_sequence_ratio], device=loss.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            flatten_acc_pt2scale_acc, flatten_acc_pt2scale_acc_counter = metrics[:len(flatten_acc_pt2scale_acc)], metrics[len(flatten_acc_pt2scale_acc):2*len(flatten_acc_pt2scale_acc)]
            flatten_acc_pt2scale_acc = flatten_acc_pt2scale_acc / (flatten_acc_pt2scale_acc_counter + 1e-16)
            acc_pt2scale_acc = list2dict(acc_pt2scale_acc, flatten_acc_pt2scale_acc)
            acc_pt2scale_acc_counter = list2dict(acc_pt2scale_acc_counter, flatten_acc_pt2scale_acc_counter)
            grad_norm_t, train_loss, train_acc, is_image_batch, valid_sequence_ratio = metrics[2*len(flatten_acc_pt2scale_acc):] / (dist.get_world_size() + 1e-16)
            if args.num_of_label_value == 1:
                key, base = 'Loss', 1
            else:
                key, base = 'Acc', 100
            metric_lg.update(L=train_loss, Acc=train_acc*base, L_i=0., Acc_i=0., L_v=0., Acc_v=0., tnm=grad_norm_t, seq_usage=valid_sequence_ratio*100.)    # 中文说明：todo: Accm, Acct
            wandb_log_dict = {
                'Overall/train_loss': train_loss,
                'Overall/train_acc': train_acc*base,
                'Overall/grad_norm_t': grad_norm_t,
                'Overall/video_batch_ratio': (1-is_image_batch)*100.,
                'Overall/valid_sequence_ratio': valid_sequence_ratio*100.,
            }
            for full_pt in acc_pt2scale_acc:
                for si in range(len(acc_pt2scale_acc[full_pt])):
                    if acc_pt2scale_acc_counter[full_pt][si] > 0:
                        duration = (full_pt-1) / args.temporal_compress_rate
                        wandb_log_dict[f'Details/{key}/t{duration:04.1f}s/s{si+1:03d}'] = acc_pt2scale_acc[full_pt][si].item() * base
                        wandb_log_dict[f'Details/Num/t{duration:04.1f}s/s{si+1:03d}'] = acc_pt2scale_acc_counter[full_pt][si]
            wandb_utils.log(wandb_log_dict, step=g_it)
        return grad_norm_t, scale_log2_t

    def __repr__(self):
        """返回训练器配置与结构摘要，便于日志打印。"""
        return (
            f'\n'
            f'[VGPTTr.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[VGPTTr.structure]: {super(InfinityTrainer, self).__repr__().replace(InfinityTrainer.__name__, "")}'
        )

    def ema_load(self):
        """把在线模型参数临时替换成 EMA 参数，常用于验证前切换。"""
        self.cached_state_not_ema = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            self.gpt_opt.paras[pi].data.copy_(p_ema)
        for pi, para in enumerate(self.gpt_opt.paras):
            dist.broadcast(para, src_rank=pi % dist.get_world_size())

    def ema_recover(self):
        """从缓存中恢复非 EMA 的原始训练参数。"""
        self.gpt_wo_ddp.load_state_dict(self.cached_state_not_ema)
        del self.cached_state_not_ema
        self.cached_state_not_ema = None

    def get_config(self):
        """导出需要随 checkpoint 一起保存的轻量训练状态。"""
        return {
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }

    def state_dict(self):
        """打包训练器状态，包括配置、VAE、模型和优化器。"""
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config(), 'vae_local': m.state_dict()}

        if self.zero:   # 中文说明：待修复；zero 路径需要单独处理 state_dict 加载细节。
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                state['gpt_fsdp'] = self.gpt.state_dict()
                if self.use_fsdp_model_ema:
                    state['gpt_ema_fsdp'] = self.gpt_ema.state_dict()
                state['gpt_fsdp_opt'] = FSDP.optim_state_dict(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=self.gpt_opt.optimizer.state_dict())
            if self.gpt_opt.scaler is not None:
                state['gpt_opt_scaler'] = self.gpt_opt.scaler.state_dict()

        else:

            for k in ('gpt_wo_ddp', 'gpt_opt'):
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True, skip_vae=False):
        """从 checkpoint 恢复模型、优化器和训练进度状态。"""
        if self.zero:
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                self.gpt.load_state_dict(state['gpt_fsdp'])
                if self.use_fsdp_model_ema:
                    self.gpt_ema.load_state_dict(state['gpt_ema_fsdp'])
                one_group_opt_state = state['gpt_fsdp_opt']
                """
                AdamW state['gpt_fsdp_opt'] 结构示意：
                {
                    'state': { <para_name>: {'exp_avg': <unsharded_tensor>, 'exp_avg_sq': <unsharded_tensor>, 'step': <int>} },
                    'param_groups': [
                        {
                            'wd_sc': 1.0, 'lr_sc': 1.0, 'lr': xxx, 'betas': (0.9, 0.97), 'eps': 1e-08, 'weight_decay': 0.02,
                            'amsgrad': False, 'foreach': None, 'maximize': False, 'capturable': False, 'differentiable': False, 'fused': True,
                            'params': [<para_name> x m]
                        } x n
                    ]
                }
                这里保留英文 key 名是因为它们来自 PyTorch optimizer state_dict。
                one_group_opt_state['param_groups'] = self.gpt_opt.optimizer.state_dict()['param_groups']
                """
                optim_state_dict = FSDP.optim_state_dict_to_load(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=one_group_opt_state)
                self.gpt_opt.optimizer.load_state_dict(optim_state_dict)

            if self.gpt_opt.scaler is not None:
                try: self.gpt_opt.scaler.load_state_dict(state['gpt_opt_scaler'])
                except Exception as e: print(f'[fp16 load_state_dict 错误] {e}')
        else:
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                if skip_vae and 'vae' in k: continue
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    ret = m.load_state_dict(state[k], strict=strict)
                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[VGPTTr.load_state_dict] {k} 缺失 keys:  {missing}')
                        print(f'[VGPTTr.load_state_dict] {k} 未预期 keys:  {unexpected}')

        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VGPT.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
