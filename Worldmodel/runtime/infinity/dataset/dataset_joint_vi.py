# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import glob
import os
import time
from os import path as osp
from typing import List, Tuple
import json
import hashlib
import copy
import collections

import tqdm
import numpy as np
import torch
import pandas as pd
from decord import VideoReader
from PIL import Image as PImage
from torchvision.transforms.functional import to_tensor
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as tdist
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta
from infinity.utils.video_decoder import EncodedVideoDecord, EncodedVideoOpencv
from transformers import AutoTokenizer

def transform(pil_img, tgt_h, tgt_w):
    """把输入图像按目标宽高比缩放、中心裁剪，并映射到 `[-1, 1]`。"""
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # 先按目标宽高比缩放，再做中心裁剪，避免直接拉伸变形。
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    # 代码/形状说明：print(f'im size {im.shape}')
    return im.add(im).add_(-1)

def get_prompt_id(prompt):
    """为文本提示生成稳定的 MD5 标识，便于缓存复用。"""
    md5 = hashlib.md5()
    md5.update(prompt.encode('utf-8'))
    prompt_id = md5.hexdigest()
    return prompt_id

def prepend_motion_score(prompt, motion_score):
    """把运动强度标签前缀拼到 caption 前面。"""
    return f'<<<motion_score: {round(motion_score):.1f}>>> {prompt}'

class VideoReaderWrapper(VideoReader):
    """读取后自动回到开头的 VideoReader 包装器。"""
    def __init__(self, *args, **kwargs):
        """初始化底层 reader，并把游标复位到第 0 帧。"""
        super().__init__(*args, **kwargs)
        self.seek(0)
    def __getitem__(self, key):
        """读取指定帧后立刻复位，避免下一次访问受上次状态影响。"""
        frames = super().__getitem__(key)
        self.seek(0)
        return frames


class JointViIterableDataset(IterableDataset):
    """混合图像与视频样本的流式数据集入口。"""
    def __init__(
        self,
        video_meta_folder: str = '',
        buffersize: int = 1000000 * 300,
        seed: int = 0,
        pn: str = '',
        video_fps: int = 1,
        num_replicas: int = 1,
        rank: int = 0,
        dataloader_workers: int = 2,
        dynamic_resolution_across_gpus: bool = True,
        enable_dynamic_length_prompt: bool = True,
        shuffle: bool = True,
        short_prob: float = 0.2,
        verbose=False,
        temp_dir= "/dev/shm",
        add_motion_score2caption=False,
        other_args=None,
        **kwargs,
    ):
        """读取数据源配置、分辨率模板、缓存策略并预生成 batch。"""
        self.video_meta_folder = video_meta_folder
        self.pn = pn
        self.verbose = verbose
        self.buffer_size = buffersize
        self.num_replicas = num_replicas
        self.rank = rank
        self.worker_id = 0
        self.global_worker_id = 0
        self.short_prob = short_prob
        self.dataloader_workers = max(1, dataloader_workers)
        self.shuffle = shuffle
        self.global_workers = self.num_replicas * self.dataloader_workers
        self.add_motion_score2caption = add_motion_score2caption
        self.seed = seed
        self.text_tokenizer = other_args.text_tokenizer
        self.feature_extraction = other_args.cache_check_mode < 0 # 特征抽取阶段不做 sequence packing。
        self.epoch_generator = None
        self.epoch_worker_generator = None
        self.epoch_global_worker_generator = None
        self.epoch_rank_generator = None
        self.other_args = other_args
        self.drop_long_video = other_args.drop_long_video
        self.dynamic_resolution_across_gpus = dynamic_resolution_across_gpus
        self.enable_dynamic_length_prompt = enable_dynamic_length_prompt
        # `other_args.epoch` 是总轮数，当前轮次保存在 `cur_epoch`。
        self.set_epoch(getattr(other_args, "cur_epoch", other_args.epoch))
        self.temporal_compress_rate = other_args.temporal_compress_rate
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(other_args.dynamic_scale_schedule, other_args.video_frames) # 这里的 `video_frames` 是允许的最大帧数。
        self.train_h_div_w_list = self.h_div_w_templates
        self.video_fps = video_fps
        self.min_training_duration = (other_args.min_video_frames-1) // self.video_fps
        self.max_training_duration = (other_args.video_frames-1) // self.video_fps
        self.append_duration2caption = other_args.append_duration2caption
        print(f"{self.rank=} 数据集 {self.seed=}, {self.append_duration2caption=} add_motion_score2caption={add_motion_score2caption}, {self.min_training_duration=} {self.max_training_duration=}, cache_check_mode={self.other_args.cache_check_mode}")
        self.token_cache_dir = other_args.token_cache_dir
        self.use_vae_token_cache = other_args.use_vae_token_cache
        self.allow_online_vae_feature_extraction = other_args.allow_online_vae_feature_extraction
        self.use_text_token_cache = other_args.use_text_token_cache
        self.max_video_frames = other_args.video_frames
        self.cached_video_frames = other_args.cached_video_frames # 缓存里允许的最大视频帧数。
        self.image_batches_multiply = other_args.image_batches_multiply
        self.down_size_limit = other_args.down_size_limit
        self.addition_pn_list = json.loads(other_args.addition_pn_list)
        self.video_caption_type = other_args.video_caption_type
        self.train_max_token_len = other_args.train_max_token_len
        self.duration_resolution = other_args.duration_resolution
        self.append_duration2caption = other_args.append_duration2caption
        self.device = other_args.device
        self.video_sample_mode = getattr(other_args, "video_sample_mode", "duration")
        print(f'self.down_size_limit: {self.down_size_limit}')
        self.max_text_len = other_args.tlen
        self.temp_dir = temp_dir.rstrip("/")
        self.metas = self.get_meta()
        self.batches, self.batch_nums = self.form_batches(self.metas)
        if self.batch_nums == 0:
            raise ValueError(
                f'在 video_meta_folder={self.video_meta_folder} 中没有找到可用训练 batch。'
                f'请检查 jsonl 路径/内容以及过滤参数（例如 min_video_frames={self.other_args.min_video_frames}）。'
            )
        print(f'{num_replicas=}, {rank=}, {dataloader_workers=}, {self.batch_nums=}, {self.drop_long_video=} {self.max_text_len=}')

    def append_duration_info(self, meta, mapped_duration):
        """把视频时长提示拼进 caption 前缀中。"""
        meta['caption'] = f'<<<t={mapped_duration}s>>>' + meta['caption']
        return meta

    def get_captions_lens(self, captions):
        """批量计算 caption 的有效 token 长度。"""
        if self.other_args.text_tokenizer_type == 'flan_t5':
            tokens = self.other_args.text_tokenizer(text=captions, max_length=self.other_args.text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')
            mask = tokens.attention_mask.cuda(non_blocking=True)
            lens: List[int] = mask.sum(dim=-1).tolist()
        else: # umt5-xxl
            ids, mask = self.other_args.text_tokenizer( captions, return_mask=True, add_special_tokens=True)
            lens = mask.gt(0).sum(dim=1).tolist()
        return lens

    def get_meta(self):
        """读取 jsonl 元数据、过滤无效样本，并补齐训练所需字段。"""
        # 优先读取 `root/<bucket>/*.jsonl` 这种分桶目录结构。
        part_filepaths = sorted(glob.glob(osp.join(self.video_meta_folder, '*/*.jsonl')))
        # 若没有分桶，则退回到平铺结构 `root/*.jsonl`。
        if len(part_filepaths) == 0:
            part_filepaths = sorted(glob.glob(osp.join(self.video_meta_folder, '*.jsonl')))
        if int(getattr(self.other_args, "shuffle_batches", 1) or 0) == 1:
            self.epoch_generator.shuffle(part_filepaths)
        print(f'jsonls 样例：{part_filepaths[:4]}')
        if self.num_replicas > 1:
            part_filepaths = part_filepaths[self.rank::self.num_replicas]
            # 当 shard 数少于 world size 时，某些 rank 会拿到空分片。
            # 这里退回到取模复用一个 shard，优先保证训练能继续跑。
            if len(part_filepaths) == 0:
                all_part_filepaths = sorted(glob.glob(osp.join(self.video_meta_folder, '*/*.jsonl')))
                if len(all_part_filepaths) == 0:
                    all_part_filepaths = sorted(glob.glob(osp.join(self.video_meta_folder, '*.jsonl')))
                if len(all_part_filepaths) > 0:
                    if int(getattr(self.other_args, "shuffle_batches", 1) or 0) == 1:
                        self.epoch_generator.shuffle(all_part_filepaths)
                    fallback_idx = int(self.rank) % len(all_part_filepaths)
                    part_filepaths = [all_part_filepaths[fallback_idx]]
                    print(
                        f'[data preprocess] rank={self.rank} 拿到空 shard slice；'
                        f'退回使用 shard {part_filepaths[0]}'
                    )

        metas = []
        pbar = tqdm.tqdm(total=len(part_filepaths))
        mapped_duration2freqs = collections.defaultdict(int)
        total, corrupt = 0, 0
        stop_read = False
        rough_h_div_w = self.h_div_w_templates[np.argmin(np.abs((9/16-self.h_div_w_templates)))]
        for part_filepath in part_filepaths:
            if stop_read:
                break
            pbar.update(1)
            with open(part_filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                total += 1
                try:
                    meta = json.loads(line)
                except Exception as e:
                    print(e)
                    corrupt += 1
                    print(e, corrupt, total, corrupt/total)
                    continue
                if 'h_div_w' in meta:
                    del meta['h_div_w']
                if 'video_path' in meta:
                    begin_frame_id, end_frame_id, fps = meta['begin_frame_id'], meta['end_frame_id'], meta['fps']
                    # 标注片段总帧数的保守估计。很多数据集把 `end_frame_id` 视为闭区间，所以这里加 1。
                    seg_frames = max(1, int(end_frame_id) - int(begin_frame_id) + 1)
                    real_duration = (end_frame_id - begin_frame_id) / fps
                    # 预处理若已给出精确采样帧列表，就直接尊重它。
                    has_frame_idxs = ('frame_idxs' in meta and isinstance(meta['frame_idxs'], list) and len(meta['frame_idxs']) > 0)

                    # 时长与过滤逻辑：若使用整段采样，就尽量与标注片段对齐。
                    if self.video_sample_mode in ("fixed_full", "segment_full", "segment_full_floor_sec"):
                        mapped_duration = real_duration
                        # 有 frame_idxs 时按真实采样帧数过滤，否则按片段总帧数过滤。
                        if (self.other_args.min_video_frames is not None) and (self.other_args.min_video_frames > 0):
                            if has_frame_idxs:
                                if len(meta['frame_idxs']) < self.other_args.min_video_frames:
                                    continue
                            else:
                                if seg_frames < self.other_args.min_video_frames:
                                    continue
                    else:
                        mapped_duration = int(real_duration / self.duration_resolution) * self.duration_resolution
                        if mapped_duration < self.min_training_duration:
                            continue
                        if mapped_duration > self.max_training_duration:
                            if self.drop_long_video:
                                continue
                            else:
                                mapped_duration = self.max_training_duration
                    caption_type = 'tarsier2_caption'
                    if ('MiniCPM_V_2_6_caption' in meta) and meta['MiniCPM_V_2_6_caption']:
                        caption_type = self.epoch_rank_generator.choice(['tarsier2_caption', 'MiniCPM_V_2_6_caption'])
                    meta['caption'] = meta[caption_type]
                    if self.enable_dynamic_length_prompt and (self.epoch_rank_generator.random() < self.short_prob):
                        meta['caption'] = self.random_drop_sentences(meta['caption'])
                    if 'quality_prompt' in meta:
                        meta['caption'] = meta['caption'] + ' ' + meta['quality_prompt']
                    if self.append_duration2caption:
                        # 时长提示也要与整段视频保持一致。
                        meta = self.append_duration_info(meta, round(float(mapped_duration), 2))
                    assert meta['caption']
                    if self.video_sample_mode == "fixed_full":
                        sample_frames = int(meta.get('sample_frames', self.max_video_frames)) if has_frame_idxs else int(self.max_video_frames)
                    elif self.video_sample_mode == "segment_full":
                        # 整段采样时保留 `4n+1` 规则，保证时间压缩后仍能对齐 VAE/模型的时间步。
                        cap = int(self.max_video_frames) if (self.max_video_frames and self.max_video_frames > 0) else seg_frames
                        sample_frames = min(seg_frames, cap)
                        # 关键公式：`1, 5, 9, ...` 这类长度都可以写成 `1 + 4k`。
                        sample_frames = max(1, 1 + 4 * ((sample_frames - 1) // 4))
                    elif self.video_sample_mode == "segment_full_floor_sec":
                        # 整段采样，但长度向下取整到 `1 + video_fps * n`，
                        # 这样更接近“整秒锚点”的预训练分布。
                        cap = int(self.max_video_frames) if (self.max_video_frames and self.max_video_frames > 0) else seg_frames
                        seg_frames_cap = min(seg_frames, cap)
                        # 合法长度遵循 `1, 1+fps, 1+2*fps, ...`。
                        n_secs = max(0, int((seg_frames_cap - 1) // max(1, int(self.video_fps))))
                        sample_frames = 1 + n_secs * int(self.video_fps)
                        # 再次投影到 `4n+1`，因为当前训练配置通常要求时间步数满足这个形式。
                        sample_frames = max(1, 1 + 4 * ((sample_frames - 1) // 4))
                        # 若追加时长标签，也同步使用取整后的秒数。
                        if self.append_duration2caption:
                            meta = self.append_duration_info(meta, n_secs * self.duration_resolution)
                    else:
                        sample_frames = int(mapped_duration * self.video_fps + 1)
                    if has_frame_idxs:
                        # 预处理给出的 frame_idxs 具有最高优先级。
                        sample_frames = len(meta['frame_idxs'])
                    pt = (sample_frames-1) // self.temporal_compress_rate + 1
                    scale_schedule = self.dynamic_resolution_h_w[rough_h_div_w][self.pn]['pt2scale_schedule'][pt]
                    meta['sample_frames'] = sample_frames
                elif 'image_path' in meta:
                    mapped_duration = -1
                    scale_schedule = self.dynamic_resolution_h_w[rough_h_div_w][self.pn]['pt2scale_schedule'][1]
                    if not meta['text']:
                        meta['caption'] = meta['long_caption']
                    elif not meta['long_caption']:
                        meta['caption'] = meta['text']
                    else:
                        if self.epoch_rank_generator.random() < self.other_args.short_cap_prob:
                            meta['caption'] = meta['text']
                        else:
                            meta['caption'] = meta['long_caption']
                    if self.enable_dynamic_length_prompt and (self.epoch_rank_generator.random() < self.short_prob):
                        meta['caption'] = self.random_drop_sentences(meta['caption'])
                else:
                    raise ValueError(f'meta 中缺少 video_path 或 image_path：{meta}')

                cum_visual_tokens = np.array(scale_schedule).prod(-1).cumsum()
                meta['cum_text_visual_tokens'] = cum_visual_tokens
                if self.other_args.cache_check_mode == 1: # 中文标题：check at the begining
                    if self.exists_cache_file(meta):
                        metas.append(meta)
                elif self.other_args.cache_check_mode == -1: # 中文说明：select unexist, used for token cache
                    if not self.exists_cache_file(meta):
                        metas.append(meta)
                else:
                    metas.append(meta)
                mapped_duration2freqs[mapped_duration] += 1
                if (self.other_args.restrict_data_size > 0) and (len(metas) > self.other_args.restrict_data_size / self.num_replicas):
                    stop_read = True
                    break

        # 给每个样本补齐文本 token 长度信息，后续 sequence packing 要用。
        metas = self.append_text_tokens(metas)

        self.epoch_rank_generator.shuffle(metas)
        for mapped_duration in sorted(mapped_duration2freqs.keys()):
            freq = mapped_duration2freqs[mapped_duration]
            proportion = freq / len(metas) * 100
            print(f'{mapped_duration=}, {freq=}, {proportion=:.1f}%')
        return metas

    def append_text_tokens(self, metas, bucket_size=100):
        """批量估算文本 token 数，并更新每个样本的总 token 占用。"""
        t1 = time.time()
        max_text_visual_tokens = -1
        pbar = tqdm.tqdm(total=len(metas) // bucket_size + 1, desc='append text tokens')
        for bucket_id in range(len(metas) // bucket_size + 1):
            pbar.update(1)
            start = bucket_id * bucket_size
            end = min(start + bucket_size, len(metas))
            if start >= end:
                break
            if self.feature_extraction:
                lens = [0 for i in range(start, end)]
            else:
                captions = [metas[i]['caption'] for i in range(start, end)]
                assert len(captions), f'{len(captions)=}'
                lens = self.get_captions_lens(captions)
            for i in range(start, end):
                metas[i]['text_tokens'] = min(self.max_text_len, lens[i-start])
                metas[i]['cum_text_visual_tokens'] = metas[i]['cum_text_visual_tokens'] + metas[i]['text_tokens']
                metas[i]['text_visual_tokens'] = metas[i]['cum_text_visual_tokens'][-1]
                max_text_visual_tokens = max(max_text_visual_tokens, metas[i]['text_visual_tokens'])
        if not self.other_args.allow_less_one_elem_in_seq:
            assert max_text_visual_tokens <= self.train_max_token_len, f'{self.train_max_token_len=} 应大于 {max_text_visual_tokens=}'
        t2 = time.time()
        print(f'追加 text tokens 耗时：{t2-t1:.1f}s')
        return metas

    def exists_cache_file(self, meta):
        """检查当前样本对应的 VAE token 缓存是否已经存在。"""
        if 'image_path' in meta:
            return osp.exists(self.get_image_cache_file(meta['image_path']))
        else:
            if '/vdataset/clip' in meta['video_path']: # 中文说明：clip
                cache_file = self.get_video_cache_file(meta['video_path'], 0, meta['end_frame_id']-meta['begin_frame_id'], self.video_fps)
            else:
                cache_file = self.get_video_cache_file(meta['video_path'], meta['begin_frame_id'], meta['end_frame_id'], self.video_fps)
            return osp.exists(cache_file)

    def form_batches(self, metas):
        """按 token 预算把样本打包成 batch，尽量提高序列利用率。"""
        st = time.time()
        if len(metas) == 0:
            # 流式场景下某个 shard / buffer 可能暂时为空；直接返回空 batch 即可。
            return [], 0
        # open-loop rollout 训练在 trainer 内部按视频推进，因此强制 batch size = 1，
        # 避免 sequence packing 干扰轨迹边界。
        if int(getattr(self.other_args, "rollout_train_open_loop", 0)) == 1:
            print(f'[data preprocess] rollout_train_open_loop=1：关闭 sequence packing（强制 batch size = 1）')
            batches = [[item] for item in range(len(metas))]
        elif self.feature_extraction: # 特征抽取模式不做 sequence packing。
            batches = [[item] for item in range(len(metas))]
        else:
            batches = []
            has_been_used = [False for _ in range(len(metas))]
            # 防御性处理：避免 `seq_pack_bucket=0` 触发除零。
            bucket_size = max(1, min(len(metas), self.other_args.seq_pack_bucket))
            print(f'[data preprocess] form_batches 处理 {len(metas)} 条 metas，bucket_size={bucket_size}...')
            step = len(metas) // bucket_size + 1
            for bucket_id in range(step):
                left_ptr = bucket_id
                while left_ptr < len(metas):
                    tmp_batch = [left_ptr]
                    tokens_remain = self.train_max_token_len - metas[left_ptr]['text_visual_tokens']
                    left_ptr += step
                    while (left_ptr < len(metas)) and (metas[left_ptr]['text_visual_tokens'] <= tokens_remain):
                        if not has_been_used[left_ptr]:
                            has_been_used[left_ptr] = True
                            tokens_remain -= metas[left_ptr]['text_visual_tokens']
                            tmp_batch.append(left_ptr)
                        left_ptr += step
                    tmp_ptr = left_ptr + step
                    while tmp_ptr < len(metas) and tokens_remain > 0:
                        if (not has_been_used[tmp_ptr]) and (metas[tmp_ptr]['text_visual_tokens'] <= tokens_remain):
                            has_been_used[tmp_ptr] = True
                            tokens_remain -= metas[tmp_ptr]['text_visual_tokens']
                            tmp_batch.append(tmp_ptr)
                        tmp_ptr += step

                    # 从剩余 token 预算还能装下的样本中继续补样本，尽量把序列塞满。
                    if tokens_remain > 0:
                        increase_seq_usage_times = 0
                        while increase_seq_usage_times == 0 or (tokens_remain > self.max_text_len):
                            increase_seq_usage_times += 1
                            if increase_seq_usage_times >= 3: break
                            select_map = {}
                            for ind in tmp_batch:
                                select_map[ind] = True
                            candidates = []
                            min_val = 99999999
                            for tmp_ind in range(bucket_id, len(metas), step):
                                if (metas[tmp_ind]['cum_text_visual_tokens'][0] <= tokens_remain) and (tmp_ind not in select_map):
                                    import bisect
                                    idx = bisect.bisect_right(metas[tmp_ind]['cum_text_visual_tokens'], tokens_remain)
                                    if tokens_remain - metas[tmp_ind]['cum_text_visual_tokens'][idx-1] < min_val:
                                        min_val = tokens_remain - metas[tmp_ind]['cum_text_visual_tokens'][idx-1]
                                        candidates = [tmp_ind]
                                    elif tokens_remain - metas[tmp_ind]['cum_text_visual_tokens'][idx-1] == min_val:
                                        candidates.append(tmp_ind)
                            if len(candidates):
                                tmp_batch.append(self.epoch_rank_generator.choice(candidates))
                                tokens_remain = min_val
                            else:
                                break
                    batches.append(tmp_batch)
                    if len(batches) % 1000 == 0:
                        print(f'已组成 {len(batches)} 个 batches，left_ptr={left_ptr}, len(metas)={len(metas)}')
        batch_num = len(batches)
        print(f'[data preprocess] form_batches 完成，得到 {len(batches)} 个 batches，耗时 {time.time()-st:.2f}s')
        try:
            if self.num_replicas > 1:
                batch_num = torch.tensor([batch_num], device=self.device)
                if tdist.is_initialized():
                    tdist.all_reduce(batch_num, op=tdist.ReduceOp.MIN)
                batch_num = batch_num.item()
        except Exception as e:
            print(e)
        batch_num = batch_num // self.dataloader_workers * self.dataloader_workers
        if batch_num == 0 and len(batches) > 0:
            # 本地 shard 太小时，也至少让每个 worker 迭代一次；
            # `__iter__` 会用取模方式重复利用 batch。
            batch_num = self.dataloader_workers
        print(f'[data preprocess] form_batches 完成，最终使用 {batch_num} 个 batches')
        return batches, batch_num

    def set_global_worker_id(self):
        """根据 DataLoader worker 信息，计算当前全局 worker 编号。"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id
        else:
            worker_id = 0
            worker_total_num = 1
        assert worker_total_num == self.dataloader_workers, print(worker_total_num, self.dataloader_workers)
        self.worker_id = worker_id
        self.global_worker_id = self.rank * self.dataloader_workers + worker_id

    def set_epoch(self, epoch):
        """更新 epoch，并据此重建随机数生成器。"""
        self.epoch = epoch
        self.set_generator()

    def set_generator(self, ):
        """分别为 epoch / worker / rank 级别创建独立随机数生成器。"""
        self.epoch_generator = np.random.default_rng(self.seed + self.epoch)
        self.epoch_worker_generator = np.random.default_rng(self.seed + self.epoch + self.worker_id)
        self.epoch_global_worker_generator = np.random.default_rng(self.seed + self.epoch + self.global_worker_id)
        self.epoch_rank_generator = np.random.default_rng(self.seed + self.epoch + self.rank)

    def __iter__(self):
        """按 worker 切分 batch，逐个产出模型训练所需的字典输入。"""
        self.set_global_worker_id()
        self.set_generator()
        if int(getattr(self.other_args, "shuffle_batches", 1) or 0) == 1:
            self.epoch_rank_generator.shuffle(self.batches)
        yield_data_cnt = 0
        batch_ind_ptr = self.worker_id
        failed_batch_cnt = 0
        last_yield_data_time = time.time()
        while yield_data_cnt < self.batch_nums // self.dataloader_workers:
            # 代码/形状说明：if True:
            try:
                if time.time() - last_yield_data_time > 600:
                    raise ValueError(f'[dataset] 产出数据耗时过长，请检查数据读取代码')
                batch_inds = self.batches[batch_ind_ptr%len(self.batches)]
                if self.other_args.cache_check_mode in [-2, 2, 3]: # 这些模式要求在每次迭代时重新检查缓存状态。
                    all_has_been_cached = True
                    all_has_not_been_cached = True
                    for j in batch_inds:
                        exist_status = self.exists_cache_file(self.metas[j])
                        if exist_status:
                            all_has_not_been_cached = False
                        if not exist_status:
                            all_has_been_cached = False
                    if self.other_args.cache_check_mode == 2: # batch 内所有样本都已经缓存后才允许使用。
                        if not all_has_been_cached:
                            batch_ind_ptr += self.dataloader_workers
                            continue
                    if self.other_args.cache_check_mode == -2: # batch 内不能全都已经缓存，否则跳过。
                        if all_has_been_cached:
                            batch_ind_ptr += self.dataloader_workers
                            # 代码/形状说明：print(f"skipping batch_inds {batch_inds}")
                            continue
                    if self.other_args.cache_check_mode == 3: # 至少有一个样本已缓存时才使用。
                        if all_has_not_been_cached:
                            batch_ind_ptr += self.dataloader_workers
                            continue

                batch_data = []
                for j in batch_inds:
                    meta = self.metas[j]
                    if 'image_path' in meta:
                        ret, model_input = self.prepare_image_input(meta)
                    elif 'video_path' in meta:
                        ret, model_input = self.prepare_video_input(meta)
                    # 代码/形状说明：if not ret: break
                    if ret:
                        batch_data.append(model_input)
                if not len(batch_data):
                    batch_ind_ptr += self.dataloader_workers
                    continue
                    # 中文说明：raise ValueError(f'[dataset] prepare_video_input failed, continue, failed meta is {meta}')

                captions4images, captions4raw_features, images, raw_features_bcthw, feature_cache_files4images, text_features  = [], [], [], [], [], []
                traj_ids = []
                hybrid_roles = []
                grpo_rewards, grpo_old_logprobs = [], []
                grpo_adv_finals = []
                grpo_reward_acts, grpo_reward_tasks = [], []
                grpo_reward_task_raws = []
                grpo_reward_task_dense_raws = []
                grpo_reward_task_success_raws = []
                grpo_succs = []
                grpo_succ_trajs = []
                grpo_task_final_costs = []
                grpo_task_final_pos_errs = []
                grpo_task_final_yaw_errs = []
                grpo_reward_ces = []
                grpo_ref_logprobs = []
                grpo_group_ids, grpo_clip_ids = [], []
                grpo_trace_files = []
                text_feature_cache_files = []
                addition_pn_images = {}
                for item in batch_data:
                    if item['raw_features_cthw'] is None:
                        images.append(item['img_T3HW'].permute(1,0,2,3)) # 代码/形状说明：# tchw -> cthw
                        for key in item:
                            if key.startswith('img_T3HW_'):
                                if key not in addition_pn_images:
                                    addition_pn_images[key] = []
                                addition_pn_images[key].append(item[key].permute(1,0,2,3))
                        feature_cache_files4images.append(item['feature_cache_file'])
                        captions4images.append(item['text_input'])
                        grpo_rewards.append(float(item.get('grpo_reward', 0.0)))
                        grpo_old_logprobs.append(float(item.get('grpo_old_logprob', 0.0)))
                        grpo_adv_finals.append(float(item.get('grpo_adv_final', item.get('grpo_weight', 0.0))))
                        # 中文说明：Prefer per-level LOO advantage if present (clip-GRPO), else fall back to raw.
                        grpo_reward_acts.append(float(item.get('grpo_reward_act_adv', item.get('grpo_reward_act', item.get('grpo_reward', 0.0)))))
                        grpo_reward_tasks.append(float(item.get('grpo_reward_task', item.get('grpo_reward_task_adv', 0.0))))
                        grpo_reward_task_raws.append(float(item.get('grpo_reward_task_raw', item.get('grpo_reward_task', 0.0))))
                        grpo_reward_task_dense_raws.append(float(item.get('grpo_reward_task_dense_raw', item.get('grpo_reward_task_raw', item.get('grpo_reward_task', 0.0)))))
                        grpo_reward_task_success_raws.append(float(item.get('grpo_reward_task_success_raw', 0.0)))
                        grpo_succs.append(float(item.get('grpo_succ', 0.0)))
                        grpo_succ_trajs.append(float(item.get('grpo_succ_traj', item.get('grpo_succ', 0.0))))
                        grpo_task_final_costs.append(float(item.get('grpo_task_final_cost', 0.0)))
                        grpo_task_final_pos_errs.append(float(item.get('grpo_task_final_pos_err_m', 0.0)))
                        grpo_task_final_yaw_errs.append(float(item.get('grpo_task_final_yaw_err_deg', 0.0)))
                        grpo_reward_ces.append(float(item.get('grpo_reward_ce_adv', item.get('grpo_reward_ce', item.get('grpo_reward_ce_raw', 0.0)))))
                        grpo_ref_logprobs.append(float(item.get('grpo_ref_logprob', item.get('grpo_old_logprob', 0.0))))
                        grpo_group_ids.append(str(item.get('grpo_group_id', '')))
                        grpo_clip_ids.append(int(item.get('grpo_clip_id', 1)))
                        grpo_trace_files.append(item.get('grpo_trace_files', []))
                    else:
                        raw_features_bcthw.append(item['raw_features_cthw'])
                        captions4raw_features.append(item['text_input'])
                        grpo_rewards.append(float(item.get('grpo_reward', 0.0)))
                        grpo_old_logprobs.append(float(item.get('grpo_old_logprob', 0.0)))
                        grpo_adv_finals.append(float(item.get('grpo_adv_final', item.get('grpo_weight', 0.0))))
                        # 中文说明：Prefer per-level LOO advantage if present (clip-GRPO), else fall back to raw.
                        grpo_reward_acts.append(float(item.get('grpo_reward_act_adv', item.get('grpo_reward_act', item.get('grpo_reward', 0.0)))))
                        grpo_reward_tasks.append(float(item.get('grpo_reward_task', item.get('grpo_reward_task_adv', 0.0))))
                        grpo_reward_task_raws.append(float(item.get('grpo_reward_task_raw', item.get('grpo_reward_task', 0.0))))
                        grpo_reward_task_dense_raws.append(float(item.get('grpo_reward_task_dense_raw', item.get('grpo_reward_task_raw', item.get('grpo_reward_task', 0.0)))))
                        grpo_reward_task_success_raws.append(float(item.get('grpo_reward_task_success_raw', 0.0)))
                        grpo_succs.append(float(item.get('grpo_succ', 0.0)))
                        grpo_succ_trajs.append(float(item.get('grpo_succ_traj', item.get('grpo_succ', 0.0))))
                        grpo_task_final_costs.append(float(item.get('grpo_task_final_cost', 0.0)))
                        grpo_task_final_pos_errs.append(float(item.get('grpo_task_final_pos_err_m', 0.0)))
                        grpo_task_final_yaw_errs.append(float(item.get('grpo_task_final_yaw_err_deg', 0.0)))
                        grpo_reward_ces.append(float(item.get('grpo_reward_ce_adv', item.get('grpo_reward_ce', item.get('grpo_reward_ce_raw', 0.0)))))
                        grpo_ref_logprobs.append(float(item.get('grpo_ref_logprob', item.get('grpo_old_logprob', 0.0))))
                        grpo_group_ids.append(str(item.get('grpo_group_id', '')))
                        grpo_clip_ids.append(int(item.get('grpo_clip_id', 1)))
                        grpo_trace_files.append(item.get('grpo_trace_files', []))
                    text_feature_cache_files.append(item['text_feature_cache_file'])
                    traj_ids.append(str(item.get('traj_id', '')))
                    hybrid_roles.append(str(item.get('hybrid_role', 'grpo')))
                captions = captions4images + captions4raw_features
                assert len(batch_data), f'len(batch_data)={len(batch_data)}'
                text_cond_tuple = None
                yield {
                    'captions': captions,
                    'images': images,
                    'addition_pn_images': addition_pn_images,
                    'feature_cache_files4images': feature_cache_files4images,
                    'raw_features_bcthw': raw_features_bcthw,
                    'text_cond_tuple': text_cond_tuple,
                    'text_feature_cache_files': text_feature_cache_files,
                    'traj_ids': traj_ids,
                    'hybrid_roles': hybrid_roles,
                    'grpo_rewards': grpo_rewards,
                    'grpo_old_logprobs': grpo_old_logprobs,
                    'grpo_adv_finals': grpo_adv_finals,
                    'grpo_reward_acts': grpo_reward_acts,
                    'grpo_reward_tasks': grpo_reward_tasks,
                    'grpo_reward_task_raws': grpo_reward_task_raws,
                    'grpo_reward_task_dense_raws': grpo_reward_task_dense_raws,
                    'grpo_reward_task_success_raws': grpo_reward_task_success_raws,
                    'grpo_succs': grpo_succs,
                    'grpo_succ_trajs': grpo_succ_trajs,
                    'grpo_task_final_costs': grpo_task_final_costs,
                    'grpo_task_final_pos_errs': grpo_task_final_pos_errs,
                    'grpo_task_final_yaw_errs': grpo_task_final_yaw_errs,
                    'grpo_reward_ces': grpo_reward_ces,
                    'grpo_ref_logprobs': grpo_ref_logprobs,
                    'grpo_group_ids': grpo_group_ids,
                    'grpo_clip_ids': grpo_clip_ids,
                    'grpo_trace_files': grpo_trace_files,
                    'media': 'videos',
                }
                yield_data_cnt += 1
                batch_ind_ptr += self.dataloader_workers
                del batch_data
                del images
                del captions
                last_yield_data_time = time.time()
            except Exception as e:
                batch_ind_ptr += self.dataloader_workers
                failed_batch_cnt += 1
                if failed_batch_cnt % 400 == 0:
                    print(f'failed_batch_cnt: {failed_batch_cnt}, yield_data_cnt: {yield_data_cnt}')
                print(f'[dataset] 错误：{e}')

    def prepare_image_input(self, info) -> Tuple:
        """读取单张图像样本，并返回 trainer 期望的标准字段。"""
        try:
            img_path, text_input = osp.abspath(info['image_path']), info['caption']
            img_T3HW, raw_features_cthw, feature_cache_file, text_features_lenxdim, text_feature_cache_file = [None] * 5
            # 代码/形状说明：text_input = process_short_text(text_input)
            if self.use_text_token_cache:
                text_feature_cache_file = osp.join(self.token_cache_dir, 'flan-t5-xl-official', get_prompt_id(text_input)+'.pt')
                if osp.exists(text_feature_cache_file):
                    text_features_lenxdim = torch.load(text_feature_cache_file, weights_only=True)

            if self.add_motion_score2caption:
                rand_motion_score = -1 + self.epoch_rank_generator.random() * 21.0 # -1.0 ~ 20.0
                text_input = prepend_motion_score(text_input, rand_motion_score)
            if self.use_vae_token_cache:
                feature_cache_file = self.get_image_cache_file(img_path)
                if osp.exists(feature_cache_file):
                    try:
                        raw_features_cthw = torch.load(feature_cache_file, weights_only=True)
                    except Exception as e:
                        print(f'加载 cache 文件失败：{e}')
                        os.remove(feature_cache_file)
                if raw_features_cthw is None and (not self.allow_online_vae_feature_extraction):
                    return False, None
            if raw_features_cthw is None:
                with open(img_path, 'rb') as f:
                    img: PImage.Image = PImage.open(f)
                    w, h = img.size
                    # 必要时可强制指定宽高比模板，例如固定走 9:16 裁剪路径。
                    forced_tpl = info.get("force_h_div_w_template", None)
                    if forced_tpl is not None:
                        h_div_w_template = float(forced_tpl)
                    else:
                        h_div_w = h / w
                        h_div_w_template = self.h_div_w_templates[np.argmin(np.abs((h_div_w-self.h_div_w_templates)))]
                    # 实际训练只允许使用离散模板，因此要吸附到最近的模板键。
                    h_div_w_template = self.h_div_w_templates[np.argmin(np.abs((h_div_w_template-self.h_div_w_templates)))]
                    tgt_h, tgt_w = self.dynamic_resolution_h_w[h_div_w_template][self.pn]['pixel']
                    img = img.convert('RGB')
                    img_T3HW = transform(img, tgt_h, tgt_w)
                    img_T3HW = img_T3HW.unsqueeze(0)
                    assert img_T3HW.shape[1] == 3
            data_item = {
                'text_input': text_input,
                'img_T3HW': img_T3HW,
                'raw_features_cthw': raw_features_cthw,
                'feature_cache_file': feature_cache_file,
                'text_features_lenxdim': text_features_lenxdim,
                'text_feature_cache_file': text_feature_cache_file,
                # 下面这些字段给 GRPO / hybrid 训练路径使用。
                'traj_id': str(info.get('traj_id', '')),
                'hybrid_role': str(info.get('hybrid_role', 'grpo')),
                'grpo_reward': float(info.get('grpo_reward', 0.0)),
                'grpo_old_logprob': float(info.get('grpo_old_logprob', 0.0)),
                'grpo_adv_final': float(info.get('grpo_adv_final', info.get('grpo_weight', 0.0))),
                'grpo_reward_act': float(info.get('grpo_reward_act', info.get('grpo_reward', 0.0))),
                'grpo_reward_task': float(info.get('grpo_reward_task', 0.0)),
                'grpo_reward_task_raw': float(info.get('grpo_reward_task_raw', info.get('grpo_reward_task', 0.0))),
                'grpo_reward_task_dense_raw': float(info.get('grpo_reward_task_dense_raw', info.get('grpo_reward_task_raw', info.get('grpo_reward_task', 0.0)))),
                'grpo_reward_task_success_raw': float(info.get('grpo_reward_task_success_raw', 0.0)),
                'grpo_succ': float(info.get('grpo_succ', 0.0)),
                'grpo_succ_traj': float(info.get('grpo_succ_traj', info.get('grpo_succ', 0.0))),
                'grpo_task_final_cost': float(info.get('grpo_task_final_cost', 0.0)),
                'grpo_task_final_pos_err_m': float(info.get('grpo_task_final_pos_err_m', 0.0)),
                'grpo_task_final_yaw_err_deg': float(info.get('grpo_task_final_yaw_err_deg', 0.0)),
                'grpo_reward_ce_raw': float(info.get('grpo_reward_ce_raw', info.get('grpo_reward_ce', 0.0))),
                'grpo_reward_ce_adv': float(info.get('grpo_reward_ce_adv', 0.0)),
                'grpo_ref_logprob': float(info.get('grpo_ref_logprob', info.get('grpo_old_logprob', 0.0))),
                'grpo_group_id': str(info.get('grpo_group_id', '')),
                'grpo_clip_id': int(info.get('grpo_clip_id', 1)),
                'grpo_trace_files': (
                    info.get('grpo_trace_files', [])
                    if isinstance(info.get('grpo_trace_files', []), list)
                    else ([info.get('grpo_trace_files')] if info.get('grpo_trace_files') else [])
                ),
            }
            return True, data_item
        except Exception as e:
            print(f'prepare_image_input 失败：{e}')
            return False, None

    def prepare_pair_image_input(self, info) -> Tuple:
        """成对图像输入的占位入口，当前实现尚未补齐。"""
        pass

    def prepare_pair_video_input(self, info) -> Tuple:
        """读取胜负成对视频，并把两段视频堆成一个二选一输入。"""
        tmp_info = copy.deepcopy(info)
        tmp_info['video_path'] = info['win_video_path']
        win_flag, win_data_item = self.prepare_video_input(tmp_info)
        assert win_data_item['raw_features_cthw'] is None

        tmp_info['video_path'] = info['lose_video_path']
        lose_flag, lose_data_item = self.prepare_video_input(tmp_info)
        assert lose_data_item['raw_features_cthw'] is None

        flag = win_flag and lose_flag
        img_T3HW = torch.stack([win_data_item['img_T3HW'], lose_data_item['img_T3HW']], dim=0) # 配对视频堆成 `[2, T, C, H, W]`。
        win_data_item['img_T3HW'] = img_T3HW
        return flag, win_data_item

    def prepare_video_input(self, info) -> Tuple:
        """读取单段视频样本，按采样策略解码帧并准备缓存字段。"""
        filename, begin_frame_id, end_frame_id = (
            info["video_path"],
            info["begin_frame_id"],
            info["end_frame_id"],
        )

        if True:
        # try:
            img_T3HW, raw_features_cthw, feature_cache_file, text_features_lenxdim, text_feature_cache_file = None, None, None, None, None
            img_T3HW_4additional_pn = {}
            text_input = info['caption']
            if '/vdataset/clip' in filename: # 某些 clip 数据集路径已经截断过起点，需要重置局部帧号。
                begin_frame_id, end_frame_id = 0, end_frame_id - begin_frame_id
            # 若预处理已经给出精确帧索引，就严格按这些帧解码。
            frame_idxs = info.get("frame_idxs", None)
            if frame_idxs is not None:
                sample_frames = len(frame_idxs)
            else:
                sample_frames = info['sample_frames']
            if self.use_vae_token_cache:
                feature_cache_file = self.get_video_cache_file(info["video_path"], begin_frame_id, end_frame_id, self.video_fps)
                if osp.exists(feature_cache_file):
                    try:
                        pt = (sample_frames-1) // self.temporal_compress_rate + 1
                        raw_features_cthw = torch.load(feature_cache_file, weights_only=True)
                        # 代码/形状说明：_, tgt_h, tgt_w = self.dynamic_resolution_h_w[h_div_w_template][self.pn]['pt2scale_schedule'][1][-1]
                        # 代码/形状说明：assert raw_features_cthw.shape[-2:] == (tgt_h, tgt_w), f'raw_features_cthw.shape[-2:] == (tgt_h, tgt_w): {raw_features_cthw.shape[-2:]} vs {(tgt_h, tgt_w)}'
                        assert raw_features_cthw.shape[1] >= pt, f'raw_features_cthw.shape[1] 应 >= pt：{raw_features_cthw.shape[1]} vs {pt}'
                        if raw_features_cthw.shape[1] > pt:
                            raw_features_cthw = raw_features_cthw[:,:pt]
                    except Exception as e:
                        print(f'加载视频 cache 文件失败：{e}')
                        os.remove(feature_cache_file)
                        raw_features_cthw = None
                if raw_features_cthw is None and (not self.allow_online_vae_feature_extraction):
                    return False, None
            pn_list = [self.pn]
            if raw_features_cthw is None:
                local_path = info["video_path"]
                if not local_path: return False, None
                if not osp.exists(local_path):
                    return False, None
                video = EncodedVideoOpencv(local_path, os.path.basename(local_path), num_threads=0)
                # 采样策略：
                # 1. `duration`：从起点开始按 `video_fps` 取前 `sample_frames` 帧。
                # 2. `fixed_full/segment_full/...`：在整段标注区间内均匀采样。
                # 片段边界优先使用元数据中的 fps，通常比容器平均 fps 更准。
                meta_fps = float(info.get("fps", video._fps)) if info.get("fps", None) is not None else float(video._fps)
                meta_fps = max(1e-6, meta_fps)
                start_interval = max(0.0, float(begin_frame_id) / meta_fps)
                sample_mode = getattr(self.other_args, "video_sample_mode", "duration")
                if sample_mode in ("fixed_full", "segment_full", "segment_full_floor_sec"):
                    # 结束时间跟随标注片段末尾，确保整段内容都可能被采到。
                    # 很多数据集把 `end_frame_id` 当成闭区间，所以这里用 `end + 1`。
                    end_frame_exclusive = float(end_frame_id) + 1.0
                    end_interval = max(start_interval + 1.0 / meta_fps, end_frame_exclusive / meta_fps)
                else:
                    end_interval = start_interval + (sample_frames - 1) / self.video_fps
                assert end_interval <= video.duration + 0.2, f'{end_interval=}，但 {video.duration=}' # 中文说明：0.2s margin
                end_interval = min(end_interval, video.duration)
                if frame_idxs is not None:
                    # `frame_idxs` 是容器内的绝对帧号。
                    raw_video, _ = video.get_frames(frame_idxs)
                else:
                    raw_video, _ = video.get_clip(start_interval, end_interval, sample_frames)
                h, w, _ = raw_video[0].shape
                forced_tpl = info.get("force_h_div_w_template", None)
                if forced_tpl is not None:
                    h_div_w_template = float(forced_tpl)
                else:
                    h_div_w = h / w
                    h_div_w_template = self.h_div_w_templates[np.argmin(np.abs((h_div_w-self.h_div_w_templates)))]
                # 训练只允许离散宽高比模板，因此要吸附到最近模板。
                h_div_w_template = self.h_div_w_templates[np.argmin(np.abs((h_div_w_template-self.h_div_w_templates)))]
                tgt_h, tgt_w = self.dynamic_resolution_h_w[h_div_w_template][self.pn]['pixel']

                for addition_pn in self.addition_pn_list:
                    pn_list = pn_list + [addition_pn]
                for pn in pn_list:
                    if isinstance(video, EncodedVideoDecord):
                        img_T3HW = [transform(Image.fromarray(frame).convert("RGB"), tgt_h, tgt_w) for frame in raw_video]
                    else:
                        img_T3HW = [transform(Image.fromarray(frame[:,:,::-1]), tgt_h, tgt_w) for frame in raw_video]
                    img_T3HW = torch.stack(img_T3HW, 0)
                    img_T3HW_4additional_pn[pn] = img_T3HW
                del video
                assert img_T3HW.shape[1] == 3
            data_item = {
                'text_input': text_input,
                'img_T3HW': img_T3HW_4additional_pn.get(self.pn, None),
                'raw_features_cthw': raw_features_cthw,
                'feature_cache_file': feature_cache_file,
                'text_features_lenxdim': text_features_lenxdim,
                'text_feature_cache_file': text_feature_cache_file,
                # 下面这些字段给 GRPO / hybrid 训练路径使用。
                'traj_id': str(info.get('traj_id', '')),
                'hybrid_role': str(info.get('hybrid_role', 'grpo')),
                'grpo_reward': float(info.get('grpo_reward', 0.0)),
                'grpo_old_logprob': float(info.get('grpo_old_logprob', 0.0)),
                'grpo_adv_final': float(info.get('grpo_adv_final', info.get('grpo_weight', 0.0))),
                'grpo_reward_act': float(info.get('grpo_reward_act', info.get('grpo_reward', 0.0))),
                'grpo_reward_task': float(info.get('grpo_reward_task', 0.0)),
                'grpo_reward_task_raw': float(info.get('grpo_reward_task_raw', info.get('grpo_reward_task', 0.0))),
                'grpo_reward_task_dense_raw': float(info.get('grpo_reward_task_dense_raw', info.get('grpo_reward_task_raw', info.get('grpo_reward_task', 0.0)))),
                'grpo_reward_task_success_raw': float(info.get('grpo_reward_task_success_raw', 0.0)),
                'grpo_succ': float(info.get('grpo_succ', 0.0)),
                'grpo_succ_traj': float(info.get('grpo_succ_traj', info.get('grpo_succ', 0.0))),
                'grpo_task_final_cost': float(info.get('grpo_task_final_cost', 0.0)),
                'grpo_task_final_pos_err_m': float(info.get('grpo_task_final_pos_err_m', 0.0)),
                'grpo_task_final_yaw_err_deg': float(info.get('grpo_task_final_yaw_err_deg', 0.0)),
                # 可选的额外奖励层级信息。
                'grpo_reward_ce_raw': float(info.get('grpo_reward_ce_raw', info.get('grpo_reward_ce', 0.0))),
                'grpo_reward_ce_adv': float(info.get('grpo_reward_ce_adv', 0.0)),
                # 参考策略的 logprob，用来算 selected-token KL。
                'grpo_ref_logprob': float(info.get('grpo_ref_logprob', info.get('grpo_old_logprob', 0.0))),
                'grpo_group_id': str(info.get('grpo_group_id', '')),
                'grpo_clip_id': int(info.get('grpo_clip_id', 1)),
                # 严格 GRPO 需要逐样本 trace 文件，以便回放 token 轨迹并重算 logprob。
                'grpo_trace_files': (
                    info.get('grpo_trace_files', [])
                    if isinstance(info.get('grpo_trace_files', []), list)
                    else ([info.get('grpo_trace_files')] if info.get('grpo_trace_files') else [])
                ),
            }
            for pn in pn_list[1:]:
                data_item.update({f'img_T3HW_{pn}': img_T3HW_4additional_pn.get(pn, None)})
            return True, data_item
        # 代码/形状说明：except Exception as e:
        # 中文说明：# print(f'prepare_video_input error: {e}, info: {info}')
        # 代码/形状说明：return False, None
        # 中文说明：finally:
        #     try:
        # 代码/形状说明：if (img_T3HW is not None) and local_path and (local_path != filename):
        # 中文说明：os.remove(local_path)
        # 代码/形状说明：except Exception as e:
        # 代码/形状说明：print(f'delete local_path: {local_path} error: {e}, info: {info}')

    @staticmethod
    def collate_function(batch, online_t5: bool = False) -> None:
        """保留给 DataLoader 的静态拼接入口，当前未实现。"""
        pass

    def random_drop_sentences(self, caption):
        """随机截断 caption 的句子数，模拟短提示词训练。"""
        elems = [item for item in caption.split('.') if item]
        if len(elems) < 2:
            return caption
        sentences = self.epoch_global_worker_generator.integers(1, len(elems)+1)
        return '.'.join(elems[:sentences]) + '.'

    def get_text_input(self, long_text_input, short_text_input, long_text_type):
        """在长短文本提示之间做采样，控制训练时的 prompt 长度分布。"""
        assert long_text_input or short_text_input
        if not long_text_input:
            return short_text_input
        if not short_text_input:
            return long_text_input
        random_value = self.epoch_global_worker_generator.random()
        assert not self.enable_dynamic_length_prompt
        if self.enable_dynamic_length_prompt and long_text_type != 'user_prompt':
            long_text_elems = [item for item in long_text_input.split('.') if item]
            if len(long_text_elems):
                first_sentence_words = [item for item in long_text_elems[0].split(' ') if item]
            else:
                first_sentence_words = 0
            if len(first_sentence_words) >= 15:
                num_sentence4short_text = 1
            else:
                num_sentence4short_text = 2
            if not short_text_input:
                short_text_input = '.'.join(long_text_elems[:num_sentence4short_text])
            if random_value < self.short_prob:
                return short_text_input
            if len(long_text_elems) <= num_sentence4short_text:
                return long_text_input
            select_sentence_num = self.epoch_global_worker_generator.integers(num_sentence4short_text+1, len(long_text_elems)+1)
            return '.'.join(long_text_elems[:select_sentence_num])
        else:
            if random_value < self.short_prob:
                return short_text_input
            return long_text_input

    def __len__(self):
        """返回当前 epoch 预生成的 batch 数。"""
        return self.batch_nums

    def get_image_cache_file(self, image_path):
        """根据图片路径生成对应的 VAE token 缓存文件路径。"""
        elems = image_path.split('/')
        elems = [item for item in elems if item]
        filename, ext = osp.splitext(elems[-1])
        filename = get_prompt_id(filename)
        save_filepath = osp.join(self.token_cache_dir, f'images_pn_{self.pn}', '/'.join(elems[4:-1]), f'{filename}.pt')
        return save_filepath

    def get_video_cache_file(self, video_path, begin_frame_id, end_frame_id, video_fps):
        """根据视频路径与采样区间生成缓存文件路径。"""
        elems = video_path.split('/')
        elems = [item for item in elems if item]
        filename, ext = osp.splitext(elems[-1])
        filename = get_prompt_id(filename)
        save_filepath = osp.join(self.token_cache_dir, f'pn_{self.pn}_sample_fps_{video_fps}', '/'.join(elems[4:-1]), f'{filename}_sf_{begin_frame_id}_ef_{end_frame_id}.pt')
        return save_filepath

if __name__ == '__main__':
    pass
