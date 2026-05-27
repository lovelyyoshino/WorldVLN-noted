# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.amp import autocast

from einops import rearrange, repeat, reduce, pack, unpack

from typing import Callable

def exists(val):
    """返回值是否不是 `None`。"""
    return val is not None

def default(val, d):
    """若 `val` 存在则返回它，否则返回默认值 `d`。"""
    return val if exists(val) else d

def noop(*args, **kwargs):
    """空操作回调，常用作可选 all-reduce / hook 的占位函数。"""
    pass

def identity(t):
    """恒等映射。"""
    return t

def l2norm(t):
    """沿最后一维做 L2 归一化。"""
    return F.normalize(t, p = 2, dim = -1)

def cdist(x, y):
    """按批量计算欧氏距离矩阵。"""
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min = 0).sqrt()

def log(t, eps = 1e-20):
    """带数值保护的对数。"""
    return torch.log(t.clamp(min = eps))

def ema_inplace(old, new, decay):
    """原地更新指数滑动平均：`old = decay * old + (1 - decay) * new`。"""
    is_mps = str(old.device).startswith('mps:')

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))

def pack_one(t, pattern):
    """对单个张量执行 `einops.pack`。"""
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    """把 `pack_one` 的结果按原始 shape 还原。"""
    return unpack(t, ps, pattern)[0]

def uniform_init(*shape):
    """用 Kaiming uniform 初始化一个临时张量。"""
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def gumbel_noise(t):
    """采样 Gumbel(0, 1) 噪声，常用于 Gumbel-Softmax。"""
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    reinmax = False,
    dim = -1,
    training = True
):
    """执行 Gumbel-Softmax / argmax 采样。

    返回 `(indices, one_hot)`。
    当 `straight_through=True` 时，前向使用离散 one-hot，反向近似使用 softmax 梯度。
    """
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (reinmax and not straight_through), 'reinmax 只能在启用 straight through gumbel softmax 时打开'

    if not straight_through or temperature <= 0. or not training:
        return ind, one_hot

    # 代码/形状说明：使用 reinmax 提高二阶精度 - https://arxiv.org/abs/2304.08612
    # 算法 2。

    if reinmax:
        π0 = logits.softmax(dim = dim)
        π1 = (one_hot + (logits / temperature).softmax(dim = dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim = 1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim = dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot

def laplace_smoothing(x, n_categories, eps = 1e-5, dim = -1):
    """做 Laplace smoothing，避免零计数导致概率和除法不稳定。"""
    denom = x.sum(dim = dim, keepdim = True)
    return (x + eps) / (denom + n_categories * eps)

def sample_vectors(samples, num):
    """从向量集合中随机采样 `num` 个样本，不足时允许重复采样。"""
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_sample_vectors(samples, num):
    """对 batch 中每个样本集合分别随机采样。"""
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

def pad_shape(shape, size, dim = 0):
    """仅替换某一维尺寸，生成用于补齐的目标 shape。"""
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_multinomial(total_count, probs):
    """按给定概率采样多项式分布计数，返回每个类别分到的样本数。"""
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)

def all_gather_sizes(x, dim):
    """在分布式环境收集各 rank 指定维度上的长度。"""
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim = 0):
    """收集不同 rank 上长度不一致的张量。"""
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def sample_vectors_distributed(local_samples, num):
    """在分布式环境下按全局比例采样向量，并通过 all-gather 拼回。"""
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    out = torch.cat(all_samples, dim = 0)

    return rearrange(out, '... -> 1 ...')

def batched_bincount(x, *, minlength):
    """对 batch 中每一行独立做 bincount。"""
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    """对多个 codebook 并行执行 k-means 初始化。

    当 `use_cosine_sim=True` 时，分配步骤基于 cosine similarity；
    否则使用欧氏距离。分布式场景可以通过 `all_reduce_fn` 同步统计量。
    """
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -cdist(samples, means)

        buckets = torch.argmax(dists, dim = -1)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins

def batched_embedding(indices, embeds):
    """批量 gather codebook embedding，返回与 `indices` 对齐的码向量。"""
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d = dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch)
    return embeds.gather(2, indices)

# 正则化损失。

def orthogonal_loss_fn(t):
    """正交正则项。

    先把 codes 归一化，再计算任意两码之间的 cosine similarity，
    惩罚 `cosine_sim^2`，鼓励码向量彼此正交、减少冗余。
    """
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)

# 距离类型。

class EuclideanCodebook(nn.Module):
    """基于欧氏距离的向量量化码本。

    码本支持：

    - 可选 k-means 初始化
    - EMA 更新 `cluster_size` 与 `embed_avg`
    - 死码（dead code）检测与替换
    - 分布式同步采样 / 统计
    """

    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True,
        affine_param = False,
        sync_affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9
    ):
        """初始化欧氏距离码本及其 EMA / k-means 相关状态。"""
        super().__init__()
        self.transform_input = identity

        self.decay = decay
        self.ema_update = ema_update

        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        assert not (use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init 目前不兼容分布式环境中的 multiple codebooks'

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

        # 仿射相关参数。

        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            return

        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay

        self.register_buffer('batch_mean', None)
        self.register_buffer('batch_variance', None)

        self.register_buffer('codebook_mean_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_mean', torch.empty(num_codebooks, 1, dim))
        self.register_buffer('codebook_variance_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_variance', torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        """用 k-means 结果初始化码本；若已初始化则跳过。"""
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        """对指定 buffer 执行指数滑动平均更新。"""
        old_value = getattr(self, buffer_name)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if not exists(old_value) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine(self, data, embed, mask = None):
        """更新 affine 参数所需的 batch / codebook 均值和方差。"""
        assert self.affine_param

        var_fn = partial(torch.var, unbiased = False)

        # 计算 codebook 自身的均值和方差，用于 affine 校准。

        embed = rearrange(embed, 'h ... d -> h (...) d')

        if self.training:
            self.update_with_decay('codebook_mean', reduce(embed, 'h n d -> h 1 d', 'mean'), self.affine_param_codebook_decay)
            self.update_with_decay('codebook_variance', reduce(embed, 'h n d -> h 1 d', var_fn), self.affine_param_codebook_decay)

        # 准备当前 batch 数据；如果有 mask，只统计有效位置。

        data = rearrange(data, 'h ... d -> h (...) d')

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        # 计算当前 batch 的均值和方差。

        if not self.sync_affine_param:
            self.update_with_decay('batch_mean', reduce(data, 'h n d -> h 1 d', 'mean'), self.affine_param_batch_decay)
            self.update_with_decay('batch_variance', reduce(data, 'h n d -> h 1 d', var_fn), self.affine_param_batch_decay)
            return

        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype

        # 记录有效向量数量，后面作为分布式平均的分母。

        num_vectors = torch.tensor([num_vectors], device = device, dtype = dtype)
        distributed.all_reduce(num_vectors)

        # 跨进程汇总后计算全局 batch 均值。

        batch_sum = reduce(data, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(batch_sum)
        batch_mean = batch_sum / num_vectors

        self.update_with_decay('batch_mean', batch_mean, self.affine_param_batch_decay)

        # 中文标题：calculate distributed variance

        variance_numer = reduce((data - batch_mean) ** 2, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(variance_numer)
        batch_variance = variance_numer / num_vectors

        self.update_with_decay('batch_variance', batch_variance, self.affine_param_batch_decay)

    def replace(self, batch_samples, batch_mask):
        """用新采样的向量替换死码，重置其计数与均值缓存。"""
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')

            self.embed.data[ind][mask] = sampled

            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        """检测 EMA 计数过低的死码，并触发替换。"""
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    @autocast("cuda", enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
    ):
        """执行码本查找并在训练时更新 EMA 统计。

        返回：

        - `quantize`: 量化后的 embedding
        - `embed_ind`: 最近邻码索引
        - `dist`: 输入到各个码的距离矩阵
        """
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        dtype = x.dtype
        flatten, ps = pack_one(x, 'h * d')

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        self.init_embed_(flatten, mask = mask)

        if self.affine_param:
            self.update_affine(flatten, self.embed, mask = mask)

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        if self.affine_param:
            codebook_std = self.codebook_variance.clamp(min = 1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min = 1e-5).sqrt()
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

        dist = -cdist(flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)

        embed_ind = unpack_one(embed_ind, ps, 'h *')

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
            quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)

        if self.training and self.ema_update and not freeze_codebook:

            if self.affine_param:
                flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean

            if exists(mask):
                embed_onehot[~mask] = 0.

            cluster_size = embed_onehot.sum(dim = 1)

            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size.data, cluster_size, self.decay)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        dist = unpack_one(dist, ps, 'h * d')

        return quantize, embed_ind, dist

class CosineSimCodebook(nn.Module):
    """基于 cosine similarity 的向量量化码本。"""

    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True
    ):
        """初始化余弦相似度码本及其 EMA 状态。"""
        super().__init__()
        self.transform_input = l2norm

        self.ema_update = ema_update
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        """使用 cosine k-means 结果初始化码本。"""
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim = True,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        """用新采样且已归一化的向量替换死码。"""
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')

            self.embed.data[ind][mask] = sampled
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size
            self.cluster_size.data[ind][mask] = self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        """检测低频死码并执行替换。"""
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    @autocast("cuda", enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
    ):
        """基于 cosine similarity 选择最近码，并在训练时更新 EMA。"""
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        dtype = x.dtype

        flatten, ps = pack_one(x, 'h * d')

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        self.init_embed_(flatten, mask = mask)

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        dist = einsum('h n d, h c d -> h n c', flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)
        embed_ind = unpack_one(embed_ind, ps, 'h *')

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
            quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)

        if self.training and self.ema_update and not freeze_codebook:
            if exists(mask):
                embed_onehot[~mask] = 0.

            bins = embed_onehot.sum(dim = 1)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size.data, bins, self.decay)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)

            self.embed.data.copy_(l2norm(embed_normalized))
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        dist = unpack_one(dist, ps, 'h * d')
        return quantize, embed_ind, dist

# 主类定义。

class VectorQuantize(nn.Module):
    """统一封装的向量量化模块。

    这个类负责把输入整理成 codebook 需要的形状，调用底层码本完成量化，
    再计算 commitment loss、usage/perplexity 等训练统计量。
    支持多头、多码本、欧氏距离 / 余弦相似度、EMA 与可学习码本。
    """

    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim = None,
        heads = 1,
        separate_codebook_per_head = False,
        decay = 0.8,
        eps = 1e-5,
        freeze_codebook = False,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        use_cosine_sim = False,
        threshold_ema_dead_code = 0,
        channel_last = True,
        accept_image_fmap = False,
        commitment_weight = 1.,
        commitment_use_cross_entropy_loss = False,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        stochastic_sample_codes = False,
        sample_codebook_temp = 1.,
        straight_through = False,
        reinmax = False,  # 中文说明：using reinmax for improved straight-through, assuming straight through helps at all
        sync_codebook = None,
        sync_affine_param = False,
        ema_update = True,
        learnable_codebook = False,
        in_place_codebook_optimizer: Callable[..., Optimizer] = None, # 中文标题：Optimizer used to update the codebook embedding if using learnable_codebook
        affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9,
        sync_update_v = 0. # 代码/形状说明：控制同步更新规则 (21) 中乐观/悲观更新的 v，见 https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
    ):
        """初始化向量量化外壳与底层 codebook。"""
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.has_projections = requires_projection

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss # 中文标题：whether to use cross entropy loss to codebook as commitment loss

        self.learnable_codebook = learnable_codebook

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        assert not (ema_update and learnable_codebook), 'learnable codebook 与 EMA update 不兼容'

        assert 0 <= sync_update_v <= 1.
        assert not (sync_update_v > 0. and not learnable_codebook), '必须开启 learnable codebook'

        self.sync_update_v = sync_update_v

        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic = stochastic_sample_codes,
            reinmax = reinmax,
            straight_through = straight_through
        )

        if not exists(sync_codebook):
            sync_codebook = distributed.is_initialized() and distributed.get_world_size() > 1

        codebook_kwargs = dict(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sync_kmeans = sync_kmeans,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            use_ddp = sync_codebook,
            learnable_codebook = has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp = sample_codebook_temp,
            gumbel_sample = gumbel_sample_fn,
            ema_update = ema_update
        )

        if affine_param:
            assert not use_cosine_sim, 'affine param 只兼容 euclidean codebook'
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param = True,
                sync_affine_param = sync_affine_param,
                affine_param_batch_decay = affine_param_batch_decay,
                affine_param_codebook_decay = affine_param_codebook_decay,
            )

        self._codebook = codebook_class(**codebook_kwargs)

        self.in_place_codebook_optimizer = in_place_codebook_optimizer(self._codebook.parameters()) if exists(in_place_codebook_optimizer) else None

        self.codebook_size = codebook_size
        self.register_buffer('codebook_usage', torch.zeros(codebook_size))
        self.call_cnt = 0

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        """返回当前可见的 codebook 张量。"""
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    @codebook.setter
    def codebook(self, codes):
        """直接替换底层 codebook 参数。"""
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, '... -> 1 ...')

        self._codebook.embed.copy_(codes)

    def get_codes_from_indices(self, indices):
        """根据索引查回码向量，支持单头和多头两种布局。"""
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, '... h d -> ... (h d)')

        indices, ps = pack_one(indices, 'b * h')
        indices = rearrange(indices, 'b n h -> b h n')

        indices = repeat(indices, 'b h n -> b h n d', d = codebook.shape[-1])
        codebook = repeat(codebook, 'h n d -> b h n d', b = indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, 'b h n d -> b n (h d)')
        codes = unpack_one(codes, ps, 'b * d')
        return codes

    def get_output_from_indices(self, indices):
        """根据索引重建量化输出，并投影回原特征维。"""
        codes = self.get_codes_from_indices(indices)
        return self.project_out(codes)

    def get_perplexity(self, encoding_indices, x):
        """根据 one-hot 平均概率计算 perplexity。

        `perplexity = exp(-sum(p * log(p)))`，数值越高通常说明码本使用越分散。
        """
        encode_onehot = F.one_hot(encoding_indices, self.codebook_size).type_as(x) # 代码/形状说明：[bthw, ncode]
        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity


    def get_usage(self, encoding_indices):
        """统计每个 code 在当前 batch 中的使用占比。"""
        # 把 batch 内所有 encoding indices 展平成一维，方便统一统计 code 使用率。
        all_indices = encoding_indices.flatten()

        # 统计当前 batch 中总共有多少个索引，用作百分比的分母。
        total_indices = all_indices.numel()

        # 初始化每个 code 的使用占比张量。
        codebook_usage_percentage = torch.zeros(self.codebook_size, device=all_indices.device)

        # 统计每个索引出现次数，再换算成使用占比。
        unique_indices, counts = torch.unique(all_indices, return_counts=True)

        # 计算占比。
        percentages = (counts.float() / total_indices)

        # 把占比写回对应 code 的位置，未出现的 code 保持 0。
        codebook_usage_percentage[unique_indices.long()] = percentages

        return codebook_usage_percentage


    def forward(
        self,
        x,
        indices = None,
        mask = None,
        sample_codebook_temp = None,
        freeze_codebook = False,
    ):
        """执行向量量化主流程。

        主要步骤：

        1. 规范输入 shape，并按需投影到 codebook 维度。
        2. 调用底层码本得到量化结果、索引和距离矩阵。
        3. 训练时使用 STE：`x + (quantize - x).detach()`。
        4. 计算 commitment loss、交叉熵损失或正交正则，并把索引整理回原布局。
        """
        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert not exists(mask)
            x = rearrange(x, 'b d -> b 1 d')

        shape, device, heads, is_multiheaded, codebook_size, return_loss = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size, exists(indices)

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        # 中文标题：rearrange inputs

        if self.accept_image_fmap:
            nframes, height, width = x.shape[-3:]
            x = rearrange(x, 'b c t h w -> b (t h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        # 先把输入投影到 codebook 使用的维度。

        x = self.project_in(x)

        # 多头量化时，把每个 head 的 token 整理到对应 codebook。

        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h = heads)

        # 如果使用 cosine 相似度，这里做 L2 归一化；否则保持原值。

        x = self._codebook.transform_input(x)

        # 中文标题：codebook forward kwargs

        codebook_forward_kwargs = dict(
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook
        )

        # 执行一次 codebook 查找/量化，得到量化向量、索引和距离矩阵。

        quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        # 可选：在当前 forward 内对 codebook 做一步原地优化。

        if should_inplace_optimize and self.training and not freeze_codebook:

            if exists(mask):
                loss = F.mse_loss(quantize, x.detach(), reduction = 'none')

                loss_mask = mask
                if is_multiheaded:
                    loss_mask = repeat(mask, 'b n -> c (b h) n', c = loss.shape[0], h = loss.shape[1] // mask.shape[0])

                loss = loss[loss_mask].mean()

            else:
                loss = F.mse_loss(quantize, x.detach())

            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()

            # codebook 更新后重新量化，避免返回旧 codebook 的结果。

            quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        if self.training:
            # 决定 commitment loss 使用 detach 后的 code，还是可学习 codebook 的原始 code。
            maybe_detach = torch.detach if not self.learnable_codebook or freeze_codebook else identity

            commit_quantize = maybe_detach(quantize)

            # straight-through：前向使用量化值，反向让梯度穿过原始输入。

            quantize = x + (quantize - x).detach()

            if self.sync_update_v > 0.:
                # 代码/形状说明：(21) in https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
                quantize = quantize + self.sync_update_v * (quantize - quantize.detach())

        # 基于距离矩阵计算 cross entropy loss 的辅助函数。
        # 用途：(1) NaturalSpeech2 residual VQ latent 对齐正确 code；(2) 基于交叉熵的 commitment loss。

        def calculate_ce_loss(codes):
            """把距离矩阵改排布后送入 `F.cross_entropy`。"""
            if not is_multiheaded:
                dist_einops_eq = '1 b n l -> b l n'
            elif self.separate_codebook_per_head:
                dist_einops_eq = 'c b n l -> b l n c'
            else:
                dist_einops_eq = '1 (b h) n l -> b l n h'

            ce_loss = F.cross_entropy(
                rearrange(distances, dist_einops_eq, b = shape[0]),
                codes,
                ignore_index = -1
            )

            return ce_loss

        # 中文标题：if returning cross entropy loss on codes that were passed in

        if return_loss:
            print(indices)
            return quantize, calculate_ce_loss(indices)

        # 中文标题：transform embedding indices

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h = heads)
            else:
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h = heads)

        if self.accept_image_fmap:
            embed_ind = rearrange(embed_ind, 'b (t h w) ... -> b t h w ...', t = nframes, h = height, w = width)

        if only_one:
            embed_ind = rearrange(embed_ind, 'b 1 ... -> b ...')

        # 中文标题：aggregate loss

        loss = torch.tensor([0.], device = device, requires_grad = self.training)

        if self.training:
            if self.commitment_weight > 0:
                if self.commitment_use_cross_entropy_loss:
                    if exists(mask):
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, 'b n -> b n h', h = heads)

                        embed_ind.masked_fill_(~ce_loss_mask, -1)

                    print(embed_ind.shape, embed_ind)
                    commit_loss = calculate_ce_loss(embed_ind)
                else:
                    if exists(mask):
                        # 中文标题：with variable lengthed sequences
                        commit_loss = F.mse_loss(commit_quantize, x, reduction = 'none')

                        loss_mask = mask
                        if is_multiheaded:
                            loss_mask = repeat(loss_mask, 'b n -> c (b h) n', c = commit_loss.shape[0], h = commit_loss.shape[1] // mask.shape[0])

                        commit_loss = commit_loss[loss_mask].mean()
                    else:
                        commit_loss = F.mse_loss(commit_quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed

                # 只对当前 batch 激活的 codes 计算 orthogonal loss。

                if self.orthogonal_reg_active_codes_only:
                    assert not (is_multiheaded and self.separate_codebook_per_head), '仅对 active codes 做 orthogonal regularization 暂不兼容 separate codebooks 的 multi-headed 模式'
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]

                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device = device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[:, rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        # 多头量化后，把每个 head 的量化 embedding 重新拼回主特征维。

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h = heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h = heads)

        # 投影回调用方期望的输出维度。

        quantize = self.project_out(quantize)

        # 如果输入曾经转置，这里把量化 embedding 排回原布局。

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (t h w) c -> b c t h w', t = nframes, h = height, w = width)

        if only_one:
            quantize = rearrange(quantize, 'b 1 d -> b d')

        # 代码/形状说明：if masking, only return quantized for where mask has True

        if exists(mask):
            quantize = torch.where(
                rearrange(mask, '... -> ... 1'),
                quantize,
                orig_input
            )

        # 代码/形状说明：return quantize, embed_ind, loss
        perplexity = self.get_perplexity(embed_ind, x)
        usage = self.get_usage(embed_ind)

        if self.call_cnt == 0:
            self.codebook_usage.data = usage
        else:
            self.codebook_usage.data = 0.99 * self.codebook_usage.data + (1 - 0.99) * usage

        self.call_cnt += 1
        # 代码/形状说明：avg_distribution = self.codebook_usage.data.sum() / self.codebook_size
        avg_usage = (self.codebook_usage.data > (1/self.codebook_size)).sum() / self.codebook_size

        return dict(embeddings=quantize, encodings=embed_ind,
                    commitment_loss=loss, perplexity=perplexity, avg_usage=avg_usage, batch_usage=usage)
