# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import torch
import torch.distributed as dist
import imageio
import os
import random

import math
import numpy as np
from einops import rearrange
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import sys
import pdb as pdb_original
from contextlib import contextmanager

COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"
ptdtype = {None: torch.float32, 'fp32': torch.float32, 'bf16': torch.bfloat16}

def rank_zero_only(fn):
    """中文说明：`rank_zero_only` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    def wrapped_fn(*args, **kwargs):
        """中文说明：`wrapped_fn` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `wrapped_fn` 步骤，供训练、推理或调试流程复用。

        新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)
    return wrapped_fn

@rank_zero_only
def print_gpu_usage(model_name) -> None:
    """中文说明：`print_gpu_usage` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `print_gpu_usage` 步骤，供训练、推理或调试流程复用。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    print(f"{model_name} backward 之后的显存: 已分配={allocated_memory}, 已保留={reserved_memory}")
    torch.cuda.empty_cache()

def seed_everything(seed=0, allow_tf32=True, benchmark=True, deterministic=False):
    """中文说明：`seed_everything` 设置随机种子，保证数据顺序、初始化和采样过程尽可能可复现。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark  # torch 2.3.1 中默认值为 False。

    # 参考：https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # 参考：https://pytorch.org/docs/stable/notes/randomness.html
    torch.use_deterministic_algorithms(deterministic)

    torch.backends.cudnn.allow_tf32 = allow_tf32  # torch 2.3.1 中默认值为 True。
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # torch 2.3.1 中默认值为 True。

# 以表格格式打印模型摘要
@rank_zero_only
def print_model_summary(models):
        # 表头
        """中文说明：`print_model_summary` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `print_model_summary` 步骤，供训练、推理或调试流程复用。

        新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        print(f"{'层名':<20} {'参数量':<20}")
        print("="*40)
        total_params = 0
        for model in models:
            for name, module in model.named_children():
                params = sum(p.numel() for p in module.parameters())
                total_params += params
                params_str = f"{params/1e6:.2f}M"
                print(f"{name:<20} {params_str:<20}")
        print("="*40)
        print(f"参数总量: {total_params/1e6:.2f}M")

def version_checker(base_version, high_version):
    """中文说明：`version_checker` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `version_checker` 步骤，供训练、推理或调试流程复用。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    try:
        from bytedance.ndtimeline import __version__
        from packaging.version import Version
        if Version(__version__) < Version(base_version) or Version(__version__) >= Version(high_version):
            raise RuntimeError(f"bytedance.ndtimeline 版本应满足 >={base_version} 且 <{high_version}，当前发现 {__version__}")
    except ImportError:
        raise RuntimeError(f"需要安装 bytedance.ndtimeline，版本应满足 >={base_version} 且 <{high_version}")

def is_torch_optim_sch(obj):
    """中文说明：`is_torch_optim_sch` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `is_torch_optim_sch` 步骤，供训练、推理或调试流程复用。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    return isinstance(obj, (optim.Optimizer, optim.lr_scheduler.LambdaLR))

def rearranged_forward(x, func):
    """中文说明：`rearranged_forward` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `rearranged_forward` 步骤，供训练、推理或调试流程复用。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    if x.ndim == 4:
        x = rearrange(x, "B C H W -> B H W C")
    elif x.ndim == 5:
        x = rearrange(x, "B C T H W -> B T H W C")
    x = func(x)
    if x.ndim == 4:
        x = rearrange(x, "B H W C -> B C H W")
    elif x.ndim == 5:
        x = rearrange(x, "B T H W C -> B C T H W")
    return x

def is_dtype_16(data):
    """中文说明：`is_dtype_16` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `is_dtype_16` 步骤，供训练、推理或调试流程复用。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    return data.dtype == torch.float16 or data.dtype == torch.bfloat16

@contextmanager
def set_tf32_flags(flag):
    """中文说明：`set_tf32_flags` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `set_tf32_flags` 步骤，供训练、推理或调试流程复用。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    old_matmul_flag = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_flag = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = flag
    torch.backends.cudnn.allow_tf32 = flag
    try:
        yield
    finally:
        # 恢复原始 flags
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_flag
        torch.backends.cudnn.allow_tf32 = old_cudnn_flag

class ByteNASManager:
    """中文说明：`ByteNASManager` 封装VideoVAE 通用张量、随机种子和可视化工具中的状态和子模块。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    bytenas_dir = {

    }
    _current_bytenas = None
    _username = None

    @classmethod
    def set_bytenas(cls, bytenas, username="zhufengda"):
        """中文说明：`set_bytenas` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `set_bytenas` 步骤，供训练、推理或调试流程复用。

        新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        cls._current_bytenas = bytenas
        cls._username = username

    @classmethod
    def get_work_dir(cls, use_username=True):
        """中文说明：`get_work_dir` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `get_work_dir` 步骤，供训练、推理或调试流程复用。

        新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if use_username:
            username = cls._username
        else:
            username = ""
        base_dir = cls.bytenas_dir[cls._current_bytenas]
        return os.path.join(base_dir, username)

    @classmethod
    def __call__(cls, rel_path, use_username=True, prefix=""):
        """中文说明：`__call__` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `__call__` 步骤，供训练、推理或调试流程复用。

        新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return os.path.join(cls.get_work_dir(use_username=use_username), prefix, rel_path)

bytenas_manager = ByteNASManager()

def get_last_ckpt(root_dir):
    """中文说明：`get_last_ckpt` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    if not os.path.exists(root_dir): return None
    ckpt_files = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.ckpt'):
                num_iter = int(filename.split('.ckpt')[0].split('_')[-1])
                ckpt_files[num_iter]=os.path.join(dirpath, filename)
    iter_list = list(ckpt_files.keys())
    if len(iter_list) == 0: return None
    max_iter = max(iter_list)
    return ckpt_files[max_iter]


# 将 src_tf 维度移动到 dest 位置
# 例：shift_dim(x, 1, -1) 会把 (b, c, t, h, w) 变为 (b, t, h, w, c)
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    """中文说明：`shift_dim` 执行张量维度重排或切片；新手阅读时建议在纸上写出每一维含义。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# 从第 i 维（含）开始重塑张量
# 到第 j 维（不含）结束，并替换为目标 shape
# 例：如果 x.shape = (b, thw, c)
# 示例：view_range(x, 1, 2, (t, h, w)) 会返回
# 输出 shape 为 (b, t, h, w, c) 的张量
def view_range(x, i, j, shape):
    """中文说明：`view_range` 执行张量维度重排或切片；新手阅读时建议在纸上写出每一维含义。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def accuracy(output, target, topk=(1,)):
    """中文说明：计算指定 top-k 取值下预测是否命中的准确率。"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def tensor_slice(x, begin, size):
    """中文说明：`tensor_slice` 执行张量维度重排或切片；新手阅读时建议在纸上写出每一维含义。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


def save_video_grid(video, fname, nrow=None, fps=16):
    """中文说明：`save_video_grid` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1).contiguous()

    video = (video.detach().cpu().numpy() * 255).astype('uint8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype='uint8')
    # 调试输出：print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    imageio.mimsave(fname, video, fps=fps)
    # 可选保存：skvideo.io.vwrite(fname, video_grid, inputdict={'-r': '5'})
    # 调试输出：print('视频已保存到', fname)


def comp_getattr(args, attr_name, default=None):
    """中文说明：`comp_getattr` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `comp_getattr` 步骤，供训练、推理或调试流程复用。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    if hasattr(args, attr_name):
        return getattr(args, attr_name)
    else:
        return default


def visualize_tensors(t, name=None, nest=0):
    """中文说明：`visualize_tensors` 实现VideoVAE 通用张量、随机种子和可视化工具中的 `visualize_tensors` 步骤，供训练、推理或调试流程复用。

    新手提示：这些函数多是训练脚手架，先看输入输出 shape，再看是否只在 rank0 执行。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    if name is not None:
        print(name, "当前嵌套层级: ", nest)
    print("类型: ", type(t))
    if 'dict' in str(type(t)):
        print(t.keys())
        for k in t.keys():
            if t[k] is None:
                print(k, "None")
            else:
                if 'Tensor' in str(type(t[k])):
                    print(k, t[k].shape)
                elif 'dict' in str(type(t[k])):
                    print(k, 'dict')
                    visualize_tensors(t[k], name, nest + 1)
                elif 'list' in str(type(t[k])):
                    print(k, len(t[k]))
                    visualize_tensors(t[k], name, nest + 1)
    elif 'list' in str(type(t)):
        print("列表长度: ", len(t))
        for t2 in t:
            visualize_tensors(t2, name, nest + 1)
    elif 'Tensor' in str(type(t)):
        print(t.shape)
    else:
        print(t)
    return ""
