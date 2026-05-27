# Copyright 2020 Ross Wightman

from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class FeatureInfo:
    """保存模型各层特征的通道数、下采样倍率和模块名等元信息。"""

    def __init__(self, feature_info: List[Dict], out_indices: Tuple[int]):
        """校验并保存特征层信息，以及调用方希望输出的层索引。"""
        prev_reduction = 1
        for fi in feature_info:
            # 检查必需字段；不同模型还可以带有额外字段。
            assert 'num_chs' in fi and fi['num_chs'] > 0
            assert 'reduction' in fi and fi['reduction'] >= prev_reduction
            prev_reduction = fi['reduction']
            assert 'module' in fi
        self.out_indices = out_indices
        self.info = feature_info

    def from_other(self, out_indices: Tuple[int]):
        """基于当前特征信息复制一份，只替换输出层索引。"""
        return FeatureInfo(deepcopy(self.info), out_indices)

    def get(self, key, idx=None):
        """按 key 和索引读取特征信息。

        如果 idx 为 None，返回每个输出索引对应的 key 值。
        如果 idx 是整数，返回该特征模块索引上的 key 值，不使用 out_indices。
        如果 idx 是 list/tuple，返回这些模块索引上的 key 值，不使用 out_indices。
        """
        if idx is None:
            return [self.info[i][key] for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i][key] for i in idx]
        else:
            return self.info[idx][key]

    def get_dicts(self, keys=None, idx=None):
        """返回指定索引处的特征信息字典，可只保留指定键。
        """
        if idx is None:
            if keys is None:
                return [self.info[i] for i in self.out_indices]
            else:
                return [{k: self.info[i][k] for k in keys} for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i] if keys is None else {k: self.info[i][k] for k in keys} for i in idx]
        else:
            return self.info[idx] if keys is None else {k: self.info[idx][k] for k in keys}

    def channels(self, idx=None):
        """读取特征图通道数 ``num_chs``。
        """
        return self.get('num_chs', idx)

    def reduction(self, idx=None):
        """读取特征图相对输入的下采样倍率 ``reduction``。
        """
        return self.get('reduction', idx)

    def module_name(self, idx=None):
        """读取产生特征的模块名 ``module``。
        """
        return self.get('module', idx)

    def __getitem__(self, item):
        """允许像列表一样用索引读取单个特征信息字典。"""
        return self.info[item]

    def __len__(self):
        """返回已记录的特征层数量。"""
        return len(self.info)


class FeatureHooks:
    """特征 hook 辅助器。

    根据模块名注册 forward 或 forward_pre hook，用于从模型内部节点取出特征。
    这种方式适合 eager Python；如果要支持 TorchScript，需要重新设计。
    """

    def __init__(self, hooks, named_modules, out_map=None, default_hook_type='forward'):
        """按 hook 配置在指定模块上注册特征收集函数。"""
        # 设置特征 hook。
        modules = {k: v for k, v in named_modules}
        for i, h in enumerate(hooks):
            hook_name = h['module']
            m = modules[hook_name]
            hook_id = out_map[i] if out_map else hook_name
            hook_fn = partial(self._collect_output_hook, hook_id)
            hook_type = h['hook_type'] if 'hook_type' in h else default_hook_type
            if hook_type == 'forward_pre':
                m.register_forward_pre_hook(hook_fn)
            elif hook_type == 'forward':
                m.register_forward_hook(hook_fn)
            else:
                assert False, "不支持的 hook 类型"
        self._feature_outputs = defaultdict(OrderedDict)

    def _collect_output_hook(self, hook_id, *args):
        """hook 回调：取出目标 tensor，并按设备和 hook_id 缓存。"""
        x = args[-1]  # 目标 tensor 是最后一个参数：forward 输出或 forward_pre 输入。
        if isinstance(x, tuple):
            x = x[0]  # 如果输入被 tuple 包住，只取第一个 tensor。
        self._feature_outputs[x.device][hook_id] = x

    def get_output(self, device) -> Dict[str, torch.tensor]:
        """读取某个设备上缓存的特征，并在读取后清空缓存。"""
        output = self._feature_outputs[device]
        self._feature_outputs[device] = OrderedDict()  # 读取后清空。
        return output


def _module_list(module, flatten_sequential=False):
    """列出直接子模块，必要时把第一层 Sequential 展平成普通子模块。"""
    # yield/迭代器写法更自然，但不兼容 TorchScript。
    ml = []
    for name, module in module.named_children():
        if flatten_sequential and isinstance(module, nn.Sequential):
            # 第一层 Sequential 容器会被展开到外层模型中。
            for child_name, child_module in module.named_children():
                combined = [name, child_name]
                ml.append(('_'.join(combined), '.'.join(combined), child_module))
        else:
            ml.append((name, name, module))
    return ml


def _get_feature_info(net, out_indices):
    """从模型中读取 ``feature_info``，并包装成 ``FeatureInfo``。"""
    feature_info = getattr(net, 'feature_info')
    if isinstance(feature_info, FeatureInfo):
        return feature_info.from_other(out_indices)
    elif isinstance(feature_info, (list, tuple)):
        return FeatureInfo(net.feature_info, out_indices)
    else:
        assert False, "提供的 feature_info 无效"


def _get_return_layers(feature_info, out_map):
    """根据 feature_info 和 out_map 建立模块名到返回 id 的映射。"""
    module_names = feature_info.module_name()
    return_layers = {}
    for i, name in enumerate(module_names):
        return_layers[name] = out_map[i] if out_map is not None else feature_info.out_indices[i]
    return return_layers


class FeatureDictNet(nn.ModuleDict):
    """返回 ``OrderedDict`` 的特征提取器。

        这个包装器会按 out_indices 指定的层抽取特征，并用原模型中的子模块
        部分重建网络。它假设模块注册顺序与实际 forward 使用顺序一致，并且
        同一个 ``nn.Module`` 不会被重复使用（包括 ``self.relu = nn.ReLU`` 这类简单模块）。
        它只能捕获直接挂在模型上的子模块（如 ``model.feature1``），或在
        ``flatten_sequential=True`` 时捕获一层 Sequential 内的模块（如 ``model.features.1``）。
        直接挂在原模型上的 Sequential 会被展开到当前模块中，名字中的点会替换为下划线。

        参数：
            model (nn.Module): 要从中抽取特征的模型。
            out_indices (tuple[int]): 需要抽取的模型输出层索引。
            out_map (sequence): 为每个输出索引指定返回 id；未指定时使用 str(index)。
            feature_concat (bool): 中间特征是 list/tuple 时，是否拼接它们；否则取第 0 个元素。
            flatten_sequential (bool): 是否展开直接挂在模型上的 Sequential 模块。

    """
    def __init__(
            self, model,
            out_indices=(0, 1, 2, 3, 4), out_map=None, feature_concat=False, flatten_sequential=False):
        """初始化特征提取网络，并记录哪些层需要返回。"""
        super(FeatureDictNet, self).__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        self.concat = feature_concat
        self.return_layers = {}
        return_layers = _get_return_layers(self.feature_info, out_map)
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        remaining = set(return_layers.keys())
        layers = OrderedDict()
        for new_name, old_name, module in modules:
            layers[new_name] = module
            if old_name in remaining:
                # 为了兼容 TorchScript，返回 id 必须统一为 str 类型。
                self.return_layers[new_name] = str(return_layers[old_name])
                remaining.remove(old_name)
            if not remaining:
                break
        assert not remaining and len(self.return_layers) == len(return_layers), \
            f'请求返回的层 ({remaining}) 不存在于模型中'
        self.update(layers)

    def _collect(self, x) -> (Dict[str, torch.Tensor]):
        """顺序执行子模块，并收集配置中指定的中间特征。"""
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_id = self.return_layers[name]
                if isinstance(x, (tuple, list)):
                    # 如果抽取点输出 tuple/list，就拼接或取第一个元素。
                    # 待修复：某些网络可能需要更通用、更灵活的处理方式。
                    out[out_id] = torch.cat(x, 1) if self.concat else x[0]
                else:
                    out[out_id] = x
        return out

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """返回按 id 组织的中间特征字典。"""
        return self._collect(x)


class FeatureListNet(FeatureDictNet):
    """返回 list 的特征提取器。

    主要逻辑见 ``FeatureDictNet``。这个类存在是为了满足 TorchScript 类型约束；
    在 eager Python 中本可以通过成员变量决定返回 ``List[Tensor]`` 还是 ``Dict[id, Tensor]``。
    """
    def __init__(
            self, model,
            out_indices=(0, 1, 2, 3, 4), out_map=None, feature_concat=False, flatten_sequential=False):
        """初始化返回 list 的特征提取包装器。"""
        super(FeatureListNet, self).__init__(
            model, out_indices=out_indices, out_map=out_map, feature_concat=feature_concat,
            flatten_sequential=flatten_sequential)

    def forward(self, x) -> (List[torch.Tensor]):
        """返回按配置顺序排列的中间特征列表。"""
        return list(self._collect(x).values())


class FeatureHookNet(nn.ModuleDict):
    """使用 forward/forward_pre hook 的特征提取器。

    这个包装器会按 out_indices 指定的位置注册 hook 并抽取特征。
    如果 ``no_rewrite`` 为 True，只通过 hook 取特征，不改写底层网络结构。
    如果 ``no_rewrite`` 为 False，会像 FeatureList/FeatureDict 一样重写模型：
    将第一层到第二层（仅 Sequential）的模块折叠到当前模块中。
    待修复：目前不支持 TorchScript，原因见 ``FeatureHooks``。
    """
    def __init__(
            self, model,
            out_indices=(0, 1, 2, 3, 4), out_map=None, out_as_dict=False, no_rewrite=False,
            feature_concat=False, flatten_sequential=False, default_hook_type='forward'):
        """初始化 hook 版本的特征提取网络，并注册需要的 hook。"""
        super(FeatureHookNet, self).__init__()
        assert not torch.jit.is_scripting()
        self.feature_info = _get_feature_info(model, out_indices)
        self.out_as_dict = out_as_dict
        layers = OrderedDict()
        hooks = []
        if no_rewrite:
            assert not flatten_sequential
            if hasattr(model, 'reset_classifier'):  # 确保分类头被移除？
                model.reset_classifier(0)
            layers['body'] = model
            hooks.extend(self.feature_info.get_dicts())
        else:
            modules = _module_list(model, flatten_sequential=flatten_sequential)
            remaining = {f['module']: f['hook_type'] if 'hook_type' in f else default_hook_type
                         for f in self.feature_info.get_dicts()}
            for new_name, old_name, module in modules:
                layers[new_name] = module
                for fn, fm in module.named_modules(prefix=old_name):
                    if fn in remaining:
                        hooks.append(dict(module=fn, hook_type=remaining[fn]))
                        del remaining[fn]
                if not remaining:
                    break
            assert not remaining, f'请求返回的层 ({remaining}) 不存在于模型中'
        self.update(layers)
        self.hooks = FeatureHooks(hooks, model.named_modules(), out_map=out_map)

    def forward(self, x):
        """执行包装后的模型，并返回 hook 收集到的特征。"""
        for name, module in self.items():
            x = module(x)
        out = self.hooks.get_output(x.device)
        return out if self.out_as_dict else list(out.values())
