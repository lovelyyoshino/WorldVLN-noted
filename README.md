# WorldVLN：面向空中视觉语言导航的自回归世界动作模型

[![arXiv](https://img.shields.io/badge/arXiv-2605.15964-b31b1b.svg)](https://arxiv.org/abs/2605.15964)
[![Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://embodiedcity.github.io/WorldVLN/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20weights-yellow.svg)](https://huggingface.co/EmbodiedCity/WorldVLN)

本仓库提供 WorldVLN 的参考实现，包含用于闭环动作预测的自回归推理服务，以及两阶段训练流程：（1）监督式 backbone 与动作解码器训练；（2）基于 action-aware GRPO 的优化。

如果想按中文路线阅读代码和理解架构：

- 第一次看代码请先看 [docs/READING_ROADMAP.zh-CN.md](./docs/READING_ROADMAP.zh-CN.md)（小白版导览，三天上手路线 + 重点公式速查 + 调试索引）。
- 想要函数级行号、张量速查、子系统细节，请看 [docs/CODE_READING_GUIDE.zh-CN.md](./docs/CODE_READING_GUIDE.zh-CN.md)（完整版，1600+ 行）。


## 安装

建议为已发布流程使用同一个 Python 3.10 环境。已验证的启动脚本会通过 `PYTHON_BIN` 显式指定 Python 解释器，因此激活环境后建议导出：

```bash
export PYTHON_BIN=$(which python)
```

### 推荐环境

1. 创建 Python 3.10 环境。

```bash
conda create -n worldvln python=3.10
conda activate worldvln
```

2. 安装与本机 CUDA 匹配的 PyTorch。对于已发布的训练和 action-aware GRPO 流程，推荐以 PyTorch 2.5.1 作为基线环境。

3. 安装已发布流程共用的依赖。

```bash
pip install -r requirements.txt
```

## 准备

### 模型权重

官方 WorldVLN 权重已发布到 Hugging Face：

- [WorldVLN 权重](https://huggingface.co/EmbodiedCity/WorldVLN)

请将权重下载到你的 checkpoint 目录，并在对应的训练或推理脚本中指向这些路径。

仿真与 benchmark 资源：

- IndoorUAV 仿真环境下载说明：[Indoor_UAV](https://modelscope.cn/datasets/valyentine/Indoor_UAV)
- UAV-Flow benchmark 与评测环境：[buaa-colalab/UAV-Flow](https://github.com/buaa-colalab/UAV-Flow)

## 推理

本仓库目前提供两个主要推理入口。

![WorldVLN 模型](./assets/model.png)

### 在线推理服务

#### 快速开始

在仓库根目录执行：

```bash
export PYTHON_BIN=$(which python)
export INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth
export STAGE2_LATENT2ACTION_CKPT=/path/to/stage2_latent2action_combined.pt

bash infer/run_server.sh
```

常用环境变量：

- `INFINITY_CKPT`：服务使用的主 InfinityStar / WorldVLN checkpoint
- `STAGE2_LATENT2ACTION_CKPT`：用于动作预测的 Stage-2 latent-to-action checkpoint
- `INFINITY_SERVER_CONFIG`：可选覆盖 `infer/config.json`
- `INFINITY_REPO_ROOT`：可选覆盖默认的 `Worldmodel/runtime/`
- `INFINITY_LATENT_CACHE_ROOT`：服务运行时 cache 目录
- `HOST`, `PORT`：Uvicorn 绑定地址

#### 细节

- 入口：[infer/run_server.sh](./infer/run_server.sh), [infer/server.py](./infer/server.py)
- 配置：[infer/config.json](./infer/config.json)
- Windows 侧客户端：[infer/client.py](./infer/client.py)

#### 自回归 I/O 约定（客户端如何调用服务）

WorldVLN 围绕轨迹 `session_id` 运行一个**自回归**闭环协议：

- **输入（每次调用）**：`images_base64`（RGB 帧）+ 首次调用时可选的 `instruction`。
  - 第一次通常发送 **1 帧预热观测**，用于建立 session 的语言条件和起始视觉状态。
  - 后续调用发送 **`step` 帧**（默认 16 帧），用于推进 session 时间线。
- **状态（服务端）**：服务端按 `session_id` 保存历史，并维护一个 streaming 世界模型 session。
- **输出（每次调用）**：`actions` 是 **cm/deg** 单位的增量动作，顺序为 `[dx, dy, dz, droll, dyaw, dpitch]`。
  - 在默认 `tsformer_latent` 模式下，只要收到足够帧，服务端就输出**一段动作**。

示例（严格闭环，启用 `allow_future_segments=1`）：

- 发送 **（1 帧真实图像 + instruction/prompt）** → 得到下一段 **`step` 个动作**（通常 16 个）。
- 执行动作，收集接下来的 **`step` 帧真实图像**，再发送给服务端 → 得到下一段 **`step` 个动作**。
- 持续使用同一个 `session_id` 重复该过程，直到轨迹结束。

本仓库中的客户端也遵循同一模式：

- **`infer/client.py`**（仿真 / 数据集示例）：在稳定的 `session_id` 下发送 `1, step, step, ...` 帧，并按段写出 `*_actions.json` / `*_poses.json`。
- **`action_aware_grpo/windows_client.py`**（Windows 侧 rollout 集成 / 调试）：使用相同的 session 协议和输出格式，但封装成更适合 Windows 使用的 action-aware GRPO 客户端。

## 训练

本仓库的训练分为两个阶段：

- **Stage 1（监督训练）**：backbone 微调 + 动作解码器训练。
- **Stage 2（action-aware GRPO）**：rollout 采集 + GRPO 训练。

![WorldVLN 框架](./assets/framework.png)

### Stage 1：监督训练

#### Backbone 训练

#### 快速开始

```bash
bash train/scripts/train_from_base.sh
```

#### 细节

backbone 微调流程位于 [train/](./train)。

- 入口：[train/scripts/train_from_base.sh](./train/scripts/train_from_base.sh)
- 主训练脚本：[train/train.py](./train/train.py)
- 详细指南：[train/TRAINING.md](./train/TRAINING.md)

#### 动作解码器训练

#### 快速开始（Stage A + Stage B）

```bash
# Stage A：adapter 蒸馏
bash train/action_decoder/scripts/train_stageA_ddp.sh

# Stage B：latent-to-action 训练
bash train/action_decoder/scripts/train_stageB_ddp.sh
```

#### 细节

动作解码器流程位于 [Worldmodel/action_decoder/src/](./Worldmodel/action_decoder/src)，并分为两个步骤（Stage A + Stage B）。

动作解码器训练入口位于 [train/action_decoder/](./train/action_decoder)，同样分为两个步骤：

- Stage A adapter 蒸馏：[train/action_decoder/scripts/train_stageA_ddp.sh](./train/action_decoder/scripts/train_stageA_ddp.sh)
- Stage B latent-to-action 训练：[train/action_decoder/scripts/train_stageB_ddp.sh](./train/action_decoder/scripts/train_stageB_ddp.sh)
- 主脚本：[train/action_decoder/tools/train_stageA_ddp.py](./train/action_decoder/tools/train_stageA_ddp.py), [train/action_decoder/tools/train_stageB_ddp.py](./train/action_decoder/tools/train_stageB_ddp.py)

该流程训练从视觉 latent 特征到 6-DoF 运动输出的映射。

数据约定（training manifest）：

```json
{
  "items_train": [
    {
      "latent_path": "path/to/latents.pt",
      "traj_json_path": "path/to/preprocessed_logs.json",
      "images_dir": "path/to/images"
    }
  ]
}
```

Stage A 必需环境变量：

- `MANIFEST_JSON`
- `TSFORMER_CKPT`
- `INF_VAE_PATH`

运行 Stage A：

```bash
bash train/action_decoder/scripts/train_stageA_ddp.sh
```

Stage B 必需环境变量：

- `MANIFEST_JSON`
- `TSFORMER_PRETRAINED`
- `ADAPTER_CKPT`
- `INFINITYSTAR_VAE_PATH`

运行 Stage B：

```bash
bash train/action_decoder/scripts/train_stageB_ddp.sh
```

### Stage 2：Action-aware GRPO

#### 快速开始（rollout + train）

启动 rollout 使用的本地推理服务：

```bash
INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth \
CHECKPOINTS_DIR=/path/to/checkpointsinf \
ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth \
ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json \
bash action_aware_grpo/run_infer_server.sh
```

运行 rollout 采集：

```bash
unset ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost

SRC_JSON=/path/to/reference_video_full_49f_trajectory_prompts.json \
INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth \
CHECKPOINTS_DIR=/path/to/checkpointsinf \
ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth \
ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json \
CUDA_VISIBLE_DEVICES=0 \
GRPO_LOCAL_GPU_IDS=0 \
NPROC_PER_NODE=1 \
NNODES=1 \
NODE_RANK=0 \
UAVFLOW_STAGEA_ROLLOUT_BACKEND=remote_sim \
UAVFLOW_SIMULATOR_BASE_URL=http://127.0.0.1:18765 \
UAVFLOW_SIMULATOR_TIMEOUT_S=120 \
UAVFLOW_TASK_JSON_ROOT=/path/to/UAV-Flow-Eval/test_jsons \
bash action_aware_grpo/scripts/run_stagea_collect.sh RUN_ID=remote_sim_smoke TOP_N=1 K_CAND=1 STAGEA_NPROC=1 STAGEA_PROGRESS_EVERY_N=1
```

运行训练（partial-freeze 优化）：

```bash
CHECKPOINTS_DIR=/path/to/checkpointsinf \
RUSH_RESUME=/path/to/infinity/global_step_xxx.pth \
REPLAY_META_DIR=/path/to/replay_meta_rollout_smoke \
bash action_aware_grpo/scripts/run_stageb_partialfreeze.sh PARTIAL_FREEZE_MODE=smoke RUN_ID=stageb_smoke
```

#### 细节

action-aware GRPO 流程位于 [action_aware_grpo/](./action_aware_grpo)，分为 **rollout** 和 **train** 两步。

- 服务端入口：[action_aware_grpo/grpo_server.py](./action_aware_grpo/grpo_server.py)
- Windows 侧客户端（用于 action-aware GRPO rollout 集成 / 调试）：[action_aware_grpo/windows_client.py](./action_aware_grpo/windows_client.py)
- Rollout 采集：[action_aware_grpo/scripts/run_stagea_collect.sh](./action_aware_grpo/scripts/run_stagea_collect.sh)
- 训练（partial-freeze 优化）：[action_aware_grpo/scripts/run_stageb_partialfreeze.sh](./action_aware_grpo/scripts/run_stageb_partialfreeze.sh)
- 远端模拟器服务封装：[action_aware_grpo/scripts/run_remote_sim_service.sh](./action_aware_grpo/scripts/run_remote_sim_service.sh)
- rollout 使用的本地推理启动脚本：[action_aware_grpo/run_infer_server.sh](./action_aware_grpo/run_infer_server.sh)

整体流程可以概括为：

- Rollout 读取 rollout 源数据和模型资产，生成 rollout cache 与 replay metadata。
- Train 读取 replay metadata 并执行优化，产出更新后的 checkpoint 和日志。

有关模拟器闭环 rollout 的细节，请参见 [action_aware_grpo/docs/remote_sim.md](./action_aware_grpo/docs/remote_sim.md)。

## 致谢

诚挚感谢以下项目的重要贡献：[InfinityStar](https://github.com/FoundationVision/InfinityStar), [TSformer-VO](https://github.com/aofrancani/TSformer-VO)。

## 引用

如果本工作对你有帮助，欢迎引用 WorldVLN 论文：

```bibtex
@misc{zhao2026worldvln,
      title={WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation},
      author={Baining Zhao and Jiacheng Xu and Weicheng Feng and Xin Zhang and Zhaolu Wang and Haoyang Wang and Shilong Ji and Ziyou Wang and Jianjie Fang and Zhiheng Zheng and Weichen Zhang and Yu Shang and Wei Wu and Chen Gao and Xinlei Chen and Yong Li},
      year={2026},
      eprint={2605.15964},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.15964},
}
```

## License

本项目基于 **CC BY 4.0** 许可发布，详见 `LICENSE`。
