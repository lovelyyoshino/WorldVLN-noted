# 训练指南

本仓库保留一个统一训练入口：

- `scripts/train_from_base.sh`

该脚本会从原始分片基础权重开始微调，并默认使用相对仓库根目录的数据、checkpoint 和输出路径。

## 开源可用性

`scripts/train_from_base.sh` 不再包含机器绑定的硬编码路径，例如用户 home 目录、集群挂载点或旧工作区位置。

默认路径会按仓库根目录解析：

- checkpoint：`checkpoints/`
- 数据：`data/`
- 输出：`outputs/`

脚本仍会把仓库根目录转换成 `PYTHONPATH` 使用的绝对运行时路径，但该路径会在启动时根据当前 clone 位置生成，因此可以跨机器迁移。

## 必需目录结构

期望的仓库结构：

```text
train/
|-- train.py
|-- scripts/
|   `-- train_from_base.sh
|-- checkpoints/
|   |-- text_encoder/
|   |   `-- flan-t5-xl-official/
|   |-- infinitystar_videovae.pth
|   `-- infinitystar_8b_480p_weights/
|-- data/
|   `-- <your jsonl shard directory>
`-- outputs/
```

## 必需权重

启动训练前，请在 `checkpoints/` 下准备这些文件，或通过环境变量覆盖路径：

1. T5 文本编码器目录

   默认路径：

   - `checkpoints/text_encoder/flan-t5-xl-official`

   环境变量覆盖：

   - `T5_PATH`

2. Video VAE checkpoint 文件

   默认路径：

   - `checkpoints/infinitystar_videovae.pth`

   环境变量覆盖：

   - `VAE_PATH`

3. 基础模型分片权重

   默认路径：

   - `checkpoints/infinitystar_8b_480p_weights`

   环境变量覆盖：

   - `TORCHSHARD_RESUME_PATH`

脚本会先检查这三个路径是否都存在，再调用 `train.py`。

## 必需训练数据

脚本要求 `VIDEO_DATA_PATH` 指向一个包含 JSONL 分片的目录。

默认搜索顺序：

1. `data/uavflow_49f_from_40_60_split8_jsonl`
2. `data/uavflow_40_60_split8_jsonl`
3. `data/split8_jsonl`

环境变量覆盖：

- `VIDEO_DATA_PATH`

支持的分片目录结构：

1. 平铺结构

```text
data/split8_jsonl/
|-- part_00.jsonl
|-- part_01.jsonl
|-- ...
`-- part_07.jsonl
```

2. 分桶结构

```text
data/split8_jsonl/
|-- bucket_0/
|   |-- part_00.jsonl
|   `-- part_01.jsonl
`-- bucket_1/
    |-- part_02.jsonl
    `-- part_03.jsonl
```

## 最小 JSONL Schema

每一行都必须是一个 JSON 对象。对于视频训练，loader 至少需要以下字段：

```json
{
  "video_path": "relative/or/absolute/path/to/video.mp4",
  "begin_frame_id": 0,
  "end_frame_id": 48,
  "fps": 16.0,
  "tarsier2_caption": "A UAV flies over a road."
}
```

推荐的可选字段：

```json
{
  "frame_idxs": [0, 1, 2, 3, 4],
  "sample_frames": 49,
  "quality_prompt": "high quality, detailed",
  "MiniCPM_V_2_6_caption": "Alternative caption text"
}
```

说明：

1. `video_path` 可以是绝对路径或相对路径，但必须能在训练机器上解析到实际视频文件。
2. loader 会把 `end_frame_id` 当作包含端点的帧索引。
3. 如果提供了 `frame_idxs`，loader 会直接使用这些显式帧索引。
4. 如果没有提供 `frame_idxs`，采样会根据标注片段和训练配置自动推断。

## 启动行为

`scripts/train_from_base.sh` 使用以下默认配置：

- 模型：`infinity_qwen8b`
- 分辨率预设：`0.40M`
- 视频帧数：`49`
- 视频 fps：`16`
- mask schedule：`infinity_elegant_clip4frames_v2_allpt`
- 优化器学习率：`1e-5`
- 总 epoch 数：`10`
- 保存频率：每 `1000` 次迭代保存一次

输出会写入：

- 日志：`outputs/run_logs/<EXP_NAME>`
- checkpoint：`outputs/checkpoints/<EXP_NAME>`
- token cache：`outputs/cache/<EXP_NAME>`

## 快速开始

在仓库根目录运行：

```bash
bash scripts/train_from_base.sh
```

也可以显式覆盖路径和参数：

```bash
CHECKPOINTS_DIR=./checkpoints \
VIDEO_DATA_PATH=./data/split8_jsonl \
OUTPUT_ROOT=./outputs \
ARNOLD_WORKER_GPU=8 \
EXP_NAME=my_train_run \
bash scripts/train_from_base.sh
```

## 环境变量

常用覆盖项：

- `PYTHON_BIN`：要使用的 Python 可执行文件
- `CHECKPOINTS_DIR`：所有默认权重路径的基础目录
- `T5_PATH`：显式 T5 路径
- `VAE_PATH`：显式 VAE checkpoint 路径
- `TORCHSHARD_RESUME_PATH`：显式基础模型分片权重路径
- `DATA_ROOT`：基础数据目录
- `VIDEO_DATA_PATH`：显式 JSONL 分片目录
- `OUTPUT_ROOT`：基础输出目录
- `LOCAL_OUT_PATH`：显式运行日志目录
- `BED_PATH`：显式 checkpoint 保存目录
- `TOKEN_CACHE_DIR`：显式 token cache 目录
- `EXP_NAME`：实验名称
- `TRAIN_EPOCHS`：总 epoch 数
- `SAVE_FREQ_ITERS`：checkpoint 保存间隔
- `TLR`：学习率
- `ARNOLD_WORKER_GPU`：每个节点的 GPU 数
- `ARNOLD_WORKER_NUM`：节点数
- `ARNOLD_ID`：节点 rank
- `ARNOLD_WORKER_0_HOST`：master host
- `ARNOLD_WORKER_0_PORT`：master port

## 训练前检查清单

训练前请确认：

1. 仓库根目录存在 `train.py`。
2. 存在 `checkpoints/text_encoder/flan-t5-xl-official`。
3. 存在 `checkpoints/infinitystar_videovae.pth`。
4. 存在 `checkpoints/infinitystar_8b_480p_weights`。
5. `VIDEO_DATA_PATH` 指向包含 JSONL 文件的目录。
6. 每条 JSONL 记录都指向可读取的本地视频文件。
7. 已安装仓库根目录 `../requirements.txt` 中要求的 Python 依赖。

如果任一必需路径缺失，脚本会立即报错退出。
