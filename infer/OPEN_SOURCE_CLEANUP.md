# WorldVLN 推理打包说明

本文档面向维护者，用于说明如何让 `infer/` 目录保持为可发布、干净的 WorldVLN 推理包。

## 主要入口

当前公开推理接口围绕以下文件组织：

- `server.py`
- `run_server.sh`
- `config.json`

## 服务路径必需文件

如果只保留 `InfinityStar -> latent2action` 在线服务路径，以下代码必须继续可用：

- `server.py`
- `config.json`
- `run_server.sh`
- `../Worldmodel/runtime/infinity/`
- `../Worldmodel/runtime/tools/closed_loop_streaming_infer_480p_81f.py`
- `../Worldmodel/runtime/tools/infinity_streaming_session.py`
- `../Worldmodel/runtime/tools/run_infinity.py`
- `../Worldmodel/action_decoder/actionhead_runtime/timesformer/`
- `../Worldmodel/action_decoder/actionhead_runtime/models/vae96_to_tsformer_adapter.py`

## 可选文件

以下文件不是主在线服务路径必需项，但对本地调试或实验可能仍有用：

- `config.local_bestrecord.json`
- `../train/action_decoder/actionhead_training/pretrain_latent_p2p.py`
- `../train/action_decoder/actionhead_training/latent_patch_embed.py`

## 不应发布的文件

为了让开源发布包保持干净，请避免提交本地运行产物、私有资产和不必要的实验文件。

例如：

- `__pycache__/`
- `cache/` 等本地 cache 目录
- 私有 checkpoint
- 日志、压缩包、临时文件和本地实验产物

清理 vendored 目录时，只保留已发布服务流程必需的源码文件；除非运行时路径仍然依赖，否则不要携带无关训练代码或旧实验代码。

## 备注

- 推理包现在复用顶层 `Worldmodel/` 和 `action_decoder/` 目录，不再 vendoring 单独副本。
- 默认路径已尽量改为相对仓库根目录解析。
- 模型权重仍应通过环境变量或挂载的本地路径提供，不应提交进仓库。
