# 远端模拟器流程

`remote_sim` 路径会把策略推理保留在训练机器上，只把环境 reset、动作执行和画面渲染交给模拟器机器。

## 服务端点

模拟器侧服务由 `runtime/client.py` 的 service 模式实现，暴露以下接口：

- GET /health
- POST /reset
- POST /step_actions

## 启动服务

在模拟器机器上运行：

```bash
python runtime/client.py \
  --mode service \
  --host 0.0.0.0 \
  --port 8765 \
  --task_json_root /path/to/UAV-Flow-Eval/test_jsons
```

## 反向端口转发

如果训练机器无法直接访问模拟器机器，可以从模拟器机器向训练机器建立反向隧道：

```bash
ssh -R 18765:127.0.0.1:8765 user@training-host
```

然后在训练机器上验证：

```bash
curl --noproxy '*' http://127.0.0.1:18765/health
```

## Rollout 设置

开源版 action-aware GRPO 包只保留 `remote_sim` 作为 rollout 后端。可使用下面的封装命令：

```bash
cd ./action_aware_grpo

unset ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost

export PYTHON_BIN=${PYTHON_BIN:-python}

export CUDA_VISIBLE_DEVICES=0
export GRPO_LOCAL_GPU_IDS=0
export NPROC_PER_NODE=1
export NNODES=1
export NODE_RANK=0

export SRC_JSON=/path/to/reference_video_full_49f_trajectory_prompts.json
export CHECKPOINTS_DIR=/path/to/checkpointsinf
export INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth
export ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth
export ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json
UAVFLOW_STAGEA_ROLLOUT_BACKEND=remote_sim
UAVFLOW_SIMULATOR_BASE_URL=http://127.0.0.1:18765
export UAVFLOW_SIMULATOR_TIMEOUT_S=120
UAVFLOW_TASK_JSON_ROOT=/path/to/UAV-Flow-Eval/test_jsons
STAGEA_NPROC=1

export RUN_ID=rl_rollout_smoke_$(date +%Y%m%d_%H%M%S)

bash scripts/run_stagea_collect.sh \
  RUN_ID=${RUN_ID} \
  TOP_N=1 \
  K_CAND=1 \
  STAGEA_NPROC=1 \
  STAGEA_PROGRESS_EVERY_N=1
```

除非模拟器服务已经扩展为支持多个并发 session，否则 `STAGEA_NPROC` 应保持为 1。
