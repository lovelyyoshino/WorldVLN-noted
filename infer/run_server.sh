#!/usr/bin/env bash
set -euo pipefail

# 这个脚本是在线推理服务的最小启动入口：
# - 只负责准备环境变量和 cache/checkpoint 路径；
# - 真正的 FastAPI 协议、session 状态和动作预测逻辑在 infer/server.py；
# - 如果要换模型或本地权重，优先覆盖下面这些环境变量，而不是直接改 server.py。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 基础服务参数。ACTION_HEAD_MODE 默认走当前推荐的 Stage-2 latent 到 TimesFormer 再到 action 的路径。
export PYTHON_BIN="${PYTHON_BIN:-python}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8001}"
export ACTION_HEAD_MODE="${ACTION_HEAD_MODE:-tsformer_latent}"

# INFINITY_SERVER_CONFIG: server.py 读取的世界模型/VAE/采样配置。
# INFINITY_REPO_ROOT: runtime 版 Infinity 代码根目录，server.py 会把它加入 sys.path。
# INFINITY_RESET_SESSION_ON_ONE_FRAME=1: 客户端只上传 1 帧时重置同名 session，适合用 1 帧预热新轨迹。
# INFINITY_REQUIRE_TGT_HW=640,640: 强制输入帧缩放/校验到训练期分辨率，避免 latent 形状偏移。
export INFINITY_SERVER_CONFIG="${INFINITY_SERVER_CONFIG:-${SCRIPT_DIR}/config.json}"
export INFINITY_REPO_ROOT="${INFINITY_REPO_ROOT:-${REPO_ROOT}/Worldmodel/runtime}"
export INFINITY_RESET_SESSION_ON_ONE_FRAME="${INFINITY_RESET_SESSION_ON_ONE_FRAME:-1}"
export INFINITY_REQUIRE_TGT_HW="${INFINITY_REQUIRE_TGT_HW:-640,640}"

# INFINITY_LATENT_CACHE_ROOT: 服务端把 segment latent/action 中间结果写到这里，方便复查。
# STAGE2_LATENT2ACTION_CKPT: Stage-2 动作头权重；默认文件名是合并后的 adapter+TimesFormer checkpoint。
export INFINITY_LATENT_CACHE_ROOT="${INFINITY_LATENT_CACHE_ROOT:-${SCRIPT_DIR}/cache}"
export STAGE2_LATENT2ACTION_CKPT="${STAGE2_LATENT2ACTION_CKPT:-${SCRIPT_DIR}/checkpoints/stage2_latent2action_combined.pt}"

mkdir -p "${INFINITY_LATENT_CACHE_ROOT}"

# uvicorn 加载的 app 模块就是 infer/server.py 中的 FastAPI app。
exec "${PYTHON_BIN}" -m uvicorn \
  server:app \
  --host "${HOST}" \
  --port "${PORT}"
