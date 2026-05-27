#!/usr/bin/env bash
set -euo pipefail

# 允许从任意工作目录启动该脚本。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
echo "[cwd] $(pwd)"

# 避免在配额受限的文件系统写 .pyc，并让日志立即刷新。
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Stage-1 (DDP)：蒸馏一个 Adapter，把 InfinityStar VAE up_block_3 feature
# 映射到 TSformer PatchEmbed tokens。
# 必需路径由下面的环境变量提供。

# 可选 conda 环境；若不设置，则使用 PATH 中的当前 `python`。
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
CONDA_ROOT_HINT="${CONDA_ROOT_HINT:-}"

activate_conda_prefix() {
  local prefix="$1"
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$prefix"; return 0
  fi
  if [[ -f "${CONDA_ROOT_HINT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT_HINT}/etc/profile.d/conda.sh"
    conda activate "$prefix"; return 0
  fi
  if [[ -f "${CONDA_ROOT_HINT}/bin/activate" ]]; then
    source "${CONDA_ROOT_HINT}/bin/activate" "$prefix"; return 0
  fi
  if [[ -f "${prefix}/bin/activate" ]]; then
    source "${prefix}/bin/activate"; return 0
  fi
  echo "[error] 激活 conda env prefix 失败：${prefix}" >&2; exit 1
}

if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
  activate_conda_prefix "${CONDA_ENV_PREFIX}"
  echo "[env] CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX}"
fi
echo "[env] python=$(command -v python)"
python -V

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# DDP rendezvous 配置：在共享机器上尽量避免端口冲突。
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(
python - <<'PY'
import socket
s=socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)}"
echo "[ddp] master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"

ITEMS_KEY="${ITEMS_KEY:-ALL}"

TSFORMER_CKPT="${TSFORMER_CKPT:-}"
INF_VAE_PATH="${INF_VAE_PATH:-}"
MANIFEST_JSON="${MANIFEST_JSON:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/stage1_adapter}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

require_env() {
  local name="$1"
  local val="${!name:-}"
  if [[ -z "${val}" ]]; then
    echo "[error] 缺少必需环境变量：${name}" >&2
    exit 2
  fi
}

require_env MANIFEST_JSON
require_env TSFORMER_CKPT
require_env INF_VAE_PATH

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
STDOUT_STDERR_LOG="${STDOUT_STDERR_LOG:-${LOG_DIR}/stdout_stderr.log}"
exec > >(tee -a "${STDOUT_STDERR_LOG}") 2>&1
echo "[log] stdout/stderr -> ${STDOUT_STDERR_LOG}"

${TORCHRUN_BIN} --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/tools/train_stageA_ddp.py" \
  --out_dir "${OUT_DIR}" \
  --tqdm --log_file "train.log" --log_dir "${LOG_DIR}" \
  --manifest_json "${MANIFEST_JSON}" \
  --items_key "${ITEMS_KEY}" \
  --tsformer_ckpt "${TSFORMER_CKPT}" \
  --infinitystar_vae_path "${INF_VAE_PATH}" \
  ${EXTRA_ARGS}
