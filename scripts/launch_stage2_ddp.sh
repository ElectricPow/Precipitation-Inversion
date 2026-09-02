#!/usr/bin/env bash
# Launch one-node multi-GPU Stage-2 training. The caller must explicitly expose
# only GPUs already confirmed idle on this shared server.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE2_MASTER_ADDR="${STAGE2_MASTER_ADDR:-127.0.0.1}"
STAGE2_MASTER_PORT="${STAGE2_MASTER_PORT:-29527}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Refusing to select GPUs implicitly on a shared server." >&2
  echo "Set CUDA_VISIBLE_DEVICES to GPU IDs confirmed idle with nvidia-smi." >&2
  exit 2
fi
IFS=',' read -r -a STAGE2_VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
STAGE2_NUM_GPUS="${STAGE2_NUM_GPUS:-${#STAGE2_VISIBLE_GPU_LIST[@]}}"
if (( STAGE2_NUM_GPUS < 1 || STAGE2_NUM_GPUS > ${#STAGE2_VISIBLE_GPU_LIST[@]} )); then
  echo "STAGE2_NUM_GPUS must be between 1 and the number of visible GPUs." >&2
  exit 2
fi

echo "[stage2-ddp] physical GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[stage2-ddp] processes: ${STAGE2_NUM_GPUS}"
echo "[stage2-ddp] rendezvous: ${STAGE2_MASTER_ADDR}:${STAGE2_MASTER_PORT}"

exec "${PROJECT_DIR}/.venv/bin/torchrun" \
  --nnodes=1 \
  --nproc-per-node="${STAGE2_NUM_GPUS}" \
  --master-addr="${STAGE2_MASTER_ADDR}" \
  --master-port="${STAGE2_MASTER_PORT}" \
  "${PROJECT_DIR}/scripts/train_stage2_unet3d.py" \
  --device auto \
  "$@"
