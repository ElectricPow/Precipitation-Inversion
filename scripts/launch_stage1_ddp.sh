#!/usr/bin/env bash
# Launch one-node multi-GPU stage-one training with an explicit loopback
# rendezvous. Set CUDA_VISIBLE_DEVICES first to avoid cards used by others.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE1_MASTER_ADDR="${STAGE1_MASTER_ADDR:-127.0.0.1}"
STAGE1_MASTER_PORT="${STAGE1_MASTER_PORT:-29517}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Refusing to select GPUs implicitly on a shared server." >&2
  echo "Set CUDA_VISIBLE_DEVICES to GPU IDs confirmed idle with nvidia-smi." >&2
  exit 2
fi
IFS=',' read -r -a STAGE1_VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
STAGE1_NUM_GPUS="${STAGE1_NUM_GPUS:-${#STAGE1_VISIBLE_GPU_LIST[@]}}"
if (( STAGE1_NUM_GPUS < 1 || STAGE1_NUM_GPUS > ${#STAGE1_VISIBLE_GPU_LIST[@]} )); then
  echo "STAGE1_NUM_GPUS must be between 1 and the number of visible GPUs." >&2
  exit 2
fi

echo "[stage1-ddp] physical GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[stage1-ddp] processes: ${STAGE1_NUM_GPUS}"
echo "[stage1-ddp] rendezvous: ${STAGE1_MASTER_ADDR}:${STAGE1_MASTER_PORT}"
echo "[stage1-ddp] c10d hostname warnings may repeat for about 60 seconds on this server; wait for the distributed JSON line."

exec "${PROJECT_DIR}/.venv/bin/torchrun" \
  --nnodes=1 \
  --nproc-per-node="${STAGE1_NUM_GPUS}" \
  --master-addr="${STAGE1_MASTER_ADDR}" \
  --master-port="${STAGE1_MASTER_PORT}" \
  "${PROJECT_DIR}/scripts/train_stage1_unet3d.py" \
  --device auto \
  "$@"
