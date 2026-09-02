#!/usr/bin/env bash
# Launch one-node multi-GPU C1-O training. The caller must expose only GPUs
# already confirmed idle on this shared server.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

STAGE3_MASTER_ADDR="${STAGE3_MASTER_ADDR:-127.0.0.1}"
STAGE3_MASTER_PORT="${STAGE3_MASTER_PORT:-29731}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Refusing to select GPUs implicitly on a shared server." >&2
  echo "Set CUDA_VISIBLE_DEVICES to GPU IDs confirmed idle with nvidia-smi." >&2
  exit 2
fi
IFS=',' read -r -a STAGE3_VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
STAGE3_NUM_GPUS="${STAGE3_NUM_GPUS:-${#STAGE3_VISIBLE_GPU_LIST[@]}}"
if (( STAGE3_NUM_GPUS < 1 || STAGE3_NUM_GPUS > ${#STAGE3_VISIBLE_GPU_LIST[@]} )); then
  echo "STAGE3_NUM_GPUS must be between 1 and the number of visible GPUs." >&2
  exit 2
fi

echo "[stage3-c1-ddp] physical GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[stage3-c1-ddp] processes: ${STAGE3_NUM_GPUS}"
echo "[stage3-c1-ddp] rendezvous: ${STAGE3_MASTER_ADDR}:${STAGE3_MASTER_PORT}"
echo "[stage3-c1-ddp] c10d hostname warnings may repeat for about 60 seconds; wait for the distributed JSON line."

exec "${PROJECT_ROOT}/.venv/bin/torchrun" \
  --nnodes=1 \
  --nproc-per-node="${STAGE3_NUM_GPUS}" \
  --master-addr="${STAGE3_MASTER_ADDR}" \
  --master-port="${STAGE3_MASTER_PORT}" \
  "${PROJECT_ROOT}/scripts/train_stage3_cascade.py" \
  --device auto \
  "$@"
