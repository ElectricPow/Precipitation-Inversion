#!/usr/bin/env bash
# Launch one-node multi-GPU S3-D0. Expose only GPUs confirmed idle first.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

STAGE3_MASTER_ADDR="${STAGE3_MASTER_ADDR:-127.0.0.1}"
STAGE3_MASTER_PORT="${STAGE3_MASTER_PORT:-29751}"

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

echo "[stage3-d0-ddp] physical GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[stage3-d0-ddp] processes: ${STAGE3_NUM_GPUS}"
echo "[stage3-d0-ddp] rendezvous: ${STAGE3_MASTER_ADDR}:${STAGE3_MASTER_PORT}"

exec "${PROJECT_ROOT}/.venv/bin/torchrun" \
  --nnodes=1 \
  --nproc-per-node="${STAGE3_NUM_GPUS}" \
  --master-addr="${STAGE3_MASTER_ADDR}" \
  --master-port="${STAGE3_MASTER_PORT}" \
  "${PROJECT_ROOT}/scripts/train_stage3_d0.py" \
  --device auto \
  "$@"
