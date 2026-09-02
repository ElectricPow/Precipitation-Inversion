#!/usr/bin/env bash
# Launch the S2-R1-O upper-bound training on explicitly selected idle GPUs.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE2_MASTER_ADDR="${STAGE2_MASTER_ADDR:-127.0.0.1}"
STAGE2_MASTER_PORT="${STAGE2_MASTER_PORT:-29531}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Refusing to select GPUs implicitly on a shared server." >&2
  exit 2
fi
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
STAGE2_NUM_GPUS="${STAGE2_NUM_GPUS:-${#VISIBLE_GPU_LIST[@]}}"
if (( STAGE2_NUM_GPUS < 1 || STAGE2_NUM_GPUS > ${#VISIBLE_GPU_LIST[@]} )); then
  echo "STAGE2_NUM_GPUS is incompatible with CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi

echo "[stage2-r1-ddp] physical GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[stage2-r1-ddp] processes: ${STAGE2_NUM_GPUS}"
exec "${PROJECT_DIR}/.venv/bin/torchrun" \
  --nnodes=1 \
  --nproc-per-node="${STAGE2_NUM_GPUS}" \
  --master-addr="${STAGE2_MASTER_ADDR}" \
  --master-port="${STAGE2_MASTER_PORT}" \
  "${PROJECT_DIR}/scripts/train_stage2_r1_oracle_sparse_value.py" \
  --device auto \
  "$@"
