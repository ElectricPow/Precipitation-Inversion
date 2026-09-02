#!/usr/bin/env bash
# Print or sequentially launch the stage-one CFB ablation experiments.
#
# Safety defaults are intentional for a shared GPU server:
#   * no experiment starts unless STAGE1_EXECUTE=1;
#   * experiments run sequentially on the same selected GPUs;
#   * a non-empty output directory is never overwritten.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE1_ABLATION_PHASE="${STAGE1_ABLATION_PHASE:-phase1}"
STAGE1_ABLATION_PARENT="${STAGE1_ABLATION_PARENT:-}"
STAGE1_EXECUTE="${STAGE1_EXECUTE:-0}"
STAGE1_ABLATION_MASTER_PORT_BASE="${STAGE1_ABLATION_MASTER_PORT_BASE:-29710}"

if [[ "${STAGE1_EXECUTE}" != "0" && "${STAGE1_EXECUTE}" != "1" ]]; then
  echo "STAGE1_EXECUTE must be 0 (print only) or 1 (run)." >&2
  exit 2
fi
if [[ ! "${STAGE1_ABLATION_MASTER_PORT_BASE}" =~ ^[0-9]+$ ]] \
  || (( STAGE1_ABLATION_MASTER_PORT_BASE < 1024 \
        || STAGE1_ABLATION_MASTER_PORT_BASE > 65530 )); then
  echo "STAGE1_ABLATION_MASTER_PORT_BASE must be an integer from 1024 to 65530." >&2
  exit 2
fi

# STAGE1_ABLATIONS overrides the phase and accepts comma-separated experiment
# IDs, for example e1,e2. Otherwise phase1 is the recommended first run.
if [[ -n "${STAGE1_ABLATIONS:-}" ]]; then
  IFS=',' read -r -a STAGE1_ABLATION_IDS <<< "${STAGE1_ABLATIONS}"
else
  case "${STAGE1_ABLATION_PHASE}" in
    phase1) STAGE1_ABLATION_IDS=(e0 e1 e2) ;;
    weighted)
      case "${STAGE1_ABLATION_PARENT}" in
        e1) STAGE1_ABLATION_IDS=(e3_from_e1) ;;
        e2) STAGE1_ABLATION_IDS=(e3_from_e2) ;;
        *)
          echo "Weighted phase requires STAGE1_ABLATION_PARENT=e1 or e2." >&2
          exit 2
          ;;
      esac
      ;;
    weak)
      case "${STAGE1_ABLATION_PARENT}" in
        e1) STAGE1_ABLATION_IDS=(e4_from_e1) ;;
        e2) STAGE1_ABLATION_IDS=(e4_from_e2) ;;
        *)
          echo "Weak phase requires STAGE1_ABLATION_PARENT=e1 or e2." >&2
          exit 2
          ;;
      esac
      ;;
    e0n) STAGE1_ABLATION_IDS=(e0_n e0_n_i e0_n_w e0_n_iw) ;;
    ig) STAGE1_ABLATION_IDS=(e0_n_i_g) ;;
    *)
      echo "Unknown STAGE1_ABLATION_PHASE=${STAGE1_ABLATION_PHASE}." >&2
      echo "Choose phase1, weighted, weak, e0n, or ig." >&2
      exit 2
      ;;
  esac
fi

config_and_output() {
  STAGE1_REQUIRED_FILE=""
  case "$1" in
    e0)
      STAGE1_CONFIG="configs/stage1_ablation_e0_baseline.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e0_baseline"
      ;;
    e1)
      STAGE1_CONFIG="configs/stage1_ablation_e1_mask_cfb.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e1_mask_cfb"
      ;;
    e2)
      STAGE1_CONFIG="configs/stage1_ablation_e2_cfb_distance.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e2_cfb_distance"
      ;;
    e3_from_e1)
      STAGE1_CONFIG="configs/stage1_ablation_e3_weighted_from_e1.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e3_weighted_from_e1"
      ;;
    e3_from_e2)
      STAGE1_CONFIG="configs/stage1_ablation_e3_weighted_from_e2.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e3_weighted_from_e2"
      ;;
    e4_from_e1)
      STAGE1_CONFIG="configs/stage1_ablation_e4_weak_from_e1.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e4_weak_from_e1"
      ;;
    e4_from_e2)
      STAGE1_CONFIG="configs/stage1_ablation_e4_weak_from_e2.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e4_weak_from_e2"
      ;;
    e0_n)
      STAGE1_CONFIG="configs/stage1_ablation_e0_n_dbz_valid.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e0_n_dbz_valid"
      STAGE1_REQUIRED_FILE="metadata/normalization/stage1_dbz_valid.json"
      ;;
    e0_n_i)
      STAGE1_CONFIG="configs/stage1_ablation_e0_n_i_intensity.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e0_n_i_intensity"
      STAGE1_REQUIRED_FILE="metadata/normalization/stage1_dbz_valid.json"
      ;;
    e0_n_w)
      STAGE1_CONFIG="configs/stage1_ablation_e0_n_w_weak_cfb.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e0_n_w_weak_cfb"
      STAGE1_REQUIRED_FILE="metadata/normalization/stage1_dbz_valid.json"
      ;;
    e0_n_iw)
      STAGE1_CONFIG="configs/stage1_ablation_e0_n_iw_combined.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e0_n_iw_combined"
      STAGE1_REQUIRED_FILE="metadata/normalization/stage1_dbz_valid.json"
      ;;
    e0_n_i_g)
      STAGE1_CONFIG="configs/stage1_ablation_e0_n_i_g_drdz_002.yaml"
      STAGE1_OUTPUT="outputs/ablations/stage1_e0_n_i_g_drdz_002"
      STAGE1_REQUIRED_FILE="metadata/normalization/stage1_dbz_valid.json"
      ;;
    *)
      echo "Unknown ablation ID '$1'." >&2
      echo "Choose e0/e1/e2, an e3/e4 branch, or e0_n/e0_n_i/e0_n_w/e0_n_iw/e0_n_i_g." >&2
      exit 2
      ;;
  esac
}

if [[ "${STAGE1_EXECUTE}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to GPUs confirmed idle before executing." >&2
  exit 2
fi

echo "[ablation-suite] phase=${STAGE1_ABLATION_PHASE} experiments=${STAGE1_ABLATION_IDS[*]}"
if [[ "${STAGE1_EXECUTE}" == "0" ]]; then
  echo "[ablation-suite] print-only mode; set STAGE1_EXECUTE=1 to train."
else
  echo "[ablation-suite] sequential execution on physical GPUs ${CUDA_VISIBLE_DEVICES}."
fi

for STAGE1_POSITION in "${!STAGE1_ABLATION_IDS[@]}"; do
  STAGE1_ID="${STAGE1_ABLATION_IDS[STAGE1_POSITION]}"
  config_and_output "${STAGE1_ID}"
  STAGE1_CONFIG_PATH="${PROJECT_DIR}/${STAGE1_CONFIG}"
  STAGE1_OUTPUT_PATH="${PROJECT_DIR}/${STAGE1_OUTPUT}"
  STAGE1_RUN_PORT=$((STAGE1_ABLATION_MASTER_PORT_BASE + STAGE1_POSITION))

  if [[ ! -f "${STAGE1_CONFIG_PATH}" ]]; then
    echo "Missing configuration: ${STAGE1_CONFIG_PATH}" >&2
    exit 2
  fi
  if [[ "${STAGE1_EXECUTE}" == "1" && -n "${STAGE1_REQUIRED_FILE}" \
    && ! -f "${PROJECT_DIR}/${STAGE1_REQUIRED_FILE}" ]]; then
    echo "Missing required experiment input: ${PROJECT_DIR}/${STAGE1_REQUIRED_FILE}" >&2
    echo "Build the dbz-valid training normalization statistics before launching." >&2
    exit 2
  fi
  if (( STAGE1_RUN_PORT > 65535 )); then
    echo "Computed master port exceeds 65535." >&2
    exit 2
  fi

  echo "[ablation-suite] ${STAGE1_ID}: ${STAGE1_CONFIG} -> ${STAGE1_OUTPUT}"
  printf 'STAGE1_MASTER_PORT=%q scripts/launch_stage1_ddp.sh --config %q --output-dir %q\n' \
    "${STAGE1_RUN_PORT}" "${STAGE1_CONFIG}" "${STAGE1_OUTPUT}"

  if [[ "${STAGE1_EXECUTE}" == "0" ]]; then
    continue
  fi
  if [[ -d "${STAGE1_OUTPUT_PATH}" ]] \
    && [[ -n "$(find "${STAGE1_OUTPUT_PATH}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty output: ${STAGE1_OUTPUT_PATH}" >&2
    echo "Move it aside or launch this experiment manually with --resume." >&2
    exit 2
  fi

  mkdir -p "${STAGE1_OUTPUT_PATH}"
  STAGE1_MASTER_PORT="${STAGE1_RUN_PORT}" \
    "${PROJECT_DIR}/scripts/launch_stage1_ddp.sh" \
    --config "${STAGE1_CONFIG_PATH}" \
    --output-dir "${STAGE1_OUTPUT_PATH}" \
    2>&1 | tee "${STAGE1_OUTPUT_PATH}/train.log"
done
