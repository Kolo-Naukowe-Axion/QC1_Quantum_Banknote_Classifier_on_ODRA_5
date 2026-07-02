#!/usr/bin/env bash
# Finish incomplete depth-4 jobs, then run depth-6 in iteration 1 and 2.
#
# Queue order (single process, resume-safe):
#   1. iteration_2 / depth 4  — resume any incomplete (ansatz_odra + ansatz_simulator)
#   2. iteration_1 / depth 6  — both ansatzes
#   3. iteration_2 / depth 6  — both ansatzes
#
# Usage (from repository root):
#   export IQM_TOKEN="..."
#   ./scripts/run_iqm_kl_finish_d4_then_d6.sh
#
# Environment overrides:
#   RUN_ID, SAMPLES, SHOTS, N_BINS, SEED,
#   HARDWARE_RETRIES, RETRY_WAIT_SECONDS, RETRY_MAX_WAIT_SECONDS,
#   SKIP_QPU, SKIP_ANALYSIS, ANALYSIS_N_BINS

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python)
else
  echo "WARNING: uv not found; falling back to python3" >&2
  PYTHON=(python3)
fi

RUN_ID="${RUN_ID:-kl_hardware}"
SAMPLES="${SAMPLES:-60}"
SHOTS="${SHOTS:-2048}"
N_BINS="${N_BINS:-400}"
SEED="${SEED:-42}"
HARDWARE_RETRIES="${HARDWARE_RETRIES:-6}"
RETRY_WAIT_SECONDS="${RETRY_WAIT_SECONDS:-60}"
RETRY_MAX_WAIT_SECONDS="${RETRY_MAX_WAIT_SECONDS:-600}"
SKIP_QPU="${SKIP_QPU:-0}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-1}"
ANALYSIS_N_BINS="${ANALYSIS_N_BINS:-75}"

PROTOCOL="${ROOT}/evaluation_and_comparison/iqm_spark/kl_hardware_protocol.json"
RUN_DIR="${ROOT}/evaluation_and_comparison/iqm_spark/iqm_kl_outputs/${RUN_ID}"
ITER1_DIR="${RUN_DIR}/iteration_1"
ITER2_DIR="${RUN_DIR}/iteration_2"

run_stage() {
  local label="$1"
  local output_dir="$2"
  shift 2
  local -a depths=("$@")

  echo
  echo "=== Stage: ${label} ==="
  echo "Output: ${output_dir}"
  echo "Depths: ${depths[*]}"

  "${PYTHON[@]}" scripts/run_iqm_kl_expressibility.py \
    --output-dir "${output_dir}" \
    --depth "${depths[@]}" \
    --samples "${SAMPLES}" \
    --shots "${SHOTS}" \
    --n-bins "${N_BINS}" \
    --seed "${SEED}" \
    --iterations 1 \
    --skip-iteration-precision \
    --resume \
    --hardware-retries "${HARDWARE_RETRIES}" \
    --retry-wait-seconds "${RETRY_WAIT_SECONDS}" \
    --retry-max-wait-seconds "${RETRY_MAX_WAIT_SECONDS}"
}

echo "=== KL queue: finish depth 4 (iter 2) -> depth 6 (iter 1) -> depth 6 (iter 2) ==="
echo "Run ID:     ${RUN_ID}"
echo "Run output: ${RUN_DIR}"
echo "Samples:    ${SAMPLES}"
echo "Shots:      ${SHOTS}"
echo "Seed:       ${SEED}"

if [[ "${SKIP_QPU}" != "1" && -z "${IQM_TOKEN:-}" ]]; then
  echo "ERROR: set IQM_TOKEN before running QPU stages." >&2
  exit 1
fi

if [[ "${SKIP_QPU}" != "1" ]]; then
  run_stage "Finish depth 4 in iteration 2" "${ITER2_DIR}" 4
  run_stage "Depth 6 in iteration 1" "${ITER1_DIR}" 6
  run_stage "Depth 6 in iteration 2" "${ITER2_DIR}" 6
else
  echo "[qpu] Skipped (SKIP_QPU=1)."
fi

if [[ "${SKIP_ANALYSIS}" != "1" ]]; then
  echo
  echo "=== Offline analysis ==="
  "${PYTHON[@]}" scripts/analyze_iqm_kl_hardware.py \
    --run-dir "${RUN_DIR}" \
    --protocol-json "${PROTOCOL}" \
    --bootstrap-trials 5000 \
    --confidence-levels 0.90 0.95 \
    --n-bins "${ANALYSIS_N_BINS}"
else
  echo "[analysis] Skipped (SKIP_ANALYSIS=1)."
fi

echo
echo "Done. Queue completed under ${RUN_DIR}"
