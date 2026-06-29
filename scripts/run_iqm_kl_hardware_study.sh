#!/usr/bin/env bash
# Fixed-budget KL expressibility on IQM Spark (depths 2 and 4 only).
# See evaluation_and_comparison/iqm_spark/iqm_kl_hardware_methodology.html
#
# Usage (from repository root):
#   export IQM_TOKEN="..."
#   ./scripts/run_iqm_kl_hardware_study.sh
#
# Second day on a separate run-id:
#   RUN_ID=kl_hardware_day2 ITERATIONS=1 SKIP_BINS=1 ./scripts/run_iqm_kl_hardware_study.sh
#   COMPARE_RUN_DIR=evaluation_and_comparison/iqm_spark/iqm_kl_outputs/kl_hardware_day2 \
#     SKIP_QPU=1 ./scripts/run_iqm_kl_hardware_study.sh
#
# Environment overrides:
#   RUN_ID=kl_hardware
#   DEPTHS="2 4"
#   SAMPLES=60
#   SHOTS=2048
#   N_BINS=400
#   SEED=42
#   ITERATIONS=2
#   COMPARE_RUN_DIR=...   (optional second run for drift analysis)
#   SKIP_BINS=0|1
#   SKIP_QPU=0|1
#   SKIP_ANALYSIS=0|1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-kl_hardware}"
DEPTHS="${DEPTHS:-2 4}"
SAMPLES="${SAMPLES:-60}"
SHOTS="${SHOTS:-2048}"
N_BINS="${N_BINS:-400}"
SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-2}"
SKIP_BINS="${SKIP_BINS:-1}"
SKIP_QPU="${SKIP_QPU:-0}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"
COMPARE_RUN_DIR="${COMPARE_RUN_DIR:-}"

PROTOCOL="${ROOT}/evaluation_and_comparison/iqm_spark/kl_hardware_protocol.json"
RUN_DIR="${ROOT}/evaluation_and_comparison/iqm_spark/iqm_kl_outputs/${RUN_ID}"

# shellcheck disable=SC2206
DEPTH_ARR=(${DEPTHS})
N_ANSATZES=2
N_DEPTHS=${#DEPTH_ARR[@]}
PAIRS=$((N_ANSATZES * N_DEPTHS * SAMPLES * ITERATIONS))
EST_HOURS=$(python3 -c "print(f'{$PAIRS * $SHOTS / 4096 / 15:.1f}')")

echo "=== KL hardware study (fixed budget) ==="
echo "Run ID:       ${RUN_ID}"
echo "Run output:   ${RUN_DIR}"
echo "Depths:       ${DEPTHS}"
echo "Samples/job:  ${SAMPLES}"
echo "Shots:        ${SHOTS}"
echo "Iterations:   ${ITERATIONS}"
echo "Est. pairs:   ${PAIRS}  (~${EST_HOURS} h QPU @ ~4 min/pair @ 4096 shots, scaled by S)"
echo

if [[ "${SKIP_QPU}" != "1" && -z "${IQM_TOKEN:-}" ]]; then
  echo "ERROR: set IQM_TOKEN before running QPU stage." >&2
  exit 1
fi

if [[ "${SKIP_BINS}" != "1" ]]; then
  echo "[bins] Offline bin sensitivity (no QPU)..."
  N_BINS="$(python3 - <<'PY'
from pathlib import Path
import sys

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
from qbanknote.metrics import choose_kl_bins, compute_kl_bin_sensitivity

_, aggregate = compute_kl_bin_sensitivity(
    num_qubits=5,
    n_samples=60,
    bin_grid=[50, 75, 100, 150, 200, 250, 300, 400],
    n_reference_bins=400,
    n_trials=100,
    seed=42,
)
chosen = choose_kl_bins(aggregate, tolerance=0.01)
print(chosen)
PY
)"
  echo "[bins] Using n_bins=${N_BINS}"
else
  echo "[bins] Skipped (SKIP_BINS=1); using N_BINS=${N_BINS}"
fi

if [[ "${SKIP_QPU}" != "1" ]]; then
  echo "[qpu] Starting hardware sweep..."
  # shellcheck disable=SC2206
  python3 scripts/run_iqm_kl_expressibility.py \
    --run-id "${RUN_ID}" \
    --depth ${DEPTH_ARR[@]} \
    --samples "${SAMPLES}" \
    --shots "${SHOTS}" \
    --n-bins "${N_BINS}" \
    --seed "${SEED}" \
    --iterations "${ITERATIONS}" \
    --skip-iteration-precision
else
  echo "[qpu] Skipped (SKIP_QPU=1)."
fi

if [[ "${SKIP_ANALYSIS}" != "1" ]]; then
  echo "[analysis] Offline bootstrap + drift + sim/Haar comparison..."
  ANALYZE_ARGS=(
    --run-dir "${RUN_DIR}"
    --protocol-json "${PROTOCOL}"
    --bootstrap-trials 5000
    --confidence-levels 0.90 0.95
    --n-bins "${N_BINS}"
  )
  if [[ -n "${COMPARE_RUN_DIR}" ]]; then
    ANALYZE_ARGS+=(--compare-run-dir "${COMPARE_RUN_DIR}")
  fi
  python3 scripts/analyze_iqm_kl_hardware.py "${ANALYZE_ARGS[@]}"
else
  echo "[analysis] Skipped (SKIP_ANALYSIS=1)."
fi

echo
echo "Done. Outputs under ${RUN_DIR}/analysis/"
