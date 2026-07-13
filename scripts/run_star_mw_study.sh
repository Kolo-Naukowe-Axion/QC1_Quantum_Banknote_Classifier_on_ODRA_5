#!/usr/bin/env bash
# Star-only Meyer-Wallach: one command runs pilot + final sweep on IQM Spark.
#
# Usage (from repository root):
#   IQM_TOKEN="your_token" ./scripts/run_star_mw_study.sh
#
# Resume after interruption (pilot done, final in progress):
#   IQM_TOKEN="..." SKIP_PILOT=1 ./scripts/run_star_mw_study.sh
#
# Environment:
#   IQM_TOKEN=...          required (unless SKIP_PILOT=1 and SKIP_FINAL=1)
#   IQM_URL=...            optional (default: https://odra5.e-science.pl/)
#   PILOT_ID=mw_pilot_star
#   RUN_ID=mw_final_star
#   SKIP_SYNC=0|1          skip uv sync
#   SKIP_PILOT=0|1
#   SKIP_FINAL=0|1
#   FORCE_PILOT=0|1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python)
else
  echo "WARNING: uv not found; falling back to python3" >&2
  PYTHON=(python3)
fi

PILOT_ID="${PILOT_ID:-mw_pilot_star}"
RUN_ID="${RUN_ID:-mw_final_star}"
SKIP_SYNC="${SKIP_SYNC:-0}"
SKIP_PILOT="${SKIP_PILOT:-0}"
SKIP_FINAL="${SKIP_FINAL:-0}"
FORCE_PILOT="${FORCE_PILOT:-0}"

PILOT_DIR="${ROOT}/evaluation_and_comparison/iqm_spark/iqm_mw_outputs/pilots/${PILOT_ID}"
PROTOCOL="${PILOT_DIR}/mw_protocol_recommendation.json"
RUN_DIR="${ROOT}/evaluation_and_comparison/iqm_spark/iqm_mw_outputs/${RUN_ID}"

needs_qpu() {
  [[ "${SKIP_PILOT}" != "1" || "${SKIP_FINAL}" != "1" ]]
}

if [[ "${SKIP_SYNC}" != "1" && -f "${ROOT}/pyproject.toml" ]] && command -v uv >/dev/null 2>&1; then
  echo "[setup] uv sync..."
  uv sync
fi

echo "=== Star MW study (pilot + final) ==="
echo "Python:       ${PYTHON[*]}"
echo "Pilot ID:     ${PILOT_ID}"
echo "Run ID:       ${RUN_ID}"
echo "Pilot output: ${PILOT_DIR}"
echo "Final output: ${RUN_DIR}"
echo

if needs_qpu && [[ -z "${IQM_TOKEN:-}" ]]; then
  echo "ERROR: set IQM_TOKEN before running QPU stages." >&2
  echo "Example: IQM_TOKEN=\"your_token\" ./scripts/run_star_mw_study.sh" >&2
  exit 1
fi

"${PYTHON[@]}" scripts/check_star_qpu_readiness.py --require-fidelity-weights=false

if [[ "${SKIP_PILOT}" != "1" ]]; then
  if [[ -f "${PROTOCOL}" && "${FORCE_PILOT}" != "1" ]]; then
    echo "[pilot] Existing recommendation: ${PROTOCOL}"
    echo "        Set FORCE_PILOT=1 to re-run the pilot."
  else
    echo "[pilot] Starting MW pilot for ansatz_star..."
    "${PYTHON[@]}" scripts/run_iqm_mw_pilot.py --pilot-id "${PILOT_ID}" --resume
  fi
else
  echo "[pilot] Skipped (SKIP_PILOT=1)."
fi

if [[ ! -f "${PROTOCOL}" ]]; then
  echo "ERROR: missing pilot recommendation: ${PROTOCOL}" >&2
  exit 1
fi

if [[ "${SKIP_FINAL}" != "1" ]]; then
  read -r CHOSEN_SHOTS CHOSEN_SAMPLES CHOSEN_ITERS < <(
    "${PYTHON[@]}" - "$PROTOCOL" <<'PY'
import json
import sys

protocol = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    protocol["chosen_shots"],
    protocol["chosen_n_samples"],
    protocol["chosen_iterations"],
)
PY
  )
  echo "[final] MW sweep (shots=${CHOSEN_SHOTS}, samples=${CHOSEN_SAMPLES}, iterations=${CHOSEN_ITERS})"
  echo "        -> ${RUN_DIR}"
  "${PYTHON[@]}" scripts/run_iqm_meyer_wallach.py \
    --run-id "${RUN_ID}" \
    --output-dir "${RUN_DIR}" \
    --shots "${CHOSEN_SHOTS}" \
    --samples "${CHOSEN_SAMPLES}" \
    --iterations "${CHOSEN_ITERS}" \
    --resume
else
  echo "[final] Skipped (SKIP_FINAL=1)."
fi

echo
echo "Done."
echo "Protocol: ${PROTOCOL}"
echo "Run data: ${RUN_DIR}"
