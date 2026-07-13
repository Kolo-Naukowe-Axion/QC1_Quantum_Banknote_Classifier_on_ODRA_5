#!/usr/bin/env bash
# Full KL expressibility study on IQM Spark ODRA 5:
#   1) pilot  -> kl_protocol_recommendation.json
#   2) production sweep on QPU
#   3) offline KL(QPU/Sim/Haar) analysis
#
# Usage (from repository root):
#   export IQM_TOKEN="..."
#   ./scripts/run_iqm_kl_full_study.sh
#
# Resume after interruption:
#   SKIP_PILOT=1 ./scripts/run_iqm_kl_full_study.sh
#   SKIP_PILOT=1 SKIP_PRODUCTION=1 ./scripts/run_iqm_kl_full_study.sh
#
# Environment overrides:
#   PILOT_ID=kl_pilot_paper
#   RUN_ID=kl_full_study
#   PROTOCOL_SCOPE=global|per_ansatz_depth
#   IQM_URL=https://odra5.e-science.pl/
#   SKIP_PILOT=0|1
#   SKIP_PRODUCTION=0|1
#   SKIP_ANALYSIS=0|1
#   FORCE_PILOT=0|1   # re-run pilot even if recommendation exists

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PILOT_ID="${PILOT_ID:-kl_pilot_star}"
RUN_ID="${RUN_ID:-kl_full_star}"
PROTOCOL_SCOPE="${PROTOCOL_SCOPE:-global}"
SKIP_PILOT="${SKIP_PILOT:-0}"
SKIP_PRODUCTION="${SKIP_PRODUCTION:-0}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"
FORCE_PILOT="${FORCE_PILOT:-0}"

PILOT_DIR="${ROOT}/evaluation_and_comparison/iqm_spark/iqm_kl_outputs/pilots/${PILOT_ID}"
PROTOCOL="${PILOT_DIR}/kl_protocol_recommendation.json"
RUN_DIR="${ROOT}/evaluation_and_comparison/iqm_spark/iqm_kl_outputs/${RUN_ID}"

needs_qpu() {
  [[ "${SKIP_PILOT}" != "1" || "${SKIP_PRODUCTION}" != "1" ]]
}

if needs_qpu && [[ -z "${IQM_TOKEN:-}" ]]; then
  echo "ERROR: set IQM_TOKEN before running QPU stages." >&2
  echo "Example: export IQM_TOKEN=\"your_token\"" >&2
  exit 1
fi

echo "=== KL full study ==="
echo "Pilot ID:     ${PILOT_ID}"
echo "Run ID:       ${RUN_ID}"
echo "Pilot output: ${PILOT_DIR}"
echo "Run output:   ${RUN_DIR}"
echo

if [[ "${SKIP_PILOT}" != "1" ]]; then
  if [[ -f "${PROTOCOL}" && "${FORCE_PILOT}" != "1" ]]; then
    echo "[pilot] Existing recommendation found: ${PROTOCOL}"
    echo "        Set FORCE_PILOT=1 to re-run the pilot."
  else
    echo "[pilot] Starting KL pilot (~42 h QPU budget with current defaults)..."
    python scripts/run_iqm_kl_pilot.py \
      --pilot-id "${PILOT_ID}" \
      --protocol-scope "${PROTOCOL_SCOPE}"
  fi
else
  echo "[pilot] Skipped (SKIP_PILOT=1)."
fi

if [[ ! -f "${PROTOCOL}" ]]; then
  echo "ERROR: missing pilot recommendation: ${PROTOCOL}" >&2
  exit 1
fi

if [[ "${SKIP_PRODUCTION}" != "1" ]]; then
  echo "[production] Starting KL expressibility sweep from ${PROTOCOL}..."
  python scripts/run_iqm_kl_expressibility.py \
    --protocol-json "${PROTOCOL}" \
    --run-id "${RUN_ID}"
else
  echo "[production] Skipped (SKIP_PRODUCTION=1)."
fi

if [[ "${SKIP_ANALYSIS}" != "1" ]]; then
  echo "[analysis] Computing KL(QPU/Sim/Haar) offline..."
  python scripts/analyze_iqm_kl_expressibility.py \
    --run-dir "${RUN_DIR}" \
    --protocol-json "${PROTOCOL}"
else
  echo "[analysis] Skipped (SKIP_ANALYSIS=1)."
fi

echo
echo "Done."
echo "Protocol:  ${PROTOCOL}"
echo "Run data:  ${RUN_DIR}"
echo "Analysis:  ${RUN_DIR}/analysis/kl_qpu_sim_haar_comparison.csv"
