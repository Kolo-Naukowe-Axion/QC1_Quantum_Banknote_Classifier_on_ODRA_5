#!/usr/bin/env bash
# Unattended state-tomography fidelity on IQM Spark — both ansatze, no weights.
#
# Uses random theta ~ Uniform[0, 2*pi] (same as MW/KL). Defaults match the
# existing pilot recommendations: shots=1024, samples=10, iterations=2.
#
# Usage (from repository root):
#   export IQM_TOKEN="..."
#   ./scripts/run_iqm_fidelity_unattended.sh
#
# Foreground:
#   DETACH=0 ./scripts/run_iqm_fidelity_unattended.sh
#
# Environment overrides:
#   RUN_ID=fidelity_hardware
#   DEPTHS="2 4 6"
#   ANSATZES="ansatz_odra ansatz_simulator"
#   SAMPLES=10
#   SHOTS=1024
#   ITERATIONS=2
#   DETACH=1|0
#   IQM_URL=https://odra5.e-science.pl/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python)
else
  echo "WARNING: uv not found; falling back to python3" >&2
  PYTHON=(python3)
fi

RUN_ID="${RUN_ID:-fidelity_hardware}"
DEPTHS="${DEPTHS:-2 4 6}"
ANSATZES="${ANSATZES:-ansatz_odra ansatz_simulator}"
SAMPLES="${SAMPLES:-10}"
SHOTS="${SHOTS:-1024}"
ITERATIONS="${ITERATIONS:-2}"
SEED="${SEED:-42}"
HARDWARE_RETRIES="${HARDWARE_RETRIES:-6}"
RETRY_WAIT_SECONDS="${RETRY_WAIT_SECONDS:-60}"
RETRY_MAX_WAIT_SECONDS="${RETRY_MAX_WAIT_SECONDS:-600}"
DETACH="${DETACH:-1}"
IQM_URL="${IQM_URL:-https://odra5.e-science.pl/}"

RUN_DIR="${ROOT}/evaluation_and_comparison/iqm_spark/iqm_fidelity_outputs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/fidelity_unattended_${STAMP}.log"
PID_FILE="${LOG_DIR}/fidelity_unattended.pid"
LATEST_LOG="${LOG_DIR}/latest.log"

# shellcheck disable=SC2206
DEPTH_ARR=(${DEPTHS})
# shellcheck disable=SC2206
ANSATZ_ARR=(${ANSATZES})
N_STATES=$((${#ANSATZ_ARR[@]} * ${#DEPTH_ARR[@]} * SAMPLES * ITERATIONS))
EST_HOURS="$(awk -v n="${N_STATES}" 'BEGIN { printf "%.1f", n * 3.5 / 60.0 }')"

if [[ -z "${IQM_TOKEN:-}" ]]; then
  echo "ERROR: set IQM_TOKEN before launching." >&2
  echo "Example: export IQM_TOKEN=\"your_token\"" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" && "${OLD_PID}" != "$$" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "ERROR: fidelity unattended job already running (pid ${OLD_PID})." >&2
    echo "  log:  ${LATEST_LOG}" >&2
    echo "  stop: kill ${OLD_PID}" >&2
    exit 1
  fi
fi

if [[ "${DETACH}" == "1" ]]; then
  echo "Launching fidelity in background (DETACH=1)."
  echo "  run dir: ${RUN_DIR}"
  echo "  log:     ${LOG_FILE}"
  echo "  est:     ~${EST_HOURS} h for ${N_STATES} tomography states (no weights)"
  echo
  echo "Monitor:   tail -f ${LOG_FILE}"
  echo "Stop:      kill \$(cat ${PID_FILE})"
  echo

  export IQM_TOKEN IQM_URL
  export RUN_ID DEPTHS ANSATZES SAMPLES SHOTS ITERATIONS SEED
  export HARDWARE_RETRIES RETRY_WAIT_SECONDS RETRY_MAX_WAIT_SECONDS

  ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG}"
  DETACH=0 nohup "$0" >"${LOG_FILE}" 2>&1 &
  echo $! >"${PID_FILE}"
  echo "Started pid $(cat "${PID_FILE}")"
  exit 0
fi

cleanup_pid() {
  rm -f "${PID_FILE}"
}
trap cleanup_pid EXIT

echo "=== Fidelity unattended study ==="
echo "Started UTC:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Python:       ${PYTHON[*]}"
echo "Run ID:       ${RUN_ID}"
echo "Run output:   ${RUN_DIR}"
echo "Ansatzes:     ${ANSATZES}"
echo "Depths:       ${DEPTHS}"
echo "Samples:      ${SAMPLES}"
echo "Shots:        ${SHOTS}"
echo "Iterations:   ${ITERATIONS}"
echo "Est. states:  ${N_STATES}  (~${EST_HOURS} h @ 3.5 min/state)"
echo "Weights:      NOT required (random thetas)"
echo

"${PYTHON[@]}" scripts/run_iqm_fidelity_hardware.py \
  --run-id "${RUN_ID}" \
  --ansatz "${ANSATZ_ARR[@]}" \
  --depth "${DEPTH_ARR[@]}" \
  --samples "${SAMPLES}" \
  --shots "${SHOTS}" \
  --seed "${SEED}" \
  --iterations "${ITERATIONS}" \
  --resume \
  --hardware-retries "${HARDWARE_RETRIES}" \
  --retry-wait-seconds "${RETRY_WAIT_SECONDS}" \
  --retry-max-wait-seconds "${RETRY_MAX_WAIT_SECONDS}" \
  --iqm-url "${IQM_URL}"

echo
echo "Done UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Outputs:  ${RUN_DIR}"
