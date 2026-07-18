#!/usr/bin/env bash
# Unattended KL expressibility on IQM Spark — both ansatze in one launch.
#
# Runs ansatz_odra + ansatz_simulator for the chosen depths, with resume,
# hardware retries, and optional background detach (nohup) so you can leave.
#
# Usage (from repository root):
#   export IQM_TOKEN="..."
#   ./scripts/run_iqm_kl_unattended.sh
#
# Foreground (stay attached):
#   DETACH=0 ./scripts/run_iqm_kl_unattended.sh
#
# Only finish missing depth 6 (faster ~16 h est.):
#   DEPTHS="6" RUN_ID=kl_hardware_depth6 ./scripts/run_iqm_kl_unattended.sh
#
# Resume after a crash / reboot (same RUN_ID):
#   ./scripts/run_iqm_kl_unattended.sh
#
# Environment overrides:
#   RUN_ID=kl_unattended
#   DEPTHS="2 4 6"
#   ANSATZES="ansatz_odra ansatz_simulator"
#   SAMPLES=60
#   SHOTS=2048
#   N_BINS=75
#   ITERATIONS=2
#   SEED=42
#   HARDWARE_RETRIES=6
#   DETACH=1|0          # default 1 — background via nohup
#   SKIP_ANALYSIS=0|1
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

RUN_ID="${RUN_ID:-kl_unattended}"
DEPTHS="${DEPTHS:-2 4 6}"
ANSATZES="${ANSATZES:-ansatz_odra ansatz_simulator}"
SAMPLES="${SAMPLES:-60}"
SHOTS="${SHOTS:-2048}"
N_BINS="${N_BINS:-75}"
ITERATIONS="${ITERATIONS:-2}"
SEED="${SEED:-42}"
HARDWARE_RETRIES="${HARDWARE_RETRIES:-6}"
RETRY_WAIT_SECONDS="${RETRY_WAIT_SECONDS:-60}"
RETRY_MAX_WAIT_SECONDS="${RETRY_MAX_WAIT_SECONDS:-600}"
DETACH="${DETACH:-1}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"
IQM_URL="${IQM_URL:-https://odra5.e-science.pl/}"

RUN_DIR="${ROOT}/evaluation_and_comparison/iqm_spark/iqm_kl_outputs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/kl_unattended_${STAMP}.log"
PID_FILE="${LOG_DIR}/kl_unattended.pid"
LATEST_LOG="${LOG_DIR}/latest.log"

# shellcheck disable=SC2206
DEPTH_ARR=(${DEPTHS})
# shellcheck disable=SC2206
ANSATZ_ARR=(${ANSATZES})
N_ANSATZES=${#ANSATZ_ARR[@]}
N_DEPTHS=${#DEPTH_ARR[@]}
PAIRS=$((N_ANSATZES * N_DEPTHS * SAMPLES * ITERATIONS))
EST_HOURS="$(awk -v p="${PAIRS}" 'BEGIN { printf "%.1f", p * 4.0 / 60.0 }')"

if [[ -z "${IQM_TOKEN:-}" ]]; then
  echo "ERROR: set IQM_TOKEN before launching." >&2
  echo "Example: export IQM_TOKEN=\"your_token\"" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  # Ignore our own pid: the DETACH parent writes it before the child reaches this check.
  if [[ -n "${OLD_PID}" && "${OLD_PID}" != "$$" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "ERROR: KL unattended job already running (pid ${OLD_PID})." >&2
    echo "  log:  ${LATEST_LOG}" >&2
    echo "  stop: kill ${OLD_PID}" >&2
    exit 1
  fi
fi

# Re-exec ourselves in the background so the terminal can be closed safely.
if [[ "${DETACH}" == "1" ]]; then
  echo "Launching in background (DETACH=1)."
  echo "  run dir: ${RUN_DIR}"
  echo "  log:     ${LOG_FILE}"
  echo "  est:     ~${EST_HOURS} h for ${PAIRS} fidelity pairs"
  echo
  echo "Monitor:   tail -f ${LOG_FILE}"
  echo "           (or:   tail -f ${LATEST_LOG})"
  echo "Stop:      kill \$(cat ${PID_FILE})"
  echo

  # Preserve overrides for the detached child.
  export IQM_TOKEN IQM_URL
  export RUN_ID DEPTHS ANSATZES SAMPLES SHOTS N_BINS ITERATIONS SEED
  export HARDWARE_RETRIES RETRY_WAIT_SECONDS RETRY_MAX_WAIT_SECONDS SKIP_ANALYSIS

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

echo "=== KL unattended study ==="
echo "Started UTC:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Python:       ${PYTHON[*]}"
echo "Run ID:       ${RUN_ID}"
echo "Run output:   ${RUN_DIR}"
echo "Ansatzes:     ${ANSATZES}"
echo "Depths:       ${DEPTHS}"
echo "Samples/job:  ${SAMPLES}"
echo "Shots:        ${SHOTS}"
echo "Bins:         ${N_BINS}"
echo "Iterations:   ${ITERATIONS}"
echo "Retries:      ${HARDWARE_RETRIES}"
echo "Est. pairs:   ${PAIRS}  (~${EST_HOURS} h @ ~4 min/pair)"
echo "Log:          ${LOG_FILE}"
echo

echo "[qpu] Starting sweep (both ansatze, resume enabled)..."
"${PYTHON[@]}" scripts/run_iqm_kl_expressibility.py \
  --run-id "${RUN_ID}" \
  --ansatz "${ANSATZ_ARR[@]}" \
  --depth "${DEPTH_ARR[@]}" \
  --samples "${SAMPLES}" \
  --shots "${SHOTS}" \
  --n-bins "${N_BINS}" \
  --seed "${SEED}" \
  --iterations "${ITERATIONS}" \
  --resume \
  --hardware-retries "${HARDWARE_RETRIES}" \
  --retry-wait-seconds "${RETRY_WAIT_SECONDS}" \
  --retry-max-wait-seconds "${RETRY_MAX_WAIT_SECONDS}" \
  --iqm-url "${IQM_URL}"

if [[ "${SKIP_ANALYSIS}" != "1" ]]; then
  echo "[analysis] Offline KL(QPU/Sim/Haar)..."
  "${PYTHON[@]}" scripts/analyze_iqm_kl_expressibility.py \
    --run-dir "${RUN_DIR}" \
    --ansatz "${ANSATZ_ARR[@]}"
else
  echo "[analysis] Skipped (SKIP_ANALYSIS=1)."
fi

echo
echo "Done UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Outputs:  ${RUN_DIR}"
echo "Analysis: ${RUN_DIR}/analysis/kl_qpu_sim_haar_comparison.csv"
