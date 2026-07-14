#!/usr/bin/env bash
# Star-only KL expressibility: fixed 120-measurement budget + bootstrap CI.
#
# Protocol: 60 pairs per (ansatz, depth) × 2 iterations = 120 pooled samples,
# then offline percentile bootstrap (5000 trials, 90%/95% CI).
#
# Usage:
#   export IQM_TOKEN="..."
#   ./scripts/run_star_kl_study.sh
#
# Environment overrides match scripts/run_iqm_kl_hardware_study.sh:
#   RUN_ID=kl_hardware_star
#   DEPTHS="2 4 6"
#   SAMPLES=60
#   ITERATIONS=2
#   SHOTS=2048
#   N_BINS=400

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python)
else
  PYTHON=(python3)
fi

export RUN_ID="${RUN_ID:-kl_hardware_star}"
export DEPTHS="${DEPTHS:-2 4 6}"
export SAMPLES="${SAMPLES:-60}"
export ITERATIONS="${ITERATIONS:-2}"

"${PYTHON[@]}" scripts/check_star_qpu_readiness.py --no-require-fidelity-weights
exec ./scripts/run_iqm_kl_hardware_study.sh
