#!/usr/bin/env bash
# Star-only state-fidelity pilot + optional final tomography sweep.
#
# Usage:
#   export IQM_TOKEN="..."
#   ./scripts/run_star_fidelity_study.sh
#
# Requires Star CV checkpoints under:
#   cross_validation/Models/Weights/depth <d>/Star/fold_<f>/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PILOT_ID="${PILOT_ID:-fidelity_pilot_star}"
RUN_ID="${RUN_ID:-fidelity_final_star}"
SKIP_PILOT="${SKIP_PILOT:-0}"
SKIP_FINAL="${SKIP_FINAL:-0}"
FORCE_PILOT="${FORCE_PILOT:-0}"

PILOT_DIR="${ROOT}/evaluation_and_comparison/iqm_spark/iqm_fidelity_outputs/pilots/${PILOT_ID}"
PROTOCOL="${PILOT_DIR}/fidelity_protocol_recommendation.json"
RUN_DIR="${ROOT}/evaluation_and_comparison/iqm_spark/iqm_fidelity_outputs/${RUN_ID}"

python scripts/check_star_qpu_readiness.py

if [[ "${SKIP_PILOT}" != "1" ]]; then
  if [[ -f "${PROTOCOL}" && "${FORCE_PILOT}" != "1" ]]; then
    echo "[pilot] Existing recommendation: ${PROTOCOL}"
  else
    echo "[pilot] Starting fidelity pilot for ansatz_star..."
    python scripts/pilot_state_fidelity.py --pilot-id "${PILOT_ID}"
  fi
else
  echo "[pilot] Skipped."
fi

if [[ ! -f "${PROTOCOL}" ]]; then
  echo "ERROR: missing ${PROTOCOL}" >&2
  exit 1
fi

if [[ "${SKIP_FINAL}" != "1" ]]; then
  CHOSEN_SHOTS="$(python - <<'PY' "${PROTOCOL}"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["chosen_shots"])
PY
)"
  CHOSEN_SAMPLES="$(python - <<'PY' "${PROTOCOL}"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["chosen_n_samples"])
PY
)"
  CHOSEN_ITERS="$(python - <<'PY' "${PROTOCOL}"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["chosen_iterations"])
PY
)"
  echo "[final] Fidelity sweep -> ${RUN_DIR}"
  python - <<'PY' \
    "${ROOT}" "${RUN_DIR}" "${CHOSEN_SHOTS}" "${CHOSEN_SAMPLES}" "${CHOSEN_ITERS}"
import os
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import DEFAULT_STAR_ANSATZES, star_ansatz_registry
from qbanknote.iqm import connect_to_iqm_backend
from qbanknote.metrics import run_iqm_fidelity_sweep

run_dir = Path(sys.argv[2])
shots = int(sys.argv[3])
samples = int(sys.argv[4])
iterations = int(sys.argv[5])
url = os.environ.get("IQM_URL", "https://odra5.e-science.pl/").strip()
backend = connect_to_iqm_backend(url, token=os.environ.get("IQM_TOKEN"))

for iteration in range(1, iterations + 1):
    out = run_dir / f"iteration_{iteration}"
    print(f"[final] iteration {iteration}/{iterations} -> {out}")
    run_iqm_fidelity_sweep(
        backend,
        ansatz_fns=star_ansatz_registry(root=ROOT),
        ansatz_names=list(DEFAULT_STAR_ANSATZES),
        depths=[2, 4, 6],
        n_qubits=5,
        n_samples=samples,
        seed=42,
        shots=shots,
        optimization_level=1,
        seed_transpiler=42,
        max_circuits_per_job=275,
        output_dir=out,
        manifest_extra={"iteration": iteration, "protocol_source": "fidelity_pilot_star"},
        verbose=True,
    )
PY
else
  echo "[final] Skipped."
fi

echo "Done."
echo "Protocol: ${PROTOCOL}"
echo "Run data: ${RUN_DIR}"
