# Star-only QPU tests (MW, KL, fidelity)

This branch runs **only** `ansatz_star` from [`star.py`](star.py).

## Prerequisites

```bash
export IQM_TOKEN="your_token"
export IQM_URL="https://odra5.e-science.pl/"   # optional, this is the default

uv sync   # opcjonalne — run_star_mw_study.sh robi to samo przy starcie
python scripts/check_star_qpu_readiness.py --require-fidelity-weights=false
```

## 1. Meyer–Wallach (pilot + final)

Jedno polecenie (instaluje zależności, pilot, final):

```bash
IQM_TOKEN="your_token" ./scripts/run_star_mw_study.sh
```

Outputs:
- Pilot: `evaluation_and_comparison/iqm_spark/iqm_mw_outputs/pilots/mw_pilot_star/`
- Final: `evaluation_and_comparison/iqm_spark/iqm_mw_outputs/mw_final_star/`

Analysis notebook: `evaluation_and_comparison/iqm_spark/iqm_meyer_wallach.ipynb`

## 2. KL expressibility (fixed 120-measurement budget + bootstrap CI)

```bash
./scripts/run_star_kl_study.sh
```

Protocol (no adaptive pilot):
- **60 pairs** per `(ansatz_star, depth)` × **2 iterations** → **120 pooled samples**
- **2048 shots**, **400 bins**, seed **42**
- Offline analysis: percentile bootstrap (**5000** trials), **90% / 95%** confidence intervals

Outputs: `evaluation_and_comparison/iqm_spark/iqm_kl_outputs/kl_hardware_star/`

Notebook: `evaluation_and_comparison/iqm_spark/iqm_kl_expressibility.ipynb`

## 3. State fidelity (pilot + final)

**Requires Star CV checkpoints** at:

```text
cross_validation/Models/Weights/depth <d>/Star/fold_<f>/Star_fold_<f>_depth_<d>_epoch_30_weights.pth
```

Train them with cross-validation using `star_ansatz`, then:

```bash
python scripts/check_star_qpu_readiness.py
./scripts/run_star_fidelity_study.sh
```

Outputs:
- Pilot: `evaluation_and_comparison/iqm_spark/iqm_fidelity_outputs/pilots/fidelity_pilot_star/`
- Final: `evaluation_and_comparison/iqm_spark/iqm_fidelity_outputs/fidelity_final_star/`

Notebook: `evaluation_and_comparison/iqm_spark/full_odra_fidelity.ipynb`

## Resume after interruption

```bash
SKIP_PILOT=1 ./scripts/run_star_mw_study.sh
SKIP_PILOT=1 ./scripts/run_star_kl_study.sh
SKIP_PILOT=1 ./scripts/run_star_fidelity_study.sh
```

## Estimated QPU time (star only, depths 2/4/6)

| Test | Pilot | Final |
|------|-------|-------|
| MW | ~4–6 h | ~1–2 h |
| KL | — | ~18 h (3 depths × 120 pairs) |
| Fidelity | ~8–12 h | ~4–8 h |

Star circuits are deeper than the old ring ansatz; add ~20–30% buffer.
