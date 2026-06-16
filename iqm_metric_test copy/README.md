# autoresearcher-odra-test

Portable bundle for measuring **accuracy** and **F1** of ODRA vs simulator ansätze on **IQM Spark**, with checkpoint-safe CSV outputs and a pilot → final evaluation protocol.

Extracted from the QC1 `tests/ansatz_odra_evaluation` workflow so it can be shared or run in a separate project.

## Layout

```
autoresearcher-odra-test/
  run_cv_experiment.py       # Main runner (pilot or final phase)
  select_protocol.py         # Freeze shots/repeats from a pilot run
  analyze_final_experiment.py
  experiment_lib.py          # Circuits, models, metrics, IQM connection
  experiment_config.toml     # Protocol and depth-specific settings
  odra_test/                 # IQM Spark estimator helpers
  setup/cross_validation/    # Test CSVs + .pth checkpoints (you provide)
  outputs/                   # Generated run artifacts
  pyproject.toml
```

## Install

From this directory:

```bash
cd autoresearcher-odra-test
uv sync
```

Requires **Python 3.12** and an **IQM token** for hardware runs.

## Data prerequisites

Before running, populate `setup/cross_validation/` with fold test CSVs and weight checkpoints. See [setup/cross_validation/README.md](setup/cross_validation/README.md).

Quick copy from the QC1 repo (paths relative to this folder):

```bash
cp -R ../tests/ansatz_odra_evaluation/setup/cross_validation/Data setup/cross_validation/
cp -R ../tests/ansatz_odra_evaluation/setup/cross_validation/Models/Weights setup/cross_validation/Models/
```

## Authentication

Hardware execution needs IQM credentials. Prefer an environment variable:

```bash
export IQM_TOKEN='your-token-here'
```

Alternatively pass `--iqm-token` on the command line, or enter the token when prompted. Do not commit tokens.

The default backend URL is `https://odra5.e-science.pl/` (override in `experiment_config.toml` under `[common].iqm_url`).

## Protocol overview

The experiment compares two ansätze (**odra**, **simulator**) on the same cross-validation folds and checkpoints.

### Phases

| Phase | Purpose | Folds | Shots | Repeats |
|-------|---------|-------|-------|---------|
| **pilot** | Explore shot count stability | 1–2 (config) | Grid: 512, 1024, 2048, 4096 | 10 |
| **final** | Full statistical comparison | 1–5 | Single frozen count | Chosen from pilot |

Edit `[pilot]` and `[final]` in `experiment_config.toml` to change defaults. CLI flags `--shots` and `--repeats` override the config for a single run.

### Fixed compilation settings

Across depths, the config fixes:

- `optimization_level = 1`
- `seed_transpiler = 42`
- `checkpoint_epoch = 30`

This keeps transpilation reproducible and avoids aggressive routing that would track live Spark calibration drift.

### Execution order

Hardware jobs are **interleaved by repeat** (both ansätze shuffled per repeat when `shuffle_execution = true`) to reduce time-drift bias between ansätze.

### Metrics

For each (fold, ansatz, shots, repeat):

1. Load test features/labels and the matching `.pth` checkpoint.
2. Run the hybrid QNN forward pass on IQM Spark (`IQMBackendEstimator`).
3. Threshold predictions at 0 → labels in `{-1, +1}`.
4. Record **accuracy** and **F1** (positive label `+1`).

A **statevector baseline** (ideal simulator) is computed first for each (fold, ansatz) before hardware starts.

### Pilot → protocol selection

After a completed pilot, `select_protocol.py`:

1. Measures shot-to-shot stability (`shot_stability.csv`).
2. Picks the **smallest shot count** where max accuracy/F1 drift between adjacent shot levels stays within `delta_accuracy` / `delta_f1`.
3. Estimates **repeat count** from pilot standard deviations to hit target half-widths (`target_half_width_accuracy`, `target_half_width_f1`).
4. Writes `protocol_recommendation.json` with `chosen_shot` and `chosen_repeats`.

Use those values for the final run (`--shots` / `--repeats` or by editing `[final]` in the config).

## Typical workflow

Run independently for each depth in `{2, 4, 6}`.

### 1. Pilot

```bash
DEPTH=2
uv run python run_cv_experiment.py --phase pilot --depth "${DEPTH}" --run-id "pilot_depth${DEPTH}"
uv run python select_protocol.py --run-dir "outputs/pilot/pilot_depth${DEPTH}"
```

Inspect `outputs/pilot/pilot_depth${DEPTH}/protocol_recommendation.json`.

### 2. Final run

Use the recommended shot count and repeats (example: 1024 shots, 7 repeats):

```bash
DEPTH=2
SHOTS=1024
REPEATS=7
uv run python run_cv_experiment.py \
  --phase final \
  --depth "${DEPTH}" \
  --run-id "final_depth${DEPTH}" \
  --shots "${SHOTS}" \
  --repeats "${REPEATS}"
```

### 3. Analysis

```bash
uv run python analyze_final_experiment.py --run-dir "outputs/final/final_depth${DEPTH}"
```

Repeat the sequence for `DEPTH=4` and `DEPTH=6`.

## Outputs

Each run writes to `outputs/<phase>/<run_id>/`:

| File | Description |
|------|-------------|
| `run_manifest.json` | Frozen run configuration |
| `statevector_results.csv` | Ideal baseline per fold/ansatz |
| `run_level_results.csv` | One row per hardware repeat (resumable) |
| `summary_comparison.csv` | Aggregated fold-level comparison |

After analysis:

| File | Description |
|------|-------------|
| `ansatz_level_summary.csv` | Mean metrics across folds |
| `paired_fold_differences.csv` | ODRA − simulator per fold |
| `paired_tests.csv` | Exact Wilcoxon / sign-test results |

Pilot runs may also include `shot_stability.csv` and `protocol_recommendation.json`.

## CLI reference

### `run_cv_experiment.py`

```text
--phase {pilot,final}   Required
--depth {2,4,6}         Required
--run-id NAME           Output subdirectory (default: timestamped)
--config PATH           Default: experiment_config.toml in this folder
--iqm-token TOKEN       Alternative to IQM_TOKEN env var
--shots N [N ...]       Override config shot list
--repeats N             Override config repeat count
--statevector-only      Skip IQM; compute ideal baseline only
--hardware-retries N    Transient failure retries (default 6)
```

Runs are **checkpoint-safe**: re-running the same `--run-id` skips rows already present in the CSVs.

### `select_protocol.py`

```text
--run-dir PATH          Pilot output directory (required)
--depth N               Optional; inferred from CSV if omitted
```

### `analyze_final_experiment.py`

```text
--run-dir PATH          Final output directory (required)
```

## Adapting for your own ansatz

This bundle is wired for the QC1 ODRA vs simulator study. To use a different ansatz in another project, edit `experiment_lib.py`:

- `odra_ansatz` / `simulator_ansatz` / `ansatz_factory`
- `ANSATZ_NAMES`
- `weight_path()` and checkpoint filename conventions
- `load_fold_test_data()` if your CSV schema differs

The reusable IQM pieces live in `odra_test/` (`IQMBackendEstimator`, counts → expectation, calibration metadata).

## Entrypoints

| Script | Role |
|--------|------|
| `run_cv_experiment.py` | Execute pilot or final evaluation |
| `select_protocol.py` | Choose frozen shots/repeats from pilot |
| `analyze_final_experiment.py` | Fold summaries and paired statistics |
| `experiment_lib.py` | Shared library (imported by the scripts above) |
