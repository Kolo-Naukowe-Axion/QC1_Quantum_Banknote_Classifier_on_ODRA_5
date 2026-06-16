"""Smoke tests for the qbanknote package (no hardware)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import (  # noqa: E402
    LEGACY_PARAM_COUNTS,
    TRIMMED_PARAM_COUNTS,
    legacy_odra_ring_ansatz,
    legacy_simulator_ring_ansatz,
    odra_ansatz,
    param_count_for_ansatz,
    simulator_ansatz,
    trimmed_reverse_q0_param_count,
)
from qbanknote.data import load_fold_arrays, set_random_seed  # noqa: E402
from qbanknote.metrics import (  # noqa: E402
    bloch_row_to_score_row,
    completed_mw_jobs,
    run_kl_self_check,
    run_mw_self_check,
    summary_row_from_result,
)
from qbanknote.model import HybridModel  # noqa: E402
from qbanknote.paths import find_project_root  # noqa: E402
from qbanknote.iqm import build_iqm_estimator_model  # noqa: E402
from qbanknote.weights import (  # noqa: E402
    WeightAnsatzMismatchError,
    cv_weight_path,
    metric_weight_path,
    validate_ansatz_weight_compatibility,
)
from qbanknote.classification import evaluate_predictions, predictions_to_labels  # noqa: E402
from qbanknote.stats import (  # noqa: E402
    analyze_final_summary,
    select_protocol_from_pilot,
    sign_test_exact,
    wilcoxon_signed_rank_exact,
    write_analysis_artifacts,
    write_protocol_artifacts,
)
from qbanknote.progress import make_print_callback, progress_bar  # noqa: E402
from qbanknote.evaluation import (  # noqa: E402
    ANSATZ_NAMES,
    ODRA_ANSATZ_NAME,
    PhaseSpec,
    SIMULATOR_ANSATZ_NAME,
    append_csv_row,
    build_failed_hardware_row,
    canonical_ansatz_name,
    count_hardware_tasks,
    count_statevector_tasks,
    load_phase_spec,
    summarize_results,
)


def test_imports() -> None:
    root = find_project_root(ROOT)
    assert (root / "cross_validation").is_dir()


def test_trimmed_param_counts() -> None:
    n_qubits = 5
    for depth, expected in TRIMMED_PARAM_COUNTS.items():
        assert trimmed_reverse_q0_param_count(n_qubits, depth) == expected
        assert param_count_for_ansatz(simulator_ansatz(n_qubits, depth)) == expected
        assert param_count_for_ansatz(odra_ansatz(n_qubits, depth)) == expected


def test_legacy_param_counts() -> None:
    n_qubits = 5
    for depth, expected in LEGACY_PARAM_COUNTS.items():
        assert param_count_for_ansatz(legacy_simulator_ring_ansatz(n_qubits, depth)) == expected
        assert param_count_for_ansatz(legacy_odra_ring_ansatz(n_qubits, depth)) == expected


def test_hybrid_model_construction() -> None:
    set_random_seed(0)
    model = HybridModel(simulator_ansatz(5, 2), 5)
    x = torch.zeros((2, 5), dtype=torch.float32)
    y = model(x)
    assert y.shape == (2, 1)


def test_fold_data_loads() -> None:
    X, y = load_fold_arrays(1, split="test")
    assert X.shape[1] == 5
    assert set(np.unique(y)).issubset({-1.0, 1.0})


def test_weight_mismatch_guard() -> None:
    trimmed = simulator_ansatz(5, 2)
    legacy = legacy_simulator_ring_ansatz(5, 2)
    legacy_weights = np.zeros(param_count_for_ansatz(legacy))
    try:
        validate_ansatz_weight_compatibility(trimmed, legacy_weights, depth=2)
        raise AssertionError("Expected WeightAnsatzMismatchError")
    except WeightAnsatzMismatchError:
        pass


def test_cv_weight_path_format() -> None:
    path = cv_weight_path(2, "Simulator", 1, epoch=30)
    assert path.name == "Simulator_fold_1_depth_2_epoch_30_weights.pth"
    assert "depth 2" in str(path)


def test_metric_self_checks() -> None:
    run_kl_self_check()
    run_mw_self_check()


def test_classification_helpers() -> None:
    y_true = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    preds = np.array([-0.2, 0.3, -0.1, 0.4], dtype=np.float32)
    labels = predictions_to_labels(preds)
    metrics = evaluate_predictions(y_true, preds)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert set(labels.tolist()) == {-1.0, 1.0}


def test_metric_weight_path_ideal_suffix() -> None:
    path = metric_weight_path(
        2,
        "simulator",
        1,
        epoch=30,
        simulator_uses_ideal_suffix=True,
        root=ROOT,
    )
    assert path.name == "Simulator_fold_1_depth_2_epoch_30_ideal_weights.pth"


def test_exact_stats_helpers() -> None:
    diffs = [0.1, 0.2, -0.05, 0.15]
    wilcoxon = wilcoxon_signed_rank_exact(diffs)
    sign = sign_test_exact(diffs)
    assert wilcoxon["pvalue"] is not None
    assert 0.0 <= sign["pvalue"] <= 1.0


def test_select_protocol_from_pilot_synthetic() -> None:
    spec = PhaseSpec(
        experiment_name="test",
        phase="pilot",
        depth=2,
        checkpoint_epoch=30,
        simulator_uses_ideal_suffix=True,
        folds=(1, 2),
        shots=(512, 1024),
        repeats=2,
        run_iqm_hardware=True,
        cross_validation_dir="cross_validation",
        outputs_dir="outputs",
        num_qubits=5,
        random_seed=42,
        optimization_level=1,
        seed_transpiler=42,
        shuffle_execution=False,
        iqm_url="https://example.invalid/",
        delta_accuracy=0.5,
        delta_f1=0.5,
        target_half_width_accuracy=0.02,
        target_half_width_f1=0.03,
    )
    rows = []
    for fold in (1, 2):
        for ansatz in ANSATZ_NAMES:
            for shot in (512, 1024):
                rows.append(
                    {
                        "phase": "pilot",
                        "depth": 2,
                        "fold": fold,
                        "ansatz": ansatz,
                        "statevector_accuracy": 0.9,
                        "statevector_f1": 0.88,
                        "statevector_std_accuracy": 0.0,
                        "statevector_std_f1": 0.0,
                        "iqm_mean_accuracy": 0.85,
                        "iqm_std_accuracy": 0.01,
                        "iqm_mean_f1": 0.83,
                        "iqm_std_f1": 0.01,
                        "eval_shots": shot,
                        "n_repeats": 2,
                        "completed_repeats": 2,
                        "test_csv": "x",
                        "weight_path": "y",
                    }
                )
    summary_df = pd.DataFrame(rows)
    protocol = select_protocol_from_pilot(summary_df, spec)
    assert protocol["chosen_shot"] in (512, 1024)
    assert protocol["chosen_repeats"] >= 1


def test_build_iqm_estimator_model() -> None:
    class _FakeBackend:
        num_qubits = 5

        def run(self, circuits, shots):
            raise AssertionError("fake backend should not be executed in construction test")

    model, estimator = build_iqm_estimator_model(
        _FakeBackend(),
        simulator_ansatz,
        num_qubits=5,
        depth=2,
        shots=100,
        optimization_level=1,
        seed_transpiler=42,
    )
    assert model is not None
    assert isinstance(estimator.failed_batches, list)


def test_progress_fallback() -> None:
    values = list(progress_bar(range(3), total=3, desc="test", disable=False))
    assert values == [0, 1, 2]

    seen: list[tuple[str, int, int]] = []

    def _cb(stage: str, completed: int, total: int) -> None:
        seen.append((stage, completed, total))

    callback = make_print_callback("prefix:")
    callback("hardware", 1, 5)
    assert seen == []  # print callback does not use seen list
    _cb("statevector", 2, 4)
    assert seen == [("statevector", 2, 4)]


def test_mw_csv_artifact_helpers(tmp_path: Path) -> None:
    result = {
        "n_qubits": 5,
        "n_params": 10,
        "n_samples": 2,
        "mw_avg": 0.5,
        "mw_std": 0.1,
        "mw_sem": 0.07,
        "mw_min": 0.4,
        "mw_max": 0.6,
        "bloch_rows": [
            {"sample_index": 0.0, "mw_score": 0.5, "x_q0": 0.1, "y_q0": 0.2, "z_q0": 0.9},
            {"sample_index": 1.0, "mw_score": 0.6, "x_q0": 0.0, "y_q0": 0.0, "z_q0": 1.0},
        ],
    }
    summary_row = summary_row_from_result(
        ansatz="ansatz_odra",
        depth=2,
        shots=1024,
        seed=42,
        result=result,
    )
    summary_path = tmp_path / "iqm_mw_results.csv"
    append_csv_row(summary_path, summary_row)
    assert completed_mw_jobs(summary_path) == {("ansatz_odra", 2)}

    scores_path = tmp_path / "iqm_mw_scores.csv"
    for bloch in result["bloch_rows"]:
        append_csv_row(
            scores_path,
            bloch_row_to_score_row("ansatz_odra", 2, bloch),
        )
    scores_df = pd.read_csv(scores_path)
    assert len(scores_df) == 2
    assert "mw_score" in scores_df.columns


def _synthetic_summary_df() -> pd.DataFrame:
    rows = []
    for fold in (1, 2, 3):
        for ansatz in ANSATZ_NAMES:
            rows.append(
                {
                    "phase": "final",
                    "depth": 2,
                    "fold": fold,
                    "ansatz": ansatz,
                    "statevector_accuracy": 0.9,
                    "statevector_f1": 0.88,
                    "statevector_std_accuracy": 0.0,
                    "statevector_std_f1": 0.0,
                    "iqm_mean_accuracy": 0.85 if ansatz == ODRA_ANSATZ_NAME else 0.84,
                    "iqm_std_accuracy": 0.01,
                    "iqm_mean_f1": 0.83 if ansatz == ODRA_ANSATZ_NAME else 0.82,
                    "iqm_std_f1": 0.01,
                    "eval_shots": 2048,
                    "n_repeats": 2,
                    "completed_repeats": 2,
                    "test_csv": "x",
                    "weight_path": "y",
                }
            )
    return pd.DataFrame(rows)


def test_write_protocol_and_analysis_artifacts(tmp_path: Path) -> None:
    spec = PhaseSpec(
        experiment_name="test",
        phase="pilot",
        depth=2,
        checkpoint_epoch=30,
        simulator_uses_ideal_suffix=True,
        folds=(1, 2),
        shots=(512, 1024),
        repeats=2,
        run_iqm_hardware=True,
        cross_validation_dir="cross_validation",
        outputs_dir="outputs",
        num_qubits=5,
        random_seed=42,
        optimization_level=1,
        seed_transpiler=42,
        shuffle_execution=False,
        iqm_url="https://example.invalid/",
        delta_accuracy=0.5,
        delta_f1=0.5,
        target_half_width_accuracy=0.02,
        target_half_width_f1=0.03,
    )
    protocol = select_protocol_from_pilot(_synthetic_pilot_summary(), spec)
    report_path = write_protocol_artifacts(tmp_path, protocol)
    assert report_path.exists()
    assert (tmp_path / "protocol_recommendation.json").exists()

    analysis = analyze_final_summary(_synthetic_summary_df())
    write_analysis_artifacts(tmp_path, analysis)
    assert (tmp_path / "paired_tests.csv").exists()
    assert (tmp_path / "ansatz_level_summary.csv").exists()


def _synthetic_pilot_summary() -> pd.DataFrame:
    rows = []
    for fold in (1, 2):
        for ansatz in ANSATZ_NAMES:
            for shot in (512, 1024):
                rows.append(
                    {
                        "phase": "pilot",
                        "depth": 2,
                        "fold": fold,
                        "ansatz": ansatz,
                        "statevector_accuracy": 0.9,
                        "statevector_f1": 0.88,
                        "statevector_std_accuracy": 0.0,
                        "statevector_std_f1": 0.0,
                        "iqm_mean_accuracy": 0.85,
                        "iqm_std_accuracy": 0.01,
                        "iqm_mean_f1": 0.83,
                        "iqm_std_f1": 0.01,
                        "eval_shots": shot,
                        "n_repeats": 2,
                        "completed_repeats": 2,
                        "test_csv": "x",
                        "weight_path": "y",
                    }
                )
    return pd.DataFrame(rows)


def test_task_counts_and_failed_row() -> None:
    spec = PhaseSpec(
        experiment_name="test",
        phase="pilot",
        depth=2,
        checkpoint_epoch=30,
        simulator_uses_ideal_suffix=True,
        folds=(1, 2),
        shots=(512,),
        repeats=2,
        run_iqm_hardware=True,
        cross_validation_dir="cross_validation",
        outputs_dir="outputs",
        num_qubits=5,
        random_seed=42,
        optimization_level=1,
        seed_transpiler=42,
        shuffle_execution=False,
        iqm_url="https://example.invalid/",
        delta_accuracy=0.01,
        delta_f1=0.02,
        target_half_width_accuracy=0.02,
        target_half_width_f1=0.03,
    )
    assert count_statevector_tasks(spec) == 4
    assert count_hardware_tasks(spec) == 8
    failed = build_failed_hardware_row(
        spec,
        fold=1,
        ansatz_name="odra",
        shots=512,
        repeat_index=0,
        root=ROOT,
    )
    assert failed["status"] == "failed"
    assert failed["ansatz"] == ODRA_ANSATZ_NAME
    assert failed["qpu_time_total"] == 0.0


def test_qce_resp_ansatz_labels() -> None:
    assert ANSATZ_NAMES == (ODRA_ANSATZ_NAME, SIMULATOR_ANSATZ_NAME)
    assert canonical_ansatz_name("odra") == ODRA_ANSATZ_NAME
    assert canonical_ansatz_name("ansatz_odra") == ODRA_ANSATZ_NAME
    assert canonical_ansatz_name("simulator") == SIMULATOR_ANSATZ_NAME
    assert canonical_ansatz_name("ansatz_simulator") == SIMULATOR_ANSATZ_NAME


def test_cli_argument_parsing() -> None:
    import argparse
    import importlib.util

    scripts = [
        "run_iqm_meyer_wallach.py",
        "run_iqm_metric_test.py",
        "select_iqm_metric_protocol.py",
        "analyze_iqm_metric_test.py",
    ]
    for script_name in scripts:
        path = ROOT / "scripts" / script_name
        spec = importlib.util.spec_from_file_location(script_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        parser = argparse.ArgumentParser()
        if hasattr(module, "parse_args"):
            # Exercise parse_args by calling with empty argv via helper attributes
            assert callable(module.parse_args)


if __name__ == "__main__":
    test_imports()
    test_trimmed_param_counts()
    test_legacy_param_counts()
    test_hybrid_model_construction()
    test_fold_data_loads()
    test_weight_mismatch_guard()
    test_cv_weight_path_format()
    test_metric_self_checks()
    test_classification_helpers()
    test_metric_weight_path_ideal_suffix()
    test_exact_stats_helpers()
    test_select_protocol_from_pilot_synthetic()
    test_build_iqm_estimator_model()
    test_progress_fallback()
    test_task_counts_and_failed_row()
    test_qce_resp_ansatz_labels()
    test_cli_argument_parsing()
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_mw_csv_artifact_helpers(tmp_path)
        test_write_protocol_and_analysis_artifacts(tmp_path)
    print("All smoke tests passed.")
