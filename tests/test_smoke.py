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
from qbanknote.metrics import run_kl_self_check, run_mw_self_check  # noqa: E402
from qbanknote.model import HybridModel  # noqa: E402
from qbanknote.paths import find_project_root  # noqa: E402
from qbanknote.weights import (  # noqa: E402
    WeightAnsatzMismatchError,
    cv_weight_path,
    metric_weight_path,
    validate_ansatz_weight_compatibility,
)
from qbanknote.classification import evaluate_predictions, predictions_to_labels  # noqa: E402
from qbanknote.evaluation import PhaseSpec, load_phase_spec, summarize_results  # noqa: E402
from qbanknote.iqm import build_iqm_estimator_model  # noqa: E402
from qbanknote.stats import (  # noqa: E402
    select_protocol_from_pilot,
    sign_test_exact,
    wilcoxon_signed_rank_exact,
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
        for ansatz in ("odra", "simulator"):
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
    print("All smoke tests passed.")
