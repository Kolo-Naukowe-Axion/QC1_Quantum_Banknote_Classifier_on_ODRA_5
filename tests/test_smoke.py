"""Smoke tests for the qbanknote package (no hardware)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
    validate_ansatz_weight_compatibility,
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


if __name__ == "__main__":
    test_imports()
    test_trimmed_param_counts()
    test_legacy_param_counts()
    test_hybrid_model_construction()
    test_fold_data_loads()
    test_weight_mismatch_guard()
    test_cv_weight_path_format()
    test_metric_self_checks()
    print("All smoke tests passed.")
