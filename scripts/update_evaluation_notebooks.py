#!/usr/bin/env python3
"""Replace duplicated notebook cells with qbanknote imports."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP = """from qbanknote.paths import ensure_importable
ensure_importable()
"""

KL_ANSATZ_CELL = """from qbanknote.ansatzes import (
    ansatz_trimmed_reverse_q0_param_count,
    odra_ansatz as ansatz_odra,
    simulator_ansatz as ansatz_simulator,
)

ANSATZES["ansatz_odra"] = ansatz_odra
ANSATZES["ansatz_simulator"] = ansatz_simulator

for depth in DEPTHS:
    n_params = ansatz_trimmed_reverse_q0_param_count(NUM_QUBITS, depth)
    print(f"depth={depth}: n_params={n_params}")
"""

KL_HELPERS_CELL = """from qbanknote.iqm import connect_to_iqm_backend, transpile_for_backend, normalize_counts
from qbanknote.tomography import (
    all_basis_settings,
    add_tomography_rotations,
    expectation_from_counts,
    reconstruct_rho,
    project_to_physical,
    hardware_overlap,
    run_tomography_jobs,
    tomography_density_matrices,
)
from qbanknote.metrics import (
    haar_pdf_fidelity,
    binned_distributions,
    kl_divergence,
    bind_ansatz,
    sample_hardware_fidelities,
    compute_kl_for_fidelities,
    circuits_per_fidelity_sample,
    total_expressibility_circuits,
    estimate_wall_time_minutes,
    run_kl_self_check as run_self_check,
)
"""

MW_ANSATZ_CELL = KL_ANSATZ_CELL.replace(
    'ANSATZES["ansatz_odra"] = ansatz_odra\nANSATZES["ansatz_simulator"] = ansatz_simulator\n',
    "",
).replace("ANSATZES", "# tuple ANSATZES")

MW_HELPERS_CELL = """from qbanknote.iqm import connect_to_iqm_backend, transpile_for_backend, normalize_counts, run_circuits_on_backend
from qbanknote.metrics import (
    BASIS_ORDER,
    BasisName,
    add_basis_measurement,
    bitstring_qubit_value,
    qubit_expectation_from_counts,
    mw_score_from_bloch,
    estimate_mw_from_hardware_counts,
    compute_iqm_mw_scores,
    meyer_wallach_score,
    single_qubit_reduced_density,
    run_mw_self_check as run_self_check,
)

DEFAULT_IQM_URL = "https://odra5.e-science.pl/"
DEFAULT_SHOTS = 4096
DEFAULT_N_SAMPLES = 20
DEFAULT_SEED = 42
DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_MAX_CIRCUITS_PER_JOB = 275
"""

LED_STATS_CELL2 = """from qbanknote.ansatzes import (
    ansatz_trimmed_reverse_q0_param_count,
    odra_ansatz,
    simulator_ansatz,
)
from qbanknote.model import HybridModel
"""

LED_STATS_DATA = """from qbanknote.data import load_fold_train
"""

FULL_ODRA_IMPORTS = """from qbanknote.paths import ensure_importable
ensure_importable()

from qbanknote.ansatzes import (
    ansatz_trimmed_reverse_q0_param_count,
    simulator_ansatz as ansatz,
    odra_ansatz as ansatz_Odra,
)
from qbanknote.data import set_random_seed, prepare_data
from qbanknote.model import angle_encoding_feature_map
from qbanknote.weights import load_trained_weights
from qbanknote.tomography import (
    all_basis_settings,
    add_tomography_rotations,
    expectation_from_counts,
    reconstruct_rho,
    project_to_physical,
    state_fidelity_pure,
    build_bound_circuit,
    run_tomography_jobs,
)
"""


def set_cell_source(nb: dict, index: int, source: str) -> None:
    nb["cells"][index]["source"] = [line + "\n" for line in source.splitlines()]
    if not source.endswith("\n"):
        nb["cells"][index]["source"][-1] = nb["cells"][index]["source"][-1].rstrip("\n")


def prepend_bootstrap(nb: dict, cell_index: int) -> None:
    src = "".join(nb["cells"][cell_index]["source"])
    if "ensure_importable" not in src:
        set_cell_source(nb, cell_index, BOOTSTRAP + "\n" + src)


def load_nb(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def save_nb(rel: str, nb: dict) -> None:
    path = ROOT / rel
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def update_kl() -> None:
    rel = "evaluation_and_comparison/iqm_spark/iqm_kl_expressibility.ipynb"
    nb = load_nb(rel)
    prepend_bootstrap(nb, 2)
    set_cell_source(nb, 4, KL_ANSATZ_CELL)
    set_cell_source(nb, 6, KL_HELPERS_CELL)
    save_nb(rel, nb)


def update_mw() -> None:
    rel = "evaluation_and_comparison/iqm_spark/iqm_meyer_wallach.ipynb"
    nb = load_nb(rel)
    prepend_bootstrap(nb, 2)
    mw_ansatz = """from qbanknote.ansatzes import (
    ansatz_trimmed_reverse_q0_param_count,
    odra_ansatz as ansatz_odra,
    simulator_ansatz as ansatz_simulator,
)

for depth in DEPTHS:
    n_params = ansatz_trimmed_reverse_q0_param_count(NUM_QUBITS, depth)
    print(f"depth={depth}: n_params={n_params}")
"""
    set_cell_source(nb, 4, mw_ansatz)
    set_cell_source(nb, 6, MW_HELPERS_CELL)
    save_nb(rel, nb)


def update_led_stats() -> None:
    rel = "evaluation_and_comparison/simulator/LED_stats.ipynb"
    nb = load_nb(rel)
    prepend_bootstrap(nb, 1)
    set_cell_source(nb, 2, LED_STATS_CELL2)
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if "def load_fold_train" in src:
            set_cell_source(nb, i, LED_STATS_DATA)
    save_nb(rel, nb)


def replace_function_block(src: str, start_marker: str, end_markers: list[str], replacement: str) -> str:
    if start_marker not in src:
        return src
    start = src.index(start_marker)
    end = len(src)
    for marker in end_markers:
        pos = src.find(marker, start + len(start_marker))
        if pos != -1:
            end = min(end, pos)
    return src[:start] + replacement + src[end:]


def update_full_odra() -> None:
    rel = "evaluation_and_comparison/iqm_spark/full_odra_fidelity.ipynb"
    nb = load_nb(rel)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "def set_random_seed" in src and "RANDOM_SEED" in src:
            set_cell_source(
                nb,
                i,
                FULL_ODRA_IMPORTS
                + "\nRANDOM_SEED = 42\nset_random_seed(RANDOM_SEED)\n",
            )
        elif "def ansatz_trimmed_reverse_q0_param_count" in src:
            set_cell_source(nb, i, "# Ansatz definitions imported from qbanknote.ansatzes (see setup cell).\n")
        elif "def load_trained_weights" in src:
            set_cell_source(
                nb,
                i,
                src.split("def load_trained_weights")[0]
                + "# load_trained_weights imported from qbanknote.weights\n"
                + src[src.index("original_circuit") :],
            )
        elif "def build_bound_circuit" in src:
            set_cell_source(
                nb,
                i,
                "# build_bound_circuit, add_tomography_rotations, all_basis_settings from qbanknote.tomography\n"
                + src[src.index("basis_settings = all_basis_settings") :],
            )
        elif "_PAULI_I = np.array" in src and "def state_fidelity_pure" in src:
            set_cell_source(nb, i, "# Tomography reconstruction helpers imported from qbanknote.tomography\n")
    save_nb(rel, nb)


def update_repeated_shot() -> None:
    rel = "evaluation_and_comparison/iqm_spark/repeated_shot_evaluation.ipynb"
    nb = load_nb(rel)
    replacement = '''from qbanknote.paths import ensure_importable
ensure_importable()

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.metrics import accuracy_score, f1_score

from qbanknote.ansatzes import odra_ansatz as ansatz
from qbanknote.data import load_fold_arrays
from qbanknote.iqm import IQMBackendEstimator, SimpleIQMJob
from qbanknote.model import HybridModel
from qbanknote.paths import find_project_root
from qbanknote.weights import cv_weight_path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("tqdm not available; falling back to plain progress output.")

RANDOM_SEED = 42
NUM_QUBITS = 5
ANSATZ_DEPTH = 2
DEFAULT_SOURCE = "noise"
DEFAULT_CHECKPOINT = "final"
OUTPUT_ROOT = Path("evaluation_and_comparison") / "repeated_shot_outputs"
EXPERIMENT_SPECS = [
    {"label": "100 x 50 shots", "shots": 50, "repeats": 100},
    {"label": "10 x 500 shots", "shots": 500, "repeats": 10},
    {"label": "1 x 5000 shots", "shots": 5000, "repeats": 1},
]


def _project_root():
    return find_project_root()


def available_folds():
    root = _project_root() / "cross_validation" / "Data"
    return sorted(
        int(path.name.split("_")[1])
        for path in root.glob("fold_*")
        if path.is_dir()
    )


def load_fold_data(fold_idx, force_reload=False):
    cache = globals().setdefault("_fold_data_cache", {})
    if not force_reload and fold_idx in cache:
        return cache[fold_idx]
    X_test, y_test = load_fold_arrays(fold_idx, split="test")
    cache[fold_idx] = (X_test, y_test)
    return X_test, y_test


def load_fold_weights(model, fold_idx, strip_prefix=True, source="noise", checkpoint="final"):
    condition = "Noise" if source == "noise" else "Ideal"
    weight_path = cv_weight_path(ANSATZ_DEPTH, "Odra", fold_idx, condition=condition)
    if not weight_path.exists():
        raise FileNotFoundError(weight_path)
    loaded_state = torch.load(weight_path, map_location="cpu", weights_only=True)
    if strip_prefix:
        loaded_state = {
            key.replace("quantum_layer.", "", 1): value
            for key, value in loaded_state.items()
        }
    model.load_state_dict(loaded_state)
    return weight_path


def build_iqm_model(iqm_backend, n_shots, num_qubits=NUM_QUBITS, depth=ANSATZ_DEPTH):
    hw_estimator = IQMBackendEstimator(iqm_backend, options={"shots": n_shots})
    hw_ansatz = ansatz(num_qubits, depth)
    hw_feature_map = HybridModel(hw_ansatz, num_qubits, gradient_backend="reverse").angle_encoding(num_qubits)

    hw_qc = QuantumCircuit(num_qubits)
    hw_qc.compose(hw_feature_map, qubits=range(num_qubits), inplace=True)
    hw_qc.compose(hw_ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("I" * (num_qubits - 1) + "Z", 1)])
    hw_qnn = EstimatorQNN(
        circuit=hw_qc,
        observables=observable,
        input_params=list(hw_feature_map.parameters),
        weight_params=list(hw_ansatz.parameters),
        estimator=hw_estimator,
    )
    hw_model = TorchConnector(hw_qnn)
    return hw_model, hw_estimator


def predictions_to_labels(predictions):
    predictions = np.asarray(predictions).reshape(-1)
    return np.where(predictions > 0, 1, -1).astype(np.float32)


def run_single_fold(iqm_backend, fold_idx, source, checkpoint, n_shots):
    X_test, y_test = load_fold_data(fold_idx)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    hw_model, hw_estimator = build_iqm_model(
        iqm_backend,
        n_shots,
        num_qubits=NUM_QUBITS,
        depth=ANSATZ_DEPTH,
    )
    weight_path = load_fold_weights(
        hw_model,
        fold_idx,
        strip_prefix=True,
        source=source,
        checkpoint=checkpoint,
    )
    hw_model.eval()

    start_time = time.time()
    with torch.no_grad():
        predictions = hw_model(X_test_tensor).detach().cpu().numpy().flatten()
    wall_time_total = time.time() - start_time

    labels = predictions_to_labels(predictions)
    accuracy = accuracy_score(y_test, labels)
    f1 = f1_score(y_test, labels, pos_label=1)

    return {
        "fold": fold_idx,
        "n_samples": len(y_test),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "qpu_time_total": float(hw_estimator.total_qpu_time),
        "qpu_time_per_sample": float(hw_estimator.total_qpu_time / len(y_test)),
        "wall_time_total": float(wall_time_total),
        "weight_path": str(weight_path),
        "y_true": y_test.copy(),
        "y_pred": labels.copy(),
    }


def make_output_dir():
    root = _project_root() / OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / f"run_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


print(f"Project root: {_project_root()}")
print(f"Available folds for depth {ANSATZ_DEPTH}: {available_folds()}")
pd.DataFrame(EXPERIMENT_SPECS)
'''
    set_cell_source(nb, 1, replacement)
    save_nb(rel, nb)


def inject_ansatz_comparison_helpers() -> None:
    rel = "evaluation_and_comparison/ansatz_comparison.ipynb"
    nb = load_nb(rel)
    helper_cell = """from qbanknote.paths import ensure_importable
ensure_importable()

from qbanknote.ansatzes import simulator_ansatz as ansatz, odra_ansatz as ansatz_Odra
from qbanknote.circuit_stats import get_circuit_stats, count_gate_types, print_gate_breakdown
from qbanknote.data import set_random_seed, prepare_data
from qbanknote.iqm import IQMBackendEstimator, SimpleIQMJob
from qbanknote.model import HybridModel
"""
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if "def set_random_seed" in src:
            set_cell_source(nb, i, helper_cell + "\nRANDOM_SEED = 42\nset_random_seed(RANDOM_SEED)\n")
        elif "def prepare_data" in src:
            set_cell_source(nb, i, "# prepare_data imported from qbanknote.data\n")
        elif src.strip().startswith("def ansatz(n_qubits") or src.strip().startswith("def ansatz_Odra"):
            set_cell_source(nb, i, "# ansatz / ansatz_Odra imported from qbanknote.ansatzes\n")
        elif "class HybridModel" in src:
            set_cell_source(nb, i, "# HybridModel imported from qbanknote.model\n")
        elif "class SimpleIQMJob" in src:
            set_cell_source(nb, i, "# IQMBackendEstimator imported from qbanknote.iqm\n")
        elif "def get_circuit_stats" in src:
            set_cell_source(nb, i, "# get_circuit_stats imported from qbanknote.circuit_stats\n")
    save_nb(rel, nb)


def update_model_evaluation() -> None:
    rel = "evaluation_and_comparison/iqm_spark/model_evaluation.ipynb"
    nb = load_nb(rel)
    helper = """from qbanknote.paths import ensure_importable
ensure_importable()

from qbanknote.ansatzes import odra_ansatz as ansatz
from qbanknote.data import load_fold_arrays
from qbanknote.iqm import IQMBackendEstimator, SimpleIQMJob
from qbanknote.model import HybridModel
from qbanknote.weights import cv_weight_path
"""
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if "def ansatz(n_qubits, depth)" in src:
            set_cell_source(nb, i, helper + "# ansatz imported from qbanknote.ansatzes\n")
        elif "class SimpleIQMJob" in src:
            set_cell_source(nb, i, "# IQMBackendEstimator imported from qbanknote.iqm\n")
        elif "class HybridModel" in src:
            set_cell_source(nb, i, "# HybridModel imported from qbanknote.model\n")
        elif "def load_depth2_data" in src:
            set_cell_source(
                nb,
                i,
                """def load_depth2_data():
    X_test, y_test = load_fold_arrays(1, split="test")
    return X_test, y_test
""",
            )
        elif "def load_depth2_weights" in src:
            set_cell_source(
                nb,
                i,
                """def load_depth2_weights(model, strip_prefix=False, source="ideal"):
    condition = "Noise" if source == "noise" else "Ideal"
    weight_path = cv_weight_path(2, "Odra", 1, condition=condition)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    if strip_prefix:
        state = {k.replace("quantum_layer.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return weight_path
""",
            )
    save_nb(rel, nb)


def inject_gate_depth_helpers() -> None:
    rel = "evaluation_and_comparison/simulator/gate_and_depth_comparison.ipynb"
    nb = load_nb(rel)
    helper_cell = """from qbanknote.paths import ensure_importable
ensure_importable()

from qbanknote.ansatzes import simulator_ansatz as ansatz, odra_ansatz as ansatz_Odra
from qbanknote.circuit_stats import get_circuit_stats, count_gate_types, print_gate_breakdown
from qbanknote.data import set_random_seed, prepare_data
from qbanknote.iqm import IQMBackendEstimator, SimpleIQMJob
from qbanknote.model import HybridModel
"""
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if "def set_random_seed" in src:
            set_cell_source(nb, i, helper_cell + "\nRANDOM_SEED = 42\nset_random_seed(RANDOM_SEED)\n")
        elif "def prepare_data" in src:
            set_cell_source(nb, i, "# prepare_data imported from qbanknote.data\n")
        elif src.strip().startswith("def ansatz(n_qubits") or src.strip().startswith("def ansatz_Odra"):
            set_cell_source(nb, i, "# ansatz / ansatz_Odra imported from qbanknote.ansatzes\n")
        elif "class HybridModel" in src:
            set_cell_source(nb, i, "# HybridModel imported from qbanknote.model\n")
        elif "class SimpleIQMJob" in src:
            set_cell_source(nb, i, "# IQMBackendEstimator imported from qbanknote.iqm\n")
        elif "def count_gate_types" in src:
            set_cell_source(nb, i, "# count_gate_types / get_circuit_stats from qbanknote.circuit_stats\n")
    save_nb(rel, nb)


def main() -> None:
    update_kl()
    update_mw()
    update_led_stats()
    update_full_odra()
    update_repeated_shot()
    update_model_evaluation()
    inject_ansatz_comparison_helpers()
    inject_gate_depth_helpers()
    print("Notebook updates complete.")


if __name__ == "__main__":
    main()
