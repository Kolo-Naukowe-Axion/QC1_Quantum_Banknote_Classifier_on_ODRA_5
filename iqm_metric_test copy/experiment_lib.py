from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import getpass
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import random
import time
import tomllib
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import ParserError
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn

from odra_test.calibration_metadata import calibration_set_id
from odra_test.iqm_backend_estimator import IQMBackendEstimator


ANSATZ_NAMES = ("odra", "simulator")
RUN_LEVEL_COLUMNS = [
    "timestamp_utc",
    "status",
    "phase",
    "depth",
    "fold",
    "ansatz",
    "shots",
    "repeat_index",
    "accuracy",
    "f1",
    "weight_path",
    "test_csv",
    "n_samples",
    "qpu_time_total",
    "wall_time_forward_s",
    "calibration_set_id",
    "optimization_level",
    "seed_transpiler",
]
LEGACY_RUN_LEVEL_COLUMNS = [column for column in RUN_LEVEL_COLUMNS if column != "status"]


@dataclass(frozen=True)
class PhaseSpec:
    experiment_name: str
    phase: str
    depth: int
    checkpoint_epoch: int
    simulator_uses_ideal_suffix: bool
    folds: tuple[int, ...]
    shots: tuple[int, ...]
    repeats: int
    run_iqm_hardware: bool
    cross_validation_dir: str
    outputs_dir: str
    num_qubits: int
    random_seed: int
    optimization_level: int
    seed_transpiler: int | None
    shuffle_execution: bool
    iqm_url: str
    delta_accuracy: float
    delta_f1: float
    target_half_width_accuracy: float
    target_half_width_f1: float


def project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not locate project root containing pyproject.toml")


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_phase_spec(
    config_path: str | Path,
    *,
    phase: str,
    depth: int,
    shots_override: list[int] | None = None,
    repeats_override: int | None = None,
    run_iqm_hardware_override: bool | None = None,
) -> PhaseSpec:
    data = load_toml(config_path)
    common = data["common"]
    depth_cfg = data["depths"][str(depth)]
    phase_cfg = data[phase]
    protocol = data["protocol"]

    shots = tuple(int(s) for s in (shots_override or phase_cfg["shots"]))
    repeats = int(repeats_override if repeats_override is not None else phase_cfg["repeats"])
    run_iqm_hardware = (
        bool(run_iqm_hardware_override)
        if run_iqm_hardware_override is not None
        else bool(phase_cfg["run_iqm_hardware"])
    )

    return PhaseSpec(
        experiment_name=str(common["experiment_name"]),
        phase=phase,
        depth=int(depth),
        checkpoint_epoch=int(depth_cfg["checkpoint_epoch"]),
        simulator_uses_ideal_suffix=bool(depth_cfg["simulator_uses_ideal_suffix"]),
        folds=tuple(int(v) for v in phase_cfg["folds"]),
        shots=shots,
        repeats=repeats,
        run_iqm_hardware=run_iqm_hardware,
        cross_validation_dir=str(common["cross_validation_dir"]),
        outputs_dir=str(common["outputs_dir"]),
        num_qubits=int(common["num_qubits"]),
        random_seed=int(common["random_seed"]),
        optimization_level=int(common["optimization_level"]),
        seed_transpiler=common.get("seed_transpiler"),
        shuffle_execution=bool(common["shuffle_execution"]),
        iqm_url=str(common["iqm_url"]),
        delta_accuracy=float(protocol["delta_accuracy"]),
        delta_f1=float(protocol["delta_f1"]),
        target_half_width_accuracy=float(protocol["target_half_width_accuracy"]),
        target_half_width_f1=float(protocol["target_half_width_f1"]),
    )


def build_run_dir(spec: PhaseSpec, run_id: str) -> Path:
    return project_root() / spec.outputs_dir / spec.phase / run_id


def timestamp_run_id(prefix: str) -> str:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def write_manifest(path: Path, spec: PhaseSpec) -> None:
    manifest = asdict(spec)
    manifest["created_utc"] = datetime.now(tz=timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def cv_root(spec: PhaseSpec) -> Path:
    return project_root() / spec.cross_validation_dir


def weights_root(spec: PhaseSpec) -> Path:
    return cv_root(spec) / "Models" / "Weights" / f"depth {spec.depth}"


def fold_test_csv_path(spec: PhaseSpec, fold: int) -> Path:
    return cv_root(spec) / "Data" / f"fold_{fold}" / "test_data.csv"


def weight_path(spec: PhaseSpec, ansatz_name: str, fold: int) -> Path:
    base = weights_root(spec) / ("Odra" if ansatz_name == "odra" else "Simulator") / f"fold_{fold}"
    if ansatz_name == "odra":
        filename = f"Odra_fold_{fold}_depth_{spec.depth}_epoch_{spec.checkpoint_epoch}_weights.pth"
    else:
        suffix = "_ideal" if spec.simulator_uses_ideal_suffix else ""
        filename = (
            f"Simulator_fold_{fold}_depth_{spec.depth}_epoch_{spec.checkpoint_epoch}{suffix}_weights.pth"
        )
    return base / filename


def load_fold_test_data(spec: PhaseSpec, fold: int) -> tuple[np.ndarray, np.ndarray]:
    path = fold_test_csv_path(spec, fold)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != 6:
        raise ValueError(f"Expected 6 columns in {path}, got {data.shape[1]}")
    x = data[:, :5].astype(np.float32)
    y = data[:, 5].astype(np.float32)
    if set(np.unique(y)).issubset({0.0, 1.0}):
        y = 2.0 * y - 1.0
    return x, y


def predictions_to_labels(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions).reshape(-1)
    return np.where(predictions > 0, 1, -1).astype(np.float32)


def ansatz_trimmed_reverse_q0_param_count(n_qubits: int, depth: int) -> int:
    m = depth // 2
    if m == 0:
        return 0
    full = 4 * n_qubits
    last = 3 * n_qubits + 2
    return (m - 1) * full + last


def odra_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    n_macro = depth // 2
    theta = ParameterVector("theta", ansatz_trimmed_reverse_q0_param_count(n_qubits, depth))
    qc = QuantumCircuit(n_qubits)
    p = 0

    for j in range(n_macro):
        last_layer = j == n_macro - 1

        for i in range(n_qubits):
            qc.ry(theta[p + i], i)
        p += n_qubits

        for i in range(n_qubits):
            control = i
            target = (i + 1) % n_qubits
            qc.rz(theta[p + i], target)
            qc.cz(control, target)
        p += n_qubits

        for i in range(n_qubits):
            qc.rx(theta[p + i], i)
        p += n_qubits

        if last_layer:
            for k in range(2):
                i = k
                control = i
                target = (i - 1) % n_qubits
                qc.ry(theta[p + k], target)
                qc.cz(control, target)
            p += 2
        else:
            for i in range(n_qubits):
                control = i
                target = (i - 1) % n_qubits
                qc.ry(theta[p + i], target)
                qc.cz(control, target)
            p += n_qubits

    assert p == len(theta)
    return qc


def simulator_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    n_macro = depth // 2
    theta = ParameterVector("theta", ansatz_trimmed_reverse_q0_param_count(n_qubits, depth))
    qc = QuantumCircuit(n_qubits)
    param_idx = 0

    for j in range(n_macro):
        last_layer = j == n_macro - 1

        for i in range(n_qubits):
            qc.ry(theta[param_idx], i)
            param_idx += 1

        for i in range(n_qubits):
            control = i
            target = (i + 1) % n_qubits
            qc.crx(theta[param_idx], control, target)
            param_idx += 1

        for i in range(n_qubits):
            qc.rx(theta[param_idx], i)
            param_idx += 1

        if last_layer:
            for k in range(2):
                i = k
                control = i
                target = (i - 1) % n_qubits
                qc.cry(theta[param_idx], control, target)
                param_idx += 1
        else:
            for i in range(n_qubits):
                control = i
                target = (i - 1) % n_qubits
                qc.cry(theta[param_idx], control, target)
                param_idx += 1

    assert param_idx == len(theta)
    return qc


def ansatz_factory(name: str):
    if name == "odra":
        return odra_ansatz
    if name == "simulator":
        return simulator_ansatz
    raise KeyError(f"Unknown ansatz: {name}")


class HybridModel(nn.Module):
    def __init__(self, ansatz_circuit: QuantumCircuit, num_qubits: int, random_seed: int):
        super().__init__()
        self.feature_map = self.angle_encoding(num_qubits)
        self.qc = QuantumCircuit(num_qubits)
        self.qc.compose(self.feature_map, qubits=range(num_qubits), inplace=True)
        self.qc.compose(ansatz_circuit, inplace=True)

        input_params = list(self.feature_map.parameters)
        weight_params = list(ansatz_circuit.parameters)
        observable = SparsePauliOp.from_list([("I" * (num_qubits - 1) + "Z", 1)])
        estimator = StatevectorEstimator(seed=random_seed)
        gradient = ParamShiftEstimatorGradient(estimator=estimator)

        self.qnn = EstimatorQNN(
            circuit=self.qc,
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
            gradient=gradient,
        )
        self.quantum_layer = TorchConnector(self.qnn)

    @staticmethod
    def angle_encoding(num_qubits: int) -> QuantumCircuit:
        qc_data = QuantumCircuit(num_qubits)
        input_params = ParameterVector("x", num_qubits)
        for i in range(num_qubits):
            qc_data.ry(input_params[i], i)
        return qc_data

    def forward(self, x):
        return self.quantum_layer(x)


def build_statevector_model(ansatz_name: str, spec: PhaseSpec) -> HybridModel:
    circ = ansatz_factory(ansatz_name)(spec.num_qubits, spec.depth)
    return HybridModel(circ, spec.num_qubits, spec.random_seed)


def build_iqm_model(iqm_backend, ansatz_name: str, spec: PhaseSpec, n_shots: int):
    estimator_options = {
        "shots": n_shots,
        "optimization_level": spec.optimization_level,
    }
    if spec.seed_transpiler is not None:
        estimator_options["seed_transpiler"] = spec.seed_transpiler

    hw_estimator = IQMBackendEstimator(iqm_backend, options=estimator_options)
    hw_ansatz = ansatz_factory(ansatz_name)(spec.num_qubits, spec.depth)
    hw_feature_map = HybridModel(hw_ansatz, spec.num_qubits, spec.random_seed).angle_encoding(spec.num_qubits)

    hw_qc = QuantumCircuit(spec.num_qubits)
    hw_qc.compose(hw_feature_map, qubits=range(spec.num_qubits), inplace=True)
    hw_qc.compose(hw_ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("I" * (spec.num_qubits - 1) + "Z", 1)])
    hw_qnn = EstimatorQNN(
        circuit=hw_qc,
        observables=observable,
        input_params=list(hw_feature_map.parameters),
        weight_params=list(hw_ansatz.parameters),
        estimator=hw_estimator,
    )
    hw_model = TorchConnector(hw_qnn)
    return hw_model, hw_estimator


def _torch_load(path: Path):
    import torch as _torch

    if _torch.__dict__.get("_utils") is None:
        _torch.__dict__["_utils"] = importlib.import_module("torch._utils")

    try:
        return _torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return _torch.load(path, map_location="cpu")
    except (AttributeError, pickle.UnpicklingError):
        return _torch.load(path, map_location="cpu", weights_only=False)


def _unwrap_state_dict(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def _strip_quantum_layer_prefix(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        if key.startswith("quantum_layer."):
            out[key.replace("quantum_layer.", "", 1)] = value
        else:
            out[key] = value
    return out


def load_checkpoint_hybrid(model: HybridModel, path: Path) -> None:
    raw = _unwrap_state_dict(_torch_load(path))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict checkpoint at {path}")
    try:
        model.load_state_dict(raw, strict=True)
        return
    except Exception:
        pass
    stripped = _strip_quantum_layer_prefix(raw)
    model.quantum_layer.load_state_dict(stripped, strict=True)


def load_checkpoint_connector(connector: TorchConnector, path: Path) -> None:
    raw = _unwrap_state_dict(_torch_load(path))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict checkpoint at {path}")
    stripped = _strip_quantum_layer_prefix(raw)
    connector.load_state_dict(stripped, strict=True)


def connect_to_iqm_backend(iqm_url: str, token: str | None = None):
    env_token = os.environ.get("IQM_TOKEN", "").strip()
    if token and env_token:
        raise ValueError("Set either --iqm-token or IQM_TOKEN, not both")
    if token is None and not env_token:
        token = getpass.getpass("Enter IQM Token: ").strip()
    from iqm.qiskit_iqm import IQMProvider

    if token:
        provider = IQMProvider(iqm_url, token=token)
    else:
        provider = IQMProvider(iqm_url)
    return provider.get_backend()


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except ParserError:
        if path.name == "run_level_results.csv":
            return _read_mixed_run_level_csv(path)
        raise


def _infer_row_status(mapping: dict[str, Any]) -> str:
    try:
        qpu_time_total = float(mapping.get("qpu_time_total", 0.0))
    except (TypeError, ValueError):
        qpu_time_total = 0.0
    return "success" if qpu_time_total > 0 else "failed"


def _read_mixed_run_level_csv(path: Path) -> pd.DataFrame:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return pd.DataFrame(columns=RUN_LEVEL_COLUMNS)

    header = rows[0]
    parsed_rows: list[dict[str, Any]] = []
    canonical_from_legacy = [header[0], "status", *header[1:]] if header == LEGACY_RUN_LEVEL_COLUMNS else None

    for raw in rows[1:]:
        if not raw:
            continue

        if header == RUN_LEVEL_COLUMNS and len(raw) == len(RUN_LEVEL_COLUMNS):
            mapping = dict(zip(RUN_LEVEL_COLUMNS, raw))
        elif header == LEGACY_RUN_LEVEL_COLUMNS and len(raw) == len(LEGACY_RUN_LEVEL_COLUMNS):
            mapping = dict(zip(LEGACY_RUN_LEVEL_COLUMNS, raw))
            mapping["status"] = _infer_row_status(mapping)
        elif canonical_from_legacy and len(raw) == len(RUN_LEVEL_COLUMNS):
            mapping = dict(zip(canonical_from_legacy, raw))
        else:
            raise ParserError(f"Could not normalize mixed-schema CSV row in {path}: {raw}")

        parsed_rows.append(mapping)

    frame = pd.DataFrame(parsed_rows)
    return frame.reindex(columns=RUN_LEVEL_COLUMNS)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    new_frame = pd.DataFrame([row])
    if not path.exists() or path.stat().st_size == 0:
        new_frame.to_csv(path, index=False)
        return

    existing = read_csv_or_empty(path)
    columns = list(dict.fromkeys([*existing.columns.tolist(), *new_frame.columns.tolist()]))
    combined = pd.concat(
        [
            existing.reindex(columns=columns),
            new_frame.reindex(columns=columns),
        ],
        ignore_index=True,
    )
    combined.to_csv(path, index=False)


def successful_run_df(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return run_df
    if "status" in run_df.columns:
        return run_df[run_df["status"] == "success"].copy()
    if "qpu_time_total" in run_df.columns:
        return run_df[run_df["qpu_time_total"] > 0].copy()
    return run_df.copy()


def compute_statevector_row(spec: PhaseSpec, fold: int, ansatz_name: str) -> dict[str, Any]:
    X_test, y_test = load_fold_test_data(spec, fold)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    model = build_statevector_model(ansatz_name, spec)
    checkpoint = weight_path(spec, ansatz_name, fold)
    load_checkpoint_hybrid(model, checkpoint)
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).detach().cpu().numpy().flatten()
    labels = predictions_to_labels(predictions)
    return {
        "phase": spec.phase,
        "depth": spec.depth,
        "fold": fold,
        "ansatz": ansatz_name,
        "statevector_accuracy": float(accuracy_score(y_test, labels)),
        "statevector_f1": float(f1_score(y_test, labels, pos_label=1)),
        "test_csv": str(fold_test_csv_path(spec, fold)),
        "weight_path": str(checkpoint),
    }


def compute_hardware_row(
    spec: PhaseSpec,
    iqm_backend,
    *,
    fold: int,
    ansatz_name: str,
    shots: int,
    repeat_index: int,
) -> dict[str, Any]:
    X_test, y_test = load_fold_test_data(spec, fold)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    checkpoint = weight_path(spec, ansatz_name, fold)
    hw_model, hw_estimator = build_iqm_model(iqm_backend, ansatz_name, spec, shots)
    load_checkpoint_connector(hw_model, checkpoint)
    hw_model.eval()

    result_calibration_id = None
    wall_t0 = time.time()
    with torch.no_grad():
        predictions = hw_model(X_tensor).detach().cpu().numpy().flatten()
    wall_time = time.time() - wall_t0

    if hw_estimator.failed_batches:
        first_failure = hw_estimator.failed_batches[0]
        raise RuntimeError(
            "Hardware evaluation failed; no result row was recorded. "
            f"First failed batch error: {first_failure['error']}"
        )

    labels = predictions_to_labels(predictions)

    if hw_estimator.timestamp_history:
        last_meta = hw_estimator.timestamp_history[-1]
        raw_ts = last_meta.get("raw_timestamps")
        if raw_ts is not None:
            result_calibration_id = calibration_set_id(raw_ts)

    return {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "status": "success",
        "phase": spec.phase,
        "depth": spec.depth,
        "fold": fold,
        "ansatz": ansatz_name,
        "shots": shots,
        "repeat_index": repeat_index,
        "accuracy": float(accuracy_score(y_test, labels)),
        "f1": float(f1_score(y_test, labels, pos_label=1)),
        "weight_path": str(checkpoint),
        "test_csv": str(fold_test_csv_path(spec, fold)),
        "n_samples": int(len(y_test)),
        "qpu_time_total": float(hw_estimator.total_qpu_time),
        "wall_time_forward_s": float(wall_time),
        "calibration_set_id": result_calibration_id,
        "optimization_level": spec.optimization_level,
        "seed_transpiler": spec.seed_transpiler,
    }


def summarize_results(
    spec: PhaseSpec,
    *,
    statevector_df: pd.DataFrame,
    run_df: pd.DataFrame,
) -> pd.DataFrame:
    if statevector_df.empty:
        return pd.DataFrame()

    statevector_rows = statevector_df.copy()
    statevector_rows["fold"] = statevector_rows["fold"].astype(int)
    successful_runs = successful_run_df(run_df)

    rows: list[dict[str, Any]] = []
    shots_values = list(spec.shots) if spec.run_iqm_hardware else [math.nan]

    for fold in spec.folds:
        for ansatz_name in ANSATZ_NAMES:
            sv_match = statevector_rows[
                (statevector_rows["fold"] == int(fold)) & (statevector_rows["ansatz"] == ansatz_name)
            ]
            if sv_match.empty:
                continue
            sv_row = sv_match.iloc[0]
            for shot in shots_values:
                if spec.run_iqm_hardware and not successful_runs.empty:
                    hw_group = successful_runs[
                        (successful_runs["fold"] == int(fold))
                        & (successful_runs["ansatz"] == ansatz_name)
                        & (successful_runs["shots"] == int(shot))
                    ]
                else:
                    hw_group = pd.DataFrame()

                if hw_group.empty:
                    mean_acc = mean_f1 = std_acc = std_f1 = float("nan")
                    completed_repeats = 0
                else:
                    mean_acc = float(hw_group["accuracy"].mean())
                    mean_f1 = float(hw_group["f1"].mean())
                    std_acc = float(hw_group["accuracy"].std(ddof=1)) if len(hw_group) > 1 else 0.0
                    std_f1 = float(hw_group["f1"].std(ddof=1)) if len(hw_group) > 1 else 0.0
                    completed_repeats = int(len(hw_group))

                rows.append(
                    {
                        "phase": spec.phase,
                        "depth": spec.depth,
                        "fold": int(fold),
                        "ansatz": ansatz_name,
                        "statevector_accuracy": float(sv_row["statevector_accuracy"]),
                        "statevector_f1": float(sv_row["statevector_f1"]),
                        "statevector_std_accuracy": 0.0,
                        "statevector_std_f1": 0.0,
                        "iqm_mean_accuracy": mean_acc,
                        "iqm_std_accuracy": std_acc,
                        "iqm_mean_f1": mean_f1,
                        "iqm_std_f1": std_f1,
                        "eval_shots": shot,
                        "n_repeats": spec.repeats if spec.run_iqm_hardware else 0,
                        "completed_repeats": completed_repeats,
                        "test_csv": str(sv_row["test_csv"]),
                        "weight_path": str(sv_row["weight_path"]),
                    }
                )

    return pd.DataFrame(rows)


def summarize_across_folds(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    group_cols = ["phase", "depth", "ansatz", "eval_shots"]
    rows = []
    for keys, group in summary_df.groupby(group_cols, dropna=False):
        phase, depth, ansatz, eval_shots = keys
        rows.append(
            {
                "phase": phase,
                "depth": depth,
                "ansatz": ansatz,
                "eval_shots": eval_shots,
                "fold_count": int(group["fold"].nunique()),
                "statevector_accuracy_mean": float(group["statevector_accuracy"].mean()),
                "statevector_f1_mean": float(group["statevector_f1"].mean()),
                "iqm_mean_accuracy_mean": float(group["iqm_mean_accuracy"].mean()),
                "iqm_mean_accuracy_std": float(group["iqm_mean_accuracy"].std(ddof=1)),
                "iqm_mean_f1_mean": float(group["iqm_mean_f1"].mean()),
                "iqm_mean_f1_std": float(group["iqm_mean_f1"].std(ddof=1)),
                "iqm_std_accuracy_mean": float(group["iqm_std_accuracy"].mean()),
                "iqm_std_f1_mean": float(group["iqm_std_f1"].mean()),
                "mean_gap_accuracy": float((group["statevector_accuracy"] - group["iqm_mean_accuracy"]).mean()),
                "mean_gap_f1": float((group["statevector_f1"] - group["iqm_mean_f1"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def iter_hardware_tasks(spec: PhaseSpec) -> list[dict[str, int | str]]:
    rng = random.Random(spec.random_seed)
    tasks: list[dict[str, int | str]] = []
    for fold in spec.folds:
        for shot in spec.shots:
            for repeat_index in range(spec.repeats):
                ansatz_order = list(ANSATZ_NAMES)
                if spec.shuffle_execution:
                    rng.shuffle(ansatz_order)
                for ansatz_name in ansatz_order:
                    tasks.append(
                        {
                            "fold": int(fold),
                            "shots": int(shot),
                            "repeat_index": int(repeat_index),
                            "ansatz": str(ansatz_name),
                        }
                    )
    return tasks


def completed_task_keys(run_df: pd.DataFrame) -> set[tuple[int, str, int, int]]:
    if run_df.empty:
        return set()
    if "status" in run_df.columns:
        completed = run_df[run_df["status"] == "success"]
    elif "qpu_time_total" in run_df.columns:
        # Backward-compatible heuristic for runs created before explicit status tracking:
        # failed hardware attempts were recorded with zero-filled metrics and zero QPU time.
        completed = run_df[run_df["qpu_time_total"] > 0]
    else:
        completed = run_df
    return {
        (int(row.fold), str(row.ansatz), int(row.shots), int(row.repeat_index))
        for row in completed.itertuples(index=False)
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    next_rank = 1
    while start < len(order):
        end = start
        while end + 1 < len(order) and math.isclose(
            values[order[end + 1]], values[order[start]], rel_tol=0.0, abs_tol=1e-12
        ):
            end += 1
        avg_rank = (next_rank + next_rank + (end - start)) / 2.0
        for pos in range(start, end + 1):
            ranks[order[pos]] = avg_rank
        next_rank += end - start + 1
        start = end + 1
    return ranks


def wilcoxon_signed_rank_exact(differences: list[float]) -> dict[str, float | int | None]:
    diffs = [float(value) for value in differences if not math.isclose(float(value), 0.0, abs_tol=1e-12)]
    if not diffs:
        return {
            "n_nonzero": 0,
            "statistic": 0.0,
            "pvalue": 1.0,
            "rank_biserial": 0.0,
            "median_difference": 0.0,
        }

    ranks = average_ranks([abs(v) for v in diffs])
    total_rank = float(sum(ranks))
    positive_rank_sum = float(sum(rank for diff, rank in zip(diffs, ranks) if diff > 0))
    negative_rank_sum = total_rank - positive_rank_sum
    statistic = float(min(positive_rank_sum, negative_rank_sum))
    rank_biserial = float((positive_rank_sum - negative_rank_sum) / total_rank) if total_rank else 0.0

    distribution = []
    for signs in itertools.product((0, 1), repeat=len(ranks)):
        signed_positive_sum = sum(rank for sign, rank in zip(signs, ranks) if sign == 1)
        distribution.append(min(signed_positive_sum, total_rank - signed_positive_sum))
    pvalue = sum(1 for value in distribution if value <= statistic + 1e-12) / len(distribution)

    return {
        "n_nonzero": int(len(diffs)),
        "statistic": statistic,
        "pvalue": float(pvalue),
        "rank_biserial": rank_biserial,
        "median_difference": float(np.median(diffs)),
    }


def sign_test_exact(differences: list[float]) -> dict[str, float | int]:
    diffs = [float(value) for value in differences if not math.isclose(float(value), 0.0, abs_tol=1e-12)]
    n = len(diffs)
    if n == 0:
        return {"n_nonzero": 0, "positive": 0, "negative": 0, "pvalue": 1.0}
    positive = sum(1 for value in diffs if value > 0)
    negative = n - positive
    tail = min(positive, negative)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    pvalue = min(1.0, 2.0 * probability)
    return {
        "n_nonzero": int(n),
        "positive": int(positive),
        "negative": int(negative),
        "pvalue": float(pvalue),
    }


def compute_paired_fold_differences(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["phase", "depth", "eval_shots", "fold"]
    for keys, group in summary_df.groupby(group_cols, dropna=False):
        phase, depth, eval_shots, fold = keys
        if set(group["ansatz"]) != set(ANSATZ_NAMES):
            continue
        odra = group[group["ansatz"] == "odra"].iloc[0]
        simulator = group[group["ansatz"] == "simulator"].iloc[0]
        rows.append(
            {
                "phase": phase,
                "depth": depth,
                "eval_shots": eval_shots,
                "fold": fold,
                "iqm_accuracy_diff_odra_minus_simulator": float(
                    odra["iqm_mean_accuracy"] - simulator["iqm_mean_accuracy"]
                ),
                "iqm_f1_diff_odra_minus_simulator": float(odra["iqm_mean_f1"] - simulator["iqm_mean_f1"]),
                "gap_accuracy_diff_odra_minus_simulator": float(
                    (odra["statevector_accuracy"] - odra["iqm_mean_accuracy"])
                    - (simulator["statevector_accuracy"] - simulator["iqm_mean_accuracy"])
                ),
                "gap_f1_diff_odra_minus_simulator": float(
                    (odra["statevector_f1"] - odra["iqm_mean_f1"])
                    - (simulator["statevector_f1"] - simulator["iqm_mean_f1"])
                ),
                "iqm_std_accuracy_diff_odra_minus_simulator": float(
                    odra["iqm_std_accuracy"] - simulator["iqm_std_accuracy"]
                ),
                "iqm_std_f1_diff_odra_minus_simulator": float(odra["iqm_std_f1"] - simulator["iqm_std_f1"]),
            }
        )
    return pd.DataFrame(rows)


def compute_paired_tests(diffs_df: pd.DataFrame) -> pd.DataFrame:
    if diffs_df.empty:
        return pd.DataFrame()

    rows = []
    metrics = [
        "iqm_accuracy_diff_odra_minus_simulator",
        "iqm_f1_diff_odra_minus_simulator",
        "gap_accuracy_diff_odra_minus_simulator",
        "gap_f1_diff_odra_minus_simulator",
        "iqm_std_accuracy_diff_odra_minus_simulator",
        "iqm_std_f1_diff_odra_minus_simulator",
    ]
    group_cols = ["phase", "depth", "eval_shots"]
    for keys, group in diffs_df.groupby(group_cols, dropna=False):
        phase, depth, eval_shots = keys
        for metric in metrics:
            values = [float(v) for v in group[metric].tolist()]
            wilcoxon = wilcoxon_signed_rank_exact(values)
            sign = sign_test_exact(values)
            rows.append(
                {
                    "phase": phase,
                    "depth": depth,
                    "eval_shots": eval_shots,
                    "metric": metric,
                    "fold_count": int(len(values)),
                    "mean_difference": float(np.mean(values)),
                    "median_difference": float(np.median(values)),
                    "wilcoxon_statistic": wilcoxon["statistic"],
                    "wilcoxon_pvalue": wilcoxon["pvalue"],
                    "wilcoxon_rank_biserial": wilcoxon["rank_biserial"],
                    "sign_test_pvalue": sign["pvalue"],
                    "positive_differences": sign["positive"],
                    "negative_differences": sign["negative"],
                }
            )
    return pd.DataFrame(rows)


def compute_shot_stability(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    shots_values = sorted(int(v) for v in summary_df["eval_shots"].dropna().unique())
    if len(shots_values) < 2:
        return pd.DataFrame(), pd.DataFrame()

    for previous_shot, current_shot in zip(shots_values[:-1], shots_values[1:]):
        prev_group = summary_df[summary_df["eval_shots"] == previous_shot]
        curr_group = summary_df[summary_df["eval_shots"] == current_shot]
        merge_cols = ["phase", "depth", "fold", "ansatz"]
        merged = prev_group.merge(
            curr_group,
            on=merge_cols,
            how="inner",
            suffixes=("_prev", "_curr"),
        )
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "phase": row.phase,
                    "depth": row.depth,
                    "fold": row.fold,
                    "ansatz": row.ansatz,
                    "previous_shot": previous_shot,
                    "current_shot": current_shot,
                    "abs_change_accuracy": abs(row.iqm_mean_accuracy_curr - row.iqm_mean_accuracy_prev),
                    "abs_change_f1": abs(row.iqm_mean_f1_curr - row.iqm_mean_f1_prev),
                }
            )

    detailed = pd.DataFrame(rows)
    if detailed.empty:
        return detailed, pd.DataFrame()

    aggregate_rows = []
    for keys, group in detailed.groupby(["phase", "depth", "previous_shot", "current_shot"], dropna=False):
        phase, depth, previous_shot, current_shot = keys
        aggregate_rows.append(
            {
                "phase": phase,
                "depth": depth,
                "previous_shot": previous_shot,
                "current_shot": current_shot,
                "mean_abs_change_accuracy": float(group["abs_change_accuracy"].mean()),
                "max_abs_change_accuracy": float(group["abs_change_accuracy"].max()),
                "mean_abs_change_f1": float(group["abs_change_f1"].mean()),
                "max_abs_change_f1": float(group["abs_change_f1"].max()),
            }
        )
    return detailed, pd.DataFrame(aggregate_rows)


def choose_shot_from_pilot(summary_df: pd.DataFrame, spec: PhaseSpec) -> int:
    _, aggregate = compute_shot_stability(summary_df)
    if aggregate.empty:
        return int(spec.shots[-1])

    for row in aggregate.sort_values("current_shot").itertuples(index=False):
        if (
            row.max_abs_change_accuracy <= spec.delta_accuracy
            and row.max_abs_change_f1 <= spec.delta_f1
        ):
            return int(row.current_shot)
    return int(max(spec.shots))


def recommended_repeats_from_pilot(summary_df: pd.DataFrame, chosen_shot: int, spec: PhaseSpec) -> dict[str, int]:
    shot_rows = summary_df[(summary_df["eval_shots"] == chosen_shot) & (summary_df["completed_repeats"] > 0)]
    if shot_rows.empty:
        return {"recommended_repeats_accuracy": spec.repeats, "recommended_repeats_f1": spec.repeats}

    conservative_std_acc = float(shot_rows["iqm_std_accuracy"].max())
    conservative_std_f1 = float(shot_rows["iqm_std_f1"].max())

    def _required_repeats(std_value: float, half_width: float) -> int:
        if half_width <= 0 or std_value <= 0:
            return 1
        return int(math.ceil(((1.96 * std_value) / half_width) ** 2))

    return {
        "recommended_repeats_accuracy": _required_repeats(
            conservative_std_acc, spec.target_half_width_accuracy
        ),
        "recommended_repeats_f1": _required_repeats(conservative_std_f1, spec.target_half_width_f1),
    }
