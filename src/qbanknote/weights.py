"""Cross-validation checkpoint discovery and safe loading."""

from __future__ import annotations

import glob
import importlib
import pickle
import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit_machine_learning.connectors import TorchConnector

from qbanknote.ansatzes import (
    LEGACY_PARAM_COUNTS,
    TRIMMED_PARAM_COUNTS,
    odra_ansatz,
    param_count_for_ansatz,
    simulator_ansatz,
    trimmed_reverse_q0_param_count,
)
from qbanknote.model import HybridModel
from qbanknote.paths import find_project_root

ModelName = Literal["Odra", "Simulator"]
AnsatzKey = Literal["odra", "simulator"]
Condition = Literal["Ideal", "Noise"]


class WeightAnsatzMismatchError(ValueError):
    """Raised when checkpoint parameter count does not match the ansatz."""


def cv_weights_dir(root: Path | None = None) -> Path:
    return find_project_root(root) / "cross_validation" / "Models" / "Weights"


def cv_weight_path(
    depth: int,
    model_name: ModelName,
    fold: int,
    *,
    epoch: int = 30,
    condition: Condition = "Ideal",
    simulator_uses_ideal_suffix: bool = False,
    root: Path | None = None,
) -> Path:
    """Canonical path to a CV checkpoint."""
    if condition == "Ideal":
        if model_name == "Simulator" and simulator_uses_ideal_suffix:
            filename = (
                f"{model_name}_fold_{fold}_depth_{depth}_epoch_{epoch}_ideal_weights.pth"
            )
        else:
            filename = f"{model_name}_fold_{fold}_depth_{depth}_epoch_{epoch}_weights.pth"
    else:
        filename = (
            f"{model_name}_fold_{fold}_depth_{depth}_noise_epoch_{epoch}_weights.pth"
        )
    return cv_weights_dir(root) / f"depth {depth}" / model_name / f"fold_{fold}" / filename


def metric_model_name(ansatz_name: AnsatzKey) -> ModelName:
    return "Odra" if ansatz_name == "odra" else "Simulator"


def metric_weight_path(
    depth: int,
    ansatz_name: AnsatzKey,
    fold: int,
    *,
    epoch: int = 30,
    simulator_uses_ideal_suffix: bool = False,
    root: Path | None = None,
) -> Path:
    """Checkpoint path used by the IQM metric test workflow."""
    return cv_weight_path(
        depth,
        metric_model_name(ansatz_name),
        fold,
        epoch=epoch,
        condition="Ideal",
        simulator_uses_ideal_suffix=simulator_uses_ideal_suffix and ansatz_name == "simulator",
        root=root,
    )


def _torch_load(path: Path):
    if torch.__dict__.get("_utils") is None:
        torch.__dict__["_utils"] = importlib.import_module("torch._utils")

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except (AttributeError, pickle.UnpicklingError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _unwrap_state_dict(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def strip_quantum_layer_prefix(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        if key.startswith("quantum_layer."):
            out[key.replace("quantum_layer.", "", 1)] = value
        else:
            out[key] = value
    return out


def load_checkpoint_hybrid(model: HybridModel, path: str | Path) -> None:
    """Load a checkpoint into a ``HybridModel``, tolerating nested state dicts."""
    raw = _unwrap_state_dict(_torch_load(Path(path)))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict checkpoint at {path}")
    try:
        model.load_state_dict(raw, strict=True)
        return
    except Exception:
        pass
    stripped = strip_quantum_layer_prefix(raw)
    model.quantum_layer.load_state_dict(stripped, strict=True)


def load_checkpoint_connector(connector: TorchConnector, path: str | Path) -> None:
    """Load a checkpoint into a bare ``TorchConnector``."""
    raw = _unwrap_state_dict(_torch_load(Path(path)))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict checkpoint at {path}")
    stripped = strip_quantum_layer_prefix(raw)
    connector.load_state_dict(stripped, strict=True)


def discover_epoch_weights(
    epoch: int = 30,
    *,
    root: Path | None = None,
    recursive: bool = True,
) -> list[Path]:
    """Glob all CV checkpoints matching ``*epoch_{epoch}_weights.pth``."""
    pattern = f"**/epoch_{epoch}_weights.pth" if recursive else f"*epoch_{epoch}_weights.pth"
    base = cv_weights_dir(root)
    return sorted(Path(p) for p in glob.glob(str(base / pattern), recursive=recursive))


def infer_weight_metadata(path: Path | str) -> dict[str, object]:
    """Parse depth, fold, model, and noise flag from a checkpoint filename."""
    path = Path(path)
    text = str(path)
    m_depth = re.search(r"depth[_ ](\d+)", text)
    m_fold = re.search(r"fold[_ ](\d+)", text)
    if not m_depth or not m_fold:
        raise ValueError(f"Could not parse depth/fold from {path}")

    environment: ModelName = "Odra" if "Odra" in text else "Simulator"
    return {
        "depth": int(m_depth.group(1)),
        "fold": int(m_fold.group(1)),
        "model": environment,
        "noise": "noise" in path.name.lower(),
        "path": path,
    }


def validate_ansatz_weight_compatibility(
    ansatz_circuit: QuantumCircuit,
    weight_tensor: torch.Tensor | np.ndarray,
    *,
    n_qubits: int = 5,
    depth: int | None = None,
    variant: Literal["trimmed", "legacy", "auto"] = "auto",
) -> None:
    """
    Ensure a weight vector matches the ansatz parameter count.

    When ``variant='auto'``, checks against known trimmed and legacy counts.
    """
    n_ansatz = param_count_for_ansatz(ansatz_circuit)
    n_weights = int(np.asarray(weight_tensor).size)
    if n_ansatz == n_weights:
        return

    hints: list[str] = []
    if depth is not None:
        if TRIMMED_PARAM_COUNTS.get(depth) == n_ansatz and LEGACY_PARAM_COUNTS.get(depth) == n_weights:
            hints.append(
                f"depth {depth}: ansatz is trimmed ({n_ansatz} params) but weights "
                f"look legacy ({n_weights} params). Use legacy ring ansatz or CV weights."
            )
        elif LEGACY_PARAM_COUNTS.get(depth) == n_ansatz and TRIMMED_PARAM_COUNTS.get(depth) == n_weights:
            hints.append(
                f"depth {depth}: ansatz is legacy ({n_ansatz} params) but weights "
                f"look trimmed ({n_weights} params). Use simulator_ansatz/odra_ansatz."
            )

    if variant == "trimmed" and depth is not None:
        expected = trimmed_reverse_q0_param_count(n_qubits, depth)
        if n_ansatz != expected:
            hints.append(f"trimmed ansatz expected {expected} params at depth {depth}.")

    msg = (
        f"Weight/ansatz mismatch: ansatz has {n_ansatz} parameters, "
        f"weights have {n_weights}."
    )
    if hints:
        msg += " " + " ".join(hints)
    raise WeightAnsatzMismatchError(msg)


def extract_weight_tensor(
    state_dict: dict,
    ansatz_circuit: QuantumCircuit,
    *,
    validate: bool = True,
) -> np.ndarray:
    """Pull the trainable weight vector from a HybridModel state_dict."""
    n_weights = param_count_for_ansatz(ansatz_circuit)
    candidate_keys = [
        "quantum_layer._weights",
        "quantum_layer.weight",
        "_weights",
    ]
    weight = None
    for key in candidate_keys:
        if key in state_dict:
            weight = state_dict[key]
            break
    if weight is None:
        for value in state_dict.values():
            if hasattr(value, "numel") and value.numel() == n_weights:
                weight = value
                break
    if weight is None:
        raise KeyError(
            f"Could not find a {n_weights}-element weight tensor. "
            f"Keys: {list(state_dict.keys())}"
        )

    arr = np.asarray(
        weight.detach().cpu().numpy() if hasattr(weight, "detach") else weight,
        dtype=float,
    )
    if validate:
        validate_ansatz_weight_compatibility(ansatz_circuit, arr)
    return arr


def load_trained_weights(
    pth_path: str | Path,
    ansatz_circuit: QuantumCircuit,
    *,
    map_location: str | torch.device = "cpu",
    validate: bool = True,
) -> np.ndarray:
    """Load weight vector from a ``.pth`` checkpoint."""
    state = torch.load(pth_path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and any(isinstance(v, torch.Tensor) for v in state.values()):
        return extract_weight_tensor(state, ansatz_circuit, validate=validate)
    if hasattr(state, "detach"):
        arr = np.asarray(state.detach().cpu().numpy(), dtype=float)
        if validate:
            validate_ansatz_weight_compatibility(ansatz_circuit, arr)
        return arr
    raise TypeError(f"Unsupported checkpoint format in {pth_path}")


def build_ansatz_for_model(model_name: ModelName, n_qubits: int, depth: int) -> QuantumCircuit:
    if model_name == "Odra":
        return odra_ansatz(n_qubits, depth)
    return simulator_ansatz(n_qubits, depth)


def load_cv_model(
    depth: int,
    model_name: ModelName,
    fold: int,
    *,
    num_qubits: int = 5,
    epoch: int = 30,
    condition: Condition = "Ideal",
    root: Path | None = None,
    gradient_backend: Literal["param_shift", "reverse"] = "param_shift",
    map_location: str | torch.device = "cpu",
) -> tuple[HybridModel, Path]:
    """Construct ``HybridModel`` and load a CV checkpoint."""
    ansatz = build_ansatz_for_model(model_name, num_qubits, depth)
    model = HybridModel(ansatz, num_qubits, gradient_backend=gradient_backend)
    weight_path = cv_weight_path(
        depth, model_name, fold, epoch=epoch, condition=condition, root=root
    )
    if not weight_path.exists():
        raise FileNotFoundError(weight_path)
    state = torch.load(weight_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, weight_path


def load_cv_weights_into_model(
    model: HybridModel,
    depth: int,
    model_name: ModelName,
    fold: int,
    *,
    epoch: int = 30,
    condition: Condition = "Ideal",
    root: Path | None = None,
    map_location: str | torch.device = "cpu",
) -> Path:
    """Load CV weights into an existing ``HybridModel``."""
    weight_path = cv_weight_path(
        depth, model_name, fold, epoch=epoch, condition=condition, root=root
    )
    if not weight_path.exists():
        raise FileNotFoundError(weight_path)
    state = torch.load(weight_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return weight_path
