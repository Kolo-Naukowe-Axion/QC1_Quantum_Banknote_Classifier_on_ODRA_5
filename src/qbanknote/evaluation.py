"""CV hardware classification evaluation workflow."""

from __future__ import annotations

import csv
import math
import random
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from pandas.errors import ParserError

from qbanknote.ansatzes import odra_ansatz, simulator_ansatz
from qbanknote.classification import evaluate_predictions
from qbanknote.data import load_fold_arrays
from qbanknote.iqm import build_iqm_estimator_model, calibration_set_id
from qbanknote.model import HybridModel
from qbanknote.paths import find_project_root
from qbanknote.progress import report_progress
from qbanknote.weights import load_checkpoint_connector, load_checkpoint_hybrid, metric_weight_path

ANSATZ_KEYS = ("odra", "simulator")
ODRA_ANSATZ_NAME = "ansatz_odra"
SIMULATOR_ANSATZ_NAME = "ansatz_simulator"
ANSATZ_NAMES = (ODRA_ANSATZ_NAME, SIMULATOR_ANSATZ_NAME)
ANSATZ_ALIASES = {
    "odra": "odra",
    ODRA_ANSATZ_NAME: "odra",
    "simulator": "simulator",
    SIMULATOR_ANSATZ_NAME: "simulator",
}
ANSATZ_LABELS = {
    "odra": ODRA_ANSATZ_NAME,
    "simulator": SIMULATOR_ANSATZ_NAME,
}
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

RETRYABLE_HARDWARE_ERROR_MARKERS = (
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "connection refused",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "httpsconnectionpool",
    "max retries exceeded",
    "batch job failed",
    "failed batch error",
)


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


def ansatz_key(name: str) -> str:
    try:
        return ANSATZ_ALIASES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown ansatz: {name}") from exc


def canonical_ansatz_name(name: str) -> str:
    return ANSATZ_LABELS[ansatz_key(name)]


def normalize_ansatz_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ansatz" not in df.columns:
        return df
    out = df.copy()
    out["ansatz"] = out["ansatz"].map(lambda value: canonical_ansatz_name(str(value)))
    return out


def ansatz_factory(name: str):
    key = ansatz_key(name)
    if key == "odra":
        return odra_ansatz
    if key == "simulator":
        return simulator_ansatz
    raise KeyError(f"Unknown ansatz: {name}")


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


def build_run_dir(spec: PhaseSpec, run_id: str, *, root: Path | None = None) -> Path:
    return find_project_root(root) / spec.outputs_dir / spec.phase / run_id


def timestamp_run_id(prefix: str) -> str:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def write_manifest(path: Path, spec: PhaseSpec) -> None:
    manifest = asdict(spec)
    manifest["created_utc"] = datetime.now(tz=timezone.utc).isoformat()
    path.write_text(
        __import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def cv_root(spec: PhaseSpec, *, root: Path | None = None) -> Path:
    return find_project_root(root) / spec.cross_validation_dir


def fold_test_csv_path(spec: PhaseSpec, fold: int, *, root: Path | None = None) -> Path:
    return cv_root(spec, root=root) / "Data" / f"fold_{fold}" / "test_data.csv"


def weight_path(spec: PhaseSpec, ansatz_name: str, fold: int, *, root: Path | None = None) -> Path:
    return metric_weight_path(
        spec.depth,
        ansatz_key(ansatz_name),
        fold,
        epoch=spec.checkpoint_epoch,
        simulator_uses_ideal_suffix=spec.simulator_uses_ideal_suffix,
        root=root,
    )


def load_fold_test_data(spec: PhaseSpec, fold: int, *, root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    return load_fold_arrays(fold, split="test", root=find_project_root(root))


def build_statevector_model(ansatz_name: str, spec: PhaseSpec) -> HybridModel:
    circuit = ansatz_factory(ansatz_name)(spec.num_qubits, spec.depth)
    return HybridModel(circuit, spec.num_qubits, random_seed=spec.random_seed)


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


def compute_statevector_row(
    spec: PhaseSpec, fold: int, ansatz_name: str, *, root: Path | None = None
) -> dict[str, Any]:
    X_test, y_test = load_fold_test_data(spec, fold, root=root)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    model = build_statevector_model(ansatz_name, spec)
    checkpoint = weight_path(spec, ansatz_name, fold, root=root)
    load_checkpoint_hybrid(model, checkpoint)
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).detach().cpu().numpy().flatten()
    metrics = evaluate_predictions(y_test, predictions)
    return {
        "phase": spec.phase,
        "depth": spec.depth,
        "fold": fold,
        "ansatz": canonical_ansatz_name(ansatz_name),
        "statevector_accuracy": metrics["accuracy"],
        "statevector_f1": metrics["f1"],
        "test_csv": str(fold_test_csv_path(spec, fold, root=root)),
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
    root: Path | None = None,
) -> dict[str, Any]:
    X_test, y_test = load_fold_test_data(spec, fold, root=root)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    checkpoint = weight_path(spec, ansatz_name, fold, root=root)
    hw_model, hw_estimator = build_iqm_estimator_model(
        iqm_backend,
        ansatz_factory(ansatz_name),
        num_qubits=spec.num_qubits,
        depth=spec.depth,
        shots=shots,
        optimization_level=spec.optimization_level,
        seed_transpiler=spec.seed_transpiler,
        random_seed=spec.random_seed,
    )
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

    metrics = evaluate_predictions(y_test, predictions)

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
        "ansatz": canonical_ansatz_name(ansatz_name),
        "shots": shots,
        "repeat_index": repeat_index,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "weight_path": str(checkpoint),
        "test_csv": str(fold_test_csv_path(spec, fold, root=root)),
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

    statevector_rows = normalize_ansatz_labels(statevector_df)
    statevector_rows["fold"] = statevector_rows["fold"].astype(int)
    successful_runs = normalize_ansatz_labels(successful_run_df(run_df))

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


def count_statevector_tasks(spec: PhaseSpec) -> int:
    return len(spec.folds) * len(ANSATZ_NAMES)


def count_hardware_tasks(spec: PhaseSpec) -> int:
    if not spec.run_iqm_hardware:
        return 0
    return len(spec.folds) * len(spec.shots) * spec.repeats * len(ANSATZ_NAMES)


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
        completed = run_df[run_df["qpu_time_total"] > 0]
    else:
        completed = run_df
    return {
        (int(row.fold), canonical_ansatz_name(str(row.ansatz)), int(row.shots), int(row.repeat_index))
        for row in completed.itertuples(index=False)
    }


def is_retryable_hardware_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in RETRYABLE_HARDWARE_ERROR_MARKERS)


def retry_wait_seconds(
    attempt: int,
    *,
    initial_wait_seconds: float,
    max_wait_seconds: float,
) -> float:
    return min(initial_wait_seconds * (2 ** max(attempt - 1, 0)), max_wait_seconds)


def build_failed_hardware_row(
    spec: PhaseSpec,
    *,
    fold: int,
    ansatz_name: str,
    shots: int,
    repeat_index: int,
    root: Path | None = None,
) -> dict[str, Any]:
    checkpoint = weight_path(spec, ansatz_name, fold, root=root)
    return {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "status": "failed",
        "phase": spec.phase,
        "depth": spec.depth,
        "fold": fold,
        "ansatz": canonical_ansatz_name(ansatz_name),
        "shots": shots,
        "repeat_index": repeat_index,
        "accuracy": float("nan"),
        "f1": float("nan"),
        "weight_path": str(checkpoint),
        "test_csv": str(fold_test_csv_path(spec, fold, root=root)),
        "n_samples": int(len(load_fold_test_data(spec, fold, root=root)[1])),
        "qpu_time_total": 0.0,
        "wall_time_forward_s": 0.0,
        "calibration_set_id": None,
        "optimization_level": spec.optimization_level,
        "seed_transpiler": spec.seed_transpiler,
    }


def run_cv_experiment(
    spec: PhaseSpec,
    run_dir: Path,
    *,
    iqm_backend=None,
    iqm_token: str | None = None,
    hardware_retries: int = 6,
    retry_wait_seconds_initial: float = 60.0,
    retry_wait_seconds_max: float = 600.0,
    root: Path | None = None,
    verbose: bool = False,
    progress_callback: Callable[[str, int, int], None] | None = None,
    persist_failed_hardware_rows: bool = True,
) -> Path:
    """Execute statevector baseline and optional IQM hardware tasks with resumable CSVs."""
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir / "run_manifest.json", spec)

    statevector_csv = run_dir / "statevector_results.csv"
    runs_csv = run_dir / "run_level_results.csv"
    summary_csv = run_dir / "summary_comparison.csv"

    statevector_df = normalize_ansatz_labels(read_csv_or_empty(statevector_csv))
    run_df = normalize_ansatz_labels(read_csv_or_empty(runs_csv))

    sv_total = count_statevector_tasks(spec)
    sv_completed = 0
    if not statevector_df.empty:
        for fold in spec.folds:
            for ansatz_name in ANSATZ_NAMES:
                existing = statevector_df[
                    (statevector_df["fold"] == int(fold)) & (statevector_df["ansatz"] == ansatz_name)
                ]
                if not existing.empty:
                    sv_completed += 1

    for fold in spec.folds:
        for ansatz_name in ANSATZ_NAMES:
            if not statevector_df.empty:
                existing = statevector_df[
                    (statevector_df["fold"] == int(fold)) & (statevector_df["ansatz"] == ansatz_name)
                ]
                if not existing.empty:
                    continue

            if verbose:
                print(f"Statevector fold={fold} ansatz={ansatz_name}", flush=True)
            row = compute_statevector_row(spec, fold, ansatz_name, root=root)
            append_csv_row(statevector_csv, row)
            statevector_df = normalize_ansatz_labels(read_csv_or_empty(statevector_csv))
            sv_completed += 1
            report_progress(progress_callback, "statevector", sv_completed, sv_total)
            summary_df = summarize_results(spec, statevector_df=statevector_df, run_df=run_df)
            summary_df.to_csv(summary_csv, index=False)

    if spec.run_iqm_hardware:
        if iqm_backend is None:
            from qbanknote.iqm import connect_to_iqm_backend

            iqm_backend = connect_to_iqm_backend(spec.iqm_url, token=iqm_token)

        done = completed_task_keys(run_df)
        tasks = iter_hardware_tasks(spec)
        hw_total = count_hardware_tasks(spec)
        hw_completed = len(done)

        for index, task in enumerate(tasks, start=1):
            task_key = (task["fold"], task["ansatz"], task["shots"], task["repeat_index"])
            if task_key in done:
                continue

            if verbose:
                print(
                    f"Hardware fold={task['fold']} ansatz={task['ansatz']} "
                    f"shots={task['shots']} repeat={task['repeat_index']}",
                    flush=True,
                )

            max_attempts = hardware_retries + 1
            row: dict[str, Any] | None = None
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    row = compute_hardware_row(
                        spec,
                        iqm_backend,
                        fold=int(task["fold"]),
                        ansatz_name=str(task["ansatz"]),
                        shots=int(task["shots"]),
                        repeat_index=int(task["repeat_index"]),
                        root=root,
                    )
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    last_exc = exc
                    retryable = is_retryable_hardware_error(exc)
                    if attempt >= max_attempts or not retryable:
                        if persist_failed_hardware_rows:
                            row = build_failed_hardware_row(
                                spec,
                                fold=int(task["fold"]),
                                ansatz_name=str(task["ansatz"]),
                                shots=int(task["shots"]),
                                repeat_index=int(task["repeat_index"]),
                                root=root,
                            )
                            if verbose:
                                print(f"Hardware task failed: {exc}", flush=True)
                            break
                        raise
                    wait_seconds = retry_wait_seconds(
                        attempt,
                        initial_wait_seconds=retry_wait_seconds_initial,
                        max_wait_seconds=retry_wait_seconds_max,
                    )
                    if verbose:
                        print(
                            f"Retryable hardware error (attempt {attempt}/{max_attempts}): {exc}; "
                            f"waiting {wait_seconds:.0f}s",
                            flush=True,
                        )
                    time.sleep(wait_seconds)

            if row is None:
                raise RuntimeError("Hardware task did not produce a result row") from last_exc

            append_csv_row(runs_csv, row)
            run_df = normalize_ansatz_labels(read_csv_or_empty(runs_csv))
            if row.get("status") == "success":
                done.add(task_key)
            hw_completed += 1
            report_progress(progress_callback, "hardware", hw_completed, hw_total)
            summary_df = summarize_results(spec, statevector_df=statevector_df, run_df=run_df)
            summary_df.to_csv(summary_csv, index=False)

    return run_dir
