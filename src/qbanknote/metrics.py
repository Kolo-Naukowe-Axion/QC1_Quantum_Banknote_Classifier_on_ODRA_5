"""Meyer-Wallach entanglement and KL expressibility metrics."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit

from qbanknote.ansatzes import trimmed_reverse_q0_param_count
from qbanknote.evaluation import append_csv_row, read_csv_or_empty
from qbanknote.iqm import run_circuits_on_backend, transpile_for_backend
from qbanknote.progress import report_progress
from qbanknote.tomography import (
    add_tomography_rotations,
    all_basis_settings,
    expectation_from_counts,
    hardware_overlap,
    project_to_physical,
    reconstruct_rho,
    tomography_density_matrices,
)

BasisName = Literal["Z", "X", "Y"]
BASIS_ORDER: tuple[BasisName, ...] = ("Z", "X", "Y")


# ---------------------------------------------------------------------------
# Meyer–Wallach
# ---------------------------------------------------------------------------


def single_qubit_reduced_density(
    state: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    arr = state.reshape((2,) * n_qubits)
    arr = np.moveaxis(arr, qubit, 0)
    psi_mat = arr.reshape(2, -1)
    return psi_mat @ psi_mat.conj().T


def meyer_wallach_score(state: np.ndarray, n_qubits: int) -> float:
    if n_qubits < 1:
        return 0.0
    acc = 0.0
    for i in range(n_qubits):
        rho_i = single_qubit_reduced_density(state, i, n_qubits)
        purity = float(np.real(np.trace(rho_i @ rho_i)))
        acc += 1.0 - purity
    q = (2.0 / n_qubits) * acc
    return float(max(0.0, min(1.0, q)))


def bitstring_qubit_value(bitstring: str, qubit: int, n_qubits: int) -> str:
    if len(bitstring) != n_qubits:
        raise ValueError(f"Expected bitstring length {n_qubits}, got {len(bitstring)!r}")
    return bitstring[-(qubit + 1)]


def qubit_expectation_from_counts(
    counts: dict[str, int], qubit: int, n_qubits: int
) -> float:
    shots = sum(counts.values())
    if shots == 0:
        return 0.0
    expval = 0.0
    for bitstring, count in counts.items():
        bit = bitstring_qubit_value(bitstring, qubit, n_qubits)
        eigenvalue = 1.0 if bit == "0" else -1.0
        expval += eigenvalue * count / shots
    return float(expval)


def mw_score_from_bloch(
    x_expectations: list[float],
    y_expectations: list[float],
    z_expectations: list[float],
) -> float:
    n_qubits = len(x_expectations)
    if n_qubits < 1:
        return 0.0
    acc = 0.0
    for x_i, y_i, z_i in zip(x_expectations, y_expectations, z_expectations):
        purity = 0.5 * (1.0 + x_i * x_i + y_i * y_i + z_i * z_i)
        acc += 1.0 - purity
    q = (2.0 / n_qubits) * acc
    return float(max(0.0, min(1.0, q)))


def add_basis_measurement(circuit: QuantumCircuit, basis: BasisName) -> QuantumCircuit:
    qc = circuit.copy()
    if basis == "X":
        for i in range(qc.num_qubits):
            qc.h(i)
    elif basis == "Y":
        for i in range(qc.num_qubits):
            qc.sdg(i)
            qc.h(i)
    qc.measure_all()
    return qc


def estimate_mw_from_hardware_counts(
    counts_by_basis: dict[BasisName, dict[str, int]],
    n_qubits: int,
) -> tuple[float, list[float], list[float], list[float]]:
    x_exp = [
        qubit_expectation_from_counts(counts_by_basis["X"], i, n_qubits)
        for i in range(n_qubits)
    ]
    y_exp = [
        qubit_expectation_from_counts(counts_by_basis["Y"], i, n_qubits)
        for i in range(n_qubits)
    ]
    z_exp = [
        qubit_expectation_from_counts(counts_by_basis["Z"], i, n_qubits)
        for i in range(n_qubits)
    ]
    score = mw_score_from_bloch(x_exp, y_exp, z_exp)
    return score, x_exp, y_exp, z_exp


def _group_counts_by_sample(
    counts_list: list[dict[str, int]],
) -> list[dict[BasisName, dict[str, int]]]:
    if len(counts_list) % len(BASIS_ORDER) != 0:
        raise ValueError("Counts list length must be a multiple of 3 (Z, X, Y per sample)")
    grouped: list[dict[BasisName, dict[str, int]]] = []
    for i in range(0, len(counts_list), len(BASIS_ORDER)):
        grouped.append(
            {
                "Z": counts_list[i],
                "X": counts_list[i + 1],
                "Y": counts_list[i + 2],
            }
        )
    return grouped


def compute_iqm_mw_scores(
    backend,
    ansatz_fn: Callable[[int, int], QuantumCircuit],
    n_qubits: int,
    depth: int,
    n_samples: int,
    seed: int,
    shots: int,
    optimization_level: int,
    seed_transpiler: int | None,
    max_circuits_per_job: int,
) -> dict[str, object]:
    qc = ansatz_fn(n_qubits, depth)
    params = list(qc.parameters)
    rng = np.random.default_rng(seed)

    scores: list[float] = []
    bloch_rows: list[dict[str, float]] = []
    pending_circuits: list[QuantumCircuit] = []
    pending_sample_indices: list[int] = []

    def flush_batch() -> None:
        nonlocal pending_circuits, pending_sample_indices
        if not pending_circuits:
            return
        counts_list = run_circuits_on_backend(
            backend,
            pending_circuits,
            shots=shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            max_circuits_per_job=max_circuits_per_job,
        )
        grouped = _group_counts_by_sample(counts_list)
        if len(grouped) != len(pending_sample_indices):
            raise RuntimeError(
                f"Batch mismatch: {len(grouped)} sample groups vs "
                f"{len(pending_sample_indices)} sample indices"
            )
        for sample_index, basis_counts in zip(pending_sample_indices, grouped):
            score, x_exp, y_exp, z_exp = estimate_mw_from_hardware_counts(
                basis_counts, n_qubits
            )
            scores.append(score)
            row: dict[str, float] = {"sample_index": float(sample_index), "mw_score": score}
            for i in range(n_qubits):
                row[f"x_q{i}"] = x_exp[i]
                row[f"y_q{i}"] = y_exp[i]
                row[f"z_q{i}"] = z_exp[i]
            bloch_rows.append(row)
        pending_circuits = []
        pending_sample_indices = []

    for sample_index in range(n_samples):
        values = rng.uniform(0.0, 2.0 * math.pi, size=len(params))
        bound = qc.assign_parameters(dict(zip(params, values)), inplace=False)
        for basis in BASIS_ORDER:
            pending_circuits.append(add_basis_measurement(bound, basis))
        pending_sample_indices.append(sample_index)
        if len(pending_circuits) >= max_circuits_per_job:
            flush_batch()
    flush_batch()

    if len(scores) != n_samples:
        raise RuntimeError(f"Expected {n_samples} MW scores, got {len(scores)}")

    arr = np.array(scores, dtype=float)
    return {
        "mw_scores": scores,
        "bloch_rows": bloch_rows,
        "mw_avg": float(np.mean(arr)),
        "mw_std": float(np.std(arr)),
        "mw_sem": float(np.std(arr) / math.sqrt(n_samples)),
        "mw_min": float(np.min(arr)),
        "mw_max": float(np.max(arr)),
        "depth": depth,
        "n_qubits": n_qubits,
        "n_params": len(params),
        "n_samples": n_samples,
    }


MW_SUMMARY_COLUMNS = [
    "ansatz",
    "depth",
    "n_qubits",
    "n_params",
    "n_samples",
    "shots",
    "seed",
    "mw_avg",
    "mw_std",
    "mw_sem",
    "mw_min",
    "mw_max",
]


def mw_job_key(ansatz: str, depth: int) -> tuple[str, int]:
    return str(ansatz), int(depth)


def completed_mw_jobs(summary_path: Path) -> set[tuple[str, int]]:
    frame = read_csv_or_empty(summary_path)
    if frame.empty:
        return set()
    return {
        mw_job_key(str(row.ansatz), int(row.depth))
        for row in frame.itertuples(index=False)
    }


def bloch_row_to_score_row(ansatz: str, depth: int, bloch: dict[str, float]) -> dict[str, object]:
    row: dict[str, object] = {
        "ansatz": ansatz,
        "depth": depth,
        "sample_index": int(bloch["sample_index"]),
        "mw_score": bloch["mw_score"],
    }
    for key, value in bloch.items():
        if key not in ("sample_index", "mw_score"):
            row[key] = value
    return row


def summary_row_from_result(
    *,
    ansatz: str,
    depth: int,
    shots: int,
    seed: int,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "ansatz": ansatz,
        "depth": depth,
        "n_qubits": result["n_qubits"],
        "n_params": result["n_params"],
        "n_samples": result["n_samples"],
        "shots": shots,
        "seed": seed,
        "mw_avg": result["mw_avg"],
        "mw_std": result["mw_std"],
        "mw_sem": result["mw_sem"],
        "mw_min": result["mw_min"],
        "mw_max": result["mw_max"],
    }


def write_mw_manifest(
    path: Path,
    *,
    backend,
    manifest: dict[str, object],
) -> None:
    payload = {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "backend": str(backend),
        **manifest,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_iqm_mw_sweep(
    backend,
    *,
    ansatz_fns: dict[str, Callable[[int, int], QuantumCircuit]],
    ansatz_names: list[str],
    depths: list[int],
    n_qubits: int,
    n_samples: int,
    seed: int,
    shots: int,
    optimization_level: int,
    seed_transpiler: int | None,
    max_circuits_per_job: int,
    output_dir: Path,
    resume: bool = True,
    verbose: bool = False,
    progress_callback: Callable[[str, int, int], None] | None = None,
    manifest_extra: dict[str, object] | None = None,
) -> Path:
    """Run Meyer-Wallach hardware sweep with resumable CSV artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "iqm_mw_results.csv"
    scores_path = output_dir / "iqm_mw_scores.csv"
    manifest_path = output_dir / "run_manifest.json"

    completed = completed_mw_jobs(summary_path) if resume else set()
    jobs = [(ansatz_name, depth) for depth in depths for ansatz_name in ansatz_names]
    total_jobs = len(jobs)
    done_jobs = sum(1 for job in jobs if job in completed)

    for ansatz_name, depth in jobs:
        if (ansatz_name, depth) in completed:
            continue

        if verbose:
            print(f"Running {ansatz_name} depth={depth}", flush=True)

        depth_seed = seed + depth * 1000 + (1 if ansatz_name == "ansatz_simulator" else 0)
        result = compute_iqm_mw_scores(
            backend,
            ansatz_fns[ansatz_name],
            n_qubits=n_qubits,
            depth=depth,
            n_samples=n_samples,
            seed=depth_seed,
            shots=shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            max_circuits_per_job=max_circuits_per_job,
        )

        summary_row = summary_row_from_result(
            ansatz=ansatz_name,
            depth=depth,
            shots=shots,
            seed=depth_seed,
            result=result,
        )
        append_csv_row(summary_path, summary_row)

        for bloch in result["bloch_rows"]:
            append_csv_row(
                scores_path,
                bloch_row_to_score_row(ansatz_name, depth, bloch),
            )

        done_jobs += 1
        report_progress(progress_callback, "mw_jobs", done_jobs, total_jobs)

    manifest = {
        "source_script": "scripts/run_iqm_meyer_wallach.py",
        "method": "local_xyz_tomography",
        "n_qubits": n_qubits,
        "depths": list(depths),
        "ansatzes": list(ansatz_names),
        "n_samples": n_samples,
        "shots": shots,
        "seed": seed,
        "optimization_level": optimization_level,
        "max_circuits_per_job": max_circuits_per_job,
        "total_circuits": len(ansatz_names) * len(depths) * n_samples * len(BASIS_ORDER),
        "outputs": ["iqm_mw_results.csv", "iqm_mw_scores.csv"],
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    write_mw_manifest(manifest_path, backend=backend, manifest=manifest)
    return output_dir


def mw_shot_noise_sd_bound(n_qubits: int, shots: int) -> float:
    """Worst-case delta-method bound for shot noise in one MW estimate."""
    if n_qubits <= 0 or shots <= 0:
        raise ValueError("n_qubits and shots must be positive")
    return float(2.0 / math.sqrt(n_qubits * shots))


def mw_mean_shot_noise_bound(n_qubits: int, shots: int, n_samples: int) -> float:
    """Shot-noise contribution after averaging MW scores over parameter samples."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    return float(mw_shot_noise_sd_bound(n_qubits, shots) / math.sqrt(n_samples))


def mw_confidence_half_width(std: float, n_samples: int, z_value: float = 1.96) -> float:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    return float(z_value * float(std) / math.sqrt(n_samples))


def required_mw_samples(std: float, target_half_width: float, z_value: float = 1.96) -> int:
    if target_half_width <= 0:
        raise ValueError("target_half_width must be positive")
    std = max(float(std), 0.0)
    if std == 0.0:
        return 1
    return int(math.ceil(((z_value * std) / target_half_width) ** 2))


def read_mw_summary(run_dir: Path, *, stage: str | None = None, iteration: int | None = None) -> np.ndarray:
    frame = read_csv_or_empty(Path(run_dir) / "iqm_mw_results.csv")
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["run_dir"] = str(Path(run_dir))
    frame["run_id"] = Path(run_dir).name
    if stage is not None:
        frame["stage"] = stage
    if iteration is not None:
        frame["iteration"] = int(iteration)
    return frame


def compute_mw_shot_stability(summary_df) -> tuple[object, object]:
    """Compare consecutive shot budgets for each ansatz/depth/sample setting."""
    if summary_df.empty or "shots" not in summary_df.columns:
        return summary_df.iloc[0:0].copy(), summary_df.iloc[0:0].copy()

    rows = []
    group_cols = ["ansatz", "depth", "n_samples"]
    for keys, group in summary_df.groupby(group_cols, dropna=False):
        ansatz, depth, n_samples = keys
        ordered_shots = sorted(int(v) for v in group["shots"].dropna().unique())
        for previous_shot, current_shot in zip(ordered_shots[:-1], ordered_shots[1:]):
            prev = group[group["shots"] == previous_shot]
            curr = group[group["shots"] == current_shot]
            if prev.empty or curr.empty:
                continue
            prev_row = prev.iloc[0]
            curr_row = curr.iloc[0]
            rows.append(
                {
                    "ansatz": ansatz,
                    "depth": int(depth),
                    "n_samples": int(n_samples),
                    "previous_shot": previous_shot,
                    "current_shot": current_shot,
                    "mw_avg_previous": float(prev_row["mw_avg"]),
                    "mw_avg_current": float(curr_row["mw_avg"]),
                    "abs_change_mw_avg": abs(float(curr_row["mw_avg"]) - float(prev_row["mw_avg"])),
                }
            )

    import pandas as pd

    detailed = pd.DataFrame(rows)
    if detailed.empty:
        return detailed, pd.DataFrame()

    aggregate_rows = []
    for keys, group in detailed.groupby(["previous_shot", "current_shot"], dropna=False):
        previous_shot, current_shot = keys
        aggregate_rows.append(
            {
                "previous_shot": int(previous_shot),
                "current_shot": int(current_shot),
                "mean_abs_change_mw_avg": float(group["abs_change_mw_avg"].mean()),
                "max_abs_change_mw_avg": float(group["abs_change_mw_avg"].max()),
            }
        )
    return detailed, pd.DataFrame(aggregate_rows)


def choose_mw_shots(summary_df, *, tolerance: float) -> int:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if summary_df.empty:
        raise ValueError("MW pilot summary is empty; cannot choose shots")
    _, aggregate = compute_mw_shot_stability(summary_df)
    if aggregate.empty:
        return int(max(summary_df["shots"]))
    for row in aggregate.sort_values("current_shot").itertuples(index=False):
        if float(row.max_abs_change_mw_avg) <= tolerance:
            return int(row.current_shot)
    return int(max(summary_df["shots"]))


def compute_mw_sample_precision(summary_df, *, target_half_width: float, z_value: float = 1.96):
    if summary_df.empty:
        return summary_df.iloc[0:0].copy(), summary_df.iloc[0:0].copy()
    rows = []
    for row in summary_df.itertuples(index=False):
        half_width = mw_confidence_half_width(float(row.mw_std), int(row.n_samples), z_value)
        rows.append(
            {
                "ansatz": row.ansatz,
                "depth": int(row.depth),
                "shots": int(row.shots),
                "n_samples": int(row.n_samples),
                "mw_std": float(row.mw_std),
                "mw_sem": float(row.mw_sem),
                "confidence_half_width": half_width,
                "target_half_width": float(target_half_width),
                "meets_target": bool(half_width <= target_half_width),
                "required_n_samples": required_mw_samples(
                    float(row.mw_std),
                    target_half_width,
                    z_value,
                ),
            }
        )

    import pandas as pd

    detailed = pd.DataFrame(rows)
    aggregate_rows = []
    for n_samples, group in detailed.groupby("n_samples", dropna=False):
        aggregate_rows.append(
            {
                "n_samples": int(n_samples),
                "max_confidence_half_width": float(group["confidence_half_width"].max()),
                "max_required_n_samples": int(group["required_n_samples"].max()),
                "all_meet_target": bool(group["meets_target"].all()),
            }
        )
    return detailed, pd.DataFrame(aggregate_rows).sort_values("n_samples")


def choose_mw_samples(summary_df, *, target_half_width: float, z_value: float = 1.96) -> int:
    if summary_df.empty:
        raise ValueError("MW sample pilot summary is empty; cannot choose n_samples")
    _, aggregate = compute_mw_sample_precision(
        summary_df,
        target_half_width=target_half_width,
        z_value=z_value,
    )
    for row in aggregate.sort_values("n_samples").itertuples(index=False):
        if bool(row.all_meet_target):
            return int(row.n_samples)
    return int(aggregate["max_required_n_samples"].max())


# Two-sided 95% Student-t critical values for df = K-1, K = 2..10
_STUDENT_T_975: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def mw_student_t_975(iterations: int) -> float:
    """Return t_{0.975, K-1} for K repeated hardware iterations."""
    if iterations < 2:
        raise ValueError("iterations must be at least 2 for a drift variance estimate")
    df = int(iterations) - 1
    if df in _STUDENT_T_975:
        return float(_STUDENT_T_975[df])
    return 1.96


def mw_iteration_half_width(std: float, iterations: int) -> float:
    """95% half-width for the mean MW across repeated hardware iterations."""
    if iterations < 2:
        raise ValueError("iterations must be at least 2 for iteration half-width")
    std = max(float(std), 0.0)
    t_value = mw_student_t_975(iterations)
    return float(t_value * std / math.sqrt(iterations))


def compute_mw_iteration_stability(summary_df):
    if summary_df.empty or "iteration" not in summary_df.columns:
        return summary_df.iloc[0:0].copy()

    import pandas as pd

    rows = []
    for keys, group in summary_df.groupby(["ansatz", "depth", "shots", "n_samples"], dropna=False):
        ansatz, depth, shots, n_samples = keys
        n_iter = int(group["iteration"].nunique())
        iter_std = float(group["mw_avg"].std(ddof=1)) if n_iter > 1 else 0.0
        iter_sem = iter_std / math.sqrt(n_iter) if n_iter > 0 else 0.0
        half_width = (
            mw_iteration_half_width(iter_std, n_iter) if n_iter >= 2 else float("nan")
        )
        rows.append(
            {
                "ansatz": ansatz,
                "depth": int(depth),
                "shots": int(shots),
                "n_samples": int(n_samples),
                "iterations": n_iter,
                "iteration_mean_mw_avg": float(group["mw_avg"].mean()),
                "iteration_std_mw_avg": iter_std,
                "iteration_sem_mw_avg": iter_sem,
                "iteration_half_width_95": half_width,
                "iteration_min_mw_avg": float(group["mw_avg"].min()),
                "iteration_max_mw_avg": float(group["mw_avg"].max()),
            }
        )
    return pd.DataFrame(rows)


def compute_mw_iteration_precision(
    summary_df,
    *,
    target_half_width: float,
) -> tuple[object, object]:
    """Compute per-configuration iteration drift precision from repeated sweeps."""
    import pandas as pd

    stability = compute_mw_iteration_stability(summary_df)
    if stability.empty:
        return stability, pd.DataFrame()

    detailed = stability.copy()
    detailed["target_half_width"] = float(target_half_width)
    detailed["meets_target"] = detailed["iteration_half_width_95"] <= float(target_half_width)

    aggregate_rows = []
    if not detailed["iteration_half_width_95"].isna().all():
        aggregate_rows.append(
            {
                "iterations": int(detailed["iterations"].max()),
                "max_iteration_half_width_95": float(detailed["iteration_half_width_95"].max()),
                "mean_iteration_half_width_95": float(detailed["iteration_half_width_95"].mean()),
                "target_half_width": float(target_half_width),
                "all_meet_target": bool(detailed["meets_target"].all()),
            }
        )
    return detailed, pd.DataFrame(aggregate_rows)


def choose_mw_iterations(
    summary_df,
    *,
    target_half_width: float,
    min_iterations: int = 3,
    max_iterations: int = 5,
) -> int:
    """Choose smallest K with worst-case iteration half-width below target."""
    if min_iterations < 2:
        raise ValueError("min_iterations must be at least 2")
    if max_iterations < min_iterations:
        raise ValueError("max_iterations must be >= min_iterations")
    if summary_df.empty:
        raise ValueError("MW iteration pilot summary is empty; cannot choose iterations")

    import pandas as pd

    for iterations in range(min_iterations, max_iterations + 1):
        subset = summary_df[summary_df["iteration"] <= iterations]
        present = sorted(int(v) for v in subset["iteration"].dropna().unique())
        if present != list(range(1, iterations + 1)):
            continue
        _, aggregate = compute_mw_iteration_precision(
            subset,
            target_half_width=target_half_width,
        )
        if aggregate.empty:
            continue
        row = aggregate.iloc[0]
        if bool(row["all_meet_target"]):
            return int(iterations)
    return int(max_iterations)


def iteration_target_met(
    summary_df,
    *,
    target_half_width: float,
    iterations: int,
) -> bool:
    """Return True if completed iterations satisfy the drift half-width target."""
    subset = summary_df[summary_df["iteration"] <= iterations]
    present = sorted(int(v) for v in subset["iteration"].dropna().unique())
    if present != list(range(1, iterations + 1)):
        return False
    _, aggregate = compute_mw_iteration_precision(
        subset,
        target_half_width=target_half_width,
    )
    if aggregate.empty:
        return False
    return bool(aggregate.iloc[0]["all_meet_target"])


def write_mw_protocol_artifacts(
    run_dir: Path,
    *,
    recommendation: dict[str, object],
    frames: dict[str, object],
) -> Path:
    import pandas as pd

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(run_dir / f"{name}.csv", index=False)
    path = run_dir / "mw_protocol_recommendation.json"
    path.write_text(json.dumps(recommendation, indent=2, sort_keys=True) + "\n")
    return path


def verify_mw_implementation() -> None:
    n = 3
    product = np.zeros(2**n, dtype=complex)
    product[0] = 1.0
    assert abs(meyer_wallach_score(product, n)) < 1e-9

    ghz = np.zeros(2**n, dtype=complex)
    ghz[0] = 1.0 / math.sqrt(2.0)
    ghz[-1] = 1.0 / math.sqrt(2.0)
    assert abs(meyer_wallach_score(ghz, n) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# KL expressibility
# ---------------------------------------------------------------------------


def haar_pdf_fidelity(f: np.ndarray, dim: int) -> np.ndarray:
    return (dim - 1.0) * (1.0 - f) ** (dim - 2.0)


def binned_distributions(
    fid_values: np.ndarray,
    dim: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    counts, edges = np.histogram(fid_values, bins=bins, density=False)
    p_emp = counts.astype(np.float64)
    if p_emp.sum() == 0:
        p_emp = np.ones_like(p_emp) / len(p_emp)
    else:
        p_emp /= p_emp.sum()

    mids = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    p_haar = haar_pdf_fidelity(mids, dim=dim) * width
    p_haar /= p_haar.sum()
    return edges, mids, p_emp, p_haar


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p_s = p + eps
    q_s = q + eps
    p_s /= p_s.sum()
    q_s /= q_s.sum()
    return float(np.sum(p_s * np.log(p_s / q_s)))


def bind_ansatz(
    ansatz_fn: Callable[[int, int], QuantumCircuit],
    n_qubits: int,
    depth: int,
    theta_values: np.ndarray,
) -> QuantumCircuit:
    qc = ansatz_fn(n_qubits, depth)
    ordered_params = list(qc.parameters)
    bind_map = {p: float(v) for p, v in zip(ordered_params, theta_values)}
    return qc.assign_parameters(bind_map, inplace=False)


def sample_hardware_fidelities(
    backend,
    ansatz_fn: Callable[[int, int], QuantumCircuit],
    n_qubits: int,
    depth: int,
    n_samples: int,
    seed: int,
    shots: int,
    optimization_level: int,
    seed_transpiler: int | None,
    max_circuits_per_job: int,
    ansatz_label: str = "",
    completed_samples: set[int] | None = None,
    on_sample_complete: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    qc_template = ansatz_fn(n_qubits, depth)
    n_params = len(qc_template.parameters)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    skip_samples = completed_samples or set()

    for sample_index in range(n_samples):
        theta_a = rng.uniform(0.0, 2.0 * np.pi, n_params)
        theta_b = rng.uniform(0.0, 2.0 * np.pi, n_params)
        if sample_index in skip_samples:
            continue

        bound_a = bind_ansatz(ansatz_fn, n_qubits, depth, theta_a)
        bound_b = bind_ansatz(ansatz_fn, n_qubits, depth, theta_b)

        print(
            f"\n  sample {sample_index + 1}/{n_samples} "
            f"({ansatz_label}, depth={depth})"
        )

        rho_a_lin, rho_a_phys, diag_a = tomography_density_matrices(
            bound_a,
            backend,
            n_qubits=n_qubits,
            shots=shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            max_circuits_per_job=max_circuits_per_job,
            label=f"{ansatz_label} state A",
        )
        rho_b_lin, rho_b_phys, diag_b = tomography_density_matrices(
            bound_b,
            backend,
            n_qubits=n_qubits,
            shots=shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            max_circuits_per_job=max_circuits_per_job,
            label=f"{ansatz_label} state B",
        )

        f_lin = hardware_overlap(rho_a_lin, rho_b_lin)
        f_phys = hardware_overlap(rho_a_phys, rho_b_phys)

        print(
            f"    F (linear inv.)    = {f_lin:.4f}\n"
            f"    F (physical proj.) = {f_phys:.4f}"
        )

        row = {
            "sample_index": sample_index,
            "fidelity_linear": f_lin,
            "fidelity_physical": f_phys,
            "trace_a_linear": diag_a["trace_linear"],
            "trace_a_physical": diag_a["trace_physical"],
            "trace_b_linear": diag_b["trace_linear"],
            "trace_b_physical": diag_b["trace_physical"],
            "purity_a_linear": diag_a["purity_linear"],
            "purity_a_physical": diag_a["purity_physical"],
            "purity_b_linear": diag_b["purity_linear"],
            "purity_b_physical": diag_b["purity_physical"],
        }
        rows.append(row)
        if on_sample_complete is not None:
            on_sample_complete(row)

    return rows


def compute_kl_for_fidelities(
    fidelities: np.ndarray,
    dim: int,
    n_bins: int,
    eps: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    _, mids, p_emp, p_haar = binned_distributions(fidelities, dim, n_bins)
    kl = kl_divergence(p_emp, p_haar, eps)
    return kl, mids, p_emp, p_haar


def sample_haar_fidelities(n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Draw fidelities F = |<psi|phi>|^2 from the Haar-random pair distribution."""
    return 1.0 - rng.random(int(n_samples)) ** (1.0 / (dim - 1))


def aggregate_kl_from_sample_rows(
    sample_rows: list[dict[str, object]],
    *,
    n_qubits: int,
    n_bins: int,
    eps: float,
) -> dict[str, object]:
    """Aggregate per-pair tomography rows into KL summary statistics."""
    dim = 2**n_qubits
    f_phys = np.array([float(row["fidelity_physical"]) for row in sample_rows], dtype=np.float64)
    f_lin = np.array([float(row["fidelity_linear"]) for row in sample_rows], dtype=np.float64)
    kl_phys, _, _, _ = compute_kl_for_fidelities(f_phys, dim, n_bins, eps)
    kl_lin, _, _, _ = compute_kl_for_fidelities(f_lin, dim, n_bins, eps)
    return {
        "n_qubits": n_qubits,
        "n_samples": len(sample_rows),
        "kl_physical": float(kl_phys),
        "kl_linear": float(kl_lin),
        "f_physical_mean": float(np.mean(f_phys)),
        "f_physical_std": float(np.std(f_phys)),
        "f_linear_mean": float(np.mean(f_lin)),
        "f_linear_std": float(np.std(f_lin)),
    }


def bootstrap_kl_std(
    fidelities: np.ndarray,
    *,
    dim: int,
    n_bins: int,
    eps: float,
    n_bootstrap: int = 400,
    seed: int = 0,
) -> float:
    """Bootstrap standard deviation of KL(P_emp || P_Haar) from fidelity samples."""
    if len(fidelities) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    n = len(fidelities)
    kl_values = np.empty(int(n_bootstrap), dtype=np.float64)
    for index in range(int(n_bootstrap)):
        draw = fidelities[rng.integers(0, n, size=n)]
        kl, _, _, _ = compute_kl_for_fidelities(draw, dim, n_bins, eps)
        kl_values[index] = kl
    return float(np.std(kl_values, ddof=1))


def kl_confidence_half_width(std: float, n_bootstrap: int, z_value: float = 1.96) -> float:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    return float(z_value * max(float(std), 0.0))


def required_kl_samples(std: float, target_half_width: float, z_value: float = 1.96) -> int:
    if target_half_width <= 0:
        raise ValueError("target_half_width must be positive")
    std = max(float(std), 0.0)
    if std == 0.0:
        return 1
    return int(math.ceil(((z_value * std) / target_half_width) ** 2))


def kl_job_key(ansatz: str, depth: int) -> tuple[str, int]:
    return str(ansatz), int(depth)


def completed_kl_jobs(summary_path: Path) -> set[tuple[str, int]]:
    frame = read_csv_or_empty(summary_path)
    if frame.empty:
        return set()
    return {
        kl_job_key(str(row.ansatz), int(row.depth))
        for row in frame.itertuples(index=False)
    }


def completed_kl_samples(
    fidelities_path: Path,
    ansatz: str,
    depth: int,
) -> set[int]:
    """Return sample indices already stored for one (ansatz, depth) job."""
    frame = read_csv_or_empty(fidelities_path)
    if frame.empty or "sample_index" not in frame.columns:
        return set()
    subset = frame[
        (frame["ansatz"].astype(str) == str(ansatz))
        & (frame["depth"].astype(int) == int(depth))
    ]
    if subset.empty:
        return set()
    return {int(value) for value in subset["sample_index"]}


def load_kl_job_fidelity_rows(
    fidelities_path: Path,
    ansatz: str,
    depth: int,
) -> list[dict[str, object]]:
    """Load per-sample fidelity rows for one (ansatz, depth) job, sorted by index."""
    frame = read_csv_or_empty(fidelities_path)
    if frame.empty or "sample_index" not in frame.columns:
        return []
    subset = frame[
        (frame["ansatz"].astype(str) == str(ansatz))
        & (frame["depth"].astype(int) == int(depth))
    ].sort_values("sample_index")
    rows: list[dict[str, object]] = []
    for record in subset.to_dict(orient="records"):
        row = {key: record[key] for key in record if key not in ("ansatz", "depth")}
        row["sample_index"] = int(row["sample_index"])
        rows.append(row)
    return rows


def kl_summary_row(
    *,
    ansatz: str,
    depth: int,
    shots: int,
    seed: int,
    n_bins: int,
    eps: float,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "ansatz": ansatz,
        "depth": depth,
        "n_qubits": result["n_qubits"],
        "n_samples": result["n_samples"],
        "shots": shots,
        "seed": seed,
        "n_bins": n_bins,
        "eps": eps,
        "kl_physical": result["kl_physical"],
        "kl_linear": result["kl_linear"],
        "f_physical_mean": result["f_physical_mean"],
        "f_physical_std": result["f_physical_std"],
        "f_linear_mean": result["f_linear_mean"],
        "f_linear_std": result["f_linear_std"],
    }


def fidelity_row_to_csv(
    ansatz: str,
    depth: int,
    row: dict[str, object],
) -> dict[str, object]:
    return {"ansatz": ansatz, "depth": depth, **row}


def read_kl_summary(
    run_dir: Path,
    *,
    stage: str | None = None,
    iteration: int | None = None,
):
    frame = read_csv_or_empty(Path(run_dir) / "iqm_kl_results.csv")
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["run_dir"] = str(Path(run_dir))
    frame["run_id"] = Path(run_dir).name
    if stage is not None:
        frame["stage"] = stage
    if iteration is not None:
        frame["iteration"] = int(iteration)
    return frame


def read_kl_fidelities(run_dir: Path):
    return read_csv_or_empty(Path(run_dir) / "iqm_kl_fidelities.csv")


def run_iqm_kl_sweep(
    backend,
    *,
    ansatz_fns: dict[str, Callable[[int, int], QuantumCircuit]],
    ansatz_names: list[str],
    depths: list[int],
    n_qubits: int,
    n_samples: int,
    seed: int,
    shots: int,
    n_bins: int,
    eps: float,
    optimization_level: int,
    seed_transpiler: int | None,
    max_circuits_per_job: int,
    output_dir: Path,
    resume: bool = True,
    verbose: bool = False,
    progress_callback: Callable[[str, int, int], None] | None = None,
    manifest_extra: dict[str, object] | None = None,
) -> Path:
    """Run KL expressibility hardware sweep with resumable CSV artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "iqm_kl_results.csv"
    fidelities_path = output_dir / "iqm_kl_fidelities.csv"
    manifest_path = output_dir / "run_manifest.json"

    completed = completed_kl_jobs(summary_path) if resume else set()
    jobs = [(ansatz_name, depth) for depth in depths for ansatz_name in ansatz_names]
    total_jobs = len(jobs)
    done_jobs = sum(1 for job in jobs if job in completed)

    for ansatz_name, depth in jobs:
        if (ansatz_name, depth) in completed:
            continue

        depth_seed = seed + 100 * depth + (1 if ansatz_name == "ansatz_simulator" else 0)
        saved_samples = completed_kl_samples(fidelities_path, ansatz_name, depth) if resume else set()
        existing_rows = (
            load_kl_job_fidelity_rows(fidelities_path, ansatz_name, depth) if resume else []
        )

        if saved_samples and verbose:
            print(
                f"Resuming {ansatz_name} depth={depth}: "
                f"{len(saved_samples)}/{n_samples} samples on disk",
                flush=True,
            )
        elif verbose:
            print(f"Running {ansatz_name} depth={depth}", flush=True)

        def _persist_sample(row: dict[str, object]) -> None:
            append_csv_row(
                fidelities_path,
                fidelity_row_to_csv(ansatz_name, depth, row),
            )

        new_rows = sample_hardware_fidelities(
            backend,
            ansatz_fns[ansatz_name],
            n_qubits=n_qubits,
            depth=depth,
            n_samples=n_samples,
            seed=depth_seed,
            shots=shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            max_circuits_per_job=max_circuits_per_job,
            ansatz_label=ansatz_name,
            completed_samples=saved_samples,
            on_sample_complete=_persist_sample if resume else None,
        )

        if not resume:
            for row in new_rows:
                append_csv_row(
                    fidelities_path,
                    fidelity_row_to_csv(ansatz_name, depth, row),
                )

        sample_rows = existing_rows + new_rows
        sample_rows.sort(key=lambda row: int(row["sample_index"]))
        if len(sample_rows) < n_samples:
            raise RuntimeError(
                f"Incomplete KL job {ansatz_name} depth={depth}: "
                f"expected {n_samples} samples, got {len(sample_rows)}"
            )

        result = aggregate_kl_from_sample_rows(
            sample_rows[:n_samples],
            n_qubits=n_qubits,
            n_bins=n_bins,
            eps=eps,
        )
        append_csv_row(
            summary_path,
            kl_summary_row(
                ansatz=ansatz_name,
                depth=depth,
                shots=shots,
                seed=depth_seed,
                n_bins=n_bins,
                eps=eps,
                result=result,
            ),
        )

        done_jobs += 1
        report_progress(progress_callback, "kl_jobs", done_jobs, total_jobs)

    manifest = {
        "source_script": "scripts/run_iqm_kl_expressibility.py",
        "method": "hardware_hardware_overlap_tomography",
        "fidelity_definition": "Tr(rho_a @ rho_b) from full 3^n Pauli tomography",
        "kl_direction": "P_hardware || P_Haar",
        "n_qubits": n_qubits,
        "depths": list(depths),
        "ansatzes": list(ansatz_names),
        "n_samples": n_samples,
        "shots": shots,
        "n_bins": n_bins,
        "eps": eps,
        "seed": seed,
        "optimization_level": optimization_level,
        "max_circuits_per_job": max_circuits_per_job,
        "circuits_per_fidelity_sample": circuits_per_fidelity_sample(n_qubits),
        "total_tomography_circuits": total_expressibility_circuits(
            len(ansatz_names),
            len(depths),
            n_samples,
            n_qubits,
        ),
        "outputs": ["iqm_kl_results.csv", "iqm_kl_fidelities.csv"],
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    write_mw_manifest(manifest_path, backend=backend, manifest=manifest)
    return output_dir


def compute_kl_shot_stability(summary_df) -> tuple[object, object]:
    """Compare consecutive shot budgets for each ansatz/depth/sample setting."""
    if summary_df.empty or "shots" not in summary_df.columns:
        return summary_df.iloc[0:0].copy(), summary_df.iloc[0:0].copy()

    rows = []
    group_cols = ["ansatz", "depth", "n_samples"]
    for keys, group in summary_df.groupby(group_cols, dropna=False):
        ansatz, depth, n_samples = keys
        ordered_shots = sorted(int(value) for value in group["shots"].dropna().unique())
        for previous_shot, current_shot in zip(ordered_shots[:-1], ordered_shots[1:]):
            prev = group[group["shots"] == previous_shot]
            curr = group[group["shots"] == current_shot]
            if prev.empty or curr.empty:
                continue
            prev_row = prev.iloc[0]
            curr_row = curr.iloc[0]
            rows.append(
                {
                    "ansatz": ansatz,
                    "depth": int(depth),
                    "n_samples": int(n_samples),
                    "previous_shot": previous_shot,
                    "current_shot": current_shot,
                    "kl_physical_previous": float(prev_row["kl_physical"]),
                    "kl_physical_current": float(curr_row["kl_physical"]),
                    "abs_change_kl_physical": abs(
                        float(curr_row["kl_physical"]) - float(prev_row["kl_physical"])
                    ),
                }
            )

    import pandas as pd

    detailed = pd.DataFrame(rows)
    if detailed.empty:
        return detailed, pd.DataFrame()

    aggregate_rows = []
    for keys, group in detailed.groupby(["previous_shot", "current_shot"], dropna=False):
        previous_shot, current_shot = keys
        aggregate_rows.append(
            {
                "previous_shot": int(previous_shot),
                "current_shot": int(current_shot),
                "mean_abs_change_kl_physical": float(group["abs_change_kl_physical"].mean()),
                "max_abs_change_kl_physical": float(group["abs_change_kl_physical"].max()),
            }
        )
    return detailed, pd.DataFrame(aggregate_rows)


def choose_kl_shots(summary_df, *, tolerance: float) -> int:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if summary_df.empty:
        raise ValueError("KL pilot summary is empty; cannot choose shots")
    _, aggregate = compute_kl_shot_stability(summary_df)
    if aggregate.empty:
        return int(max(summary_df["shots"]))
    for row in aggregate.sort_values("current_shot").itertuples(index=False):
        if float(row.max_abs_change_kl_physical) <= tolerance:
            return int(row.current_shot)
    return int(max(summary_df["shots"]))


def compute_kl_prefix_precision(
    fidelities_df,
    *,
    sample_grid: list[int],
    dim: int,
    n_bins: int,
    eps: float,
    target_half_width: float,
    n_bootstrap: int = 400,
    seed: int = 0,
    fidelity_column: str = "fidelity_physical",
    z_value: float = 1.96,
):
    """Bootstrap KL uncertainty from fidelity-prefix subsamples."""
    if fidelities_df.empty:
        return fidelities_df.iloc[0:0].copy(), fidelities_df.iloc[0:0].copy()

    rows = []
    group_cols = ["ansatz", "depth"]
    for keys, group in fidelities_df.groupby(group_cols, dropna=False):
        ansatz, depth = keys
        ordered = group.sort_values("sample_index")
        fidelities = ordered[fidelity_column].to_numpy(dtype=np.float64)
        available = len(fidelities)
        for n_samples in sample_grid:
            if n_samples > available:
                continue
            prefix = fidelities[: int(n_samples)]
            kl, _, _, _ = compute_kl_for_fidelities(prefix, dim, n_bins, eps)
            boot_std = bootstrap_kl_std(
                prefix,
                dim=dim,
                n_bins=n_bins,
                eps=eps,
                n_bootstrap=n_bootstrap,
                seed=seed + int(depth) * 1000 + (1 if ansatz == "ansatz_simulator" else 0),
            )
            half_width = kl_confidence_half_width(boot_std, n_bootstrap, z_value)
            rows.append(
                {
                    "ansatz": ansatz,
                    "depth": int(depth),
                    "n_samples": int(n_samples),
                    "available_samples": available,
                    "kl_physical": float(kl),
                    "bootstrap_std": boot_std,
                    "confidence_half_width": half_width,
                    "target_half_width": float(target_half_width),
                    "meets_target": bool(half_width <= target_half_width),
                    "required_n_samples": required_kl_samples(
                        boot_std,
                        target_half_width,
                        z_value,
                    ),
                }
            )

    import pandas as pd

    detailed = pd.DataFrame(rows)
    if detailed.empty:
        return detailed, pd.DataFrame()

    aggregate_rows = []
    for n_samples, group in detailed.groupby("n_samples", dropna=False):
        aggregate_rows.append(
            {
                "n_samples": int(n_samples),
                "max_confidence_half_width": float(group["confidence_half_width"].max()),
                "max_required_n_samples": int(group["required_n_samples"].max()),
                "all_meet_target": bool(group["meets_target"].all()),
            }
        )
    return detailed, pd.DataFrame(aggregate_rows).sort_values("n_samples")


def choose_kl_samples(
    fidelities_df,
    *,
    sample_grid: list[int],
    dim: int,
    n_bins: int,
    eps: float,
    target_half_width: float,
    n_bootstrap: int = 400,
    seed: int = 0,
    z_value: float = 1.96,
) -> int:
    _, aggregate = compute_kl_prefix_precision(
        fidelities_df,
        sample_grid=sample_grid,
        dim=dim,
        n_bins=n_bins,
        eps=eps,
        target_half_width=target_half_width,
        n_bootstrap=n_bootstrap,
        seed=seed,
        z_value=z_value,
    )
    if aggregate.empty:
        raise ValueError("KL prefix precision table is empty; cannot choose n_samples")
    for row in aggregate.sort_values("n_samples").itertuples(index=False):
        if bool(row.all_meet_target):
            return int(row.n_samples)
    return int(aggregate["max_required_n_samples"].max())


def compute_kl_bin_sensitivity(
    *,
    num_qubits: int,
    n_samples: int,
    bin_grid: list[int],
    n_reference_bins: int = 400,
    eps: float = 1e-12,
    seed: int = 0,
    n_trials: int = 100,
):
    """Estimate histogram discretization bias for KL via Haar-random fidelity draws."""
    dim = 2**num_qubits
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(int(n_trials)):
        fidelities = sample_haar_fidelities(n_samples, dim, rng)
        kl_ref, _, _, _ = compute_kl_for_fidelities(
            fidelities,
            dim,
            n_reference_bins,
            eps,
        )
        for n_bins in bin_grid:
            kl_bins, _, _, _ = compute_kl_for_fidelities(fidelities, dim, int(n_bins), eps)
            rows.append(
                {
                    "trial": trial,
                    "n_samples": int(n_samples),
                    "n_bins": int(n_bins),
                    "kl_reference": float(kl_ref),
                    "kl_physical": float(kl_bins),
                    "abs_bias": abs(float(kl_bins) - float(kl_ref)),
                }
            )

    import pandas as pd

    detailed = pd.DataFrame(rows)
    if detailed.empty:
        return detailed, pd.DataFrame()

    aggregate_rows = []
    for n_bins, group in detailed.groupby("n_bins", dropna=False):
        aggregate_rows.append(
            {
                "n_bins": int(n_bins),
                "mean_abs_bias": float(group["abs_bias"].mean()),
                "max_abs_bias": float(group["abs_bias"].max()),
                "p95_abs_bias": float(group["abs_bias"].quantile(0.95)),
            }
        )
    return detailed, pd.DataFrame(aggregate_rows).sort_values("n_bins")


def choose_kl_bins(
    bin_aggregate,
    *,
    tolerance: float,
    fallback: int | None = None,
) -> int:
    if bin_aggregate.empty:
        if fallback is None:
            raise ValueError("KL bin sensitivity table is empty; cannot choose n_bins")
        return int(fallback)
    for row in bin_aggregate.sort_values("n_bins").itertuples(index=False):
        if float(row.max_abs_bias) <= tolerance:
            return int(row.n_bins)
    return int(bin_aggregate["n_bins"].max())


def compute_kl_iteration_stability(summary_df):
    if summary_df.empty or "iteration" not in summary_df.columns:
        return summary_df.iloc[0:0].copy()

    import pandas as pd

    rows = []
    for keys, group in summary_df.groupby(
        ["ansatz", "depth", "shots", "n_samples", "n_bins"], dropna=False
    ):
        ansatz, depth, shots, n_samples, n_bins = keys
        n_iter = int(group["iteration"].nunique())
        iter_std = float(group["kl_physical"].std(ddof=1)) if n_iter > 1 else 0.0
        iter_sem = iter_std / math.sqrt(n_iter) if n_iter > 0 else 0.0
        half_width = (
            mw_iteration_half_width(iter_std, n_iter) if n_iter >= 2 else float("nan")
        )
        rows.append(
            {
                "ansatz": ansatz,
                "depth": int(depth),
                "shots": int(shots),
                "n_samples": int(n_samples),
                "n_bins": int(n_bins),
                "iterations": n_iter,
                "iteration_mean_kl_physical": float(group["kl_physical"].mean()),
                "iteration_std_kl_physical": iter_std,
                "iteration_sem_kl_physical": iter_sem,
                "iteration_half_width_95": half_width,
                "iteration_min_kl_physical": float(group["kl_physical"].min()),
                "iteration_max_kl_physical": float(group["kl_physical"].max()),
            }
        )
    return pd.DataFrame(rows)


def compute_kl_iteration_precision(
    summary_df,
    *,
    target_half_width: float,
) -> tuple[object, object]:
    import pandas as pd

    stability = compute_kl_iteration_stability(summary_df)
    if stability.empty:
        return stability, pd.DataFrame()

    detailed = stability.copy()
    detailed["target_half_width"] = float(target_half_width)
    detailed["meets_target"] = detailed["iteration_half_width_95"] <= float(target_half_width)

    aggregate_rows = []
    if not detailed["iteration_half_width_95"].isna().all():
        aggregate_rows.append(
            {
                "iterations": int(detailed["iterations"].max()),
                "max_iteration_half_width_95": float(detailed["iteration_half_width_95"].max()),
                "mean_iteration_half_width_95": float(detailed["iteration_half_width_95"].mean()),
                "target_half_width": float(target_half_width),
                "all_meet_target": bool(detailed["meets_target"].all()),
            }
        )
    return detailed, pd.DataFrame(aggregate_rows)


def choose_kl_iterations(
    summary_df,
    *,
    target_half_width: float,
    min_iterations: int = 2,
    max_iterations: int = 4,
) -> int:
    if min_iterations < 2:
        raise ValueError("min_iterations must be at least 2")
    if max_iterations < min_iterations:
        raise ValueError("max_iterations must be >= min_iterations")
    if summary_df.empty:
        raise ValueError("KL iteration pilot summary is empty; cannot choose iterations")

    for iterations in range(min_iterations, max_iterations + 1):
        subset = summary_df[summary_df["iteration"] <= iterations]
        present = sorted(int(value) for value in subset["iteration"].dropna().unique())
        if present != list(range(1, iterations + 1)):
            continue
        _, aggregate = compute_kl_iteration_precision(
            subset,
            target_half_width=target_half_width,
        )
        if aggregate.empty:
            continue
        if bool(aggregate.iloc[0]["all_meet_target"]):
            return int(iterations)
    return int(max_iterations)


def kl_iteration_target_met(
    summary_df,
    *,
    target_half_width: float,
    iterations: int,
) -> bool:
    subset = summary_df[summary_df["iteration"] <= iterations]
    present = sorted(int(value) for value in subset["iteration"].dropna().unique())
    if present != list(range(1, iterations + 1)):
        return False
    _, aggregate = compute_kl_iteration_precision(
        subset,
        target_half_width=target_half_width,
    )
    if aggregate.empty:
        return False
    return bool(aggregate.iloc[0]["all_meet_target"])


def write_kl_protocol_artifacts(
    run_dir: Path,
    *,
    recommendation: dict[str, object],
    frames: dict[str, object],
) -> Path:
    import pandas as pd

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(run_dir / f"{name}.csv", index=False)
    path = run_dir / "kl_protocol_recommendation.json"
    path.write_text(json.dumps(recommendation, indent=2, sort_keys=True) + "\n")
    return path


def circuits_per_fidelity_sample(n_qubits: int) -> int:
    return 2 * (3**n_qubits)


def total_expressibility_circuits(
    n_ansatzes: int,
    n_depths: int,
    n_samples: int,
    n_qubits: int,
) -> int:
    return n_ansatzes * n_depths * n_samples * circuits_per_fidelity_sample(n_qubits)


def estimate_wall_time_minutes(
    n_ansatzes: int,
    n_depths: int,
    n_samples: int,
    minutes_per_state: float,
) -> float:
    n_pairs = n_ansatzes * n_depths * n_samples
    return n_pairs * 2.0 * minutes_per_state


# ---------------------------------------------------------------------------
# Self-checks (no hardware)
# ---------------------------------------------------------------------------


def verify_expectation_endianness() -> None:
    counts_all_zero = {"00000": 1000}
    assert abs(expectation_from_counts(counts_all_zero, "ZZZZZ") - 1.0) < 1e-12
    counts_q0_one = {"00001": 1000}
    assert abs(expectation_from_counts(counts_q0_one, "ZZZZZ") - (-1.0)) < 1e-12
    assert abs(expectation_from_counts(counts_q0_one, "IZZZI") - 1.0) < 1e-12


def verify_projection_psd() -> None:
    dim = 8
    rng = np.random.default_rng(1)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    rho_noisy = rho + 0.01 * (
        rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    )
    rho_noisy = 0.5 * (rho_noisy + rho_noisy.conj().T)
    rho_phys = project_to_physical(rho_noisy)
    eigvals = np.linalg.eigvalsh(rho_phys)
    assert np.all(eigvals >= -1e-10)
    trace = float(np.real(np.trace(rho_phys)))
    assert 0.0 < trace <= 1.0 + 1e-8


def verify_haar_kl_helpers(num_qubits: int = 5, n_bins: int = 20, eps: float = 1e-12) -> None:
    dim = 2**num_qubits
    rng = np.random.default_rng(0)
    haar_samples = 1.0 - rng.random(50_000) ** (1.0 / (dim - 1))
    kl, _, _, _ = compute_kl_for_fidelities(haar_samples, dim, n_bins, eps)
    assert kl < 0.05, f"Haar self-samples should have low KL, got {kl}"


def verify_statevector_vs_tomography_overlap() -> None:
    from qiskit.quantum_info import Statevector

    n_qubits = 2
    qc_a = QuantumCircuit(n_qubits)
    qc_a.h(0)
    qc_a.cx(0, 1)
    qc_b = QuantumCircuit(n_qubits)
    qc_b.ry(0.7, 0)
    qc_b.rz(1.1, 1)

    psi_a = Statevector.from_instruction(qc_a).data
    psi_b = Statevector.from_instruction(qc_b).data
    ideal = abs(np.vdot(psi_a, psi_b)) ** 2

    simulated_counts: dict[tuple[str, ...], dict[str, int]] = {}
    for basis_tuple in all_basis_settings(n_qubits):
        qc = qc_b.copy()
        tomo = add_tomography_rotations(qc, basis_tuple)
        sv = Statevector.from_instruction(tomo.remove_final_measurements(inplace=False))
        probs = sv.probabilities_dict()
        counts = {k.replace(" ", ""): int(round(v * 10_000)) for k, v in probs.items()}
        simulated_counts[basis_tuple] = counts

    rho_lin = reconstruct_rho(simulated_counts, n_qubits)
    rho_phys = project_to_physical(rho_lin)
    rho_a = np.outer(psi_a, psi_a.conj())
    tomography_overlap = hardware_overlap(rho_a, rho_phys)
    assert abs(tomography_overlap - ideal) < 0.05, (
        f"tomography overlap {tomography_overlap} vs ideal {ideal}"
    )


def verify_count_expectation_helper() -> None:
    n = 5
    counts_all_zero = {"00000": 1000}
    for q in range(n):
        assert abs(qubit_expectation_from_counts(counts_all_zero, q, n) - 1.0) < 1e-12

    counts_q0_one = {"00001": 1000}
    assert abs(qubit_expectation_from_counts(counts_q0_one, 0, n) - (-1.0)) < 1e-12
    assert abs(qubit_expectation_from_counts(counts_q0_one, 1, n) - 1.0) < 1e-12

    counts_by_basis: dict[BasisName, dict[str, int]] = {
        "Z": counts_all_zero,
        "X": counts_all_zero,
        "Y": counts_all_zero,
    }
    assert abs(estimate_mw_from_hardware_counts(counts_by_basis, n)[0]) < 1e-12


def verify_bloch_mw_matches_statevector() -> None:
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

    n = 5
    rng = np.random.default_rng(0)
    for _ in range(20):
        psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        psi /= np.linalg.norm(psi)
        sv_mw = meyer_wallach_score(psi, n)
        x_exp: list[float] = []
        y_exp: list[float] = []
        z_exp: list[float] = []
        for i in range(n):
            rho_i = single_qubit_reduced_density(psi, i, n)
            x_exp.append(float(np.real(np.trace(rho_i @ pauli_x))))
            y_exp.append(float(np.real(np.trace(rho_i @ pauli_y))))
            z_exp.append(float(np.real(np.trace(rho_i @ pauli_z))))
        bloch_mw = mw_score_from_bloch(x_exp, y_exp, z_exp)
        assert abs(sv_mw - bloch_mw) < 1e-9, f"sv={sv_mw}, bloch={bloch_mw}"


def run_kl_self_check(num_qubits: int = 5, n_bins: int = 20, eps: float = 1e-12) -> None:
    verify_haar_kl_helpers(num_qubits, n_bins, eps)
    verify_expectation_endianness()
    verify_projection_psd()
    verify_statevector_vs_tomography_overlap()


def run_mw_self_check() -> None:
    verify_mw_implementation()
    verify_count_expectation_helper()
    verify_bloch_mw_matches_statevector()


def trimmed_param_counts_for_depths(
    n_qubits: int, depths: list[int]
) -> dict[int, int]:
    return {depth: trimmed_reverse_q0_param_count(n_qubits, depth) for depth in depths}
