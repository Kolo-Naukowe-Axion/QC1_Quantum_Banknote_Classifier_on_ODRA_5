#!/usr/bin/env python3
"""Compute KL(Sim||Haar) and persist per-pair fidelities for later re-binning.

Workflow
--------
1. Sample noiseless pairwise fidelities on the statevector simulator.
2. Append each sample to ``sim_kl_fidelities.csv`` (resume-safe).
3. Aggregate KL(Sim||Haar) for the requested bin count into ``sim_kl_results.csv``.

Re-bin later without re-sampling::

    python scripts/compute_sim_kl_expressibility.py \\
        --rebin --run-dir evaluation_and_comparison/simulator/sim_kl_outputs \\
        --n-bins 150
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import odra_ansatz, simulator_ansatz  # noqa: E402

ANSATZ_FNS: dict[str, Callable[[int, int], QuantumCircuit]] = {
    "ansatz_odra": odra_ansatz,
    "ansatz_simulator": simulator_ansatz,
}

FIDELITIES_CSV = "sim_kl_fidelities.csv"
RESULTS_CSV = "sim_kl_results.csv"
MANIFEST_JSON = "run_manifest.json"


def kl_depth_seed(base_seed: int, depth: int, ansatz: str) -> int:
    return int(base_seed) + 100 * int(depth) + (1 if str(ansatz) == "ansatz_simulator" else 0)


def bind_ansatz(
    ansatz_fn: Callable[[int, int], QuantumCircuit],
    n_qubits: int,
    depth: int,
    theta_values: np.ndarray,
) -> QuantumCircuit:
    qc = ansatz_fn(n_qubits, depth)
    bind_map = {p: float(v) for p, v in zip(qc.parameters, theta_values)}
    return qc.assign_parameters(bind_map, inplace=False)


def statevector_pairwise_fidelity(
    ansatz_fn: Callable[[int, int], QuantumCircuit],
    n_qubits: int,
    depth: int,
    theta_a: np.ndarray,
    theta_b: np.ndarray,
) -> float:
    psi_a = Statevector.from_instruction(bind_ansatz(ansatz_fn, n_qubits, depth, theta_a)).data
    psi_b = Statevector.from_instruction(bind_ansatz(ansatz_fn, n_qubits, depth, theta_b)).data
    return float(abs(np.vdot(psi_a, psi_b)) ** 2)


def haar_pdf_fidelity(f: np.ndarray, dim: int) -> np.ndarray:
    return (dim - 1.0) * (1.0 - f) ** (dim - 2.0)


def compute_kl_for_fidelities(
    fidelities: np.ndarray,
    dim: int,
    n_bins: int,
    eps: float,
) -> float:
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    counts, edges = np.histogram(fidelities, bins=bins, density=False)
    p_emp = counts.astype(np.float64)
    if p_emp.sum() == 0:
        p_emp = np.ones_like(p_emp) / len(p_emp)
    else:
        p_emp /= p_emp.sum()

    mids = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    p_haar = haar_pdf_fidelity(mids, dim=dim) * width
    p_haar /= p_haar.sum()

    p_s = p_emp + eps
    q_s = p_haar + eps
    p_s /= p_s.sum()
    q_s /= q_s.sum()
    return float(np.sum(p_s * np.log(p_s / q_s)))


def append_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_sample_indices(path: Path, ansatz: str, depth: int) -> set[int]:
    indices: set[int] = set()
    for row in read_csv_rows(path):
        if row.get("ansatz") == ansatz and int(row.get("depth", -1)) == int(depth):
            indices.add(int(row["sample_index"]))
    return indices


def load_job_fidelities(path: Path, ansatz: str, depth: int) -> np.ndarray:
    rows = [
        row
        for row in read_csv_rows(path)
        if row.get("ansatz") == ansatz and int(row.get("depth", -1)) == int(depth)
    ]
    if not rows:
        return np.array([], dtype=np.float64)
    rows.sort(key=lambda row: int(row["sample_index"]))
    return np.array([float(row["fidelity"]) for row in rows], dtype=np.float64)


def sample_and_persist_job(
    *,
    ansatz_name: str,
    ansatz_fn: Callable[[int, int], QuantumCircuit],
    n_qubits: int,
    depth: int,
    n_samples: int,
    seed: int,
    fidelities_path: Path,
    resume: bool,
    progress_every: int,
) -> np.ndarray:
    n_params = len(ansatz_fn(n_qubits, depth).parameters)
    rng = np.random.default_rng(int(seed))
    done = completed_sample_indices(fidelities_path, ansatz_name, depth) if resume else set()

    if done:
        print(
            f"  resuming {ansatz_name} depth={depth}: "
            f"{len(done)}/{n_samples} samples on disk",
            flush=True,
        )

    # Advance RNG to match samples already written for this job.
    for sample_index in range(n_samples):
        theta_a = rng.uniform(0.0, 2.0 * np.pi, n_params)
        theta_b = rng.uniform(0.0, 2.0 * np.pi, n_params)
        if sample_index in done:
            continue

        fidelity = statevector_pairwise_fidelity(
            ansatz_fn,
            n_qubits,
            depth,
            theta_a,
            theta_b,
        )
        append_csv_row(
            fidelities_path,
            {
                "ansatz": ansatz_name,
                "depth": int(depth),
                "sample_index": int(sample_index),
                "fidelity": float(fidelity),
                "seed": int(seed),
            },
        )
        if progress_every > 0 and (sample_index + 1) % progress_every == 0:
            print(
                f"  {ansatz_name} depth={depth}: "
                f"{sample_index + 1}/{n_samples} samples",
                flush=True,
            )

    fidelities = load_job_fidelities(fidelities_path, ansatz_name, depth)
    if len(fidelities) < n_samples:
        raise RuntimeError(
            f"Incomplete job {ansatz_name} depth={depth}: "
            f"expected {n_samples}, got {len(fidelities)}"
        )
    return fidelities[:n_samples]


def upsert_results_row(results_path: Path, row: dict[str, object]) -> None:
    rows = read_csv_rows(results_path)
    key = (str(row["ansatz"]), int(row["depth"]), int(row["n_bins"]))
    kept = [
        existing
        for existing in rows
        if (
            existing.get("ansatz"),
            int(existing.get("depth", -1)),
            int(existing.get("n_bins", -1)),
        )
        != key
    ]
    kept.append({key: str(value) for key, value in row.items()})
    fieldnames = list(row.keys())
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def iter_jobs(
    ansatz_names: Iterable[str],
    depths: Iterable[int],
) -> list[tuple[str, Callable[[int, int], QuantumCircuit], int]]:
    jobs: list[tuple[str, Callable[[int, int], QuantumCircuit], int]] = []
    for ansatz_name in ansatz_names:
        if ansatz_name not in ANSATZ_FNS:
            raise ValueError(f"Unknown ansatz: {ansatz_name}")
        for depth in depths:
            jobs.append((ansatz_name, ANSATZ_FNS[ansatz_name], int(depth)))
    return jobs


def run_sampling(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fidelities_path = run_dir / FIDELITIES_CSV
    results_path = run_dir / RESULTS_CSV
    dim = 2 ** int(args.n_qubits)

    jobs = iter_jobs(args.ansatz, args.depths)
    started = time.perf_counter()

    for ansatz_name, ansatz_fn, depth in jobs:
        depth_seed = kl_depth_seed(args.seed, depth, ansatz_name)
        job_started = time.perf_counter()
        print(
            f"[{ansatz_name} depth={depth}] target {args.n_samples} pairs "
            f"(seed={depth_seed})",
            flush=True,
        )
        fidelities = sample_and_persist_job(
            ansatz_name=ansatz_name,
            ansatz_fn=ansatz_fn,
            n_qubits=args.n_qubits,
            depth=depth,
            n_samples=args.n_samples,
            seed=depth_seed,
            fidelities_path=fidelities_path,
            resume=args.resume,
            progress_every=args.progress_every,
        )
        kl = compute_kl_for_fidelities(fidelities, dim, args.n_bins, args.eps)
        elapsed = time.perf_counter() - job_started
        upsert_results_row(
            results_path,
            {
                "ansatz": ansatz_name,
                "depth": int(depth),
                "n_qubits": int(args.n_qubits),
                "n_samples": int(args.n_samples),
                "n_bins": int(args.n_bins),
                "eps": float(args.eps),
                "seed": int(depth_seed),
                "kl_sim_haar": float(kl),
                "fidelity_mean": float(np.mean(fidelities)),
                "fidelity_std": float(np.std(fidelities)),
                "wall_seconds": float(elapsed),
            },
        )
        print(f"  kl_sim_haar={kl:.6g}  ({elapsed:.1f}s)", flush=True)

    total_elapsed = time.perf_counter() - started
    write_manifest(
        run_dir / MANIFEST_JSON,
        {
            "mode": "sample",
            "metric": "kl_sim_haar",
            "base_seed": int(args.seed),
            "n_qubits": int(args.n_qubits),
            "depths": list(args.depths),
            "ansatz": list(args.ansatz),
            "n_samples": int(args.n_samples),
            "n_bins": int(args.n_bins),
            "eps": float(args.eps),
            "total_wall_seconds": float(total_elapsed),
            "outputs": [FIDELITIES_CSV, RESULTS_CSV],
        },
    )
    print(f"\nTotal wall time: {total_elapsed / 60:.1f} min")
    print(f"Saved fidelities: {fidelities_path}")
    print(f"Saved KL summary: {results_path}")


def run_rebin(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    fidelities_path = run_dir / FIDELITIES_CSV
    results_path = run_dir / RESULTS_CSV
    if not fidelities_path.is_file():
        raise SystemExit(f"Missing fidelity archive: {fidelities_path}")

    dim = 2 ** int(args.n_qubits)
    jobs = iter_jobs(args.ansatz, args.depths)
    rows: list[dict[str, object]] = []

    for ansatz_name, _, depth in jobs:
        fidelities = load_job_fidelities(fidelities_path, ansatz_name, depth)
        if fidelities.size == 0:
            raise SystemExit(f"No saved fidelities for {ansatz_name} depth={depth}")
        depth_seed = kl_depth_seed(args.seed, depth, ansatz_name)
        kl = compute_kl_for_fidelities(fidelities, dim, args.n_bins, args.eps)
        row = {
            "ansatz": ansatz_name,
            "depth": int(depth),
            "n_qubits": int(args.n_qubits),
            "n_samples": int(len(fidelities)),
            "n_bins": int(args.n_bins),
            "eps": float(args.eps),
            "seed": int(depth_seed),
            "kl_sim_haar": float(kl),
            "fidelity_mean": float(np.mean(fidelities)),
            "fidelity_std": float(np.std(fidelities)),
            "wall_seconds": 0.0,
        }
        upsert_results_row(results_path, row)
        rows.append(row)
        print(
            f"[{ansatz_name} depth={depth}] "
            f"n={len(fidelities)} bins={args.n_bins} -> kl_sim_haar={kl:.6g}",
            flush=True,
        )

    write_manifest(
        run_dir / MANIFEST_JSON,
        {
            "mode": "rebin",
            "metric": "kl_sim_haar",
            "base_seed": int(args.seed),
            "n_qubits": int(args.n_qubits),
            "depths": list(args.depths),
            "ansatz": list(args.ansatz),
            "n_bins": int(args.n_bins),
            "eps": float(args.eps),
            "rows": rows,
            "outputs": [RESULTS_CSV],
        },
    )
    print(f"\nUpdated KL summary: {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample Sim||Haar fidelities, save per-pair rows, aggregate KL."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "evaluation_and_comparison" / "simulator" / "sim_kl_outputs",
        help="Directory for sim_kl_fidelities.csv and sim_kl_results.csv.",
    )
    parser.add_argument(
        "--rebin",
        action="store_true",
        help="Recompute KL from saved fidelities only (no statevector resampling).",
    )
    parser.add_argument("--n-qubits", type=int, default=5)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4])
    parser.add_argument(
        "--ansatz",
        nargs="+",
        default=["ansatz_odra", "ansatz_simulator"],
        choices=sorted(ANSATZ_FNS),
    )
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--n-bins", type=int, default=75)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sample indices already present in sim_kl_fidelities.csv.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print sampling progress every N pairs (0 disables).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebin:
        run_rebin(args)
    else:
        run_sampling(args)


if __name__ == "__main__":
    main()
