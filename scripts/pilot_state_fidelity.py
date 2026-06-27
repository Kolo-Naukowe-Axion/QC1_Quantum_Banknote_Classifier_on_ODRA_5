#!/usr/bin/env python3
"""Run the state-fidelity pilot protocol and write precision recommendations.

State fidelity here is the hardware state-tomography fidelity computed in
``full_odra_fidelity.ipynb``:

    F = <psi_ideal | rho_hardware | psi_ideal>

where ``psi_ideal`` is the noiseless statevector of the bound circuit
(feature map + trained ansatz) and ``rho_hardware`` is reconstructed from a
full 3^n Pauli-basis tomography sweep on IQM Spark, then projected to the
physical set. The headline metric is the physical-projection fidelity
(``F_phys``).

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import (  # noqa: E402
    odra_ansatz as ansatz_odra,
    simulator_ansatz as ansatz_simulator,
)
from qbanknote.iqm import connect_to_iqm_backend  # noqa: E402
from qbanknote.metrics import (  # noqa: E402
    choose_fidelity_iterations,
    choose_fidelity_samples,
    choose_fidelity_shots,
    compute_fidelity_iteration_precision,
    compute_fidelity_sample_precision,
    compute_fidelity_shot_stability,
    fidelity_mean_shot_noise_bound,
    fidelity_shot_noise_sd_bound,
    iteration_target_met,
    read_fidelity_summary,
    run_iqm_fidelity_sweep,
    write_fidelity_protocol_artifacts,
)
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402
from qbanknote.progress import make_print_callback  # noqa: E402

DEFAULT_OUTPUT_ROOT = "evaluation_and_comparison/iqm_spark/iqm_fidelity_outputs"
DEFAULT_DEPTHS = [2, 4, 6]
DEFAULT_ANSATZES = ("ansatz_odra", "ansatz_simulator")
DEFAULT_SHOT_GRID = [512, 1024, 2048, 4096]
DEFAULT_SAMPLE_GRID = [10, 20, 40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pilot-calibrated state-fidelity protocol on IQM Spark"
    )
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    parser.add_argument("--shot-grid", type=int, nargs="+", default=DEFAULT_SHOT_GRID)
    parser.add_argument("--sample-grid", type=int, nargs="+", default=DEFAULT_SAMPLE_GRID)
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Fixed tomography shot count per Pauli basis for --drift-only mode.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Fixed test-input count for --drift-only mode.",
    )
    parser.add_argument(
        "--pilot-samples",
        type=int,
        default=10,
        help="Test inputs used for the shot-stability pilot.",
    )
    parser.add_argument(
        "--shot-tolerance",
        type=float,
        default=0.02,
        help="Max allowed fidelity mean change between consecutive shot budgets.",
    )
    parser.add_argument(
        "--target-half-width",
        type=float,
        default=0.03,
        help="Target 95%% half-width for per-input fidelity mean estimates.",
    )
    parser.add_argument(
        "--target-iteration-half-width",
        type=float,
        default=0.01,
        help="Target 95%% half-width for run-to-run fidelity drift across repeated iterations.",
    )
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=3,
        help="Minimum repeated frozen sweeps in the iteration pilot.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum repeated frozen sweeps in the iteration pilot.",
    )
    parser.add_argument(
        "--drift-only",
        action="store_true",
        help="Skip shot/sample pilots and run only repeated frozen iterations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--max-circuits-per-job", type=int, default=275)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Pilot output root (default: <project>/iqm_fidelity_outputs/pilots/<pilot_id>)",
    )
    parser.add_argument("--pilot-id", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iqm-token", default=None)
    parser.add_argument(
        "--iqm-url",
        default=os.environ.get("IQM_URL", "https://odra5.e-science.pl/").strip(),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _validate_positive_ints(name: str, values: list[int]) -> list[int]:
    clean = sorted(set(int(value) for value in values))
    if not clean or any(value <= 0 for value in clean):
        raise SystemExit(f"{name} must contain positive integers")
    return clean


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_iteration_pilot(
    *,
    backend,
    ansatz_fns,
    args,
    output_root: Path,
    pilot_id: str,
    chosen_shots: int,
    chosen_samples: int,
    progress_callback,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, bool]:
    iteration_frames: list[pd.DataFrame] = []
    chosen_iterations = 0
    target_met = False

    for iteration in range(1, args.max_iterations + 1):
        run_dir = output_root / "iteration_pilot" / f"iteration_{iteration}"
        if not args.quiet:
            print(
                f"\n[iteration pilot] iteration={iteration}, "
                f"n_samples={chosen_samples}, shots={chosen_shots} -> {run_dir}"
            )
        run_iqm_fidelity_sweep(
            backend,
            ansatz_fns=ansatz_fns,
            ansatz_names=list(args.ansatz),
            depths=list(args.depth),
            n_qubits=args.num_qubits,
            n_samples=chosen_samples,
            seed=args.seed,
            shots=chosen_shots,
            optimization_level=args.optimization_level,
            seed_transpiler=None,
            max_circuits_per_job=args.max_circuits_per_job,
            output_dir=run_dir,
            resume=args.resume,
            verbose=not args.quiet,
            progress_callback=progress_callback,
            manifest_extra={
                "pilot_id": pilot_id,
                "pilot_stage": "iteration_pilot",
                "iteration": iteration,
                "iqm_url": args.iqm_url,
            },
        )
        iteration_frames.append(
            read_fidelity_summary(run_dir, stage="iteration_pilot", iteration=iteration)
        )
        iteration_summary = _concat(iteration_frames)
        chosen_iterations = iteration
        target_met = iteration_target_met(
            iteration_summary,
            target_half_width=args.target_iteration_half_width,
            iterations=iteration,
        )
        if iteration >= args.min_iterations and target_met:
            break

    iteration_summary = _concat(iteration_frames)
    iteration_precision, iteration_precision_aggregate = compute_fidelity_iteration_precision(
        iteration_summary,
        target_half_width=args.target_iteration_half_width,
    )
    if not target_met and chosen_iterations >= args.min_iterations:
        chosen_iterations = choose_fidelity_iterations(
            iteration_summary,
            target_half_width=args.target_iteration_half_width,
            min_iterations=args.min_iterations,
            max_iterations=args.max_iterations,
        )
        target_met = iteration_target_met(
            iteration_summary,
            target_half_width=args.target_iteration_half_width,
            iterations=chosen_iterations,
        )

    return (
        iteration_summary,
        iteration_precision,
        iteration_precision_aggregate,
        chosen_iterations,
        target_met,
    )


def main() -> None:
    ensure_importable()
    args = parse_args()
    project_root = find_project_root(ROOT)

    ansatz_fns = {
        "ansatz_odra": ansatz_odra,
        "ansatz_simulator": ansatz_simulator,
    }
    for name in args.ansatz:
        if name not in ansatz_fns:
            raise SystemExit(f"Unknown ansatz: {name}. Choose from {sorted(ansatz_fns)}")

    if args.min_iterations < 2:
        raise SystemExit("--min-iterations must be at least 2")
    if args.max_iterations < args.min_iterations:
        raise SystemExit("--max-iterations must be >= --min-iterations")

    pilot_id = args.pilot_id or datetime.now(tz=timezone.utc).strftime(
        "fidelity_pilot_%Y%m%d_%H%M%S"
    )
    output_root = (
        Path(args.output_root)
        if args.output_root
        else project_root / DEFAULT_OUTPUT_ROOT / "pilots" / pilot_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    backend = connect_to_iqm_backend(args.iqm_url, token=args.iqm_token)
    progress_callback = None if args.quiet else make_print_callback()

    shot_summary = pd.DataFrame()
    shot_stability = pd.DataFrame()
    shot_stability_aggregate = pd.DataFrame()
    sample_summary = pd.DataFrame()
    sample_precision = pd.DataFrame()
    sample_precision_aggregate = pd.DataFrame()
    chosen_shots = args.shots
    chosen_samples = args.samples

    if args.drift_only:
        if chosen_shots is None or chosen_samples is None:
            raise SystemExit("--drift-only requires fixed --shots and --samples")
        if not args.quiet:
            print(f"Drift-only pilot at shots={chosen_shots}, samples={chosen_samples}")
    else:
        shot_grid = _validate_positive_ints("shot-grid", args.shot_grid)
        sample_grid = _validate_positive_ints("sample-grid", args.sample_grid)
        if args.pilot_samples <= 0:
            raise SystemExit("--pilot-samples must be positive")
        if not args.quiet:
            print(f"Pilot output root: {output_root}")
            print(f"Shot grid: {shot_grid} with pilot samples={args.pilot_samples}")
            print(f"Sample grid: {sample_grid}")

        shot_frames: list[pd.DataFrame] = []
        for shots in shot_grid:
            run_dir = output_root / "shot_pilot" / f"shots_{shots}"
            if not args.quiet:
                print(f"\n[shot pilot] shots={shots} -> {run_dir}")
            run_iqm_fidelity_sweep(
                backend,
                ansatz_fns=ansatz_fns,
                ansatz_names=list(args.ansatz),
                depths=list(args.depth),
                n_qubits=args.num_qubits,
                n_samples=args.pilot_samples,
                seed=args.seed,
                shots=shots,
                optimization_level=args.optimization_level,
                seed_transpiler=None,
                max_circuits_per_job=args.max_circuits_per_job,
                output_dir=run_dir,
                resume=args.resume,
                verbose=not args.quiet,
                progress_callback=progress_callback,
                manifest_extra={
                    "pilot_id": pilot_id,
                    "pilot_stage": "shot_pilot",
                    "iqm_url": args.iqm_url,
                },
            )
            shot_frames.append(read_fidelity_summary(run_dir, stage="shot_pilot"))

        shot_summary = _concat(shot_frames)
        shot_stability, shot_stability_aggregate = compute_fidelity_shot_stability(shot_summary)
        chosen_shots = choose_fidelity_shots(shot_summary, tolerance=args.shot_tolerance)
        if not args.quiet:
            print(f"\nChosen shots: {chosen_shots}")

        sample_frames: list[pd.DataFrame] = []
        for n_samples in sample_grid:
            run_dir = output_root / "sample_pilot" / f"samples_{n_samples}"
            if not args.quiet:
                print(
                    f"\n[sample pilot] n_samples={n_samples}, shots={chosen_shots} -> {run_dir}"
                )
            run_iqm_fidelity_sweep(
                backend,
                ansatz_fns=ansatz_fns,
                ansatz_names=list(args.ansatz),
                depths=list(args.depth),
                n_qubits=args.num_qubits,
                n_samples=n_samples,
                seed=args.seed,
                shots=chosen_shots,
                optimization_level=args.optimization_level,
                seed_transpiler=None,
                max_circuits_per_job=args.max_circuits_per_job,
                output_dir=run_dir,
                resume=args.resume,
                verbose=not args.quiet,
                progress_callback=progress_callback,
                manifest_extra={
                    "pilot_id": pilot_id,
                    "pilot_stage": "sample_pilot",
                    "iqm_url": args.iqm_url,
                },
            )
            sample_frames.append(read_fidelity_summary(run_dir, stage="sample_pilot"))

        sample_summary = _concat(sample_frames)
        sample_precision, sample_precision_aggregate = compute_fidelity_sample_precision(
            sample_summary,
            target_half_width=args.target_half_width,
            z_value=args.confidence_z,
        )
        chosen_samples = choose_fidelity_samples(
            sample_summary,
            target_half_width=args.target_half_width,
            z_value=args.confidence_z,
        )
        if not args.quiet:
            print(f"\nChosen n_samples: {chosen_samples}")

    assert chosen_shots is not None and chosen_samples is not None

    (
        iteration_summary,
        iteration_precision,
        iteration_precision_aggregate,
        chosen_iterations,
        iteration_target_met_flag,
    ) = _run_iteration_pilot(
        backend=backend,
        ansatz_fns=ansatz_fns,
        args=args,
        output_root=output_root,
        pilot_id=pilot_id,
        chosen_shots=int(chosen_shots),
        chosen_samples=int(chosen_samples),
        progress_callback=progress_callback,
    )

    if not args.quiet:
        print(f"\nChosen iterations: {chosen_iterations}")
        print(f"Iteration target met: {iteration_target_met_flag}")

    iteration_stability = iteration_precision.drop(
        columns=["target_half_width", "meets_target"], errors="ignore"
    )

    recommendation = {
        "pilot_id": pilot_id,
        "iqm_url": args.iqm_url,
        "depths": list(args.depth),
        "ansatzes": list(args.ansatz),
        "drift_only": bool(args.drift_only),
        "shot_grid": [] if args.drift_only else list(args.shot_grid),
        "sample_grid": [] if args.drift_only else list(args.sample_grid),
        "pilot_samples": args.pilot_samples,
        "shot_tolerance": args.shot_tolerance,
        "target_half_width": args.target_half_width,
        "target_iteration_half_width": args.target_iteration_half_width,
        "confidence_z": args.confidence_z,
        "min_iterations": int(args.min_iterations),
        "max_iterations": int(args.max_iterations),
        "chosen_shots": int(chosen_shots),
        "chosen_n_samples": int(chosen_samples),
        "chosen_iterations": int(chosen_iterations),
        "iteration_target_met": bool(iteration_target_met_flag),
        "iterations_run": int(chosen_iterations),
        "single_sample_shot_noise_sd_bound": fidelity_shot_noise_sd_bound(
            args.num_qubits,
            int(chosen_shots),
        ),
        "mean_shot_noise_bound_at_chosen_samples": fidelity_mean_shot_noise_bound(
            args.num_qubits,
            int(chosen_shots),
            int(chosen_samples),
        ),
        "output_root": str(output_root),
        "methodology": {
            "shot_rule": (
                "Choose smallest tomography shot count (per Pauli basis) whose max "
                "consecutive physical-projection fidelity mean change is <= shot_tolerance."
            ),
            "sample_rule": (
                "Choose smallest n_samples (test inputs) whose worst 95% half-width is "
                "<= target_half_width; otherwise use required_n_samples."
            ),
            "iteration_rule": (
                "Repeat frozen protocol with identical seed; choose smallest K >= min_iterations "
                "such that max 95% iteration half-width <= target_iteration_half_width, "
                "otherwise use max_iterations."
            ),
        },
    }

    recommendation_path = write_fidelity_protocol_artifacts(
        output_root,
        recommendation=recommendation,
        frames={
            "shot_pilot_summary": shot_summary,
            "shot_stability": shot_stability,
            "shot_stability_aggregate": shot_stability_aggregate,
            "sample_pilot_summary": sample_summary,
            "sample_precision": sample_precision,
            "sample_precision_aggregate": sample_precision_aggregate,
            "iteration_summary": iteration_summary,
            "iteration_stability": iteration_stability,
            "iteration_precision": iteration_precision,
            "iteration_precision_aggregate": iteration_precision_aggregate,
        },
    )

    print(json.dumps(recommendation, indent=2, sort_keys=True))
    print(f"Wrote state-fidelity protocol recommendation to {recommendation_path}")


if __name__ == "__main__":
    main()
