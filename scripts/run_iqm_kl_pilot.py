#!/usr/bin/env python3
"""Run the KL expressibility pilot protocol and write hyperparameter recommendations."""

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
    choose_kl_bins,
    choose_kl_iterations,
    choose_kl_samples,
    choose_kl_shots,
    compute_kl_bin_sensitivity,
    compute_kl_iteration_precision,
    compute_kl_prefix_precision,
    compute_kl_shot_stability,
    kl_iteration_target_met,
    read_kl_fidelities,
    read_kl_summary,
    run_iqm_kl_sweep,
    write_kl_protocol_artifacts,
)
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402
from qbanknote.progress import make_print_callback  # noqa: E402

DEFAULT_OUTPUT_ROOT = "evaluation_and_comparison/iqm_spark/iqm_kl_outputs"
DEFAULT_DEPTHS = [2, 4, 6]
DEFAULT_SHOT_PILOT_DEPTHS = [2, 4, 6]
DEFAULT_ANSATZES = ("ansatz_odra", "ansatz_simulator")
DEFAULT_SHOT_GRID = [512, 1024, 2048, 4096]
DEFAULT_SAMPLE_GRID = [5, 8, 10, 12, 15]
DEFAULT_BIN_GRID = [50, 75, 100, 150, 200]
DEFAULT_PILOT_SAMPLES = 3
MINUTES_PER_FIDELITY_PAIR = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pilot-calibrated KL expressibility protocol on IQM Spark"
    )
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument(
        "--shot-pilot-depth",
        type=int,
        nargs="+",
        default=DEFAULT_SHOT_PILOT_DEPTHS,
        help="Depths used in the shot-stability pilot (default: 2, 4, 6 — MW-aligned).",
    )
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    parser.add_argument("--shot-grid", type=int, nargs="+", default=DEFAULT_SHOT_GRID)
    parser.add_argument("--sample-grid", type=int, nargs="+", default=DEFAULT_SAMPLE_GRID)
    parser.add_argument("--bin-grid", type=int, nargs="+", default=DEFAULT_BIN_GRID)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=15,
        help="Fidelity pairs collected once for offline prefix/sample analysis.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Fixed shot count for --drift-only mode.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Fixed sample count for --drift-only mode.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Fixed histogram bin count for --drift-only mode.",
    )
    parser.add_argument(
        "--pilot-samples",
        type=int,
        default=DEFAULT_PILOT_SAMPLES,
        help="Fidelity pairs used in the shot-stability pilot (default: 3).",
    )
    parser.add_argument(
        "--shot-tolerance",
        type=float,
        default=0.02,
        help="Max allowed |Delta KL_physical| between consecutive shot budgets.",
    )
    parser.add_argument(
        "--bin-tolerance",
        type=float,
        default=0.01,
        help="Max allowed KL discretization bias vs a fine reference histogram.",
    )
    parser.add_argument(
        "--target-half-width",
        type=float,
        default=0.03,
        help="Target 95%% bootstrap half-width for KL_physical estimates.",
    )
    parser.add_argument(
        "--target-iteration-half-width",
        type=float,
        default=0.02,
        help="Target 95%% half-width for run-to-run KL drift across repeated iterations.",
    )
    parser.add_argument("--bootstrap-trials", type=int, default=400)
    parser.add_argument("--bin-trials", type=int, default=100)
    parser.add_argument("--reference-bins", type=int, default=400)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=2,
        help="Minimum repeated frozen sweeps in the iteration pilot.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=4,
        help="Maximum repeated frozen sweeps in the iteration pilot.",
    )
    parser.add_argument(
        "--drift-only",
        action="store_true",
        help="Skip shot/sample/bin pilots and run only repeated frozen iterations.",
    )
    parser.add_argument(
        "--skip-bins",
        action="store_true",
        help="Skip offline bin-sensitivity pilot and keep --n-bins or default 150.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--max-circuits-per-job", type=int, default=250)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Pilot output root (default: <project>/iqm_kl_outputs/pilots/<pilot_id>)",
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


def estimate_kl_pilot_budget(
    *,
    n_ansatzes: int,
    shot_grid: list[int],
    shot_pilot_depths: list[int],
    pilot_samples: int,
    depths: list[int],
    max_samples: int,
    max_iterations: int,
    drift_only: bool,
    minutes_per_pair: float = MINUTES_PER_FIDELITY_PAIR,
) -> dict[str, int | float]:
    """Estimate fidelity-pair counts and wall time for each pilot stage."""
    if drift_only:
        shot_pairs = 0
        sample_pairs = 0
    else:
        shot_pairs = len(shot_grid) * n_ansatzes * len(shot_pilot_depths) * pilot_samples
        sample_pairs = n_ansatzes * len(depths) * max_samples
    iteration_pairs = max_iterations * n_ansatzes * len(depths) * max_samples
    total_pairs = shot_pairs + sample_pairs + iteration_pairs
    return {
        "shot_pairs": int(shot_pairs),
        "sample_pairs": int(sample_pairs),
        "iteration_pairs": int(iteration_pairs),
        "total_pairs": int(total_pairs),
        "shot_hours": float(shot_pairs * minutes_per_pair / 60.0),
        "sample_hours": float(sample_pairs * minutes_per_pair / 60.0),
        "iteration_hours": float(iteration_pairs * minutes_per_pair / 60.0),
        "total_hours": float(total_pairs * minutes_per_pair / 60.0),
    }


def _run_iteration_pilot(
    *,
    backend,
    ansatz_fns,
    args,
    output_root: Path,
    pilot_id: str,
    chosen_shots: int,
    chosen_samples: int,
    chosen_bins: int,
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
                f"n_samples={chosen_samples}, shots={chosen_shots}, "
                f"n_bins={chosen_bins} -> {run_dir}"
            )
        run_iqm_kl_sweep(
            backend,
            ansatz_fns=ansatz_fns,
            ansatz_names=list(args.ansatz),
            depths=list(args.depth),
            n_qubits=args.num_qubits,
            n_samples=chosen_samples,
            seed=args.seed,
            shots=chosen_shots,
            n_bins=chosen_bins,
            eps=args.eps,
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
            read_kl_summary(run_dir, stage="iteration_pilot", iteration=iteration)
        )
        iteration_summary = _concat(iteration_frames)
        chosen_iterations = iteration
        target_met = kl_iteration_target_met(
            iteration_summary,
            target_half_width=args.target_iteration_half_width,
            iterations=iteration,
        )
        if iteration >= args.min_iterations and target_met:
            break

    iteration_summary = _concat(iteration_frames)
    iteration_precision, iteration_precision_aggregate = compute_kl_iteration_precision(
        iteration_summary,
        target_half_width=args.target_iteration_half_width,
    )
    if not target_met and chosen_iterations >= args.min_iterations:
        chosen_iterations = choose_kl_iterations(
            iteration_summary,
            target_half_width=args.target_iteration_half_width,
            min_iterations=args.min_iterations,
            max_iterations=args.max_iterations,
        )
        target_met = kl_iteration_target_met(
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
    dim = 2 ** args.num_qubits

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
    if args.max_samples < max(args.sample_grid, default=0):
        raise SystemExit("--max-samples must be >= max(sample-grid)")

    pilot_id = args.pilot_id or datetime.now(tz=timezone.utc).strftime("kl_pilot_%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else project_root / DEFAULT_OUTPUT_ROOT / "pilots" / pilot_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.quiet and not args.drift_only:
        budget = estimate_kl_pilot_budget(
            n_ansatzes=len(args.ansatz),
            shot_grid=list(args.shot_grid),
            shot_pilot_depths=list(args.shot_pilot_depth),
            pilot_samples=args.pilot_samples,
            depths=list(args.depth),
            max_samples=args.max_samples,
            max_iterations=args.max_iterations,
            drift_only=args.drift_only,
        )
        print(
            "Estimated QPU budget (MW-aligned shot pilot defaults, "
            f"~{MINUTES_PER_FIDELITY_PAIR:.0f} min/fidelity pair):"
        )
        print(
            f"  shot pilot:      {budget['shot_pairs']:>4} pairs  ~{budget['shot_hours']:.1f} h"
        )
        print(
            f"  sample pilot:    {budget['sample_pairs']:>4} pairs  ~{budget['sample_hours']:.1f} h"
        )
        print(
            f"  iteration pilot: {budget['iteration_pairs']:>4} pairs  "
            f"~{budget['iteration_hours']:.1f} h  (up to {args.max_iterations} iterations)"
        )
        print(f"  total:           {budget['total_pairs']:>4} pairs  ~{budget['total_hours']:.1f} h")

    bin_sensitivity = pd.DataFrame()
    bin_sensitivity_aggregate = pd.DataFrame()
    chosen_bins = args.n_bins if args.n_bins is not None else 150

    if not args.skip_bins and not args.drift_only:
        bin_grid = _validate_positive_ints("bin-grid", args.bin_grid)
        if not args.quiet:
            print(
                f"Offline bin pilot: grid={bin_grid}, "
                f"planning_n_samples={max(args.sample_grid)}"
            )
        bin_sensitivity, bin_sensitivity_aggregate = compute_kl_bin_sensitivity(
            num_qubits=args.num_qubits,
            n_samples=max(args.sample_grid),
            bin_grid=bin_grid,
            n_reference_bins=args.reference_bins,
            eps=args.eps,
            seed=args.seed,
            n_trials=args.bin_trials,
        )
        chosen_bins = choose_kl_bins(
            bin_sensitivity_aggregate,
            tolerance=args.bin_tolerance,
            fallback=max(bin_grid),
        )
        if not args.quiet:
            print(f"Chosen n_bins: {chosen_bins}")

    shot_summary = pd.DataFrame()
    shot_stability = pd.DataFrame()
    shot_stability_aggregate = pd.DataFrame()
    sample_precision = pd.DataFrame()
    sample_precision_aggregate = pd.DataFrame()
    sample_fidelities = pd.DataFrame()
    chosen_shots = args.shots
    chosen_samples = args.samples

    backend = None
    progress_callback = None

    if args.drift_only:
        if chosen_shots is None or chosen_samples is None:
            raise SystemExit("--drift-only requires fixed --shots and --samples")
        if args.n_bins is not None:
            chosen_bins = int(args.n_bins)
        if not args.quiet:
            print(
                f"Drift-only pilot at shots={chosen_shots}, "
                f"samples={chosen_samples}, n_bins={chosen_bins}"
            )
        backend = connect_to_iqm_backend(args.iqm_url, token=args.iqm_token)
        progress_callback = None if args.quiet else make_print_callback()
    else:
        backend = connect_to_iqm_backend(args.iqm_url, token=args.iqm_token)
        progress_callback = None if args.quiet else make_print_callback()

        shot_grid = _validate_positive_ints("shot-grid", args.shot_grid)
        sample_grid = _validate_positive_ints("sample-grid", args.sample_grid)
        if args.pilot_samples <= 0:
            raise SystemExit("--pilot-samples must be positive")
        if not args.quiet:
            print(f"Pilot output root: {output_root}")
            print(
                f"Shot pilot depths={list(args.shot_pilot_depth)}, "
                f"grid={shot_grid}, pilot_samples={args.pilot_samples}"
            )
            print(f"Sample prefix grid={sample_grid}, max_samples={args.max_samples}")

        shot_frames: list[pd.DataFrame] = []
        for shots in shot_grid:
            run_dir = output_root / "shot_pilot" / f"shots_{shots}"
            if not args.quiet:
                print(f"\n[shot pilot] shots={shots} -> {run_dir}")
            run_iqm_kl_sweep(
                backend,
                ansatz_fns=ansatz_fns,
                ansatz_names=list(args.ansatz),
                depths=list(args.shot_pilot_depth),
                n_qubits=args.num_qubits,
                n_samples=args.pilot_samples,
                seed=args.seed,
                shots=shots,
                n_bins=chosen_bins,
                eps=args.eps,
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
            shot_frames.append(read_kl_summary(run_dir, stage="shot_pilot"))

        shot_summary = _concat(shot_frames)
        shot_stability, shot_stability_aggregate = compute_kl_shot_stability(shot_summary)
        chosen_shots = choose_kl_shots(shot_summary, tolerance=args.shot_tolerance)
        if not args.quiet:
            print(f"\nChosen shots: {chosen_shots}")

        sample_run_dir = output_root / "sample_pilot" / f"samples_{args.max_samples}"
        if not args.quiet:
            print(
                f"\n[sample pilot] collecting max_samples={args.max_samples}, "
                f"shots={chosen_shots} -> {sample_run_dir}"
            )
        run_iqm_kl_sweep(
            backend,
            ansatz_fns=ansatz_fns,
            ansatz_names=list(args.ansatz),
            depths=list(args.depth),
            n_qubits=args.num_qubits,
            n_samples=args.max_samples,
            seed=args.seed,
            shots=int(chosen_shots),
            n_bins=chosen_bins,
            eps=args.eps,
            optimization_level=args.optimization_level,
            seed_transpiler=None,
            max_circuits_per_job=args.max_circuits_per_job,
            output_dir=sample_run_dir,
            resume=args.resume,
            verbose=not args.quiet,
            progress_callback=progress_callback,
            manifest_extra={
                "pilot_id": pilot_id,
                "pilot_stage": "sample_pilot",
                "iqm_url": args.iqm_url,
            },
        )
        sample_fidelities = read_kl_fidelities(sample_run_dir)
        sample_precision, sample_precision_aggregate = compute_kl_prefix_precision(
            sample_fidelities,
            sample_grid=sample_grid,
            dim=dim,
            n_bins=chosen_bins,
            eps=args.eps,
            target_half_width=args.target_half_width,
            n_bootstrap=args.bootstrap_trials,
            seed=args.seed,
            z_value=args.confidence_z,
        )
        chosen_samples = choose_kl_samples(
            sample_fidelities,
            sample_grid=sample_grid,
            dim=dim,
            n_bins=chosen_bins,
            eps=args.eps,
            target_half_width=args.target_half_width,
            n_bootstrap=args.bootstrap_trials,
            seed=args.seed,
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
        chosen_bins=int(chosen_bins),
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
        "shot_pilot_depths": list(args.shot_pilot_depth),
        "ansatzes": list(args.ansatz),
        "drift_only": bool(args.drift_only),
        "skip_bins": bool(args.skip_bins),
        "shot_grid": [] if args.drift_only else list(args.shot_grid),
        "sample_grid": [] if args.drift_only else list(args.sample_grid),
        "bin_grid": [] if args.skip_bins or args.drift_only else list(args.bin_grid),
        "pilot_samples": args.pilot_samples,
        "max_samples": args.max_samples,
        "shot_tolerance": args.shot_tolerance,
        "bin_tolerance": args.bin_tolerance,
        "target_half_width": args.target_half_width,
        "target_iteration_half_width": args.target_iteration_half_width,
        "confidence_z": args.confidence_z,
        "bootstrap_trials": args.bootstrap_trials,
        "bin_trials": args.bin_trials,
        "reference_bins": args.reference_bins,
        "eps": args.eps,
        "min_iterations": int(args.min_iterations),
        "max_iterations": int(args.max_iterations),
        "chosen_shots": int(chosen_shots),
        "chosen_n_samples": int(chosen_samples),
        "chosen_n_bins": int(chosen_bins),
        "chosen_iterations": int(chosen_iterations),
        "iteration_target_met": bool(iteration_target_met_flag),
        "iterations_run": int(chosen_iterations),
        "output_root": str(output_root),
        "methodology": {
            "bin_rule": (
                "Offline Haar draws: choose smallest n_bins whose max discretization "
                "bias vs a fine reference histogram is <= bin_tolerance."
            ),
            "shot_rule": (
                "Choose smallest shot count whose max consecutive |Delta KL_physical| "
                "is <= shot_tolerance across all ansatz/depth settings in the shot pilot."
            ),
            "sample_rule": (
                "Collect one max_samples hardware run, then choose smallest prefix "
                "length whose bootstrap KL half-width is <= target_half_width."
            ),
            "iteration_rule": (
                "Repeat frozen protocol with identical seed; choose smallest K >= "
                "min_iterations such that max iteration half-width <= "
                "target_iteration_half_width, otherwise use max_iterations."
            ),
        },
    }

    recommendation_path = write_kl_protocol_artifacts(
        output_root,
        recommendation=recommendation,
        frames={
            "bin_sensitivity": bin_sensitivity,
            "bin_sensitivity_aggregate": bin_sensitivity_aggregate,
            "shot_pilot_summary": shot_summary,
            "shot_stability": shot_stability,
            "shot_stability_aggregate": shot_stability_aggregate,
            "sample_fidelities": sample_fidelities,
            "sample_precision": sample_precision,
            "sample_precision_aggregate": sample_precision_aggregate,
            "iteration_summary": iteration_summary,
            "iteration_stability": iteration_stability,
            "iteration_precision": iteration_precision,
            "iteration_precision_aggregate": iteration_precision_aggregate,
        },
    )

    print(json.dumps(recommendation, indent=2, sort_keys=True))
    print(f"Wrote KL protocol recommendation to {recommendation_path}")


if __name__ == "__main__":
    main()
