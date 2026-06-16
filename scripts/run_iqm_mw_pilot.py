#!/usr/bin/env python3
"""Run the Meyer-Wallach pilot protocol and write precision recommendations."""

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
    choose_mw_samples,
    choose_mw_shots,
    compute_mw_iteration_stability,
    compute_mw_sample_precision,
    compute_mw_shot_stability,
    mw_mean_shot_noise_bound,
    mw_shot_noise_sd_bound,
    read_mw_summary,
    run_iqm_mw_sweep,
    write_mw_protocol_artifacts,
)
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402
from qbanknote.progress import make_print_callback  # noqa: E402

DEFAULT_OUTPUT_ROOT = "evaluation_and_comparison/iqm_spark/iqm_mw_outputs"
DEFAULT_DEPTHS = [2, 4, 6]
DEFAULT_ANSATZES = ("ansatz_odra", "ansatz_simulator")
DEFAULT_SHOT_GRID = [512, 1024, 2048, 4096]
DEFAULT_SAMPLE_GRID = [10, 20, 40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pilot-calibrated Meyer-Wallach protocol on IQM Spark"
    )
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    parser.add_argument("--shot-grid", type=int, nargs="+", default=DEFAULT_SHOT_GRID)
    parser.add_argument("--sample-grid", type=int, nargs="+", default=DEFAULT_SAMPLE_GRID)
    parser.add_argument(
        "--pilot-samples",
        type=int,
        default=10,
        help="Samples used for the shot-stability pilot.",
    )
    parser.add_argument(
        "--shot-tolerance",
        type=float,
        default=0.02,
        help="Max allowed MW mean change between consecutive shot budgets.",
    )
    parser.add_argument(
        "--target-half-width",
        type=float,
        default=0.03,
        help="Target 95%% half-width for random-parameter MW mean estimates.",
    )
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Optional repeated fixed-seed final sweeps to estimate hardware drift.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--max-circuits-per-job", type=int, default=275)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Pilot output root (default: <project>/iqm_mw_outputs/pilots/<pilot_id>)",
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

    shot_grid = _validate_positive_ints("shot-grid", args.shot_grid)
    sample_grid = _validate_positive_ints("sample-grid", args.sample_grid)
    if args.pilot_samples <= 0:
        raise SystemExit("--pilot-samples must be positive")
    if args.iterations < 0:
        raise SystemExit("--iterations must be non-negative")

    pilot_id = args.pilot_id or datetime.now(tz=timezone.utc).strftime("mw_pilot_%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else project_root / DEFAULT_OUTPUT_ROOT / "pilots" / pilot_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Pilot output root: {output_root}")
        print(f"Shot grid: {shot_grid} with pilot samples={args.pilot_samples}")
        print(f"Sample grid: {sample_grid}")

    backend = connect_to_iqm_backend(args.iqm_url, token=args.iqm_token)
    progress_callback = None if args.quiet else make_print_callback()

    shot_frames: list[pd.DataFrame] = []
    for shots in shot_grid:
        run_dir = output_root / "shot_pilot" / f"shots_{shots}"
        if not args.quiet:
            print(f"\n[shot pilot] shots={shots} -> {run_dir}")
        run_iqm_mw_sweep(
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
        shot_frames.append(read_mw_summary(run_dir, stage="shot_pilot"))

    shot_summary = _concat(shot_frames)
    shot_stability, shot_stability_aggregate = compute_mw_shot_stability(shot_summary)
    chosen_shots = choose_mw_shots(shot_summary, tolerance=args.shot_tolerance)

    if not args.quiet:
        print(f"\nChosen shots: {chosen_shots}")

    sample_frames: list[pd.DataFrame] = []
    for n_samples in sample_grid:
        run_dir = output_root / "sample_pilot" / f"samples_{n_samples}"
        if not args.quiet:
            print(f"\n[sample pilot] n_samples={n_samples}, shots={chosen_shots} -> {run_dir}")
        run_iqm_mw_sweep(
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
        sample_frames.append(read_mw_summary(run_dir, stage="sample_pilot"))

    sample_summary = _concat(sample_frames)
    sample_precision, sample_precision_aggregate = compute_mw_sample_precision(
        sample_summary,
        target_half_width=args.target_half_width,
        z_value=args.confidence_z,
    )
    chosen_samples = choose_mw_samples(
        sample_summary,
        target_half_width=args.target_half_width,
        z_value=args.confidence_z,
    )

    if not args.quiet:
        print(f"\nChosen n_samples: {chosen_samples}")

    iteration_frames: list[pd.DataFrame] = []
    for iteration in range(1, args.iterations + 1):
        run_dir = output_root / "iteration_pilot" / f"iteration_{iteration}"
        if not args.quiet:
            print(
                f"\n[iteration pilot] iteration={iteration}, "
                f"n_samples={chosen_samples}, shots={chosen_shots} -> {run_dir}"
            )
        run_iqm_mw_sweep(
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
        iteration_frames.append(read_mw_summary(run_dir, stage="iteration_pilot", iteration=iteration))

    iteration_summary = _concat(iteration_frames)
    iteration_stability = compute_mw_iteration_stability(iteration_summary)

    recommendation = {
        "pilot_id": pilot_id,
        "iqm_url": args.iqm_url,
        "depths": list(args.depth),
        "ansatzes": list(args.ansatz),
        "shot_grid": shot_grid,
        "sample_grid": sample_grid,
        "pilot_samples": args.pilot_samples,
        "shot_tolerance": args.shot_tolerance,
        "target_half_width": args.target_half_width,
        "confidence_z": args.confidence_z,
        "chosen_shots": int(chosen_shots),
        "chosen_n_samples": int(chosen_samples),
        "recommended_iterations_minimum": 3,
        "iterations_run": int(args.iterations),
        "single_sample_shot_noise_sd_bound": mw_shot_noise_sd_bound(
            args.num_qubits,
            chosen_shots,
        ),
        "mean_shot_noise_bound_at_chosen_samples": mw_mean_shot_noise_bound(
            args.num_qubits,
            chosen_shots,
            chosen_samples,
        ),
        "output_root": str(output_root),
        "methodology": {
            "shot_rule": "Choose smallest shot count whose max consecutive MW mean change is <= shot_tolerance.",
            "sample_rule": "Choose smallest n_samples whose worst 95% half-width is <= target_half_width; otherwise use required_n_samples.",
            "iteration_rule": "Repeat frozen protocol with identical seed to estimate hardware drift separately.",
        },
    }

    recommendation_path = write_mw_protocol_artifacts(
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
        },
    )

    print(json.dumps(recommendation, indent=2, sort_keys=True))
    print(f"Wrote MW protocol recommendation to {recommendation_path}")


if __name__ == "__main__":
    main()
