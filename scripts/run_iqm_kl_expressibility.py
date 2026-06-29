#!/usr/bin/env python3
"""Run KL expressibility sweep on IQM Spark with resumable CSV output."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import (  # noqa: E402
    odra_ansatz as ansatz_odra,
    simulator_ansatz as ansatz_simulator,
)
from qbanknote.iqm import connect_to_iqm_backend  # noqa: E402
from qbanknote.metrics import (  # noqa: E402
    compute_kl_drift_summary,
    compute_kl_iteration_precision,
    estimate_wall_time_minutes,
    protocol_by_job_to_flat_maps,
    read_kl_summary,
    run_iqm_kl_sweep,
    total_expressibility_circuits,
    write_kl_protocol_artifacts,
)
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402
from qbanknote.progress import make_print_callback  # noqa: E402

DEFAULT_OUTPUT_ROOT = "evaluation_and_comparison/iqm_spark/iqm_kl_outputs"
DEFAULT_DEPTHS = [2, 4, 6]
DEFAULT_ANSATZES = ("ansatz_odra", "ansatz_simulator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KL expressibility sweep on IQM Spark")
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--n-bins", type=int, default=150)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of repeated fixed-seed hardware sweeps (default: 1).",
    )
    parser.add_argument(
        "--target-iteration-half-width",
        type=float,
        default=0.02,
        help="Target 95%% half-width used when writing iteration precision artifacts.",
    )
    parser.add_argument(
        "--skip-iteration-precision",
        action="store_true",
        help="Skip Student-t iteration precision artifacts (hardware methodology runs).",
    )
    parser.add_argument(
        "--protocol-json",
        default=None,
        help="Optional kl_protocol_recommendation.json from a completed pilot.",
    )
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--max-circuits-per-job", type=int, default=250)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <project>/iqm_kl_outputs/<run_id>)",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hardware-retries", type=int, default=6)
    parser.add_argument("--retry-wait-seconds", type=float, default=60.0)
    parser.add_argument("--retry-max-wait-seconds", type=float, default=600.0)
    parser.add_argument("--iqm-token", default=None)
    parser.add_argument(
        "--iqm-url",
        default=os.environ.get("IQM_URL", "https://odra5.e-science.pl/").strip(),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _apply_protocol(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.protocol_json:
        args.shots_by_job = None
        args.n_samples_by_job = None
        return None
    protocol_path = Path(args.protocol_json)
    if not protocol_path.is_file():
        raise SystemExit(f"Protocol file not found: {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    scope = str(protocol.get("protocol_scope", "global"))
    args.shots_by_job = None
    args.n_samples_by_job = None
    if scope == "per_ansatz_depth":
        protocol_by_job = protocol.get("protocol_by_job")
        if not isinstance(protocol_by_job, dict):
            raise SystemExit("per_ansatz_depth protocol missing protocol_by_job")
        shots_by_job, n_samples_by_job, iterations_by_job = protocol_by_job_to_flat_maps(
            protocol_by_job
        )
        args.shots_by_job = shots_by_job
        args.n_samples_by_job = n_samples_by_job
        args.shots = int(max(shots_by_job.values()))
        args.samples = int(max(n_samples_by_job.values()))
        args.n_bins = int(protocol.get("chosen_n_bins", args.n_bins))
        args.eps = float(protocol.get("eps", args.eps))
        if protocol.get("chosen_iterations") is not None and args.iterations == 1:
            args.iterations = int(max(iterations_by_job.values()))
    else:
        args.shots = int(protocol.get("chosen_shots", args.shots))
        args.samples = int(protocol.get("chosen_n_samples", args.samples))
        args.n_bins = int(protocol.get("chosen_n_bins", args.n_bins))
        args.eps = float(protocol.get("eps", args.eps))
        if protocol.get("chosen_iterations") is not None and args.iterations == 1:
            args.iterations = int(protocol["chosen_iterations"])
    return protocol


def main() -> None:
    ensure_importable()
    args = parse_args()
    project_root = find_project_root(ROOT)
    protocol = _apply_protocol(args)

    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")

    ansatz_fns = {
        "ansatz_odra": ansatz_odra,
        "ansatz_simulator": ansatz_simulator,
    }
    for name in args.ansatz:
        if name not in ansatz_fns:
            raise SystemExit(f"Unknown ansatz: {name}. Choose from {sorted(ansatz_fns)}")

    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("kl_run_%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / DEFAULT_OUTPUT_ROOT / run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    backend = connect_to_iqm_backend(args.iqm_url, token=args.iqm_token)
    progress_callback = None if args.quiet else make_print_callback()

    iteration_frames = []
    for iteration in range(1, args.iterations + 1):
        run_dir = output_root if args.iterations == 1 else output_root / f"iteration_{iteration}"
        if not args.quiet and args.iterations > 1:
            print(f"\n[iteration {iteration}/{args.iterations}] -> {run_dir}")
        run_iqm_kl_sweep(
            backend,
            ansatz_fns=ansatz_fns,
            ansatz_names=list(args.ansatz),
            depths=list(args.depth),
            n_qubits=args.num_qubits,
            n_samples=args.samples,
            seed=args.seed,
            shots=args.shots,
            n_bins=args.n_bins,
            eps=args.eps,
            optimization_level=args.optimization_level,
            seed_transpiler=None,
            max_circuits_per_job=args.max_circuits_per_job,
            output_dir=run_dir,
            resume=args.resume,
            verbose=not args.quiet,
            progress_callback=progress_callback,
            shots_by_job=getattr(args, "shots_by_job", None),
            n_samples_by_job=getattr(args, "n_samples_by_job", None),
            manifest_extra={
                "run_id": run_id,
                "iteration": iteration,
                "protocol_scope": (
                    str(protocol.get("protocol_scope", "global")) if protocol else "global"
                ),
                "iqm_url": args.iqm_url,
                "protocol_json": str(args.protocol_json) if args.protocol_json else None,
            },
            hardware_retries=args.hardware_retries,
            retry_wait_seconds_initial=args.retry_wait_seconds,
            retry_wait_seconds_max=args.retry_max_wait_seconds,
            show_sample_progress=not args.quiet,
        )
        if args.iterations > 1:
            iteration_frames.append(read_kl_summary(run_dir, iteration=iteration))

    total_circuits = total_expressibility_circuits(
        len(args.ansatz),
        len(args.depth),
        args.samples,
        args.num_qubits,
    )
    est_minutes = estimate_wall_time_minutes(
        len(args.ansatz),
        len(args.depth),
        args.samples,
        minutes_per_state=2.0,
    )
    if not args.quiet:
        print(
            f"\nCompleted KL sweep: {total_circuits} tomography circuits, "
            f"estimated wall time ~{est_minutes:.0f} min"
        )
        print(f"Outputs: {output_root}")

    if args.iterations > 1 and iteration_frames:
        import pandas as pd

        iteration_summary = pd.concat(iteration_frames, ignore_index=True)
        if "execution_index" not in iteration_summary.columns:
            iteration_summary = iteration_summary.copy()
            iteration_summary["execution_index"] = iteration_summary["iteration"]
        drift_summary = compute_kl_drift_summary(iteration_summary)
        if not drift_summary.empty:
            drift_path = output_root / "kl_drift_summary.csv"
            drift_summary.to_csv(drift_path, index=False)
            if not args.quiet:
                print(f"Wrote drift summary: {drift_path}")

        if args.skip_iteration_precision:
            iteration_summary.to_csv(output_root / "iteration_summary.csv", index=False)
        else:
            iteration_precision, iteration_precision_aggregate = compute_kl_iteration_precision(
                iteration_summary,
                target_half_width=args.target_iteration_half_width,
            )
            write_kl_protocol_artifacts(
                output_root,
                recommendation={
                    "run_id": run_id,
                    "iterations": args.iterations,
                    "target_iteration_half_width": args.target_iteration_half_width,
                    "shots": args.shots,
                    "n_samples": args.samples,
                    "n_bins": args.n_bins,
                    "eps": args.eps,
                    "protocol_json": str(args.protocol_json) if args.protocol_json else None,
                },
                frames={
                    "iteration_summary": iteration_summary,
                    "iteration_precision": iteration_precision,
                    "iteration_precision_aggregate": iteration_precision_aggregate,
                },
            )


if __name__ == "__main__":
    main()
