#!/usr/bin/env python3
"""Run Meyer-Wallach entanglement sweep on IQM Spark with resumable CSV output."""

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
    DEFAULT_STAR_ANSATZES,
    star_ansatz_registry,
)
from qbanknote.iqm import connect_to_iqm_backend  # noqa: E402
from qbanknote.metrics import (  # noqa: E402
    compute_mw_iteration_precision,
    read_mw_summary,
    run_iqm_mw_sweep,
    write_mw_protocol_artifacts,
)
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402
from qbanknote.progress import make_print_callback  # noqa: E402

DEFAULT_OUTPUT_ROOT = "evaluation_and_comparison/iqm_spark/iqm_mw_outputs"
DEFAULT_DEPTHS = [2, 4, 6]
DEFAULT_ANSATZES = DEFAULT_STAR_ANSATZES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Meyer-Wallach entanglement sweep on IQM Spark")
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--shots", type=int, default=4096)
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
        default=0.01,
        help="Target 95%% half-width used when writing iteration precision artifacts.",
    )
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--max-circuits-per-job", type=int, default=275)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <project>/iqm_mw_outputs/<run_id>)",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iqm-token", default=None)
    parser.add_argument(
        "--iqm-url",
        default=os.environ.get("IQM_URL", "https://odra5.e-science.pl/").strip(),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_importable()
    args = parse_args()
    project_root = find_project_root(ROOT)

    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")

    ansatz_fns = star_ansatz_registry()
    for name in args.ansatz:
        if name not in ansatz_fns:
            raise SystemExit(f"Unknown ansatz: {name}. Choose from {sorted(ansatz_fns)}")

    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / DEFAULT_OUTPUT_ROOT / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    circuits_per_sweep = len(args.ansatz) * len(args.depth) * args.samples * 3
    if not args.quiet:
        print(
            f"Planned run: {len(args.ansatz)} ansatze x {len(args.depth)} depths x "
            f"{args.samples} samples x 3 bases x {args.iterations} iterations = "
            f"{circuits_per_sweep * args.iterations} circuits ({args.shots} shots each)"
        )
        print(f"Output directory: {output_dir}")

    backend = connect_to_iqm_backend(args.iqm_url, token=args.iqm_token)
    if not args.quiet:
        print(f"Connected to backend: {backend}")

    progress_callback = None if args.quiet else make_print_callback()

    iteration_frames = []
    for iteration in range(1, args.iterations + 1):
        iter_dir = output_dir if args.iterations == 1 else output_dir / f"iteration_{iteration}"
        if args.iterations > 1 and not args.quiet:
            print(f"\n[iteration {iteration}/{args.iterations}] -> {iter_dir}")
        run_iqm_mw_sweep(
            backend,
            ansatz_fns=ansatz_fns,
            ansatz_names=list(args.ansatz),
            depths=list(args.depth),
            n_qubits=args.num_qubits,
            n_samples=args.samples,
            seed=args.seed,
            shots=args.shots,
            optimization_level=args.optimization_level,
            seed_transpiler=None,
            max_circuits_per_job=args.max_circuits_per_job,
            output_dir=iter_dir,
            resume=args.resume,
            verbose=not args.quiet,
            progress_callback=progress_callback,
            manifest_extra={
                "run_id": run_id,
                "iqm_url": args.iqm_url,
                "iteration": iteration,
                "iterations": args.iterations,
                "source_notebook": "evaluation_and_comparison/iqm_spark/iqm_meyer_wallach.ipynb",
            },
        )
        if args.iterations > 1:
            iteration_frames.append(
                read_mw_summary(iter_dir, stage="final", iteration=iteration)
            )

    if args.iterations > 1:
        import pandas as pd

        iteration_summary = (
            pd.concat(iteration_frames, ignore_index=True) if iteration_frames else pd.DataFrame()
        )
        iteration_precision, iteration_precision_aggregate = compute_mw_iteration_precision(
            iteration_summary,
            target_half_width=args.target_iteration_half_width,
        )
        iteration_stability = iteration_precision.drop(
            columns=["target_half_width", "meets_target"], errors="ignore"
        )
        manifest = {
            "run_id": run_id,
            "iterations": args.iterations,
            "shots": args.shots,
            "n_samples": args.samples,
            "seed": args.seed,
            "target_iteration_half_width": args.target_iteration_half_width,
            "source_script": "scripts/run_iqm_meyer_wallach.py",
        }
        write_mw_protocol_artifacts(
            output_dir,
            recommendation=manifest,
            frames={
                "iteration_summary": iteration_summary,
                "iteration_stability": iteration_stability,
                "iteration_precision": iteration_precision,
                "iteration_precision_aggregate": iteration_precision_aggregate,
            },
        )
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    if not args.quiet:
        print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
