#!/usr/bin/env python3
"""Run Meyer-Wallach entanglement sweep on IQM Spark with resumable CSV output."""

from __future__ import annotations

import argparse
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
from qbanknote.metrics import run_iqm_mw_sweep  # noqa: E402
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402
from qbanknote.progress import make_print_callback  # noqa: E402

DEFAULT_OUTPUT_ROOT = "evaluation_and_comparison/iqm_spark/iqm_mw_outputs"
DEFAULT_DEPTHS = [2, 4, 6]
DEFAULT_ANSATZES = ("ansatz_odra", "ansatz_simulator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Meyer-Wallach entanglement sweep on IQM Spark")
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
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

    ansatz_fns = {
        "ansatz_odra": ansatz_odra,
        "ansatz_simulator": ansatz_simulator,
    }
    for name in args.ansatz:
        if name not in ansatz_fns:
            raise SystemExit(f"Unknown ansatz: {name}. Choose from {sorted(ansatz_fns)}")

    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / DEFAULT_OUTPUT_ROOT / run_id
    )

    if not args.quiet:
        total_circuits = len(args.ansatz) * len(args.depth) * args.samples * 3
        print(
            f"Planned run: {len(args.ansatz)} ansatze x {len(args.depth)} depths x "
            f"{args.samples} samples x 3 bases = {total_circuits} circuits "
            f"({args.shots} shots each)"
        )
        print(f"Output directory: {output_dir}")

    backend = connect_to_iqm_backend(args.iqm_url, token=args.iqm_token)
    if not args.quiet:
        print(f"Connected to backend: {backend}")

    progress_callback = None if args.quiet else make_print_callback()

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
        output_dir=output_dir,
        resume=args.resume,
        verbose=not args.quiet,
        progress_callback=progress_callback,
        manifest_extra={
            "run_id": run_id,
            "iqm_url": args.iqm_url,
            "source_notebook": "evaluation_and_comparison/iqm_spark/iqm_meyer_wallach.ipynb",
        },
    )

    if not args.quiet:
        print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
