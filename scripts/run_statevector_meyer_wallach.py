#!/usr/bin/env python3
"""Run exact statevector Meyer-Wallach entanglement sweep with resumable CSV output."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import (  # noqa: E402
    odra_ansatz as ansatz_odra,
    simulator_ansatz as ansatz_simulator,
    star_ansatz as ansatz_star,
)
from qbanknote.metrics import run_statevector_mw_sweep  # noqa: E402
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402
from qbanknote.progress import make_print_callback  # noqa: E402

DEFAULT_OUTPUT_ROOT = "evaluation_and_comparison/iqm_spark/statevector_mw_outputs"
DEFAULT_DEPTHS = [2, 4, 6]
DEFAULT_ANSATZES = ("ansatz_odra", "ansatz_simulator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact statevector Meyer-Wallach entanglement sweep"
    )
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <project>/statevector_mw_outputs/<run_id>)",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_importable()
    args = parse_args()
    project_root = find_project_root(ROOT)

    ansatz_fns = {
        "ansatz_odra": ansatz_odra,
        "ansatz_simulator": ansatz_simulator,
        "ansatz_star": ansatz_star,
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
    output_dir.mkdir(parents=True, exist_ok=True)

    circuits_total = len(args.ansatz) * len(args.depth) * args.samples
    if not args.quiet:
        print(
            f"Planned run: {len(args.ansatz)} ansatze x {len(args.depth)} depths x "
            f"{args.samples} samples = {circuits_total} statevector evaluations"
        )
        print(f"Output directory: {output_dir}")

    progress_callback = None if args.quiet else make_print_callback()

    run_statevector_mw_sweep(
        ansatz_fns=ansatz_fns,
        ansatz_names=list(args.ansatz),
        depths=list(args.depth),
        n_qubits=args.num_qubits,
        n_samples=args.samples,
        seed=args.seed,
        output_dir=output_dir,
        resume=args.resume,
        verbose=not args.quiet,
        progress_callback=progress_callback,
        manifest_extra={
            "run_id": run_id,
            "source_notebook": (
                "evaluation_and_comparison/iqm_spark/meyer_wallach_comparison.ipynb"
            ),
        },
    )

    if not args.quiet:
        print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
