#!/usr/bin/env python3
"""Run state-tomography fidelity sweep on IQM Spark (no trained weights).

Uses random ansatz parameters theta ~ Uniform[0, 2*pi] — same rule as MW/KL.
Default budget matches the existing pilot recommendations (shots=1024,
n_samples=10, iterations=2).
"""

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
from qbanknote.metrics import run_iqm_fidelity_sweep  # noqa: E402
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402
from qbanknote.progress import make_print_callback  # noqa: E402

DEFAULT_OUTPUT_ROOT = "evaluation_and_comparison/iqm_spark/iqm_fidelity_outputs"
DEFAULT_DEPTHS = [2, 4, 6]
DEFAULT_ANSATZES = ("ansatz_odra", "ansatz_simulator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run state-tomography fidelity on IQM Spark (random thetas, no weights)"
    )
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--max-circuits-per-job", type=int, default=250)
    parser.add_argument(
        "--protocol-json",
        default=None,
        help="Optional fidelity_protocol_recommendation.json (overrides shots/samples/iterations).",
    )
    parser.add_argument("--output-dir", default=None)
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
        return None
    path = Path(args.protocol_json)
    if not path.is_file():
        raise SystemExit(f"Protocol file not found: {path}")
    protocol = json.loads(path.read_text())
    args.shots = int(protocol.get("chosen_shots", args.shots))
    args.samples = int(protocol.get("chosen_n_samples", args.samples))
    if protocol.get("chosen_iterations") is not None and args.iterations == 2:
        args.iterations = int(protocol["chosen_iterations"])
    if protocol.get("depths"):
        # Keep CLI depths if user overrode; only fill when still default-ish.
        pass
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

    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("fidelity_run_%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / DEFAULT_OUTPUT_ROOT / run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # One tomography state per sample (~half a KL pair).
    n_states = len(args.ansatz) * len(args.depth) * args.samples * args.iterations
    if not args.quiet:
        print(f"Run ID:     {run_id}")
        print(f"Output:     {output_root}")
        print(f"Ansatzes:   {args.ansatz}")
        print(f"Depths:     {args.depth}")
        print(f"Samples:    {args.samples}")
        print(f"Shots:      {args.shots}")
        print(f"Iterations: {args.iterations}")
        print(f"Est. states:{n_states}  (~{n_states * 3.5 / 60:.1f} h @ 3.5 min/state)")
        print("Note:       random thetas — no trained weights required")
        print()

    backend = connect_to_iqm_backend(args.iqm_url, token=args.iqm_token)
    progress_callback = None if args.quiet else make_print_callback()

    for iteration in range(1, args.iterations + 1):
        run_dir = output_root if args.iterations == 1 else output_root / f"iteration_{iteration}"
        if not args.quiet and args.iterations > 1:
            print(f"\n[iteration {iteration}/{args.iterations}] -> {run_dir}")
        run_iqm_fidelity_sweep(
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
            output_dir=run_dir,
            resume=args.resume,
            verbose=not args.quiet,
            progress_callback=progress_callback,
            hardware_retries=args.hardware_retries,
            retry_wait_seconds_initial=args.retry_wait_seconds,
            retry_wait_seconds_max=args.retry_max_wait_seconds,
            manifest_extra={
                "run_id": run_id,
                "iteration": iteration,
                "iqm_url": args.iqm_url,
                "protocol_json": str(args.protocol_json) if args.protocol_json else None,
                "weight_sampling": "uniform_random_0_2pi",
                "hardware_retries": args.hardware_retries,
            },
        )

    if not args.quiet:
        print(f"\nDone. Outputs under {output_root}")


if __name__ == "__main__":
    main()
