#!/usr/bin/env python3
"""Offline Aer fidelity aligned with the QPU tomography protocol.

Default mode (simulator-vs-QPU gap, same as current QPU fidelity pilot):
  1. Full 3^n Pauli tomography on Aer (same reconstruction as IQM),
  2. Logical noiseless statevector as ideal reference (same as QPU),
  3. Random ansatz parameters theta ~ Uniform[0, 2*pi] (same as MW/KL/QPU fidelity).

No QPU / IQM token required. Use --weights trained for task-state fidelity
(CV checkpoints + test inputs), or --protocol exact for fast screening.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import (  # noqa: E402
    odra_ansatz as ansatz_odra,
    odra_star_ansatz as ansatz_odra_star,
    simulator_ansatz as ansatz_simulator,
)
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402
from qbanknote.simulator_fidelity import (  # noqa: E402
    DEFAULT_BASIS_GATES,
    DEFAULT_ERR1,
    DEFAULT_ERR2,
    DEFAULT_GATE1_S,
    DEFAULT_GATE2_S,
    DEFAULT_SHOTS,
    DEFAULT_T1,
    DEFAULT_T2,
    run_simulator_fidelity_sweep,
)

DEFAULT_OUTPUT_ROOT = (
    "evaluation_and_comparison/iqm_spark/iqm_fidelity_outputs/simulator"
)
DEFAULT_ANSATZES = ("ansatz_odra", "ansatz_simulator", "ansatz_odra_star")
DEFAULT_DEPTHS = [2, 4, 6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Aer fidelity (no QPU). Default = tomography + random weights "
            "+ logical ideal, matching the current IQM fidelity pilot."
        )
    )
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocol",
        choices=["tomography", "exact"],
        default="tomography",
        help="tomography = QPU-matched 3^n shot tomography; exact = fast density-matrix.",
    )
    parser.add_argument(
        "--weights",
        choices=["trained", "random"],
        default="random",
        help="random = Uniform[0,2pi] ansatz only (default, same as QPU fidelity); "
        "trained = CV checkpoints + test inputs.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help="Shots per Pauli basis for --protocol tomography (default 1024).",
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--err1", type=float, default=DEFAULT_ERR1)
    parser.add_argument("--err2", type=float, default=DEFAULT_ERR2)
    parser.add_argument("--err2-sweep", type=float, nargs="+", default=None)
    parser.add_argument("--noiseless", action="store_true")
    parser.add_argument("--basis-gates", nargs="+", default=list(DEFAULT_BASIS_GATES))
    parser.add_argument("--topology", choices=["star", "none"], default="star")
    parser.add_argument("--star-center", type=int, default=0)
    parser.add_argument(
        "--thermal",
        action="store_true",
        help="Add T1/T2 on top of depolarizing (may double-count datasheet RB errors).",
    )
    parser.add_argument("--t1", type=float, default=DEFAULT_T1)
    parser.add_argument("--t2", type=float, default=DEFAULT_T2)
    parser.add_argument("--gate1-s", type=float, default=DEFAULT_GATE1_S)
    parser.add_argument("--gate2-s", type=float, default=DEFAULT_GATE2_S)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--max-circuits-per-job", type=int, default=275)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_importable()
    args = parse_args()
    project_root = find_project_root(ROOT)

    ansatz_fns = {
        "ansatz_odra": ansatz_odra,
        "ansatz_simulator": ansatz_simulator,
        "ansatz_odra_star": ansatz_odra_star,
    }
    for name in args.ansatz:
        if name not in ansatz_fns:
            raise SystemExit(f"Unknown ansatz: {name}. Choose from {sorted(ansatz_fns)}")
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be positive")
    if args.protocol == "tomography" and args.shots <= 0:
        raise SystemExit("--shots must be positive for tomography")
    if args.thermal and not args.noiseless and not args.quiet:
        print(
            "Warning: --thermal with datasheet err1/err2 can double-count decoherence.",
            flush=True,
        )

    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime(
        "sim_fidelity_%Y%m%d_%H%M%S"
    )
    output_root = (
        Path(args.output_root)
        if args.output_root
        else project_root / DEFAULT_OUTPUT_ROOT / run_id
    )

    print(f"Output: {output_root}")
    print(
        f"protocol={args.protocol} weights={args.weights} "
        f"ansatzes={list(args.ansatz)} depths={list(args.depth)} "
        f"n_samples={args.n_samples} noiseless={args.noiseless}"
    )
    if args.protocol == "tomography":
        print(f"Tomography shots/basis: {args.shots} (3^{args.num_qubits} bases)")
    print(f"Basis gates: {list(args.basis_gates)} | topology: {args.topology}")
    if not args.noiseless:
        if args.err2_sweep:
            print(f"Noise: err1={args.err1}, err2 sweep={args.err2_sweep}")
        else:
            print(f"Noise: err1={args.err1}, err2={args.err2}")

    _, summary, summary_path = run_simulator_fidelity_sweep(
        ansatz_fns=ansatz_fns,
        ansatz_names=list(args.ansatz),
        depths=list(args.depth),
        n_qubits=args.num_qubits,
        n_samples=args.n_samples,
        seed=args.seed,
        err1=args.err1,
        err2=args.err2,
        err2_grid=args.err2_sweep,
        noiseless=args.noiseless,
        basis_gates=list(args.basis_gates),
        topology=args.topology,
        star_center=args.star_center,
        thermal=args.thermal,
        t1=args.t1,
        t2=args.t2,
        gate1_s=args.gate1_s,
        gate2_s=args.gate2_s,
        optimization_level=args.optimization_level,
        protocol=args.protocol,
        weights=args.weights,
        shots=args.shots,
        fold=args.fold,
        epoch=args.epoch,
        max_circuits_per_job=args.max_circuits_per_job,
        root=project_root,
        output_dir=output_root,
        verbose=not args.quiet,
    )

    print("\nSummary (headline = fidelity_physical mean):")
    print(summary.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output_root": str(output_root),
                "protocol": args.protocol,
                "weights": args.weights,
                "requires_qpu": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
