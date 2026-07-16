#!/usr/bin/env python3
"""Compare ansatz state fidelity fully offline (no QPU).

Uses exact Aer density-matrix simulation under a Spark-like noise model and the
same random Uniform[0, 2*pi] parameter sampling as MW/KL/fidelity.

Methodology:
  * native IQM Spark basis (r, cz),
  * star coupling map so ring ansatze pay their SWAP routing cost,
  * gate errors given as average gate error (1 - F_gate), converted to the
    correct depolarizing probability internally,
  * optional T1/T2 thermal relaxation,
  * optional two-qubit error sweep to check ranking stability.
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
            "Offline Aer fidelity comparison for ansatze (no IQM / no QPU). "
            "Computes F = <psi_ideal|rho_noisy|psi_ideal> with random weights."
        )
    )
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--err1",
        type=float,
        default=DEFAULT_ERR1,
        help="Average 1-qubit gate error (1 - F_1q). Default from Spark >=99.9%%.",
    )
    parser.add_argument(
        "--err2",
        type=float,
        default=DEFAULT_ERR2,
        help="Average 2-qubit (CZ) gate error (1 - F_cz). Default from Spark >=99%%.",
    )
    parser.add_argument(
        "--err2-sweep",
        type=float,
        nargs="+",
        default=None,
        help="Sweep several CZ errors to check ranking stability (overrides --err2).",
    )
    parser.add_argument(
        "--noiseless",
        action="store_true",
        help="Sanity check: skip noise (fidelity should be ~1 for all ansatze).",
    )
    parser.add_argument(
        "--basis-gates",
        nargs="+",
        default=list(DEFAULT_BASIS_GATES),
        help="Native transpile basis (default IQM Spark: r cz).",
    )
    parser.add_argument(
        "--topology",
        choices=["star", "none"],
        default="star",
        help="Coupling map for routing cost. 'star' = Spark; 'none' = no routing.",
    )
    parser.add_argument("--star-center", type=int, default=0)
    parser.add_argument(
        "--thermal",
        action="store_true",
        help="Add T1/T2 thermal relaxation on top of depolarizing noise.",
    )
    parser.add_argument("--t1", type=float, default=DEFAULT_T1, help="T1 (seconds).")
    parser.add_argument("--t2", type=float, default=DEFAULT_T2, help="T2 (seconds).")
    parser.add_argument("--gate1-s", type=float, default=DEFAULT_GATE1_S)
    parser.add_argument("--gate2-s", type=float, default=DEFAULT_GATE2_S)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
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
        f"Ansatzes={list(args.ansatz)} depths={list(args.depth)} "
        f"n_samples={args.n_samples} noiseless={args.noiseless}"
    )
    print(f"Basis gates: {list(args.basis_gates)} | topology: {args.topology}")
    if not args.noiseless:
        if args.err2_sweep:
            print(f"Noise: err1={args.err1}, err2 sweep={args.err2_sweep}")
        else:
            print(f"Noise: err1={args.err1}, err2={args.err2}")
        if args.thermal:
            print(f"Thermal: T1={args.t1}s T2={args.t2}s g1={args.gate1_s}s g2={args.gate2_s}s")

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
        output_dir=output_root,
    )

    print("\nSummary (higher fidelity = better under this noise model):")
    print(summary.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(json.dumps({"run_id": run_id, "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
