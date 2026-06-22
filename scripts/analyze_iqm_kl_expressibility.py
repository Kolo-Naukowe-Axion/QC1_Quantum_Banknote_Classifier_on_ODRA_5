#!/usr/bin/env python3
"""Offline KL(QPU/Sim/Haar) analysis for a completed KL expressibility run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import (  # noqa: E402
    odra_ansatz as ansatz_odra,
    simulator_ansatz as ansatz_simulator,
)
from qbanknote.metrics import (  # noqa: E402
    analyze_kl_qpu_sim_haar_jobs,
    resolve_kl_run_data_dir,
    write_kl_comparison_artifacts,
)
from qbanknote.paths import ensure_importable  # noqa: E402

DEFAULT_ANSATZES = ("ansatz_odra", "ansatz_simulator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze KL(QPU||Haar), KL(Sim||Haar), and KL(QPU||Sim) offline"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed KL run directory (or parent with iteration_* subdirs).",
    )
    parser.add_argument(
        "--protocol-json",
        default=None,
        help="Optional kl_protocol_recommendation.json for metadata attachment.",
    )
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    return parser.parse_args()


def main() -> None:
    ensure_importable()
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    ansatz_fns = {
        "ansatz_odra": ansatz_odra,
        "ansatz_simulator": ansatz_simulator,
    }
    for name in args.ansatz:
        if name not in ansatz_fns:
            raise SystemExit(f"Unknown ansatz: {name}. Choose from {sorted(ansatz_fns)}")

    protocol = None
    if args.protocol_json:
        protocol_path = Path(args.protocol_json)
        if not protocol_path.is_file():
            raise SystemExit(f"Protocol file not found: {protocol_path}")
        protocol = json.loads(protocol_path.read_text())

    data_dir = resolve_kl_run_data_dir(run_dir)
    comparison_df = analyze_kl_qpu_sim_haar_jobs(
        run_dir,
        ansatz_fns=ansatz_fns,
        base_seed=args.base_seed,
        n_bins=args.n_bins,
        eps=args.eps,
    )
    csv_path = write_kl_comparison_artifacts(
        run_dir,
        comparison_df,
        protocol=protocol,
    )

    print(f"Resolved data directory: {data_dir}")
    print(f"Wrote comparison table: {csv_path}")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
