#!/usr/bin/env python3
"""Offline hardware KL analysis: bootstrap CI, drift, and sim/Haar comparison."""

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
    compute_kl_bootstrap_qpu_sim_uncertainty,
    compute_kl_bootstrap_uncertainty,
    compute_kl_drift_summary,
    list_kl_hardware_executions,
    list_kl_run_data_dirs,
    read_kl_fidelities,
    read_kl_summary,
    write_kl_comparison_artifacts,
    write_kl_hardware_report,
)
from qbanknote.paths import ensure_importable  # noqa: E402

DEFAULT_ANSATZES = ("ansatz_odra", "ansatz_simulator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze KL hardware runs: percentile bootstrap CI, drift, and "
            "KL(QPU/Sim/Haar) comparison"
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Primary KL run directory (flat or iteration_* subdirs).",
    )
    parser.add_argument(
        "--compare-run-dir",
        default=None,
        help="Optional second run for day-to-day drift (separate run-id).",
    )
    parser.add_argument(
        "--protocol-json",
        default=None,
        help="Optional kl_hardware_protocol.json for report metadata.",
    )
    parser.add_argument("--bootstrap-trials", type=int, default=5000)
    parser.add_argument(
        "--confidence-levels",
        type=float,
        nargs="+",
        default=[0.90, 0.95],
    )
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    return parser.parse_args()


def main() -> None:
    ensure_importable()
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    compare_run_dir = Path(args.compare_run_dir) if args.compare_run_dir else None
    if compare_run_dir is not None and not compare_run_dir.is_dir():
        raise SystemExit(f"Compare run directory not found: {compare_run_dir}")

    all_ansatz_fns = {
        "ansatz_odra": ansatz_odra,
        "ansatz_simulator": ansatz_simulator,
    }
    for name in args.ansatz:
        if name not in all_ansatz_fns:
            raise SystemExit(f"Unknown ansatz: {name}. Choose from {sorted(all_ansatz_fns)}")
    ansatz_fns = {name: all_ansatz_fns[name] for name in args.ansatz}

    protocol = None
    if args.protocol_json:
        protocol_path = Path(args.protocol_json)
        if not protocol_path.is_file():
            raise SystemExit(f"Protocol file not found: {protocol_path}")
        protocol = json.loads(protocol_path.read_text())

    dim = 2 ** int(args.num_qubits)
    confidence_levels = tuple(float(level) for level in args.confidence_levels)
    n_bootstrap = int(args.bootstrap_trials)
    bootstrap_seed = int(args.base_seed if args.base_seed is not None else (protocol or {}).get("seed", 42))
    selected_ansatze = set(args.ansatz)

    executions = list_kl_hardware_executions(
        run_dir,
        compare_run_dir=compare_run_dir,
    )
    primary_executions = list_kl_run_data_dirs(run_dir)
    execution_meta: list[dict[str, object]] = []
    bootstrap_frames = []
    qpu_sim_boot_frames = []
    summary_frames = []

    for execution_index, (data_dir, run_label, iteration) in enumerate(executions):
        fidelities_df = read_kl_fidelities(data_dir)
        if fidelities_df.empty:
            raise SystemExit(f"No fidelity rows in {data_dir}")
        fidelities_df = fidelities_df[fidelities_df["ansatz"].isin(selected_ansatze)]
        if fidelities_df.empty:
            raise SystemExit(
                f"No fidelity rows for selected ansatze {sorted(selected_ansatze)} in {data_dir}"
            )

        summary_df = read_kl_summary(
            data_dir,
            iteration=iteration,
        )
        if summary_df.empty:
            raise SystemExit(f"No summary rows in {data_dir / 'iqm_kl_results.csv'}")
        summary_df = summary_df[summary_df["ansatz"].isin(selected_ansatze)]
        if summary_df.empty:
            raise SystemExit(
                f"No summary rows for selected ansatze {sorted(selected_ansatze)} in {data_dir}"
            )

        summary_df = summary_df.copy()
        summary_df["execution_index"] = execution_index
        if "run_label" not in summary_df.columns:
            summary_df["run_label"] = run_label
        summary_frames.append(summary_df)

        default_n_bins = args.n_bins if args.n_bins is not None else 400
        if args.n_bins is None and not summary_df.empty:
            default_n_bins = int(summary_df.iloc[0]["n_bins"])
        if args.n_bins is not None:
            summary_df["n_bins"] = int(args.n_bins)

        bootstrap_frames.append(
            compute_kl_bootstrap_uncertainty(
                fidelities_df,
                dim=dim,
                n_bins=default_n_bins,
                eps=args.eps,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
                confidence_levels=confidence_levels,
                run_label=run_label,
                iteration=iteration,
                data_dir=str(data_dir),
                summary_df=summary_df,
            )
        )
        qpu_sim_boot_frames.append(
            compute_kl_bootstrap_qpu_sim_uncertainty(
                fidelities_df,
                ansatz_fns=ansatz_fns,
                n_bins=default_n_bins,
                eps=args.eps,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
                base_seed=bootstrap_seed,
                confidence_levels=confidence_levels,
                run_label=run_label,
                iteration=iteration,
                data_dir=str(data_dir),
            )
        )
        execution_meta.append(
            {
                "data_dir": str(data_dir),
                "run_label": run_label,
                "iteration": iteration,
            }
        )

    import pandas as pd

    bootstrap_df = (
        pd.concat(bootstrap_frames, ignore_index=True)
        if bootstrap_frames
        else pd.DataFrame()
    )
    qpu_sim_boot_df = (
        pd.concat(qpu_sim_boot_frames, ignore_index=True)
        if qpu_sim_boot_frames
        else pd.DataFrame()
    )
    if not bootstrap_df.empty and not qpu_sim_boot_df.empty:
        merge_keys = ["ansatz", "depth", "run_label", "iteration", "data_dir"]
        merge_keys = [key for key in merge_keys if key in bootstrap_df.columns and key in qpu_sim_boot_df.columns]
        qpu_sim_cols = [
            col
            for col in qpu_sim_boot_df.columns
            if col.startswith("kl_qpu_sim") or col.startswith("bootstrap_qpu_sim")
        ]
        bootstrap_df = bootstrap_df.merge(
            qpu_sim_boot_df[merge_keys + qpu_sim_cols],
            on=merge_keys,
            how="left",
        )
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_path = analysis_dir / "kl_bootstrap_uncertainty.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)

    iteration_summary = (
        pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    )
    if len(primary_executions) >= 2:
        primary_dirs = {str(path) for path, _, _ in primary_executions}
        drift_source = iteration_summary[iteration_summary["run_dir"].isin(primary_dirs)]
    elif compare_run_dir is not None and len(primary_executions) == 1:
        drift_source = iteration_summary
    else:
        drift_source = iteration_summary.iloc[0:0]
    drift_df = compute_kl_drift_summary(drift_source)
    drift_path = analysis_dir / "kl_drift_summary.csv"
    drift_df.to_csv(drift_path, index=False)

    comparison_df = analyze_kl_qpu_sim_haar_jobs(
        run_dir,
        ansatz_fns=ansatz_fns,
        base_seed=args.base_seed,
        n_bins=args.n_bins,
        eps=args.eps,
    )
    comparison_path = write_kl_comparison_artifacts(
        run_dir,
        comparison_df,
        protocol=protocol,
    )

    report_path = write_kl_hardware_report(
        run_dir,
        bootstrap_df=bootstrap_df,
        drift_df=drift_df,
        comparison_df=comparison_df,
        protocol=protocol,
        executions=execution_meta,
    )

    print(f"Executions analyzed: {len(executions)}")
    print(f"Bootstrap uncertainty: {bootstrap_path}")
    print(f"Drift summary: {drift_path}")
    print(f"QPU/Sim/Haar comparison: {comparison_path}")
    print(f"Hardware report: {report_path}")
    if not bootstrap_df.empty:
        print("\nBootstrap uncertainty (head):")
        print(bootstrap_df.to_string(index=False))
    if not drift_df.empty:
        print("\nDrift summary:")
        print(drift_df.to_string(index=False))


if __name__ == "__main__":
    main()
