#!/usr/bin/env python3
"""Plot KL vs histogram bin count: Odra vs Simulator with bootstrap 95% CI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import (  # noqa: E402
    odra_ansatz as ansatz_odra,
    simulator_ansatz as ansatz_simulator,
)
from qbanknote.metrics import (  # noqa: E402
    bootstrap_kl_interval,
    compute_kl_between_fidelity_samples,
    compute_statevector_fidelities_for_job,
    kl_depth_seed,
    list_kl_run_data_dirs,
    read_kl_fidelities,
)
from qbanknote.paths import ensure_importable  # noqa: E402

ANSATZ_FNS: dict[str, Callable[[int, int], QuantumCircuit]] = {
    "ansatz_odra": ansatz_odra,
    "ansatz_simulator": ansatz_simulator,
}
ANSATZ_LABELS = {
    "ansatz_odra": "Odra",
    "ansatz_simulator": "Simulator",
}
ANSATZ_COLORS = {
    "ansatz_odra": "#1f77b4",
    "ansatz_simulator": "#ff7f0e",
}
ITERATION_LINESTYLES = {
    1: "-",
    2: "--",
    None: "-.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep histogram bin count and plot Odra vs Simulator for "
            "KL(QPU||Haar) and KL(QPU||Sim) with percentile bootstrap CI."
        )
    )
    parser.add_argument(
        "--run-dir",
        default="evaluation_and_comparison/iqm_spark/iqm_kl_outputs/kl_hardware",
        help="KL run root (flat dir or iteration_* subdirs).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV/PNG/PDF (default: <run-dir>/analysis/bin_sweep).",
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[2])
    parser.add_argument("--ansatz", nargs="+", default=["ansatz_odra", "ansatz_simulator"])
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Optional single iteration to plot (default: all available).",
    )
    parser.add_argument("--bin-min", type=int, default=50)
    parser.add_argument("--bin-max", type=int, default=400)
    parser.add_argument("--n-bin-points", type=int, default=15)
    parser.add_argument("--bootstrap-trials", type=int, default=5000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def build_bin_grid(bin_min: int, bin_max: int, n_points: int) -> list[int]:
    if bin_min < 2:
        raise ValueError("bin_min must be at least 2")
    if bin_max < bin_min:
        raise ValueError("bin_max must be >= bin_min")
    if n_points < 2:
        raise ValueError("n_bin_points must be at least 2")
    grid = np.unique(np.round(np.linspace(bin_min, bin_max, n_points)).astype(int))
    if len(grid) < 2:
        raise ValueError("bin grid collapsed to fewer than 2 unique values")
    return [int(value) for value in grid]


def percentile_ci(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    alpha = 1.0 - float(confidence_level)
    lo, hi = np.percentile(values, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def sweep_bins_qpu_haar(
    fidelities: np.ndarray,
    *,
    dim: int,
    bin_grid: list[int],
    eps: float,
    n_bootstrap: int,
    seed: int,
    confidence_level: float,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    confidence_levels = (float(confidence_level),)
    pct = int(round(float(confidence_level) * 100))
    for n_bins in bin_grid:
        interval = bootstrap_kl_interval(
            fidelities,
            dim=dim,
            n_bins=int(n_bins),
            eps=eps,
            n_bootstrap=n_bootstrap,
            seed=seed,
            confidence_levels=confidence_levels,
        )
        rows.append(
            {
                "n_bins": int(n_bins),
                "kl_qpu_haar": float(interval["kl_physical"]),
                "kl_qpu_haar_lo": float(interval[f"bootstrap_lo_{pct}"]),
                "kl_qpu_haar_hi": float(interval[f"bootstrap_hi_{pct}"]),
            }
        )
    return rows


def sweep_bins_qpu_sim(
    f_qpu: np.ndarray,
    f_sim: np.ndarray,
    *,
    bin_grid: list[int],
    eps: float,
    n_bootstrap: int,
    seed: int,
    confidence_level: float,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    n = len(f_qpu)
    rng = np.random.default_rng(int(seed))
    for n_bins in bin_grid:
        kl_point = compute_kl_between_fidelity_samples(
            f_qpu,
            f_sim,
            n_bins=int(n_bins),
            eps=eps,
        )
        if n < 2 or n_bootstrap < 2:
            lo = hi = float(kl_point)
        else:
            kl_boot = np.empty(int(n_bootstrap), dtype=np.float64)
            for index in range(int(n_bootstrap)):
                draw = rng.integers(0, n, size=n)
                kl_boot[index] = compute_kl_between_fidelity_samples(
                    f_qpu[draw],
                    f_sim[draw],
                    n_bins=int(n_bins),
                    eps=eps,
                )
            lo, hi = percentile_ci(kl_boot, confidence_level)
        rows.append(
            {
                "n_bins": int(n_bins),
                "kl_qpu_sim": float(kl_point),
                "kl_qpu_sim_lo": float(lo),
                "kl_qpu_sim_hi": float(hi),
            }
        )
    return rows


def plot_metric_comparison(
    sweep_df,
    *,
    depth: int,
    metric_prefix: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    confidence_level: float,
    dpi: int,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import pandas as pd

    if not isinstance(sweep_df, pd.DataFrame):
        raise TypeError("sweep_df must be a pandas DataFrame")

    depth_df = sweep_df[sweep_df["depth"] == depth].copy()
    if depth_df.empty:
        raise ValueError(f"No sweep rows for depth={depth}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=dpi)
    pct = int(round(confidence_level * 100))

    for (ansatz, iteration), group in depth_df.groupby(["ansatz", "iteration"], dropna=False):
        ordered = group.sort_values("n_bins")
        x = ordered["n_bins"].to_numpy()
        y = ordered[f"{metric_prefix}"].to_numpy()
        lo = ordered[f"{metric_prefix}_lo"].to_numpy()
        hi = ordered[f"{metric_prefix}_hi"].to_numpy()
        ansatz_label = ANSATZ_LABELS.get(str(ansatz), str(ansatz))
        iter_label = f", iter {int(iteration)}" if iteration is not None else ""
        label = f"{ansatz_label}{iter_label}"
        color = ANSATZ_COLORS.get(str(ansatz), "#333333")
        linestyle = ITERATION_LINESTYLES.get(
            int(iteration) if iteration is not None else None,
            "-",
        )
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel("Number of histogram bins")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.95)
    ax.text(
        0.02,
        0.98,
        f"depth $L={depth}$, {pct}% bootstrap CI",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.25",
    )
    fig.tight_layout()

    stem = f"{metric_prefix}_odra_vs_sim_depth_{depth}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_depth_comparison(
    sweep_df,
    *,
    depth: int,
    output_dir: Path,
    confidence_level: float,
    dpi: int,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import pandas as pd

    if not isinstance(sweep_df, pd.DataFrame):
        raise TypeError("sweep_df must be a pandas DataFrame")

    depth_df = sweep_df[sweep_df["depth"] == depth].copy()
    if depth_df.empty:
        raise ValueError(f"No sweep rows for depth={depth}")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=dpi)
    pct = int(round(confidence_level * 100))
    panels = [
        (
            axes[0],
            "kl_qpu_haar",
            r"$D_{\mathrm{KL}}(P_{\mathrm{QPU}}\,\|\,P_{\mathrm{Haar}})$",
            "QPU vs Haar",
        ),
        (
            axes[1],
            "kl_qpu_sim",
            r"$D_{\mathrm{KL}}(P_{\mathrm{QPU}}\,\|\,P_{\mathrm{Sim}})$",
            "QPU vs Sim",
        ),
    ]

    for ax, metric_prefix, ylabel, panel_title in panels:
        for (ansatz, iteration), group in depth_df.groupby(["ansatz", "iteration"], dropna=False):
            ordered = group.sort_values("n_bins")
            x = ordered["n_bins"].to_numpy()
            y = ordered[f"{metric_prefix}"].to_numpy()
            lo = ordered[f"{metric_prefix}_lo"].to_numpy()
            hi = ordered[f"{metric_prefix}_hi"].to_numpy()
            ansatz_label = ANSATZ_LABELS.get(str(ansatz), str(ansatz))
            iter_label = f", iter {int(iteration)}" if iteration is not None else ""
            label = f"{ansatz_label}{iter_label}"
            color = ANSATZ_COLORS.get(str(ansatz), "#333333")
            linestyle = ITERATION_LINESTYLES.get(
                int(iteration) if iteration is not None else None,
                "-",
            )
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=5,
                color=color,
                linestyle=linestyle,
                label=label,
            )
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)

        ax.set_title(panel_title)
        ax.set_xlabel("Number of histogram bins")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", framealpha=0.95, fontsize=8)
        ax.text(
            0.02,
            0.98,
            f"$L={depth}$, {pct}% CI",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="0.25",
        )

    fig.suptitle(
        r"Odra vs Simulator: KL vs. histogram bins ($N=60$, $S=2048$)",
        fontsize=11,
    )
    fig.tight_layout()

    png_path = output_dir / f"kl_bin_sweep_odra_vs_sim_depth_{depth}.png"
    pdf_path = output_dir / f"kl_bin_sweep_odra_vs_sim_depth_{depth}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    ensure_importable()
    args = parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required") from exc
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install with: uv pip install matplotlib"
        ) from exc

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis" / "bin_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_grid = build_bin_grid(args.bin_min, args.bin_max, args.n_bin_points)
    dim = 2 ** int(args.num_qubits)
    selected_ansatze = set(args.ansatz)
    selected_depths = set(args.depths)
    confidence_level = float(args.confidence_level)
    pct = int(round(confidence_level * 100))

    rows: list[dict[str, object]] = []
    executions = list_kl_run_data_dirs(run_dir)
    for data_dir, run_label, iteration in executions:
        if args.iteration is not None and iteration != args.iteration:
            continue

        fidelities_df = read_kl_fidelities(data_dir)
        if fidelities_df.empty:
            continue
        fidelities_df = fidelities_df[fidelities_df["ansatz"].isin(selected_ansatze)]
        fidelities_df = fidelities_df[fidelities_df["depth"].isin(selected_depths)]

        for (ansatz, depth), group in fidelities_df.groupby(["ansatz", "depth"], dropna=False):
            ansatz_name = str(ansatz)
            if ansatz_name not in ANSATZ_FNS:
                continue
            ordered = group.sort_values("sample_index")
            f_qpu = ordered["fidelity_physical"].to_numpy(dtype=np.float64)
            if len(f_qpu) < 2:
                continue

            depth_seed = kl_depth_seed(int(args.base_seed), int(depth), ansatz_name)
            f_sim = compute_statevector_fidelities_for_job(
                ANSATZ_FNS[ansatz_name],
                n_qubits=int(args.num_qubits),
                depth=int(depth),
                n_samples=len(f_qpu),
                seed=depth_seed,
            )
            job_seed = int(args.base_seed) + int(depth) * 1000 + (
                1 if ansatz_name == "ansatz_simulator" else 0
            )

            haar_rows = sweep_bins_qpu_haar(
                f_qpu,
                dim=dim,
                bin_grid=bin_grid,
                eps=args.eps,
                n_bootstrap=int(args.bootstrap_trials),
                seed=job_seed,
                confidence_level=confidence_level,
            )
            sim_rows = sweep_bins_qpu_sim(
                f_qpu,
                f_sim,
                bin_grid=bin_grid,
                eps=args.eps,
                n_bootstrap=int(args.bootstrap_trials),
                seed=job_seed,
                confidence_level=confidence_level,
            )
            for haar_row, sim_row in zip(haar_rows, sim_rows, strict=True):
                rows.append(
                    {
                        "ansatz": ansatz_name,
                        "depth": int(depth),
                        "run_label": run_label,
                        "iteration": iteration,
                        "n_samples": int(len(f_qpu)),
                        "confidence_level": confidence_level,
                        **haar_row,
                        **sim_row,
                    }
                )

    if not rows:
        raise SystemExit("No sweep rows produced; check run-dir, ansatz, and depth filters.")

    sweep_df = pd.DataFrame(rows).sort_values(
        ["depth", "ansatz", "iteration", "n_bins"]
    ).reset_index(drop=True)
    csv_path = output_dir / "kl_bin_sweep.csv"
    sweep_df.to_csv(csv_path, index=False)

    manifest = {
        "run_dir": str(run_dir),
        "bin_grid": bin_grid,
        "bin_min": int(args.bin_min),
        "bin_max": int(args.bin_max),
        "n_bin_points_requested": int(args.n_bin_points),
        "bootstrap_trials": int(args.bootstrap_trials),
        "confidence_level": confidence_level,
        "iteration_filter": args.iteration,
        "depths": sorted(selected_depths),
        "ansatzes": sorted(selected_ansatze),
        "metrics": ["kl_qpu_haar", "kl_qpu_sim"],
    }
    manifest_path = output_dir / "kl_bin_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    plot_paths: list[Path] = []
    for depth in sorted(selected_depths):
        combined_png, combined_pdf = plot_depth_comparison(
            sweep_df,
            depth=int(depth),
            output_dir=output_dir,
            confidence_level=confidence_level,
            dpi=int(args.dpi),
        )
        plot_paths.extend([combined_png, combined_pdf])

        for metric_prefix, ylabel, title in [
            (
                "kl_qpu_haar",
                r"$D_{\mathrm{KL}}(P_{\mathrm{QPU}}\,\|\,P_{\mathrm{Haar}})$",
                "Odra vs Simulator: QPU vs Haar",
            ),
            (
                "kl_qpu_sim",
                r"$D_{\mathrm{KL}}(P_{\mathrm{QPU}}\,\|\,P_{\mathrm{Sim}})$",
                "Odra vs Simulator: QPU vs Sim",
            ),
        ]:
            png_path, pdf_path = plot_metric_comparison(
                sweep_df,
                depth=int(depth),
                metric_prefix=metric_prefix,
                ylabel=ylabel,
                title=title,
                output_dir=output_dir,
                confidence_level=confidence_level,
                dpi=int(args.dpi),
            )
            plot_paths.extend([png_path, pdf_path])

    print(f"Bin grid ({len(bin_grid)} values): {bin_grid}")
    print(f"Sweep table: {csv_path}")
    print(f"Manifest: {manifest_path}")
    for path in plot_paths:
        print(f"Plot: {path}")
    preview = sweep_df[
        [
            "ansatz",
            "depth",
            "iteration",
            "n_bins",
            "kl_qpu_haar",
            "kl_qpu_haar_lo",
            "kl_qpu_haar_hi",
            "kl_qpu_sim",
            "kl_qpu_sim_lo",
            "kl_qpu_sim_hi",
        ]
    ]
    print(f"\nPreview ({pct}% CI):")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
