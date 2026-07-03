#!/usr/bin/env python3
"""Three presentation figures for KL(QPU || Haar) hardware results (no drift emphasis)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.paths import ensure_importable  # noqa: E402

ANSATZ_LABELS = {
    "ansatz_odra": "Odra",
    "ansatz_simulator": "Simulator",
}
ANSATZ_COLORS = {
    "ansatz_odra": "#212531",
    "ansatz_simulator": "#C4302B",
}
ITER_COLORS = {1: "#212531", 2: "#C4302B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KL(QPU||Haar) presentation figures.")
    parser.add_argument(
        "--run-dir",
        default="evaluation_and_comparison/iqm_spark/iqm_kl_outputs/kl_hardware",
    )
    parser.add_argument(
        "--bootstrap-csv",
        default=None,
        help="Defaults to <run-dir>/analysis/kl_bootstrap_uncertainty.csv",
    )
    parser.add_argument(
        "--bin-sweep-csv",
        default=None,
        help="Defaults to <run-dir>/analysis/bin_sweep/kl_bin_sweep.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <run-dir>/analysis/presentation",
    )
    parser.add_argument("--n-bins", type=int, default=75)
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_bootstrap_table(path: Path, *, n_bins: int, min_samples: int):
    import pandas as pd

    df = pd.read_csv(path)
    df = df[df["n_bins"] == int(n_bins)].copy()
    df = df[df["n_samples"] >= int(min_samples)].copy()
    if df.empty:
        raise ValueError(f"No complete rows at n_bins={n_bins} in {path}")
    df["ansatz_label"] = df["ansatz"].map(ANSATZ_LABELS).fillna(df["ansatz"])
    return df.sort_values(["depth", "ansatz", "iteration"]).reset_index(drop=True)


def aggregate_mean_envelope(group_df):
    """Mean point estimate; CI envelope = min(lo), max(hi) across rows."""
    return {
        "kl": float(group_df["kl_physical"].mean()),
        "lo": float(group_df["bootstrap_lo_95"].min()),
        "hi": float(group_df["bootstrap_hi_95"].max()),
        "n_executions": int(len(group_df)),
    }


def save_figure(fig, output_dir: Path, stem: str) -> tuple[Path, Path]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def plot_v1_executions_with_ci(df, *, output_dir: Path, n_bins: int, dpi: int) -> tuple[Path, Path]:
    """Grouped bars: both iterations, Odra vs Sim, 95% bootstrap CI."""
    import matplotlib.pyplot as plt

    depths = sorted(int(v) for v in df["depth"].unique())
    fig, axes = plt.subplots(1, len(depths), figsize=(5.5 * len(depths), 4.8), dpi=dpi, sharey=True)
    if len(depths) == 1:
        axes = [axes]

    width = 0.18
    for ax, depth in zip(axes, depths, strict=True):
        subset = df[df["depth"] == depth]
        x_base = np.arange(2)
        for offset, iteration in enumerate(sorted(subset["iteration"].unique())):
            iter_df = subset[subset["iteration"] == iteration]
            odra = iter_df[iter_df["ansatz"] == "ansatz_odra"]
            sim = iter_df[iter_df["ansatz"] == "ansatz_simulator"]
            positions = x_base + (offset - 0.5) * width * 2.2
            for pos, row, ansatz in [
                (positions[0], odra.iloc[0] if not odra.empty else None, "ansatz_odra"),
                (positions[1], sim.iloc[0] if not sim.empty else None, "ansatz_simulator"),
            ]:
                if row is None:
                    continue
                color = ITER_COLORS.get(int(iteration), "#444444")
                ax.bar(
                    pos,
                    row["kl_physical"],
                    width=width,
                    color=color,
                    alpha=0.85 if ansatz == "ansatz_odra" else 0.65,
                    edgecolor="white",
                    linewidth=0.8,
                    label=f"Iteration {int(iteration)}" if pos == positions[0] else "",
                )
                lo_err = float(row["kl_physical"] - row["bootstrap_lo_95"])
                hi_err = float(row["bootstrap_hi_95"] - row["kl_physical"])
                ax.errorbar(
                    pos,
                    row["kl_physical"],
                    yerr=np.array([[lo_err], [hi_err]]),
                    fmt="none",
                    ecolor="0.15",
                    capsize=4,
                    linewidth=1.2,
                )

        ax.set_xticks(x_base)
        ax.set_xticklabels(["Odra", "Simulator"])
        ax.set_title(f"Depth $L={depth}$")
        ax.set_xlabel("Ansatz")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    axes[0].set_ylabel(r"$D_{\mathrm{KL}}(P_{\mathrm{QPU}}\,\|\,P_{\mathrm{Haar}})$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, framealpha=0.95, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Version 1 — both executions with 95% bootstrap CI ($B={n_bins}$, $N=60$)",
        fontsize=11,
        y=1.08,
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "v1_executions_with_ci")


def plot_v2_headline_odra_vs_sim(df, *, output_dir: Path, n_bins: int, dpi: int) -> tuple[Path, Path]:
    """Clean headline: Odra vs Simulator, mean KL across complete executions."""
    import matplotlib.pyplot as plt

    depths = sorted(int(v) for v in df["depth"].unique())
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=dpi)

    x = np.arange(len(depths))
    width = 0.34
    summary_rows = []
    for idx, ansatz in enumerate(["ansatz_odra", "ansatz_simulator"]):
        means = []
        yerr_lo = []
        yerr_hi = []
        for depth in depths:
            group = df[(df["depth"] == depth) & (df["ansatz"] == ansatz)]
            if group.empty:
                means.append(np.nan)
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
                continue
            agg = aggregate_mean_envelope(group)
            summary_rows.append({"depth": depth, "ansatz": ansatz, **agg})
            means.append(agg["kl"])
            yerr_lo.append(agg["kl"] - agg["lo"])
            yerr_hi.append(agg["hi"] - agg["kl"])

        offset = -width / 2 if ansatz == "ansatz_odra" else width / 2
        ax.bar(
            x + offset,
            means,
            width=width,
            color=ANSATZ_COLORS[ansatz],
            label=ANSATZ_LABELS[ansatz],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor="0.15",
            capsize=5,
            linewidth=1.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"$L={depth}$" for depth in depths])
    ax.set_ylabel(r"$D_{\mathrm{KL}}(P_{\mathrm{QPU}}\,\|\,P_{\mathrm{Haar}})$")
    ax.set_xlabel("Circuit depth")
    ax.set_title(
        f"Version 2 — headline comparison (mean over complete executions; CI envelope)\n"
        f"$B={n_bins}$, $N=60$"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    return save_figure(fig, output_dir, "v2_headline_odra_vs_sim")


def plot_v3_interval_overview(df, *, output_dir: Path, n_bins: int, dpi: int) -> tuple[Path, Path]:
    """Horizontal interval plot: each execution as its own CI segment."""
    import matplotlib.pyplot as plt

    ordered = df.sort_values(["depth", "ansatz", "iteration"]).reset_index(drop=True)
    labels = [
        f"{row.ansatz_label} · $L={int(row.depth)}$ · iter {int(row.iteration)}"
        for row in ordered.itertuples(index=False)
    ]
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.55 * len(ordered) + 1.5)), dpi=dpi)
    for yi, row in zip(y, ordered.itertuples(index=False), strict=True):
        color = ANSATZ_COLORS.get(str(row.ansatz), "#333333")
        ax.plot(
            [row.bootstrap_lo_95, row.bootstrap_hi_95],
            [yi, yi],
            color=color,
            linewidth=3.0,
            alpha=0.35,
            solid_capstyle="round",
        )
        ax.scatter(row.kl_physical, yi, color=color, s=55, zorder=3, edgecolors="white", linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$D_{\mathrm{KL}}(P_{\mathrm{QPU}}\,\|\,P_{\mathrm{Haar}})$")
    ax.set_title(f"Version 3 — interval overview (95% bootstrap CI, $B={n_bins}$)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    return save_figure(fig, output_dir, "v3_interval_overview")


def main() -> None:
    ensure_importable()
    args = parse_args()

    try:
        import matplotlib.pyplot  # noqa: F401
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Requires pandas and matplotlib.") from exc

    run_dir = Path(args.run_dir)
    bootstrap_csv = (
        Path(args.bootstrap_csv)
        if args.bootstrap_csv
        else run_dir / "analysis" / "kl_bootstrap_uncertainty.csv"
    )
    if not bootstrap_csv.is_file():
        raise SystemExit(
            f"Missing {bootstrap_csv}. Run analyze_iqm_kl_hardware.py first "
            f"(e.g. --n-bins {args.n_bins})."
        )

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis" / "presentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_bootstrap_table(bootstrap_csv, n_bins=args.n_bins, min_samples=args.min_samples)
    manifest = {
        "metric": "kl_qpu_haar",
        "n_bins": int(args.n_bins),
        "min_samples": int(args.min_samples),
        "rows_used": df.to_dict(orient="records"),
        "figures": [],
    }

    plots = [
        plot_v1_executions_with_ci(df, output_dir=output_dir, n_bins=args.n_bins, dpi=args.dpi),
        plot_v2_headline_odra_vs_sim(df, output_dir=output_dir, n_bins=args.n_bins, dpi=args.dpi),
        plot_v3_interval_overview(df, output_dir=output_dir, n_bins=args.n_bins, dpi=args.dpi),
    ]
    stems = ["v1_executions_with_ci", "v2_headline_odra_vs_sim", "v3_interval_overview"]
    for stem, (png, pdf) in zip(stems, plots, strict=True):
        manifest["figures"].append({"stem": stem, "png": str(png), "pdf": str(pdf)})

    summary_path = output_dir / "presentation_summary.csv"
    summary_rows = []
    for (ansatz, depth), group in df.groupby(["ansatz", "depth"]):
        agg = aggregate_mean_envelope(group)
        summary_rows.append(
            {
                "ansatz": ansatz,
                "ansatz_label": ANSATZ_LABELS.get(str(ansatz), str(ansatz)),
                "depth": int(depth),
                "n_bins": int(args.n_bins),
                **agg,
            }
        )
    pd.DataFrame(summary_rows).sort_values(["depth", "ansatz"]).to_csv(summary_path, index=False)

    manifest_path = output_dir / "presentation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Rows used ({len(df)} execution summaries @ B={args.n_bins}):")
    print(
        df[["ansatz", "depth", "iteration", "kl_physical", "bootstrap_lo_95", "bootstrap_hi_95"]]
        .to_string(index=False)
    )
    print(f"\nSummary table: {summary_path}")
    print(f"Manifest: {manifest_path}")
    for _, (png, pdf) in zip(stems, plots, strict=True):
        print(f"Plot: {png}")
        print(f"Plot: {pdf}")


if __name__ == "__main__":
    main()
