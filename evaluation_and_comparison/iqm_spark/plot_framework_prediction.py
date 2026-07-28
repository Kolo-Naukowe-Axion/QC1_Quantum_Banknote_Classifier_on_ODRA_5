#!/usr/bin/env python3
"""Wide single-panel bar chart: all metrics overlaid by depth.

X-axis: depth 2, depth 4, depth 6.
Within each depth group, all metrics are shown as bars.
Palette is consistent with earlier figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_metric_framework_variants import load_framework_df

ROOT = Path(__file__).resolve().parents[2]
SPARK = ROOT / "evaluation_and_comparison/iqm_spark"

C_ACC = "#c0392b"     # Accuracy
C_F1 = "#7b241c"      # F1
C_FID = "#1f6fb2"     # Fidelity
C_MW = "#e08a1e"      # MW
C_KL = "#2e8b57"      # KL

depths = [2, 4, 6]

def main() -> None:
    df = load_framework_df()

    sim = df[df["ansatz"] == "ansatz_simulator"].sort_values("depth")
    odra = df[df["ansatz"] == "ansatz_odra"].sort_values("depth")
    fig, ax = plt.subplots(figsize=(16.8, 5.2), facecolor="white")

    bar_w = 0.09
    metric_gap = 0.04
    depth_gap = 0.42
    pair_step = 2 * bar_w + metric_gap
    depth_width = 5 * pair_step + depth_gap
    depth_centers = []

    panel_cfg = [
        ("accuracy", "Accuracy", C_ACC, "score"),
        ("f1", r"$F_1$", C_F1, "score"),
        ("fidelity", "Fidelity", C_FID, "score"),
        ("mw", "MW", C_MW, "score"),
        ("kl_inv", "KL-inv", C_KL, "score"),
    ]

    for d_i, depth in enumerate(depths):
        base = d_i * depth_width
        depth_centers.append(base + 2.5 * pair_step - pair_step / 2)

        ax.axvspan(
            base - 0.08,
            base + 5 * pair_step - metric_gap + 0.08,
            color="#f7f7f7" if d_i % 2 == 0 else "#ffffff",
            zorder=0,
        )

        for m_i, (col, title, color, _kind) in enumerate(panel_cfg):
            sv = float(sim[col].values[d_i])
            ov = float(odra[col].values[d_i])
            center = base + m_i * pair_step
            sim_x = center - bar_w / 2
            odra_x = center + bar_w / 2

            ax.bar(
                sim_x, sv, width=bar_w, color=color, alpha=0.52,
                edgecolor="white", linewidth=0.8, hatch="//", zorder=3,
            )
            ax.bar(
                odra_x, ov, width=bar_w, color=color, alpha=0.95,
                edgecolor="white", linewidth=0.8, zorder=3,
            )

            ax.text(
                sim_x, sv + 0.03, f"{sv:.2f}",
                ha="center", va="bottom", fontsize=7.2,
                fontweight="bold", color=color, rotation=90 if m_i >= 2 else 0,
            )
            ax.text(
                odra_x, ov + 0.03, f"{ov:.2f}",
                ha="center", va="bottom", fontsize=7.2,
                fontweight="bold", color=color, rotation=90 if m_i >= 2 else 0,
            )

            if d_i == 0:
                ax.text(
                    center, -0.11, title,
                    ha="center", va="top", fontsize=8.5,
                    fontweight="bold", color=color,
                    transform=ax.get_xaxis_transform(),
                )

        if d_i > 0:
            ax.axvline(base - depth_gap / 2, color="#cccccc", linewidth=1.0, linestyle=":", zorder=1)

    ax.set_xticks(depth_centers)
    ax.set_xticklabels([f"depth {d}" for d in depths], fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.16)
    ax.set_xlim(-0.18, depth_centers[-1] + depth_width / 2 - 0.08)
    ax.set_ylabel("Score (0 - 1, higher = better)", fontsize=11)
    ax.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.65, zorder=1)
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=C_ACC, lw=8, alpha=0.95, label="Accuracy"),
        Line2D([0], [0], color=C_F1, lw=8, alpha=0.95, label=r"$F_1$"),
        Line2D([0], [0], color=C_FID, lw=8, alpha=0.95, label="Fidelity"),
        Line2D([0], [0], color=C_MW, lw=8, alpha=0.95, label="MW"),
        Line2D([0], [0], color=C_KL, lw=8, alpha=0.95, label="KL-inv"),
        Line2D([0], [0], color="#666666", lw=8, alpha=0.55, label="ansatz_simulator (hatched)"),
        Line2D([0], [0], color="#666666", lw=8, alpha=0.95, label="ansatz_odra (solid)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=7,
        fontsize=9.0,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )

    fig.suptitle(
        "Simulator vs Odra gap by depth (all metrics in one plot)",
        fontsize=14.5,
        fontweight="bold",
        y=1.09,
    )
    fig.text(
        0.5, 0.01,
        "Numbers above bars show real plotted values. KL is shown as KL-inv on 0-1 scale; KL values illustrative.",
        ha="center", fontsize=9, color="#777777",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    out = SPARK / "framework_prediction.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
