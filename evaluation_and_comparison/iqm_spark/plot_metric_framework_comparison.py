#!/usr/bin/env python3
"""Two-panel framework comparison: simulator vs odra, red=task / blue=hardware."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPARK = ROOT / "evaluation_and_comparison/iqm_spark"

METRIC_NAMES = ["Accuracy", r"$F_1$", "MW", "Fidelity", "KL"]
# Task metrics (Acc, F1) → reds; hardware (MW, Fid, KL) → blues
SHADES_TASK = ["#f4a9a3", "#c94438"]       # Acc, F1
SHADES_HW = ["#9ec5e8", "#4a8fc4", "#1a4d7a"]  # MW, Fidelity, KL
METRIC_COLORS = SHADES_TASK + SHADES_HW

COLOR_TASK_BG = "#fff5f4"
COLOR_HW_BG = "#f2f8fc"
DIVIDER = "#bbbbbb"
KL_SCALE = 0.11

depths = [2, 4, 6]


def load_fidelity():
    df = pd.read_csv(SPARK / "iqm_fidelity_outputs/cv/full_odra_fidelity_cv_summary.csv")
    odra = df[df["ansatz"] == "ansatz_odra"].set_index("depth").reindex(depths)
    sim = df[df["ansatz"] == "ansatz_simulator"].set_index("depth").reindex(depths)
    return (
        odra["cv_mean_of_fold_means"].values,
        sim["cv_mean_of_fold_means"].values,
        odra["cv_std_across_folds"].values,
        sim["cv_std_across_folds"].values,
    )


def load_mw():
    df = pd.read_csv(SPARK / "iqm_mw_outputs/mw_final_depths_2_4_6/iqm_mw_results.csv")
    odra = df[df["ansatz"] == "ansatz_odra"].set_index("depth").reindex(depths)
    sim = df[df["ansatz"] == "ansatz_simulator"].set_index("depth").reindex(depths)
    return (
        odra["mw_avg"].values,
        sim["mw_avg"].values,
        odra["mw_sem"].values,
        sim["mw_sem"].values,
    )


def task_metrics():
    """IQM Spark QPU accuracy / F1 from CV results table (depths 2, 4, 6)."""
    acc_odra = np.array([0.863, 0.876, 0.590])
    acc_sim = np.array([0.797, 0.745, 0.565])
    f1_odra = np.array([0.841, 0.846, 0.142])
    f1_sim = np.array([0.687, 0.583, 0.044])
    return acc_odra, acc_sim, f1_odra, f1_sim


def kl_metrics():
    odra = np.array([0.096967, 0.000893, 0.000256])
    sim = np.array([0.068663, 0.002184, 0.000323])
    return odra, sim


def kl_display(kl: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - kl / KL_SCALE, 0.0, 1.0)


def bar_label(metric_i: int, val: float, kl_raw: float | None = None) -> str:
    if metric_i == 4 and kl_raw is not None:
        return f"{kl_raw:.3f}"
    return f"{100 * val:.1f}%"


def draw_ansatz_panel(ax, values, errs, kl_raw, title: str) -> None:
    """One ansatz: 5 metrics × 3 depths on a single axes."""
    bar_w = 0.55
    metric_gap = 0.35
    depth_gap = 0.7
    metric_span = bar_w + metric_gap

    depth_centers = []
    for d_i, depth in enumerate(depths):
        base = d_i * (5 * metric_span + depth_gap)
        depth_center = base + 2 * metric_span
        depth_centers.append(depth_center)

        ax.axvspan(base - 0.05, base + 2 * metric_span - metric_gap / 2,
                   color=COLOR_TASK_BG, zorder=0)
        ax.axvspan(base + 2 * metric_span - metric_gap / 2, base + 5 * metric_span,
                   color=COLOR_HW_BG, zorder=0)

        for m_i, name in enumerate(METRIC_NAMES):
            cx = base + m_i * metric_span
            v = values[m_i][d_i]
            e = errs[m_i][d_i] if errs[m_i] is not None else 0
            kl_r = kl_raw[d_i] if m_i == 4 else None

            ax.bar(
                cx, v, bar_w, yerr=e if errs[m_i] is not None else None,
                capsize=3, color=METRIC_COLORS[m_i], edgecolor="white", linewidth=0.9, zorder=3,
                error_kw={"elinewidth": 0.9, "ecolor": "#555555"},
            )
            ax.text(
                cx, v + e + 0.028, bar_label(m_i, v, kl_r),
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=METRIC_COLORS[m_i], rotation=90 if m_i >= 2 else 0, zorder=5,
            )

        x_div = base + 2 * metric_span - metric_gap / 2
        ax.axvline(x_div, color=DIVIDER, linewidth=1.0, linestyle=":", alpha=0.8, zorder=1)

        if d_i == 0:
            for m_i, name in enumerate(METRIC_NAMES):
                cx = base + m_i * metric_span
                ax.text(cx, 1.04, name, ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color="#444444", transform=ax.get_xaxis_transform())

    ax.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.9, alpha=0.65, zorder=1)
    ax.set_xticks(depth_centers)
    ax.set_xticklabels([f"depth {d}" for d in depths], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-0.3, depth_centers[-1] + 2.5 * metric_span)
    ax.set_ylabel("Score (0 – 1)", fontsize=9)
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=28)


def main() -> None:
    from plot_metric_framework_variants import main as run_all
    run_all()


if __name__ == "__main__":
    main()
