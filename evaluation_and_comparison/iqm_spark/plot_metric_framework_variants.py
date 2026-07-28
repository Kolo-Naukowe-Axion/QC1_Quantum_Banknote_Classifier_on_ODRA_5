#!/usr/bin/env python3
"""Four framework visualizations: scatter, dual-axis, heatmap, grouped bars."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
SPARK = ROOT / "evaluation_and_comparison/iqm_spark"

METRIC_NAMES = ["Accuracy", r"$F_1$", "MW", "Fidelity", "KL"]
SHADES_TASK = ["#f4a9a3", "#c94438"]
SHADES_HW = ["#9ec5e8", "#4a8fc4", "#1a4d7a"]
METRIC_COLORS = SHADES_TASK + SHADES_HW
COLOR_TASK_BG = "#fff5f4"
COLOR_HW_BG = "#f2f8fc"
DIVIDER = "#bbbbbb"
KL_SCALE = 0.11
depths = [2, 4, 6]

DEPTH_COLORS = {2: "#4C72B0", 4: "#55A868", 6: "#DD8452"}
ANSATZ_MARKERS = {"ansatz_simulator": "o", "ansatz_odra": "s"}


def kl_display(kl: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - kl / KL_SCALE, 0.0, 1.0)


def load_framework_df() -> pd.DataFrame:
    fid_o, fid_s, _, _ = _load_fidelity()
    mw_o, mw_s, _, _ = _load_mw()
    acc_o, acc_s, f1_o, f1_s = _task_metrics()
    kl_o, kl_s = _kl_metrics()

    rows = []
    for d_i, depth in enumerate(depths):
        for ansatz, acc, f1, mw, fid, kl in [
            ("ansatz_simulator", acc_s[d_i], f1_s[d_i], mw_s[d_i], fid_s[d_i], kl_s[d_i]),
            ("ansatz_odra", acc_o[d_i], f1_o[d_i], mw_o[d_i], fid_o[d_i], kl_o[d_i]),
        ]:
            kl_inv = float(kl_display(np.array([kl]))[0])
            rows.append({
                "depth": depth,
                "ansatz": ansatz,
                "ansatz_short": "simulator" if "simulator" in ansatz else "odra",
                "accuracy": acc,
                "f1": f1,
                "mw": mw,
                "fidelity": fid,
                "kl_raw": kl,
                "kl_inv": kl_inv,
                "hw_composite": 0.5 * fid + 0.5 * kl_inv,
                "label": f"d{depth}·{('sim' if 'simulator' in ansatz else 'odra')}",
            })
    return pd.DataFrame(rows)


def _load_fidelity():
    df = pd.read_csv(SPARK / "iqm_fidelity_outputs/cv/full_odra_fidelity_cv_summary.csv")
    odra = df[df["ansatz"] == "ansatz_odra"].set_index("depth").reindex(depths)
    sim = df[df["ansatz"] == "ansatz_simulator"].set_index("depth").reindex(depths)
    return (
        odra["cv_mean_of_fold_means"].values,
        sim["cv_mean_of_fold_means"].values,
        odra["cv_std_across_folds"].values,
        sim["cv_std_across_folds"].values,
    )


def _load_mw():
    df = pd.read_csv(SPARK / "iqm_mw_outputs/mw_final_depths_2_4_6/iqm_mw_results.csv")
    odra = df[df["ansatz"] == "ansatz_odra"].set_index("depth").reindex(depths)
    sim = df[df["ansatz"] == "ansatz_simulator"].set_index("depth").reindex(depths)
    return (
        odra["mw_avg"].values,
        sim["mw_avg"].values,
        odra["mw_sem"].values,
        sim["mw_sem"].values,
    )


def _task_metrics():
    acc_odra = np.array([0.863, 0.876, 0.590])
    acc_sim = np.array([0.797, 0.745, 0.565])
    f1_odra = np.array([0.841, 0.846, 0.142])
    f1_sim = np.array([0.687, 0.583, 0.044])
    return acc_odra, acc_sim, f1_odra, f1_sim


def _kl_metrics():
    odra = np.array([0.096967, 0.000893, 0.000256])
    sim = np.array([0.068663, 0.002184, 0.000323])
    return odra, sim


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def plot_scatter(df: pd.DataFrame, out: Path) -> None:
    """1. Diagnostic vs task performance scatter (color=depth, shape=ansatz)."""
    diag_cols = [
        ("fidelity", r"$\mathcal{F}_{\mathrm{phys}}$"),
        ("kl_inv", r"KL inverted ($1 - D_{KL}/0.11$)"),
        ("mw", "Meyer–Wallach $Q$"),
        ("hw_composite", r"Composite ($\mathcal{F}$ + KL inv.) / 2"),
    ]
    task_cols = [("accuracy", "Accuracy"), ("f1", r"$F_1$")]

    fig, axes = plt.subplots(len(diag_cols), len(task_cols), figsize=(10, 13), facecolor="white")
    for i, (dx, dx_lbl) in enumerate(diag_cols):
        for j, (ty, ty_lbl) in enumerate(task_cols):
            ax = axes[i, j]
            for _, row in df.iterrows():
                ax.scatter(
                    row[dx], row[ty],
                    c=DEPTH_COLORS[row["depth"]],
                    marker=ANSATZ_MARKERS[row["ansatz"]],
                    s=90, edgecolors="white", linewidths=1.2, zorder=3,
                )
                ax.annotate(row["label"], (row[dx], row[ty]),
                            textcoords="offset points", xytext=(5, 4), fontsize=7, color="#444444")
            r = _pearson(df[dx].values, df[ty].values)
            ax.set_xlabel(dx_lbl, fontsize=9)
            ax.set_ylabel(ty_lbl, fontsize=9)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if not np.isnan(r):
                ax.text(0.04, 0.96, f"$r$ = {r:.2f}", transform=ax.transAxes,
                        va="top", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#ccc"))

    depth_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                            markersize=8, label=f"depth {d}") for d, c in DEPTH_COLORS.items()]
    ansatz_handles = [
        Line2D([0], [0], marker="o", color="#555", linestyle="None", markersize=8, label="simulator"),
        Line2D([0], [0], marker="s", color="#555", linestyle="None", markersize=8, label="odra"),
    ]
    fig.legend(handles=depth_handles + ansatz_handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.01), fontsize=8, frameon=True)
    fig.suptitle("Diagnostic metrics vs. task performance\n(each point = depth × ansatz)",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def plot_dual_axis(df: pd.DataFrame, out: Path) -> None:
    """2. Dual-axis trend over depth (task left, diagnostics right)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white", sharex=True)

    for ax, ansatz, title in zip(
        axes,
        ["ansatz_simulator", "ansatz_odra"],
        ["ansatz_simulator", "ansatz_odra"],
    ):
        sub = df[df["ansatz"] == ansatz].sort_values("depth")
        x = sub["depth"].values
        ax2 = ax.twinx()

        l1, = ax.plot(x, sub["accuracy"], "o-", color=SHADES_TASK[0], linewidth=2,
                      markersize=8, label="Accuracy")
        l2, = ax.plot(x, sub["f1"], "s--", color=SHADES_TASK[1], linewidth=2,
                      markersize=8, label=r"$F_1$")
        l3, = ax2.plot(x, sub["fidelity"], "^-", color=SHADES_HW[1], linewidth=2,
                       markersize=8, label=r"$\mathcal{F}_{\mathrm{phys}}$")
        l4, = ax2.plot(x, sub["kl_inv"], "d--", color=SHADES_HW[2], linewidth=2,
                       markersize=8, label="KL inverted")

        ax.set_xlabel("Circuit depth", fontsize=10)
        ax.set_ylabel("Task metrics (Accuracy, $F_1$)", fontsize=9, color="#8b3030")
        ax2.set_ylabel("Hardware diagnostics", fontsize=9, color="#1a4d7a")
        ax.set_xticks(depths)
        ax.set_ylim(0, 1.05)
        ax2.set_ylim(0, 1.05)
        ax.tick_params(axis="y", labelcolor="#8b3030")
        ax2.tick_params(axis="y", labelcolor="#1a4d7a")
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(handles=[l1, l2], loc="upper left", fontsize=8)
        ax2.legend(handles=[l3, l4], loc="upper right", fontsize=8)

    fig.suptitle("Task vs. hardware metrics across depth\n(dual axis — drops should track together)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def plot_correlation(df: pd.DataFrame, out: Path) -> None:
    """3. Pearson correlation heatmap across all five metrics."""
    cols = ["accuracy", "f1", "mw", "fidelity", "kl_inv"]
    labels = ["Accuracy", r"$F_1$", "MW", r"$\mathcal{F}_{\mathrm{phys}}$", "KL inv."]
    corr = df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson $r$")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            highlight = abs(val) >= 0.8 and i != j
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10,
                    fontweight="bold" if highlight else "normal",
                    color="white" if abs(val) > 0.55 else "#222222")
            if highlight:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#ffd700", linewidth=3))

    ax.set_title(
        f"Correlation matrix ($n$={len(df)} configurations)\n"
        "gold border: $|r| \\geq 0.8$ between task & hardware metrics",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def plot_grouped_bars(df: pd.DataFrame, out: Path) -> None:
    """4. Two-panel grouped bars (red task / blue hardware) — original layout."""
    fid_o, fid_s, fid_o_e, fid_s_e = _load_fidelity()
    mw_o, mw_s, mw_o_e, mw_s_e = _load_mw()
    kl_o, kl_s = _kl_metrics()

    def panel_data(ansatz: str):
        sub = df[df["ansatz"] == ansatz].sort_values("depth")
        kl_raw = kl_s if ansatz == "ansatz_simulator" else kl_o
        errs = [None, None,
                mw_s_e if ansatz == "ansatz_simulator" else mw_o_e,
                fid_s_e if ansatz == "ansatz_simulator" else fid_o_e,
                None]
        return (
            [sub["accuracy"].values, sub["f1"].values, sub["mw"].values,
             sub["fidelity"].values, kl_display(kl_raw)],
            errs, kl_raw,
        )

    fig, (ax_sim, ax_odra) = plt.subplots(1, 2, figsize=(15, 5.8), facecolor="white", sharey=True)
    for ax, ansatz, title in [
        (ax_sim, "ansatz_simulator", "ansatz_simulator"),
        (ax_odra, "ansatz_odra", "ansatz_odra"),
    ]:
        values, errs, kl_raw = panel_data(ansatz)
        _draw_ansatz_panel(ax, values, errs, kl_raw, title)

    task_patches = [mpatches.Patch(facecolor=c, edgecolor="white", label=n)
                    for c, n in zip(SHADES_TASK, ["Accuracy", r"$F_1$"])]
    hw_patches = [mpatches.Patch(facecolor=c, edgecolor="white", label=n)
                  for c, n in zip(SHADES_HW, ["MW", "Fidelity", "KL"])]
    fig.legend(handles=task_patches + hw_patches, loc="upper center", ncol=5, fontsize=8,
               title="red = task  ·  blue = hardware", title_fontsize=9,
               bbox_to_anchor=(0.5, 1.02), frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.suptitle("Five-dimensional ansatz evaluation on IQM Spark", fontsize=13, fontweight="bold", y=1.08)
    fig.text(0.5, 0.01, "KL inverted on 0–1 axis  ·  dashed = 50% fidelity",
             ha="center", fontsize=8, color="#666666")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def _bar_label(metric_i: int, val: float, kl_raw: float | None = None) -> str:
    if metric_i == 4 and kl_raw is not None:
        return f"{kl_raw:.3f}"
    return f"{100 * val:.1f}%"


def _draw_ansatz_panel(ax, values, errs, kl_raw, title: str) -> None:
    # Make the "gap" between task metrics (Accuracy/F1) and hardware diagnostics
    # (MW/Fidelity/KL) intentionally large, so it is readable at a glance.
    bar_w = 0.55
    intra_gap = 0.14        # spacing between bars within the same group
    group_gap = 1.05        # spacing between task group and HW group
    depth_gap = 0.7         # spacing between different depths

    task_step = bar_w + intra_gap
    depth_group_span = 4 * task_step + group_gap + bar_w  # total x-width per depth
    depth_centers = []
    for d_i, depth in enumerate(depths):
        base = d_i * (depth_group_span + depth_gap)

        # Bar centers (Accuracy, F1) then a large group gap then (MW, Fidelity, KL).
        task_centers = [base, base + task_step]
        hw_start = base + 2 * task_step + group_gap
        hw_centers = [hw_start, hw_start + task_step, hw_start + 2 * task_step]
        metric_xs = task_centers + hw_centers

        # Center x-position for the depth label
        depth_centers.append(base + 2 * task_step + group_gap / 2)

        # Colored background blocks for the two groups.
        task_left = base - bar_w / 2 - 0.05
        task_right = task_centers[-1] + bar_w / 2 + 0.02
        hw_left = hw_centers[0] - bar_w / 2 - 0.02
        hw_right = hw_centers[-1] + bar_w / 2 + 0.05
        ax.axvspan(task_left, task_right, color=COLOR_TASK_BG, zorder=0)
        ax.axvspan(hw_left, hw_right, color=COLOR_HW_BG, zorder=0)

        # Clear divider line in the middle of the big gap.
        divider_x = (task_centers[-1] + hw_centers[0]) / 2
        ax.axvline(
            divider_x,
            color=DIVIDER,
            linewidth=2.0,
            linestyle="-",
            alpha=0.95,
            zorder=1,
        )
        for m_i in range(5):
            cx = metric_xs[m_i]
            v = values[m_i][d_i]
            e = errs[m_i][d_i] if errs[m_i] is not None else 0
            kl_r = kl_raw[d_i] if m_i == 4 else None
            ax.bar(cx, v, bar_w, yerr=e if errs[m_i] is not None else None, capsize=3,
                   color=METRIC_COLORS[m_i], edgecolor="white", linewidth=0.9, zorder=3,
                   error_kw={"elinewidth": 0.9, "ecolor": "#555555"})
            ax.text(cx, v + e + 0.028, _bar_label(m_i, v, kl_r), ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color=METRIC_COLORS[m_i],
                    rotation=90 if m_i >= 2 else 0, zorder=5)
        if d_i == 0:
            for m_i, name in enumerate(METRIC_NAMES):
                ax.text(metric_xs[m_i], 1.04, name, ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#444444",
                        transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.9, alpha=0.65, zorder=1)
    ax.set_xticks(depth_centers)
    ax.set_xticklabels([f"depth {d}" for d in depths], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(depth_centers[0] - depth_group_span / 2 - 0.15, depth_centers[-1] + depth_group_span / 2 + 0.15)
    ax.set_ylabel("Score (0 – 1)", fontsize=9)
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=28)


def main() -> None:
    df = load_framework_df()
    plot_scatter(df, SPARK / "metric_framework_scatter.png")
    plot_dual_axis(df, SPARK / "metric_framework_dual_axis.png")
    plot_correlation(df, SPARK / "metric_framework_correlation.png")
    plot_grouped_bars(df, SPARK / "metric_framework_comparison.png")


if __name__ == "__main__":
    main()
