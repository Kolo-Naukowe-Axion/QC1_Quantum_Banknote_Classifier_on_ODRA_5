#!/usr/bin/env python3
"""Plot simulator fidelity band (datasheet err2 sweep) vs. QPU fidelity.

The simulator is shown as a shaded band spanning the IQM Spark datasheet
two-qubit error range (err2 = 0.010..0.020), with the nominal err2 = 0.010
curve drawn as a line. QPU physical-projection fidelity is overlaid as points
with SEM error bars. One panel per ansatz.

Reads the actual result CSVs under evaluation_and_comparison/iqm_spark/
iqm_fidelity_outputs and writes a PNG + PDF next to them.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "evaluation_and_comparison" / "iqm_spark" / "iqm_fidelity_outputs"
HW = OUT_DIR / "fidelity_hardware"
STAR_PILOT = (
    OUT_DIR
    / "pilots"
    / "fidelity_odra_star_depths_2_4_6"
    / "sample_pilot"
    / "samples_10"
    / "iqm_fidelity_results.csv"
)
SIM_SWEEP = OUT_DIR / "simulator" / "sim_fidelity_spark_datasheet_range" / "simulator_fidelity_summary.csv"

DEPTHS = [2, 4, 6]
ANSATZE = ["ansatz_odra", "ansatz_odra_star", "ansatz_simulator"]
TITLES = {
    "ansatz_odra": "Odra",
    "ansatz_odra_star": "Odra-star",
    "ansatz_simulator": "Simulator",
}
NOMINAL_ERR2 = 0.01


def load_qpu() -> pd.DataFrame:
    """Mean physical fidelity over the two hardware iterations (odra, simulator)."""
    frames = []
    for it in ("iteration_1", "iteration_2"):
        f = HW / it / "iqm_fidelity_results.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    hw = pd.concat(frames, ignore_index=True)
    agg = (
        hw.groupby(["ansatz", "depth"])
        .agg(f=("f_phys_avg", "mean"), sem=("f_phys_sem", "mean"))
        .reset_index()
    )
    # odra_star hardware pilot (depths 2, 4 available; depth 6 pending).
    if STAR_PILOT.exists():
        star = pd.read_csv(STAR_PILOT)[["ansatz", "depth", "f_phys_avg", "f_phys_sem"]]
        star = star.rename(columns={"f_phys_avg": "f", "f_phys_sem": "sem"})
        agg = pd.concat([agg, star], ignore_index=True)
    return agg


def main() -> None:
    sim = pd.read_csv(SIM_SWEEP)
    qpu = load_qpu()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    band_color = "#4C72B0"
    qpu_color = "#C44E52"

    for ax, ansatz in zip(axes, ANSATZE):
        s = sim[sim["ansatz"] == ansatz]
        lo, hi, nom = [], [], []
        for d in DEPTHS:
            sd = s[s["depth"] == d]
            lo.append(sd["f_avg"].min())
            hi.append(sd["f_avg"].max())
            nrow = sd[np.isclose(sd["err2"], NOMINAL_ERR2)]
            nom.append(float(nrow["f_avg"].iloc[0]) if len(nrow) else np.nan)

        ax.fill_between(
            DEPTHS, lo, hi, color=band_color, alpha=0.22,
            label="Simulator (err$_2$=0.010–0.020)",
        )
        ax.plot(DEPTHS, nom, color=band_color, lw=1.6, marker="o", ms=4,
                label="Simulator (err$_2$=0.010)")

        q = qpu[qpu["ansatz"] == ansatz].sort_values("depth")
        ax.errorbar(
            q["depth"], q["f"], yerr=q["sem"], color=qpu_color, lw=0, marker="s",
            ms=7, capsize=3, elinewidth=1.4, markeredgecolor="white",
            markeredgewidth=0.6, label="QPU (IQM Spark)", zorder=5,
        )
        # Flag pending QPU depth-6 point for odra_star.
        if ansatz == "ansatz_odra_star" and 6 not in set(q["depth"]):
            ax.scatter([6], [0.0], marker="x", color="gray", s=40, zorder=4)
            ax.annotate(
                "QPU d=6\npending", (6, 0.02), ha="center", va="bottom",
                fontsize=7.5, color="gray",
            )

        ax.set_title(TITLES[ansatz], fontsize=12)
        ax.set_xlabel("Ansatz depth")
        ax.set_xticks(DEPTHS)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"Fidelity $F_{\mathrm{phys}}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    png = OUT_DIR / "fidelity_sim_vs_qpu.png"
    pdf = OUT_DIR / "fidelity_sim_vs_qpu.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
