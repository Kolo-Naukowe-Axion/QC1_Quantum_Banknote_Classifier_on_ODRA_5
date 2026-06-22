"""Plot KL(sim || Haar) expressibility table from main.tex (tab:kl_haar)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Table~\ref{tab:kl_haar} in main.tex (depths used in main experiments)
DEPTHS = np.array([2, 4, 6])
ODRA = np.array([0.096967, 0.000893, 0.000256])
SIMULATOR = np.array([0.068663, 0.002184, 0.000323])

COLOR_ODRA = "#1f77b4"
COLOR_SIM = "#ff7f0e"


def _format_kl_value(value: float) -> str:
    if value >= 0.01:
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return f"{value:.2e}"


def _annotate_bars(ax, bars, values: np.ndarray, *, y_factor: float = 1.18) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * y_factor,
            _format_kl_value(value),
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.15",
        )


def plot_kl_haar_depth(output_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = output_dir or Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    x = np.arange(len(DEPTHS))
    width = 0.35

    bars_odra = ax.bar(
        x - width / 2,
        ODRA,
        width,
        color=COLOR_ODRA,
        label="ansatz_odra (hardware-aligned)",
        edgecolor="white",
        linewidth=0.8,
    )
    bars_sim = ax.bar(
        x + width / 2,
        SIMULATOR,
        width,
        color=COLOR_SIM,
        label="ansatz_simulator (simulator-oriented)",
        edgecolor="white",
        linewidth=0.8,
    )
    _annotate_bars(ax, bars_odra, ODRA, y_factor=1.15)
    _annotate_bars(ax, bars_sim, SIMULATOR, y_factor=1.35)

    ax.set_yscale("log")
    ax.set_title(r"$D_{\mathrm{KL}}(p_{\mathrm{emp}}\,\|\,q_{\mathrm{Haar}})$ vs. nominal depth")
    ax.set_xlabel("Nominal circuit depth")
    ax.set_ylabel(r"$D_{\mathrm{KL}}$ (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(DEPTHS)
    ax.set_ylim(1.8e-4, 0.28)
    ax.grid(True, which="major", axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95)

    fig.text(
        0.5,
        0.01,
        "Lower is closer to Haar. 10⁵ Monte Carlo pairs, 150 bins, 5 qubits, noiseless statevector.",
        ha="center",
        fontsize=8,
        color="0.35",
    )

    png_path = out_dir / "kl_haar_depth.png"
    pdf_path = out_dir / "kl_haar_depth.pdf"
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    png, pdf = plot_kl_haar_depth()
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
