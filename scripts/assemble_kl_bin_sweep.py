#!/usr/bin/env python3
"""Assemble combined KL fidelities and sweep KL(QPU||Haar) vs n_bins for presentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.metrics import compute_kl_for_fidelities  # noqa: E402
from qbanknote.paths import ensure_importable  # noqa: E402

DEFAULT_COMBINED = (
    ROOT
    / "evaluation_and_comparison"
    / "iqm_spark"
    / "iqm_kl_outputs"
    / "kl_combined"
)

ANSATZ_LABELS = {
    "ansatz_odra": "Odra",
    "ansatz_simulator": "Simulator",
    "ansatz_star": "Star",
}
ANSATZ_COLORS = {
    "ansatz_odra": "#212531",
    "ansatz_simulator": "#C4302B",
    "ansatz_star": "#2A6F97",
}
ITERATION_LINESTYLES = {1: "-", 2: "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute KL(QPU||Haar) across a bin grid from combined fidelities "
            "and write CSV + presentation plots."
        )
    )
    parser.add_argument("--combined-dir", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--bin-min", type=int, default=50)
    parser.add_argument("--bin-max", type=int, default=400)
    parser.add_argument("--n-bin-points", type=int, default=15)
    parser.add_argument("--n-qubits", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--reference-bins",
        type=int,
        nargs="+",
        default=[75, 150, 400],
        help="Also write point-estimate results tables at these bin counts.",
    )
    return parser.parse_args()


def build_bin_grid(bin_min: int, bin_max: int, n_points: int) -> list[int]:
    grid = np.unique(np.round(np.linspace(bin_min, bin_max, n_points)).astype(int))
    return [int(v) for v in grid]


def bootstrap_kl(
    fidelities: np.ndarray,
    *,
    dim: int,
    n_bins: int,
    eps: float,
    n_bootstrap: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float]:
    fidelities = np.asarray(fidelities, dtype=np.float64)
    kl_point, _, _, _ = compute_kl_for_fidelities(fidelities, dim, n_bins, eps)
    pct = int(round(confidence_level * 100))
    if len(fidelities) < 2 or n_bootstrap < 2:
        return {
            "kl_qpu_haar": float(kl_point),
            "kl_qpu_haar_lo": float(kl_point),
            "kl_qpu_haar_hi": float(kl_point),
        }

    rng = np.random.default_rng(seed)
    n = len(fidelities)
    kl_boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        draw = fidelities[rng.integers(0, n, size=n)]
        kl_boot[i], _, _, _ = compute_kl_for_fidelities(draw, dim, n_bins, eps)

    alpha = 1.0 - float(confidence_level)
    lo, hi = np.percentile(kl_boot, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "kl_qpu_haar": float(kl_point),
        "kl_qpu_haar_lo": float(lo),
        "kl_qpu_haar_hi": float(hi),
        "bootstrap_std": float(np.std(kl_boot, ddof=1)),
    }


def plot_depth_sweep(
    sweep_df: pd.DataFrame,
    *,
    depth: int,
    output_dir: Path,
    confidence_level: float,
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    depth_df = sweep_df[sweep_df["depth"] == depth].copy()
    if depth_df.empty:
        return []

    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=dpi)
    pct = int(round(confidence_level * 100))
    for (ansatz, iteration), group in depth_df.groupby(["ansatz", "iteration"], dropna=False):
        ordered = group.sort_values("n_bins")
        color = ANSATZ_COLORS.get(str(ansatz), "#333333")
        linestyle = ITERATION_LINESTYLES.get(int(iteration), "-")
        label = f"{ANSATZ_LABELS.get(str(ansatz), ansatz)}, iter {int(iteration)}"
        ax.plot(
            ordered["n_bins"],
            ordered["kl_qpu_haar"],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        ax.fill_between(
            ordered["n_bins"],
            ordered["kl_qpu_haar_lo"],
            ordered["kl_qpu_haar_hi"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    ax.set_xlabel("Number of histogram bins")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(P_{\mathrm{QPU}}\,\|\,P_{\mathrm{Haar}})$")
    ax.set_title(rf"KL vs bins — depth $L={depth}$ ($N=60$, $S=2048$)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8, framealpha=0.95)
    ax.text(
        0.02,
        0.98,
        f"{pct}% bootstrap CI",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout()
    paths = []
    for ext in ("png", "pdf"):
        path = output_dir / f"kl_qpu_haar_vs_bins_depth_{depth}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_facet_by_ansatz(
    sweep_df: pd.DataFrame,
    *,
    output_dir: Path,
    confidence_level: float,
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    ansatze = sorted(sweep_df["ansatz"].unique())
    fig, axes = plt.subplots(1, len(ansatze), figsize=(4.2 * len(ansatze), 4.2), dpi=dpi, sharey=True)
    if len(ansatze) == 1:
        axes = [axes]
    pct = int(round(confidence_level * 100))
    depth_colors = {2: "#1B998B", 4: "#E84855", 6: "#2E294E"}

    for ax, ansatz in zip(axes, ansatze, strict=True):
        sub = sweep_df[sweep_df["ansatz"] == ansatz]
        for (depth, iteration), group in sub.groupby(["depth", "iteration"], dropna=False):
            ordered = group.sort_values("n_bins")
            color = depth_colors.get(int(depth), "#333333")
            linestyle = ITERATION_LINESTYLES.get(int(iteration), "-")
            ax.plot(
                ordered["n_bins"],
                ordered["kl_qpu_haar"],
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                color=color,
                linestyle=linestyle,
                label=f"L={int(depth)}, iter {int(iteration)}",
            )
            ax.fill_between(
                ordered["n_bins"],
                ordered["kl_qpu_haar_lo"],
                ordered["kl_qpu_haar_hi"],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
        ax.set_title(ANSATZ_LABELS.get(str(ansatz), str(ansatz)))
        ax.set_xlabel("n_bins")
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(fontsize=7, loc="best", framealpha=0.9)

    axes[0].set_ylabel(r"$D_{\mathrm{KL}}(P_{\mathrm{QPU}}\,\|\,P_{\mathrm{Haar}})$")
    fig.suptitle(f"KL vs histogram bins by ansatz ({pct}% CI)", fontsize=11)
    fig.tight_layout()
    paths = []
    for ext in ("png", "pdf"):
        path = output_dir / f"kl_qpu_haar_vs_bins_by_ansatz.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def write_reference_results(
    combined_dir: Path,
    *,
    bin_counts: list[int],
    n_qubits: int,
    eps: float,
) -> Path:
    dim = 2**n_qubits
    rows: list[dict[str, object]] = []
    for iteration in (1, 2):
        fid_path = combined_dir / f"iteration_{iteration}" / "iqm_kl_fidelities.csv"
        fid = pd.read_csv(fid_path)
        for (ansatz, depth), group in fid.groupby(["ansatz", "depth"], dropna=False):
            f_phys = group.sort_values("sample_index")["fidelity_physical"].to_numpy(dtype=np.float64)
            f_lin = group.sort_values("sample_index")["fidelity_linear"].to_numpy(dtype=np.float64)
            source = str(group["source"].iloc[0]) if "source" in group.columns else ""
            for n_bins in bin_counts:
                kl_phys, _, _, _ = compute_kl_for_fidelities(f_phys, dim, int(n_bins), eps)
                kl_lin, _, _, _ = compute_kl_for_fidelities(f_lin, dim, int(n_bins), eps)
                rows.append(
                    {
                        "ansatz": str(ansatz),
                        "depth": int(depth),
                        "iteration": int(iteration),
                        "n_qubits": int(n_qubits),
                        "n_samples": int(len(f_phys)),
                        "n_bins": int(n_bins),
                        "eps": float(eps),
                        "kl_physical": float(kl_phys),
                        "kl_linear": float(kl_lin),
                        "f_physical_mean": float(np.mean(f_phys)),
                        "f_physical_std": float(np.std(f_phys, ddof=1)),
                        "f_linear_mean": float(np.mean(f_lin)),
                        "f_linear_std": float(np.std(f_lin, ddof=1)),
                        "source": source,
                    }
                )
            # also write per-iteration results at first reference bin for run compatibility
        ref = int(bin_counts[0])
        iter_rows = [r for r in rows if r["iteration"] == iteration and r["n_bins"] == ref]
        pd.DataFrame(iter_rows).drop(columns=["iteration"]).to_csv(
            combined_dir / f"iteration_{iteration}" / "iqm_kl_results.csv",
            index=False,
        )

    out = combined_dir / "analysis" / "kl_results_by_bins.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows).sort_values(["n_bins", "depth", "ansatz", "iteration"])
    result_df.to_csv(out, index=False)
    return out


def main() -> None:
    ensure_importable()
    args = parse_args()
    combined_dir = args.combined_dir.resolve()
    if not combined_dir.is_dir():
        raise SystemExit(f"Combined directory not found: {combined_dir}")

    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    bin_grid = build_bin_grid(args.bin_min, args.bin_max, args.n_bin_points)
    dim = 2 ** int(args.n_qubits)
    output_dir = combined_dir / "analysis" / "bin_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "note": (
            "Depth 2/4 (odra, simulator) from branch iqm-kl-pilot-data "
            "(not present on main); depth 6 (odra, simulator) from main "
            "kl_hardware_depth6; star depths 2/4/6 from branch star-2."
        ),
        "sources": {
            "d2_d4_odra_simulator": "iqm-kl-pilot-data:evaluation_and_comparison/iqm_spark/iqm_kl_outputs/kl_hardware",
            "d6_odra_simulator": "main working tree:evaluation_and_comparison/iqm_spark/iqm_kl_outputs/kl_hardware_depth6",
            "d2_d4_d6_star": "star-2:evaluation_and_comparison/iqm_spark/iqm_kl_outputs/kl_hardware_star",
        },
        "matrix": {
            "depths": [2, 4, 6],
            "ansatze": ["ansatz_odra", "ansatz_simulator", "ansatz_star"],
            "iterations": [1, 2],
            "n_samples_per_cell": 60,
            "shots": 2048,
        },
        "incomplete_excluded": (
            "iqm-kl-pilot-data kl_hardware iteration_1 had incomplete "
            "ansatz_odra depth=6 (31/60); replaced by main kl_hardware_depth6."
        ),
    }
    (combined_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    ref_path = write_reference_results(
        combined_dir,
        bin_counts=list(args.reference_bins),
        n_qubits=int(args.n_qubits),
        eps=float(args.eps),
    )

    rows: list[dict[str, object]] = []
    for iteration in (1, 2):
        fid = pd.read_csv(combined_dir / f"iteration_{iteration}" / "iqm_kl_fidelities.csv")
        for (ansatz, depth), group in fid.groupby(["ansatz", "depth"], dropna=False):
            f_qpu = group.sort_values("sample_index")["fidelity_physical"].to_numpy(dtype=np.float64)
            job_seed = int(args.base_seed) + int(depth) * 1000 + hash(str(ansatz)) % 997
            for n_bins in bin_grid:
                stats = bootstrap_kl(
                    f_qpu,
                    dim=dim,
                    n_bins=int(n_bins),
                    eps=float(args.eps),
                    n_bootstrap=int(args.bootstrap_trials),
                    seed=job_seed + int(n_bins),
                    confidence_level=float(args.confidence_level),
                )
                rows.append(
                    {
                        "ansatz": str(ansatz),
                        "depth": int(depth),
                        "iteration": int(iteration),
                        "n_samples": int(len(f_qpu)),
                        "n_bins": int(n_bins),
                        "confidence_level": float(args.confidence_level),
                        **stats,
                    }
                )

    sweep_df = pd.DataFrame(rows).sort_values(["depth", "ansatz", "iteration", "n_bins"])
    csv_path = output_dir / "kl_bin_sweep.csv"
    sweep_df.to_csv(csv_path, index=False)

    manifest = {
        "combined_dir": str(combined_dir),
        "bin_grid": bin_grid,
        "bootstrap_trials": int(args.bootstrap_trials),
        "confidence_level": float(args.confidence_level),
        "metric": "kl_qpu_haar",
        "n_rows": int(len(sweep_df)),
        "reference_results": str(ref_path),
    }
    (output_dir / "kl_bin_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    plot_paths: list[Path] = []
    for depth in sorted(sweep_df["depth"].unique()):
        plot_paths.extend(
            plot_depth_sweep(
                sweep_df,
                depth=int(depth),
                output_dir=output_dir,
                confidence_level=float(args.confidence_level),
                dpi=int(args.dpi),
            )
        )
    plot_paths.extend(
        plot_facet_by_ansatz(
            sweep_df,
            output_dir=output_dir,
            confidence_level=float(args.confidence_level),
            dpi=int(args.dpi),
        )
    )

    print(f"Provenance: {combined_dir / 'provenance.json'}")
    print(f"Results by bins: {ref_path}")
    print(f"Bin sweep CSV: {csv_path}")
    print(f"Bin grid ({len(bin_grid)}): {bin_grid}")
    for path in plot_paths:
        print(f"Plot: {path}")
    preview = sweep_df[
        ["ansatz", "depth", "iteration", "n_bins", "kl_qpu_haar", "kl_qpu_haar_lo", "kl_qpu_haar_hi"]
    ]
    print("\nPreview:")
    print(preview.head(24).to_string(index=False))


if __name__ == "__main__":
    main()
