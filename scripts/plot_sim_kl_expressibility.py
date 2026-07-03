#!/usr/bin/env python3
"""Headline KL(Sim||Haar) figure from compute_sim_kl_expressibility.py outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.paths import ensure_importable  # noqa: E402

ANSATZ_ORDER = ("ansatz_odra", "ansatz_simulator")
ANSATZ_LABELS = {
    "ansatz_odra": "Odra",
    "ansatz_simulator": "Simulator",
}
ANSATZ_COLORS = {
    "ansatz_odra": "#212531",
    "ansatz_simulator": "#C4302B",
}

FIDELITIES_CSV = "sim_kl_fidelities.csv"
RESULTS_CSV = "sim_kl_results.csv"


def haar_pdf_fidelity(f: np.ndarray, dim: int) -> np.ndarray:
    return (dim - 1.0) * (1.0 - f) ** (dim - 2.0)


def compute_kl_for_fidelities(
    fidelities: np.ndarray,
    dim: int,
    n_bins: int,
    eps: float,
) -> float:
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    counts, edges = np.histogram(fidelities, bins=bins, density=False)
    p_emp = counts.astype(np.float64)
    if p_emp.sum() == 0:
        p_emp = np.ones_like(p_emp) / len(p_emp)
    else:
        p_emp /= p_emp.sum()

    mids = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    p_haar = haar_pdf_fidelity(mids, dim=dim) * width
    p_haar /= p_haar.sum()

    p_s = p_emp + eps
    q_s = p_haar + eps
    p_s /= p_s.sum()
    q_s /= q_s.sum()
    return float(np.sum(p_s * np.log(p_s / q_s)))


def bootstrap_kl_interval(
    fidelities: np.ndarray,
    *,
    dim: int,
    n_bins: int,
    eps: float,
    n_bootstrap: int,
    seed: int,
    ci_level: float,
) -> dict[str, float | int]:
    fidelities = np.asarray(fidelities, dtype=np.float64)
    kl_point = compute_kl_for_fidelities(fidelities, dim, n_bins, eps)
    result: dict[str, float | int] = {
        "kl_physical": float(kl_point),
        "n_samples": int(len(fidelities)),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_std": 0.0,
    }
    if len(fidelities) < 2 or n_bootstrap < 2:
        pct = int(round(ci_level * 100))
        result[f"bootstrap_lo_{pct}"] = float(kl_point)
        result[f"bootstrap_hi_{pct}"] = float(kl_point)
        return result

    rng = np.random.default_rng(seed)
    n = len(fidelities)
    kl_boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for index in range(int(n_bootstrap)):
        draw = fidelities[rng.integers(0, n, size=n)]
        kl_boot[index] = compute_kl_for_fidelities(draw, dim, n_bins, eps)

    result["bootstrap_std"] = float(np.std(kl_boot, ddof=1))
    alpha = 1.0 - float(ci_level)
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    lo, hi = np.percentile(kl_boot, [lo_pct, hi_pct])
    pct = int(round(ci_level * 100))
    result[f"bootstrap_lo_{pct}"] = float(lo)
    result[f"bootstrap_hi_{pct}"] = float(hi)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot headline KL(Sim||Haar) Odra vs Simulator from saved fidelities."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "evaluation_and_comparison" / "simulator" / "sim_kl_outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/analysis/presentation",
    )
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.90,
        help="Bootstrap confidence level for error bars (default: 0.90).",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_job_fidelities(path: Path, ansatz: str, depth: int) -> np.ndarray:
    rows = [
        row
        for row in read_csv_rows(path)
        if row.get("ansatz") == ansatz and int(row.get("depth", -1)) == int(depth)
    ]
    if not rows:
        return np.array([], dtype=np.float64)
    rows.sort(key=lambda row: int(row["sample_index"]))
    return np.array([float(row["fidelity"]) for row in rows], dtype=np.float64)


def infer_run_params(run_dir: Path) -> dict[str, object]:
    results_path = run_dir / RESULTS_CSV
    manifest_path = run_dir / "run_manifest.json"
    if results_path.is_file():
        rows = read_csv_rows(results_path)
        if rows:
            first = rows[0]
            return {
                "n_qubits": int(first["n_qubits"]),
                "n_bins": int(first["n_bins"]),
                "eps": float(first["eps"]),
                "depths": sorted({int(row["depth"]) for row in rows}),
                "ansatz": [row["ansatz"] for row in rows],
            }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "n_qubits": int(manifest["n_qubits"]),
            "n_bins": int(manifest["n_bins"]),
            "eps": float(manifest["eps"]),
            "depths": [int(depth) for depth in manifest["depths"]],
            "ansatz": list(manifest["ansatz"]),
        }
    raise FileNotFoundError(f"Missing {results_path} and {manifest_path}")


def summarize_jobs(
    run_dir: Path,
    *,
    depths: list[int],
    n_qubits: int,
    n_bins: int,
    eps: float,
    n_bootstrap: int,
    seed: int,
    ci_level: float,
) -> list[dict[str, object]]:
    fidelities_path = run_dir / FIDELITIES_CSV
    if not fidelities_path.is_file():
        raise FileNotFoundError(f"Missing fidelity archive: {fidelities_path}")

    dim = 2 ** int(n_qubits)
    rows: list[dict[str, object]] = []
    for ansatz in ANSATZ_ORDER:
        for depth in depths:
            fidelities = load_job_fidelities(fidelities_path, ansatz, depth)
            if fidelities.size == 0:
                raise ValueError(f"No fidelities for {ansatz} depth={depth}")
            interval = bootstrap_kl_interval(
                fidelities,
                dim=dim,
                n_bins=n_bins,
                eps=eps,
                n_bootstrap=n_bootstrap,
                seed=seed + int(depth) + (1 if ansatz == "ansatz_simulator" else 0),
                ci_level=ci_level,
            )
            pct = int(round(ci_level * 100))
            rows.append(
                {
                    "ansatz": ansatz,
                    "ansatz_label": ANSATZ_LABELS[ansatz],
                    "depth": int(depth),
                    "n_bins": int(n_bins),
                    "n_samples": int(len(fidelities)),
                    "kl_sim_haar": float(interval["kl_physical"]),
                    "ci_lo": float(interval[f"bootstrap_lo_{pct}"]),
                    "ci_hi": float(interval[f"bootstrap_hi_{pct}"]),
                    "bootstrap_std": float(interval["bootstrap_std"]),
                    "n_bootstrap": int(interval["n_bootstrap"]),
                }
            )
    return rows


def plot_headline(
    summary_rows: list[dict[str, object]],
    *,
    output_dir: Path,
    ci_level: float,
    dpi: int,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    depths = sorted({int(row["depth"]) for row in summary_rows})
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=dpi)

    x = np.arange(len(depths))
    width = 0.34
    for ansatz in ANSATZ_ORDER:
        subset = [row for row in summary_rows if row["ansatz"] == ansatz]
        by_depth = {int(row["depth"]): row for row in subset}
        means = [float(by_depth[depth]["kl_sim_haar"]) for depth in depths]
        yerr_lo = [
            max(0.0, float(by_depth[depth]["kl_sim_haar"] - by_depth[depth]["ci_lo"]))
            for depth in depths
        ]
        yerr_hi = [
            max(0.0, float(by_depth[depth]["ci_hi"] - by_depth[depth]["kl_sim_haar"]))
            for depth in depths
        ]
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
    ax.set_xticklabels([f"depth {depth}" for depth in depths])
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(P_{\mathrm{Sim}}\,\|\,P_{\mathrm{Haar}})$")
    ax.set_title("Expressibility on Simulator")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()

    pct = int(round(ci_level * 100))
    stem = f"v2_headline_odra_vs_sim_pooled_ci{pct}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    ensure_importable()
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    try:
        import pandas as pd  # noqa: F401
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Requires pandas and matplotlib.") from exc

    params = infer_run_params(run_dir)
    n_bins = int(args.n_bins if args.n_bins is not None else params["n_bins"])
    eps = float(args.eps if args.eps is not None else params["eps"])
    depths = sorted(int(depth) for depth in params["depths"])

    summary_rows = summarize_jobs(
        run_dir,
        depths=depths,
        n_qubits=int(params["n_qubits"]),
        n_bins=n_bins,
        eps=eps,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        ci_level=args.ci_level,
    )

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis" / "presentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "sim_kl_presentation_summary.csv"
    write_summary_csv(summary_csv, summary_rows)
    png_path, pdf_path = plot_headline(
        summary_rows,
        output_dir=output_dir,
        ci_level=args.ci_level,
        dpi=args.dpi,
    )

    manifest = {
        "metric": "kl_sim_haar",
        "run_dir": str(run_dir),
        "n_bins": n_bins,
        "eps": eps,
        "n_bootstrap": int(args.n_bootstrap),
        "ci_level": float(args.ci_level),
        "rows": summary_rows,
        "figures": {"png": str(png_path), "pdf": str(pdf_path)},
        "summary_csv": str(summary_csv),
    }
    manifest_path = output_dir / "presentation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print("Summary:")
    for row in summary_rows:
        print(
            f"  {row['ansatz_label']:10s} depth={row['depth']:d}  "
            f"KL={row['kl_sim_haar']:.6g}  "
            f"CI{int(round(args.ci_level * 100))}=[{row['ci_lo']:.6g}, {row['ci_hi']:.6g}]  "
            f"N={row['n_samples']}"
        )
    print(f"\nSummary CSV: {summary_csv}")
    print(f"Plot: {png_path}")
    print(f"Plot: {pdf_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
