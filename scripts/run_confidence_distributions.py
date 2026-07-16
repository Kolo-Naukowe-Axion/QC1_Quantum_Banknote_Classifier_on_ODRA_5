#!/usr/bin/env python3
"""Compute <Z_0> confidence distributions (ideal / shot / hardware) per class.

Offline by default (statevector + classical shot resample); add ``--with-hardware``
to evaluate on the IQM Spark QPU. Per-sample rows are appended to a resumable CSV,
so a dropped hardware repeat can be re-run without losing completed work.

``--env``, ``--depth`` and ``--fold`` each accept multiple values; the script
sweeps their full cartesian product, one resumable CSV set per config, reusing a
single IQM connection. A config that fails is reported and skipped, not fatal.

Examples
--------
Offline preview, single config (no QPU)::

    python scripts/run_confidence_distributions.py --env Odra --depth 2 --fold 1

Sweep both ansatzes, three depths, five folds on hardware::

    python scripts/run_confidence_distributions.py \
        --env Odra Simulator --depth 2 4 6 --fold 1 2 3 4 5 \
        --with-hardware --repeats 3 --shots 2048 --max-circuits-per-job 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote import confidence as C  # noqa: E402
from qbanknote.ansatzes import odra_ansatz, simulator_ansatz  # noqa: E402
from qbanknote.data import load_fold_arrays  # noqa: E402
from qbanknote.evaluation import append_csv_row, read_csv_or_empty  # noqa: E402
from qbanknote.iqm import build_iqm_estimator_model, connect_to_iqm_backend  # noqa: E402
from qbanknote.paths import find_project_root  # noqa: E402
from qbanknote.weights import (  # noqa: E402
    cv_weight_path,
    load_checkpoint_connector,
    load_cv_model,
)

DEFAULT_OUT = "evaluation_and_comparison/iqm_spark/confidence_outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="<Z_0> confidence distributions across classes")
    p.add_argument("--env", choices=("Odra", "Simulator"), nargs="+", default=["Odra"],
                   help="one or more ansatz environments to sweep")
    p.add_argument("--depth", type=int, choices=(2, 4, 6), nargs="+", default=[2],
                   help="one or more circuit depths to sweep")
    p.add_argument("--fold", type=int, nargs="+", default=[1],
                   help="one or more CV folds to sweep")
    p.add_argument("--condition", choices=("Ideal", "Noise"), default="Ideal",
                   help="which trained weights to load (clean vs noise-trained theta*)")
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=3, help="hardware drift repeats")
    p.add_argument("--skip-shot", action="store_true",
                   help="skip the classical shot-noise control (compute only ideal); "
                        "use offline so the shot control can be regenerated at the "
                        "pilot-chosen shot count during the hardware run")
    p.add_argument("--with-hardware", action="store_true")
    p.add_argument("--iqm-url", default="https://odra5.e-science.pl/")
    p.add_argument("--iqm-token", default=None)
    p.add_argument("--max-circuits-per-job", type=int, default=100)
    p.add_argument("--optimization-level", type=int, default=1)
    p.add_argument("--seed-transpiler", type=int, default=42)
    p.add_argument("--seed", type=int, default=42, help="RNG seed for the shot resample")
    p.add_argument("--num-qubits", type=int, default=5)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def ansatz_fn(env: str):
    return odra_ansatz if env == "Odra" else simulator_ansatz


def completed_runs(csv_path: Path) -> set[tuple[str, int]]:
    """Return the set of (Condition, RunIndex) groups already fully recorded."""
    df = read_csv_or_empty(csv_path)
    if df.empty or "Condition" not in df.columns:
        return set()
    return {(str(c), int(r)) for c, r in zip(df["Condition"], df["RunIndex"])}


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    for row in rows:
        append_csv_row(csv_path, row)


def run_one_config(args, root: Path, out_dir: Path, backend, env: str,
                   depth: int, fold: int) -> None:
    """Run the ideal/shot/(hardware) pipeline for one (env, depth, fold)."""
    tag = f"{env}_d{depth}_fold{fold}"
    samples_csv = out_dir / f"confidence_distributions_{tag}.csv"
    summary_csv = out_dir / f"confidence_summary_{tag}.csv"
    noisefit_csv = out_dir / f"confidence_noisefit_{tag}.csv"

    done = completed_runs(samples_csv)
    rng = np.random.default_rng(args.seed)

    print(f"\n{'=' * 70}\n=== {tag} ===\n{'=' * 70}")

    # --- Data + trained model (statevector) -------------------------------
    X, y = load_fold_arrays(fold, split="test")
    model, weight_used = load_cv_model(
        depth, env, fold, condition=args.condition, root=root
    )
    print(f"Model: {weight_used}")
    print(f"Test set: {len(y)} samples "
          f"(class0={int((y < 0).sum())}, class1={int((y > 0).sum())})")

    # --- ideal (exact) ----------------------------------------------------
    z_ideal = C.statevector_confidences(model, X)
    if ("ideal", 0) not in done:
        write_rows(samples_csv, C.confidence_rows(
            environment=env, depth=depth, fold=fold,
            condition="ideal", run_index=0, y=y, z=z_ideal, shots=float("nan")))
    acc_ideal = C.accuracy_from_z(z_ideal, y)
    print(f"[ideal] accuracy={acc_ideal:.4f}  sep={C.class_separation(z_ideal, y):.3f}")

    # --- shot-only (classical, no QPU) ------------------------------------
    acc_shot = float("nan")
    if args.skip_shot:
        print("[shot ] skipped (--skip-shot); regenerate at the chosen shots later")
    else:
        z_shot = C.shot_confidences(z_ideal, shots=args.shots, rng=rng)
        if ("shot", 0) not in done:
            write_rows(samples_csv, C.confidence_rows(
                environment=env, depth=depth, fold=fold,
                condition="shot", run_index=0, y=y, z=z_shot, shots=args.shots))
        acc_shot = C.accuracy_from_z(z_shot, y)
        print(f"[shot ] accuracy={acc_shot:.4f}  sep={C.class_separation(z_shot, y):.3f}  "
              f"(boundary shot-std={C.shot_noise_std(0.0, args.shots):.4f})")

    # --- hardware (IQM Spark) ---------------------------------------------
    if args.with_hardware:
        hw_model, hw_estimator = build_iqm_estimator_model(
            backend, ansatz_fn(env),
            num_qubits=args.num_qubits, depth=depth, shots=args.shots,
            optimization_level=args.optimization_level,
            seed_transpiler=args.seed_transpiler,
            max_circuits_per_job=args.max_circuits_per_job,
        )
        load_checkpoint_connector(
            hw_model,
            cv_weight_path(depth, env, fold, condition=args.condition, root=root),
        )
        for r in range(args.repeats):
            if ("hw", r) in done:
                print(f"[hw r{r}] already recorded, skipping")
                continue
            # Interleave classes so slow drift cannot correlate with a class.
            order, inverse = C.interleave_by_class(y)
            z_hw = C.hardware_confidences(hw_model, hw_estimator, X[order])[inverse]
            write_rows(samples_csv, C.confidence_rows(
                environment=env, depth=depth, fold=fold,
                condition="hw", run_index=r, y=y, z=z_hw, shots=args.shots))
            print(f"[hw r{r}] accuracy={C.accuracy_from_z(z_hw, y):.4f}  "
                  f"sep={C.class_separation(z_hw, y):.3f}")
        hw_estimator.print_timing_summary()

    # --- summary + noise-channel fit --------------------------------------
    df = read_csv_or_empty(samples_csv)
    summary_rows = []
    for cond in sorted(df["Condition"].unique()):
        sub = df[df["Condition"] == cond]
        s = C.summarize_condition(sub["z"].to_numpy(), sub["TrueLabel"].to_numpy())
        summary_rows.append({"Environment": env, "Depth": depth,
                             "Fold": fold, "Condition": cond,
                             "n_rows": len(sub), **s})
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\nWrote {summary_csv}")

    hw_df = df[df["Condition"] == "hw"]
    if not hw_df.empty:
        z_hw_mean = hw_df.groupby("SampleID")["z"].mean().reindex(range(len(y))).to_numpy()
        # Paired statevector-vs-QPU comparison: affine channel (lambda, b, R2)
        # plus per-class mean shift and per-class boundary-crossing change.
        cmp = C.compare_conditions(z_ideal, z_hw_mean, y)
        fit = cmp.fit
        pd.DataFrame([{
            "Environment": env, "Depth": depth, "Fold": fold,
            "lambda": fit.lam, "b": fit.b, "R2": fit.r2, "n": fit.n,
            "acc_ideal": acc_ideal, "acc_shot": acc_shot,
            "acc_hw": cmp.accuracy_test, "acc_pred": cmp.accuracy_predicted,
            "delta_mu_class0": cmp.delta_mu_class0,
            "delta_mu_class1": cmp.delta_mu_class1,
            "delta_error_class0": cmp.delta_error_class0,
            "delta_error_class1": cmp.delta_error_class1,
        }]).to_csv(noisefit_csv, index=False)
        print(f"[noise-fit] lambda={fit.lam:.3f} b={fit.b:+.3f} R2={fit.r2:.3f}")
        print(f"            acc_hw={cmp.accuracy_test:.4f}  "
              f"acc_pred(lambda,b)={cmp.accuracy_predicted:.4f}")
        print(f"            d_mu=({cmp.delta_mu_class0:+.3f}, {cmp.delta_mu_class1:+.3f})  "
              f"d_err=({cmp.delta_error_class0:+.3f}, {cmp.delta_error_class1:+.3f})")
        print(f"Wrote {noisefit_csv}")

    print(f"\nSamples CSV: {samples_csv}")


def main() -> None:
    args = parse_args()
    root = find_project_root(ROOT)
    out_dir = Path(args.out_dir) if args.out_dir else root / DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    # Connect once and reuse the backend across every config.
    backend = (connect_to_iqm_backend(args.iqm_url, token=args.iqm_token)
               if args.with_hardware else None)

    configs = [(env, depth, fold)
               for env in args.env for depth in args.depth for fold in args.fold]
    print(f"Sweeping {len(configs)} config(s): "
          f"envs={args.env} depths={args.depth} folds={args.fold}")

    failures: list[tuple[str, str]] = []
    for env, depth, fold in configs:
        try:
            run_one_config(args, root, out_dir, backend, env, depth, fold)
        except Exception as exc:  # keep sweeping; report at the end
            tag = f"{env}_d{depth}_fold{fold}"
            print(f"[FAILED] {tag}: {exc}")
            failures.append((tag, str(exc)))

    if failures:
        print(f"\n{len(failures)}/{len(configs)} config(s) failed:")
        for tag, msg in failures:
            print(f"  {tag}: {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
