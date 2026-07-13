#!/usr/bin/env python3
"""Validate that the repository is ready to launch star-only QPU pilots."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import (  # noqa: E402
    DEFAULT_STAR_ANSATZES,
    STAR_ANSATZ_NAME,
    star_ansatz,
    star_ansatz_registry,
    star_param_count,
)
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402

DEFAULT_DEPTHS = [2, 4, 6]
REQUIRED_SCRIPTS = (
    "scripts/run_iqm_mw_pilot.py",
    "scripts/run_iqm_meyer_wallach.py",
    "scripts/run_iqm_kl_pilot.py",
    "scripts/run_iqm_kl_expressibility.py",
    "scripts/pilot_state_fidelity.py",
    "scripts/run_star_mw_study.sh",
    "scripts/run_star_kl_study.sh",
    "scripts/run_star_fidelity_study.sh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check star-only QPU test readiness")
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument(
        "--require-fidelity-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if Star CV checkpoints are missing (needed for fidelity pilot).",
    )
    return parser.parse_args()


def main() -> int:
    ensure_importable()
    args = parse_args()
    project_root = find_project_root(ROOT)
    errors: list[str] = []
    warnings: list[str] = []

    if not (project_root / "star.py").is_file():
        errors.append(f"Missing star ansatz module: {project_root / 'star.py'}")

    for rel_path in REQUIRED_SCRIPTS:
        if not (project_root / rel_path).is_file():
            errors.append(f"Missing runner: {rel_path}")

    registry = star_ansatz_registry(root=project_root)
    if set(registry) != set(DEFAULT_STAR_ANSATZES):
        errors.append(
            f"Unexpected star registry keys: {sorted(registry)} "
            f"(expected {list(DEFAULT_STAR_ANSATZES)})"
        )

    for depth in args.depth:
        circuit = star_ansatz(5, depth, root=project_root)
        expected = star_param_count(5, depth, root=project_root)
        actual = len(circuit.parameters)
        if actual != expected:
            errors.append(
                f"star_ansatz depth={depth}: expected {expected} params, got {actual}"
            )

    if not os.environ.get("IQM_TOKEN", "").strip():
        warnings.append("IQM_TOKEN is not set (required before submitting QPU jobs).")

    if args.require_fidelity_weights:
        from qbanknote.weights import metric_weight_path

        missing_weights: list[str] = []
        for depth in args.depth:
            weight_path = metric_weight_path(
                depth,
                "star",
                args.fold,
                epoch=args.epoch,
                root=project_root,
            )
            if not weight_path.is_file():
                missing_weights.append(str(weight_path.relative_to(project_root)))
        if missing_weights:
            warnings.append(
                "Fidelity pilot needs Star CV checkpoints. Missing:\n  - "
                + "\n  - ".join(missing_weights)
            )
            warnings.append(
                "Train Star weights in cross_validation (Models/Training) or copy "
                "checkpoints into cross_validation/Models/Weights/depth <d>/Star/."
            )

    print(f"Project root: {project_root}")
    print(f"Star ansatz:  {STAR_ANSATZ_NAME}")
    print(f"Depths:       {args.depth}")
    print(f"Registry:     {sorted(registry)}")

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"  - {item}")

    if errors:
        print("\nErrors:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print("\nReady for star-only MW and KL pilots.")
    if args.require_fidelity_weights and warnings:
        print("Fidelity pilot will fail until Star checkpoints are available.")
    elif not warnings:
        print("Ready for star-only fidelity pilot as well.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
