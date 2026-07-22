#!/usr/bin/env python3
"""Run CV classification metric test on IQM Spark with resumable CSV output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.evaluation import (  # noqa: E402
    build_run_dir,
    count_hardware_tasks,
    count_statevector_tasks,
    load_phase_spec,
    run_cv_experiment,
    timestamp_run_id,
)
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402
from qbanknote.progress import make_print_callback  # noqa: E402

DEFAULT_CONFIG = ROOT / "evaluation_and_comparison/iqm_spark/iqm_metric_test_config.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IQM Spark CV classification metric test")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    parser.add_argument(
        "--depth",
        type=int,
        choices=(2, 4, 6),
        nargs="+",
        required=True,
        metavar="DEPTH",
        help="One or more circuit depths, e.g. '--depth 2 4 6'. Each is run in turn "
        "into the same run directory.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--ansatz",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "Subset of ansatze to execute (e.g. 'star', or 'odra simulator'). "
            "Accepts keys (odra/simulator/star) or canonical names. Defaults to all. "
            "Reuse the same --run-id to append a subset (e.g. star) to an existing run."
        ),
    )
    parser.add_argument("--iqm-token", default=None)
    parser.add_argument("--shots", type=int, nargs="*", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--statevector-only", action="store_true")
    parser.add_argument("--hardware-retries", type=int, default=6)
    parser.add_argument("--retry-wait-seconds", type=float, default=60.0)
    parser.add_argument("--retry-max-wait-seconds", type=float, default=600.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_importable()
    args = parse_args()
    project_root = find_project_root(ROOT)

    depths = list(dict.fromkeys(args.depth))  # dedupe, preserve order

    # One shared run id (and run directory) across all depths; every depth's rows
    # land in the same CSVs, distinguished by their `depth` column.
    depth_tag = "-".join(str(d) for d in depths)
    run_id = args.run_id or timestamp_run_id(f"{args.phase}_depth{depth_tag}")
    progress_callback = None if args.quiet else make_print_callback()

    iqm_backend = None
    for depth in depths:
        spec = load_phase_spec(
            args.config,
            phase=args.phase,
            depth=depth,
            shots_override=args.shots,
            repeats_override=args.repeats,
            run_iqm_hardware_override=False if args.statevector_only else None,
            ansatze_override=args.ansatz,
        )
        run_dir = build_run_dir(spec, run_id, root=project_root)

        if not args.quiet:
            print(f"=== depth {depth} ===")
            print(f"Run directory: {run_dir}")
            print(
                f"Phase={spec.phase}, depth={spec.depth}, folds={spec.folds}, "
                f"shots={spec.shots}, repeats={spec.repeats}, hardware={spec.run_iqm_hardware}"
            )
            print(f"Ansatze (executing): {spec.ansatze}")
            print(
                f"Tasks: statevector={count_statevector_tasks(spec)}, "
                f"hardware={count_hardware_tasks(spec)}"
            )

        # Connect once and reuse the backend across depths (avoids re-prompting for
        # a token and re-establishing a session per depth).
        if spec.run_iqm_hardware and iqm_backend is None:
            from qbanknote.iqm import connect_to_iqm_backend

            iqm_backend = connect_to_iqm_backend(spec.iqm_url, token=args.iqm_token)

        run_cv_experiment(
            spec,
            run_dir,
            iqm_backend=iqm_backend,
            hardware_retries=args.hardware_retries,
            retry_wait_seconds_initial=args.retry_wait_seconds,
            retry_wait_seconds_max=args.retry_max_wait_seconds,
            root=project_root,
            verbose=not args.quiet,
            progress_callback=progress_callback,
        )

        if not args.quiet:
            print(f"Finished depth {depth}: {run_dir}")


if __name__ == "__main__":
    main()
