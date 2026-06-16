from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiment_lib import (
    ANSATZ_NAMES,
    append_csv_row,
    build_run_dir,
    completed_task_keys,
    compute_hardware_row,
    compute_statevector_row,
    connect_to_iqm_backend,
    iter_hardware_tasks,
    load_phase_spec,
    read_csv_or_empty,
    summarize_results,
    timestamp_run_id,
    write_manifest,
)

RETRYABLE_HARDWARE_ERROR_MARKERS = (
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "connection refused",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "httpsconnectionpool",
    "max retries exceeded",
    "batch job failed",
    "failed batch error",
)


def is_retryable_hardware_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in RETRYABLE_HARDWARE_ERROR_MARKERS)


def retry_wait_seconds(
    attempt: int,
    *,
    initial_wait_seconds: float,
    max_wait_seconds: float,
) -> float:
    return min(initial_wait_seconds * (2 ** max(attempt - 1, 0)), max_wait_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IQM Spark ansatz evaluation")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "experiment_config.toml"),
    )
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    parser.add_argument("--depth", type=int, choices=(2, 4, 6), required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--iqm-token", default=None)
    parser.add_argument("--shots", type=int, nargs="*", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--statevector-only", action="store_true")
    parser.add_argument("--hardware-retries", type=int, default=6)
    parser.add_argument("--retry-wait-seconds", type=float, default=60.0)
    parser.add_argument("--retry-max-wait-seconds", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_phase_spec(
        args.config,
        phase=args.phase,
        depth=args.depth,
        shots_override=args.shots,
        repeats_override=args.repeats,
        run_iqm_hardware_override=False if args.statevector_only else None,
    )

    run_id = args.run_id or timestamp_run_id(f"{spec.phase}_depth{spec.depth}")
    run_dir = build_run_dir(spec, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir / "run_manifest.json", spec)

    statevector_csv = run_dir / "statevector_results.csv"
    runs_csv = run_dir / "run_level_results.csv"
    summary_csv = run_dir / "summary_comparison.csv"

    statevector_df = read_csv_or_empty(statevector_csv)
    run_df = read_csv_or_empty(runs_csv)

    for fold in spec.folds:
        for ansatz_name in ANSATZ_NAMES:
            if not statevector_df.empty:
                existing = statevector_df[
                    (statevector_df["fold"] == int(fold)) & (statevector_df["ansatz"] == ansatz_name)
                ]
                if not existing.empty:
                    continue

            print(f"Statevector | depth={spec.depth} fold={fold} ansatz={ansatz_name}")
            row = compute_statevector_row(spec, fold, ansatz_name)
            append_csv_row(statevector_csv, row)
            statevector_df = read_csv_or_empty(statevector_csv)
            summary_df = summarize_results(spec, statevector_df=statevector_df, run_df=run_df)
            summary_df.to_csv(summary_csv, index=False)

    if spec.run_iqm_hardware:
        print(
            f"Connecting to IQM for phase={spec.phase}, depth={spec.depth}, "
            f"shots={list(spec.shots)}, repeats={spec.repeats}"
        )
        backend = connect_to_iqm_backend(spec.iqm_url, token=args.iqm_token)
        done = completed_task_keys(run_df)
        tasks = iter_hardware_tasks(spec)

        for index, task in enumerate(tasks, start=1):
            task_key = (task["fold"], task["ansatz"], task["shots"], task["repeat_index"])
            if task_key in done:
                continue

            print(
                f"Hardware {index}/{len(tasks)} | depth={spec.depth} fold={task['fold']} "
                f"ansatz={task['ansatz']} shots={task['shots']} repeat={task['repeat_index']}"
            )
            max_attempts = args.hardware_retries + 1
            for attempt in range(1, max_attempts + 1):
                try:
                    row = compute_hardware_row(
                        spec,
                        backend,
                        fold=int(task["fold"]),
                        ansatz_name=str(task["ansatz"]),
                        shots=int(task["shots"]),
                        repeat_index=int(task["repeat_index"]),
                    )
                    break
                except KeyboardInterrupt:
                    print(
                        "\nInterrupted during hardware evaluation. "
                        "The current task was not written to the results CSV and will be retried on resume."
                    )
                    raise
                except Exception as exc:
                    retryable = is_retryable_hardware_error(exc)
                    if attempt >= max_attempts or not retryable:
                        print(f"Hardware task failed and was not recorded: {exc}")
                        raise SystemExit(1) from exc

                    wait_seconds = retry_wait_seconds(
                        attempt,
                        initial_wait_seconds=args.retry_wait_seconds,
                        max_wait_seconds=args.retry_max_wait_seconds,
                    )
                    print(
                        f"Transient hardware failure on attempt {attempt}/{max_attempts}: {exc}\n"
                        f"Retrying in {wait_seconds:.1f}s..."
                    )
                    time.sleep(wait_seconds)
            append_csv_row(runs_csv, row)
            run_df = read_csv_or_empty(runs_csv)
            done.add(task_key)
            summary_df = summarize_results(spec, statevector_df=statevector_df, run_df=run_df)
            summary_df.to_csv(summary_csv, index=False)
    else:
        print("Skipping IQM hardware because statevector-only mode is active.")

    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
