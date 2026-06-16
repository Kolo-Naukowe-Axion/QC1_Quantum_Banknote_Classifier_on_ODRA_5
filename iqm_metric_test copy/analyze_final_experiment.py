from __future__ import annotations

import argparse
from pathlib import Path

from experiment_lib import (
    compute_paired_fold_differences,
    compute_paired_tests,
    compute_shot_stability,
    read_csv_or_empty,
    summarize_across_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a completed ansatz experiment run")
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary_df = read_csv_or_empty(run_dir / "summary_comparison.csv")
    if summary_df.empty:
        raise FileNotFoundError(f"No summary_comparison.csv found in {run_dir}")

    ansatz_level = summarize_across_folds(summary_df)
    paired_differences = compute_paired_fold_differences(summary_df)
    paired_tests = compute_paired_tests(paired_differences)
    shot_stability, shot_stability_aggregate = compute_shot_stability(summary_df)

    ansatz_level.to_csv(run_dir / "ansatz_level_summary.csv", index=False)
    paired_differences.to_csv(run_dir / "paired_fold_differences.csv", index=False)
    paired_tests.to_csv(run_dir / "paired_tests.csv", index=False)

    if not shot_stability.empty:
        shot_stability.to_csv(run_dir / "shot_stability.csv", index=False)
    if not shot_stability_aggregate.empty:
        shot_stability_aggregate.to_csv(run_dir / "shot_stability_aggregate.csv", index=False)

    print(f"Wrote analysis outputs to {run_dir}")


if __name__ == "__main__":
    main()
