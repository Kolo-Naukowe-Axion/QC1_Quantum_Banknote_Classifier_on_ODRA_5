#!/usr/bin/env python3
"""Analyze a completed final (or pilot) metric-test run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.evaluation import read_csv_or_empty  # noqa: E402
from qbanknote.paths import ensure_importable  # noqa: E402
from qbanknote.stats import analyze_final_summary, write_analysis_artifacts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a completed metric-test run directory")
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    ensure_importable()
    args = parse_args()
    run_dir = Path(args.run_dir)

    summary_df = read_csv_or_empty(run_dir / "summary_comparison.csv")
    if summary_df.empty:
        raise SystemExit(f"No summary_comparison.csv found in {run_dir}")

    analysis = analyze_final_summary(summary_df)
    write_analysis_artifacts(run_dir, analysis)
    print(f"Wrote analysis outputs to {run_dir}")


if __name__ == "__main__":
    main()
