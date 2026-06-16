#!/usr/bin/env python3
"""Select frozen shot/repeat protocol from a completed pilot run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.evaluation import (  # noqa: E402
    load_phase_spec,
    read_csv_or_empty,
    summarize_results,
)
from qbanknote.paths import ensure_importable  # noqa: E402
from qbanknote.stats import select_protocol_from_pilot, write_protocol_artifacts  # noqa: E402

DEFAULT_CONFIG = ROOT / "evaluation_and_comparison/iqm_spark/iqm_metric_test_config.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select frozen protocol from a pilot run directory")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--depth", type=int, choices=(2, 4, 6), default=None)
    return parser.parse_args()


def main() -> None:
    ensure_importable()
    args = parse_args()
    run_dir = Path(args.run_dir)

    statevector_csv = run_dir / "statevector_results.csv"
    runs_csv = run_dir / "run_level_results.csv"
    summary_csv = run_dir / "summary_comparison.csv"

    statevector_df = read_csv_or_empty(statevector_csv)
    run_df = read_csv_or_empty(runs_csv)
    if statevector_df.empty:
        raise SystemExit(f"No statevector results found at {statevector_csv}")

    depth_source = statevector_df if not statevector_df.empty else run_df
    depth = args.depth if args.depth is not None else int(depth_source["depth"].iloc[0])
    spec = load_phase_spec(args.config, phase="pilot", depth=depth)

    summary_df = summarize_results(spec, statevector_df=statevector_df, run_df=run_df)
    summary_df.to_csv(summary_csv, index=False)
    if summary_df.empty:
        raise SystemExit(f"Could not build pilot summary at {summary_csv}")

    protocol = select_protocol_from_pilot(summary_df, spec, require_complete=True)
    protocol["pilot_run_dir"] = str(run_dir.resolve())
    report_path = write_protocol_artifacts(run_dir, protocol)

    report = {k: v for k, v in protocol.items() if not hasattr(v, "to_csv")}
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"Wrote protocol recommendation to {report_path}")


if __name__ == "__main__":
    main()
