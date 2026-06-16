from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_lib import (
    choose_shot_from_pilot,
    compute_shot_stability,
    load_phase_spec,
    read_csv_or_empty,
    recommended_repeats_from_pilot,
    summarize_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select frozen protocol from a pilot run")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "experiment_config.toml"),
    )
    parser.add_argument("--depth", type=int, choices=(2, 4, 6), required=False)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    statevector_csv = run_dir / "statevector_results.csv"
    runs_csv = run_dir / "run_level_results.csv"
    summary_csv = run_dir / "summary_comparison.csv"
    statevector_df = read_csv_or_empty(statevector_csv)
    run_df = read_csv_or_empty(runs_csv)
    if statevector_df.empty:
        raise FileNotFoundError(f"No statevector results found at {statevector_csv}")

    depth_source = statevector_df if not statevector_df.empty else run_df
    if depth_source.empty:
        raise FileNotFoundError(f"No pilot inputs found in {run_dir}")

    depth = args.depth if args.depth is not None else int(depth_source["depth"].iloc[0])
    spec = load_phase_spec(args.config, phase="pilot", depth=depth)
    summary_df = summarize_results(spec, statevector_df=statevector_df, run_df=run_df)
    summary_df.to_csv(summary_csv, index=False)

    if summary_df.empty:
        raise FileNotFoundError(f"Could not build pilot summary at {summary_csv}")

    incomplete = summary_df[summary_df["completed_repeats"] < spec.repeats]
    if not incomplete.empty:
        missing = (
            incomplete[["fold", "ansatz", "eval_shots", "completed_repeats"]]
            .sort_values(["fold", "ansatz", "eval_shots"])
            .to_dict(orient="records")
        )
        raise SystemExit(
            "Pilot run is incomplete; no recommendation was generated.\n"
            f"Expected {spec.repeats} repeats per (fold, ansatz, shots).\n"
            f"Incomplete entries: {missing}"
        )

    detailed, aggregate = compute_shot_stability(summary_df)
    if not detailed.empty:
        detailed.to_csv(run_dir / "shot_stability.csv", index=False)
    if not aggregate.empty:
        aggregate.to_csv(run_dir / "shot_stability_aggregate.csv", index=False)

    chosen_shot = choose_shot_from_pilot(summary_df, spec)
    repeat_recommendations = recommended_repeats_from_pilot(summary_df, chosen_shot, spec)
    chosen_repeats = max(
        repeat_recommendations["recommended_repeats_accuracy"],
        repeat_recommendations["recommended_repeats_f1"],
    )

    report = {
        "pilot_run_dir": str(run_dir),
        "depth": depth,
        "delta_accuracy": spec.delta_accuracy,
        "delta_f1": spec.delta_f1,
        "target_half_width_accuracy": spec.target_half_width_accuracy,
        "target_half_width_f1": spec.target_half_width_f1,
        "chosen_shot": chosen_shot,
        "recommended_repeats_accuracy": repeat_recommendations["recommended_repeats_accuracy"],
        "recommended_repeats_f1": repeat_recommendations["recommended_repeats_f1"],
        "chosen_repeats": chosen_repeats,
    }

    report_path = run_dir / "protocol_recommendation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
