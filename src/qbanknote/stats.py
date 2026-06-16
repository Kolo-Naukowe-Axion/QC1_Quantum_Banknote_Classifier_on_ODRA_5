"""Statistical helpers for CV hardware metric experiments."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd

from qbanknote.evaluation import ANSATZ_NAMES, PhaseSpec

__all__ = [
    "average_ranks",
    "wilcoxon_signed_rank_exact",
    "sign_test_exact",
    "compute_paired_fold_differences",
    "compute_paired_tests",
    "compute_shot_stability",
    "choose_shot_from_pilot",
    "recommended_repeats_from_pilot",
    "summarize_across_folds",
    "select_protocol_from_pilot",
    "analyze_final_summary",
]


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    next_rank = 1
    while start < len(order):
        end = start
        while end + 1 < len(order) and math.isclose(
            values[order[end + 1]], values[order[start]], rel_tol=0.0, abs_tol=1e-12
        ):
            end += 1
        avg_rank = (next_rank + next_rank + (end - start)) / 2.0
        for pos in range(start, end + 1):
            ranks[order[pos]] = avg_rank
        next_rank += end - start + 1
        start = end + 1
    return ranks


def wilcoxon_signed_rank_exact(differences: list[float]) -> dict[str, float | int | None]:
    diffs = [float(value) for value in differences if not math.isclose(float(value), 0.0, abs_tol=1e-12)]
    if not diffs:
        return {
            "n_nonzero": 0,
            "statistic": 0.0,
            "pvalue": 1.0,
            "rank_biserial": 0.0,
            "median_difference": 0.0,
        }

    ranks = average_ranks([abs(v) for v in diffs])
    total_rank = float(sum(ranks))
    positive_rank_sum = float(sum(rank for diff, rank in zip(diffs, ranks) if diff > 0))
    negative_rank_sum = total_rank - positive_rank_sum
    statistic = float(min(positive_rank_sum, negative_rank_sum))
    rank_biserial = float((positive_rank_sum - negative_rank_sum) / total_rank) if total_rank else 0.0

    distribution = []
    for signs in itertools.product((0, 1), repeat=len(ranks)):
        signed_positive_sum = sum(rank for sign, rank in zip(signs, ranks) if sign == 1)
        distribution.append(min(signed_positive_sum, total_rank - signed_positive_sum))
    pvalue = sum(1 for value in distribution if value <= statistic + 1e-12) / len(distribution)

    return {
        "n_nonzero": int(len(diffs)),
        "statistic": statistic,
        "pvalue": float(pvalue),
        "rank_biserial": rank_biserial,
        "median_difference": float(np.median(diffs)),
    }


def sign_test_exact(differences: list[float]) -> dict[str, float | int]:
    diffs = [float(value) for value in differences if not math.isclose(float(value), 0.0, abs_tol=1e-12)]
    n = len(diffs)
    if n == 0:
        return {"n_nonzero": 0, "positive": 0, "negative": 0, "pvalue": 1.0}
    positive = sum(1 for value in diffs if value > 0)
    negative = n - positive
    tail = min(positive, negative)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    pvalue = min(1.0, 2.0 * probability)
    return {
        "n_nonzero": int(n),
        "positive": int(positive),
        "negative": int(negative),
        "pvalue": float(pvalue),
    }


def summarize_across_folds(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    group_cols = ["phase", "depth", "ansatz", "eval_shots"]
    rows = []
    for keys, group in summary_df.groupby(group_cols, dropna=False):
        phase, depth, ansatz, eval_shots = keys
        rows.append(
            {
                "phase": phase,
                "depth": depth,
                "ansatz": ansatz,
                "eval_shots": eval_shots,
                "fold_count": int(group["fold"].nunique()),
                "statevector_accuracy_mean": float(group["statevector_accuracy"].mean()),
                "statevector_f1_mean": float(group["statevector_f1"].mean()),
                "iqm_mean_accuracy_mean": float(group["iqm_mean_accuracy"].mean()),
                "iqm_mean_accuracy_std": float(group["iqm_mean_accuracy"].std(ddof=1)),
                "iqm_mean_f1_mean": float(group["iqm_mean_f1"].mean()),
                "iqm_mean_f1_std": float(group["iqm_mean_f1"].std(ddof=1)),
                "iqm_std_accuracy_mean": float(group["iqm_std_accuracy"].mean()),
                "iqm_std_f1_mean": float(group["iqm_std_f1"].mean()),
                "mean_gap_accuracy": float(
                    (group["statevector_accuracy"] - group["iqm_mean_accuracy"]).mean()
                ),
                "mean_gap_f1": float((group["statevector_f1"] - group["iqm_mean_f1"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_paired_fold_differences(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["phase", "depth", "eval_shots", "fold"]
    for keys, group in summary_df.groupby(group_cols, dropna=False):
        phase, depth, eval_shots, fold = keys
        if set(group["ansatz"]) != set(ANSATZ_NAMES):
            continue
        odra = group[group["ansatz"] == "odra"].iloc[0]
        simulator = group[group["ansatz"] == "simulator"].iloc[0]
        rows.append(
            {
                "phase": phase,
                "depth": depth,
                "eval_shots": eval_shots,
                "fold": fold,
                "iqm_accuracy_diff_odra_minus_simulator": float(
                    odra["iqm_mean_accuracy"] - simulator["iqm_mean_accuracy"]
                ),
                "iqm_f1_diff_odra_minus_simulator": float(
                    odra["iqm_mean_f1"] - simulator["iqm_mean_f1"]
                ),
                "gap_accuracy_diff_odra_minus_simulator": float(
                    (odra["statevector_accuracy"] - odra["iqm_mean_accuracy"])
                    - (simulator["statevector_accuracy"] - simulator["iqm_mean_accuracy"])
                ),
                "gap_f1_diff_odra_minus_simulator": float(
                    (odra["statevector_f1"] - odra["iqm_mean_f1"])
                    - (simulator["statevector_f1"] - simulator["iqm_mean_f1"])
                ),
                "iqm_std_accuracy_diff_odra_minus_simulator": float(
                    odra["iqm_std_accuracy"] - simulator["iqm_std_accuracy"]
                ),
                "iqm_std_f1_diff_odra_minus_simulator": float(
                    odra["iqm_std_f1"] - simulator["iqm_std_f1"]
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_paired_tests(diffs_df: pd.DataFrame) -> pd.DataFrame:
    if diffs_df.empty:
        return pd.DataFrame()

    rows = []
    metrics = [
        "iqm_accuracy_diff_odra_minus_simulator",
        "iqm_f1_diff_odra_minus_simulator",
        "gap_accuracy_diff_odra_minus_simulator",
        "gap_f1_diff_odra_minus_simulator",
        "iqm_std_accuracy_diff_odra_minus_simulator",
        "iqm_std_f1_diff_odra_minus_simulator",
    ]
    group_cols = ["phase", "depth", "eval_shots"]
    for keys, group in diffs_df.groupby(group_cols, dropna=False):
        phase, depth, eval_shots = keys
        for metric in metrics:
            values = [float(v) for v in group[metric].tolist()]
            wilcoxon = wilcoxon_signed_rank_exact(values)
            sign = sign_test_exact(values)
            rows.append(
                {
                    "phase": phase,
                    "depth": depth,
                    "eval_shots": eval_shots,
                    "metric": metric,
                    "fold_count": int(len(values)),
                    "mean_difference": float(np.mean(values)),
                    "median_difference": float(np.median(values)),
                    "wilcoxon_statistic": wilcoxon["statistic"],
                    "wilcoxon_pvalue": wilcoxon["pvalue"],
                    "wilcoxon_rank_biserial": wilcoxon["rank_biserial"],
                    "sign_test_pvalue": sign["pvalue"],
                    "positive_differences": sign["positive"],
                    "negative_differences": sign["negative"],
                }
            )
    return pd.DataFrame(rows)


def compute_shot_stability(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    shots_values = sorted(int(v) for v in summary_df["eval_shots"].dropna().unique())
    if len(shots_values) < 2:
        return pd.DataFrame(), pd.DataFrame()

    for previous_shot, current_shot in zip(shots_values[:-1], shots_values[1:]):
        prev_group = summary_df[summary_df["eval_shots"] == previous_shot]
        curr_group = summary_df[summary_df["eval_shots"] == current_shot]
        merge_cols = ["phase", "depth", "fold", "ansatz"]
        merged = prev_group.merge(
            curr_group,
            on=merge_cols,
            how="inner",
            suffixes=("_prev", "_curr"),
        )
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "phase": row.phase,
                    "depth": row.depth,
                    "fold": row.fold,
                    "ansatz": row.ansatz,
                    "previous_shot": previous_shot,
                    "current_shot": current_shot,
                    "abs_change_accuracy": abs(row.iqm_mean_accuracy_curr - row.iqm_mean_accuracy_prev),
                    "abs_change_f1": abs(row.iqm_mean_f1_curr - row.iqm_mean_f1_prev),
                }
            )

    detailed = pd.DataFrame(rows)
    if detailed.empty:
        return detailed, pd.DataFrame()

    aggregate_rows = []
    for keys, group in detailed.groupby(
        ["phase", "depth", "previous_shot", "current_shot"], dropna=False
    ):
        phase, depth, previous_shot, current_shot = keys
        aggregate_rows.append(
            {
                "phase": phase,
                "depth": depth,
                "previous_shot": previous_shot,
                "current_shot": current_shot,
                "mean_abs_change_accuracy": float(group["abs_change_accuracy"].mean()),
                "max_abs_change_accuracy": float(group["abs_change_accuracy"].max()),
                "mean_abs_change_f1": float(group["abs_change_f1"].mean()),
                "max_abs_change_f1": float(group["abs_change_f1"].max()),
            }
        )
    return detailed, pd.DataFrame(aggregate_rows)


def choose_shot_from_pilot(summary_df: pd.DataFrame, spec: PhaseSpec) -> int:
    _, aggregate = compute_shot_stability(summary_df)
    if aggregate.empty:
        return int(spec.shots[-1])

    for row in aggregate.sort_values("current_shot").itertuples(index=False):
        if (
            row.max_abs_change_accuracy <= spec.delta_accuracy
            and row.max_abs_change_f1 <= spec.delta_f1
        ):
            return int(row.current_shot)
    return int(max(spec.shots))


def recommended_repeats_from_pilot(
    summary_df: pd.DataFrame, chosen_shot: int, spec: PhaseSpec
) -> dict[str, int]:
    shot_rows = summary_df[
        (summary_df["eval_shots"] == chosen_shot) & (summary_df["completed_repeats"] > 0)
    ]
    if shot_rows.empty:
        return {"recommended_repeats_accuracy": spec.repeats, "recommended_repeats_f1": spec.repeats}

    conservative_std_acc = float(shot_rows["iqm_std_accuracy"].max())
    conservative_std_f1 = float(shot_rows["iqm_std_f1"].max())

    def _required_repeats(std_value: float, half_width: float) -> int:
        if half_width <= 0 or std_value <= 0:
            return 1
        return int(math.ceil(((1.96 * std_value) / half_width) ** 2))

    return {
        "recommended_repeats_accuracy": _required_repeats(
            conservative_std_acc, spec.target_half_width_accuracy
        ),
        "recommended_repeats_f1": _required_repeats(
            conservative_std_f1, spec.target_half_width_f1
        ),
    }


def select_protocol_from_pilot(
    summary_df: pd.DataFrame,
    spec: PhaseSpec,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    if summary_df.empty:
        raise ValueError("Pilot summary is empty; cannot select protocol.")

    if require_complete:
        incomplete = summary_df[summary_df["completed_repeats"] < spec.repeats]
        if not incomplete.empty:
            missing = (
                incomplete[["fold", "ansatz", "eval_shots", "completed_repeats"]]
                .sort_values(["fold", "ansatz", "eval_shots"])
                .to_dict(orient="records")
            )
            raise ValueError(
                f"Pilot run is incomplete; expected {spec.repeats} repeats per task. "
                f"Incomplete entries: {missing}"
            )

    detailed, aggregate = compute_shot_stability(summary_df)
    chosen_shot = choose_shot_from_pilot(summary_df, spec)
    repeat_recommendations = recommended_repeats_from_pilot(summary_df, chosen_shot, spec)
    chosen_repeats = max(
        repeat_recommendations["recommended_repeats_accuracy"],
        repeat_recommendations["recommended_repeats_f1"],
    )
    return {
        "depth": spec.depth,
        "delta_accuracy": spec.delta_accuracy,
        "delta_f1": spec.delta_f1,
        "target_half_width_accuracy": spec.target_half_width_accuracy,
        "target_half_width_f1": spec.target_half_width_f1,
        "chosen_shot": chosen_shot,
        "recommended_repeats_accuracy": repeat_recommendations["recommended_repeats_accuracy"],
        "recommended_repeats_f1": repeat_recommendations["recommended_repeats_f1"],
        "chosen_repeats": chosen_repeats,
        "shot_stability": detailed,
        "shot_stability_aggregate": aggregate,
    }


def analyze_final_summary(summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if summary_df.empty:
        raise ValueError("summary_comparison.csv is empty; nothing to analyze.")

    ansatz_level = summarize_across_folds(summary_df)
    paired_differences = compute_paired_fold_differences(summary_df)
    paired_tests = compute_paired_tests(paired_differences)
    shot_stability, shot_stability_aggregate = compute_shot_stability(summary_df)
    return {
        "ansatz_level_summary": ansatz_level,
        "paired_fold_differences": paired_differences,
        "paired_tests": paired_tests,
        "shot_stability": shot_stability,
        "shot_stability_aggregate": shot_stability_aggregate,
    }
