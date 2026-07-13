#!/usr/bin/env python3
"""Top up shot-pilot jobs that failed tolerance using cached fidelity samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import (  # noqa: E402
    odra_ansatz as ansatz_odra,
    simulator_ansatz as ansatz_simulator,
)
from qbanknote.iqm import connect_to_iqm_backend  # noqa: E402
from qbanknote.metrics import (  # noqa: E402
    KlPilotFallbackError,
    choose_kl_shots_by_job,
    compute_kl_shot_stability_by_job,
    count_kl_job_samples,
    failed_kl_shot_pilot_jobs,
    infer_shot_pilot_kl_params,
    kl_depth_seed,
    kl_job_key,
    nested_job_int_map_to_flat,
    read_kl_summary,
    refresh_kl_job_summary_from_fidelities,
    run_iqm_kl_sweep,
)
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402
from qbanknote.progress import make_print_callback  # noqa: E402

DEFAULT_OUTPUT_ROOT = "evaluation_and_comparison/iqm_spark/iqm_kl_outputs"
DEFAULT_DEPTHS = [2, 4, 6]
DEFAULT_ANSATZES = ("ansatz_odra", "ansatz_simulator")
DEFAULT_SHOT_GRID = [512, 1024, 2048, 4096, 8192]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume failed KL shot-pilot jobs: reuse cached samples and collect "
            "additional fidelity pairs before re-evaluating shot tolerance."
        )
    )
    parser.add_argument("--pilot-id", default="kl_pilot_paper")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--depth", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--ansatz", nargs="+", default=list(DEFAULT_ANSATZES))
    parser.add_argument("--shot-grid", type=int, nargs="+", default=DEFAULT_SHOT_GRID)
    parser.add_argument(
        "--pilot-samples",
        type=int,
        default=10,
        help=(
            "Target fidelity pairs per job after top-up (default: 10). "
            "Existing sample_index rows on disk are kept; only missing indices are collected."
        ),
    )
    parser.add_argument(
        "--shot-tolerance",
        type=float,
        default=0.025,
        help="Tolerance for shot selection after top-up.",
    )
    parser.add_argument(
        "--identify-failure-tolerance",
        type=float,
        default=0.02,
        help=(
            "Tolerance for labelling jobs as failed in the report. "
            "When --use-topup-manifest-jobs is set, the manifest job list takes precedence."
        ),
    )
    parser.add_argument(
        "--use-topup-manifest-jobs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse topup_failed_jobs from a previous top-up manifest so sample extensions "
            "keep the same job set even if shot selection later passes (default: true)."
        ),
    )
    parser.add_argument(
        "--topup-jobs-mode",
        choices=("manifest", "not-passing", "identify-failed"),
        default="manifest",
        help=(
            "manifest: use topup_failed_jobs from a prior manifest (default). "
            "not-passing: only jobs with no passing shot step at --baseline-samples. "
            "identify-failed: use --identify-failure-tolerance on the full summary."
        ),
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=None,
        help=(
            "Sample count used to detect not-passing jobs (default: --pilot-samples minus 5 "
            "when pilot_samples > 5, else pilot_samples)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Histogram bins for KL summaries. Default: infer from existing shot pilot (n=3 rows).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="KL smoothing epsilon. Default: infer from existing shot pilot artifacts.",
    )
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--max-circuits-per-job", type=int, default=250)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iqm-token", default=None)
    parser.add_argument(
        "--iqm-url",
        default=os.environ.get("IQM_URL", "https://odra5.e-science.pl/").strip(),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Identify failed jobs and print work estimate without QPU calls.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_shot_pilot_summary(output_root: Path, shot_grid: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for shots in shot_grid:
        run_dir = output_root / "shot_pilot" / f"shots_{shots}"
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing shot pilot directory: {run_dir}")
        frames.append(read_kl_summary(run_dir, stage="shot_pilot"))
    return _concat(frames)


def _resolve_pilot_kl_params(
    args: argparse.Namespace,
    output_root: Path,
    shot_grid: list[int],
    *,
    reference_n_samples: int = 3,
) -> tuple[int, float]:
    if args.n_bins is not None and args.eps is not None:
        return int(args.n_bins), float(args.eps)

    inferred = infer_shot_pilot_kl_params(
        output_root,
        shot_grid,
        reference_n_samples=reference_n_samples,
    )
    n_bins = int(args.n_bins if args.n_bins is not None else inferred["n_bins"])
    eps = float(args.eps if args.eps is not None else inferred["eps"])
    return n_bins, eps


def _refresh_stale_summaries(
    *,
    output_root: Path,
    shot_grid: list[int],
    failed_jobs: set[tuple[str, int]],
    pilot_samples: int,
    n_qubits: int,
    seed: int,
    n_bins: int,
    eps: float,
    quiet: bool,
) -> int:
    refreshed = 0
    for shots in shot_grid:
        run_dir = output_root / "shot_pilot" / f"shots_{shots}"
        fidelities_path = run_dir / "iqm_kl_fidelities.csv"
        summary_path = run_dir / "iqm_kl_results.csv"
        for ansatz, depth in sorted(failed_jobs):
            depth_seed = kl_depth_seed(seed, depth, ansatz)
            if refresh_kl_job_summary_from_fidelities(
                fidelities_path=fidelities_path,
                summary_path=summary_path,
                ansatz=ansatz,
                depth=depth,
                shots=int(shots),
                seed=depth_seed,
                n_qubits=n_qubits,
                n_samples=int(pilot_samples),
                n_bins=n_bins,
                eps=eps,
            ):
                refreshed += 1
    if not quiet and refreshed:
        print(
            f"Refreshed {refreshed} stale KL summary row(s) at n_bins={n_bins}, "
            f"n_samples={pilot_samples} from cached fidelities."
        )
    return refreshed


def _load_topup_jobs_from_manifest(output_root: Path, shot_grid: list[int]) -> set[tuple[str, int]]:
    """Return the failed-job set recorded by a previous top-up run, if any."""
    for shots in reversed(shot_grid):
        manifest_path = output_root / "shot_pilot" / f"shots_{shots}" / "run_manifest.json"
        if not manifest_path.exists():
            continue
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_jobs = data.get("topup_failed_jobs")
        if raw_jobs:
            return {kl_job_key(str(job[0]), int(job[1])) for job in raw_jobs}
    return set()


def _estimate_topup_work(
    output_root: Path,
    shot_grid: list[int],
    jobs: set[tuple[str, int]],
    pilot_samples: int,
) -> tuple[dict[tuple[str, int], dict[int, int]], int]:
    """Return per-job/per-shot cached counts and total new sample slots needed."""
    per_job: dict[tuple[str, int], dict[int, int]] = {}
    total_new = 0
    for job in jobs:
        per_shot: dict[int, int] = {}
        for shots in shot_grid:
            fidelities_path = output_root / "shot_pilot" / f"shots_{shots}" / "iqm_kl_fidelities.csv"
            have = count_kl_job_samples(fidelities_path, job[0], job[1])
            per_shot[shots] = have
            total_new += max(0, int(pilot_samples) - have)
        per_job[job] = per_shot
    return per_job, total_new


def _summary_at_n_samples(summary_df, n_samples: int):
    if summary_df.empty:
        return summary_df
    return summary_df.loc[summary_df["n_samples"] == int(n_samples)].copy()


def _resolve_topup_jobs(
    *,
    shot_summary,
    output_root: Path,
    shot_grid: list[int],
    identify_tolerance: float,
    shot_tolerance: float,
    use_manifest_jobs: bool,
    topup_jobs_mode: str,
    baseline_samples: int | None,
    pilot_samples: int,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    """Choose which (ansatz, depth) jobs to extend and which passed shot selection."""
    all_jobs = {
        kl_job_key(str(row.ansatz), int(row.depth))
        for row in shot_summary.drop_duplicates(["ansatz", "depth"]).itertuples(index=False)
    }

    if topup_jobs_mode == "not-passing":
        baseline = (
            int(baseline_samples)
            if baseline_samples is not None
            else (int(pilot_samples) - 5 if int(pilot_samples) > 5 else int(pilot_samples))
        )
        baseline_summary = _summary_at_n_samples(shot_summary, baseline)
        if baseline_summary.empty:
            raise SystemExit(
                f"No summary rows with n_samples={baseline} in shot pilot; "
                "cannot detect not-passing jobs."
            )
        topup_jobs = failed_kl_shot_pilot_jobs(baseline_summary, tolerance=shot_tolerance)
    elif topup_jobs_mode == "identify-failed":
        topup_jobs = failed_kl_shot_pilot_jobs(shot_summary, tolerance=identify_tolerance)
    elif use_manifest_jobs:
        manifest_jobs = _load_topup_jobs_from_manifest(output_root, shot_grid)
        topup_jobs = manifest_jobs if manifest_jobs else failed_kl_shot_pilot_jobs(
            shot_summary, tolerance=identify_tolerance
        )
    else:
        topup_jobs = failed_kl_shot_pilot_jobs(shot_summary, tolerance=identify_tolerance)

    passing_jobs = all_jobs - topup_jobs
    return topup_jobs, passing_jobs


def _print_topup_work_estimate(
    *,
    jobs: set[tuple[str, int]],
    per_job: dict[tuple[str, int], dict[int, int]],
    pilot_samples: int,
    total_new: int,
) -> None:
    print("\nCached samples per shot level (indices 0..n-1 are reused):")
    for job in sorted(jobs):
        counts = per_job[job]
        min_have = min(counts.values())
        max_have = max(counts.values())
        need = max(0, pilot_samples - min_have)
        print(
            f"  {job[0]} depth={job[1]}: {min_have}-{max_have}/{pilot_samples} "
            f"(+{max(0, pilot_samples - max_have)} to {pilot_samples} on sparsest level)"
        )
        for shots, have in sorted(counts.items()):
            if have < pilot_samples:
                new_indices = pilot_samples - have
                print(f"    shots={shots}: {have}/{pilot_samples} -> collect indices {have}-{pilot_samples - 1} ({new_indices} new)")
    print(f"\nNew fidelity pairs to collect: {total_new} (job x shot settings)")
    if total_new == 0:
        print("Nothing to collect — all jobs already at target sample count.")


def main() -> None:
    args = parse_args()
    ensure_importable()
    project_root = find_project_root()
    output_root = (
        Path(args.output_root)
        if args.output_root
        else project_root / DEFAULT_OUTPUT_ROOT / "pilots" / args.pilot_id
    )
    shot_grid = sorted(set(int(value) for value in args.shot_grid))
    ansatz_fns = {
        "ansatz_odra": ansatz_odra,
        "ansatz_simulator": ansatz_simulator,
    }

    shot_summary = _load_shot_pilot_summary(output_root, shot_grid)
    if shot_summary.empty:
        raise SystemExit("No existing shot pilot summaries found; nothing to top up.")

    failed_jobs, passing_jobs = _resolve_topup_jobs(
        shot_summary=shot_summary,
        output_root=output_root,
        shot_grid=shot_grid,
        identify_tolerance=args.identify_failure_tolerance,
        shot_tolerance=args.shot_tolerance,
        use_manifest_jobs=args.use_topup_manifest_jobs,
        topup_jobs_mode=args.topup_jobs_mode,
        baseline_samples=args.baseline_samples,
        pilot_samples=int(args.pilot_samples),
    )

    if not args.quiet:
        print(f"Pilot output root: {output_root}")
        print(f"Top-up jobs mode: {args.topup_jobs_mode}")
        print(f"Shot grid: {shot_grid}")
        print(f"Top-up target: pilot_samples={args.pilot_samples}")
        print(f"Re-select shots at tolerance={args.shot_tolerance}")
        print(f"Jobs to extend ({len(failed_jobs)}): {sorted(failed_jobs)}")
        print(f"Passing jobs (unchanged, {len(passing_jobs)}): {sorted(passing_jobs)}")

    if not failed_jobs:
        print("No jobs to top up.")
        return

    n_bins, eps = _resolve_pilot_kl_params(
        args,
        output_root,
        sorted(set(DEFAULT_SHOT_GRID + shot_grid)),
    )
    if not args.quiet:
        print(f"Using n_bins={n_bins}, eps={eps} (matched to existing shot pilot)")

    _refresh_stale_summaries(
        output_root=output_root,
        shot_grid=shot_grid,
        failed_jobs=failed_jobs,
        pilot_samples=int(args.pilot_samples),
        n_qubits=int(args.num_qubits),
        seed=int(args.seed),
        n_bins=n_bins,
        eps=eps,
        quiet=args.quiet,
    )

    per_job_counts, total_new_samples = _estimate_topup_work(
        output_root, shot_grid, failed_jobs, int(args.pilot_samples)
    )
    if not args.quiet:
        _print_topup_work_estimate(
            jobs=failed_jobs,
            per_job=per_job_counts,
            pilot_samples=int(args.pilot_samples),
            total_new=total_new_samples,
        )

    if total_new_samples == 0:
        print("All jobs already at target sample count.")
        if args.dry_run:
            return

    if args.dry_run:
        print("Dry run — no QPU work performed.")
        return

    backend = connect_to_iqm_backend(args.iqm_url, token=args.iqm_token)
    progress_callback = None if args.quiet else make_print_callback()
    n_samples_by_job = {job: int(args.pilot_samples) for job in failed_jobs}

    for shots in shot_grid:
        run_dir = output_root / "shot_pilot" / f"shots_{shots}"
        if not args.quiet:
            print(f"\n[top-up] shots={shots} -> {run_dir}")
        run_iqm_kl_sweep(
            backend,
            ansatz_fns=ansatz_fns,
            ansatz_names=list(args.ansatz),
            depths=list(args.depth),
            n_qubits=args.num_qubits,
            n_samples=int(args.pilot_samples),
            seed=args.seed,
            shots=shots,
            n_bins=n_bins,
            eps=eps,
            optimization_level=args.optimization_level,
            seed_transpiler=None,
            max_circuits_per_job=args.max_circuits_per_job,
            output_dir=run_dir,
            resume=args.resume,
            verbose=not args.quiet,
            progress_callback=progress_callback,
            jobs_filter=failed_jobs,
            n_samples_by_job=n_samples_by_job,
            manifest_extra={
                "pilot_id": args.pilot_id,
                "pilot_stage": "shot_pilot_topup",
                "pilot_samples_target": int(args.pilot_samples),
                "topup_failed_jobs": [list(job) for job in sorted(failed_jobs)],
                "iqm_url": args.iqm_url,
            },
        )

    shot_summary = _load_shot_pilot_summary(output_root, shot_grid)
    shot_stability, shot_stability_by_job = compute_kl_shot_stability_by_job(shot_summary)
    summary_path = output_root / "shot_pilot_summary.csv"
    shot_summary.to_csv(summary_path, index=False)
    shot_stability.to_csv(output_root / "shot_stability.csv", index=False)
    shot_stability_by_job.to_csv(output_root / "shot_stability_by_job.csv", index=False)

    report = {
        "pilot_id": args.pilot_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_samples": int(args.pilot_samples),
        "identify_failure_tolerance": float(args.identify_failure_tolerance),
        "shot_tolerance": float(args.shot_tolerance),
        "failed_jobs_topped_up": [list(job) for job in sorted(failed_jobs)],
        "passing_jobs_unchanged": [list(job) for job in sorted(passing_jobs)],
    }

    try:
        shots_by_job = choose_kl_shots_by_job(shot_summary, tolerance=args.shot_tolerance)
        shots_by_job_flat = nested_job_int_map_to_flat(shots_by_job)
        report["shots_by_job"] = shots_by_job
        report["shots_by_job_flat"] = shots_by_job_flat
        report["shot_selection_status"] = "ok"
        if not args.quiet:
            print(f"\nShot selection OK at tolerance={args.shot_tolerance}:")
            print(f"  per job: {shots_by_job}")
            print(f"  conservative max: {max(shots_by_job_flat.values())}")
    except KlPilotFallbackError as exc:
        report["shot_selection_status"] = "fallback"
        report["shot_selection_error"] = str(exc)
        if not args.quiet:
            print(f"\nShot selection still failed at tolerance={args.shot_tolerance}:")
            print(f"  {exc}")

    report_path = output_root / "shot_pilot_topup_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.quiet:
        print(f"\nWrote {summary_path}")
        print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
