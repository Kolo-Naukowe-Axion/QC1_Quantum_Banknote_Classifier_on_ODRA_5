#!/usr/bin/env python3
"""Inspect a state-fidelity pilot and resume it from the last incomplete step."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.metrics import completed_fidelity_jobs, completed_fidelity_samples  # noqa: E402
from qbanknote.paths import ensure_importable, find_project_root  # noqa: E402

DEFAULT_OUTPUT_ROOT = "evaluation_and_comparison/iqm_spark/iqm_fidelity_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show pilot progress and rerun pilot_state_fidelity.py with resume enabled"
    )
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Pilot directory (default: <project>/iqm_fidelity_outputs/pilots/<pilot_id>)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the resume command instead of only printing it",
    )
    parser.add_argument("--iqm-token", default=None)
    parser.add_argument(
        "--iqm-url",
        default=None,
        help="Defaults to the URL stored in the first run manifest, if present.",
    )
    return parser.parse_args()


def _read_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _expected_jobs(manifest: dict) -> list[tuple[str, int]]:
    ansatzes = [str(name) for name in manifest.get("ansatzes", [])]
    depths = [int(depth) for depth in manifest.get("depths", [])]
    return [(ansatz, depth) for depth in depths for ansatz in ansatzes]


def _inspect_run_dir(stage: str, run_dir: Path, fallback_manifest: dict) -> dict:
    manifest = _read_manifest(run_dir) or fallback_manifest
    summary_path = run_dir / "iqm_fidelity_results.csv"
    scores_path = run_dir / "iqm_fidelity_scores.csv"
    expected_jobs = _expected_jobs(manifest)
    completed_jobs = completed_fidelity_jobs(summary_path)
    n_samples = int(manifest.get("n_samples", 0))

    pending_jobs: list[tuple[str, int]] = []
    partial_jobs: list[tuple[str, int, int, int]] = []
    for job in expected_jobs:
        if job in completed_jobs:
            continue
        saved = completed_fidelity_samples(scores_path, job[0], job[1])
        if saved:
            partial_jobs.append((job[0], job[1], len(saved), n_samples))
        else:
            pending_jobs.append(job)

    return {
        "stage": stage,
        "run_dir": run_dir,
        "manifest": manifest,
        "expected_jobs": expected_jobs,
        "completed_jobs": [job for job in expected_jobs if job in completed_jobs],
        "pending_jobs": pending_jobs,
        "partial_jobs": partial_jobs,
        "done": not pending_jobs and not partial_jobs and bool(expected_jobs),
    }


def _collect_stage_statuses(pilot_root: Path) -> list[dict]:
    stages = [
        ("shot_pilot", sorted((pilot_root / "shot_pilot").glob("shots_*"))),
        ("sample_pilot", sorted((pilot_root / "sample_pilot").glob("samples_*"))),
        ("iteration_pilot", sorted((pilot_root / "iteration_pilot").glob("iteration_*"))),
    ]
    provisional: list[dict] = []
    for stage_name, run_dirs in stages:
        if not run_dirs:
            continue
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            provisional.append(_inspect_run_dir(stage_name, run_dir, {}))

    fallback_manifest = _fallback_manifest(pilot_root, provisional)
    statuses: list[dict] = []
    for stage_name, run_dirs in stages:
        if not run_dirs:
            continue
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            statuses.append(_inspect_run_dir(stage_name, run_dir, fallback_manifest))
    return statuses


def _first_incomplete(statuses: list[dict]) -> dict | None:
    for status in statuses:
        if not status["done"]:
            return status
    return None


def _infer_cli_args(pilot_root: Path, statuses: list[dict]) -> tuple[list[str], list[int], str | None]:
    manifest = {}
    for status in statuses:
        if status["manifest"]:
            manifest = status["manifest"]
            break
    if not manifest:
        raise SystemExit(f"No run manifests found under {pilot_root}")

    ansatzes = [str(name) for name in manifest.get("ansatzes", [])]
    depths = [int(depth) for depth in manifest.get("depths", [])]
    iqm_url = manifest.get("iqm_url")
    if not ansatzes or not depths:
        raise SystemExit(f"Could not infer ansatzes/depths from manifests in {pilot_root}")
    return ansatzes, depths, str(iqm_url) if iqm_url else None


def _fallback_manifest(pilot_root: Path, statuses: list[dict]) -> dict:
    for status in statuses:
        if status["manifest"]:
            return status["manifest"]
    raise SystemExit(f"No run manifests found under {pilot_root}")


def _format_job(job: tuple[str, int]) -> str:
    return f"{job[0]} depth={job[1]}"


def main() -> None:
    ensure_importable()
    args = parse_args()
    project_root = find_project_root(ROOT)
    pilot_root = (
        Path(args.output_root)
        if args.output_root
        else project_root / DEFAULT_OUTPUT_ROOT / "pilots" / args.pilot_id
    )
    if not pilot_root.exists():
        raise SystemExit(f"Pilot directory not found: {pilot_root}")

    statuses = _collect_stage_statuses(pilot_root)
    if not statuses:
        raise SystemExit(f"No pilot run directories found under {pilot_root}")

    print(f"Pilot root: {pilot_root}")
    for status in statuses:
        run_name = status["run_dir"].name
        completed = len(status["completed_jobs"])
        expected = len(status["expected_jobs"])
        print(f"\n[{status['stage']}/{run_name}] {completed}/{expected} jobs complete")
        for job in status["completed_jobs"]:
            print(f"  done: {_format_job(job)}")
        for ansatz, depth, saved, total in status["partial_jobs"]:
            print(f"  partial: {_format_job((ansatz, depth))} ({saved}/{total} samples)")
        for job in status["pending_jobs"]:
            print(f"  pending: {_format_job(job)}")

    incomplete = _first_incomplete(statuses)
    recommendation_path = pilot_root / "fidelity_protocol_recommendation.json"
    if incomplete is None and recommendation_path.exists():
        print("\nPilot appears complete.")
        print(f"Recommendation: {recommendation_path}")
        return

    if incomplete is None:
        print("\nAll discovered run directories are complete, but no final recommendation was written.")
        print("Rerunning the pilot will continue with any later stages still missing.")
        incomplete = statuses[-1]

    resume_target = incomplete["run_dir"]
    if incomplete["partial_jobs"]:
        ansatz, depth, saved, total = incomplete["partial_jobs"][0]
        stop_note = (
            f"{incomplete['stage']}/{resume_target.name}: "
            f"resume {_format_job((ansatz, depth))} from sample {saved + 1}/{total}"
        )
    elif incomplete["pending_jobs"]:
        stop_note = (
            f"{incomplete['stage']}/{resume_target.name}: "
            f"start {_format_job(incomplete['pending_jobs'][0])}"
        )
    else:
        stop_note = f"{incomplete['stage']}/{resume_target.name}: rerun this stage"

    print(f"\nStopped at: {stop_note}")

    ansatzes, depths, manifest_url = _infer_cli_args(pilot_root, statuses)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "pilot_state_fidelity.py"),
        "--pilot-id",
        args.pilot_id,
        "--ansatz",
        *ansatzes,
        "--depth",
        *[str(depth) for depth in depths],
        "--resume",
    ]
    iqm_url = args.iqm_url or manifest_url
    if iqm_url:
        cmd.extend(["--iqm-url", iqm_url])
    if args.iqm_token:
        cmd.extend(["--iqm-token", args.iqm_token])

    print("\nResume command:")
    print(" ".join(cmd))

    if args.run:
        print("\nLaunching resume run...\n", flush=True)
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
