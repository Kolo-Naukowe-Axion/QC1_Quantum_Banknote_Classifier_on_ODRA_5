"""Smoke tests for the qbanknote package (no hardware)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.ansatzes import (  # noqa: E402
    LEGACY_PARAM_COUNTS,
    TRIMMED_PARAM_COUNTS,
    legacy_odra_ring_ansatz,
    legacy_simulator_ring_ansatz,
    odra_ansatz,
    param_count_for_ansatz,
    simulator_ansatz,
    trimmed_reverse_q0_param_count,
)
from qbanknote.data import load_fold_arrays, set_random_seed  # noqa: E402
from qbanknote.metrics import (  # noqa: E402
    bloch_row_to_score_row,
    KlPilotFallbackError,
    aggregate_kl_from_sample_rows,
    analyze_kl_qpu_sim_haar_jobs,
    choose_kl_bins,
    choose_kl_iterations,
    choose_kl_samples,
    choose_kl_shots,
    choose_mw_iterations,
    choose_mw_samples,
    choose_mw_shots,
    completed_kl_jobs,
    completed_kl_samples,
    completed_mw_jobs,
    compute_kl_bin_sensitivity,
    compute_kl_between_fidelity_samples,
    compute_kl_iteration_precision,
    compute_kl_prefix_precision,
    compute_kl_shot_stability,
    compute_mw_iteration_precision,
    compute_mw_iteration_stability,
    compute_mw_sample_precision,
    compute_mw_shot_stability,
    compute_statevector_fidelities_for_job,
    kl_depth_seed,
    kl_summary_row,
    load_kl_job_fidelity_rows,
    mw_iteration_half_width,
    mw_mean_shot_noise_bound,
    mw_shot_noise_sd_bound,
    mw_student_t_975,
    reproduce_pairwise_thetas,
    resolve_kl_run_data_dir,
    run_kl_self_check,
    run_mw_self_check,
    summary_row_from_result,
    write_kl_comparison_artifacts,
    write_kl_protocol_artifacts,
    write_mw_protocol_artifacts,
)
from qbanknote.model import HybridModel  # noqa: E402
from qbanknote.paths import find_project_root  # noqa: E402
from qbanknote.iqm import build_iqm_estimator_model  # noqa: E402
from qbanknote.weights import (  # noqa: E402
    WeightAnsatzMismatchError,
    cv_weight_path,
    metric_weight_path,
    validate_ansatz_weight_compatibility,
)
from qbanknote.classification import evaluate_predictions, predictions_to_labels  # noqa: E402
from qbanknote.stats import (  # noqa: E402
    analyze_final_summary,
    select_protocol_from_pilot,
    sign_test_exact,
    wilcoxon_signed_rank_exact,
    write_analysis_artifacts,
    write_protocol_artifacts,
)
from qbanknote.progress import make_print_callback, progress_bar  # noqa: E402
from qbanknote.evaluation import (  # noqa: E402
    ANSATZ_NAMES,
    ODRA_ANSATZ_NAME,
    PhaseSpec,
    SIMULATOR_ANSATZ_NAME,
    append_csv_row,
    build_failed_hardware_row,
    canonical_ansatz_name,
    count_hardware_tasks,
    count_statevector_tasks,
    load_phase_spec,
    summarize_results,
)


def test_imports() -> None:
    root = find_project_root(ROOT)
    assert (root / "cross_validation").is_dir()


def test_trimmed_param_counts() -> None:
    n_qubits = 5
    for depth, expected in TRIMMED_PARAM_COUNTS.items():
        assert trimmed_reverse_q0_param_count(n_qubits, depth) == expected
        assert param_count_for_ansatz(simulator_ansatz(n_qubits, depth)) == expected
        assert param_count_for_ansatz(odra_ansatz(n_qubits, depth)) == expected


def test_legacy_param_counts() -> None:
    n_qubits = 5
    for depth, expected in LEGACY_PARAM_COUNTS.items():
        assert param_count_for_ansatz(legacy_simulator_ring_ansatz(n_qubits, depth)) == expected
        assert param_count_for_ansatz(legacy_odra_ring_ansatz(n_qubits, depth)) == expected


def test_hybrid_model_construction() -> None:
    set_random_seed(0)
    model = HybridModel(simulator_ansatz(5, 2), 5)
    x = torch.zeros((2, 5), dtype=torch.float32)
    y = model(x)
    assert y.shape == (2, 1)


def test_fold_data_loads() -> None:
    X, y = load_fold_arrays(1, split="test")
    assert X.shape[1] == 5
    assert set(np.unique(y)).issubset({-1.0, 1.0})


def test_weight_mismatch_guard() -> None:
    trimmed = simulator_ansatz(5, 2)
    legacy = legacy_simulator_ring_ansatz(5, 2)
    legacy_weights = np.zeros(param_count_for_ansatz(legacy))
    try:
        validate_ansatz_weight_compatibility(trimmed, legacy_weights, depth=2)
        raise AssertionError("Expected WeightAnsatzMismatchError")
    except WeightAnsatzMismatchError:
        pass


def test_cv_weight_path_format() -> None:
    path = cv_weight_path(2, "Simulator", 1, epoch=30)
    assert path.name == "Simulator_fold_1_depth_2_epoch_30_weights.pth"
    assert "depth 2" in str(path)


def test_metric_self_checks() -> None:
    run_kl_self_check()
    run_mw_self_check()


def test_classification_helpers() -> None:
    y_true = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    preds = np.array([-0.2, 0.3, -0.1, 0.4], dtype=np.float32)
    labels = predictions_to_labels(preds)
    metrics = evaluate_predictions(y_true, preds)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert set(labels.tolist()) == {-1.0, 1.0}


def test_metric_weight_path_ideal_suffix() -> None:
    path = metric_weight_path(
        2,
        "simulator",
        1,
        epoch=30,
        simulator_uses_ideal_suffix=True,
        root=ROOT,
    )
    assert path.name == "Simulator_fold_1_depth_2_epoch_30_ideal_weights.pth"


def test_exact_stats_helpers() -> None:
    diffs = [0.1, 0.2, -0.05, 0.15]
    wilcoxon = wilcoxon_signed_rank_exact(diffs)
    sign = sign_test_exact(diffs)
    assert wilcoxon["pvalue"] is not None
    assert 0.0 <= sign["pvalue"] <= 1.0


def test_select_protocol_from_pilot_synthetic() -> None:
    spec = PhaseSpec(
        experiment_name="test",
        phase="pilot",
        depth=2,
        checkpoint_epoch=30,
        simulator_uses_ideal_suffix=True,
        folds=(1, 2),
        shots=(512, 1024),
        repeats=2,
        run_iqm_hardware=True,
        cross_validation_dir="cross_validation",
        outputs_dir="outputs",
        num_qubits=5,
        random_seed=42,
        optimization_level=1,
        seed_transpiler=42,
        shuffle_execution=False,
        iqm_url="https://example.invalid/",
        delta_accuracy=0.5,
        delta_f1=0.5,
        target_half_width_accuracy=0.02,
        target_half_width_f1=0.03,
    )
    rows = []
    for fold in (1, 2):
        for ansatz in ANSATZ_NAMES:
            for shot in (512, 1024):
                rows.append(
                    {
                        "phase": "pilot",
                        "depth": 2,
                        "fold": fold,
                        "ansatz": ansatz,
                        "statevector_accuracy": 0.9,
                        "statevector_f1": 0.88,
                        "statevector_std_accuracy": 0.0,
                        "statevector_std_f1": 0.0,
                        "iqm_mean_accuracy": 0.85,
                        "iqm_std_accuracy": 0.01,
                        "iqm_mean_f1": 0.83,
                        "iqm_std_f1": 0.01,
                        "eval_shots": shot,
                        "n_repeats": 2,
                        "completed_repeats": 2,
                        "test_csv": "x",
                        "weight_path": "y",
                    }
                )
    summary_df = pd.DataFrame(rows)
    protocol = select_protocol_from_pilot(summary_df, spec)
    assert protocol["chosen_shot"] in (512, 1024)
    assert protocol["chosen_repeats"] >= 1


def test_build_iqm_estimator_model() -> None:
    class _FakeBackend:
        num_qubits = 5

        def run(self, circuits, shots):
            raise AssertionError("fake backend should not be executed in construction test")

    model, estimator = build_iqm_estimator_model(
        _FakeBackend(),
        simulator_ansatz,
        num_qubits=5,
        depth=2,
        shots=100,
        optimization_level=1,
        seed_transpiler=42,
    )
    assert model is not None
    assert isinstance(estimator.failed_batches, list)


def test_progress_fallback() -> None:
    values = list(progress_bar(range(3), total=3, desc="test", disable=False))
    assert values == [0, 1, 2]

    seen: list[tuple[str, int, int]] = []

    def _cb(stage: str, completed: int, total: int) -> None:
        seen.append((stage, completed, total))

    callback = make_print_callback("prefix:")
    callback("hardware", 1, 5)
    assert seen == []  # print callback does not use seen list
    _cb("statevector", 2, 4)
    assert seen == [("statevector", 2, 4)]


def test_mw_csv_artifact_helpers(tmp_path: Path) -> None:
    result = {
        "n_qubits": 5,
        "n_params": 10,
        "n_samples": 2,
        "mw_avg": 0.5,
        "mw_std": 0.1,
        "mw_sem": 0.07,
        "mw_min": 0.4,
        "mw_max": 0.6,
        "bloch_rows": [
            {"sample_index": 0.0, "mw_score": 0.5, "x_q0": 0.1, "y_q0": 0.2, "z_q0": 0.9},
            {"sample_index": 1.0, "mw_score": 0.6, "x_q0": 0.0, "y_q0": 0.0, "z_q0": 1.0},
        ],
    }
    summary_row = summary_row_from_result(
        ansatz="ansatz_odra",
        depth=2,
        shots=1024,
        seed=42,
        result=result,
    )
    summary_path = tmp_path / "iqm_mw_results.csv"
    append_csv_row(summary_path, summary_row)
    assert completed_mw_jobs(summary_path) == {("ansatz_odra", 2)}

    scores_path = tmp_path / "iqm_mw_scores.csv"
    for bloch in result["bloch_rows"]:
        append_csv_row(
            scores_path,
            bloch_row_to_score_row("ansatz_odra", 2, bloch),
        )
    scores_df = pd.read_csv(scores_path)
    assert len(scores_df) == 2
    assert "mw_score" in scores_df.columns


def test_mw_protocol_precision_helpers(tmp_path: Path) -> None:
    assert 0.013 < mw_shot_noise_sd_bound(5, 4096) < 0.015
    assert mw_mean_shot_noise_bound(5, 4096, 20) < 0.004

    shot_summary = pd.DataFrame(
        [
            {"ansatz": "ansatz_odra", "depth": 2, "n_samples": 10, "shots": 512, "mw_avg": 0.50},
            {"ansatz": "ansatz_odra", "depth": 2, "n_samples": 10, "shots": 1024, "mw_avg": 0.515},
            {"ansatz": "ansatz_odra", "depth": 2, "n_samples": 10, "shots": 2048, "mw_avg": 0.517},
            {"ansatz": "ansatz_simulator", "depth": 2, "n_samples": 10, "shots": 512, "mw_avg": 0.40},
            {"ansatz": "ansatz_simulator", "depth": 2, "n_samples": 10, "shots": 1024, "mw_avg": 0.411},
            {"ansatz": "ansatz_simulator", "depth": 2, "n_samples": 10, "shots": 2048, "mw_avg": 0.412},
        ]
    )
    detailed, aggregate = compute_mw_shot_stability(shot_summary)
    assert not detailed.empty
    assert not aggregate.empty
    assert choose_mw_shots(shot_summary, tolerance=0.02) == 1024

    sample_summary = pd.DataFrame(
        [
            {
                "ansatz": "ansatz_odra",
                "depth": 2,
                "shots": 1024,
                "n_samples": 10,
                "mw_std": 0.08,
                "mw_sem": 0.08 / np.sqrt(10),
            },
            {
                "ansatz": "ansatz_simulator",
                "depth": 2,
                "shots": 1024,
                "n_samples": 10,
                "mw_std": 0.07,
                "mw_sem": 0.07 / np.sqrt(10),
            },
            {
                "ansatz": "ansatz_odra",
                "depth": 2,
                "shots": 1024,
                "n_samples": 40,
                "mw_std": 0.08,
                "mw_sem": 0.08 / np.sqrt(40),
            },
            {
                "ansatz": "ansatz_simulator",
                "depth": 2,
                "shots": 1024,
                "n_samples": 40,
                "mw_std": 0.07,
                "mw_sem": 0.07 / np.sqrt(40),
            },
        ]
    )
    precision, precision_aggregate = compute_mw_sample_precision(
        sample_summary,
        target_half_width=0.03,
    )
    assert not precision.empty
    assert not precision_aggregate.empty
    assert choose_mw_samples(sample_summary, target_half_width=0.03) == 40

    iteration_summary = pd.DataFrame(
        [
            {
                "ansatz": "ansatz_odra",
                "depth": 2,
                "shots": 1024,
                "n_samples": 40,
                "iteration": 1,
                "mw_avg": 0.5,
            },
            {
                "ansatz": "ansatz_odra",
                "depth": 2,
                "shots": 1024,
                "n_samples": 40,
                "iteration": 2,
                "mw_avg": 0.52,
            },
        ]
    )
    iteration_stability = compute_mw_iteration_stability(iteration_summary)
    assert float(iteration_stability["iteration_std_mw_avg"].iloc[0]) > 0

    assert abs(mw_student_t_975(3) - 4.303) < 1e-3
    assert abs(mw_student_t_975(4) - 3.182) < 1e-3
    assert abs(mw_student_t_975(5) - 2.776) < 1e-3
    assert mw_student_t_975(20) == 1.96
    half_width = mw_iteration_half_width(0.02, 5)
    assert abs(half_width - 2.776 * 0.02 / np.sqrt(5)) < 1e-9

    stable_iteration_summary = pd.DataFrame(
        [
            {
                "ansatz": "ansatz_odra",
                "depth": 2,
                "shots": 1024,
                "n_samples": 40,
                "iteration": iteration,
                "mw_avg": 0.50 + 0.001 * iteration,
            }
            for iteration in range(1, 4)
        ]
        + [
            {
                "ansatz": "ansatz_simulator",
                "depth": 2,
                "shots": 1024,
                "n_samples": 40,
                "iteration": iteration,
                "mw_avg": 0.40 + 0.001 * iteration,
            }
            for iteration in range(1, 4)
        ]
    )
    iteration_precision, iteration_precision_aggregate = compute_mw_iteration_precision(
        stable_iteration_summary,
        target_half_width=0.01,
    )
    assert not iteration_precision.empty
    assert bool(iteration_precision_aggregate.iloc[0]["all_meet_target"])
    assert choose_mw_iterations(
        stable_iteration_summary,
        target_half_width=0.01,
        min_iterations=3,
        max_iterations=5,
    ) == 3

    noisy_iteration_summary = pd.DataFrame(
        [
            {
                "ansatz": "ansatz_odra",
                "depth": 2,
                "shots": 1024,
                "n_samples": 40,
                "iteration": iteration,
                "mw_avg": 0.50 + 0.05 * ((-1) ** iteration),
            }
            for iteration in range(1, 4)
        ]
        + [
            {
                "ansatz": "ansatz_simulator",
                "depth": 2,
                "shots": 1024,
                "n_samples": 40,
                "iteration": iteration,
                "mw_avg": 0.40 + 0.04 * ((-1) ** iteration),
            }
            for iteration in range(1, 4)
        ]
    )
    noisy_precision, noisy_aggregate = compute_mw_iteration_precision(
        noisy_iteration_summary,
        target_half_width=0.01,
    )
    assert not bool(noisy_aggregate.iloc[0]["all_meet_target"])
    assert choose_mw_iterations(
        noisy_iteration_summary,
        target_half_width=0.01,
        min_iterations=3,
        max_iterations=5,
    ) == 5

    report_path = write_mw_protocol_artifacts(
        tmp_path,
        recommendation={"chosen_shots": 1024, "chosen_n_samples": 40, "chosen_iterations": 3},
        frames={
            "shot_stability": detailed,
            "sample_precision": precision,
            "iteration_stability": iteration_stability,
            "iteration_precision": iteration_precision,
        },
    )
    assert report_path.exists()
    assert (tmp_path / "shot_stability.csv").exists()


def test_kl_protocol_precision_helpers(tmp_path: Path) -> None:
    dim = 32
    rng = np.random.default_rng(0)
    fidelities = 1.0 - rng.random(20) ** (1.0 / (dim - 1))

    _, bin_aggregate = compute_kl_bin_sensitivity(
        num_qubits=5,
        n_samples=15,
        bin_grid=[50, 100, 150],
        n_reference_bins=400,
        seed=0,
        n_trials=20,
    )
    assert not bin_aggregate.empty
    chosen_bins = choose_kl_bins(bin_aggregate, tolerance=0.05)
    assert chosen_bins in {50, 100, 150}

    try:
        choose_kl_bins(bin_aggregate, tolerance=1e-6)
    except KlPilotFallbackError:
        pass
    else:
        raise AssertionError("expected KlPilotFallbackError for strict bin tolerance")

    shot_summary = pd.DataFrame(
        [
            {
                "ansatz": "ansatz_odra",
                "depth": 2,
                "n_samples": 3,
                "shots": 512,
                "kl_physical": 0.50,
            },
            {
                "ansatz": "ansatz_odra",
                "depth": 2,
                "n_samples": 3,
                "shots": 1024,
                "kl_physical": 0.515,
            },
            {
                "ansatz": "ansatz_odra",
                "depth": 2,
                "n_samples": 3,
                "shots": 2048,
                "kl_physical": 0.517,
            },
            {
                "ansatz": "ansatz_simulator",
                "depth": 2,
                "n_samples": 3,
                "shots": 512,
                "kl_physical": 0.40,
            },
            {
                "ansatz": "ansatz_simulator",
                "depth": 2,
                "n_samples": 3,
                "shots": 1024,
                "kl_physical": 0.411,
            },
            {
                "ansatz": "ansatz_simulator",
                "depth": 2,
                "n_samples": 3,
                "shots": 2048,
                "kl_physical": 0.412,
            },
        ]
    )
    detailed, aggregate = compute_kl_shot_stability(shot_summary)
    assert not detailed.empty
    assert not aggregate.empty
    assert choose_kl_shots(shot_summary, tolerance=0.02) == 1024

    fidelities_df = pd.DataFrame(
        {
            "ansatz": ["ansatz_odra"] * len(fidelities),
            "depth": [2] * len(fidelities),
            "sample_index": list(range(len(fidelities))),
            "fidelity_physical": fidelities,
        }
    )
    precision, precision_aggregate = compute_kl_prefix_precision(
        fidelities_df,
        sample_grid=[5, 10, 15],
        dim=dim,
        n_bins=chosen_bins,
        eps=1e-12,
        target_half_width=0.10,
        n_bootstrap=50,
        seed=0,
    )
    assert not precision.empty
    assert not precision_aggregate.empty
    chosen_samples = choose_kl_samples(
        fidelities_df,
        sample_grid=[5, 10, 15],
        dim=dim,
        n_bins=chosen_bins,
        eps=1e-12,
        target_half_width=0.10,
        n_bootstrap=50,
        seed=0,
    )
    assert chosen_samples in {5, 10, 15}

    summary_path = tmp_path / "iqm_kl_results.csv"
    append_csv_row(
        summary_path,
        kl_summary_row(
            ansatz="ansatz_odra",
            depth=2,
            shots=1024,
            seed=42,
            n_bins=chosen_bins,
            eps=1e-12,
            result={
                "n_qubits": 5,
                "n_samples": chosen_samples,
                "kl_physical": 0.5,
                "kl_linear": 0.55,
                "f_physical_mean": 0.1,
                "f_physical_std": 0.05,
                "f_linear_mean": 0.1,
                "f_linear_std": 0.05,
            },
        ),
    )
    assert completed_kl_jobs(summary_path) == {("ansatz_odra", 2)}

    fidelities_path = tmp_path / "iqm_kl_fidelities.csv"
    for sample_index in (0, 2):
        append_csv_row(
            fidelities_path,
            {
                "ansatz": "ansatz_odra",
                "depth": 4,
                "sample_index": sample_index,
                "fidelity_linear": 0.1 + sample_index,
                "fidelity_physical": 0.2 + sample_index,
            },
        )
    assert completed_kl_samples(fidelities_path, "ansatz_odra", 4) == {0, 2}
    loaded = load_kl_job_fidelity_rows(fidelities_path, "ansatz_odra", 4)
    assert [int(row["sample_index"]) for row in loaded] == [0, 2]
    assert loaded[1]["fidelity_physical"] == 0.4

    iteration_summary = pd.DataFrame(
        [
            {
                "ansatz": "ansatz_odra",
                "depth": 2,
                "shots": 1024,
                "n_samples": 10,
                "n_bins": chosen_bins,
                "iteration": iteration,
                "kl_physical": 0.50 + 0.001 * iteration,
            }
            for iteration in range(1, 3)
        ]
        + [
            {
                "ansatz": "ansatz_simulator",
                "depth": 2,
                "shots": 1024,
                "n_samples": 10,
                "n_bins": chosen_bins,
                "iteration": iteration,
                "kl_physical": 0.40 + 0.001 * iteration,
            }
            for iteration in range(1, 3)
        ]
    )
    iteration_precision, iteration_precision_aggregate = compute_kl_iteration_precision(
        iteration_summary,
        target_half_width=0.02,
    )
    assert not iteration_precision.empty
    assert bool(iteration_precision_aggregate.iloc[0]["all_meet_target"])
    assert choose_kl_iterations(
        iteration_summary,
        target_half_width=0.02,
        min_iterations=2,
        max_iterations=4,
    ) == 2

    report_path = write_kl_protocol_artifacts(
        tmp_path,
        recommendation={
            "chosen_shots": 1024,
            "chosen_n_samples": chosen_samples,
            "chosen_n_bins": chosen_bins,
            "chosen_iterations": 2,
        },
        frames={
            "shot_stability": detailed,
            "sample_precision": precision,
            "iteration_precision": iteration_precision,
        },
    )
    assert report_path.exists()
    assert (tmp_path / "shot_stability.csv").exists()


def test_kl_qpu_sim_haar_analysis(tmp_path: Path) -> None:
    n_qubits = 5
    depth = 2
    ansatz = "ansatz_odra"
    base_seed = 42
    depth_seed = kl_depth_seed(base_seed, depth, ansatz)
    n_samples = 5
    n_bins = 50
    eps = 1e-12

    f_sim = compute_statevector_fidelities_for_job(
        odra_ansatz,
        n_qubits=n_qubits,
        depth=depth,
        n_samples=n_samples,
        seed=depth_seed,
    )
    assert len(f_sim) == n_samples

    fidelities_path = tmp_path / "iqm_kl_fidelities.csv"
    for sample_index, fidelity in enumerate(f_sim):
        append_csv_row(
            fidelities_path,
            {
                "ansatz": ansatz,
                "depth": depth,
                "sample_index": sample_index,
                "fidelity_linear": float(fidelity),
                "fidelity_physical": float(fidelity),
            },
        )

    result = aggregate_kl_from_sample_rows(
        load_kl_job_fidelity_rows(fidelities_path, ansatz, depth),
        n_qubits=n_qubits,
        n_bins=n_bins,
        eps=eps,
    )
    append_csv_row(
        tmp_path / "iqm_kl_results.csv",
        kl_summary_row(
            ansatz=ansatz,
            depth=depth,
            shots=1024,
            seed=depth_seed,
            n_bins=n_bins,
            eps=eps,
            result=result,
        ),
    )

    comparison = analyze_kl_qpu_sim_haar_jobs(
        tmp_path,
        ansatz_fns={"ansatz_odra": odra_ansatz, "ansatz_simulator": simulator_ansatz},
    )
    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert abs(float(row["kl_qpu_haar"]) - float(row["kl_sim_haar"])) < 1e-9
    assert float(row["kl_qpu_sim"]) < 1e-9
    assert float(row["f_gap_mean_abs"]) < 1e-9

    out_csv = write_kl_comparison_artifacts(tmp_path, comparison)
    assert out_csv.exists()
    assert resolve_kl_run_data_dir(tmp_path) == tmp_path

    theta_a1, theta_b1 = reproduce_pairwise_thetas(depth_seed, 10, 0)
    theta_a2, theta_b2 = reproduce_pairwise_thetas(depth_seed, 10, 0)
    assert np.allclose(theta_a1, theta_a2)
    assert np.allclose(theta_b1, theta_b2)

    kl_cross = compute_kl_between_fidelity_samples(
        f_sim,
        f_sim,
        n_bins=n_bins,
        eps=eps,
    )
    assert kl_cross < 1e-9


def _synthetic_summary_df() -> pd.DataFrame:
    rows = []
    for fold in (1, 2, 3):
        for ansatz in ANSATZ_NAMES:
            rows.append(
                {
                    "phase": "final",
                    "depth": 2,
                    "fold": fold,
                    "ansatz": ansatz,
                    "statevector_accuracy": 0.9,
                    "statevector_f1": 0.88,
                    "statevector_std_accuracy": 0.0,
                    "statevector_std_f1": 0.0,
                    "iqm_mean_accuracy": 0.85 if ansatz == ODRA_ANSATZ_NAME else 0.84,
                    "iqm_std_accuracy": 0.01,
                    "iqm_mean_f1": 0.83 if ansatz == ODRA_ANSATZ_NAME else 0.82,
                    "iqm_std_f1": 0.01,
                    "eval_shots": 2048,
                    "n_repeats": 2,
                    "completed_repeats": 2,
                    "test_csv": "x",
                    "weight_path": "y",
                }
            )
    return pd.DataFrame(rows)


def test_write_protocol_and_analysis_artifacts(tmp_path: Path) -> None:
    spec = PhaseSpec(
        experiment_name="test",
        phase="pilot",
        depth=2,
        checkpoint_epoch=30,
        simulator_uses_ideal_suffix=True,
        folds=(1, 2),
        shots=(512, 1024),
        repeats=2,
        run_iqm_hardware=True,
        cross_validation_dir="cross_validation",
        outputs_dir="outputs",
        num_qubits=5,
        random_seed=42,
        optimization_level=1,
        seed_transpiler=42,
        shuffle_execution=False,
        iqm_url="https://example.invalid/",
        delta_accuracy=0.5,
        delta_f1=0.5,
        target_half_width_accuracy=0.02,
        target_half_width_f1=0.03,
    )
    protocol = select_protocol_from_pilot(_synthetic_pilot_summary(), spec)
    report_path = write_protocol_artifacts(tmp_path, protocol)
    assert report_path.exists()
    assert (tmp_path / "protocol_recommendation.json").exists()

    analysis = analyze_final_summary(_synthetic_summary_df())
    write_analysis_artifacts(tmp_path, analysis)
    assert (tmp_path / "paired_tests.csv").exists()
    assert (tmp_path / "ansatz_level_summary.csv").exists()


def _synthetic_pilot_summary() -> pd.DataFrame:
    rows = []
    for fold in (1, 2):
        for ansatz in ANSATZ_NAMES:
            for shot in (512, 1024):
                rows.append(
                    {
                        "phase": "pilot",
                        "depth": 2,
                        "fold": fold,
                        "ansatz": ansatz,
                        "statevector_accuracy": 0.9,
                        "statevector_f1": 0.88,
                        "statevector_std_accuracy": 0.0,
                        "statevector_std_f1": 0.0,
                        "iqm_mean_accuracy": 0.85,
                        "iqm_std_accuracy": 0.01,
                        "iqm_mean_f1": 0.83,
                        "iqm_std_f1": 0.01,
                        "eval_shots": shot,
                        "n_repeats": 2,
                        "completed_repeats": 2,
                        "test_csv": "x",
                        "weight_path": "y",
                    }
                )
    return pd.DataFrame(rows)


def test_task_counts_and_failed_row() -> None:
    spec = PhaseSpec(
        experiment_name="test",
        phase="pilot",
        depth=2,
        checkpoint_epoch=30,
        simulator_uses_ideal_suffix=True,
        folds=(1, 2),
        shots=(512,),
        repeats=2,
        run_iqm_hardware=True,
        cross_validation_dir="cross_validation",
        outputs_dir="outputs",
        num_qubits=5,
        random_seed=42,
        optimization_level=1,
        seed_transpiler=42,
        shuffle_execution=False,
        iqm_url="https://example.invalid/",
        delta_accuracy=0.01,
        delta_f1=0.02,
        target_half_width_accuracy=0.02,
        target_half_width_f1=0.03,
    )
    assert count_statevector_tasks(spec) == 4
    assert count_hardware_tasks(spec) == 8
    failed = build_failed_hardware_row(
        spec,
        fold=1,
        ansatz_name="odra",
        shots=512,
        repeat_index=0,
        root=ROOT,
    )
    assert failed["status"] == "failed"
    assert failed["ansatz"] == ODRA_ANSATZ_NAME
    assert failed["qpu_time_total"] == 0.0


def test_qce_resp_ansatz_labels() -> None:
    assert ANSATZ_NAMES == (ODRA_ANSATZ_NAME, SIMULATOR_ANSATZ_NAME)
    assert canonical_ansatz_name("odra") == ODRA_ANSATZ_NAME
    assert canonical_ansatz_name("ansatz_odra") == ODRA_ANSATZ_NAME
    assert canonical_ansatz_name("simulator") == SIMULATOR_ANSATZ_NAME
    assert canonical_ansatz_name("ansatz_simulator") == SIMULATOR_ANSATZ_NAME


def test_cli_argument_parsing() -> None:
    import argparse
    import importlib.util

    scripts = [
        "run_iqm_meyer_wallach.py",
        "run_iqm_mw_pilot.py",
        "run_iqm_kl_pilot.py",
        "run_iqm_kl_expressibility.py",
        "run_iqm_metric_test.py",
        "select_iqm_metric_protocol.py",
        "analyze_iqm_metric_test.py",
    ]
    for script_name in scripts:
        path = ROOT / "scripts" / script_name
        spec = importlib.util.spec_from_file_location(script_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        assert callable(module.parse_args)

    pilot_path = ROOT / "scripts" / "run_iqm_mw_pilot.py"
    pilot_spec = importlib.util.spec_from_file_location("run_iqm_mw_pilot", pilot_path)
    pilot_module = importlib.util.module_from_spec(pilot_spec)
    assert pilot_spec.loader is not None
    pilot_spec.loader.exec_module(pilot_module)
    original_argv = sys.argv
    try:
        sys.argv = [
            "run_iqm_mw_pilot.py",
            "--drift-only",
            "--shots",
            "1024",
            "--samples",
            "60",
            "--target-iteration-half-width",
            "0.01",
            "--min-iterations",
            "3",
            "--max-iterations",
            "5",
        ]
        pilot_args = pilot_module.parse_args()
    finally:
        sys.argv = original_argv
    assert pilot_args.drift_only is True
    assert pilot_args.shots == 1024
    assert pilot_args.samples == 60
    assert pilot_args.target_iteration_half_width == 0.01

    kl_pilot_path = ROOT / "scripts" / "run_iqm_kl_pilot.py"
    kl_pilot_spec = importlib.util.spec_from_file_location("run_iqm_kl_pilot", kl_pilot_path)
    kl_pilot_module = importlib.util.module_from_spec(kl_pilot_spec)
    assert kl_pilot_spec.loader is not None
    kl_pilot_spec.loader.exec_module(kl_pilot_module)
    try:
        sys.argv = ["run_iqm_kl_pilot.py"]
        kl_pilot_args = kl_pilot_module.parse_args()
    finally:
        sys.argv = original_argv
    assert kl_pilot_args.shot_grid == [512, 1024, 2048, 4096, 8192]
    assert kl_pilot_args.sample_grid == [5, 8, 10, 12, 15, 20, 25, 30]
    assert kl_pilot_args.bin_grid == [50, 75, 100, 150, 200, 250, 300, 400]
    assert kl_pilot_args.max_samples == 30
    assert kl_pilot_args.shot_pilot_depth == [2, 4, 6]
    assert kl_pilot_args.pilot_samples == 3
    budget = kl_pilot_module.estimate_kl_pilot_budget(
        n_ansatzes=2,
        shot_grid=kl_pilot_args.shot_grid,
        shot_pilot_depths=kl_pilot_args.shot_pilot_depth,
        pilot_samples=kl_pilot_args.pilot_samples,
        depths=kl_pilot_args.depth,
        max_samples=kl_pilot_args.max_samples,
        max_iterations=kl_pilot_args.max_iterations,
        drift_only=False,
    )
    assert budget["shot_pairs"] == 90
    assert budget["total_pairs"] == 630

    mw_path = ROOT / "scripts" / "run_iqm_meyer_wallach.py"
    mw_spec = importlib.util.spec_from_file_location("run_iqm_meyer_wallach", mw_path)
    mw_module = importlib.util.module_from_spec(mw_spec)
    assert mw_spec.loader is not None
    mw_spec.loader.exec_module(mw_module)
    try:
        sys.argv = [
            "run_iqm_meyer_wallach.py",
            "--iterations",
            "3",
            "--target-iteration-half-width",
            "0.02",
        ]
        mw_args = mw_module.parse_args()
    finally:
        sys.argv = original_argv
    assert mw_args.iterations == 3
    assert mw_args.target_iteration_half_width == 0.02


if __name__ == "__main__":
    test_imports()
    test_trimmed_param_counts()
    test_legacy_param_counts()
    test_hybrid_model_construction()
    test_fold_data_loads()
    test_weight_mismatch_guard()
    test_cv_weight_path_format()
    test_metric_self_checks()
    test_classification_helpers()
    test_metric_weight_path_ideal_suffix()
    test_exact_stats_helpers()
    test_select_protocol_from_pilot_synthetic()
    test_build_iqm_estimator_model()
    test_progress_fallback()
    test_task_counts_and_failed_row()
    test_qce_resp_ansatz_labels()
    test_cli_argument_parsing()
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_mw_csv_artifact_helpers(tmp_path)
        test_mw_protocol_precision_helpers(tmp_path)
        test_kl_protocol_precision_helpers(tmp_path)
        test_kl_qpu_sim_haar_analysis(tmp_path)
        test_write_protocol_and_analysis_artifacts(tmp_path)
    print("All smoke tests passed.")
