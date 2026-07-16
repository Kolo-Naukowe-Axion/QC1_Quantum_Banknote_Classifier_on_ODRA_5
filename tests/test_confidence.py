"""Tests for the confidence-distribution module (no hardware required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qbanknote.confidence import (  # noqa: E402
    ConditionComparison,
    NoiseFit,
    accuracy_from_z,
    choose_iterations,
    choose_shots,
    class_descriptors,
    class_moments,
    class_separation,
    compare_conditions,
    confidence_rows,
    fit_noise_channel,
    hardware_confidences,
    interleave_by_class,
    iteration_precision,
    mean_half_width,
    predicted_accuracy,
    shot_confidences,
    shot_noise_std,
    shot_stability_table,
    statevector_confidences,
    summarize_condition,
)
from qbanknote.data import load_fold_arrays  # noqa: E402
from qbanknote.iqm import IQMBackendEstimator  # noqa: E402
from qbanknote.weights import load_cv_model  # noqa: E402

DEPTH, ENV, FOLD = 2, "Odra", 1


# --------------------------------------------------------------------------- #
# Shot-noise model
# --------------------------------------------------------------------------- #
def test_shot_confidence_unbiased() -> None:
    """Mean of the binomial resample -> z_ideal (no shift) as shots grow."""
    rng = np.random.default_rng(0)
    z_true = np.array([-0.8, -0.2, 0.0, 0.3, 0.9])
    samples = np.stack([shot_confidences(z_true, shots=8000, rng=rng) for _ in range(400)])
    mean = samples.mean(axis=0)
    assert np.allclose(mean, z_true, atol=0.01), mean


def test_shot_variance_matches_formula() -> None:
    """Empirical variance matches (1 - z^2) / N at several z."""
    rng = np.random.default_rng(1)
    shots = 2048
    for z in (-0.6, 0.0, 0.5):
        draws = shot_confidences(np.full(20000, z), shots=shots, rng=rng)
        emp_std = draws.std()
        assert abs(emp_std - shot_noise_std(z, shots)) < 0.002, (z, emp_std)


def test_shot_noise_std_max_at_boundary() -> None:
    assert shot_noise_std(0.0, 100) == 1.0 / np.sqrt(100)
    assert shot_noise_std(1.0, 100) == 0.0
    assert shot_noise_std(0.5, 2048) < shot_noise_std(0.0, 2048)


# --------------------------------------------------------------------------- #
# Noise-channel fit
# --------------------------------------------------------------------------- #
def test_fit_noise_channel_recovers_params() -> None:
    rng = np.random.default_rng(2)
    z_ideal = rng.uniform(-1, 1, 500)
    lam_true, b_true = 0.7, 0.05
    z_hw = lam_true * z_ideal + b_true + rng.normal(0, 0.01, z_ideal.size)
    fit = fit_noise_channel(z_ideal, z_hw)
    assert abs(fit.lam - lam_true) < 0.02
    assert abs(fit.b - b_true) < 0.02
    assert fit.r2 > 0.99
    assert fit.n == 500


def test_fit_noise_channel_handles_nans() -> None:
    z_ideal = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    z_hw = np.array([0.05, np.nan, 0.3, 0.2, 0.25])
    fit = fit_noise_channel(z_ideal, z_hw)
    assert fit.n == 3  # only fully finite pairs
    assert np.isfinite(fit.lam)


def test_predicted_accuracy_matches_when_identity() -> None:
    y = np.array([-1, -1, 1, 1])
    z_ideal = np.array([-0.5, -0.2, 0.3, 0.6])
    identity = NoiseFit(lam=1.0, b=0.0, r2=1.0, n=4)
    assert predicted_accuracy(z_ideal, y, identity) == accuracy_from_z(z_ideal, y)


def test_contraction_preserves_sign_but_shift_can_flip() -> None:
    y = np.array([-1, 1])
    z_ideal = np.array([-0.1, 0.1])
    # Pure contraction keeps signs -> accuracy preserved.
    assert predicted_accuracy(z_ideal, y, NoiseFit(0.3, 0.0, 1.0, 2)) == 1.0
    # A positive readout bias can push the negative-class score across 0.
    assert predicted_accuracy(z_ideal, y, NoiseFit(0.3, 0.2, 1.0, 2)) == 0.5


# --------------------------------------------------------------------------- #
# Drift-control permutation
# --------------------------------------------------------------------------- #
def test_interleave_is_permutation_and_alternates() -> None:
    y = np.array([-1, -1, -1, 1, 1])
    order, inverse = interleave_by_class(y)
    assert sorted(order.tolist()) == list(range(len(y)))  # is a permutation
    assert np.array_equal(order[inverse], np.arange(len(y)))  # inverse restores
    submitted_labels = y[order]
    # First four alternate -, +, -, + before the surplus class-0 sample.
    assert submitted_labels[:4].tolist() == [-1, 1, -1, 1]


def test_interleave_round_trips_values() -> None:
    y = np.array([-1, 1, -1, 1, -1, 1])
    z = np.arange(6, dtype=float)
    order, inverse = interleave_by_class(y)
    z_submitted = z[order]  # what hardware would receive
    restored = z_submitted[inverse]  # restore to original order
    assert np.array_equal(restored, z)


# --------------------------------------------------------------------------- #
# Decision-relative per-class descriptors
# --------------------------------------------------------------------------- #
def test_class_separation_grows_with_separation() -> None:
    rng = np.random.default_rng(3)
    n = 200
    y = np.array([-1] * n + [1] * n)
    # Same within-class spread; well-separated means vs. nearly-merged means.
    well = np.concatenate([rng.normal(-0.6, 0.15, n), rng.normal(0.6, 0.15, n)])
    merged = np.concatenate([rng.normal(-0.05, 0.15, n), rng.normal(0.05, 0.15, n)])
    assert class_separation(well, y) > class_separation(merged, y)
    # Raw mean gap is in z-units: ~1.2 when means sit at -+0.6.
    assert abs(class_separation(well, y) - 1.2) < 0.1


def test_class_separation_is_not_contraction_invariant() -> None:
    """Unlike the dropped d', the raw gap shrinks under the noise channel's
    contraction -- which is exactly why it can distinguish ansatzes."""
    rng = np.random.default_rng(4)
    n = 200
    y = np.array([-1] * n + [1] * n)
    z = np.concatenate([rng.normal(-0.5, 0.15, n), rng.normal(0.5, 0.15, n)])
    contracted = 0.5 * z  # lam = 0.5
    assert class_separation(contracted, y) < 0.6 * class_separation(z, y)


def test_class_descriptors_per_class_error_and_margin() -> None:
    # class -1: three correct (z<0), one crosses (z>0); class +1 mirrored.
    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)
    z = np.array([-0.6, -0.4, -0.2, 0.1, 0.6, 0.4, 0.2, -0.1], dtype=float)
    d = class_descriptors(z, y)
    assert d["error_class0"] == 0.25  # one of four crosses the boundary
    assert d["error_class1"] == 0.25
    assert d["mean_abs_z_class0"] > 0 and d["mean_abs_z_class1"] > 0
    # Normalized margin carries the sign of the class mean.
    assert d["margin_norm_class0"] < 0 < d["margin_norm_class1"]


def test_summarize_condition_keys() -> None:
    y = np.array([-1, -1, 1, 1])
    z = np.array([-0.5, -0.3, 0.4, 0.6])
    summary = summarize_condition(z, y)
    expected = {
        "accuracy", "class_separation",
        "mu_class0", "sigma_class0", "mu_class1", "sigma_class1",
        "margin_norm_class0", "margin_norm_class1",
        "error_class0", "error_class1",
        "mean_abs_z_class0", "mean_abs_z_class1",
    }
    assert set(summary) == expected
    assert summary["accuracy"] == 1.0
    # Dropped metrics must be gone.
    assert "dprime" not in summary and "bhattacharyya" not in summary


def test_class_moments() -> None:
    y = np.array([-1, -1, 1, 1])
    z = np.array([-0.5, -0.3, 0.4, 0.6])
    m = class_moments(z, y)
    assert abs(m["mu_class0"] - (-0.4)) < 1e-9
    assert abs(m["mu_class1"] - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# Cross-condition comparison (statevector vs QPU)
# --------------------------------------------------------------------------- #
def test_compare_conditions_recovers_channel_and_shifts() -> None:
    """lam~=0.83, b~=0.04 as read off the Odra depth-2 fold-1 plot; per-class
    mean shift must equal the affine prediction (lam-1)*mu_ref + b."""
    rng = np.random.default_rng(5)
    y = np.array([-1] * 200 + [1] * 200, dtype=float)
    z_ref = np.concatenate([rng.normal(-0.35, 0.18, 200), rng.normal(0.17, 0.18, 200)])
    lam_true, b_true = 0.83, 0.04
    z_hw = lam_true * z_ref + b_true  # noiseless affine map

    cmp = compare_conditions(z_ref, z_hw, y)
    assert isinstance(cmp, ConditionComparison)
    assert abs(cmp.fit.lam - lam_true) < 1e-6 and abs(cmp.fit.b - b_true) < 1e-6

    m_ref = class_moments(z_ref, y)
    assert abs(cmp.delta_mu_class0 - ((lam_true - 1) * m_ref["mu_class0"] + b_true)) < 1e-6
    assert abs(cmp.delta_mu_class1 - ((lam_true - 1) * m_ref["mu_class1"] + b_true)) < 1e-6
    # Noiseless affine map -> predicted accuracy equals measured.
    assert np.isclose(cmp.accuracy_predicted, cmp.accuracy_test)


def test_compare_conditions_identity_is_flat_zero_delta() -> None:
    y = np.array([-1, -1, 1, 1])
    z = np.array([-0.5, -0.3, 0.4, 0.6])
    d = compare_conditions(z, z, y).as_dict()
    assert {"fit_lam", "fit_b", "fit_r2", "fit_n"} <= set(d)
    assert {"delta_mu_class0", "delta_error_class1", "accuracy_predicted"} <= set(d)
    assert np.isclose(d["fit_lam"], 1.0) and np.isclose(d["fit_b"], 0.0)
    assert d["delta_error_class0"] == 0.0 and d["delta_error_class1"] == 0.0


# --------------------------------------------------------------------------- #
# Row builder
# --------------------------------------------------------------------------- #
def test_confidence_rows_schema() -> None:
    y = np.array([-1, 1])
    z = np.array([-0.2, 0.7])
    rows = confidence_rows(
        environment=ENV, depth=DEPTH, fold=FOLD, condition="ideal",
        run_index=0, y=y, z=z, shots=float("nan"),
    )
    assert len(rows) == 2
    expected = {
        "Environment", "Depth", "Fold", "Condition", "RunIndex",
        "SampleID", "TrueLabel", "Shots", "z", "PredLabel",
    }
    assert set(rows[0]) == expected
    assert rows[0]["PredLabel"] == -1 and rows[1]["PredLabel"] == 1


# --------------------------------------------------------------------------- #
# Hardware guard (no real backend; failed-batch sentinel must raise)
# --------------------------------------------------------------------------- #
def test_hardware_confidences_raises_on_failed_batch() -> None:
    class _Estimator:
        failed_batches = [{"start": 0, "end": 4, "n_circuits": 4, "error": "boom"}]

    class _Model:
        def eval(self):
            return self

        def __call__(self, x):
            return torch.zeros((x.shape[0], 1))

    try:
        hardware_confidences(_Model(), _Estimator(), np.zeros((4, 5)))
        raise AssertionError("expected RuntimeError on failed batch")
    except RuntimeError as exc:
        assert "z=0.0" in str(exc)


# --------------------------------------------------------------------------- #
# End-to-end offline pipeline against the real trained checkpoint
# --------------------------------------------------------------------------- #
def test_statevector_pipeline_matches_reported_accuracy() -> None:
    """Consistency gate: sign(z_ideal) accuracy must equal the model's forward
    accuracy, and the shot condition must reproduce the ideal class means."""
    model, _ = load_cv_model(DEPTH, ENV, FOLD, condition="Ideal")
    X, y = load_fold_arrays(FOLD, split="test")

    z_ideal = statevector_confidences(model, X)
    assert z_ideal.shape == (len(y),)
    assert np.all(np.abs(z_ideal) <= 1.0 + 1e-9)

    # sign(z_ideal) accuracy == direct evaluate_predictions on forward output.
    with torch.no_grad():
        direct = model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()
    assert accuracy_from_z(z_ideal, y) == accuracy_from_z(direct, y)
    # Banknote classifier is strong; sanity floor well above chance.
    assert accuracy_from_z(z_ideal, y) > 0.8

    # Shot condition is unbiased: class means track the ideal means (broadening
    # only, no shift) at a realistic shot budget.
    rng = np.random.default_rng(42)
    z_shot = shot_confidences(z_ideal, shots=4096, rng=rng)
    ideal_m = class_moments(z_ideal, y)
    shot_m = class_moments(z_shot, y)
    assert abs(ideal_m["mu_class0"] - shot_m["mu_class0"]) < 0.02
    assert abs(ideal_m["mu_class1"] - shot_m["mu_class1"]) < 0.02


def test_runner_cli_parses() -> None:
    import importlib.util

    path = ROOT / "scripts" / "run_confidence_distributions.py"
    spec = importlib.util.spec_from_file_location("run_confidence_distributions", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    original_argv = sys.argv
    try:
        sys.argv = [
            "run_confidence_distributions.py",
            "--env", "Odra", "Simulator", "--depth", "2", "4", "--fold", "1", "2",
            "--with-hardware", "--repeats", "3", "--shots", "2048",
            "--max-circuits-per-job", "100",
        ]
        args = module.parse_args()
    finally:
        sys.argv = original_argv
    # env/depth/fold are now lists (sweepable).
    assert args.env == ["Odra", "Simulator"]
    assert args.depth == [2, 4] and args.fold == [1, 2]
    assert args.with_hardware is True and args.repeats == 3
    assert args.shots == 2048 and args.max_circuits_per_job == 100
    assert module.ansatz_fn("Odra").__name__ == "odra_ansatz"
    assert module.ansatz_fn("Simulator").__name__ == "simulator_ansatz"


# --------------------------------------------------------------------------- #
# Pilot calibration helpers
# --------------------------------------------------------------------------- #
def test_mean_half_width_basic() -> None:
    mean, hw = mean_half_width([0.80, 0.84, 0.82], use_t=False)
    assert abs(mean - 0.82) < 1e-9
    assert hw > 0
    # n=1 -> value, nan half-width; n=0 -> nan, nan.
    one_mean, one_hw = mean_half_width([0.5])
    assert one_mean == 0.5 and np.isnan(one_hw)
    assert all(np.isnan(v) for v in mean_half_width([]))


def test_choose_shots_picks_converged_budget() -> None:
    shots = [512, 1024, 2048, 4096]
    # lambda settles by 1024->2048 (delta 0.005); b is flat.
    lams = [0.70, 0.78, 0.785, 0.787]
    bs = [0.04, 0.04, 0.04, 0.04]
    # 512->1024 moves lambda by 0.08 (> tol); 1024->2048 by 0.005 (<= tol).
    assert choose_shots(shots, lams, bs, tolerance=0.02) == 1024
    # Tight tolerance never converges -> largest budget.
    assert choose_shots(shots, lams, bs, tolerance=0.001) == 4096
    # Order-independence.
    assert choose_shots(shots[::-1], lams[::-1], bs[::-1], tolerance=0.02) == 1024


def test_choose_shots_respects_b_movement() -> None:
    shots = [512, 1024, 2048]
    lams = [0.78, 0.785, 0.787]  # lambda converged
    bs = [0.00, 0.05, 0.051]     # but b jumps 512->1024
    # 512->1024 fails on b; 1024->2048 converges on both -> 1024.
    assert choose_shots(shots, lams, bs, tolerance=0.02) == 1024


def test_shot_stability_table_deltas() -> None:
    rows = shot_stability_table([1024, 512], [0.78, 0.70], [0.04, 0.02])
    assert rows[0]["shots_from"] == 512 and rows[0]["shots_to"] == 1024
    assert abs(rows[0]["delta_lambda"] - 0.08) < 1e-9
    assert abs(rows[0]["delta_b"] - 0.02) < 1e-9


def test_choose_iterations_and_precision() -> None:
    # Very stable lambda/b -> target met at the minimum iteration count.
    lams = [0.830, 0.831, 0.829, 0.830]
    bs = [0.040, 0.041, 0.039, 0.040]
    k, met = choose_iterations(lams, bs, target_half_width=0.02,
                               min_iterations=2, max_iterations=4)
    assert met and k == 2
    # Unreachable target -> max iterations, not met.
    noisy = [0.6, 0.9, 0.7, 1.0]
    k2, met2 = choose_iterations(noisy, bs, target_half_width=0.001,
                                 min_iterations=2, max_iterations=4)
    assert not met2 and k2 == 4
    prec = iteration_precision(lams, bs)
    assert abs(prec["lambda_mean"] - 0.830) < 1e-3 and prec["n_iterations"] == 4


def test_confidence_pilot_cli_parses() -> None:
    import importlib.util

    path = ROOT / "scripts" / "run_iqm_confidence_pilot.py"
    spec = importlib.util.spec_from_file_location("run_iqm_confidence_pilot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    original_argv = sys.argv
    try:
        sys.argv = [
            "run_iqm_confidence_pilot.py",
            "--env", "Odra", "--depth", "2", "--fold", "1",
            "--shot-grid", "512", "1024", "2048",
            "--shot-tolerance", "0.02", "--max-iterations", "4",
        ]
        args = module.parse_args()
    finally:
        sys.argv = original_argv
    assert args.env == "Odra" and args.depth == 2 and args.fold == 1
    assert args.shot_grid == [512, 1024, 2048] and args.shot_tolerance == 0.02
    assert module.ansatz_fn("Odra").__name__ == "odra_ansatz"
    # Stratified subset is class-balanced and a strict subset.
    y = np.array([-1] * 120 + [1] * 80, dtype=float)
    sub = module._stratified_subset(y, 40, seed=1)
    assert sub.size <= 44 and sub.size >= 36
    assert (y[sub] < 0).sum() > 0 and (y[sub] > 0).sum() > 0
    assert module._stratified_subset(y, 999, seed=1).size == y.size  # n>=N -> all


def test_iqm_counts_to_expectation_endianness() -> None:
    """Qubit 0 is the last bit; all-zeros -> z=+1, qubit-0 set -> contributes -1."""
    est = IQMBackendEstimator(backend=None, options={"shots": 100})
    assert est._counts_to_expectation({"00000": 100}) == 1.0
    assert est._counts_to_expectation({"00001": 100}) == -1.0  # qubit 0 set
    assert est._counts_to_expectation({"10000": 100}) == 1.0   # qubit 4 set, q0=0
    assert abs(est._counts_to_expectation({"00000": 50, "00001": 50})) < 1e-9
