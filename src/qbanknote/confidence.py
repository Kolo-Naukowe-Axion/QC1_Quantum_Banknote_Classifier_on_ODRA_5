"""Decision-variable (<Z_0>) confidence distributions across classes.

For a trained banknote classifier the decision variable is the single-qubit
expectation value ``z = <Z_0>(x) = model.forward(x)``, and the predicted label is
``sign(z)`` (threshold 0). This module computes ``z`` per test sample under three
conditions and compares the resulting class-conditional distributions:

* ``ideal`` -- exact statevector expectation (deterministic ground truth).
* ``shot``  -- classical binomial resample of the exact value (isolates shot
  noise; costs no QPU). This is the control that distinguishes genuine device
  noise from finite-sampling broadening.
* ``hw``    -- IQM Spark hardware expectation (real noisy value).

The headline quantitative result is the effective noise channel
``z_hw ~= lambda * z_ideal + b`` (least squares), whose contraction factor
``lambda`` predicts the hardware accuracy drop.

Everything except :func:`hardware_confidences` runs without a QPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import zip_longest

import numpy as np
import torch

from qbanknote.classification import evaluate_predictions, predictions_to_labels

Condition = str  # "ideal" | "shot" | "hw"


# --------------------------------------------------------------------------- #
# 1. The three confidence sources
# --------------------------------------------------------------------------- #
def statevector_confidences(model, X: np.ndarray) -> np.ndarray:
    """Exact ``<Z_0>(x_i) = model.forward(x_i)`` for every row of ``X``.

    Deterministic ground truth (StatevectorEstimator is exact, not sampled).
    """
    model.eval()
    X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    with torch.no_grad():
        z = model(X_tensor).detach().cpu().numpy().ravel()
    return z


def shot_confidences(
    z_ideal: np.ndarray, shots: int, rng: np.random.Generator
) -> np.ndarray:
    """Classical binomial resample of the exact ``<Z_0>`` -- isolates shot noise.

    Each shot measures qubit 0, so ``n0 ~ Binom(N, p0)`` with ``p0 = (1 + z)/2``
    and the estimator ``z_hat = 2 n0 / N - 1`` has ``E[z_hat] = z`` (unbiased,
    no shift) and ``Var[z_hat] = (1 - z^2) / N``. Costs no QPU.
    """
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    z_ideal = np.asarray(z_ideal, dtype=float)
    p0 = np.clip((1.0 + z_ideal) / 2.0, 0.0, 1.0)
    n0 = rng.binomial(shots, p0)
    return 2.0 * n0 / shots - 1.0


def shot_noise_std(z: np.ndarray | float, shots: int) -> np.ndarray:
    """Analytic shot-noise standard deviation ``sqrt(1 - z^2) / sqrt(N)``.

    Maximal (``1/sqrt(N)``) at the decision boundary ``z = 0``.
    """
    z = np.asarray(z, dtype=float)
    return np.sqrt(np.clip(1.0 - z**2, 0.0, 1.0)) / np.sqrt(shots)


def hardware_confidences(hw_model, hw_estimator, X: np.ndarray) -> np.ndarray:
    """Run the IQM-backed QNN forward and return per-sample ``<Z_0>``.

    Raises if any batch failed. This guard is mandatory: ``IQMBackendEstimator``
    fills failed batches with ``z = 0.0``, which is *exactly* the decision
    boundary -- silently accepting it would inject fake boundary samples that
    corrupt both the histogram and the accuracy (``sign(0)``). Re-run the repeat
    instead.
    """
    hw_model.eval()
    X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    with torch.no_grad():
        z = hw_model(X_tensor).detach().cpu().numpy().ravel()
    if getattr(hw_estimator, "failed_batches", None):
        first = hw_estimator.failed_batches[0]
        raise RuntimeError(
            f"{len(hw_estimator.failed_batches)} hardware batch(es) failed; "
            f"first error: {first['error']}. Refusing to record -- failed "
            "circuits return z=0.0 (the decision boundary). Re-run this repeat."
        )
    return z


# --------------------------------------------------------------------------- #
# 2. Drift control: class-interleaved submission order
# --------------------------------------------------------------------------- #
def interleave_by_class(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Permutation that alternates the two classes in submission order.

    Slow QPU drift over a long submission would otherwise correlate with class
    if the test set were class-sorted. Submit ``X[order]`` to the hardware, then
    restore original sample order with ``z_submitted[inverse]``.

    Returns ``(order, inverse)`` where ``order[p]`` is the original index of the
    p-th submitted sample and ``inverse`` is its argsort.
    """
    y = np.asarray(y)
    idx_neg = np.where(y < 0)[0]
    idx_pos = np.where(y > 0)[0]
    order: list[int] = []
    for a, b in zip_longest(idx_neg.tolist(), idx_pos.tolist()):
        if a is not None:
            order.append(a)
        if b is not None:
            order.append(b)
    order_arr = np.asarray(order, dtype=int)
    inverse = np.argsort(order_arr)
    return order_arr, inverse


# --------------------------------------------------------------------------- #
# 3. Headline noise-channel fit and accuracy prediction
# --------------------------------------------------------------------------- #
@dataclass
class NoiseFit:
    """Effective noise channel ``z_hw ~= lam * z_ideal + b`` (least squares)."""

    lam: float  # contraction / effective depolarising factor (margin shrink)
    b: float  # additive offset (readout bias)
    r2: float  # coefficient of determination
    n: int  # number of finite (z_ideal, z_hw) pairs used

    def apply(self, z_ideal: np.ndarray) -> np.ndarray:
        return self.lam * np.asarray(z_ideal, dtype=float) + self.b

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_noise_channel(z_ideal: np.ndarray, z_hw: np.ndarray) -> NoiseFit:
    """Least-squares fit ``z_hw ~= lam * z_ideal + b``; NaNs are dropped."""
    z_ideal = np.asarray(z_ideal, dtype=float)
    z_hw = np.asarray(z_hw, dtype=float)
    if z_ideal.shape != z_hw.shape:
        raise ValueError(f"shape mismatch: {z_ideal.shape} vs {z_hw.shape}")
    mask = np.isfinite(z_ideal) & np.isfinite(z_hw)
    n = int(mask.sum())
    if n < 2:
        return NoiseFit(float("nan"), float("nan"), float("nan"), n)
    zi, zh = z_ideal[mask], z_hw[mask]
    A = np.vstack([zi, np.ones(n)]).T
    (lam, b), *_ = np.linalg.lstsq(A, zh, rcond=None)
    resid = zh - (lam * zi + b)
    ss_tot = float(np.sum((zh - zh.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    return NoiseFit(float(lam), float(b), float(r2), n)


def accuracy_from_z(z: np.ndarray, y: np.ndarray) -> float:
    """Accuracy of ``sign(z)`` against labels in ``{-1, +1}`` (threshold 0)."""
    return evaluate_predictions(np.asarray(y), np.asarray(z))["accuracy"]


def predicted_accuracy(z_ideal: np.ndarray, y: np.ndarray, fit: NoiseFit) -> float:
    """Apply the fitted ``(lam, b)`` to ideal scores and threshold -- closes the
    loop between the noise channel and the measured hardware accuracy drop."""
    return accuracy_from_z(fit.apply(z_ideal), y)


# --------------------------------------------------------------------------- #
# 4. Decision-relative per-class descriptors (one condition at a time)
# --------------------------------------------------------------------------- #
# These describe each class cloud *relative to the sign(z)=0 decision boundary*,
# the rule the classifier actually uses.
def class_moments(z: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Per-class mean/std of ``z`` for labels in ``{-1, +1}``.

    ``mu_class0``/``mu_class1`` are the signed margins (distance of each class
    from the boundary is ``|mu|``; the sign should match the class label).
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y)
    z0, z1 = z[y < 0], z[y > 0]
    return {
        "mu_class0": float(np.mean(z0)) if z0.size else float("nan"),
        "sigma_class0": float(np.std(z0)) if z0.size else float("nan"),
        "mu_class1": float(np.mean(z1)) if z1.size else float("nan"),
        "sigma_class1": float(np.std(z1)) if z1.size else float("nan"),
    }


def _normalized_margin(mu: float, sigma: float) -> float:
    """Signed margin in units of the class spread, ``mu / sigma``."""
    if sigma > 0:
        return float(mu / sigma)
    return float("nan") if mu == 0 else float(np.sign(mu) * np.inf)


def _class_error_fractions(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Per-class boundary-crossing fraction ``P(sign(z) != true label)``.

    Uses the same ``predictions_to_labels`` rule as the accuracy (``z>0 -> +1``,
    else ``-1``), so the two errors are the exact per-class decomposition of the
    overall misclassification rate.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y)
    labels = predictions_to_labels(z)
    l0, l1 = labels[y < 0], labels[y > 0]
    err0 = float(np.mean(l0 != -1)) if l0.size else float("nan")
    err1 = float(np.mean(l1 != 1)) if l1.size else float("nan")
    return err0, err1


def class_descriptors(z: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Decision-relative per-class descriptors for a single condition.

    For labels in ``{-1, +1}`` with the boundary at ``z = 0``:

    * ``margin_norm_class{0,1}`` -- signed margin in units of spread
      (``mu_c / sigma_c``); how many standard deviations the class mean sits
      from the boundary (per-class confidence).
    * ``error_class{0,1}`` -- boundary-crossing fraction ``P(sign(z) != c)``.
    * ``mean_abs_z_class{0,1}`` -- mean ``|z|`` (mean confidence magnitude;
      hardware contraction pulls this toward 0 even when the sign is right).

    The signed margin itself is ``mu_class{0,1}`` from :func:`class_moments`.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y)
    z0, z1 = z[y < 0], z[y > 0]
    moments = class_moments(z, y)
    err0, err1 = _class_error_fractions(z, y)
    return {
        "margin_norm_class0": _normalized_margin(
            moments["mu_class0"], moments["sigma_class0"]
        ),
        "margin_norm_class1": _normalized_margin(
            moments["mu_class1"], moments["sigma_class1"]
        ),
        "error_class0": err0,
        "error_class1": err1,
        "mean_abs_z_class0": float(np.mean(np.abs(z0))) if z0.size else float("nan"),
        "mean_abs_z_class1": float(np.mean(np.abs(z1))) if z1.size else float("nan"),
    }


def class_separation(z: np.ndarray, y: np.ndarray) -> float:
    """Raw between-class mean gap ``|mu_class1 - mu_class0|`` (in z-units).

    The ideal-separation scalar for the ansatz comparison: not normalized by
    spread and directly interpretable on the ``[-1, 1]`` scale.
    """
    m = class_moments(z, y)
    if np.isnan(m["mu_class0"]) or np.isnan(m["mu_class1"]):
        return float("nan")
    return float(abs(m["mu_class1"] - m["mu_class0"]))


def summarize_condition(z: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Per-condition snapshot: accuracy, class separation, moments, descriptors.

    Covers the "classes (per condition)" axis -- location/margin/confidence --
    plus the ideal-separation scalar used to compare ansatzes.
    """
    out: dict[str, float] = {
        "accuracy": accuracy_from_z(z, y),
        "class_separation": class_separation(z, y),
    }
    out.update(class_moments(z, y))
    out.update(class_descriptors(z, y))
    return out


# --------------------------------------------------------------------------- #
# 4b. Cross-condition comparison (statevector vs QPU, per ansatz)
# --------------------------------------------------------------------------- #
@dataclass
class ConditionComparison:
    """How faithfully a noisy condition reproduces a reference condition.

    Bundles the paired affine noise channel ``z_test ~= lam*z_ref + b`` with the
    per-class mean shift and per-class error change -- the full "statevector vs
    QPU" row. ``delta_mu_*`` and ``delta_error_*`` are ``test - ref`` (a positive
    ``delta_error`` means the noisy condition misclassifies *more* of that class).
    """

    fit: NoiseFit  # lam, b, r2, n of z_test ~= lam*z_ref + b
    accuracy_ref: float
    accuracy_test: float
    accuracy_predicted: float  # fit applied to z_ref, then thresholded
    delta_mu_class0: float
    delta_mu_class1: float
    delta_error_class0: float
    delta_error_class1: float

    def as_dict(self) -> dict[str, float]:
        out = {f"fit_{k}": v for k, v in self.fit.as_dict().items()}
        out.update(
            accuracy_ref=self.accuracy_ref,
            accuracy_test=self.accuracy_test,
            accuracy_predicted=self.accuracy_predicted,
            delta_mu_class0=self.delta_mu_class0,
            delta_mu_class1=self.delta_mu_class1,
            delta_error_class0=self.delta_error_class0,
            delta_error_class1=self.delta_error_class1,
        )
        return out


def compare_conditions(
    z_ref: np.ndarray, z_test: np.ndarray, y: np.ndarray
) -> ConditionComparison:
    """Compare a noisy condition (``z_test``, e.g. hardware) to a reference
    (``z_ref``, e.g. statevector) on the *same* samples ``y``.

    Returns the affine noise-channel fit (``lam, b, r2``) plus the per-class mean
    shift and per-class error change. ``accuracy_predicted`` applies the fitted
    channel to ``z_ref`` and thresholds it, closing the loop between the channel
    and the measured accuracy drop.
    """
    z_ref = np.asarray(z_ref, dtype=float)
    z_test = np.asarray(z_test, dtype=float)
    fit = fit_noise_channel(z_ref, z_test)
    m_ref = class_moments(z_ref, y)
    m_test = class_moments(z_test, y)
    err_ref0, err_ref1 = _class_error_fractions(z_ref, y)
    err_test0, err_test1 = _class_error_fractions(z_test, y)
    return ConditionComparison(
        fit=fit,
        accuracy_ref=accuracy_from_z(z_ref, y),
        accuracy_test=accuracy_from_z(z_test, y),
        accuracy_predicted=predicted_accuracy(z_ref, y, fit),
        delta_mu_class0=m_test["mu_class0"] - m_ref["mu_class0"],
        delta_mu_class1=m_test["mu_class1"] - m_ref["mu_class1"],
        delta_error_class0=err_test0 - err_ref0,
        delta_error_class1=err_test1 - err_ref1,
    )


# --------------------------------------------------------------------------- #
# 5. Pilot calibration helpers (shot stability, iteration precision)
# --------------------------------------------------------------------------- #
# The headline quantity is the noise-channel fit, so precision targets live on
# (lam, b) -- not on a raw expectation value. The shot pilot watches how much
# (lam, b) move when shots double; the iteration pilot watches the run-to-run
# spread of (lam, b) across repeated frozen runs.

# Student-t 0.975 quantiles (two-sided 95%) for small samples; df -> t.
_STUDENT_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}


def mean_half_width(
    values, *, use_t: bool = True, z: float = 1.96
) -> tuple[float, float]:
    """``(mean, 95% half-width)`` of repeated scalar estimates.

    ``half-width = t_{0.975, n-1} * s / sqrt(n)`` (Student-t for small ``n``;
    falls back to the normal ``z`` when ``use_t`` is False or ``n-1 > 10``).
    Non-finite values are dropped; returns ``(value, nan)`` for ``n == 1`` and
    ``(nan, nan)`` for ``n == 0``.
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(arr[0]), float("nan")
    s = float(np.std(arr, ddof=1))
    crit = _STUDENT_T_975.get(n - 1, z) if use_t else z
    return float(arr.mean()), float(crit * s / np.sqrt(n))


def shot_stability_table(shots, lams, bs) -> list[dict[str, float]]:
    """Per-consecutive-pair ``|delta lam|`` / ``|delta b|``, sorted by shots.

    Each row compares a shot budget with the next-larger one -- the doubling
    sensitivity the shot pilot thresholds against.
    """
    order = np.argsort(np.asarray(shots, dtype=float))
    s = np.asarray(shots, dtype=int)[order]
    lam = np.asarray(lams, dtype=float)[order]
    b = np.asarray(bs, dtype=float)[order]
    return [
        {
            "shots_from": int(s[i]),
            "shots_to": int(s[i + 1]),
            "delta_lambda": float(abs(lam[i + 1] - lam[i])),
            "delta_b": float(abs(b[i + 1] - b[i])),
        }
        for i in range(len(s) - 1)
    ]


def choose_shots(shots, lams, bs, *, tolerance: float) -> int:
    """Smallest shot budget whose doubling moves both ``lam`` and ``b`` by
    ``<= tolerance``.

    Returns the *smaller* budget of the first converged consecutive pair; if no
    pair converges, returns the largest budget in the grid.
    """
    order = np.argsort(np.asarray(shots, dtype=float))
    s = np.asarray(shots, dtype=int)[order]
    lam = np.asarray(lams, dtype=float)[order]
    b = np.asarray(bs, dtype=float)[order]
    for i in range(len(s) - 1):
        if abs(lam[i + 1] - lam[i]) <= tolerance and abs(b[i + 1] - b[i]) <= tolerance:
            return int(s[i])
    return int(s[-1])


def iteration_precision(
    lams, bs, *, use_t: bool = True
) -> dict[str, float]:
    """Mean and 95% half-width of ``lam`` and ``b`` across repeated runs."""
    lam_mean, lam_hw = mean_half_width(lams, use_t=use_t)
    b_mean, b_hw = mean_half_width(bs, use_t=use_t)
    return {
        "lambda_mean": lam_mean,
        "lambda_half_width": lam_hw,
        "b_mean": b_mean,
        "b_half_width": b_hw,
        "n_iterations": int(np.isfinite(np.asarray(lams, dtype=float)).sum()),
    }


def choose_iterations(
    lams,
    bs,
    *,
    target_half_width: float,
    min_iterations: int,
    max_iterations: int,
    use_t: bool = True,
) -> tuple[int, bool]:
    """Smallest ``K`` in ``[min, max]`` whose first-``K`` prefix puts both the
    ``lam`` and ``b`` mean half-widths ``<= target_half_width``.

    Returns ``(K, met)``; if the target is never met, ``K`` is the largest prefix
    available (capped at ``max_iterations``) and ``met`` is False.
    """
    lam = list(lams)
    b = list(bs)
    k_max = min(max_iterations, len(lam))
    k_min = max(min_iterations, 2)
    for k in range(k_min, k_max + 1):
        _, hw_lam = mean_half_width(lam[:k], use_t=use_t)
        _, hw_b = mean_half_width(b[:k], use_t=use_t)
        if hw_lam <= target_half_width and hw_b <= target_half_width:
            return k, True
    return max(k_max, k_min), False


# --------------------------------------------------------------------------- #
# 6. Tidy long-format row builder (for resumable CSV append)
# --------------------------------------------------------------------------- #
def confidence_rows(
    *,
    environment: str,
    depth: int,
    fold: int,
    condition: Condition,
    run_index: int,
    y: np.ndarray,
    z: np.ndarray,
    shots: int | float,
) -> list[dict[str, object]]:
    """One row per sample for ``confidence_distributions.csv``."""
    y = np.asarray(y)
    z = np.asarray(z, dtype=float)
    if y.shape[0] != z.shape[0]:
        raise ValueError(f"y/z length mismatch: {y.shape[0]} vs {z.shape[0]}")
    labels = predictions_to_labels(z)
    return [
        {
            "Environment": environment,
            "Depth": int(depth),
            "Fold": int(fold),
            "Condition": condition,
            "RunIndex": int(run_index),
            "SampleID": int(i),
            "TrueLabel": int(y[i]),
            "Shots": shots,
            "z": float(z[i]),
            "PredLabel": int(labels[i]),
        }
        for i in range(len(z))
    ]
