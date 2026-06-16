"""Meyer-Wallach entanglement and KL expressibility metrics."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit

from qbanknote.ansatzes import trimmed_reverse_q0_param_count
from qbanknote.iqm import run_circuits_on_backend, transpile_for_backend
from qbanknote.tomography import (
    add_tomography_rotations,
    all_basis_settings,
    expectation_from_counts,
    hardware_overlap,
    project_to_physical,
    reconstruct_rho,
    tomography_density_matrices,
)

BasisName = Literal["Z", "X", "Y"]
BASIS_ORDER: tuple[BasisName, ...] = ("Z", "X", "Y")


# ---------------------------------------------------------------------------
# Meyer–Wallach
# ---------------------------------------------------------------------------


def single_qubit_reduced_density(
    state: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    arr = state.reshape((2,) * n_qubits)
    arr = np.moveaxis(arr, qubit, 0)
    psi_mat = arr.reshape(2, -1)
    return psi_mat @ psi_mat.conj().T


def meyer_wallach_score(state: np.ndarray, n_qubits: int) -> float:
    if n_qubits < 1:
        return 0.0
    acc = 0.0
    for i in range(n_qubits):
        rho_i = single_qubit_reduced_density(state, i, n_qubits)
        purity = float(np.real(np.trace(rho_i @ rho_i)))
        acc += 1.0 - purity
    q = (2.0 / n_qubits) * acc
    return float(max(0.0, min(1.0, q)))


def bitstring_qubit_value(bitstring: str, qubit: int, n_qubits: int) -> str:
    if len(bitstring) != n_qubits:
        raise ValueError(f"Expected bitstring length {n_qubits}, got {len(bitstring)!r}")
    return bitstring[-(qubit + 1)]


def qubit_expectation_from_counts(
    counts: dict[str, int], qubit: int, n_qubits: int
) -> float:
    shots = sum(counts.values())
    if shots == 0:
        return 0.0
    expval = 0.0
    for bitstring, count in counts.items():
        bit = bitstring_qubit_value(bitstring, qubit, n_qubits)
        eigenvalue = 1.0 if bit == "0" else -1.0
        expval += eigenvalue * count / shots
    return float(expval)


def mw_score_from_bloch(
    x_expectations: list[float],
    y_expectations: list[float],
    z_expectations: list[float],
) -> float:
    n_qubits = len(x_expectations)
    if n_qubits < 1:
        return 0.0
    acc = 0.0
    for x_i, y_i, z_i in zip(x_expectations, y_expectations, z_expectations):
        purity = 0.5 * (1.0 + x_i * x_i + y_i * y_i + z_i * z_i)
        acc += 1.0 - purity
    q = (2.0 / n_qubits) * acc
    return float(max(0.0, min(1.0, q)))


def add_basis_measurement(circuit: QuantumCircuit, basis: BasisName) -> QuantumCircuit:
    qc = circuit.copy()
    if basis == "X":
        for i in range(qc.num_qubits):
            qc.h(i)
    elif basis == "Y":
        for i in range(qc.num_qubits):
            qc.sdg(i)
            qc.h(i)
    qc.measure_all()
    return qc


def estimate_mw_from_hardware_counts(
    counts_by_basis: dict[BasisName, dict[str, int]],
    n_qubits: int,
) -> tuple[float, list[float], list[float], list[float]]:
    x_exp = [
        qubit_expectation_from_counts(counts_by_basis["X"], i, n_qubits)
        for i in range(n_qubits)
    ]
    y_exp = [
        qubit_expectation_from_counts(counts_by_basis["Y"], i, n_qubits)
        for i in range(n_qubits)
    ]
    z_exp = [
        qubit_expectation_from_counts(counts_by_basis["Z"], i, n_qubits)
        for i in range(n_qubits)
    ]
    score = mw_score_from_bloch(x_exp, y_exp, z_exp)
    return score, x_exp, y_exp, z_exp


def _group_counts_by_sample(
    counts_list: list[dict[str, int]],
) -> list[dict[BasisName, dict[str, int]]]:
    if len(counts_list) % len(BASIS_ORDER) != 0:
        raise ValueError("Counts list length must be a multiple of 3 (Z, X, Y per sample)")
    grouped: list[dict[BasisName, dict[str, int]]] = []
    for i in range(0, len(counts_list), len(BASIS_ORDER)):
        grouped.append(
            {
                "Z": counts_list[i],
                "X": counts_list[i + 1],
                "Y": counts_list[i + 2],
            }
        )
    return grouped


def compute_iqm_mw_scores(
    backend,
    ansatz_fn: Callable[[int, int], QuantumCircuit],
    n_qubits: int,
    depth: int,
    n_samples: int,
    seed: int,
    shots: int,
    optimization_level: int,
    seed_transpiler: int | None,
    max_circuits_per_job: int,
) -> dict[str, object]:
    qc = ansatz_fn(n_qubits, depth)
    params = list(qc.parameters)
    rng = np.random.default_rng(seed)

    scores: list[float] = []
    bloch_rows: list[dict[str, float]] = []
    pending_circuits: list[QuantumCircuit] = []
    pending_sample_indices: list[int] = []

    def flush_batch() -> None:
        nonlocal pending_circuits, pending_sample_indices
        if not pending_circuits:
            return
        counts_list = run_circuits_on_backend(
            backend,
            pending_circuits,
            shots=shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            max_circuits_per_job=max_circuits_per_job,
        )
        grouped = _group_counts_by_sample(counts_list)
        if len(grouped) != len(pending_sample_indices):
            raise RuntimeError(
                f"Batch mismatch: {len(grouped)} sample groups vs "
                f"{len(pending_sample_indices)} sample indices"
            )
        for sample_index, basis_counts in zip(pending_sample_indices, grouped):
            score, x_exp, y_exp, z_exp = estimate_mw_from_hardware_counts(
                basis_counts, n_qubits
            )
            scores.append(score)
            row: dict[str, float] = {"sample_index": float(sample_index), "mw_score": score}
            for i in range(n_qubits):
                row[f"x_q{i}"] = x_exp[i]
                row[f"y_q{i}"] = y_exp[i]
                row[f"z_q{i}"] = z_exp[i]
            bloch_rows.append(row)
        pending_circuits = []
        pending_sample_indices = []

    for sample_index in range(n_samples):
        values = rng.uniform(0.0, 2.0 * math.pi, size=len(params))
        bound = qc.assign_parameters(dict(zip(params, values)), inplace=False)
        for basis in BASIS_ORDER:
            pending_circuits.append(add_basis_measurement(bound, basis))
        pending_sample_indices.append(sample_index)
        if len(pending_circuits) >= max_circuits_per_job:
            flush_batch()
    flush_batch()

    if len(scores) != n_samples:
        raise RuntimeError(f"Expected {n_samples} MW scores, got {len(scores)}")

    arr = np.array(scores, dtype=float)
    return {
        "mw_scores": scores,
        "bloch_rows": bloch_rows,
        "mw_avg": float(np.mean(arr)),
        "mw_std": float(np.std(arr)),
        "mw_sem": float(np.std(arr) / math.sqrt(n_samples)),
        "mw_min": float(np.min(arr)),
        "mw_max": float(np.max(arr)),
        "depth": depth,
        "n_qubits": n_qubits,
        "n_params": len(params),
        "n_samples": n_samples,
    }


def verify_mw_implementation() -> None:
    n = 3
    product = np.zeros(2**n, dtype=complex)
    product[0] = 1.0
    assert abs(meyer_wallach_score(product, n)) < 1e-9

    ghz = np.zeros(2**n, dtype=complex)
    ghz[0] = 1.0 / math.sqrt(2.0)
    ghz[-1] = 1.0 / math.sqrt(2.0)
    assert abs(meyer_wallach_score(ghz, n) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# KL expressibility
# ---------------------------------------------------------------------------


def haar_pdf_fidelity(f: np.ndarray, dim: int) -> np.ndarray:
    return (dim - 1.0) * (1.0 - f) ** (dim - 2.0)


def binned_distributions(
    fid_values: np.ndarray,
    dim: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    counts, edges = np.histogram(fid_values, bins=bins, density=False)
    p_emp = counts.astype(np.float64)
    if p_emp.sum() == 0:
        p_emp = np.ones_like(p_emp) / len(p_emp)
    else:
        p_emp /= p_emp.sum()

    mids = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    p_haar = haar_pdf_fidelity(mids, dim=dim) * width
    p_haar /= p_haar.sum()
    return edges, mids, p_emp, p_haar


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p_s = p + eps
    q_s = q + eps
    p_s /= p_s.sum()
    q_s /= q_s.sum()
    return float(np.sum(p_s * np.log(p_s / q_s)))


def bind_ansatz(
    ansatz_fn: Callable[[int, int], QuantumCircuit],
    n_qubits: int,
    depth: int,
    theta_values: np.ndarray,
) -> QuantumCircuit:
    qc = ansatz_fn(n_qubits, depth)
    ordered_params = list(qc.parameters)
    bind_map = {p: float(v) for p, v in zip(ordered_params, theta_values)}
    return qc.assign_parameters(bind_map, inplace=False)


def sample_hardware_fidelities(
    backend,
    ansatz_fn: Callable[[int, int], QuantumCircuit],
    n_qubits: int,
    depth: int,
    n_samples: int,
    seed: int,
    shots: int,
    optimization_level: int,
    seed_transpiler: int | None,
    max_circuits_per_job: int,
    ansatz_label: str = "",
) -> list[dict[str, object]]:
    qc_template = ansatz_fn(n_qubits, depth)
    n_params = len(qc_template.parameters)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for sample_index in range(n_samples):
        theta_a = rng.uniform(0.0, 2.0 * np.pi, n_params)
        theta_b = rng.uniform(0.0, 2.0 * np.pi, n_params)
        bound_a = bind_ansatz(ansatz_fn, n_qubits, depth, theta_a)
        bound_b = bind_ansatz(ansatz_fn, n_qubits, depth, theta_b)

        print(
            f"\n  sample {sample_index + 1}/{n_samples} "
            f"({ansatz_label}, depth={depth})"
        )

        rho_a_lin, rho_a_phys, diag_a = tomography_density_matrices(
            bound_a,
            backend,
            n_qubits=n_qubits,
            shots=shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            max_circuits_per_job=max_circuits_per_job,
            label=f"{ansatz_label} state A",
        )
        rho_b_lin, rho_b_phys, diag_b = tomography_density_matrices(
            bound_b,
            backend,
            n_qubits=n_qubits,
            shots=shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            max_circuits_per_job=max_circuits_per_job,
            label=f"{ansatz_label} state B",
        )

        f_lin = hardware_overlap(rho_a_lin, rho_b_lin)
        f_phys = hardware_overlap(rho_a_phys, rho_b_phys)

        print(
            f"    F (linear inv.)    = {f_lin:.4f}\n"
            f"    F (physical proj.) = {f_phys:.4f}"
        )

        rows.append(
            {
                "sample_index": sample_index,
                "fidelity_linear": f_lin,
                "fidelity_physical": f_phys,
                "trace_a_linear": diag_a["trace_linear"],
                "trace_a_physical": diag_a["trace_physical"],
                "trace_b_linear": diag_b["trace_linear"],
                "trace_b_physical": diag_b["trace_physical"],
                "purity_a_linear": diag_a["purity_linear"],
                "purity_a_physical": diag_a["purity_physical"],
                "purity_b_linear": diag_b["purity_linear"],
                "purity_b_physical": diag_b["purity_physical"],
            }
        )

    return rows


def compute_kl_for_fidelities(
    fidelities: np.ndarray,
    dim: int,
    n_bins: int,
    eps: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    _, mids, p_emp, p_haar = binned_distributions(fidelities, dim, n_bins)
    kl = kl_divergence(p_emp, p_haar, eps)
    return kl, mids, p_emp, p_haar


def circuits_per_fidelity_sample(n_qubits: int) -> int:
    return 2 * (3**n_qubits)


def total_expressibility_circuits(
    n_ansatzes: int,
    n_depths: int,
    n_samples: int,
    n_qubits: int,
) -> int:
    return n_ansatzes * n_depths * n_samples * circuits_per_fidelity_sample(n_qubits)


def estimate_wall_time_minutes(
    n_ansatzes: int,
    n_depths: int,
    n_samples: int,
    minutes_per_state: float,
) -> float:
    n_pairs = n_ansatzes * n_depths * n_samples
    return n_pairs * 2.0 * minutes_per_state


# ---------------------------------------------------------------------------
# Self-checks (no hardware)
# ---------------------------------------------------------------------------


def verify_expectation_endianness() -> None:
    counts_all_zero = {"00000": 1000}
    assert abs(expectation_from_counts(counts_all_zero, "ZZZZZ") - 1.0) < 1e-12
    counts_q0_one = {"00001": 1000}
    assert abs(expectation_from_counts(counts_q0_one, "ZZZZZ") - (-1.0)) < 1e-12
    assert abs(expectation_from_counts(counts_q0_one, "IZZZI") - 1.0) < 1e-12


def verify_projection_psd() -> None:
    dim = 8
    rng = np.random.default_rng(1)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    rho_noisy = rho + 0.01 * (
        rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    )
    rho_noisy = 0.5 * (rho_noisy + rho_noisy.conj().T)
    rho_phys = project_to_physical(rho_noisy)
    eigvals = np.linalg.eigvalsh(rho_phys)
    assert np.all(eigvals >= -1e-10)
    trace = float(np.real(np.trace(rho_phys)))
    assert 0.0 < trace <= 1.0 + 1e-8


def verify_haar_kl_helpers(num_qubits: int = 5, n_bins: int = 20, eps: float = 1e-12) -> None:
    dim = 2**num_qubits
    rng = np.random.default_rng(0)
    haar_samples = 1.0 - rng.random(50_000) ** (1.0 / (dim - 1))
    kl, _, _, _ = compute_kl_for_fidelities(haar_samples, dim, n_bins, eps)
    assert kl < 0.05, f"Haar self-samples should have low KL, got {kl}"


def verify_statevector_vs_tomography_overlap() -> None:
    from qiskit.quantum_info import Statevector

    n_qubits = 2
    qc_a = QuantumCircuit(n_qubits)
    qc_a.h(0)
    qc_a.cx(0, 1)
    qc_b = QuantumCircuit(n_qubits)
    qc_b.ry(0.7, 0)
    qc_b.rz(1.1, 1)

    psi_a = Statevector.from_instruction(qc_a).data
    psi_b = Statevector.from_instruction(qc_b).data
    ideal = abs(np.vdot(psi_a, psi_b)) ** 2

    simulated_counts: dict[tuple[str, ...], dict[str, int]] = {}
    for basis_tuple in all_basis_settings(n_qubits):
        qc = qc_b.copy()
        tomo = add_tomography_rotations(qc, basis_tuple)
        sv = Statevector.from_instruction(tomo.remove_final_measurements(inplace=False))
        probs = sv.probabilities_dict()
        counts = {k.replace(" ", ""): int(round(v * 10_000)) for k, v in probs.items()}
        simulated_counts[basis_tuple] = counts

    rho_lin = reconstruct_rho(simulated_counts, n_qubits)
    rho_phys = project_to_physical(rho_lin)
    rho_a = np.outer(psi_a, psi_a.conj())
    tomography_overlap = hardware_overlap(rho_a, rho_phys)
    assert abs(tomography_overlap - ideal) < 0.05, (
        f"tomography overlap {tomography_overlap} vs ideal {ideal}"
    )


def verify_count_expectation_helper() -> None:
    n = 5
    counts_all_zero = {"00000": 1000}
    for q in range(n):
        assert abs(qubit_expectation_from_counts(counts_all_zero, q, n) - 1.0) < 1e-12

    counts_q0_one = {"00001": 1000}
    assert abs(qubit_expectation_from_counts(counts_q0_one, 0, n) - (-1.0)) < 1e-12
    assert abs(qubit_expectation_from_counts(counts_q0_one, 1, n) - 1.0) < 1e-12

    counts_by_basis: dict[BasisName, dict[str, int]] = {
        "Z": counts_all_zero,
        "X": counts_all_zero,
        "Y": counts_all_zero,
    }
    assert abs(estimate_mw_from_hardware_counts(counts_by_basis, n)[0]) < 1e-12


def verify_bloch_mw_matches_statevector() -> None:
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

    n = 5
    rng = np.random.default_rng(0)
    for _ in range(20):
        psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        psi /= np.linalg.norm(psi)
        sv_mw = meyer_wallach_score(psi, n)
        x_exp: list[float] = []
        y_exp: list[float] = []
        z_exp: list[float] = []
        for i in range(n):
            rho_i = single_qubit_reduced_density(psi, i, n)
            x_exp.append(float(np.real(np.trace(rho_i @ pauli_x))))
            y_exp.append(float(np.real(np.trace(rho_i @ pauli_y))))
            z_exp.append(float(np.real(np.trace(rho_i @ pauli_z))))
        bloch_mw = mw_score_from_bloch(x_exp, y_exp, z_exp)
        assert abs(sv_mw - bloch_mw) < 1e-9, f"sv={sv_mw}, bloch={bloch_mw}"


def run_kl_self_check(num_qubits: int = 5, n_bins: int = 20, eps: float = 1e-12) -> None:
    verify_haar_kl_helpers(num_qubits, n_bins, eps)
    verify_expectation_endianness()
    verify_projection_psd()
    verify_statevector_vs_tomography_overlap()


def run_mw_self_check() -> None:
    verify_mw_implementation()
    verify_count_expectation_helper()
    verify_bloch_mw_matches_statevector()


def trimmed_param_counts_for_depths(
    n_qubits: int, depths: list[int]
) -> dict[int, int]:
    return {depth: trimmed_reverse_q0_param_count(n_qubits, depth) for depth in depths}
