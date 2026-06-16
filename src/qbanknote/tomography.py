"""State tomography helpers shared by fidelity and expressibility notebooks."""

from __future__ import annotations

import time
from itertools import product

import numpy as np
from qiskit import QuantumCircuit

from qbanknote.iqm import normalize_counts, transpile_for_backend
from qbanknote.model import angle_encoding_feature_map

_PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": _PAULI_I, "X": _PAULI_X, "Y": _PAULI_Y, "Z": _PAULI_Z}


def all_basis_settings(n_qubits: int) -> list[tuple[str, ...]]:
    return list(product(["X", "Y", "Z"], repeat=n_qubits))


def add_tomography_rotations(
    circuit: QuantumCircuit, bases: tuple[str, ...]
) -> QuantumCircuit:
    qc = circuit.copy()
    for qubit, basis in enumerate(bases):
        if basis == "X":
            qc.h(qubit)
        elif basis == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
        elif basis == "Z":
            pass
        else:
            raise ValueError(f"Unknown basis {basis!r}, expected X / Y / Z")
    qc.measure_all()
    return qc


def expectation_from_counts(counts: dict[str, int], pauli_string: str) -> float:
    shots = sum(counts.values())
    if shots == 0:
        return 0.0
    expval = 0.0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")[::-1]
        value = 1
        for pauli, bit in zip(pauli_string, bits):
            if pauli == "I":
                continue
            value *= 1 if bit == "0" else -1
        expval += value * count / shots
    return float(expval)


def _kron_all(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def reconstruct_rho(
    tomography_counts: dict[tuple[str, ...], dict[str, int]],
    n_qubits: int,
) -> np.ndarray:
    dim = 2**n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    for pauli_tuple in product("IXYZ", repeat=n_qubits):
        pauli_string = "".join(pauli_tuple)
        basis_tuple = tuple(p if p != "I" else "Z" for p in pauli_tuple)
        counts = tomography_counts[basis_tuple]
        expval = expectation_from_counts(counts, pauli_string)
        P = _kron_all([PAULI[p] for p in reversed(pauli_tuple)])
        rho += expval * P
    rho /= dim
    return rho


def project_to_physical(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + rho.conj().T)
    eigvals, _ = np.linalg.eigh(rho)
    eigvals = np.sort(eigvals)[::-1]
    accum = 0.0
    n = len(eigvals)
    proj = np.zeros_like(eigvals)
    for i in range(n - 1, -1, -1):
        ev = eigvals[i] + accum / (i + 1)
        if ev >= 0:
            for j in range(i + 1):
                proj[j] = eigvals[j] + accum / (i + 1)
            break
        accum += eigvals[i]

    eigvals_unsorted, eigvecs_unsorted = np.linalg.eigh(0.5 * (rho + rho.conj().T))
    order = np.argsort(eigvals_unsorted)[::-1]
    proj_in_orig_order = np.empty_like(eigvals_unsorted)
    proj_in_orig_order[order] = proj
    return eigvecs_unsorted @ np.diag(proj_in_orig_order) @ eigvecs_unsorted.conj().T


def hardware_overlap(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    return float(np.real(np.trace(rho_a @ rho_b)))


def state_fidelity_pure(psi_ideal: np.ndarray, rho: np.ndarray) -> float:
    psi = np.asarray(psi_ideal, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    return float(np.real(np.conj(psi) @ rho @ psi))


def build_bound_circuit(
    ansatz_circuit: QuantumCircuit,
    x_value: np.ndarray,
    weight_values: np.ndarray,
    n_qubits: int,
) -> QuantumCircuit:
    """Compose angle encoding + ansatz and bind feature/weight parameters."""
    feature_map = angle_encoding_feature_map(n_qubits)
    full = QuantumCircuit(n_qubits)
    full.compose(feature_map, qubits=range(n_qubits), inplace=True)
    full.compose(ansatz_circuit, inplace=True)

    binding = {p: float(v) for p, v in zip(feature_map.parameters, x_value)}
    binding.update(
        {p: float(v) for p, v in zip(ansatz_circuit.parameters, weight_values)}
    )
    return full.assign_parameters(binding)


def bind_circuit(
    qc: QuantumCircuit,
    input_params,
    weight_params,
    x_value: np.ndarray,
    weight_values: np.ndarray,
) -> QuantumCircuit:
    """Bind separate input and weight parameter lists (fidelity notebook API)."""
    binding = {p: float(v) for p, v in zip(input_params, x_value)}
    binding.update({p: float(v) for p, v in zip(weight_params, weight_values)})
    return qc.assign_parameters(binding)


def run_tomography_jobs(
    state_circuit: QuantumCircuit,
    backend,
    n_qubits: int,
    shots: int,
    optimization_level: int,
    seed_transpiler: int | None,
    max_circuits_per_job: int,
    label: str = "",
) -> dict[tuple[str, ...], dict[str, int]]:
    bases = all_basis_settings(n_qubits)
    tomo_circuits = [add_tomography_rotations(state_circuit, b) for b in bases]
    transpiled = [
        transpile_for_backend(qc, backend, optimization_level, seed_transpiler)
        for qc in tomo_circuits
    ]

    if label:
        print(
            f"  [{label}] submitting {len(transpiled)} tomography circuits "
            f"in batches of {max_circuits_per_job} ({shots} shots each)..."
        )

    counts_per_basis: dict[tuple[str, ...], dict[str, int]] = {}
    submitted = 0
    while submitted < len(transpiled):
        batch = transpiled[submitted : submitted + max_circuits_per_job]
        batch_bases = bases[submitted : submitted + max_circuits_per_job]
        t0 = time.perf_counter()
        result = backend.run(batch, shots=shots).result()
        dt = time.perf_counter() - t0
        counts_list = result.get_counts()
        if not isinstance(counts_list, list):
            counts_list = [counts_list]
        if len(counts_list) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} count dicts, backend returned {len(counts_list)}"
            )
        for j, basis_tuple in enumerate(batch_bases):
            counts_per_basis[basis_tuple] = normalize_counts(counts_list[j])
        submitted += len(batch)
        if label:
            print(
                f"    batch done: {submitted:>4}/{len(transpiled)}  ({dt:.1f}s on backend)"
            )

    return counts_per_basis


def tomography_density_matrices(
    state_circuit: QuantumCircuit,
    backend,
    n_qubits: int,
    shots: int,
    optimization_level: int,
    seed_transpiler: int | None,
    max_circuits_per_job: int,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    counts = run_tomography_jobs(
        state_circuit,
        backend,
        n_qubits=n_qubits,
        shots=shots,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        max_circuits_per_job=max_circuits_per_job,
        label=label,
    )
    rho_lin = reconstruct_rho(counts, n_qubits)
    rho_phys = project_to_physical(rho_lin)
    diagnostics = {
        "trace_linear": float(np.real(np.trace(rho_lin))),
        "trace_physical": float(np.real(np.trace(rho_phys))),
        "purity_linear": float(np.real(np.trace(rho_lin @ rho_lin))),
        "purity_physical": float(np.real(np.trace(rho_phys @ rho_phys))),
    }
    return rho_lin, rho_phys, diagnostics
