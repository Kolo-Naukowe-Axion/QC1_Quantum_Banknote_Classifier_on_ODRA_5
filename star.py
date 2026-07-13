"""Star-topology ansatz for IQM Spark (hub qubit QB2)."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def star_param_count(n_qubits: int, depth: int) -> int:
    """Parameter count for star_ansatz at the given depth."""
    return n_qubits * depth * 3 + 2 * n_qubits


def star_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    hub = 2
    # +2*n_qubits for the final rotation layer (Rz + Rx only; last Rz would commute with Z)
    total_params = star_param_count(n_qubits, depth)
    theta = ParameterVector("theta", total_params)
    qc = QuantumCircuit(n_qubits)
    p = 0

    for d in range(depth):
        for i in range(n_qubits):
            qc.rz(theta[p + i], i)
        p += n_qubits
        for i in range(n_qubits):
            qc.rx(theta[p + i], i)
        p += n_qubits
        for i in range(n_qubits):
            qc.rz(theta[p + i], i)
        p += n_qubits
        for target in range(n_qubits):
            if target == hub:
                continue
            qc.cz(hub, target)
        qc.barrier()

    # Final rotation layer (makes every CZ fan useful)
    for i in range(n_qubits):
        qc.rz(theta[p + i], i)
    p += n_qubits
    for i in range(n_qubits):
        qc.rx(theta[p + i], i)
    p += n_qubits

    assert p == len(theta)
    return qc
