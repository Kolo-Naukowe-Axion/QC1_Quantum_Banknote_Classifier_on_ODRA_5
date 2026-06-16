"""Circuit depth and gate-count utilities."""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile


def count_gate_types(circuit: QuantumCircuit) -> dict[str, object]:
    total_gates = 0
    single_qubit_gates = 0
    two_qubit_gates = 0
    single_qubit_gate_counts = {qubit: 0 for qubit in range(circuit.num_qubits)}
    two_qubit_gate_counts = {
        (q1, q2): 0
        for q1 in range(circuit.num_qubits)
        for q2 in range(q1 + 1, circuit.num_qubits)
    }

    for instruction in circuit.data:
        operation = (
            instruction.operation if hasattr(instruction, "operation") else instruction[0]
        )
        qubits = instruction.qubits if hasattr(instruction, "qubits") else instruction[1]

        if operation.name in {"barrier", "measure"}:
            continue

        qubit_indices = tuple(circuit.find_bit(qubit).index for qubit in qubits)
        total_gates += 1

        if operation.num_qubits == 1:
            single_qubit_gates += 1
            single_qubit_gate_counts[qubit_indices[0]] += 1
        elif operation.num_qubits == 2:
            two_qubit_gates += 1
            pair = tuple(sorted(qubit_indices))
            two_qubit_gate_counts[pair] += 1

    return {
        "Total Gates": total_gates,
        "Single-Qubit Gates": single_qubit_gates,
        "Two-Qubit Gates": two_qubit_gates,
        "Single-Qubit Gates by Qubit": single_qubit_gate_counts,
        "Two-Qubit Gates by Pair": two_qubit_gate_counts,
    }


def print_gate_breakdown(stats: dict[str, object]) -> None:
    print(f"Total Gates:        {stats['Total Gates']}")
    print(f"Single-Qubit Gates: {stats['Single-Qubit Gates']}")
    print(f"Two-Qubit Gates:    {stats['Two-Qubit Gates']}")
    print("Single-Qubit Gates on Each Qubit:")
    for qubit, count in stats["Single-Qubit Gates by Qubit"].items():
        print(f"  q{qubit}: {count}")
    print("Two-Qubit Gates by Pair:")
    for (q1, q2), count in stats["Two-Qubit Gates by Pair"].items():
        print(f"  q{q1}-q{q2}: {count}")


def get_circuit_stats(circuit: QuantumCircuit, backend, *, return_transpiled: bool = False):
    """
    Transpile ``circuit`` for ``backend`` and return depth/SWAP/gate statistics.

    When ``return_transpiled`` is True, returns ``(transpiled_circuit, stats)``.
    """
    t_qc = transpile(circuit, backend, optimization_level=0)
    ops = t_qc.count_ops()
    stats = {
        "Depth": t_qc.depth(),
        "SWAPs": ops.get("swap", 0),
        "CNOTs/CZs": ops.get("cz", 0) + ops.get("cx", 0),
    }
    stats.update(count_gate_types(t_qc))
    if return_transpiled:
        return t_qc, stats
    return stats
