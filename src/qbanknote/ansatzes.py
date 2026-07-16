"""Canonical trimmed ansatze, star topology, and legacy ring variants."""

from __future__ import annotations

from collections.abc import Callable

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# Parameter counts for n_qubits=5 at depths 2/4/6 (trimmed vs legacy full ring).
TRIMMED_PARAM_COUNTS = {2: 17, 4: 37, 6: 57}
LEGACY_PARAM_COUNTS = {2: 20, 4: 40, 6: 60}


def trimmed_reverse_q0_param_count(n_qubits: int, depth: int) -> int:
    """Parameter count when only the last macro-layer trims reverse entanglement to q0 edges."""
    n_macro = depth // 2
    if n_macro == 0:
        return 0
    full = 4 * n_qubits
    last = 3 * n_qubits + 2
    return (n_macro - 1) * full + last


# Backwards-compatible alias used in notebooks.
ansatz_trimmed_reverse_q0_param_count = trimmed_reverse_q0_param_count


def odra_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    """Trimmed ODRA-native ansatz (RZ/CZ ring)."""
    n_macro = depth // 2
    theta = ParameterVector("theta", trimmed_reverse_q0_param_count(n_qubits, depth))
    qc = QuantumCircuit(n_qubits)
    p = 0

    for j in range(n_macro):
        last_layer = j == n_macro - 1

        for i in range(n_qubits):
            qc.ry(theta[p + i], i)
        p += n_qubits

        for i in range(n_qubits):
            control = i
            target = (i + 1) % n_qubits
            qc.rz(theta[p + i], target)
            qc.cz(control, target)
        p += n_qubits

        for i in range(n_qubits):
            qc.rx(theta[p + i], i)
        p += n_qubits

        if last_layer:
            for k in range(2):
                i = k
                control = i
                target = (i - 1) % n_qubits
                qc.ry(theta[p + k], target)
                qc.cz(control, target)
            p += 2
        else:
            for i in range(n_qubits):
                control = i
                target = (i - 1) % n_qubits
                qc.ry(theta[p + i], target)
                qc.cz(control, target)
            p += n_qubits

    assert p == len(theta)
    return qc


def simulator_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    """Trimmed simulator-optimized ansatz (CRX/CRY ring)."""
    n_macro = depth // 2
    theta = ParameterVector("theta", trimmed_reverse_q0_param_count(n_qubits, depth))
    qc = QuantumCircuit(n_qubits)
    param_idx = 0

    for j in range(n_macro):
        last_layer = j == n_macro - 1

        for i in range(n_qubits):
            qc.ry(theta[param_idx], i)
            param_idx += 1

        for i in range(n_qubits):
            control = i
            target = (i + 1) % n_qubits
            qc.crx(theta[param_idx], control, target)
            param_idx += 1

        for i in range(n_qubits):
            qc.rx(theta[param_idx], i)
            param_idx += 1

        if last_layer:
            for k in range(2):
                i = k
                control = i
                target = (i - 1) % n_qubits
                qc.cry(theta[param_idx], control, target)
                param_idx += 1
        else:
            for i in range(n_qubits):
                control = i
                target = (i - 1) % n_qubits
                qc.cry(theta[param_idx], control, target)
                param_idx += 1

    assert param_idx == len(theta)
    return qc


# Notebook naming aliases.
ansatz_odra = odra_ansatz
ansatz_simulator = simulator_ansatz
ansatz_Odra = odra_ansatz


def legacy_simulator_ring_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    """Full-parameter simulator ring ansatz (4n params per macro-layer)."""
    params_per_iter = 4 * n_qubits
    theta = ParameterVector("theta", params_per_iter * (depth // 2))
    qc = QuantumCircuit(n_qubits)
    param_idx = 0

    for _ in range(depth // 2):
        for i in range(n_qubits):
            qc.ry(theta[param_idx], i)
            param_idx += 1

        for i in range(n_qubits):
            control = i
            target = (i + 1) % n_qubits
            qc.crx(theta[param_idx], control, target)
            param_idx += 1

        for i in range(n_qubits):
            qc.rx(theta[param_idx], i)
            param_idx += 1

        for i in range(n_qubits):
            control = i
            target = (i - 1) % n_qubits
            qc.cry(theta[param_idx], control, target)
            param_idx += 1

    return qc


def legacy_odra_ring_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    """Full-parameter ODRA ring ansatz (4n params per macro-layer, RZ/CZ)."""
    params_per_iter = 4 * n_qubits
    theta = ParameterVector("theta", params_per_iter * (depth // 2))
    qc = QuantumCircuit(n_qubits)

    for j in range(depth // 2):
        offset = j * params_per_iter

        for i in range(n_qubits):
            qc.ry(theta[offset + i], i)

        for i in range(n_qubits):
            control = i
            target = (i + 1) % n_qubits
            param_idx = offset + n_qubits + i
            qc.rz(theta[param_idx], target)
            qc.cz(control, target)

        offset_l2 = offset + 2 * n_qubits

        for i in range(n_qubits):
            qc.rx(theta[offset_l2 + i], i)

        for i in range(n_qubits):
            control = i
            target = (i - 1) % n_qubits
            param_idx = offset_l2 + n_qubits + i
            qc.ry(theta[param_idx], target)
            qc.cz(control, target)

    return qc


def param_count_for_ansatz(ansatz_circuit: QuantumCircuit) -> int:
    return len(ansatz_circuit.parameters)


def expected_param_count(
    n_qubits: int,
    depth: int,
    *,
    variant: str = "trimmed",
    family: str = "simulator",
) -> int:
    """Return expected parameter count for a known ansatz variant."""
    if variant == "star":
        return star_param_count(n_qubits, depth)
    if variant == "trimmed":
        return trimmed_reverse_q0_param_count(n_qubits, depth)
    if variant == "legacy":
        return (depth // 2) * 4 * n_qubits
    raise ValueError(f"Unknown variant {variant!r}; expected 'star', 'trimmed', or 'legacy'.")


STAR_ANSATZ_NAME = "ansatz_star"
DEFAULT_STAR_ANSATZES = (STAR_ANSATZ_NAME,)

# Hub qubit for the star topology (QB2 on IQM Spark / Odra 5).
STAR_HUB_QUBIT = 2


def star_param_count(n_qubits: int, depth: int) -> int:
    """Parameter count for the star-topology ansatz at the given depth."""
    # Three rotation layers (Rz-Rx-Rz) per depth block, plus a final Rz-Rx layer.
    return n_qubits * depth * 3 + 2 * n_qubits


def star_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    """Star-topology ansatz for IQM Spark (hub qubit QB2)."""
    hub = STAR_HUB_QUBIT
    theta = ParameterVector("theta", star_param_count(n_qubits, depth))
    qc = QuantumCircuit(n_qubits)
    p = 0

    for _ in range(depth):
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

    # Final rotation layer (makes every CZ fan useful); the trailing Rz would
    # commute with the Z measurement, so only Rz + Rx are applied here.
    for i in range(n_qubits):
        qc.rz(theta[p + i], i)
    p += n_qubits
    for i in range(n_qubits):
        qc.rx(theta[p + i], i)
    p += n_qubits

    assert p == len(theta)
    return qc


def star_ansatz_registry() -> dict[str, Callable[[int, int], QuantumCircuit]]:
    """Return the QPU ansatz registry for star-only studies."""
    return {STAR_ANSATZ_NAME: star_ansatz}
