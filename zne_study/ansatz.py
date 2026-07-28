"""VQC ansatz for the ZNE / SWAP-overhead study.

A hardware-efficient, data-reuploading variational classifier built with the
IQM-native gate set (RX/RY/RZ + CZ) and the Odra star hub convention
(hub = QB2), consistent with ``qbanknote.ansatzes``.

Unlike the pure-state ansatze in ``qbanknote.ansatzes`` (used for fidelity /
expressibility studies), this circuit carries *two* parameter groups:

* ``x``     -- input features (data re-uploaded on every layer), bound per sample.
* ``theta`` -- trainable weights, trained once in simulation and then frozen.

The entangling pattern is the primary experimental knob for the study:

* ``"star"``       -- CZ only between the hub and each outer qubit. Native to the
                      Odra star topology -> (near) zero routing / SWAP overhead.
* ``"all_to_all"`` -- CZ between every qubit pair. On the star device this forces
                      the transpiler to insert SWAPs to connect non-hub pairs ->
                      large, controllable two-qubit-gate overhead.
* ``"linear"`` / ``"ring"`` -- intermediate patterns for finer sweeps.

Both parameter groups are returned separately so the circuit plugs directly into
``qiskit_machine_learning`` (EstimatorQNN: ``input_params`` vs ``weight_params``).
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp

try:  # keep in sync with the rest of the project, but stay importable standalone.
    from qbanknote.ansatzes import STAR_HUB_QUBIT as _DEFAULT_HUB
except Exception:  # pragma: no cover - fallback when package not on path.
    _DEFAULT_HUB = 2

ENTANGLEMENT_PATTERNS = ("star", "all_to_all", "linear", "ring")


@dataclass(frozen=True)
class VQCSpec:
    """Fully describes a built VQC circuit and its parameter layout."""

    circuit: QuantumCircuit
    feature_params: ParameterVector
    weight_params: ParameterVector
    n_qubits: int
    n_features: int
    depth: int
    entanglement: str
    hub: int
    feature_qubits: tuple[int, ...]
    readout_qubit: int

    @property
    def num_cz(self) -> int:
        """Number of (logical) CZ gates in the circuit -- the routing-cost proxy."""
        return sum(1 for inst in self.circuit.data if inst.operation.name == "cz")


def _entangling_pairs(
    n_qubits: int, hub: int, entanglement: str
) -> list[tuple[int, int]]:
    """Return the ordered list of (control, target) CZ edges for a pattern."""
    if entanglement == "star":
        return [(hub, q) for q in range(n_qubits) if q != hub]
    if entanglement == "all_to_all":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if entanglement == "ring":
        edges = [(i, i + 1) for i in range(n_qubits - 1)]
        if n_qubits > 2:
            edges.append((n_qubits - 1, 0))
        return edges
    raise ValueError(
        f"Unknown entanglement {entanglement!r}; expected one of {ENTANGLEMENT_PATTERNS}."
    )


def build_vqc(
    n_qubits: int = 5,
    n_features: int = 4,
    depth: int = 2,
    entanglement: str = "star",
    *,
    hub: int | None = None,
    feature_qubits: tuple[int, ...] | None = None,
    readout_qubit: int | None = None,
    add_barriers: bool = True,
) -> VQCSpec:
    """Build a data-reuploading hardware-efficient VQC.

    Parameters
    ----------
    n_qubits:
        Total qubits used (default 5 = full Odra star).
    n_features:
        Number of input features (default 4, e.g. Iris / two_curves d=4).
        Features are pre-scaled classically to roughly [-pi/2, pi/2] before binding.
    depth:
        Number of re-upload layers (encoding + variational + entangler).
    entanglement:
        One of ``ENTANGLEMENT_PATTERNS``. This is the SWAP-overhead knob.
    hub:
        Central qubit for the star pattern (defaults to the project hub, QB2 = 2).
    feature_qubits:
        Which qubits carry the encoded features. Defaults to all non-hub qubits so
        the hub acts purely as the router (matches the "4 features on 4 outer
        qubits, hub routes" design). Features cycle if there are fewer qubits.
    readout_qubit:
        Qubit measured for the class prediction (defaults to the hub).
    add_barriers:
        Insert a barrier after each entangling block (keeps layers visually and
        structurally separated; also helps prevent later passes from merging
        layers unexpectedly).

    Returns
    -------
    VQCSpec
        Circuit plus its feature/weight parameter vectors and metadata.
    """
    if n_qubits < 2:
        raise ValueError("n_qubits must be >= 2 to have any entanglement.")
    hub = _DEFAULT_HUB if hub is None else hub
    if not 0 <= hub < n_qubits:
        hub = 0  # hub out of range for small circuits -> fall back to qubit 0.

    if feature_qubits is None:
        outer = tuple(q for q in range(n_qubits) if q != hub)
        feature_qubits = outer if outer else tuple(range(n_qubits))
    if readout_qubit is None:
        readout_qubit = hub

    x = ParameterVector("x", n_features)
    theta = ParameterVector("theta", 2 * n_qubits * depth)

    qc = QuantumCircuit(n_qubits, name=f"vqc_{entanglement}_d{depth}")
    edges = _entangling_pairs(n_qubits, hub, entanglement)
    p = 0

    for _layer in range(depth):
        # (1) Data encoding, re-uploaded every layer. Feature f -> its qubit.
        for idx, q in enumerate(feature_qubits):
            qc.rx(x[idx % n_features], q)

        # (2) Trainable single-qubit rotations on all qubits.
        for q in range(n_qubits):
            qc.ry(theta[p + q], q)
        p += n_qubits
        for q in range(n_qubits):
            qc.rz(theta[p + q], q)
        p += n_qubits

        # (3) Entangling block -- the routing-cost knob.
        for control, target in edges:
            qc.cz(control, target)

        if add_barriers:
            qc.barrier()

    assert p == len(theta), (p, len(theta))

    return VQCSpec(
        circuit=qc,
        feature_params=x,
        weight_params=theta,
        n_qubits=n_qubits,
        n_features=n_features,
        depth=depth,
        entanglement=entanglement,
        hub=hub,
        feature_qubits=tuple(feature_qubits),
        readout_qubit=readout_qubit,
    )


def default_observable(n_qubits: int, readout_qubit: int) -> SparsePauliOp:
    """Return <Z> on the readout qubit as the classification observable.

    The sign of this expectation value is the predicted class (+/-1); ZNE is
    applied to the raw expectation value *before* the decision rule.
    """
    return SparsePauliOp.from_sparse_list(
        [("Z", [readout_qubit], 1.0)], num_qubits=n_qubits
    )


def weight_count(n_qubits: int, depth: int) -> int:
    """Trainable-parameter count (excludes data features)."""
    return 2 * n_qubits * depth


if __name__ == "__main__":
    # Sanity check: build across n = 2..5 and confirm the SWAP-overhead ordering.
    print(f"{'n':>2} {'depth':>5} {'pattern':>11} {'#weights':>9} {'#CZ':>5}")
    print("-" * 38)
    for n in range(2, 6):
        for pattern in ENTANGLEMENT_PATTERNS:
            spec = build_vqc(n_qubits=n, n_features=min(4, n), depth=2, entanglement=pattern)
            assert len(spec.weight_params) == weight_count(n, 2)
            assert spec.circuit.num_qubits == n
            print(f"{n:>2} {2:>5} {pattern:>11} {len(spec.weight_params):>9} {spec.num_cz:>5}")

    # Core invariant for the study: all_to_all must never route cheaper than star.
    for n in range(3, 6):
        star = build_vqc(n_qubits=n, depth=2, entanglement="star").num_cz
        full = build_vqc(n_qubits=n, depth=2, entanglement="all_to_all").num_cz
        assert full >= star, (n, full, star)
    print("\nOK: circuits build for n=2..5 and all_to_all >= star CZ count.")
