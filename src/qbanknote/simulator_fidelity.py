"""Offline (no-QPU) state fidelity under a synthetic Aer noise model.

Computes the same headline quantity used on hardware:

    F = <psi_ideal | rho_noisy | psi_ideal>

but obtains ``rho_noisy`` from an exact Aer density-matrix simulation instead of
IQM tomography. Random ansatz parameters are drawn with the same Uniform[0, 2*pi]
rule as the MW / KL / fidelity pilots.

Methodology notes (why this is Spark-faithful rather than a toy):

* Native gate basis ``("r", "cz")`` -- IQM Spark exposes a phased-RX (``r``)
  single-qubit gate and a ``cz`` two-qubit gate. Using an IBM-style
  ``rz/sx/x`` basis would mis-count single-qubit gates and add noise to virtual
  ``rz`` gates that are error-free on hardware.
* Star coupling map -- Spark is a star (one center qubit connected to the rest).
  Routing non-native two-qubit gates (e.g. the ring in ``ansatz_simulator``)
  costs SWAPs, which is exactly the hardware cost a topology comparison must
  charge. Without a coupling map every ansatz looks artificially equal.
* Gate error -> depolarizing probability conversion -- Qiskit's
  ``depolarizing_error(p, k)`` parameter ``p`` is NOT the average gate error.
  For dimension ``d = 2**k`` the average gate error is ``r = p (d-1)/d``, so we
  invert this to hit a requested per-gate error ``r`` (= ``1 - F_gate``).
* Optional thermal relaxation -- ``T1``/``T2`` with gate durations, composed on
  top of depolarizing, for a more realistic Spark-like model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
)

from qbanknote.metrics import bind_ansatz, reproduce_random_thetas

# IQM Spark native gates: phased-RX (Qiskit ``r``) + ``cz``.
DEFAULT_BASIS_GATES = ("r", "cz")

# Spark-typical published specs (datasheet): F_1q >= 99.9%, F_cz >= 99.0%.
DEFAULT_ERR1 = 0.001
DEFAULT_ERR2 = 0.01

# Spark-typical coherence / durations (seconds).
DEFAULT_T1 = 0.964e-3
DEFAULT_T2 = 1.155e-3
DEFAULT_GATE1_S = 20e-9
DEFAULT_GATE2_S = 60e-9

ONE_QUBIT_GATE_NAMES = {"id", "x", "sx", "rz", "ry", "rx", "r", "u", "u1", "u2", "u3"}
TWO_QUBIT_GATE_NAMES = {"cz", "cx", "ecr", "rzz"}


def average_error_to_depolarizing_p(avg_gate_error: float, num_qubits: int) -> float:
    """Convert an average gate error ``r = 1 - F`` to Qiskit's depolarizing ``p``.

    For a depolarizing channel on ``d = 2**num_qubits`` levels,
    ``r = p (d - 1) / d`` so ``p = r * d / (d - 1)``. The result is clamped to
    the physical range ``[0, 1]``.
    """
    if avg_gate_error < 0.0:
        raise ValueError("avg_gate_error must be non-negative")
    d = 2**num_qubits
    p = avg_gate_error * d / (d - 1)
    return float(min(max(p, 0.0), 1.0))


def star_coupling_map(n_qubits: int, center: int = 0) -> list[list[int]]:
    """Bidirectional star coupling map (center connected to every other qubit)."""
    if not (0 <= center < n_qubits):
        raise ValueError("center must be a valid qubit index")
    edges: list[list[int]] = []
    for q in range(n_qubits):
        if q == center:
            continue
        edges.append([center, q])
        edges.append([q, center])
    return edges


def build_noise_model(
    *,
    err1: float = DEFAULT_ERR1,
    err2: float = DEFAULT_ERR2,
    basis_gates: Sequence[str] = DEFAULT_BASIS_GATES,
    thermal: bool = False,
    t1: float = DEFAULT_T1,
    t2: float = DEFAULT_T2,
    gate1_s: float = DEFAULT_GATE1_S,
    gate2_s: float = DEFAULT_GATE2_S,
) -> NoiseModel:
    """Build a 1q/2q depolarizing (optionally + thermal) noise model.

    ``err1`` / ``err2`` are *average gate errors* (``1 - F_gate``), not raw
    depolarizing probabilities; they are converted internally.
    """
    if not (0.0 <= err1 <= 1.0 and 0.0 <= err2 <= 1.0):
        raise ValueError("err1 and err2 must lie in [0, 1]")
    if thermal and not (0.0 < t2 <= 2.0 * t1):
        raise ValueError("thermal relaxation requires 0 < T2 <= 2*T1")

    p1 = average_error_to_depolarizing_p(err1, 1)
    p2 = average_error_to_depolarizing_p(err2, 2)

    err1_channel = depolarizing_error(p1, 1)
    err2_channel = depolarizing_error(p2, 2)

    if thermal:
        relax1 = thermal_relaxation_error(t1, t2, gate1_s)
        relax2 = thermal_relaxation_error(t1, t2, gate2_s).tensor(
            thermal_relaxation_error(t1, t2, gate2_s)
        )
        err1_channel = err1_channel.compose(relax1)
        err2_channel = err2_channel.compose(relax2)

    noise_model = NoiseModel(basis_gates=list(basis_gates))
    one_qubit = [g for g in basis_gates if g in ONE_QUBIT_GATE_NAMES]
    two_qubit = [g for g in basis_gates if g in TWO_QUBIT_GATE_NAMES]
    for gate in one_qubit:
        noise_model.add_all_qubit_quantum_error(err1_channel, gate)
    for gate in two_qubit:
        noise_model.add_all_qubit_quantum_error(err2_channel, gate)
    return noise_model


def circuit_gate_stats(circuit: QuantumCircuit) -> dict[str, int]:
    skip = {"barrier", "measure", "save_density_matrix", "delay"}
    total_gates = 0
    two_qubit_gates = 0
    for instr in circuit.data:
        name = instr.operation.name
        if name in skip:
            continue
        total_gates += 1
        if instr.operation.num_qubits >= 2:
            two_qubit_gates += 1
    return {
        "transpiled_depth": int(circuit.depth()),
        "total_gates": int(total_gates),
        "two_qubit_gates": int(two_qubit_gates),
    }


def exact_noisy_fidelity(
    bound_circuit: QuantumCircuit,
    *,
    noise_model: NoiseModel | None,
    basis_gates: Sequence[str] = DEFAULT_BASIS_GATES,
    coupling_map: list[list[int]] | None = None,
    optimization_level: int = 1,
    seed_transpiler: int | None = None,
) -> dict[str, float | int]:
    """Exact Aer density-matrix fidelity of one bound ansatz circuit.

    The reference state is the *routed* noiseless statevector, i.e. the ideal
    output of the transpiled circuit including any SWAP-induced qubit
    permutation. This is important: routing relabels qubits (final layout), so
    comparing the noisy routed state against the un-routed logical state would
    conflate a harmless relabeling with error. Using the routed noiseless state
    still charges the SWAP *noise* cost (extra CZ errors) while removing the
    permutation artifact.
    """
    transpiled = transpile(
        bound_circuit,
        basis_gates=list(basis_gates),
        coupling_map=coupling_map,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    stats = circuit_gate_stats(transpiled)

    # Reference = noiseless routed state (matches the physical output ordering).
    psi_ideal = Statevector.from_instruction(transpiled)

    if noise_model is None:
        fidelity = 1.0
    else:
        sim_circuit = transpiled.copy()
        sim_circuit.save_density_matrix()
        sim = AerSimulator(method="density_matrix", noise_model=noise_model)
        result = sim.run(sim_circuit, shots=1).result()
        rho_noisy = DensityMatrix(result.data(0)["density_matrix"])
        fidelity = float(state_fidelity(psi_ideal, rho_noisy, validate=False))

    return {"fidelity": float(fidelity), **stats}


def _summarize(scores: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for keys, group in scores.groupby(group_cols, sort=True):
        values = group["fidelity"].to_numpy(dtype=float)
        n = len(values)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                "n_samples": n,
                "n_params": int(group["n_params"].iloc[0]),
                "f_avg": float(np.mean(values)),
                "f_std": float(np.std(values)),
                "f_sem": float(np.std(values) / math.sqrt(n)) if n else float("nan"),
                "f_min": float(np.min(values)),
                "f_max": float(np.max(values)),
                "transpiled_depth_avg": float(group["transpiled_depth"].mean()),
                "two_qubit_gates_avg": float(group["two_qubit_gates"].mean()),
                "total_gates_avg": float(group["total_gates"].mean()),
            }
        )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def run_simulator_fidelity_sweep(
    *,
    ansatz_fns: dict[str, Callable[[int, int], QuantumCircuit]],
    ansatz_names: list[str],
    depths: list[int],
    n_qubits: int = 5,
    n_samples: int = 20,
    seed: int = 42,
    err1: float = DEFAULT_ERR1,
    err2: float = DEFAULT_ERR2,
    err2_grid: Sequence[float] | None = None,
    noiseless: bool = False,
    basis_gates: Sequence[str] = DEFAULT_BASIS_GATES,
    topology: str = "star",
    star_center: int = 0,
    thermal: bool = False,
    t1: float = DEFAULT_T1,
    t2: float = DEFAULT_T2,
    gate1_s: float = DEFAULT_GATE1_S,
    gate2_s: float = DEFAULT_GATE2_S,
    optimization_level: int = 1,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Compare ansatze offline with random weights and a Spark-like noise model.

    If ``err2_grid`` is provided (and not noiseless), the sweep is repeated for
    each two-qubit error value so the ansatz ranking can be checked for
    stability across a noise range instead of a single arbitrary point.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if topology == "star":
        coupling_map: list[list[int]] | None = star_coupling_map(n_qubits, star_center)
    elif topology == "none":
        coupling_map = None
    else:
        raise ValueError(f"Unknown topology: {topology!r} (use 'star' or 'none')")

    if noiseless:
        err2_values: list[float | None] = [None]
    elif err2_grid:
        err2_values = [float(v) for v in err2_grid]
    else:
        err2_values = [float(err2)]

    score_rows: list[dict[str, object]] = []
    for err2_value in err2_values:
        noise_model = None
        if not noiseless:
            noise_model = build_noise_model(
                err1=err1,
                err2=float(err2_value),
                basis_gates=basis_gates,
                thermal=thermal,
                t1=t1,
                t2=t2,
                gate1_s=gate1_s,
                gate2_s=gate2_s,
            )

        for depth in depths:
            for ansatz_name in ansatz_names:
                ansatz_fn = ansatz_fns[ansatz_name]
                n_params = len(ansatz_fn(n_qubits, depth).parameters)
                depth_seed = seed + depth * 1000 + (1 if ansatz_name == "ansatz_simulator" else 0)

                for sample_index in range(n_samples):
                    theta = reproduce_random_thetas(depth_seed, n_params, sample_index)
                    bound = bind_ansatz(ansatz_fn, n_qubits, depth, theta)
                    result = exact_noisy_fidelity(
                        bound,
                        noise_model=noise_model,
                        basis_gates=basis_gates,
                        coupling_map=coupling_map,
                        optimization_level=optimization_level,
                        seed_transpiler=depth_seed + sample_index,
                    )
                    score_rows.append(
                        {
                            "ansatz": ansatz_name,
                            "depth": int(depth),
                            "err2": float("nan") if err2_value is None else float(err2_value),
                            "sample_index": int(sample_index),
                            "n_params": int(n_params),
                            "seed": int(depth_seed),
                            "fidelity": float(result["fidelity"]),
                            "transpiled_depth": int(result["transpiled_depth"]),
                            "total_gates": int(result["total_gates"]),
                            "two_qubit_gates": int(result["two_qubit_gates"]),
                        }
                    )

    scores = pd.DataFrame(score_rows)
    group_cols = ["ansatz", "depth"]
    if not noiseless and len(err2_values) > 1:
        group_cols = ["ansatz", "depth", "err2"]
    summary = _summarize(scores, group_cols)

    scores_path = output_dir / "simulator_fidelity_scores.csv"
    summary_path = output_dir / "simulator_fidelity_summary.csv"
    scores.to_csv(scores_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "method": "aer_density_matrix_exact_fidelity",
        "fidelity_definition": "F = <psi_ideal | rho_noisy | psi_ideal>",
        "weight_sampling": "uniform_random_0_2pi",
        "noiseless": bool(noiseless),
        "basis_gates": list(basis_gates),
        "topology": topology,
        "star_center": int(star_center) if topology == "star" else None,
        "err1": None if noiseless else float(err1),
        "err2": None if noiseless else (list(err2_values) if len(err2_values) > 1 else float(err2_values[0])),
        "error_convention": "err = 1 - F_gate (avg gate error); converted to depolarizing p = err*d/(d-1)",
        "thermal": bool(thermal),
        "t1_s": float(t1) if thermal else None,
        "t2_s": float(t2) if thermal else None,
        "gate1_s": float(gate1_s) if thermal else None,
        "gate2_s": float(gate2_s) if thermal else None,
        "n_qubits": int(n_qubits),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "optimization_level": int(optimization_level),
        "ansatzes": list(ansatz_names),
        "depths": list(depths),
        "outputs": [
            "simulator_fidelity_scores.csv",
            "simulator_fidelity_summary.csv",
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return scores, summary, summary_path
