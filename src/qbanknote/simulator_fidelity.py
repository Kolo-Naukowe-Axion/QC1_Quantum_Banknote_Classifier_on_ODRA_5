"""Offline Aer fidelity aligned with the QPU tomography protocol.

Default path (for fair simulator-vs-QPU comparison):

1. Same tomography protocol as IQM: full ``3^n`` Pauli settings, shot counts,
   linear inversion, physical projection.
2. Same ideal reference as QPU: logical noiseless statevector of the prepared
   circuit.
3. Same parameter sampling as the current QPU fidelity / MW / KL pilots:
   ``theta ~ Uniform[0, 2*pi]`` (``weights="random"``). Optional
   ``weights="trained"`` uses CV checkpoints + test inputs + feature map.

Everything runs on ``AerSimulator`` — no IQM / QPU access required.
"""

from __future__ import annotations

import json
import math
import time
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

from qbanknote.data import load_fold_arrays
from qbanknote.evaluation import ansatz_key
from qbanknote.iqm import normalize_counts
from qbanknote.metrics import bind_ansatz, reproduce_random_thetas
from qbanknote.paths import find_project_root
from qbanknote.tomography import (
    add_tomography_rotations,
    all_basis_settings,
    build_bound_circuit,
    project_to_physical,
    reconstruct_rho,
    state_fidelity_pure,
)
from qbanknote.weights import load_trained_weights, metric_weight_path

DEFAULT_BASIS_GATES = ("r", "cz")
DEFAULT_ERR1 = 0.001
DEFAULT_ERR2 = 0.01
DEFAULT_T1 = 0.964e-3
DEFAULT_T2 = 1.155e-3
DEFAULT_GATE1_S = 20e-9
DEFAULT_GATE2_S = 60e-9
DEFAULT_SHOTS = 1024

ONE_QUBIT_GATE_NAMES = {"id", "x", "sx", "rz", "ry", "rx", "r", "u", "u1", "u2", "u3"}
TWO_QUBIT_GATE_NAMES = {"cz", "cx", "ecr", "rzz"}


def average_error_to_depolarizing_p(avg_gate_error: float, num_qubits: int) -> float:
    """Convert average gate error ``r = 1 - F`` to Qiskit depolarizing ``p``."""
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
    """Build 1q/2q depolarizing (+ optional thermal) noise. ``err*`` = avg gate error."""
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
    for gate in basis_gates:
        if gate in ONE_QUBIT_GATE_NAMES:
            noise_model.add_all_qubit_quantum_error(err1_channel, gate)
        elif gate in TWO_QUBIT_GATE_NAMES:
            noise_model.add_all_qubit_quantum_error(err2_channel, gate)
    return noise_model


def make_aer_backend(
    *,
    noise_model: NoiseModel | None,
    basis_gates: Sequence[str],
    coupling_map: list[list[int]] | None,
) -> AerSimulator:
    """Fully offline Aer backend (optionally noisy + topology-constrained)."""
    kwargs: dict[str, object] = {"basis_gates": list(basis_gates)}
    if noise_model is not None:
        kwargs["noise_model"] = noise_model
    if coupling_map is not None:
        kwargs["coupling_map"] = coupling_map
    return AerSimulator(**kwargs)


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


def _transpile_kwargs(
    *,
    basis_gates: Sequence[str],
    coupling_map: list[list[int]] | None,
    initial_layout: list[int] | None,
    optimization_level: int,
    seed_transpiler: int | None,
) -> dict[str, object]:
    # Do not pass ``backend=`` together with basis_gates/coupling_map (Qiskit warning).
    kwargs: dict[str, object] = {
        "basis_gates": list(basis_gates),
        "optimization_level": optimization_level,
    }
    if coupling_map is not None:
        kwargs["coupling_map"] = coupling_map
    if initial_layout is not None:
        kwargs["initial_layout"] = initial_layout
    if seed_transpiler is not None:
        kwargs["seed_transpiler"] = seed_transpiler
    return kwargs


def run_aer_tomography_counts(
    state_circuit: QuantumCircuit,
    backend: AerSimulator,
    *,
    n_qubits: int,
    shots: int,
    basis_gates: Sequence[str],
    coupling_map: list[list[int]] | None,
    initial_layout: list[int] | None,
    optimization_level: int,
    seed_transpiler: int | None,
    max_circuits_per_job: int,
) -> tuple[dict[tuple[str, ...], dict[str, int]], dict[str, int]]:
    """Shot-based full Pauli tomography on Aer (same bases as QPU path)."""
    bases = all_basis_settings(n_qubits)
    tomo_circuits = [add_tomography_rotations(state_circuit, b) for b in bases]
    tkwargs = _transpile_kwargs(
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        initial_layout=initial_layout,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    transpiled = [transpile(qc, **tkwargs) for qc in tomo_circuits]
    stats = circuit_gate_stats(transpiled[0].remove_final_measurements(inplace=False))

    counts_per_basis: dict[tuple[str, ...], dict[str, int]] = {}
    submitted = 0
    while submitted < len(transpiled):
        batch = transpiled[submitted : submitted + max_circuits_per_job]
        batch_bases = bases[submitted : submitted + max_circuits_per_job]
        result = backend.run(batch, shots=shots).result()
        counts_list = result.get_counts()
        if not isinstance(counts_list, list):
            counts_list = [counts_list]
        if len(counts_list) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} count dicts, Aer returned {len(counts_list)}"
            )
        for basis_tuple, counts in zip(batch_bases, counts_list):
            counts_per_basis[basis_tuple] = normalize_counts(counts)
        submitted += len(batch)

    return counts_per_basis, stats


def sample_aer_tomography_fidelity(
    state_circuit: QuantumCircuit,
    backend: AerSimulator,
    *,
    n_qubits: int,
    shots: int,
    basis_gates: Sequence[str],
    coupling_map: list[list[int]] | None,
    initial_layout: list[int] | None,
    optimization_level: int,
    seed_transpiler: int | None,
    max_circuits_per_job: int,
) -> dict[str, float | int]:
    """QPU-matching fidelity: logical ideal vs tomography-reconstructed rho on Aer."""
    # Point 2: same ideal as QPU — logical noiseless statevector.
    psi_ideal = Statevector.from_instruction(state_circuit).data

    counts, stats = run_aer_tomography_counts(
        state_circuit,
        backend,
        n_qubits=n_qubits,
        shots=shots,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        initial_layout=initial_layout,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        max_circuits_per_job=max_circuits_per_job,
    )
    rho_lin = reconstruct_rho(counts, n_qubits)
    rho_phys = project_to_physical(rho_lin)
    return {
        "fidelity_linear": float(state_fidelity_pure(psi_ideal, rho_lin)),
        "fidelity_physical": float(state_fidelity_pure(psi_ideal, rho_phys)),
        "fidelity": float(state_fidelity_pure(psi_ideal, rho_phys)),
        "trace_linear": float(np.real(np.trace(rho_lin))),
        "trace_physical": float(np.real(np.trace(rho_phys))),
        "purity_linear": float(np.real(np.trace(rho_lin @ rho_lin))),
        "purity_physical": float(np.real(np.trace(rho_phys @ rho_phys))),
        **stats,
    }


def sample_aer_exact_fidelity(
    state_circuit: QuantumCircuit,
    *,
    noise_model: NoiseModel | None,
    basis_gates: Sequence[str],
    coupling_map: list[list[int]] | None,
    initial_layout: list[int] | None,
    optimization_level: int,
    seed_transpiler: int | None,
) -> dict[str, float | int]:
    """Fast exact density-matrix path (screening only; not QPU-protocol-matched).

    Uses the *routed* noiseless statevector as reference so SWAP permutation does
    not fake-kill fidelity. Prefer ``protocol=tomography`` for simulator-vs-QPU
    comparisons.
    """
    tkwargs: dict[str, object] = {
        "basis_gates": list(basis_gates),
        "optimization_level": optimization_level,
    }
    if coupling_map is not None:
        tkwargs["coupling_map"] = coupling_map
    if initial_layout is not None:
        tkwargs["initial_layout"] = initial_layout
    if seed_transpiler is not None:
        tkwargs["seed_transpiler"] = seed_transpiler
    transpiled = transpile(state_circuit, **tkwargs)
    stats = circuit_gate_stats(transpiled)
    psi_ref = Statevector.from_instruction(transpiled)

    if noise_model is None:
        fidelity = 1.0
    else:
        sim_circuit = transpiled.copy()
        sim_circuit.save_density_matrix()
        sim = AerSimulator(method="density_matrix", noise_model=noise_model)
        rho_noisy = DensityMatrix(
            sim.run(sim_circuit, shots=1).result().data(0)["density_matrix"]
        )
        fidelity = float(state_fidelity(psi_ref, rho_noisy, validate=False))

    return {
        "fidelity": float(fidelity),
        "fidelity_physical": float(fidelity),
        "fidelity_linear": float(fidelity),
        **stats,
    }


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
    protocol: str = "tomography",
    weights: str = "random",
    shots: int = DEFAULT_SHOTS,
    fold: int = 1,
    epoch: int = 30,
    max_circuits_per_job: int = 275,
    root: Path | None = None,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Offline fidelity sweep aligned with QPU methodology by default."""
    if protocol not in {"tomography", "exact"}:
        raise ValueError("protocol must be 'tomography' or 'exact'")
    if weights not in {"trained", "random"}:
        raise ValueError("weights must be 'trained' or 'random'")
    if thermal and not noiseless:
        # Datasheet RB errors already include some decoherence.
        pass

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = find_project_root(root)

    if topology == "star":
        coupling_map: list[list[int]] | None = star_coupling_map(n_qubits, star_center)
        # Pin logical i -> physical i so odra_star hub (q0) stays on star center.
        initial_layout: list[int] | None = list(range(n_qubits))
    elif topology == "none":
        coupling_map = None
        initial_layout = None
    else:
        raise ValueError(f"Unknown topology: {topology!r} (use 'star' or 'none')")

    if noiseless:
        err2_values: list[float | None] = [None]
    elif err2_grid:
        err2_values = [float(v) for v in err2_grid]
    else:
        err2_values = [float(err2)]

    # Optional task-state mode: trained weights + real test inputs.
    inputs: np.ndarray | None = None
    if weights == "trained":
        X_test, _ = load_fold_arrays(fold, split="test", root=project_root)
        if len(X_test) < n_samples:
            raise ValueError(
                f"Fold {fold} test split has {len(X_test)} rows, need n_samples={n_samples}"
            )
        inputs = np.asarray(X_test[:n_samples], dtype=float)

    score_rows: list[dict[str, object]] = []
    t_start = time.perf_counter()

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
        backend = make_aer_backend(
            noise_model=noise_model,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
        )

        for depth in depths:
            for ansatz_name in ansatz_names:
                ansatz_fn = ansatz_fns[ansatz_name]
                ansatz_circuit = ansatz_fn(n_qubits, depth)
                n_params = len(ansatz_circuit.parameters)
                depth_seed = seed + depth * 1000 + (1 if ansatz_name == "ansatz_simulator" else 0)

                weight_values: np.ndarray | None = None
                if weights == "trained":
                    weight_file = metric_weight_path(
                        depth,
                        ansatz_key(ansatz_name),
                        fold,
                        epoch=epoch,
                        root=project_root,
                    )
                    if not weight_file.exists():
                        raise FileNotFoundError(
                            f"Trained weights not found for {ansatz_name} "
                            f"depth={depth} fold={fold}: {weight_file}"
                        )
                    weight_values = load_trained_weights(weight_file, ansatz_circuit)

                if verbose:
                    print(
                        f"[{protocol}/{weights}] {ansatz_name} depth={depth} "
                        f"err2={err2_value} n_samples={n_samples}",
                        flush=True,
                    )

                for sample_index in range(n_samples):
                    if weights == "trained":
                        assert inputs is not None and weight_values is not None
                        state_circuit = build_bound_circuit(
                            ansatz_circuit,
                            inputs[sample_index],
                            weight_values,
                            n_qubits,
                        )
                    else:
                        theta = reproduce_random_thetas(depth_seed, n_params, sample_index)
                        state_circuit = bind_ansatz(ansatz_fn, n_qubits, depth, theta)

                    seed_transpiler = depth_seed + sample_index
                    if protocol == "tomography":
                        result = sample_aer_tomography_fidelity(
                            state_circuit,
                            backend,
                            n_qubits=n_qubits,
                            shots=shots,
                            basis_gates=basis_gates,
                            coupling_map=coupling_map,
                            initial_layout=initial_layout,
                            optimization_level=optimization_level,
                            seed_transpiler=seed_transpiler,
                            max_circuits_per_job=max_circuits_per_job,
                        )
                    else:
                        result = sample_aer_exact_fidelity(
                            state_circuit,
                            noise_model=noise_model,
                            basis_gates=basis_gates,
                            coupling_map=coupling_map,
                            initial_layout=initial_layout,
                            optimization_level=optimization_level,
                            seed_transpiler=seed_transpiler,
                        )

                    score_rows.append(
                        {
                            "ansatz": ansatz_name,
                            "depth": int(depth),
                            "err2": float("nan") if err2_value is None else float(err2_value),
                            "sample_index": int(sample_index),
                            "n_params": int(n_params),
                            "seed": int(depth_seed),
                            "protocol": protocol,
                            "weights": weights,
                            "shots": int(shots) if protocol == "tomography" else 0,
                            "fidelity": float(result["fidelity"]),
                            "fidelity_physical": float(result["fidelity_physical"]),
                            "fidelity_linear": float(result["fidelity_linear"]),
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
        "elapsed_s": float(time.perf_counter() - t_start),
        "method": (
            "aer_full_pauli_tomography_fidelity"
            if protocol == "tomography"
            else "aer_density_matrix_exact_fidelity"
        ),
        "fidelity_definition": (
            "F = <psi_logical | rho_tomography | psi_logical> "
            "(same as QPU tomography protocol)"
            if protocol == "tomography"
            else "F = <psi_logical | rho_exact | psi_logical>"
        ),
        "ideal_reference": "logical_statevector",
        "weight_mode": weights,
        "weight_sampling": (
            "trained_cv_checkpoint_plus_test_inputs"
            if weights == "trained"
            else "uniform_random_0_2pi"
        ),
        "protocol": protocol,
        "shots": int(shots) if protocol == "tomography" else None,
        "noiseless": bool(noiseless),
        "basis_gates": list(basis_gates),
        "topology": topology,
        "star_center": int(star_center) if topology == "star" else None,
        "initial_layout": initial_layout,
        "err1": None if noiseless else float(err1),
        "err2": None
        if noiseless
        else (list(err2_values) if len(err2_values) > 1 else float(err2_values[0])),
        "error_convention": "err = 1 - F_gate (avg gate error); p = err*d/(d-1)",
        "thermal": bool(thermal),
        "t1_s": float(t1) if thermal else None,
        "t2_s": float(t2) if thermal else None,
        "fold": int(fold) if weights == "trained" else None,
        "checkpoint_epoch": int(epoch) if weights == "trained" else None,
        "n_qubits": int(n_qubits),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "optimization_level": int(optimization_level),
        "ansatzes": list(ansatz_names),
        "depths": list(depths),
        "backend": "AerSimulator",
        "requires_qpu": False,
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
