"""IQM backend connection helpers and EstimatorV2 shim."""

from __future__ import annotations

import getpass
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import PrimitiveResult, PubResult
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers.data_bin import DataBin
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from qbanknote.model import HybridModel

try:
    from iqm.qiskit_iqm import transpile_to_IQM as _iqm_transpile
    from iqm.qiskit_iqm.iqm_backend import IQMBackendBase as _IQMBackendBase
except ImportError:
    _iqm_transpile = None
    _IQMBackendBase = None


class SimpleIQMJob:
    """Minimal job wrapper holding a primitive result."""

    def __init__(self, result: PrimitiveResult):
        self._result = result

    def result(self) -> PrimitiveResult:
        return self._result


class IQMBackendEstimator(BaseEstimatorV2):
    """Run EstimatorQNN forwards on IQM hardware via batched measurement jobs."""

    def __init__(self, backend, options: dict[str, Any] | None = None):
        super().__init__()
        self._backend = backend
        self._options = options or {"shots": 100}
        self.timestamp_history: list[dict[str, Any]] = []
        self.total_qpu_time = 0.0
        self.failed_batches: list[dict[str, Any]] = []

    def _transpile_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "optimization_level": self._options.get("optimization_level", 3)
        }
        seed = self._options.get("seed_transpiler")
        if seed is not None:
            kwargs["seed_transpiler"] = seed
        return kwargs

    def _transpile_for_backend(self, circuit: QuantumCircuit) -> QuantumCircuit:
        kwargs = self._transpile_kwargs()
        if (
            _iqm_transpile is not None
            and _IQMBackendBase is not None
            and isinstance(self._backend, _IQMBackendBase)
        ):
            return _iqm_transpile(circuit, self._backend, **kwargs)
        return transpile(circuit, self._backend, **kwargs)

    def _extract_timestamps(self, result) -> dict[str, Any] | None:
        try:
            timeline = result._metadata.get("timeline", [])
            if not timeline:
                return None
            return {entry.status: entry.timestamp for entry in timeline}
        except Exception:
            return None

    def _counts_to_expectation(self, counts) -> float:
        if isinstance(counts, list):
            counts = counts[0]
        shots = sum(counts.values())
        count_0 = sum(c for bitstring, c in counts.items() if bitstring[-1] == "0")
        p0 = count_0 / shots if shots else 0.0
        return p0 - (1 - p0)

    def run(self, pubs, precision=None):
        if not isinstance(pubs, list):
            pubs = [pubs]

        self.failed_batches = []
        job_results = []
        shots_opt = self._options["shots"]
        max_circuits = self._options.get("max_circuits_per_job")

        base_circuit = pubs[0][0]
        circuit_with_meas = base_circuit.copy()
        if circuit_with_meas.num_clbits == 0:
            circuit_with_meas.measure_all()
        transpiled_qc = self._transpile_for_backend(circuit_with_meas)

        for pub in pubs:
            _, observables, parameter_values = pub
            if parameter_values.ndim == 1:
                parameter_values = [parameter_values]

            bound_circuits = [
                transpiled_qc.assign_parameters(params) for params in parameter_values
            ]
            n_circuits = len(bound_circuits)
            pub_expectations = []

            for start in range(0, n_circuits, max_circuits or n_circuits):
                end = min(start + (max_circuits or n_circuits), n_circuits)
                batch = bound_circuits[start:end]
                try:
                    job = self._backend.run(batch, shots=shots_opt)
                    result = job.result()

                    ts = self._extract_timestamps(result)
                    if ts:
                        exec_start = ts.get("execution_started")
                        exec_end = ts.get("execution_ended")
                        comp_start = ts.get("compilation_started")
                        comp_end = ts.get("compilation_ended")
                        job_created = ts.get("created")
                        job_completed = ts.get("completed")
                        if exec_start and exec_end:
                            execution_time = (exec_end - exec_start).total_seconds()
                            compile_time = (
                                (comp_end - comp_start).total_seconds()
                                if comp_start and comp_end
                                else 0.0
                            )
                            job_time = (
                                (job_completed - job_created).total_seconds()
                                if job_created and job_completed
                                else 0.0
                            )
                            self.timestamp_history.append(
                                {
                                    "execution_time_qpu": execution_time,
                                    "job_time_total": job_time,
                                    "compile_time": compile_time,
                                    "raw_timestamps": ts,
                                    "n_circuits": len(batch),
                                }
                            )
                            self.total_qpu_time += execution_time

                    counts_list = result.get_counts()
                    if not isinstance(counts_list, list):
                        counts_list = [counts_list]
                    for counts in counts_list:
                        pub_expectations.append(self._counts_to_expectation(counts))
                except Exception as exc:
                    self.failed_batches.append(
                        {
                            "start": start,
                            "end": end,
                            "n_circuits": len(batch),
                            "error": str(exc),
                        }
                    )
                    print(f"Batch job failed: {exc}")
                    pub_expectations.extend([0.0] * len(batch))

            data = DataBin(evs=np.array(pub_expectations), shape=(len(pub_expectations),))
            job_results.append(PubResult(data=data))

        return SimpleIQMJob(PrimitiveResult(job_results))

    def print_timing_summary(self) -> None:
        if not self.timestamp_history:
            print("No timestamp data collected.")
            return

        print("\n" + "=" * 60)
        print("DETAILED SUMMARY OF THE TIMESTAMPS")
        print("=" * 60)
        print(f"Number of executed jobs: {len(self.timestamp_history)}")

        qpu_times: list[float] = []
        compile_times: list[float] = []
        queue_times: list[float] = []
        network_times: list[float] = []

        for entry in self.timestamp_history:
            ts = entry["raw_timestamps"]
            if ts.get("execution_started") and ts.get("execution_ended"):
                qpu_times.append(
                    (ts["execution_ended"] - ts["execution_started"]).total_seconds()
                )
            if ts.get("compilation_started") and ts.get("compilation_ended"):
                compile_times.append(
                    (ts["compilation_ended"] - ts["compilation_started"]).total_seconds()
                )
            if ts.get("pending_execution") and ts.get("execution_started"):
                queue_times.append(
                    (ts["execution_started"] - ts["pending_execution"]).total_seconds()
                )
            net_time = 0.0
            if ts.get("created") and ts.get("received"):
                net_time += (ts["received"] - ts["created"]).total_seconds()
            if ts.get("ready") and ts.get("completed"):
                net_time += (ts["completed"] - ts["ready"]).total_seconds()
            network_times.append(net_time)

        total_job = sum(t["job_time_total"] for t in self.timestamp_history)
        total_measured = (
            sum(qpu_times) + sum(compile_times) + sum(queue_times) + sum(network_times)
        )
        other = total_job - total_measured

        print(
            f"\nTIME ON QPU :     {sum(qpu_times) * 1000:8.2f} ms  "
            f"(mean: {np.mean(qpu_times) * 1000:.2f} ms/job)"
            if qpu_times
            else "\nTIME ON QPU :          n/a"
        )
        if compile_times:
            print(
                f"Compilation:       {sum(compile_times) * 1000:8.2f} ms  "
                f"(mean: {np.mean(compile_times) * 1000:.2f} ms/job)"
            )
        if queue_times:
            print(
                f"Queue:             {sum(queue_times) * 1000:8.2f} ms  "
                f"(mean: {np.mean(queue_times) * 1000:.2f} ms/job)"
            )
        if network_times:
            print(
                f"Network:           {sum(network_times) * 1000:8.2f} ms  "
                f"(mean: {np.mean(network_times) * 1000:.2f} ms/job)"
            )
        print(f"Others:            {other * 1000:8.2f} ms")
        print(f"\nTIME OVERALL:       {total_job * 1000:8.2f} ms ({total_job:.3f} s)")
        print("=" * 60 + "\n")


def calibration_set_id(result) -> str | None:
    """Return calibration set id if the provider exposes it."""
    for path in (
        lambda r: getattr(r, "parameters", None),
        lambda r: getattr(r, "metadata", None),
        lambda r: getattr(r, "_metadata", None),
    ):
        try:
            obj = path(result)
            if obj is None:
                continue
            if isinstance(obj, dict):
                cid = obj.get("calibration_set_id") or obj.get("calibration_set")
                if cid is not None:
                    return str(cid)
            cid = getattr(obj, "calibration_set_id", None)
            if cid is not None:
                return str(cid)
        except Exception:
            continue
    return None


def build_iqm_estimator_model(
    iqm_backend,
    ansatz_fn: Callable[[int, int], QuantumCircuit],
    *,
    num_qubits: int,
    depth: int,
    shots: int,
    optimization_level: int = 1,
    seed_transpiler: int | None = None,
    random_seed: int = 42,
    max_circuits_per_job: int | None = None,
):
    """Build a ``TorchConnector`` QNN that forwards on IQM hardware."""
    estimator_options: dict[str, Any] = {
        "shots": shots,
        "optimization_level": optimization_level,
    }
    if seed_transpiler is not None:
        estimator_options["seed_transpiler"] = seed_transpiler
    if max_circuits_per_job is not None:
        estimator_options["max_circuits_per_job"] = max_circuits_per_job

    hw_estimator = IQMBackendEstimator(iqm_backend, options=estimator_options)
    hw_ansatz = ansatz_fn(num_qubits, depth)
    hw_feature_map = HybridModel(
        hw_ansatz, num_qubits, random_seed=random_seed, gradient_backend="param_shift"
    ).angle_encoding(num_qubits)

    hw_qc = QuantumCircuit(num_qubits)
    hw_qc.compose(hw_feature_map, qubits=range(num_qubits), inplace=True)
    hw_qc.compose(hw_ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("I" * (num_qubits - 1) + "Z", 1)])
    hw_qnn = EstimatorQNN(
        circuit=hw_qc,
        observables=observable,
        input_params=list(hw_feature_map.parameters),
        weight_params=list(hw_ansatz.parameters),
        estimator=hw_estimator,
    )
    hw_model = TorchConnector(hw_qnn)
    return hw_model, hw_estimator


def connect_to_iqm_backend(iqm_url: str, token: str | None = None):
    """Connect to an IQM backend via ``IQMProvider``."""
    env_token = os.environ.get("IQM_TOKEN", "").strip()
    if token and env_token:
        raise ValueError("Set either token argument or IQM_TOKEN, not both")
    if token is None and not env_token:
        token = getpass.getpass("Enter IQM Token: ").strip()
    from iqm.qiskit_iqm import IQMProvider

    provider = IQMProvider(iqm_url, token=token) if token else IQMProvider(iqm_url)
    return provider.get_backend()


def transpile_for_backend(
    circuit: QuantumCircuit,
    backend,
    optimization_level: int,
    seed_transpiler: int | None = None,
):
    kwargs: dict[str, object] = {"optimization_level": optimization_level}
    if seed_transpiler is not None:
        kwargs["seed_transpiler"] = seed_transpiler
    if (
        _iqm_transpile is not None
        and _IQMBackendBase is not None
        and isinstance(backend, _IQMBackendBase)
    ):
        return _iqm_transpile(circuit, backend, **kwargs)
    return transpile(circuit, backend, **kwargs)


def normalize_counts(counts) -> dict[str, int]:
    if isinstance(counts, list):
        if len(counts) != 1:
            raise ValueError(f"Expected one counts dict, got {len(counts)}")
        counts = counts[0]
    return {str(k): int(v) for k, v in counts.items()}


def run_circuits_on_backend(
    backend,
    circuits: list[QuantumCircuit],
    shots: int,
    optimization_level: int,
    seed_transpiler: int | None,
    max_circuits_per_job: int,
) -> list[dict[str, int]]:
    """Transpile and execute circuits; return counts in input order."""
    transpiled = [
        transpile_for_backend(qc, backend, optimization_level, seed_transpiler)
        for qc in circuits
    ]
    all_counts: list[dict[str, int]] = []
    batch_size = max(1, max_circuits_per_job)
    for start in range(0, len(transpiled), batch_size):
        batch = transpiled[start : start + batch_size]
        result = backend.run(batch, shots=shots).result()
        counts_list = result.get_counts()
        if not isinstance(counts_list, list):
            counts_list = [counts_list]
        if len(counts_list) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} count dicts, backend returned {len(counts_list)}"
            )
        all_counts.extend(normalize_counts(c) for c in counts_list)
    return all_counts
