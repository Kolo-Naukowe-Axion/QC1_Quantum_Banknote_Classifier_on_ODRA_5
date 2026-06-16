"""EstimatorV2 wrapper for IQM Qiskit backend (same logic as evaluation notebook, + transpile seed)."""

import numpy as np
from qiskit import transpile
from qiskit.primitives import PrimitiveResult, PubResult
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers.data_bin import DataBin

try:
    from iqm.qiskit_iqm import transpile_to_IQM as _iqm_transpile
    from iqm.qiskit_iqm.iqm_backend import IQMBackendBase as _IQMBackendBase
except ImportError:
    _iqm_transpile = None
    _IQMBackendBase = None

from .expectation_from_counts import counts_to_expectation


class SimpleIQMJob:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class IQMBackendEstimator(BaseEstimatorV2):
    def __init__(self, backend, options=None):
        super().__init__()
        self._backend = backend
        self._options = options or {"shots": 100}
        self.timestamp_history = []
        self.total_qpu_time = 0.0
        self.failed_batches = []

    def _transpile_kwargs(self) -> dict:
        kw = {"optimization_level": self._options.get("optimization_level", 3)}
        seed = self._options.get("seed_transpiler")
        if seed is not None:
            kw["seed_transpiler"] = seed
        return kw

    def _transpile_for_backend(self, circuit):
        kwargs = self._transpile_kwargs()
        if (
            _iqm_transpile is not None
            and _IQMBackendBase is not None
            and isinstance(self._backend, _IQMBackendBase)
        ):
            return _iqm_transpile(circuit, self._backend, **kwargs)
        return transpile(circuit, self._backend, **kwargs)

    def _extract_timestamps(self, result):
        try:
            timeline = result._metadata.get("timeline", [])
            if not timeline:
                return None
            timestamps = {}
            for entry in timeline:
                timestamps[entry.status] = entry.timestamp
            return timestamps
        except Exception:
            return None

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
            _, _observables, parameter_values = pub
            if parameter_values.ndim == 1:
                parameter_values = [parameter_values]

            bound_circuits = [transpiled_qc.assign_parameters(params) for params in parameter_values]
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
                                (comp_end - comp_start).total_seconds() if comp_start and comp_end else 0.0
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
                        pub_expectations.append(counts_to_expectation(counts))
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
