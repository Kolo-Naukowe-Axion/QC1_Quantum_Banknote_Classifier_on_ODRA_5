"""Hybrid quantum-classical model wrapper."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN


def angle_encoding_feature_map(num_qubits: int) -> QuantumCircuit:
    """RY angle encoding on ``num_qubits`` qubits."""
    qc_data = QuantumCircuit(num_qubits)
    input_params = ParameterVector("x", num_qubits)
    for i in range(num_qubits):
        qc_data.ry(input_params[i], i)
    return qc_data


class HybridModel(nn.Module):
    """Feature map + ansatz + EstimatorQNN, wrapped with TorchConnector."""

    def __init__(
        self,
        ansatz_circuit: QuantumCircuit,
        num_qubits: int,
        *,
        random_seed: int | None = 42,
        gradient_backend: Literal["param_shift", "reverse"] = "param_shift",
    ):
        super().__init__()
        self.feature_map = angle_encoding_feature_map(num_qubits)
        self.num_qubits = num_qubits

        self.qc = QuantumCircuit(num_qubits)
        self.qc.compose(self.feature_map, qubits=range(num_qubits), inplace=True)
        self.qc.compose(ansatz_circuit, inplace=True)

        input_params = list(self.feature_map.parameters)
        weight_params = list(ansatz_circuit.parameters)
        observable = SparsePauliOp.from_list([("I" * (num_qubits - 1) + "Z", 1)])

        estimator_kwargs = {}
        if random_seed is not None:
            estimator_kwargs["seed"] = random_seed
        estimator = StatevectorEstimator(**estimator_kwargs)

        if gradient_backend == "param_shift":
            from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient

            gradient = ParamShiftEstimatorGradient(estimator)
        elif gradient_backend == "reverse":
            from qiskit_algorithms.gradients import ReverseEstimatorGradient

            gradient = ReverseEstimatorGradient(estimator)
        else:
            raise ValueError(f"Unknown gradient_backend: {gradient_backend!r}")

        self.qnn = EstimatorQNN(
            circuit=self.qc,
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
            gradient=gradient,
        )
        self.quantum_layer = TorchConnector(self.qnn)

    # Notebook compatibility aliases.
    def _create_angle_encoding(self, num_qubits: int) -> QuantumCircuit:
        return angle_encoding_feature_map(num_qubits)

    def angle_encoding(self, num_qubits: int) -> QuantumCircuit:
        return angle_encoding_feature_map(num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum_layer(x)
