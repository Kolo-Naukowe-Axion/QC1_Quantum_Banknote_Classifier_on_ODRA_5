#!/usr/bin/env python
# coding: utf-8

# # Variational Quantum Classifier for Binary Classification
# 
# **Authors:** Koło Naukowe Axion  
# **Dataset:** Banknote Authentication (UCI ML Repository)  
# **Framework:** Qiskit Machine Learning + PyTorch
# 
# ## Abstract
# 
# This notebook demonstrates a hybrid Variational Quantum Circuit (VQC) for binary classification using the banknote authentication dataset. The model leverages:
# 
# - **Feature encoding**: Angle encoding via RY rotations
# - **Parametrized quantum circuit (VQC)**: Hardware-efficient ansatz with ring topology entanglement
# - **Measurement**: Expectation value of the Pauli-Z observable on the first qubit
# - **Classical optimizer**: Adam optimizer via PyTorch's automatic differentiation
# 
# The hybrid architecture uses Qiskit's `EstimatorQNN` bridged to PyTorch via `TorchConnector`, enabling seamless gradient-based training on a simulated quantum computer.

# ## 1. Environment Setup
# 
# This section handles dependency installation and imports. For reproducibility, all package versions should be pinned in a production environment.

# ### 1.1 Package Installation (Optional)
# 
# Set `INSTALL_DEPS = True` if running in a fresh environment. For production use, pin specific versions.

# In[1]:


# Optional: Install dependencies if not already present
INSTALL_DEPS = False

if INSTALL_DEPS:
    import sys
    import subprocess

    packages = [
        'numpy',
        'scikit-learn',
        'ucimlrepo',
        'qiskit',
        'qiskit-machine-learning',
        'torch',
        'matplotlib'
    ]

    for pkg in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])


# ### 1.2 Imports

# In[2]:


# Standard library
import os
import random
import copy 

# Third-party: Scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Third-party: Machine learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from ucimlrepo import fetch_ucirepo

# Third-party: Quantum computing
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN


# ## 2. Reproducibility and Random Seed Control

# In[3]:


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy, PyTorch, and Python's random module.

    Parameters
    ----------
    seed : int
        Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False    
    algorithm_globals.random_seed = seed

# Set global seed
RANDOM_SEED = 42
set_random_seed(RANDOM_SEED)


# ## 3. Quantum Circuit Architecture
# 
# The quantum model consists of two components:
# 1. **Feature map** (angle encoding): Encodes classical data into quantum states
# 2. **Ansatz** (trainable circuit): Parametrized quantum circuit with learnable weights

# ### 3.1 Parametrized Ansatz

# In[4]:


def star_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    hub = 2
    # +2*n_qubits for the final rotation layer (Rz + Rx only; last Rz would commute with Z)
    total_params = n_qubits * depth * 3 + 2 * n_qubits
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


# ### 3.2 Hybrid Variational Quantum Circuit
# 
# The `HybridModel` class implements a VQC by integrating the quantum circuit with PyTorch's autograd system via Qiskit's `TorchConnector`. This enables gradient-based optimization of quantum parameters using classical optimizers.

# In[5]:


class HybridModel(nn.Module):
    """
    Hybrid Variational Quantum Circuit (VQC) for binary classification.

    The model combines:
    1. Angle encoding feature map (classical data → quantum state)
    2. Parametrized ansatz (trainable quantum circuit)
    3. Observable measurement (quantum state → classical expectation value)
    4. PyTorch integration via TorchConnector (enables backpropagation)

    Parameters
    ----------
    ansatz_circuit : QuantumCircuit
        Parametrized quantum circuit with trainable weights
    num_qubits : int
        Number of qubits (must match feature dimension)

    Attributes
    ----------
    qnn : EstimatorQNN
        Qiskit's EstimatorQNN that computes expectation values
    quantum_layer : TorchConnector
        PyTorch-compatible wrapper enabling gradient computation

    Notes
    -----
    - **Feature encoding**: RY(x_i) on qubit i encodes feature x_i
    - **Observable**: Pauli-Z on qubit 2, measuring spin in computational basis
    - **Output range**: [-1, +1] (expectation value of Z operator)
    - **Gradient method**: Parameter shift rule for quantum gradients
    - **Simulator**: StatevectorEstimator (change for real quantum hardware)
    """

    def __init__(self, ansatz_circuit, num_qubits):
        super().__init__()

        # Create angle encoding feature map
        self.feature_map = self._create_angle_encoding(num_qubits)

        # Compose full quantum circuit: feature_map → ansatz
        self.qc = QuantumCircuit(num_qubits)
        self.qc.compose(self.feature_map, qubits=range(num_qubits), inplace=True)
        self.qc.compose(ansatz_circuit, inplace=True)

        # Separate input parameters (from feature map) and weight parameters (from ansatz)
        # This distinction is crucial for EstimatorQNN to correctly handle data vs. trainable weights
        input_params = list(self.feature_map.parameters)
        weight_params = list(ansatz_circuit.parameters)

        # Define observable: measure Z on qubit 2 (identity on other qubits)
        # Pauli string ordering: rightmost character = qubit 0
        observable = SparsePauliOp.from_list([("IIZII", 1)])

        # Initialize statevector simulator for noiseless quantum simulation
        # NOTE: Replace with Sampler or real backend for quantum hardware deployment
        estimator = StatevectorEstimator()


        gradient = ParamShiftEstimatorGradient(estimator)
        # Create variational quantum circuit using EstimatorQNN
        # EstimatorQNN computes <ψ|O|ψ> where |ψ> = ansatz(weights)|feature_map(x)>
        self.qnn = EstimatorQNN(
            circuit=self.qc,
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
            gradient=gradient
        )

        # Wrap the VQC as a PyTorch module
        # TorchConnector bridges Qiskit and PyTorch autograd systems,
        # allowing standard PyTorch optimizers (SGD, Adam, etc.) to train quantum parameters
        self.quantum_layer = TorchConnector(self.qnn)

    def _create_angle_encoding(self, num_qubits: int) -> QuantumCircuit:
        """
        Create angle encoding feature map: |0⟩ → RY(x₀) ⊗ RY(x₁) ⊗ ... ⊗ RY(xₙ) |0⟩

        Each classical feature x_i ∈ [0, π] is encoded as a rotation angle on qubit i.
        This maps the feature vector to the amplitude of the quantum state.

        Parameters
        ----------
        num_qubits : int
            Number of qubits (and features)

        Returns
        -------
        QuantumCircuit
            Feature map circuit with n_qubits input parameters
        """
        qc_data = QuantumCircuit(num_qubits)
        input_params = ParameterVector('x', num_qubits)
        for i in range(num_qubits):
            qc_data.ry(input_params[i], i)
        return qc_data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid variational quantum circuit.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch_size, num_qubits)

        Returns
        -------
        torch.Tensor
            Expectation values, shape (batch_size, 1), range [-1, +1]
        """
        return self.quantum_layer(x)


# ## 4. Training Configuration and Data Loading

# In[6]:


# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.01
NUM_QUBITS = 5
DEPTHS = [2, 4, 6]
K_FOLDS = 5
EVAL_EVERY = 5 # train-only diagnostics every N epochs


# ## 5. Model Training
# 
# We train the hybrid VQC using Mean Squared Error (MSE) loss and the Adam optimizer, sweeping over every depth in `DEPTHS`. During training we log **train-only** diagnostics (batch loss, train MSE, and train accuracy) every `EVAL_EVERY` epochs; to preserve the clean design, the **test set is evaluated only once, at the final epoch**. The full printed output of every fold — including the periodic diagnostics — is also written to a per-depth `training_log_depth_<d>.txt` file inside that depth's weights directory.

# In[ ]:


# Define relative paths based on the notebook's location
DATA_DIR = "../../Data"

# Validate if the data directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found at: {DATA_DIR}. Check relative path.")

# ============================ DEPTH SWEEP ============================
for ANSATZ_DEPTH in DEPTHS:
    WEIGHTS_DIR = f"../Weights/depth {ANSATZ_DEPTH}/Star"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Open a per-depth log file. log() prints to the notebook AND appends to the file,
    # so the full printed output of every fold is saved to disk.
    log_path = f"{WEIGHTS_DIR}/training_log_depth_{ANSATZ_DEPTH}.txt"
    log_file = open(log_path, "w")

    def log(msg=""):
        print(msg)
        log_file.write(f"{msg}\n")
        log_file.flush()

    log(f"================ DEPTH {ANSATZ_DEPTH} (params={NUM_QUBITS * ANSATZ_DEPTH * 3 + 2 * NUM_QUBITS}) ================")

    all_y_true = []
    all_y_pred = []
    fold_final_accuracies = []
    fold_final_f1s = []

    for fold in range(1, K_FOLDS + 1):
        log(f"\n--- FOLD {fold}/{K_FOLDS} ---")

        # Ensure deterministic behavior and independent initialization for this specific fold
        set_random_seed(RANDOM_SEED)

        # 1. Load data from the Data directory
        fold_data_dir = f"{DATA_DIR}/fold_{fold}"
        train_csv = f"{fold_data_dir}/train_data.csv"
        test_csv = f"{fold_data_dir}/test_data.csv"

        try:
            train_df = pd.read_csv(train_csv)
            test_df = pd.read_csv(test_csv)
        except FileNotFoundError:
            log(f"Error: Could not find data files at {fold_data_dir}")
            break

        X_train_scaled = train_df.drop('target', axis=1).values
        y_train = train_df['target'].values
        X_test_scaled = test_df.drop('target', axis=1).values
        y_test = test_df['target'].values

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        # Create an explicit generator for the DataLoader to guarantee deterministic shuffling
        g = torch.Generator()
        g.manual_seed(RANDOM_SEED + fold)

        # Create DataLoader for batched training
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=g
        )

        # Ensure the directory for storing weights exists inside the Weights structure
        model_fold_dir = f"{WEIGHTS_DIR}/fold_{fold}"
        os.makedirs(model_fold_dir, exist_ok=True)

        # Initialize a new ansatz and hybrid model for the current fold
        current_ansatz = star_ansatz(NUM_QUBITS, ANSATZ_DEPTH)
        model = HybridModel(current_ansatz, NUM_QUBITS)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_function = torch.nn.MSELoss()

        epoch_20_weights = None

        # Training loop (Clean: No evaluation on the test set during training)
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = loss_function(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Capture the model weights exactly after the 20th epoch (index 19)
            if epoch == 19:
                epoch_20_weights = copy.deepcopy(model.state_dict())

            # --- Train-only overfit monitoring every EVAL_EVERY epochs (no test-set peeking) ---
            if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
                model.eval()
                with torch.no_grad():
                    train_out = model(X_train_tensor)
                    train_mse = loss_function(train_out, y_train_tensor).item()
                    train_preds = (torch.round(train_out, decimals=5) > 0).float() * 2 - 1
                    train_acc = accuracy_score(
                        y_train_tensor.numpy().flatten(),
                        train_preds.numpy().flatten()
                    )
                log(f"    [epoch {epoch + 1:02d}/{EPOCHS}] "
                    f"avg_batch_loss={epoch_loss / len(train_loader):.4f} | "
                    f"train_mse={train_mse:.4f} | train_acc={train_acc:.4f}")

        # Calculate final metrics using the model from the last epoch (Epoch 30)
        # This is the ONLY time the model sees the test set, preventing data leakage
        model.eval()
        with torch.no_grad():
            preds = (torch.round(model(X_test_tensor), decimals=5) > 0).float() * 2 - 1
            y_true_np = y_test_tensor.numpy().flatten()
            preds_np = preds.numpy().flatten()

            all_y_true.extend(y_true_np)
            all_y_pred.extend(preds_np)

            final_acc = accuracy_score(y_true_np, preds_np)
            final_f1 = f1_score(y_true_np, preds_np, pos_label=1)

        fold_final_accuracies.append(final_acc)
        fold_final_f1s.append(final_f1)

        # Define paths for saving weights from Epoch 20 and Epoch 30
        path_epoch_20 = f"{model_fold_dir}/Star_fold_{fold}_depth_{ANSATZ_DEPTH}_epoch_20_weights.pth"
        path_epoch_30 = f"{model_fold_dir}/Star_fold_{fold}_depth_{ANSATZ_DEPTH}_epoch_30_weights.pth"

        # Save both checkpoints to the designated Weights directory
        if epoch_20_weights is not None:
            torch.save(epoch_20_weights, path_epoch_20)
        torch.save(model.state_dict(), path_epoch_30)

        log(f"Results fold {fold} (Epoch 30):")
        log(f"  Accuracy: {final_acc:.4f} | F1: {final_f1:.4f}")
        log(f"  -> Saved Epoch 20 weights to: {path_epoch_20}")
        log(f"  -> Saved Epoch 30 weights to: {path_epoch_30}")

    # Per-depth cross-validation summary
    if fold_final_accuracies:
        log(f"\n=== DEPTH {ANSATZ_DEPTH} CV SUMMARY ===")
        log(f"  Mean Accuracy: {np.mean(fold_final_accuracies):.4f} +/- {np.std(fold_final_accuracies):.4f}")
        log(f"  Mean F1:       {np.mean(fold_final_f1s):.4f} +/- {np.std(fold_final_f1s):.4f}")
    log(f"  -> Training log saved to: {log_path}")

    log_file.close()

