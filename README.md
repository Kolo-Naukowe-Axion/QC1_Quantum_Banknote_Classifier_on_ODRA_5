# **Quantum Banknote Classifier on ODRA 5**

> **Official Research Project:** A hybrid quantum-classical machine learning study on **ODRA 5** — Poland's first superconducting quantum computer, hosted at **Wrocław University of Science and Technology (WUST)**.

This repository implements and evaluates a **Variational Quantum Classifier (VQC)** for banknote authentication on the **IQM Spark ODRA 5** processor. The central comparison is between a **simulator-oriented ansatz** (CRX/CRY ring) and a **hardware-aligned ODRA ansatz** (RZ/CZ ring) tailored to the native gate set of IQM Spark.

The full empirical study and five-dimensional evaluation framework are documented in [`main.tex`](main.tex).

---

## Student Research Group: **KN Axion**

* **Affiliation**: Wrocław University of Science and Technology (**WUST**).
* **Mission**: Exploring practical IT applications of Quantum Information Science.
* **Project Role**: Hardware-native ansatz design, multi-dimensional benchmarking, and end-to-end classification on **IQM Spark ODRA 5**.
* **Members**: Iwo Wojtakajtis, Rafał Balicki, Karina Leśkiewicz, Maria Płatek, Michał Szczęsny.

---

## Repository Structure

* **`src/qbanknote/`** – Installable Python package with shared project logic:
  * `ansatzes.py` – Simulator-oriented and ODRA-native ansatz families (trimmed and legacy variants).
  * `model.py` – Hybrid quantum-classical `HybridModel` (angle encoding + EstimatorQNN + TorchConnector).
  * `data.py` – UCI dataset loading, feature engineering, and five-fold CV fold I/O.
  * `evaluation.py` – Resumable IQM Spark CV classification metric-test workflow.
  * `iqm.py` – IQM backend connection and hardware estimator helpers.
  * `metrics.py`, `tomography.py`, `stats.py` – Expressibility, Meyer–Wallach, and statistical analysis utilities.
  * `weights.py` – Cross-validation checkpoint discovery and safe loading.
* **`cross_validation/`** – Stratified five-fold data splits, training notebooks, and CV checkpoints under `Models/Weights/`.
* **`models/`** – Reference notebooks for classical ML, simulator VQC, and ODRA-tuned VQC training.
* **`evaluation_and_comparison/`** – Analysis notebooks and experiment outputs:
  * `simulator/` – Depth/gate comparisons, LED analysis, RAM profiling.
  * `iqm_spark/` – Fidelity proxies, KL expressibility, shot noise, Meyer–Wallach tomography, and physical QPU classification runs.
* **`scripts/`** – CLI runners for IQM hardware experiments (metric test, Meyer–Wallach pilot, protocol selection).
* **`tests/`** – Smoke tests for package imports, ansatz parameter counts, and workflow helpers.
* **`eda.ipynb`** – Exploratory Data Analysis of the UCI dataset.
* **`main.tex`** – Manuscript: *Hardware-Efficient Ansatz Design and Noise-Aware Analysis of a Variational Quantum Classifier for IQM Spark*.
* **`pyproject.toml`** – Package metadata and dependencies for the `qbanknote` distribution.

---

## Project Overview

The core objective is to compare ansatz designs under the constraints of a real near-term QPU, not only in simulation. Using the **UCI Banknote Authentication** dataset as a binary classification benchmark, we evaluate:

* **Hardware-native co-design**: A simulator-oriented ansatz vs. an ODRA-aligned ansatz built from native RZ/CZ entanglers.
* **Multi-dimensional benchmarking**: Five complementary evaluation axes (see below), rather than training accuracy alone.
* **Reproducible methodology**: Shared library code, versioned CV folds, checkpointed weights, and scripted hardware workflows.

At depth \(L=4\) on the physical IQM Spark QPU, the hardware-aligned ansatz reaches **87.6% accuracy** (\(F_1\): 0.846) vs. **74.5%** (\(F_1\): 0.583) for the simulator-oriented baseline. At \(L=6\) with Qiskit optimization level 1, compiled physical depth drops from 263 to 127 (51.7% reduction) and native two-qubit gates from 87 to 60.

---

## Dataset: Banknote Authentication

The dataset is sourced from the **UCI Machine Learning Repository** (ID: 267) and used to validate classification on **IQM Spark ODRA 5**.

### Data Characteristics

* **Source**: UCI Banknote Authentication Dataset (ID: 267).
* **Origin**: Features extracted from genuine and forged banknote-like specimens using **Wavelet Transform**.
* **Size**: 1372 instances with 4 continuous raw features and a binary target class.

### Features (Input for Quantum Encoding)

The pipeline engineers a fifth interaction feature and scales all inputs to \([-\pi/4, \pi/4]\) for **angle encoding** (\(R_y\) gates on five qubits):

1. **Variance** of Wavelet Transformed image.
2. **Skewness** of Wavelet Transformed image.
3. **Kurtosis** of Wavelet Transformed image.
4. **Entropy** of image.
5. **Variance × Skewness** interaction (engineered).

### Target

* **Class 0** / **-1**: Authentic banknote.
* **Class 1** / **+1**: Counterfeit banknote.

Cross-validation folds store labels as \(\{-1, +1\}\); the standalone `prepare_data()` helper can return \(\{0, 1\}\) labels when preferred.

### Automatic Data Loading

Data are fetched via `ucimlrepo` or loaded from pre-generated fold CSVs in `cross_validation/Data/`:

```python
from qbanknote.data import prepare_data, load_fold_arrays

# Fresh fetch with feature engineering and scaling
X_train, X_test, y_train, y_test = prepare_data()

# Pre-split CV fold (5 features, labels in {-1, +1})
X, y = load_fold_arrays(fold=1, split="test")
```

---

## Benchmarks & Performance

The evaluation framework assesses five complementary dimensions:

1. **Compiled resource costs** – Physical depth and native gate counts after transpilation.
2. **Estimated fidelity proxies** – Composite \(\mathcal{F}_{est}\) from single-qubit, two-qubit, and measurement error channels.
3. **Theoretical expressibility** – KL divergence of pairwise fidelities to the Haar reference distribution.
4. **Optimization robustness** – Five-fold cross-validation under a phenomenological expectation-value noise model.
5. **End-to-end QPU performance** – Accuracy and \(F_1\) on the physical IQM Spark processor.

Key notebooks and scripts for each axis live under `evaluation_and_comparison/` and `scripts/`. See [`main.tex`](main.tex) for full results and discussion.

---

## Installation

Requires **Python ≥ 3.11**.

```bash
# Clone the repository, then from the project root:
python -m pip install -e ".[dev]"
```

The `[dev]` extra installs `pytest`, `qiskit-aer`, and `matplotlib` for local testing and notebook support.

Verify the install:

```bash
pytest
```

---

## Usage & Workflow

### 1. Explore the data

Start with [`eda.ipynb`](eda.ipynb) for input distribution and class balance.

### 2. Train or inspect models

* [`models/classical_ml.ipynb`](models/classical_ml.ipynb) – Classical baseline.
* [`models/model_simulator.ipynb`](models/model_simulator.ipynb) – Simulator-oriented VQC.
* [`models/model_odra.ipynb`](models/model_odra.ipynb) – ODRA-native VQC.

Cross-validation training and noise-aware runs are under `cross_validation/Models/Training/`. Checkpoints are stored in `cross_validation/Models/Weights/`.

### 3. Run simulator-side analysis

Notebooks in `evaluation_and_comparison/simulator/` cover depth/gate comparisons, LED statistics, and RAM profiling. Expressibility and fidelity analyses for IQM Spark are in `evaluation_and_comparison/iqm_spark/`.

### 4. Run IQM Spark hardware experiments

Hardware classification uses CV checkpoints and a TOML-driven pilot/final protocol:

```bash
# Pilot phase (shot/repeat selection) at depth 4
python scripts/run_iqm_metric_test.py --phase pilot --depth 4

# Final phase with protocol chosen from pilot
python scripts/run_iqm_metric_test.py --phase final --depth 4

# Statevector-only dry run (no QPU access required)
python scripts/run_iqm_metric_test.py --phase pilot --depth 2 --statevector-only
```

Configuration: [`evaluation_and_comparison/iqm_spark/iqm_metric_test_config.toml`](evaluation_and_comparison/iqm_spark/iqm_metric_test_config.toml).

Meyer–Wallach entanglement experiments use `scripts/run_iqm_mw_pilot.py` and `scripts/run_iqm_meyer_wallach.py`; methodology is documented in [`evaluation_and_comparison/iqm_spark/iqm_mw_pilot_methodology.md`](evaluation_and_comparison/iqm_spark/iqm_mw_pilot_methodology.md).

IQM hardware access requires a valid token (via `--iqm-token` or environment configuration expected by `iqm-client`).

### 5. Use the shared library in notebooks

```python
from qbanknote.ansatzes import odra_ansatz, simulator_ansatz
from qbanknote.model import HybridModel

model = HybridModel(odra_ansatz(n_qubits=5, depth=4), num_qubits=5)
```

---

## Technical Stack

* **Hardware**: ODRA 5 (superconducting qubits, IQM Spark)
* **Software**: Qiskit, Qiskit Machine Learning, PyTorch, scikit-learn, `ucimlrepo`, `iqm-client`
* **Optimization**: Hybrid quantum-classical training (Adam optimizer via parameter-shift rule)

---
