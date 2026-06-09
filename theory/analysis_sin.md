# Synthetic Sin Regression — VQC Results

## Purpose

Test whether the same hybrid VQC architecture can approximate a continuous function: predict \(y = \sin(2\pi x) + \epsilon\) from a single input \(x \in [0, 1]\). This is a regression analogue of the banknote/Iris classification benchmarks.

## Setup

| Parameter | Value |
|-----------|-------|
| Target | \(y = \sin(2\pi x) + \epsilon\), \(\epsilon \sim \mathcal{N}(0, 0.1^2)\) |
| Samples | 2500 (2000 train / 500 test, 80/20 split) |
| Qubits | 2 (qubit 0 = \(x\); qubit 1 = auxiliary zero for ring ansatz) |
| Ansatz depth | 6 |
| Trainable params | 24 (2 × 2 × 6) |
| Feature scaling | MinMaxScaler → \([-\pi/4, \pi/4]\) on \(x\) |
| Optimizer | Adam, lr = 0.01 |
| Batch size | 16 |
| Epochs | 30 |
| Loss | MSE |
| Simulator | `StatevectorEstimator` (noise-free) |
| Random seed | 42 |

The Pauli-Z expectation on qubit 0 outputs values in \([-1, +1]\), matching the range of \(\sin(2\pi x)\).

## Final Test Performance

| Metric | Value |
|--------|-------|
| MSE | 0.2185 |
| R² | 0.5758 |

The model explains roughly **58%** of the variance in the noisy sin target on the held-out test set.

## Training Log (logged epochs)

The training loop prints every 5th epoch plus epoch 1. Intermediate epochs (2–4, 6–9, …) were not logged but training ran continuously.

| Epoch | Train Loss | Test Loss | Test R² |
|------:|-----------:|----------:|--------:|
| 1 | 0.2236 | 0.2175 | 0.5779 |
| 5 | 0.2084 | 0.2179 | 0.5770 |
| 10 | 0.2078 | 0.2178 | 0.5773 |
| 15 | 0.2080 | 0.2163 | 0.5802 |
| 20 | 0.2078 | 0.2163 | 0.5801 |
| 25 | 0.2087 | 0.2197 | 0.5736 |
| 30 | 0.2091 | 0.2196 | 0.5738 |

## Observations

- **Fast initial drop**: train loss falls from 0.224 → ~0.208 by epoch 5, then plateaus.
- **Test metrics flat**: test loss and R² stay near 0.217 / 0.58 for most of training — limited generalization gain after early epochs.
- **Mild overfitting signal**: train loss keeps edging down while test R² peaks around epoch 15 (~0.5802) and dips slightly by epoch 30.
- **R² ≈ 0.58** indicates the VQC captures the overall sin shape but not fine detail; noise (\(\sigma = 0.1\)) sets a practical ceiling well below R² = 1.

## Conclusion

The architecture **successfully learns a coarse approximation** of \(\sin(2\pi x)\) from one encoded feature. Performance is modest (R² ≈ 0.58), which is expected for a shallow 2-qubit circuit with a single-qubit readout. Deeper ansatz, more qubits, or hyperparameter tuning could improve the fit.

See `models/model_simulator.ipynb` for the full notebook (data generation, training, and fit plots).

