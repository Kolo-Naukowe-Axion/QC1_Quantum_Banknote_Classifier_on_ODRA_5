# Iris Classification — VQC Results

## Purpose

Test whether the same hybrid VQC architecture can solve a multiclass classification task: predict the species of an Iris flower from four morphological features. This is a classification analogue of the synthetic sin regression benchmark.

## Setup

| Parameter | Value |
|-----------|-------|
| Target | Iris species: Setosa / Versicolor / Virginica |
| Samples | 150 (120 train / 30 test, 80/20 stratified split) |
| Qubits | 4 (one per feature: sepal length, sepal width, petal length, petal width) |
| Ansatz depth | 6 |
| Trainable params | 48 (2 × 4 × 6) |
| Feature scaling | MinMaxScaler → \([-\pi/4, \pi/4]\) on all 4 features |
| Optimizer | Adam, lr = 0.01 |
| Batch size | 16 |
| Epochs | 30 |
| Loss | MSE |
| Simulator | `StatevectorEstimator` (noise-free) |
| Random seed | 42 |

Class labels \(\{0, 1, 2\}\) are mapped to \(\{-1.0, 0.0, +1.0\}\) to match the Pauli-Z expectation output range of the VQC. At inference, the continuous output is snapped to the nearest class target.

## Final Test Performance

| Metric | Value |
|--------|-------|
| Test Loss (MSE) | 0.1151 |
| Test Accuracy | 0.8667 (26 / 30) |

The model correctly classifies **87%** of held-out samples after 30 epochs.

## Training Log

| Epoch | Train Loss | Test Loss | Test Acc |
|------:|-----------:|----------:|---------:|
| 1 | 1.1460 | 1.0378 | 0.3333 |
| 2 | 0.9906 | 0.8647 | 0.3333 |
| 3 | 0.8101 | 0.6935 | 0.3333 |
| 4 | 0.6499 | 0.5663 | 0.3333 |
| 5 | 0.5165 | 0.4620 | 0.3333 |
| 6 | 0.4153 | 0.3742 | 0.3333 |
| 7 | 0.3348 | 0.2966 | 0.3667 |
| 8 | 0.2750 | 0.2469 | 0.4667 |
| 9 | 0.2132 | 0.2100 | 0.6667 |
| 10 | 0.1819 | 0.1898 | 0.8000 |
| 11 | 0.1673 | 0.1669 | 0.8333 |
| 12 | 0.1543 | 0.1592 | 0.8667 |
| 13 | 0.1392 | 0.1513 | 0.8333 |
| 14 | 0.1376 | 0.1493 | 0.8667 |
| 15 | 0.1311 | 0.1498 | 0.8333 |
| 16 | 0.1289 | 0.1391 | 0.8667 |
| 17 | 0.1210 | 0.1385 | 0.8667 |
| 18 | 0.1215 | 0.1370 | 0.8667 |
| 19 | 0.1193 | 0.1363 | 0.8667 |
| 20 | 0.1164 | 0.1386 | 0.8667 |
| 21 | 0.1220 | 0.1308 | 0.8667 |
| 22 | 0.1118 | 0.1297 | 0.8667 |
| 23 | 0.1142 | 0.1306 | 0.9000 |
| 24 | 0.1142 | 0.1279 | 0.8667 |
| 25 | 0.1112 | 0.1284 | 0.8667 |
| 26 | 0.1072 | 0.1226 | 0.8667 |
| 27 | 0.1061 | 0.1217 | 0.8667 |
| 28 | 0.1106 | 0.1231 | 0.9000 |
| 29 | 0.1028 | 0.1183 | 0.8667 |
| 30 | 0.1013 | 0.1151 | 0.8667 |

## Observations

- **Slow start — random baseline for 6 epochs**: test accuracy stays at 33.3% (random chance for 3 classes) through epoch 6, indicating the circuit has not yet learned any discriminative structure.
- **Sharp learning transition (epochs 7–10)**: accuracy jumps from 33% → 80% in just four epochs as the loss drops steeply from ~0.30 to ~0.19. This mirrors a phase transition typical of quantum circuits finding a useful parameter regime.
- **Rapid convergence**: the model reaches 86.7% accuracy by epoch 12 and essentially plateaus there for the remainder of training.
- **Peak accuracy at epochs 23 and 28**: test accuracy briefly hits 90% (27/30 correct) at epochs 23 and 28 before settling back at 86.7% — likely a fluctuation from the small test set (30 samples).
- **No significant overfitting**: train loss and test loss track closely throughout, with the gap narrowing toward epoch 30 rather than widening.
- **Steady late-stage improvement**: both train and test loss continue to decline slowly from epoch 12 onward (0.159 → 0.115), suggesting the model has not fully converged and could benefit from additional epochs.

## Conclusion

The architecture **successfully learns to classify all three Iris species** using a 4-qubit VQC with a single Pauli-Z readout mapped to \(\{-1, 0, +1\}\). Reaching ~87% accuracy on a 30-sample test set is strong for a shallow circuit with only 48 trainable parameters. The Setosa class (linearly separable in feature space) is likely classified near-perfectly, with most errors falling on the Versicolor / Virginica boundary. Deeper ansatz, additional epochs, or a multi-observable readout could push accuracy higher.

See `models/model_simulator.ipynb` for the full notebook (data preparation, training, evaluation, and plots).
