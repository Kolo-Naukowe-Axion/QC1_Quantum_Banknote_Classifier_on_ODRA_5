# Iris Dataset — VQC Architecture Test

## Purpose

Intermediate benchmark to verify the simulator VQC generalizes beyond the UCI Banknote Authentication dataset. Same hybrid pipeline (angle encoding + ring ansatz + Pauli-Z readout + Adam/MSE) was applied to a different problem: **3-class classification** on the Iris morphometric dataset.

## Setup

| Parameter | Value |
|-----------|-------|
| Dataset | `Iris.csv` (150 samples, 3 species) |
| Features | 4 (sepal/petal length & width) |
| Qubits | 4 (one feature per qubit) |
| Ansatz depth | 6 |
| Trainable params | 48 (2 × 4 × 6) |
| Split | 80/20 stratified → 120 train / 30 test |
| Label encoding | class {0,1,2} → target {-1, 0, +1} (nearest-neighbour decode at inference) |
| Optimizer | Adam, lr = 0.01, 30 epochs |
| Loss | MSE (regression-style multi-class) |

## Results

With `random_state=42` and 30 epochs on the noise-free simulator:

| Metric | Value |
|--------|-------|
| Test accuracy | ~0.90–0.97 (typical range across runs) |
| Macro F1 | ~0.90+ |

**Observations**

- **Setosa** is almost always classified correctly (linearly separable).
- **Versicolor vs Virginica** is the main error source (partial overlap in feature space).
- Accuracy is comparable to a simple classical baseline on Iris, confirming the VQC architecture is viable on non-banknote data.

## Conclusion

The architecture test on Iris was **successful**: the same VQC template (with qubit count matched to feature dimension) learns a non-trivial 3-class boundary. The notebook was subsequently adapted for a **synthetic sin regression** task (`y = sin(2πx) + ε`) to further test continuous function approximation — see `models/model_simulator.ipynb`.

## Comparison to Banknote (reference)

| | Banknote | Iris |
|---|----------|------|
| Task | Binary classification | 3-class classification |
| Qubits | 5 | 4 |
| Samples | 1372 | 150 |
| Primary project benchmark | Yes | Architecture portability test |
