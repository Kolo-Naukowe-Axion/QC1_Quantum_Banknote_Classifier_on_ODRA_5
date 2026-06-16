# Mapowanie badań → pliki w repozytorium

Krótki przewodnik: który notebook (lub plik) odpowiada za które badanie w projekcie **Quantum Banknote Classifier on ODRA 5**.

Legenda:
- **QPU** — fizyczne wykonanie na IQM Spark (Odra 5), `backend.run` / `IQMProvider`
- **Symulator** — `StatevectorEstimator` lub `AerSimulator` (bez kolejki na hardware)
- **Brak** — wzmianka w dokumentacji, ale brak implementacji w repo

---

## 1. Fidelity na QPU

| Plik | Rola |
|------|------|
| `evaluation_and_comparison/full_odra_fidelity.ipynb` | **Główne badanie.** State fidelity z quantum state tomography na IQM Spark: submit $3^n$ baz pomiarowych, rekonstrukcja $\rho$, $\mathcal{F}=\langle\psi\vert\rho\vert\psi\rangle$. |

**Powiązane (nie QPU):**

| Plik | Rola |
|------|------|
| `evaluation_and_comparison/ansatz_comparison_fidelity.ipynb` | Proxy fidelity przez `AerSimulator` + model szumu z live kalibracji Odry — **symulator**, nie fizyczny QPU. |

---

## 2. KL divergence na QPU

| Status | Uwagi |
|--------|--------|
| **Brak pliku** | W `README.md` jest wzmianka o benchmarku: *„theoretical expressibility via Kullback-Leibler divergence to the Haar distribution”*, ale w repozytorium nie ma notebooka ani skryptu liczącego KL na QPU. |

---

## 3. Ewaluacja modelu na QPU (accuracy, F1, shot schedules)

*(Wcześniej oznaczane skrótem „MW” — w repo nie występuje taka etykieta.)*

| Plik | Rola |
|------|------|
| `evaluation_and_comparison/model_evaluation.ipynb` | Porównanie accuracy przy różnych liczbach strzałów: IQM hardware vs `StatevectorEstimator`. |
| `evaluation_and_comparison/repeated_shot_evaluation.ipynb` | Powtarzane uruchomienia na IQM (harmonogramy typu 100×50, 10×500, 1×5000 shots); zapis accuracy i F1 per run. |
| `models/model_odra.ipynb` | Sekcje **7–8**: inference na Odra 5 (pojedynczy input / batch). |
| `evaluation_and_comparison/ansatz_comparison.ipynb` | Sekcje **7–9**: porównanie ansatzów z wykonaniem na hardware. |
| `evaluation_and_comparison/gate_and_depth_comparison.ipynb` | Sekcje **7–9**: to samo w kontekście głębokości i bramek. |

**Powiązane (pomiary czasu / zasobów na QPU, bez pełnej ewaluacji klasyfikatora):**

| Plik | Rola |
|------|------|
| `evaluation_and_comparison/depth_comparison.ipynb` | Czasy wykonania i koszty obwodów na QPU dla głębokości 2, 4, 6. |
| `evaluation_and_comparison/shot_noise.ipynb` | Wizualizacja szumu strzał po strzale na IQM Spark. |

---

## 4. Matryki modeli (confusion matrix) i metryki klasyfikacji

| Plik | Gdzie w notebooku | Na czym liczone |
|------|-------------------|-----------------|
| `models/model_odra.ipynb` | Sekcja **6.2** | **Symulator** (`StatevectorEstimator`) — `confusion_matrix`, accuracy, F1 |
| `evaluation_and_comparison/ansatz_comparison.ipynb` | Sekcja **6.2** (oba ansatze) | **Symulator** |
| `evaluation_and_comparison/gate_and_depth_comparison.ipynb` | Sekcja **6.2** | **Symulator** |
| `models/model_simulator.ipynb` | Sekcja ewaluacji | **Symulator** (model referencyjny) |
| `models/classical_ml.ipynb` | Confusion matrix | Model klasyczny (Logistic Regression) |

**Na QPU** w sekcjach 8–9 powyższych notebooków (`ansatz_comparison`, `gate_and_depth_comparison`, `model_odra`) raportowane są **accuracy i F1**, ale **nie** rysowana jest macierz pomyłek z wyników hardware.

---

## Branch `Dania`

Na branchu `Dania` zebrano wyłącznie notebooki z badań **na QPU** (bez całego drzewa `main`):

- `evaluation_and_comparison/full_odra_fidelity.ipynb`
- `evaluation_and_comparison/model_evaluation.ipynb`
- `evaluation_and_comparison/repeated_shot_evaluation.ipynb`
- `models/model_odra.ipynb`
- `evaluation_and_comparison/ansatz_comparison.ipynb`
- `evaluation_and_comparison/gate_and_depth_comparison.ipynb`

KL divergence — nadal brak pliku w repozytorium.

---

## Pliki wspomniane w notebookach, ale nieobecne w repo

| Nazwa | Gdzie wspomniane |
|-------|------------------|
| `2_1_ansatz_comparison_error_gate.ipynb` | `full_odra_fidelity.ipynb` (readout error rates) |
| `all_folds_evaluation.ipynb` | `repeated_shot_evaluation.ipynb` |
