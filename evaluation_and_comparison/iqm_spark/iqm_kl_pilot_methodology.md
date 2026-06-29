# KL Expressibility Pilot Methodology

> **See also:** For production QPU KL runs with a fixed budget (no half-width pilot),
> use [`iqm_kl_hardware_methodology.html`](iqm_kl_hardware_methodology.html) and
> `scripts/run_iqm_kl_hardware_study.sh` / `scripts/analyze_iqm_kl_hardware.py`.

This document explains the pilot protocol used to choose the hardware and analysis
budget for the KL expressibility workflow on IQM Spark:

- the shot count per Pauli tomography basis,
- the number of hardware–hardware fidelity pairs,
- the histogram bin count used in $D_{\mathrm{KL}}(P_{\mathrm{hardware}}\|P_{\mathrm{Haar}})$,
- and the number of repeated fixed-seed hardware iterations used to quantify drift.

The corresponding runner is:

```bash
python scripts/run_iqm_kl_pilot.py --pilot-id kl_pilot_paper
```

Per-(ansatz, depth) protocol selection (independent shots/samples/iterations per job):

```bash
python scripts/run_iqm_kl_pilot.py --pilot-id kl_pilot_paper --protocol-scope per_ansatz_depth
```

The pilot writes artifacts under:

```text
evaluation_and_comparison/iqm_spark/iqm_kl_outputs/pilots/<pilot_id>/
```

After the pilot completes, run the full study with:

```bash
python scripts/run_iqm_kl_expressibility.py \
  --protocol-json evaluation_and_comparison/iqm_spark/iqm_kl_outputs/pilots/<pilot_id>/kl_protocol_recommendation.json
```

## Quantity Being Estimated

For each random parameter pair $(\theta_a, \theta_b)$ the notebook prepares two
states on hardware, reconstructs $\rho_a$ and $\rho_b$ from full $3^n$ Pauli
tomography, and computes the overlap

$$
F = \mathrm{Tr}(\rho_a \rho_b).
$$

The empirical histogram of $F$ is compared to the Haar-random analytic density
via

$$
D_{\mathrm{KL}}(P_{\mathrm{hardware}}(F)\,\|\,P_{\mathrm{Haar}}(F)).
$$

Lower KL means the ansatz explores state space more like a Haar-random ensemble
at the tested depth.

## Cost Model

For $n=5$ qubits each fidelity pair requires

$$
2 \times 3^5 = 486
$$

tomography circuits. With about 2 minutes per state on IQM Spark, one pair costs about
**4 minutes** (shallow ansatze; depth $6$ can be slower in practice).

**Pair count for one pilot stage:**

$$
N_{\mathrm{pairs}} =
|\text{shot grid}| \times |\text{ansatze}| \times |\text{shot-pilot depths}| \times n_{\mathrm{pilot}}
$$

for the shot pilot, where $n_{\mathrm{pilot}}$ is `--pilot-samples` and each entry in
the shot grid triggers a **separate** hardware sweep (not a multiplier inside one run).

The pilot is designed to align the **shot-stability stage** with the Meyer-Wallach
pilot while keeping cheaper KL-specific choices elsewhere:

1. **Bin sensitivity** is measured offline from Haar-random draws.
2. **Shot stability** defaults to MW-aligned ranges: depths $\{2,4,6\}$, shot grid
   $\{512,1024,2048,4096,8192\}$ (max 8192), and `pilot_samples = 3` (MW uses `pilot_samples = 10`).
3. **Sample precision** collects one larger run and analyses prefixes offline (MW runs
   a separate QPU sweep for every sample count in the grid).
4. **Iteration drift** repeats the frozen protocol only after shots/samples/bins are fixed.

## Stage 1: Histogram Bins (Offline)

For a candidate bin count $B$, discretize $[0,1]$ uniformly and estimate

$$
D_{\mathrm{KL}}(P_{\mathrm{emp}}^{(B)}\,\|\,P_{\mathrm{Haar}}^{(B)}).
$$

Using Haar-random fidelity draws with the planned final sample size $N$, compare
each $B$ against a fine reference histogram ($B_{\mathrm{ref}}=400$ by default).
The chosen bin count is the smallest $B$ satisfying

$$
\max_{\text{trial}} \left| D_{\mathrm{KL}}^{(B)} - D_{\mathrm{KL}}^{(B_{\mathrm{ref}})} \right|
\leq \epsilon_B,
$$

with default

```text
epsilon_B = 0.01
```

Default tested grid:

```text
B in {50, 75, 100, 150, 200, 250, 300, 400}
```

Artifacts:

- `bin_sensitivity.csv`
- `bin_sensitivity_aggregate.csv`

## Stage 2: Shot Budget (Hardware Pilot)

**Defaults** (`scripts/run_iqm_kl_pilot.py`, MW-aligned shot coverage):

- shot-pilot depths `{2, 4, 6}`,
- `pilot_samples = 3` fidelity pairs,
- shot grid `{512, 1024, 2048, 4096, 8192}` (max 8192 shots).

Use `--pilot-samples 10` only if you need full MW parity on the shot stage (~16 h
for shot pilot alone).

For consecutive shot budgets $(S_{k-1}, S_k)$ the pilot records

$$
\Delta_S(a,L) = \left| D_{\mathrm{KL},S_k}(a,L) - D_{\mathrm{KL},S_{k-1}}(a,L) \right|.
$$

The chosen shot count is the smallest $S_k$ with

$$
\max_{a,L} \Delta_S(a,L) \leq \epsilon_S,
$$

default

```text
epsilon_S = 0.02 KL units
```

**Global scope (default):** choose the smallest $S_k$ with $\max_{a,L} \Delta_S(a,L) \leq \epsilon_S$.

**Per-job scope** (`--protocol-scope per_ansatz_depth`): for each ansatz $a$ and depth $L$
independently, choose the smallest $S_k$ with $\Delta_S(a,L) \leq \epsilon_S$.

Artifacts:

- `shot_pilot_summary.csv`
- `shot_stability.csv`
- `shot_stability_aggregate.csv`
- `shot_stability_by_job.csv` (per-job scope only)

## Stage 3: Sample Count (One Hardware Run + Offline Prefix Analysis)

After shots are fixed, collect **one** hardware run per ansatz/depth with
`max_samples` pairs (default 30). Because sampling is deterministic from the
seed, the first $N$ pairs in that run are identical to a dedicated run with
only $N$ pairs.

For each prefix length $N$ in the sample grid, bootstrap-resample the first $N$
physical fidelities and estimate the standard deviation of the resulting KL.
The 95% half-width is

$$
h_N = 1.96 \cdot \mathrm{SD}_{\mathrm{boot}}(D_{\mathrm{KL}}).
$$

Choose the smallest $N$ with

$$
\max_{a,L} h_N(a,L) \leq h_{\mathrm{target}},
$$

(default global scope). With `--protocol-scope per_ansatz_depth`, each $(a,L)$ job
gets its own smallest $N$ satisfying $h_N(a,L) \leq h_{\mathrm{target}}$, using
job-specific shots from the shot pilot.

default

```text
h_target = 0.03 KL units
```

Default prefix grid:

```text
N in {5, 8, 10, 12, 15, 20, 25, 30}
```

Artifacts:

- `sample_fidelities.csv`
- `sample_precision.csv`
- `sample_precision_aggregate.csv`

## Stage 4: Hardware Iterations And Drift

Repeat the frozen protocol with the same seed and measure run-to-run variation in
$D_{\mathrm{KL}}$. For repeated sweeps $\bar D_1,\dots,\bar D_K$, use the
Student-$t$ 95% half-width

$$
h_{\mathrm{iter}} = t_{0.975,K-1}\,\frac{s_{\mathrm{iter}}}{\sqrt{K}}.
$$

Choose the smallest $K \ge K_{\min}$ such that

$$
\max_{\text{ansatz, depth}} h_{\mathrm{iter}} \le h_{\mathrm{target}}.
$$

KL defaults are more conservative than MW because of cost:

```text
K_min = 2
K_max = 4
h_target,iter = 0.02
```

Drift-only mode after shots/samples/bins are frozen:

```bash
python scripts/run_iqm_kl_pilot.py \
  --drift-only \
  --shots 1024 \
  --samples 15 \
  --n-bins 150 \
  --target-iteration-half-width 0.02 \
  --min-iterations 2 \
  --max-iterations 4 \
  --pilot-id kl_drift_pilot_final
```

Artifacts:

- `iteration_summary.csv`
- `iteration_stability.csv`
- `iteration_precision.csv`
- `iteration_precision_aggregate.csv`

## Recommended Default Pilot

This matches `scripts/run_iqm_kl_pilot.py` defaults (MW-aligned shot stage):

```bash
python scripts/run_iqm_kl_pilot.py --pilot-id kl_pilot_paper
```

Equivalent explicit invocation:

```bash
python scripts/run_iqm_kl_pilot.py \
  --pilot-id kl_full_precision_pilot \
  --shot-grid 512 1024 2048 4096 8192 \
  --shot-pilot-depth 2 4 6 \
  --pilot-samples 3 \
  --sample-grid 5 8 10 12 15 20 25 30 \
  --max-samples 30 \
  --bin-grid 50 75 100 150 200 250 300 400 \
  --shot-tolerance 0.02 \
  --bin-tolerance 0.01 \
  --target-half-width 0.03 \
  --target-iteration-half-width 0.02 \
  --min-iterations 2 \
  --max-iterations 4
```

### Pilot wall-time estimates (@ ~4 min per fidelity pair)

Let $P_{\mathrm{shot}} = |\text{shot grid}| \times 2 \times |\text{shot-pilot depths}| \times n_{\mathrm{pilot}}$.

| Stage | Default ($n_{\mathrm{pilot}}{=}3$) | With $n_{\mathrm{pilot}}{=}10$ |
|-------|-----------------------------------|--------------------------------|
| Bins | 0 (offline) | 0 |
| Shot pilot | $5 \times 2 \times 3 \times 3 = 90$ pairs → **~6.0 h** | $5 \times 2 \times 3 \times 10 = 300$ pairs → **~20 h** |
| Sample pilot | $2 \times 3 \times 30 = 180$ pairs → **~12 h** | same **~12 h** |
| Iteration pilot ($K{=}2$, $N{=}30$) | $2 \times 2 \times 3 \times 30 = 360$ pairs → **~24 h** | same **~24 h** |

**Totals (@ 4 min/pair, $N_{\mathrm{chosen}}{=}30$, $K{=}2$):**

| Configuration | Total pairs (QPU) | Wall time |
|---------------|-------------------|-----------|
| **Default ($n_{\mathrm{pilot}}{=}3$)** | $90 + 180 + 360 = 630$ | **~42 h** |
| **$n_{\mathrm{pilot}}{=}10$** | $300 + 180 + 360 = 840$ | **~56 h** |

The runner prints this budget estimate at startup. Iteration pilot dominates because
each iteration repeats the full ansatz $\times$ depth grid at up to `max_samples` pairs.

A full notebook sweep (2 ansatze, depths $\{2,4,6\}$, 30 pairs, no pilot) is

$$
2 \times 3 \times 30 = 180\ \text{pairs} \approx 12\ \text{hours}.
$$

**Note:** MW full pilot at comparable shot/depth settings is far cheaper because each
MW sample uses only 3 circuits ($Z$, $X$, $Y$), not $486$ tomography circuits per
fidelity pair.

The recommendation file

```text
kl_protocol_recommendation.json
```

records:

- `protocol_scope` (`global` or `per_ansatz_depth`)
- `chosen_shots`, `chosen_n_samples`, `chosen_n_bins`, `chosen_iterations` (conservative max over jobs)
- `protocol_by_job` (per-job scope): nested ansatz → depth → `{shots, n_samples, n_bins, iterations}`
- tolerances and target half-widths
- and the exact rule used for each choice.

If any stage cannot meet its tolerance within the tested grid, the pilot **aborts**
with `KlPilotFallbackError` instead of silently picking the largest grid value.
Expand the relevant grid or relax the tolerance, then re-run.

## Full Study (Pilot + Production + Offline Analysis)

One command runs all three stages:

```bash
export IQM_TOKEN="..."
./scripts/run_iqm_kl_full_study.sh
```

Defaults:

- pilot id: `kl_pilot_paper`
- production run id: `kl_full_study`
- outputs:
  - `evaluation_and_comparison/iqm_spark/iqm_kl_outputs/pilots/kl_pilot_paper/`
  - `evaluation_and_comparison/iqm_spark/iqm_kl_outputs/kl_full_study/`
  - `evaluation_and_comparison/iqm_spark/iqm_kl_outputs/kl_full_study/analysis/kl_qpu_sim_haar_comparison.csv`

Resume after interruption:

```bash
SKIP_PILOT=1 ./scripts/run_iqm_kl_full_study.sh
SKIP_PILOT=1 SKIP_PRODUCTION=1 ./scripts/run_iqm_kl_full_study.sh
```

Stage-by-stage equivalents:

```bash
# 1) Pilot (~42 h QPU)
python scripts/run_iqm_kl_pilot.py --pilot-id kl_pilot_paper

# 2) Production sweep (uses protocol from pilot)
python scripts/run_iqm_kl_expressibility.py \
  --protocol-json evaluation_and_comparison/iqm_spark/iqm_kl_outputs/pilots/kl_pilot_paper/kl_protocol_recommendation.json \
  --run-id kl_full_study

# 3) Offline KL(QPU/Sim/Haar) — no QPU time
python scripts/analyze_iqm_kl_expressibility.py \
  --run-dir evaluation_and_comparison/iqm_spark/iqm_kl_outputs/kl_full_study \
  --protocol-json evaluation_and_comparison/iqm_spark/iqm_kl_outputs/pilots/kl_pilot_paper/kl_protocol_recommendation.json
```

The offline analysis recomputes noiseless statevector fidelities from the same
RNG seeds as the QPU tomography run and reports, per ansatz and depth:

- `kl_qpu_haar` — $D_{\mathrm{KL}}(P_{\mathrm{QPU}}\|P_{\mathrm{Haar}})$
- `kl_sim_haar` — $D_{\mathrm{KL}}(P_{\mathrm{Sim}}\|P_{\mathrm{Haar}})$
- `kl_qpu_sim` — $D_{\mathrm{KL}}(P_{\mathrm{QPU}}\|P_{\mathrm{Sim}})$
- `delta_kl_haar_qpu_minus_sim` and per-pair fidelity-gap statistics

## Mapping Back To The Notebook

After the pilot, copy the recommended values into
`iqm_kl_expressibility.ipynb`:

```python
N_SAMPLES = <chosen_n_samples>
SHOTS = <chosen_shots>
N_BINS = <chosen_n_bins>
EPS = 1e-12
```

Or run the production script and analyse its CSV outputs directly.

## References

- Sim, Seron and Aspuru-Guzik, "Expressibility and entangling capability of
  parameterized quantum circuits", *Advanced Quantum Technologies*, 2019.
- Paris and Rehacek, *Quantum State Estimation*, 2004.
- Notebook protocol: `evaluation_and_comparison/iqm_spark/iqm_kl_expressibility.ipynb`
- Tomography methodology: `evaluation_and_comparison/iqm_spark/full_odra_fidelity.ipynb`
