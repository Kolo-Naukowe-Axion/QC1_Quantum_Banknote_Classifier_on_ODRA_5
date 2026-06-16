# Meyer-Wallach Pilot Methodology

This document explains the pilot protocol used to choose the hardware budget for
the Meyer-Wallach (MW) entanglement workflow on IQM Spark:

- the shot count used for each Pauli measurement basis,
- the number of random ansatz parameter samples,
- and the number of repeated fixed-seed hardware iterations used to quantify drift.

The corresponding runner is:

```bash
python scripts/run_iqm_mw_pilot.py --pilot-id mw_pilot_paper
```

The pilot writes artifacts under:

```text
evaluation_and_comparison/iqm_spark/iqm_mw_outputs/pilots/<pilot_id>/
```

## Quantity Being Estimated

For an $n$-qubit pure state, the Meyer-Wallach score is

$$
Q(|\psi\rangle) =
\frac{2}{n}\sum_{i=1}^{n}\left(1-\mathrm{Tr}(\rho_i^2)\right),
$$

where $\rho_i$ is the one-qubit reduced density matrix of qubit $i$. The
hardware workflow avoids full $2^n \times 2^n$ state tomography and instead
estimates only the one-qubit marginals needed by this formula.

Each reduced state is represented by its Bloch vector:

$$
\rho_i = \frac{1}{2}(I + x_i X + y_i Y + z_i Z),
$$

where

$$
x_i = \langle X_i\rangle,\qquad
y_i = \langle Y_i\rangle,\qquad
z_i = \langle Z_i\rangle.
$$

The corresponding purity is

$$
\mathrm{Tr}(\rho_i^2)
= \frac{1}{2}(1+x_i^2+y_i^2+z_i^2).
$$

Substituting this into the Meyer-Wallach expression gives

$$
Q = 1 - \frac{1}{n}\sum_{i=1}^{n}(x_i^2+y_i^2+z_i^2).
$$

Thus, for each random ansatz parameter sample, the hardware run only needs
three circuits: one measured in the $Z$ basis, one in the $X$ basis, and one
in the $Y$ basis.

## Shot-Count Justification

For a fixed qubit $i$ and Pauli basis $P \in \{X,Y,Z\}$, each shot returns
a binary outcome

$$
Y_s \in \{-1,+1\}.
$$

The Pauli expectation estimator is the sample mean

$$
\hat p_{iP} = \frac{1}{S}\sum_{s=1}^{S}Y_s,
$$

where $S$ is the number of shots in that basis. If
$p_{iP} = \mathbb{E}[Y_s]$, then $Y_s^2=1$, so

$$
\mathrm{Var}(Y_s) =
\mathbb{E}[Y_s^2] - \mathbb{E}[Y_s]^2
= 1-p_{iP}^2.
$$

For independent shots,

$$
\mathrm{Var}(\hat p_{iP})
= \frac{1-p_{iP}^2}{S}
\leq \frac{1}{S}.
$$

Because

$$
Q = 1 - \frac{1}{n}\sum_{i,P}p_{iP}^2,
$$

the delta method gives the approximate propagation of Pauli-expectation shot
noise into one MW score:

$$
\mathrm{Var}(\hat Q)
\approx
\sum_{i,P}
\left(\frac{\partial Q}{\partial p_{iP}}\right)^2
\mathrm{Var}(\hat p_{iP})
=
\sum_{i,P}
\left(\frac{2p_{iP}}{n}\right)^2
\frac{1-p_{iP}^2}{S}.
$$

Using the single-qubit Bloch-vector constraint
$x_i^2+y_i^2+z_i^2 \leq 1$, this is bounded conservatively by

$$
\mathrm{SD}_{\mathrm{shot}}(\hat Q)
\lesssim
\frac{2}{\sqrt{nS}}.
$$

For the five-qubit IQM Spark experiment, $n=5$. With $S=4096$ shots,

$$
\frac{2}{\sqrt{5\cdot4096}} \approx 0.014.
$$

If the final MW estimate averages over $N=20$ independent random parameter
samples, the shot-noise contribution to the mean is further reduced by
$\sqrt{N}$:

$$
\frac{0.014}{\sqrt{20}} \approx 0.003.
$$

This makes finite-shot Pauli-estimation error small compared with the typical
Monte Carlo variability over random ansatz parameters.

## Pilot Rule For Choosing Shots

The pilot evaluates a grid of shot counts using the same ansatzes, depths,
random seed, and pilot parameter samples:

```text
S in {512, 1024, 2048, 4096}
```

For every ansatz $a$, depth $L$, and consecutive shot pair
$(S_{k-1}, S_k)$, the pilot computes

$$
\Delta_S(a,L)
=
\left|
\bar Q_{a,L,S_k}
-
\bar Q_{a,L,S_{k-1}}
\right|.
$$

The chosen shot count is the smallest $S_k$ satisfying

$$
\max_{a,L}\Delta_S(a,L) \leq \epsilon_S,
$$

with the default tolerance

```text
epsilon_S = 0.02 MW units
```

If no shot count satisfies this stability criterion, the protocol falls back to
the largest tested shot budget.

The script writes:

- `shot_pilot_summary.csv`: per-run MW summaries for each shot count,
- `shot_stability.csv`: per ansatz/depth consecutive-shot differences,
- `shot_stability_aggregate.csv`: max/mean differences per shot transition.

## Sample-Count Justification

Once shots are fixed, the remaining statistical uncertainty comes primarily
from Monte Carlo sampling over random ansatz parameters. For each ansatz/depth
configuration, let $Q_1,\dots,Q_N$ be the MW scores from $N$ random
parameter samples. The sample mean is

$$
\bar Q = \frac{1}{N}\sum_{j=1}^{N}Q_j,
$$

with empirical standard deviation $s_Q$. The estimated 95% half-width is

$$
h_N = z_{0.975}\frac{s_Q}{\sqrt{N}},
$$

where the implementation uses $z_{0.975}=1.96$ by default. A Student-$t$
critical value can also be used for very small $N$, but $1.96$ gives a
simple conservative planning rule once pilot variability is measured.

The pilot chooses the smallest sample count $N$ for which the worst-case
half-width across ansatz/depth configurations satisfies

$$
\max_{a,L} h_N(a,L) \leq h_{\mathrm{target}},
$$

with the default target

```text
h_target = 0.03 MW units.
```

Equivalently, for a pilot-estimated standard deviation $s_Q$, the required
number of samples is

$$
N_{\mathrm{req}}
=
\left\lceil
\left(
\frac{1.96\,s_Q}{h_{\mathrm{target}}}
\right)^2
\right\rceil.
$$

For example, if $s_Q \leq 0.06$ and $h_{\mathrm{target}}=0.03$, then
$N=20$ gives an approximate half-width

$$
1.96\frac{0.06}{\sqrt{20}} \approx 0.026.
$$

The script writes:

- `sample_pilot_summary.csv`: MW summaries for each tested sample count,
- `sample_precision.csv`: per ansatz/depth confidence half-widths,
- `sample_precision_aggregate.csv`: worst-case precision by sample count.

## Hardware Iterations And Drift

The shot and sample pilots quantify statistical precision. They do not, by
themselves, quantify time-dependent hardware drift. For that, repeat the frozen
protocol with the same seed and choose the smallest iteration count $K$ whose
worst-case drift half-width meets a target.

For repeated hardware sweeps $\bar Q_1,\bar Q_2,\dots,\bar Q_K$ at fixed shots
and $N$, define the iteration standard deviation $s_{\mathrm{iter}}$ and
Student-$t$ 95% half-width

$$
h_{\mathrm{iter}}
=
t_{0.975,K-1}\,
\frac{s_{\mathrm{iter}}}{\sqrt{K}}.
$$

Choose the smallest $K \ge K_{\min}$ such that

$$
\max_{\text{ansatz, depth}} h_{\mathrm{iter}} \le h_{\mathrm{target}}.
$$

Defaults are `--min-iterations 3`, `--max-iterations 5`, and
`--target-iteration-half-width 0.01`. Use `0.02` for qualitative depth trends.

After shots and samples are already frozen, run only the drift pilot:

```bash
python scripts/run_iqm_mw_pilot.py \
  --drift-only \
  --shots 1024 \
  --samples 60 \
  --target-iteration-half-width 0.01 \
  --min-iterations 3 \
  --max-iterations 5 \
  --pilot-id mw_drift_pilot_final
```

Using the same seed keeps the random parameter samples fixed across iterations,
so run-to-run differences mainly reflect QPU drift, calibration changes, queue
conditions, and residual shot noise rather than different random points in
parameter space.

The script writes:

- `iteration_summary.csv`: per-iteration MW summaries,
- `iteration_stability.csv`: run-to-run mean, standard deviation, min, and max,
- `iteration_precision.csv`: per ansatz/depth drift half-widths and target flags,
- `iteration_precision_aggregate.csv`: worst-case drift precision by iteration count.

## Recommended Paper-Quality Defaults

A practical default pilot is:

```bash
python scripts/run_iqm_mw_pilot.py \
  --pilot-id mw_full_precision_pilot \
  --shot-grid 512 1024 2048 4096 \
  --sample-grid 10 20 40 60 \
  --pilot-samples 10 \
  --shot-tolerance 0.02 \
  --target-half-width 0.03 \
  --target-iteration-half-width 0.01 \
  --min-iterations 3 \
  --max-iterations 5
```

For the current two-ansatz, three-depth setup, one full MW sweep costs

$$
2 \times 3 \times N \times 3
$$

circuits, because each random parameter sample requires $Z$, $X$, and
$Y$-basis measurements. At $N=20$, this is

$$
2 \times 3 \times 20 \times 3 = 360
$$

circuits per full final sweep.

The recommendation file

```text
mw_protocol_recommendation.json
```

records:

- `chosen_shots`,
- `chosen_n_samples`,
- `chosen_iterations`,
- `target_iteration_half_width`,
- `min_iterations`,
- `max_iterations`,
- `iteration_target_met`,
- `single_sample_shot_noise_sd_bound`,
- `mean_shot_noise_bound_at_chosen_samples`,
- and the exact rule used for each choice.

## References

- Meyer and Wallach, "Global entanglement in multiparticle systems", Journal of
  Mathematical Physics, 2002.
- Nielsen and Chuang, *Quantum Computation and Quantum Information*, for the
  Bloch-vector representation and Pauli measurement formalism.
- Paris and Rehacek, *Quantum State Estimation*, 2004, for quantum tomography
  methodology.
- Hoeffding, "Probability inequalities for sums of bounded random variables",
  1963, for concentration of bounded measurement outcomes.
