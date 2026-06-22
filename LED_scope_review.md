# Methodological Review — Is the Effective Dimension (LED) the right experiment for *our* setup?

**Purpose.** A decision memo for the team: should the Local Effective Dimension (LED) analysis stay in the main-track presentation, be rescoped, or be dropped? This lays out — with exact references — where our experimental setup falls *outside* the assumptions and validated regime of the source work, what LED still legitimately gives us, and the options.

**Reference paper.** Abbas, Sutter, Zoufal, Lucchi, Figalli, Woerner, *"The power of quantum neural networks,"* **Nat. Comput. Sci. 1, 403–409 (2021)** (arXiv:2011.00027) — peer-reviewed. (The *local* effective-dimension definition + its bound, Def. 4 / Thm 5, are in the **follow-up preprint** arXiv:2112.04807, not peer-reviewed.) Page numbers below refer to the arXiv PDF.

---

## TL;DR (recommendation)

The Abbas effective-dimension framework is **natively a cross-entropy / log-likelihood loss, large-$n$, QNN-vs-classical construction.** Our study is **MSE loss, $n\approx10^3$, QNN-vs-QNN, using the paper's own "easy" (non-entangling) feature map.** Consequences:

- LED is still **computable and meaningful as a *relative, descriptive* capacity diagnostic** (with the MSE-consistent Gaussian Fisher). ✔
- LED **does NOT license a generalisation claim** in our setup (loss-class assumptions, full-rank assumption, and large-$n$ requirement are not satisfied/established). ✘

**Bottom line:** as configured, LED is an *off-label, descriptive-only* probe. For a first main-track talk there are three defensible choices — **(A) drop it**, **(B) keep it but rescope to descriptive-only with explicit limitations**. Dropping is well-justified; rescoping is the minimum honest fix. See §5.

---

## 1. What we built (recap)

MSE-consistent (Gaussian) Local Effective Dimension for two ansätze (Odra, Simulator), depths 2/4/6 ($d=17/37/57$), 5-fold CV ($n\approx1097$–$1098$), $k=200$ Monte-Carlo inputs, evaluated at the trained $\theta^\star$. Results: models use ~7–18 % of nominal capacity; Odra uses more than Simulator; robust to the Gaussian-vs-categorical Fisher choice ($\Delta\le0.17$, because nothing saturates).

This part is internally correct (the Gaussian Fisher fix made the FIM self-consistent with our MSE objective). The concerns below are not about our code — they are about whether the *metric and its claims* fit our experimental design.

---

## 2. The core mismatch

| Dimension | Abbas et al. (the framework) | Our project |
|---|---|---|
| Loss | cross-entropy / log-likelihood / relative entropy | **MSE** (regression-style on $\langle Z\rangle$) |
| Sample size $n$ | $10^5$–$10^7$ (large-$n$ regime) | **$\approx10^3$** |
| Feature map | "hard" ZZ map (the *powerful* one) | plain RY angle-encoding = the paper's **"easy"** map |
| Fisher rank | full-rank assumed | **rank-deficient** (rank 3–6 ≪ $d$) |

Every row is a place where we are outside the source's design.

---

## 3. Evidence, point by point (with exact locations)

| # | Concern | Where in Abbas (2011.00027) | Our setup → consequence | Severity |
|---|---|---|---|---|
| 1 | **Loss must be on distributions, bounded, $\alpha$-Hölder w.r.t. total-variation.** Theorem 3.2 (Eq. 5) assumes $L:\mathrm{P}(\mathcal Y)\times\mathrm{P}(\mathcal Y)\to[-B/2,B/2]$, $\alpha$-Hölder in the 1st arg w.r.t. TV; Eq. (4) assumes $\theta\mapsto p(\cdot,\cdot;\theta)$ Lipschitz. | p.5 (Eq. 4, Thm 3.2) | Our MSE acts on the **point** output $\langle Z\rangle$, not a predictive distribution. Recasting as Gaussian NLL: boundedness holds on our restricted domain, but the **$\alpha$-Hölder-w.r.t.-TV property is *not established*** (not by Abbas, not by us). Invoking the generalisation bound for our MSE models is **unsupported**. | **High** |
| 2 | **The framework's loss IS cross-entropy / relative entropy.** "Minimising the empirical risk with a log-likelihood loss coincides with the relative entropy… we can choose $L=D(\cdot\|\cdot)$." | p.7, footnote 14 | Confirms the theory is proper-scoring-rule native. MSE is the outlier. | **High** (context) |
| 3 | **All the paper's experiments use cross-entropy.** "Using a cross-entropy loss function, optimised with ADAM…" | p.10, §4.3 | The empirical validation of ED-as-capacity/trainability was done with CE, never MSE. Our MSE setting is unvalidated for this metric. | **High** |
| 4 | **Operational interpretation requires large $n$.** "The geometric operational interpretation of the effective dimension only holds if $n$ is sufficiently large." | p.6, footnote 12 | Our $n\approx1098$ is 2–4 orders of magnitude below their experiments ($n$ up to $10^6$–$10^7$, Fig. 3a). LED is still computable ($\kappa\approx25>1$), but its **operational/capacity meaning is not on firm ground at our $n$.** | **High** |
| 5 | **Theorem 3.2 assumes a full-rank Fisher.** | p.5 (Thm 3.2), p.6 (Remark 3.4) | Our Fisher is rank-deficient (rank 3–6 ≪ $d=57$). The paper *extends* to non-full-rank via a discretisation argument (Remark 3.4 / App. B.5), so this is **mitigable** — but the base theorem still doesn't apply directly. | **Medium** |
| 6 | **Our feature map is the paper's "easy quantum model."** "…angle encoding via RY-gates on each qubit, **without entangling them**, rotations = feature values in $[-1,1]$." The paper shows this map has a **worse, more barren-plateau-prone** spectrum and **lower** effective dimension (Fig. 2, blue) than the hard ZZ map (green). | p.3, footnote 6; §2; Fig. 2 (p.8) | Our low LED (7–18 %) is partly an **artifact of using the encoding the paper identifies as weak.** Any "powerful QNN / good capacity" framing is undercut. | **High** |
| 7 | **Fisher = Hessian holds for *log-likelihood* losses.** "For certain loss functions, the FIM coincides with the Hessian." | p.2; p.6 footnote 13 | For MSE this holds only via the Gaussian (Fisher = Gauss–Newton = expected Hessian) reading we adopted — fine, but again underscores the likelihood framing. | **Low** |

---

## 4. What LED *still* legitimately gives us (the fair counter-argument)

To avoid an overstated case (which a teammate could rebut), here is what survives:

- **The effective-dimension *formula* (Def. 3.1, Eq. 2) is a functional of whatever Fisher you supply** — it is well-defined for our Gaussian Fisher and yields a meaningful number. Nothing about computing it is wrong.
- As a **relative, descriptive spectral diagnostic** — "how many parameter directions does this trained solution effectively use, and how does that compare across ansätze/depths" — it is loss-agnostic at the level of *capacity utilisation* and is internally consistent.
- We already report **Meyer–Wallach (entanglement)** and **KL (expressibility)** (`mw.tex`). As a *third* descriptive capacity probe, ED is coherent — provided it is **not load-bearing alone** and the generalisation/advantage claims are dropped.
- The Gaussian-Fisher choice is a genuine, defensible methodological contribution (matching the FIM to the training loss), arguably *more* correct than the naïve Qiskit default.

So the honest framing is **not** "ED is wrong" — it is **"ED here is a descriptive comparator, used off-label, with the inferential claims removed."** The decision is whether that is strong enough to headline a main-track talk.

---

## 5. Decision options

**(A) Drop LED from the main track.**
- *Rationale:* it cannot support generalisation or quantum-advantage claims in our setup (§3, #1–4, #6–7), and defending an off-label metric live, in a first presentation, is risky.
- *Cost:* lose one analysis; MW + KL still carry the expressibility/entanglement story.
- *Defensible?* **Yes** — §3 is solid proof of inconsistency for any inferential use.

**(B) Keep LED, rescoped to descriptive-only with explicit limitations.**
- *Required:* state plainly on the slide/paper: "We use ED as a *relative capacity descriptor* with the MSE-consistent Gaussian Fisher; we do **not** invoke the generalisation bound (loss-class, full-rank, and large-$n$ assumptions of Abbas Thm 3.2 are not satisfied in our MSE / $n\approx10^3$ setting), and we do not claim a quantum advantage (we compare two QNNs and use the simple encoding)."
- *Cost:* honest but weaker claim; invites the "then why show it?" question — answer with the multi-metric (ED+MW+KL) framing.
- *Defensible?* **Yes**, and minimal effort.

---

## 6. Recommendation

For a **first main-track presentation**, where defensibility under live questioning matters most:

- If timeline is tight: **(A) drop** or **(B) rescope.** Both are honestly defensible; (B) keeps the work visible if paired with MW/KL and the limitations are stated up front.
- **Do not** present ED with a generalisation or quantum-advantage claim under the current MSE / small-$n$ / easy-feature-map configuration — that is the one option §3 rules out.
---

## 7. References (with status)

- Abbas et al., *"The power of quantum neural networks,"* **Nat. Comput. Sci. 1, 403–409 (2021)**, arXiv:2011.00027 — **peer-reviewed.** Global effective dimension (Def. 3.1, Eq. 2); generalisation bound (Thm 3.2, Eq. 5, **global**); loss = cross-entropy/relative entropy (fn 14); large-$n$ caveat (fn 12); "easy" feature map (fn 6, §2); QNN-vs-classical Fisher spectra (Fig. 2).
- Abbas et al., *"Effective dimension of machine learning models,"* arXiv:2112.04807 (2021) — **preprint only.** Local effective dimension (Def. 4) + local generalisation bound (Thm 5) — the object we compute.
- Holmes et al. (2022) — barren plateaus / Fisher spectrum.
- MSE ⟺ Gaussian MLE: Bishop *PRML* §3.1.1; Goodfellow et al. *Deep Learning* §5.5.1. Fisher = Gauss–Newton / expected-vs-empirical Fisher: Martens, *JMLR* 2020; Kunstner et al., *NeurIPS* 2019.
