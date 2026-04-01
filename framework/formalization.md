# Formalization: Conjecture and Falsification Criteria

**Version**: 0.3 (revised after 7 experiments — domain-specific narrowing)
**Date**: 2026-04-01
**Depends on**: `framework/state_definition.md`

---

## The Conjecture (Revised)

~~**Original Conjecture (Approximate Markov Property on Shape Space).**~~
~~The projected process $(s_t) = (h_t/\|h_t\|)$ on $S^{D-1}$ is approximately
Markov with $\epsilon(1,D) = 0$.~~

**Status**: SUPERSEDED. The k=1 case was disproved (the norm leaks history).
See `proofs/k1_case_disproof.md`. The state is $h_t \in \mathbb{R}^D$, not
$s_t \in S^{D-1}$.

---

**Revised Conjecture (Domain-Specific Geometric Inductive Bias).**

*Let $(h_t)_{t \geq 0}$ be the process defined by the recurrence:*

$$h_{t+1} = \lambda_t \cdot u_t h_t u_t^{-1} + \overline{B}_t x_t, \quad u_t = \exp(B_t(x_t)/2) \in \text{Spin}(D), \quad \lambda_t = \sigma(s_\lambda(x_t)) \in (0,1)$$

*where $(x_t)$ is drawn from a source of Markov order $k$ over a finite alphabet,
$B_t: \mathbb{R}^D \to \bigwedge^2 \mathbb{R}^D$ is a learned bivector-valued
function, and $\overline{B}_t: \mathbb{R}^D \to \mathbb{R}^D$ is a learned input
projection. The state is $s_t = h_t \in \mathbb{R}^D$.*

*Then the Spinor+Decay SSM achieves lower next-token prediction loss than a
Diagonal SSM $h_{t+1} = \Lambda_t \odot h_t + \overline{B}_t x_t$ at matched
parameter count **when the source $(x_t)$ has genuine rotational structure**,
because the rotation constraint acts as a geometric inductive bias that:*

1. *Preserves angular relationships between state components (rotation is
   norm-preserving)*
2. *Couples all dimensions through off-diagonal structure (rotation mixes
   components that diagonal scaling cannot)*
3. *Reduces the effective number of free parameters while maintaining
   expressivity over the relevant symmetry group*

*Conversely, when the source lacks rotational symmetry (e.g., natural language),
the rotation constraint is a liability: it imposes structure the data does not
have, and the Diagonal SSM outperforms.*

---

## Scope of Claim

This conjecture is explicitly:

1. **Within-class**: Both the Spinor+Decay and Diagonal SSMs are linear
   recurrences in TC^0 [@merrill2024illusion]. The claim is a constant-factor
   improvement within this complexity class, not a class-crossing improvement
   [@deletang2023chomsky].

2. **Domain-specific**: The prediction advantage holds on data with rotational
   structure (SO(3) walks, toy Markov chains) but **not on natural language**,
   where non-Markovian induction heads dominate [@olsson2022incontext] and the
   rotation constraint is actively harmful (issue #31, resolved).

3. **An inductive-bias-efficiency claim, not an expressivity claim**: Both
   models are universal approximators given sufficient parameters
   [@yun2020transformers]. The conjecture claims that the Spin(D) constraint
   improves *sample efficiency* — better generalization at matched parameter
   count — not that it enables a function class the diagonal model cannot reach.

4. **Distinguished from mere non-diagonality at D>=9**: The dense ablation
   (#45) shows Spinor+Decay beats a generic Dense SSM at D>=9 despite 6-10x
   fewer parameters. At small D, the advantage is attributable to non-diagonality
   [@cirone2024theoretical]; at larger D, the Spin(D) geometry carries genuine
   load.

---

## Conditions

The conjecture requires:

1. **Stable reparameterization**: The bivector $B_t$ is parameterized via a
   stable function (StableSSM), ensuring the transition does not converge
   to the stability boundary.

2. **Input-dependent selection**: $B_t$ and $\overline{B}_t$ depend on $x_t$
   but NOT on $h_{t-1}$ (otherwise the raw-state Markov property (B) from
   state_definition.md is violated).

3. **Non-degeneracy**: $\overline{B}_t$ has rank $\geq k$ (sufficient capacity
   to inject $k$-step dependencies from the input).

---

## Empirical Evidence

### Summary Table

| # | Experiment | Domain | Winner | Key Result | Status |
|---|-----------|--------|--------|------------|--------|
| #30 | D=3 toy Markov | Toy | Spinor+Decay | 8.7% lower test loss vs Diagonal | Partial support |
| #38, #45 | Dimension scaling + Dense ablation | Toy | Spinor+Decay | Beats Dense at D>=9 (6-10x fewer params) | **Support** |
| #44 | Block size ablation | Toy | Rotation (any) | Sharp transition at diagonal→rotation, flat across types | **Support** |
| #31 | Language modeling (WikiText-2) | Language | Diagonal | 469 PPL vs QuatBlock 562, Pascal 535 | **Falsification (language)** |
| #50 | SO(3) rotation prediction | Rotation | Givens/QuatBlock | 20-22% lower error vs Diagonal (2.24°/2.30° vs 2.88°) | **Support (domain-specific)** |
| #40 | Discrete spinor | Toy | Inconclusive | Gumbel-softmax collapses with ES optimizer | Needs re-run |
| #47 | Convergence tracking | Toy | N/A | No spontaneous polyhedral convergence | Null result |
| #25 | Epsilon bound | Theory | N/A | Simple norm-variance bound fails; two-channel model needed | Partial |

### Detailed Results

**D=3 toy experiment** (2nd-order Markov source, 3 seeds, n=2000, issue #30):

| Model | Params | Test Loss | Markov Score |
|-------|--------|-----------|-------------|
| Spinor+Decay | 14 | **0.5882 ± 0.004** | 0.1793 |
| Diagonal | 12 | 0.6446 ± 0.006 | **0.0087** |

**Dense ablation at scale** (issue #45): At D>=9, Spinor+Decay beats a generic
Dense SSM despite having 6-10x fewer parameters. This resolves the Cirone et al.
challenge — the advantage is not mere non-diagonality but geometric structure.

**Language modeling** (WikiText-2, D=768, issue #31):

| Model | Val PPL |
|-------|---------|
| Diagonal SSM | **469** |
| Pascal (grade 1+2) | 535 |
| QuatBlock | 562 |

**Finding**: Rotation is the wrong inductive bias for language. The Diagonal SSM
wins by 14-20%. The rotation constraint assumes symmetry that language embeddings
lack. CUDA kernel achieves 732x speedup over Python.

**SO(3) rotation prediction** (D=12, 3 seeds, issue #50):

| Model | Mean Angular Error |
|-------|-------------------|
| Givens | **2.24°** |
| QuatBlock | 2.30° |
| Pascal (grade 1+2) | 2.60° |
| Diagonal | 2.88° |

**Finding**: On data with genuine rotational structure, geometric SSMs beat Diagonal
by 20-22%. Givens (SO(2) rotations) slightly outperforms QuatBlock (SU(2)) at D=12,
possibly because the paired rotation structure matches the data dimension better.
Pascal (combining per-dim decay with rotation) underperforms the tighter rotation
constraints.

### Interpretation: Clifford Grade Hierarchy

The experimental results align with a **Clifford grade hierarchy** prediction:

- **Grade 0** (scalar): Global decay — present in all models.
- **Grade 1** (vector/per-dim): Per-dimension decay ($\Lambda_t \odot h_t$) — **universal**. Helps on all domains. This is what the Diagonal SSM provides.
- **Grade 2** (bivector/rotation): Coupled rotation ($u_t h_t u_t^{-1}$) — **domain-dependent**. Helps when the data has rotational structure; hurts when it doesn't.

The Pascal model (grade 1 + grade 2) underperforms pure grade-2 models on rotation
data but outperforms them on language — suggesting that per-dim flexibility and
rotation are partially competing constraints, not additive.

---

## What Would Disprove the Revised Conjecture

The domain-specific conjecture is falsified if any of the following are
demonstrated:

1. ~~**Empirical falsification (language)**: On real language data, the
   Diagonal SSM matches or beats the Spinor+Decay SSM on test loss.~~
   **CONFIRMED** — issue #31 shows Diagonal wins on WikiText-2 (469 vs 562
   PPL). This falsified the *universal* version of the conjecture and motivated
   the domain-specific narrowing.

2. **Empirical falsification (rotation domain)**: On data with genuine
   rotational structure (SO(3) walks, molecular dynamics, N-body physics),
   the Diagonal SSM matches or beats the Spinor+Decay SSM on test loss.
   This would mean the geometric bias does not help even on its home turf.
   Current status: **not triggered** — issue #50 shows 20-22% advantage.

3. **Scaling falsification**: The prediction advantage of Spinor+Decay on
   rotation-domain data shrinks with $D$ rather than growing. This would
   suggest the inductive bias is a small-$D$ artifact.

4. **Structural falsification**: A demonstration that the sandwich product
   constraint ($u_t \in \text{Spin}(D)$) reduces expressivity so much that
   the model cannot match an unconstrained SSM on benchmarks with rotational
   structure. (Issue #50 provides evidence *against* this — rotation models
   win on rotation data.)

---

## Falsification Protocol

### Completed: D=3 toy experiment (issue #30)

Setup, models, and results as described in "Empirical Evidence" above.
Result: **partial support** — Spinor+Decay wins on prediction, loses on
Markovianity.

### Completed: Dimension scaling + Dense ablation (issues #38, #45)

Spinor+Decay beats Dense SSM at D>=9 despite 6-10x fewer parameters. The
geometric structure (not mere non-diagonality) carries the load at scale.
This resolves the Cirone et al. challenge for the toy domain.

### Completed: Language modeling (issue #31) — FALSIFICATION

WikiText-2 at D=768: Diagonal SSM (469 PPL) beats QuatBlock (562 PPL) and
Pascal grade-hierarchy (535 PPL). The rotation constraint is a liability on
language data. **This falsified the universal conjecture** and motivated the
domain-specific revision.

### Completed: SO(3) rotation prediction (issue #50) — SUPPORT

Synthetic SO(3) random walks at D=12: Givens (2.24°) and QuatBlock (2.30°)
beat Diagonal (2.88°) by 20-22%. **The geometric bias helps substantially
when the data has genuine rotational structure.**

### Next: Scale rotation experiment (issues #48, #50)

1. **D=3 and D=48 rotation prediction**: Test whether QuatBlock dominates at
   D=3 (exact SO(3) match) where it should have maximum advantage.
2. **N-body physics** (issue #48): Established benchmark with genuine SO(3)
   symmetry — a real-world test of the domain-specific conjecture.
3. **Learnable coupling topology** (issue #51): Gated rotation instead of
   fixed blocks — can the model learn *when* to rotate?

---

## Connection to Existing Results

| Result | Connection to Conjecture |
|--------|------------------------|
| Rajaraman 2024 (Thm 4.1) | Tokenization achieves $\epsilon \to 0$ as dictionary size $d \to \infty$; our $D$ plays the role of $d$ |
| Wang & Li 2023 (Thm 3.3) | Curse of memory gives $\epsilon > 0$ for exponentially decaying targets without reparameterization; our stable reparameterization is the fix |
| Bohde 2024 (ForgetNet) | Removing history when task IS Markov improves performance; our conjecture predicts this for the geometric SSM |
| Cohen 2019 (Gauge) | Gauge equivariance is necessary for the process on $S^{D-1}$ to be well-defined; our spinor constraint ensures this |

---

## Adversarial Literature

Seven papers challenge aspects of this conjecture. Full analysis in
`literature/reviews/adversarial_literature.md`. Key challenges:

| Paper | Challenge | Status |
|-------|-----------|--------|
| Deletang et al. 2023 | Expressivity is Chomsky-class-bounded | Addressed (within-class claim) |
| Olsson et al. 2022 | Real prediction uses non-Markovian induction heads | **Confirmed** — #31 shows rotation hurts on language; conjecture narrowed to domain-specific |
| Yun et al. 2020 | Universal approximation makes Markov question underdetermined | Addressed (bias, not expressivity) |
| Merrill et al. 2024 | SSM "state" is illusory; both are TC^0 | Addressed (below the ceiling) |
| Sarrof et al. 2024 | SSM expressivity limited by linearity, not geometry | Orthogonal (smooth sources) |
| **Cirone et al. 2024** | **Both are log-signature projections; geometry may be irrelevant** | **Partially resolved** — #45 shows Spinor beats Dense at D>=9; geometry matters beyond non-diagonality at scale |
| Movahedi et al. 2024 | Wrong geometric constraint can hurt | **Confirmed** — #31 shows rotation hurts on language; exactly Movahedi's warning |

**Resolved critical question**: The dense ablation (#45) shows Spinor+Decay beats
generic Dense at D>=9 with 6-10x fewer parameters. The geometric structure carries
genuine load beyond non-diagonality — but **only on data with matching symmetry**.
On language data (#31), Movahedi et al.'s warning is borne out: the wrong geometric
constraint actively hurts.

---

## Next Steps

1. ~~**Implement Toy Example** (issue #7)~~ **DONE** — `framework/toy_example.py`, `framework/train_toy.py`
2. ~~**Prove k=1 case**~~ **DISPROVED** — `framework/proofs/k1_case_disproof.md`
3. ~~**Dimension scaling experiment** (issue #38)~~ **DONE** — geometric advantage confirmed at D>=9
4. ~~**Dense ablation** (issue #45)~~ **DONE** — Spinor beats Dense at D>=9 (6-10x fewer params); Cirone challenge partially resolved
5. ~~**Scale to factored quaternion SSM** (issue #31)~~ **DONE** — Diagonal wins on language (469 vs 562 PPL); falsified universal conjecture
6. ~~**Domain-specific test** (issue #50)~~ **DONE** — Givens/QuatBlock beat Diagonal by 20-22% on SO(3) walks; confirms domain-specific thesis
7. **N-body physics benchmark** (issue #48): Real-world SO(3) data to validate beyond synthetic rotation walks
8. **Learnable coupling topology** (issue #51): Gated rotation — can the model learn when to apply rotation vs decay?
9. **Re-run #40 with gradient-based optimizer**: Fair discrete spinor comparison (ES caused Gumbel-softmax collapse)
10. **Norm concentration bound** (issue #25): Two-channel leakage model needed (simple bound failed)

---

*The strength of this conjecture is that it is specific enough to be wrong.
If it is wrong, we learn something. If it is right, we have a framework.*
