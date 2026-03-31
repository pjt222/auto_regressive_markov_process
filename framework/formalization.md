# Formalization: Conjecture and Falsification Criteria

**Version**: 0.2 (revised after k=1 disproof and training experiment)
**Date**: 2026-03-31
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

**Revised Conjecture (Geometric Inductive Bias).**

*Let $(h_t)_{t \geq 0}$ be the process defined by the recurrence:*

$$h_{t+1} = \lambda_t \cdot u_t h_t u_t^{-1} + \overline{B}_t x_t, \quad u_t = \exp(B_t(x_t)/2) \in \text{Spin}(D), \quad \lambda_t = \sigma(s_\lambda(x_t)) \in (0,1)$$

*where $(x_t)$ is drawn from a source of Markov order $k$ over a finite alphabet,
$B_t: \mathbb{R}^D \to \bigwedge^2 \mathbb{R}^D$ is a learned bivector-valued
function, and $\overline{B}_t: \mathbb{R}^D \to \mathbb{R}^D$ is a learned input
projection. The state is $s_t = h_t \in \mathbb{R}^D$.*

*Then the Spinor+Decay SSM achieves lower next-token prediction loss than a
Diagonal SSM $h_{t+1} = \Lambda_t \odot h_t + \overline{B}_t x_t$ at matched
parameter count, because the rotation constraint acts as a geometric inductive
bias that:*

1. *Preserves angular relationships between state components (rotation is
   norm-preserving)*
2. *Couples all dimensions through off-diagonal structure (rotation mixes
   components that diagonal scaling cannot)*
3. *Reduces the effective number of free parameters while maintaining
   expressivity over the relevant symmetry group*

---

## Scope of Claim

This conjecture is explicitly:

1. **Within-class**: Both the Spinor+Decay and Diagonal SSMs are linear
   recurrences in TC^0 [@merrill2024illusion]. The claim is a constant-factor
   improvement within this complexity class, not a class-crossing improvement
   [@deletang2023chomsky].

2. **For finite-Markov-order sources**: The prediction advantage is demonstrated
   on a 2nd-order Markov source. Whether it transfers to natural language —
   where non-Markovian induction heads dominate [@olsson2022incontext] — is an
   open empirical question (issue #31).

3. **An inductive-bias-efficiency claim, not an expressivity claim**: Both
   models are universal approximators given sufficient parameters
   [@yun2020transformers]. The conjecture claims that the Spin(D) constraint
   improves *sample efficiency* — better generalization at matched parameter
   count — not that it enables a function class the diagonal model cannot reach.

4. **Not yet distinguished from non-diagonality**: The observed advantage could
   be attributable to any dense (non-diagonal) transition projecting more of the
   input's log-signature [@cirone2024theoretical], rather than to the specific
   geometry of Spin(D). A generic dense SSM ablation is needed to resolve this
   (see Adversarial Literature below).

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

**D=3 toy experiment** (2nd-order Markov source, 3 seeds, n=2000):

| Model | Params | Test Loss | Markov Score |
|-------|--------|-----------|-------------|
| Spinor+Decay | 14 | **0.5882 ± 0.004** | 0.1793 |
| Diagonal | 12 | 0.6446 ± 0.006 | **0.0087** |

**Finding**: Spinor+Decay achieves 8.7% lower test loss (better prediction) but
carries more history (20x higher Markov score). The geometric bias helps prediction
at the cost of Markovianity. See `framework/train_toy.py` and issue #30.

**Interpretation**: For a 2nd-order source, remembering lag-2 history IS useful.
The spinor's information-preserving rotation enables this. The question at higher
D (issue #38) is whether the Markov score gap narrows — whether geometric structure
becomes a more *efficient* encoder that wastes less capacity on irrelevant history.

---

## What Would Disprove the Revised Conjecture

The revised conjecture is falsified if any of the following are demonstrated:

1. **Empirical falsification**: At higher $D$ or on real language data, the
   Diagonal SSM matches or beats the Spinor+Decay SSM on test loss. This
   would mean the geometric bias does not help at practical scales.

2. **Scaling falsification**: The prediction advantage of Spinor+Decay
   shrinks with $D$ rather than growing. This would suggest the inductive
   bias is a small-$D$ artifact.

3. **Structural falsification**: A demonstration that the sandwich product
   constraint ($u_t \in \text{Spin}(D)$) reduces expressivity so much that
   the model cannot match an unconstrained SSM on standard benchmarks.
   (This would mean the geometric structure is harmful, not helpful.)

---

## Falsification Protocol

### Completed: D=3 toy experiment (issue #30)

Setup, models, and results as described in "Empirical Evidence" above.
Result: **partial support** — Spinor+Decay wins on prediction, loses on
Markovianity.

### Next: Dimension scaling experiment (issue #38)

1. **Setup**: Run `train_toy.py` at $D \in \{3, 8, 16, 32\}$, 2nd-order
   Markov source, 3 seeds each.
2. **Metric**: Track both test loss gap and Markov score gap across dimensions.
3. **Prediction**: If the conjecture holds at scale, the prediction advantage
   should persist or grow with $D$.
4. **Bonus**: If the Markov score gap shrinks with $D$, this suggests a
   crossover dimension where geometric structure dominates on both metrics.

### Future: Real language data (issue #27, #31)

Test on actual token sequences from a language model. Compare factored
quaternion SSM (Option B from state_definition.md §5) against Mamba's
diagonal SSM at D=768.

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
| Olsson et al. 2022 | Real prediction uses non-Markovian induction heads | Open (issue #31 is the test) |
| Yun et al. 2020 | Universal approximation makes Markov question underdetermined | Addressed (bias, not expressivity) |
| Merrill et al. 2024 | SSM "state" is illusory; both are TC^0 | Addressed (below the ceiling) |
| Sarrof et al. 2024 | SSM expressivity limited by linearity, not geometry | Orthogonal (smooth sources) |
| **Cirone et al. 2024** | **Both are log-signature projections; geometry may be irrelevant** | **Not addressed — needs ablation** |
| Movahedi et al. 2024 | Wrong geometric constraint can hurt | Partial (falsification criterion 3) |

**Critical open question**: Does the prediction advantage come from the Spin(D)
geometry specifically, or from any non-diagonal transition? A **generic dense SSM
ablation** at matched parameter count would resolve this. If Spinor+Decay beats
Generic Dense, the geometric structure carries the load. If not, the advantage is
merely non-diagonality.

---

## Next Steps

1. ~~**Implement Toy Example** (issue #7)~~ **DONE** — `framework/toy_example.py`, `framework/train_toy.py`
2. ~~**Prove k=1 case**~~ **DISPROVED** — `framework/proofs/k1_case_disproof.md`
3. **Dimension scaling experiment** (issue #38): Test prediction advantage across D
4. **Derive norm concentration bound**: When does $\text{Var}[\log\|h_t\|] \to 0$?
5. **Scale to factored quaternion SSM** (issue #31): Real language data at D=768
6. **Generic dense SSM ablation**: Distinguish geometric structure from non-diagonality (Cirone et al. challenge)

---

*The strength of this conjecture is that it is specific enough to be wrong.
If it is wrong, we learn something. If it is right, we have a framework.*
