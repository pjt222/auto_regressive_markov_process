# Formalization: Conjecture and Falsification Criteria

**Version**: 0.1 (working draft)
**Date**: 2026-03-31
**Depends on**: `framework/state_definition.md`

---

## The Conjecture

**Conjecture (Approximate Markov Property on Shape Space).**

*Let $(h_t)_{t \geq 0}$ be the process defined by the recurrence:*

$$h_{t+1} = u_t h_t u_t^{-1} + \overline{B}_t x_t, \quad u_t = \exp(B_t(x_t)/2) \in \text{Spin}(D)$$

*where $(x_t)$ is drawn from a source of Markov order $k$ over a finite alphabet,
$B_t: \mathbb{R}^D \to \bigwedge^2 \mathbb{R}^D$ is a learned bivector-valued
function, and $\overline{B}_t: \mathbb{R}^D \to \mathbb{R}^D$ is a learned input
projection. Let $s_t = h_t / \|h_t\| \in S^{D-1}$ be the projected state.*

*Then the conditional mutual information satisfies:*

$$\mathcal{I}(s_{t+1}; s_{0:t-1} \mid s_t) \leq \epsilon(k, D)$$

*where $\epsilon(k, D) \to 0$ as $D \to \infty$ for fixed $k$, and
$\epsilon(1, D) = 0$ for all $D$ (the process is exactly Markov when the source
is itself first-order Markov).*

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

## What Would Disprove the Conjecture

The conjecture is falsified if any of the following are demonstrated:

1. **Empirical falsification**: For a trained model with $D \gg k$, the
   Markov score $\mathcal{I}(s_{t+1}; s_{t-j} \mid s_t)$ does NOT decay
   with lag $j$ -- i.e., past states carry irreducible information about the
   future even after conditioning on $s_t$.

2. **Theoretical falsification**: A proof that for ANY recurrence of the form
   in Definition 3.1, with $D$ finite and source order $k > 1$, the residual
   $\epsilon(k, D) \geq c > 0$ for some constant $c$ independent of $D$.
   (This would mean the curse of memory is insuperable even with geometric
   structure.)

3. **Structural falsification**: A demonstration that the sandwich product
   constraint ($u_t \in \text{Spin}(D)$) reduces expressivity so much that
   the model cannot match an unconstrained SSM on standard benchmarks.
   (This would mean the geometric structure is harmful, not helpful.)

---

## Falsification Protocol (Toy Example)

To test the conjecture at minimal scale:

1. **Setup**: $D = 3$, source = 2nd-order Markov chain on $\{0, 1\}$,
   embeddings $x_0 = (1,0,0)$, $x_1 = (0,1,0)$.

2. **Model**: Spinor SSM with $u_t = \exp(B_t/2) \in \text{Spin}(3) \cong SU(2)$,
   $B_t = \alpha(x_t) e_{12} + \beta(x_t) e_{13} + \gamma(x_t) e_{23}$ where
   $\alpha, \beta, \gamma$ are learned functions of $x_t$.

3. **Baseline**: Standard diagonal SSM (Mamba-style) with same parameter count.

4. **Metric**: Compute $\mathcal{I}(s_{t+1}; s_{t-1} \mid s_t)$ via binning
   on $S^2$ (discretize into icosahedral cells).

5. **Prediction**: If the conjecture holds, the spinor SSM should have
   lower Markov score (closer to zero) than the diagonal SSM for the same
   $D$ and $k$.

---

## Connection to Existing Results

| Result | Connection to Conjecture |
|--------|------------------------|
| Rajaraman 2024 (Thm 4.1) | Tokenization achieves $\epsilon \to 0$ as dictionary size $d \to \infty$; our $D$ plays the role of $d$ |
| Wang & Li 2023 (Thm 3.3) | Curse of memory gives $\epsilon > 0$ for exponentially decaying targets without reparameterization; our stable reparameterization is the fix |
| Bohde 2024 (ForgetNet) | Removing history when task IS Markov improves performance; our conjecture predicts this for the geometric SSM |
| Cohen 2019 (Gauge) | Gauge equivariance is necessary for the process on $S^{D-1}$ to be well-defined; our spinor constraint ensures this |

---

## Next Steps

1. **Implement Toy Example** (issue #7): Python script for the 3D falsification protocol
2. **Prove special cases**: $k=1$ (source is Markov) should be provable analytically
3. **Derive $\epsilon(k, D)$ bound**: Upper bound on the Markov score as a function of source order and state dimension
4. **Scale to medium instantiation**: Factored quaternion SSM on actual language data

---

*The strength of this conjecture is that it is specific enough to be wrong.
If it is wrong, we learn something. If it is right, we have a framework.*
