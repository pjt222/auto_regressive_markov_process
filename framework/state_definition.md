# Formal State Definition: Autoregressive Processes on Shape Space

**Version**: 0.1 (working draft)
**Date**: 2026-03-31
**Status**: Hypothesis — not a theorem. Every claim marked *Conjecture* requires proof or empirical falsification.

---

## 1. Notation and Preliminaries

Let $\mathbb{R}^D$ denote the embedding space of dimension $D$. Denote by $Cl(p,q)$ the real Clifford algebra over a vector space with signature $(p,q)$, i.e., with $p$ basis vectors squaring to $+1$ and $q$ squaring to $-1$; $p+q = n$. The Clifford algebra $Cl(p,q)$ has dimension $2^n$ as a vector space and decomposes into graded subspaces (grade-0 scalars, grade-1 vectors, grade-2 bivectors, etc.). The spin group $\text{Spin}(p,q) \subset Cl(p,q)$ consists of unit-norm even-grade elements; its Lie algebra is $\mathfrak{spin}(p,q) \cong \mathfrak{so}(p+q)$.

For an element $u \in \text{Spin}(p,q)$, the **sandwich product** (twisted adjoint representation) acts on grade-1 vectors $v \in \mathbb{R}^n$ by:
$$\rho_u(v) = u v u^{-1}$$
This map is an isometry in $O(p,q)$, and for $u \in \text{Spin}(p,q)$ it lands in $SO(p,q)$. For the positive-definite case $q=0$, $\rho_u \in SO(n)$.

**Notation summary:**

| Symbol | Type | Meaning |
|--------|------|---------|
| $D$ | $\mathbb{Z}_{>0}$ | Model embedding dimension |
| $h_t \in \mathbb{R}^D$ | Vector | Hidden state at step $t$ |
| $x_t \in \mathbb{R}^D$ | Vector | Input token embedding at step $t$ |
| $s_t$ | Point in $\mathcal{S}$ | Formal state (defined below) |
| $G$ | Lie group | Symmetry group acting on $\mathbb{R}^D$ |
| $\mathcal{S} = \mathbb{R}^D / G$ | Quotient space | State space modulo symmetry |
| $u_t \in \text{Spin}(p,q)$ | Spinor | Rotation component of transition |
| $\overline{B}_t \in \mathbb{R}^{D \times D}$ | Matrix | Input projection, input-dependent |
| $\pi: \mathbb{R}^D \to \mathcal{S}$ | Projection | Quotient map |

---

## 2. State Space

**Definition 2.1 (Raw state).** The raw state at step $t$ is the hidden vector $h_t \in \mathbb{R}^D$ produced by the autoregressive recurrence. In an SSM-type model (e.g., Mamba), $h_t \in \mathbb{R}^{D \times N}$ where $N$ is the state expansion factor; for simplicity we write $h_t \in \mathbb{R}^D$ with $D$ absorbing this product.

**The symmetry problem.** The map $h_t \mapsto Rh_t$ for any $R \in O(D)$ produces a numerically different hidden state that encodes identical semantic content — this is the gauge freedom in representation learning. A transition kernel $P(h_{t+1} \mid h_t)$ that is well-defined as a process over *semantic content* must factor through the quotient.

**Definition 2.2 (Symmetry group).** We take $G$ to be a subgroup of $O(D)$ that acts as a semantic gauge symmetry on $\mathbb{R}^D$. Three candidate choices, in increasing size:

1. $G = \mathbb{Z}_2 = \{I, -I\}$ (sign flip): minimal, quotient is a projective space
2. $G = U(1)^{D/2}$ (phase rotations per complex pair): intermediate, quotient is a torus bundle
3. $G = SO(D)$ (full rotation group): maximal, but $\mathbb{R}^D / SO(D) \cong \mathbb{R}_{\geq 0}$ (norm only) — **this is too coarse**, collapsing all directional information

*Remark.* The full rotation quotient $\mathbb{R}^D / SO(D)$ retains only $\|h_t\|$, destroying all angular structure. The correct symmetry is a *proper subgroup* of $SO(D)$ corresponding to the actual gauge freedom of the embedding geometry. In practice, cosine similarity is invariant under positive scaling ($G \supseteq \mathbb{R}_{>0}$), so the projective quotient $\mathbb{R}^D \setminus \{0\} / \mathbb{R}_{>0} \cong S^{D-1}$ (the unit sphere) is a natural intermediate choice.

**Definition 2.3 (State space).** The **formal state space** is:
$$\mathcal{S} = S^{D-1} = \mathbb{R}^D \setminus \{0\} / \mathbb{R}_{>0}$$
equipped with the round metric. The state at step $t$ is $s_t = \pi(h_t) \in S^{D-1}$, where $\pi(h) = h / \|h\|$.

*Remark on Clifford structure.* If we embed $\mathbb{R}^D$ as the grade-1 subspace of $Cl(D, 0)$, then $S^{D-1} \subset Cl(D,0)$ and the sandwich product $\rho_u$ for $u \in \text{Spin}(D)$ acts isometrically on it. This gives $\mathcal{S}$ the structure of a homogeneous space $\text{Spin}(D) / \text{Spin}(D-1) \cong S^{D-1}$, and transitions in $\mathcal{S}$ are naturally expressed in the Clifford language.

---

## 3. Transition Kernel

**Definition 3.1 (Raw transition).** The proposed recurrence for the raw hidden state is:
$$h_{t+1} = u_t h_t u_t^{-1} + \overline{B}_t x_t, \qquad u_t = \exp(B_t / 2) \in \text{Spin}(p,q)$$
where $B_t \in \mathfrak{spin}(p,q)$ (a bivector in $Cl(p,q)$) and $\overline{B}_t \in \mathbb{R}^{D \times D}$ are both **input-dependent**: $B_t = B_t(x_t)$, $\overline{B}_t = \overline{B}_t(x_t, \Delta_t)$, with $\Delta_t$ a learned input-dependent step size (the Mamba selection mechanism).

The first term $u_t h_t u_t^{-1}$ is a rotation (parallel transport) of the previous state; the second term $\overline{B}_t x_t$ is an additive input injection.

**Definition 3.2 (State transition on $\mathcal{S}$).** The induced transition on $s_t = \pi(h_t) \in S^{D-1}$ is:
$$s_{t+1} = \pi\!\left(u_t h_t u_t^{-1} + \overline{B}_t x_t\right)$$
This is well-defined because $\pi$ is equivariant under $\text{Spin}(D)$ (by construction of the sandwich product) and the denominator $\|u_t h_t u_t^{-1} + \overline{B}_t x_t\|$ is generically nonzero.

**Connection to continuous time.** The continuous-time analogue of the transition in Definition 3.1 is the matrix ODE:
$$\frac{dh}{d\tau} = A(\tau) h(\tau) + B(\tau) x(\tau)$$
with $A(\tau) = [B(\tau), \cdot]$ (Lie bracket / commutator in $\mathfrak{spin}$). The Euler discretization with step $\Delta_t$ yields Definition 3.1 to first order in $\Delta_t$; the ZOH (zero-order hold) discretization used in Mamba gives $\overline{A}_t = \exp(\Delta_t A)$, $\overline{B}_t = (\overline{A}_t - I) A^{-1} B$.

**Connection to rate matrix / master equation.** For the purely rotational part (setting $\overline{B}_t x_t = 0$), the transition $h \mapsto u h u^{-1}$ is a deterministic rotation. To obtain a stochastic process, we add noise $\xi_t \sim \mathcal{N}(0, \sigma^2 I)$ post-normalization. The resulting Fokker-Planck equation on $S^{D-1}$ is a diffusion on the sphere with drift given by the rotation field $u_t$. In the small-noise limit, this reduces to the deterministic transition; the rate matrix formalism of NeuralMJP applies when the state space is further discretized into a finite set of cells on $S^{D-1}$.

---

## 4. Markov Property

Three distinct senses of "Markov" are in play:

| Sense | Statement | Status |
|-------|-----------|--------|
| (A) Sequence Markov | Token sequence $(x_t)$ is 1st-order Markov | False for natural language; approximately true after tokenization (Rajaraman 2024) |
| (B) Raw-state Markov | $P(h_{t+1} \mid h_0, \ldots, h_t) = P(h_{t+1} \mid h_t)$ | Holds by construction in Definition 3.1 IF $B_t$ depends only on $x_t$ (not on $h_{t-1}$) |
| (C) Semantic-state Markov | $P(s_{t+1} \mid s_0, \ldots, s_t) = P(s_{t+1} \mid s_t)$ | **This is the project's central claim; its truth depends on the transition structure** |

Sense (B) is automatic from the recurrence definition. Sense (C) is not, because the input $x_t$ introduces correlations from outside the state.

**Conjecture 4.1 (Approximate Markov property).** Let $(s_t)$ be the process on $S^{D-1}$ defined by Definition 3.2, with $x_t$ drawn from a source process of Markov order $k$. Then:
$$\mathcal{I}(s_{t+1}; s_{t-1}, \ldots, s_0 \mid s_t) \leq \epsilon(k, D, \sigma^2)$$
where $\epsilon \to 0$ as $D \to \infty$ (high-dimensional state captures sufficient statistics) or as $k \to 1$ (source is itself Markov).

*Remark.* The **curse of memory** (Wang & Li 2023, StableSSM) states that SSMs without reparameterization can only stably approximate exponentially decaying targets, implying the residual $\epsilon$ is nonzero in finite $D$. The stable reparameterization lifts this to polynomial decay, bringing $\epsilon$ closer to zero. The conjecture is approximate, not exact.

**Empirical test criterion.** The quantity $\mathcal{I}(s_{t+1}; s_{t-j} \mid s_t)$ (conditional mutual information, lag $j \geq 2$) serves as a falsifiable Markov score. If this quantity decays rapidly with $j$, the process is approximately Markov in $\mathcal{S}$. Gate norms $\|u_t\|_F$ and $\|\overline{B}_t\|_F$ provide proxy signals for how much history is being mixed at each step.

---

## 5. Scalability

The full Clifford algebra $Cl(D, 0)$ has dimension $2^D$ — computationally intractable for $D \geq 32$. Three tractable alternatives:

**Option A: Grade truncation.** Restrict to grades 0, 1, and 2 only (scalars, vectors, bivectors). The bivector subspace has dimension $\binom{D}{2} = D(D-1)/2$. The spinor $u_t = \exp(B_t/2)$ where $B_t$ is a grade-2 element. Cost: $O(D^2)$ parameters for the rotation component. This covers the full $\mathfrak{so}(D)$ Lie algebra (since $\mathfrak{spin}(D) \cong \mathfrak{so}(D)$) while discarding higher-grade interference terms.

**Option B: Factored Clifford algebra.** Decompose $\mathbb{R}^D = \bigoplus_{i=1}^{D/3} \mathbb{R}^3$ and apply $Cl(3,0)$ independently to each 3D block. The spinor group $\text{Spin}(3) \cong SU(2)$ (unit quaternions) with only 4 real parameters per block, giving $4D/3$ total parameters. Block-diagonal rotations capture local geometric structure at $O(D)$ cost.

**Option C: Low-rank spin generators.** Parameterize $B_t = \sum_{i=1}^r v_i \wedge w_i$ as a sum of $r$ simple bivectors (rank-$r$ approximation to the rotation generator). Cost: $O(rD)$ parameters. For $r \ll D$, this is a substantial compression.

**Practical recommendation.** For initial experiments, Option B (factored $Cl(3,0)^{D/3}$) gives quaternion arithmetic per block — well-studied, numerically stable, and $O(D)$ in parameters. Option A is preferable when the full $SO(D)$ orbit structure matters and $D \leq 64$.

---

## 6. Candidate Instantiations

**Instantiation 1: Toy — 3D spherical state.**
$D = 3$, $\mathcal{S} = S^2$, $G = \mathbb{R}_{>0}$. The spinor $u_t \in \text{Spin}(3) \cong SU(2)$ is a unit quaternion with 3 degrees of freedom. The state $s_t \in S^2$ is a unit 3-vector. The transition is $s_{t+1} = \pi(u_t s_t u_t^{-1} + \overline{b}_t x_t)$ where $\overline{b}_t \in \mathbb{R}^3$. The Markov score $\mathcal{I}(s_{t+1}; s_{t-1} \mid s_t)$ is computable in closed form for Gaussian inputs. This is the minimal falsifiable instance.

**Instantiation 2: Medium — factored quaternion state.**
$D = 768$ (BERT-base), block size 3, so $D/3 = 256$ blocks. Each block uses a unit quaternion $u_t^{(i)} \in SU(2)$. Total spinor parameters: $256 \times 3 = 768$. The state lives on the product manifold $(S^2)^{256}$. The Markov property is tested block-wise. This is computationally tractable and directly comparable to a standard transformer hidden state.

**Instantiation 3: Full — grade-restricted $Cl(D,0)$.**
$D = 64$ (small model), grades 0+1+2. State space $\mathcal{S} = S^{63}$. Bivector parameters: $\binom{64}{2} = 2016$ per step. The spinor $u_t = \exp(B_t/2)$ computed via matrix exponential of the $64 \times 64$ antisymmetric matrix representation of $B_t$. Full $SO(64)$ equivariance. This is the theoretically complete instance for a small model.

---

## 7. Open Questions

**Unresolved theoretical questions (require proof):**

1. **Quotient structure**: What is the correct gauge group $G$ for embedding spaces trained with cosine similarity objectives? Is $G = \mathbb{R}_{>0}$ (projective) sufficient, or does training induce additional rotational symmetry?

2. **Sufficient statistic theorem**: Under what conditions on the transition (Definition 3.1) does $s_t = \pi(h_t)$ constitute a sufficient statistic for $x_{t+1}$? The answer likely involves the rank of $\overline{B}_t$ relative to $D$.

3. **Stationary distribution**: Does the process on $S^{D-1}$ have a stationary distribution? For the purely rotational case ($\overline{B}_t = 0$), the answer is yes (uniform measure on $S^{D-1}$). With input injection, existence and uniqueness of a stationary measure is open.

4. **Mixing time**: How many steps does it take for the state distribution to approach stationarity? This determines the effective "memory length" of the process and is related to the spectral gap of the transition operator.

**Unresolved empirical questions (require experiment):**

5. **Markov score**: Does $\mathcal{I}(s_{t+1}; s_{t-j} \mid s_t)$ decay rapidly with lag $j$ in a trained model? If yes, the approximate Markov property (Conjecture 4.1) is supported.

6. **Grade truncation adequacy**: Does restricting to grades 0+1+2 lose information that grades 3+ would provide, at the scales of interest?

7. **Barbour correspondence**: Does the norm $\|h_t\|$ (which is discarded by the quotient map $\pi$) carry information analogous to Barbour's entaxy (complexity)? If so, the full state should be $(s_t, \|h_t\|) \in S^{D-1} \times \mathbb{R}_{>0}$, i.e., $\mathbb{R}^D$ itself with the product structure.

---

## References (directly informing this document)

- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752
- Wang & Li (2023). *StableSSM: Alleviating the Curse of Memory in State-Space Models.*
- Bronstein et al. (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* arXiv:2104.13478
- Cohen et al. (2019). *Gauge Equivariant Convolutional Networks.* ICML 2019
- Brehmer et al. (2023). *Geometric Algebra Transformers.* NeurIPS 2023
- Rajaraman, Jiao & Ramchandran (2024). *Toward a Theory of Tokenization in LLMs.* arXiv:2404.08335
- Bohde et al. (2024). *On the Markov Property of Neural Algorithmic Reasoning.* ICLR 2024
- Seifner & Sanchez (2023). *Neural Markov Jump Processes.* ICML 2023
- Morad et al. (2023). *Reinforcement Learning with Fast and Forgetful Memory.* arXiv:2310.04128
- Hu et al. (2023). *Latent State Models of Training Dynamics.* TMLR 2023

---

*This document is a working draft. Every conjecture is falsifiable; falsification is progress.*
