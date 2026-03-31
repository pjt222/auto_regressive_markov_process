# Formal State Definition: Autoregressive Processes on Shape Space

**Version**: 0.2 (revised after k=1 disproof and training experiment)
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
| $s_t = h_t$ | $\mathbb{R}^D$ | Formal state (= raw state; revised from $S^{D-1}$) |
| $\hat{h}_t$ | $S^{D-1}$ | Directional component $h_t / \|h_t\|$ |
| $u_t \in \text{Spin}(p,q)$ | Spinor | Rotation component of transition |
| $\overline{B}_t \in \mathbb{R}^{D \times D}$ | Matrix | Input projection, input-dependent |
| $\pi: \mathbb{R}^D \to S^{D-1}$ | Projection | Normalization map $h \mapsto h/\|h\|$ (no longer defines the state) |

---

## 2. State Space

**Definition 2.1 (Raw state).** The raw state at step $t$ is the hidden vector $h_t \in \mathbb{R}^D$ produced by the autoregressive recurrence. In an SSM-type model (e.g., Mamba), $h_t \in \mathbb{R}^{D \times N}$ where $N$ is the state expansion factor; for simplicity we write $h_t \in \mathbb{R}^D$ with $D$ absorbing this product.

**The symmetry problem.** The map $h_t \mapsto Rh_t$ for any $R \in O(D)$ produces a numerically different hidden state that encodes identical semantic content — this is the gauge freedom in representation learning. A transition kernel $P(h_{t+1} \mid h_t)$ that is well-defined as a process over *semantic content* must factor through the quotient.

**Definition 2.2 (Symmetry group).** We take $G$ to be a subgroup of $O(D)$ that acts as a semantic gauge symmetry on $\mathbb{R}^D$. Three candidate choices, in increasing size:

1. $G = \mathbb{Z}_2 = \{I, -I\}$ (sign flip): minimal, quotient is a projective space
2. $G = U(1)^{D/2}$ (phase rotations per complex pair): intermediate, quotient is a torus bundle
3. $G = SO(D)$ (full rotation group): maximal, but $\mathbb{R}^D / SO(D) \cong \mathbb{R}_{\geq 0}$ (norm only) — **this is too coarse**, collapsing all directional information

*Remark.* The full rotation quotient $\mathbb{R}^D / SO(D)$ retains only $\|h_t\|$, destroying all angular structure. The correct symmetry is a *proper subgroup* of $SO(D)$ corresponding to the actual gauge freedom of the embedding geometry. In practice, cosine similarity is invariant under positive scaling ($G \supseteq \mathbb{R}_{>0}$), so the projective quotient $\mathbb{R}^D \setminus \{0\} / \mathbb{R}_{>0} \cong S^{D-1}$ (the unit sphere) is a natural intermediate choice.

**Definition 2.3 (State space — revised).** The **formal state space** is:
$$\mathcal{S} = \mathbb{R}^D$$
The state at step $t$ is $s_t = h_t \in \mathbb{R}^D$.

*Remark (why not $S^{D-1}$).* The original formulation used $\mathcal{S} = S^{D-1}$ with the projection $\pi(h) = h/\|h\|$. This was **disproved** for the k=1 case: the norm $\|h_t\|$ carries history through the mixing weight $\lambda_t \|h_t\|$ in the recurrence, so the projected process $(s_t)$ on $S^{D-1}$ is not Markov even when the source is first-order. See `framework/proofs/k1_case_disproof.md`.

*Remark (product decomposition).* The state $h_t \in \mathbb{R}^D$ decomposes naturally into direction and magnitude: $h_t = \|h_t\| \cdot \hat{h}_t$ where $\hat{h}_t \in S^{D-1}$. Equivalently, $\mathbb{R}^D \setminus \{0\} \cong S^{D-1} \times \mathbb{R}_{>0}$. The direction $\hat{h}_t$ carries the "content" and the norm $\|h_t\|$ carries a geometrically-decayed summary of injection magnitudes — potentially analogous to Barbour's complexity (issue #35).

*Remark on Clifford structure.* The grade-1 subspace of $Cl(D, 0)$ is $\mathbb{R}^D$ itself. The sandwich product $\rho_u$ for $u \in \text{Spin}(D)$ preserves the norm and acts isometrically on the directional component. The spinor constraint is a geometric *inductive bias* on the transition, not a constraint on the state space.

---

## 3. Transition Kernel

**Definition 3.1 (Raw transition).** The proposed recurrence for the raw hidden state is:
$$h_{t+1} = \lambda_t \cdot u_t h_t u_t^{-1} + \overline{B}_t x_t, \qquad u_t = \exp(B_t / 2) \in \text{Spin}(p,q)$$
where $\lambda_t = \sigma(s_\lambda(x_t)) \in (0, 1)$ is an input-dependent scalar decay, $B_t \in \mathfrak{spin}(p,q)$ (a bivector in $Cl(p,q)$) and $\overline{B}_t \in \mathbb{R}^{D \times D}$ are both **input-dependent**: $B_t = B_t(x_t)$, $\overline{B}_t = \overline{B}_t(x_t, \Delta_t)$, with $\Delta_t$ a learned input-dependent step size (the Mamba selection mechanism).

The first term $\lambda_t \cdot u_t h_t u_t^{-1}$ combines rotation (parallel transport) with decay (forgetting); the second term $\overline{B}_t x_t$ is an additive input injection. The scalar decay $\lambda_t$ is essential: without it, the sandwich product preserves all information and the process is not approximately Markov (see toy example, issue #22).

**Definition 3.2 (State transition on $\mathcal{S} = \mathbb{R}^D$ — revised).** The state transition on $\mathcal{S} = \mathbb{R}^D$ is simply Definition 3.1 itself:
$$s_{t+1} = h_{t+1} = \lambda_t \cdot u_t h_t u_t^{-1} + \overline{B}_t x_t$$
This is Markov by construction: $h_{t+1}$ depends only on $h_t$ and $x_t$, and $x_t$ is drawn from a source independent of the state's history given $h_t$.

*Remark (the old Definition 3.2).* The original formulation used $s_t = \pi(h_t) = h_t / \|h_t\| \in S^{D-1}$ and required showing the projection preserves the Markov property. The k=1 disproof shows it does not: the norm $\|h_t\|$ enters as a hidden variable that breaks Markovianity on the sphere. The revised definition avoids this problem by taking the full state.

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
| (C) Projected-state Markov | $P(s_{t+1} \mid s_0, \ldots, s_t) = P(s_{t+1} \mid s_t)$ for $s_t = h_t/\|h_t\|$ | **DISPROVED** for k=1 (the norm leaks history). See `proofs/k1_case_disproof.md` |

Sense (B) is automatic from the recurrence definition. Sense (C) is **false** — the quotient to $S^{D-1}$ discards the norm, which carries history.

**The revised central question** is not "is the process Markov?" (it is, trivially, in $\mathbb{R}^D$) but rather: **does geometric structure (spinor transitions) on the Markov process improve next-token prediction compared to unconstrained transitions?**

**Conjecture 4.1 (Geometric inductive bias — revised).** Let $(h_t)$ be the process defined by Definition 3.1. Compare two parameterizations of the transition:

- **Spinor+Decay**: $h_{t+1} = \lambda_t \cdot u_t h_t u_t^{-1} + \overline{B}_t x_t$ with $u_t \in \text{Spin}(D)$
- **Diagonal**: $h_{t+1} = \Lambda_t \odot h_t + \overline{B}_t x_t$ with $\Lambda_t \in (0,1)^D$

*Then the Spinor+Decay model achieves lower test loss (better next-token prediction) than the Diagonal model at matched parameter count, because the rotation constraint encodes geometric structure of the embedding space as an inductive bias.*

**Empirical status** (D=3 toy, 3 seeds):

| Model | Test Loss | Markov Score |
|-------|-----------|-------------|
| Spinor+Decay (14 params) | 0.5882 ± 0.004 | 0.1793 |
| Diagonal (12 params) | 0.6446 ± 0.006 | 0.0087 |

The Spinor+Decay model achieves 8.7% lower test loss but carries more history (higher Markov score). This is **partial support**: the geometric bias helps prediction, but the rotation is information-preserving and the model uses that "extra memory" productively for a 2nd-order source.

**Open**: Does this advantage persist at higher D? Does the Markov score gap narrow? See issue #38.

*Remark (old Conjecture 4.1).* The original conjecture claimed $\epsilon(1, D) = 0$ (exact Markov on $S^{D-1}$ for 1st-order sources). This was disproved. The revised conjecture replaces a claim about Markovianity with a claim about prediction quality — a more productive and empirically testable question.

*Remark (Markov score interpretation).* The Markov score $\mathcal{I}(s_{t+1}; s_{t-j} \mid s_t)$ remains useful as a diagnostic, but it is no longer the central quantity. A model with higher Markov score may be better if it uses the extra history productively. The training experiment confirms this: Spinor+Decay has higher Markov score AND lower test loss.

---

## 5. Scalability

The full Clifford algebra $Cl(D, 0)$ has dimension $2^D$ — computationally intractable for $D \geq 32$. Three tractable alternatives:

**Option A: Grade truncation.** Restrict to grades 0, 1, and 2 only (scalars, vectors, bivectors). The bivector subspace has dimension $\binom{D}{2} = D(D-1)/2$. The spinor $u_t = \exp(B_t/2)$ where $B_t$ is a grade-2 element. Cost: $O(D^2)$ parameters for the rotation component. This covers the full $\mathfrak{so}(D)$ Lie algebra (since $\mathfrak{spin}(D) \cong \mathfrak{so}(D)$) while discarding higher-grade interference terms.

**Option B: Factored Clifford algebra.** Decompose $\mathbb{R}^D = \bigoplus_{i=1}^{D/3} \mathbb{R}^3$ and apply $Cl(3,0)$ independently to each 3D block. The spinor group $\text{Spin}(3) \cong SU(2)$ (unit quaternions) with only 4 real parameters per block, giving $4D/3$ total parameters. Block-diagonal rotations capture local geometric structure at $O(D)$ cost.

**Option C: Low-rank spin generators.** Parameterize $B_t = \sum_{i=1}^r v_i \wedge w_i$ as a sum of $r$ simple bivectors (rank-$r$ approximation to the rotation generator). Cost: $O(rD)$ parameters. For $r \ll D$, this is a substantial compression.

**Practical recommendation.** For initial experiments, Option B (factored $Cl(3,0)^{D/3}$) gives quaternion arithmetic per block — well-studied, numerically stable, and $O(D)$ in parameters. Option A is preferable when the full $SO(D)$ orbit structure matters and $D \leq 64$.

**Remark (Discrete spinor parameterizations).** The continuous $\text{Spin}(3)$
group used in each QuatBlock contains discrete polyhedral subgroups as special
cases: the binary tetrahedral group $2T$ (order 24), binary octahedral group
$2O$ (order 48), and binary icosahedral group $2I$ (order 120)
[@dechant2014clifford]. These are the only finite subgroups of $SU(2)$ beyond
cyclic and binary dihedral groups (ADE classification). If one restricts the
learned spinors $u_t$ to one of these discrete groups, the resulting model has
built-in polyhedral symmetry as an inductive bias — a stronger constraint than
continuous $\text{Spin}(3)$ but potentially more interpretable. The binary
icosahedral group $2I$ is the largest and connects to $E_8$ through spinor
induction: the $H_3$ root system in 3D generates the $H_4$ root system
(600-cell) in 4D under the spinor map [@dechant2016birth]. Whether discrete
polyhedral spinors improve or harm prediction is an open empirical question
(issue #39).

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

1. ~~**Quotient structure**: What is the correct gauge group $G$?~~ **SUPERSEDED**: The k=1 disproof shows the quotient to $S^{D-1}$ discards load-bearing information (the norm). The state is $h_t \in \mathbb{R}^D$, full stop. The quotient question becomes: under what conditions does the norm concentrate, making the spherical approximation adequate?

2. **Sufficient statistic theorem**: Under what conditions does $h_t$ constitute a sufficient statistic for $x_{t+1}$? By construction of the recurrence, $h_t$ summarizes all past inputs — the question is whether the summary is lossy or lossless for prediction. (issue #34)

3. **Stationary distribution**: Does the process on $\mathbb{R}^D$ have a stationary distribution? The decay $\lambda_t \in (0,1)$ is contractive, suggesting yes (bounded in expectation). Formal proof needed. (issue #32)

4. **Mixing time**: How many steps does it take for the state distribution to approach stationarity? This determines the effective "memory length" of the process. (issue #32)

5. **Norm concentration**: Does $\text{Var}[\log \|h_t\|] \to 0$ as $D \to \infty$? If so, the spherical approximation $(S^{D-1})$ becomes asymptotically valid, recovering the original conjecture in the large-$D$ limit.

**Unresolved empirical questions (require experiment):**

6. ~~**Markov score decay**~~ **PARTIALLY RESOLVED**: Training experiment (issue #30) shows Markov scores decay with training for both models, but Spinor+Decay retains more history than Diagonal. The revised question is whether this extra history is *productive* (it appears to be — lower test loss).

7. **Dimension scaling**: Does the prediction advantage of Spinor+Decay over Diagonal persist (or grow) at higher D? Does the Markov score gap narrow? (issue #38)

8. **Grade truncation adequacy**: Does restricting to grades 0+1+2 lose information that grades 3+ would provide, at the scales of interest?

9. **Barbour correspondence**: ~~Does the norm carry Barbour-like information?~~ **PARTIALLY CONFIRMED**: The k=1 disproof shows the norm carries a geometrically-decayed summary of injection magnitudes. Whether this maps to Barbour's entaxy remains open. (issue #35)

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
