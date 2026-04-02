# Complete Mathematical Formula Extraction
## Auto-Regressive Markov Process Project

**Date**: 2026-04-02  
**Purpose**: Comprehensive inventory of all mathematical formulas, equations, and formal definitions  
**Extraction Source**: state_definition.md, formalization.md, ssm_cells.py, proofs/, reports/

---

## 1. Core State Space Formulation

### 1.1 State Definition

| Formula | LaTeX | Concept | File | Notes |
|---------|-------|---------|------|-------|
| **State vector** | $h_t \in \mathbb{R}^D$ | Hidden state at timestep $t$ | state_definition.md:24 | Revised from $S^{D-1}$; includes norm information |
| **Directional component** | $\hat{h}_t = h_t / \|h_t\|$ | Unit direction on sphere $S^{D-1}$ | state_definition.md:25 | Used for normalized representation |
| **Input token** | $x_t \in \mathbb{R}^D$ | Input embedding | state_definition.md:23 | From finite token alphabet |
| **Formal state space** | $\mathcal{S} = \mathbb{R}^D$ | Space of hidden states | state_definition.md:47 | **Revised definition** (k=1 disproof) |

### 1.2 Clifford Algebra Background

| Formula | LaTeX | Concept | File | Notes |
|---------|-------|---------|------|-------|
| **Clifford algebra** | $Cl(p,q)$ | Real Clifford algebra, signature $(p,q)$ | state_definition.md:11 | Dimension $2^{p+q}$ as vector space |
| **Spin group** | $\text{Spin}(p,q)$ | Unit-norm even-grade elements of $Cl(p,q)$ | state_definition.md:11 | Lie algebra: $\mathfrak{spin}(p,q) \cong \mathfrak{so}(p+q)$ |
| **Sandwich product** | $\rho_u(v) = u v u^{-1}$ | Twisted adjoint representation | state_definition.md:14 | Acts isometrically on grade-1 vectors $v \in \mathbb{R}^n$ |
| **Special case (positive-definite)** | $u \in \text{Spin}(D), \rho_u \in SO(D)$ | Spinor action on Euclidean space | state_definition.md:15 | For $q=0$: spinor acts as $SO(D)$ rotation |

---

## 2. Main Recurrence & Transition Kernel

### 2.1 Fundamental Recurrence Equation

**Equation (Definition 3.1 from state_definition.md:61)**

$$h_{t+1} = \lambda_t \cdot u_t h_t u_t^{-1} + \overline{B}_t x_t$$

where:
- $\lambda_t = \sigma(s_\lambda(x_t)) \in (0,1)$ — scalar decay (input-dependent)
- $u_t = \exp(B_t(x_t)/2) \in \text{Spin}(p,q)$ — spinor rotation generated from bivector
- $B_t \in \mathfrak{spin}(p,q)$ — bivector (grade-2 element) derived from input
- $\overline{B}_t \in \mathbb{R}^{D \times D}$ — input projection matrix (input-dependent)
- $\Delta_t$ — step size (Mamba selection mechanism, input-dependent)

**Components:**
1. **Rotation + Decay term**: $\lambda_t \cdot u_t h_t u_t^{-1}$
   - Parallel transport via spinor sandwich product
   - Scaled by decay weight
2. **Injection term**: $\overline{B}_t x_t$
   - Additive input contribution

### 2.2 Bivector Exponential (Spinor Generator)

$$u_t = \exp(B_t / 2) \in \text{Spin}(p,q)$$

where $B_t \in \mathfrak{spin}(p,q)$ is a bivector (grade-2 element).

**For the special case $Cl(3,0)$ (quaternions):**

If $B_t = \alpha e_{23} + \beta e_{13} + \gamma e_{12}$ (three linearly independent bivectors in 3D), then:

$$u_t = \cos(|B_t|/2) + \sin(|B_t|/2) \cdot \frac{B_t}{|B_t|}$$

where $|B_t| = \sqrt{\alpha^2 + \beta^2 + \gamma^2}$.

### 2.3 Continuous-Time Analogue

$$\frac{dh}{d\tau} = A(\tau) h(\tau) + B(\tau) x(\tau)$$

where:
- $A(\tau) = [B(\tau), \cdot]$ — Lie bracket (commutator in $\mathfrak{spin}$)
- Euler discretization with step $\Delta_t$ gives Definition 3.1 to first order

---

## 3. Comparison Models

### 3.1 Diagonal SSM (Baseline / Mamba-style)

$$h_{t+1} = \text{diag}(\lambda_t) \odot h_t + \overline{B}_t x_t$$

where:
- $\lambda_t = \sigma(W_\lambda x_t + b_\lambda) \in (0,1)^D$ — per-dimension scalar decay
- $\odot$ — element-wise (Hadamard) product
- No inter-dimensional coupling

**Parameters**: $2D$ (decay projections + input injection)

### 3.2 Quaternion Block SSM (QuatBlockSSM)

**Partition state into blocks of 3**: $h_t = [h_t^{(1)}, \ldots, h_t^{(D/3)}]$ where each $h_t^{(i)} \in \mathbb{R}^3$

**Quaternion sandwich product for block $i$**:

$$h_{t+1}^{(i)} = \lambda_t \cdot q_t^{(i)} h_t^{(i)} (q_t^{(i)})^{-1} + \overline{b}_t^{(i)} x_t$$

where $q_t^{(i)} \in SU(2)$ is a unit quaternion parameterized by 3 bivector coefficients $[\alpha_t^{(i)}, \beta_t^{(i)}, \gamma_t^{(i)}]$.

**Rodrigues formula for quaternion rotation**:

Given $q = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$ (where $w = \cos(\theta/2)$, $(x,y,z) = \sin(\theta/2) \hat{n}$), the rotation of $v = (v_x, v_y, v_z)$ is:

$$v' = v + 2w(v \times (x,y,z)) + 2((x,y,z) \times v) \times (x,y,z)$$

Alternative (from ssm_cells.py:100-109):

$$\begin{aligned}
\mathbf{t} &= 2(q_{xyz} \times v) \\
v' &= v + w \cdot \mathbf{t} + (q_{xyz} \times \mathbf{t})
\end{aligned}$$

**Parameters**: $D/3$ blocks × (3 bivector + 1 decay + 3 injection) ≈ $7D/3$ per timestep

### 3.3 Givens SSM (Paired Rotation)

**Partition state into $D/2$ pairs**: $h_t = [(h_t^{(1)}, h_t^{(2)}), (h_t^{(3)}, h_t^{(4)}), \ldots]$

**Givens rotation on pair $i$**:

$$\begin{pmatrix} h_{t+1}^{(2i-1)} \\ h_{t+1}^{(2i)} \end{pmatrix} = \lambda_t \begin{pmatrix} \cos\theta_t^{(i)} & -\sin\theta_t^{(i)} \\ \sin\theta_t^{(i)} & \cos\theta_t^{(i)} \end{pmatrix} \begin{pmatrix} h_t^{(2i-1)} \\ h_t^{(2i)} \end{pmatrix} + \text{injection}$$

where $\theta_t^{(i)}$ is the input-dependent rotation angle.

**Matrix form**:
$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**Parameters**: $D/2$ (angles) + $1$ (scalar decay) + $D$ (injection) = $D/2 + D + 1$ per timestep

### 3.4 Pascal SSM (Grade Hierarchy: 1+2)

**Combines Clifford grades 1 and 2**:

$$h_{t+1} = \text{diag}(\lambda_t^{(1)}) \odot [u_t h_t u_t^{-1}] + \text{diag}(\lambda_t^{(2)}) \odot h_t + \overline{B}_t x_t$$

where:
- $\lambda_t^{(1)}$ — per-dimension decay (grade 1)
- $u_t h_t u_t^{-1}$ — rotation component (grade 2)
- $\lambda_t^{(2)}$ — additional per-dim scaling on rotated output

**Interpretation**: Each dimension has independent decay, AND blocks of 3 rotate. Tests whether per-dim flexibility and rotation cooperate or compete.

### 3.5 Gated Rotation SSM (Learnable Coupling)

**Learned blending of rotation and diagonal paths**:

$$h_{t+1} = \text{gate}_t \odot h_{\text{rotated}} + (1 - \text{gate}_t) \odot h_{\text{diagonal}} + \overline{B}_t x_t$$

where:
- $\text{gate}_t = \sigma(W_{\text{gate}} x_t + b_{\text{gate}}) \in [0,1]^{D/3}$ — per-block gate
- $h_{\text{rotated}} = \lambda_t^{\text{rot}} \cdot u_t h_t u_t^{-1}$ — quaternion rotation path
- $h_{\text{diagonal}} = \text{diag}(\lambda_t^{\text{diag}}) \odot h_t$ — diagonal decay path

**Purpose**: Learn whether rotation or diagonal is more useful per block, per timestep. On language data, gate should approach 0; on rotation data, gate should approach 1.

---

## 4. Markov Property & Scoring

### 4.1 Three Senses of "Markov"

| Sense | Formula | Status | Notes |
|-------|---------|--------|-------|
| **(A) Sequence Markov** | $(x_t)$ is 1st-order Markov | False for language; approximately true after tokenization | Rajaraman 2024 |
| **(B) Raw-state Markov** | $P(h_{t+1} \mid h_0, \ldots, h_t) = P(h_{t+1} \mid h_t)$ | **Holds by construction** | $B_t$ depends only on $x_t$, not $h_{t-1}$ |
| **(C) Projected-state Markov** | $P(s_{t+1} \mid s_0, \ldots, s_t) = P(s_{t+1} \mid s_t)$ where $s_t = h_t/\|h_t\|$ | **DISPROVED** for k=1 | The norm $\|h_t\|$ carries history (k1_case_disproof.md) |

### 4.2 Markov Score (Mutual Information Gap)

$$\mathcal{I}(s_{t+1}; s_{t-j} \mid s_t) = \epsilon(j, D)$$

**Interpretation**: Conditional mutual information measuring how much the projected state $s_{t+1}$ depends on $s_{t-j}$ *given* $s_t$. If zero, process is Markov.

**Empirical measurement** (state_definition.md:104-106):

| Model | Markov Score | Test Loss | Interpretation |
|-------|-------------|-----------|-----------------|
| Spinor+Decay | 0.1793 | **0.5882** | Higher Markov score, but **uses extra history productively** |
| Diagonal | 0.0087 | 0.6446 | Lower Markov score, worse prediction |

---

## 5. Norm Evolution & Leakage

### 5.1 Norm Recurrence (k=1 Disproof)

$$\|h_{t+1}\|^2 = \lambda_t^2 \|h_t\|^2 + 2\lambda_t \|h_t\| \langle v_t, \overline{B}_t x_t \rangle + \|\overline{B}_t x_t\|^2$$

where:
- $v_t = u_t s_t u_t^{-1} \in S^{D-1}$ — rotated directional component
- $\|h_t\|$ — magnitude of hidden state
- Cross-term explicitly depends on norm

**Key insight**: The projection $s_t = h_t / \|h_t\|$ discards the norm, which is a hidden state carrying history through $\lambda_t \|h_t\|$ weighting.

### 5.2 Log-Norm Evolution (Epsilon Bound)

Define $r_t = \log \|h_t\|$. Then:

$$r_{t+1} = r_t + \frac{1}{2}\log\left(\lambda_t^2 + 2\lambda_t e^{-r_t} \langle v_t, \overline{B}_t x_t \rangle + e^{-2r_t}\|\overline{B}_t x_t\|^2\right)$$

**For large $r_t$** (large norm, first-order approximation):

$$r_{t+1} \approx r_t + \log\lambda_t + \frac{\langle v_t, \overline{B}_t x_t \rangle}{\lambda_t e^{r_t}} + O(e^{-2r_t})$$

### 5.3 Epsilon Bound (Non-Markovianity Bound)

**Claim (Theorem sketch, epsilon_bound.md:71)**:

$$\epsilon(1, D) \leq C \cdot \frac{\text{Var}[\log\|h_t\|]}{D}$$

**Interpretation**: 
- Non-Markovianity of the sphere process is controlled by norm variance
- Decreases as $1/D$ (concentration of measure)
- At stationarity and high D, the spherical approximation becomes asymptotically valid

**Derivation sketch**:
1. Non-Markovianity arises from hidden norm $\|h_t\|$
2. By data processing: $I(s_{t+1}; s_{t-j} \mid s_t) \leq I(s_{t+1}; r_t \mid s_t)$
3. Sensitivity of $s_{t+1}$ to $r_t$: $\|\partial s_{t+1}/\partial r_t\| \leq C_1 / \sqrt{D}$
4. Final bound uses Gaussian MI: $I(X; f(X,Z)) \leq \frac{1}{2}\|f'\|^2 \text{Var}[X]$

**Comparison to Rajaraman et al. 2024**:

| Bound | Decreases with | Structure |
|-------|---|---|
| Ours | Dimension $D$ | Continuous state, norm concentration |
| Rajaraman | Alphabet size $d$ | Discrete state, tokenization |

Both: non-Markovianity $\propto 1 / \text{state resolution}$

---

## 6. Hypothesis & Falsification

### 6.1 Revised Conjecture (Formalization.md)

**Conjecture statement** (formalization.md:25-46):

Let $(h_t)$ satisfy the recurrence:
$$h_{t+1} = \lambda_t \cdot u_t h_t u_t^{-1} + \overline{B}_t x_t, \quad u_t = \exp(B_t(x_t)/2) \in \text{Spin}(D)$$

Then the **Spinor+Decay SSM achieves lower next-token prediction loss than a Diagonal SSM at matched parameter count** **when the source $(x_t)$ has genuine rotational structure**, because the rotation constraint acts as a geometric inductive bias.

**Quantitative claim**:

For the toy Markov domain (2nd-order, 4-token vocabulary, n=2000):
- Spinor+Decay: 0.5882 ± 0.004 test loss (14 params)
- Diagonal: 0.6446 ± 0.006 test loss (12 params)
- **Advantage**: 8.7% lower test loss

### 6.2 Falsification Protocol Results

| Experiment | Domain | Result | Status |
|------------|--------|--------|--------|
| #30 | D=3 toy Markov | Spinor beats Diagonal by 8.7% | Partial support |
| #38, #45 | Dimension scaling + Dense ablation | Spinor beats Dense at D≥9 (6-10x fewer params) | **Support** |
| #31 | Language (WikiText-2, D=768) | Diagonal wins: 469 PPL vs QuatBlock 562 | **Falsification** (language domain) |
| #50 | SO(3) rotation prediction (D=12) | Givens/QuatBlock beat Diagonal by 20-22% (2.24°/2.30° vs 2.88°) | **Support** (rotation domain) |
| #40 | Discrete spinor | Gumbel-softmax collapses | Inconclusive |
| #47 | Convergence tracking | No spontaneous polyhedral convergence | Null result |
| #25 | Epsilon bound | Simple norm-variance bound: R² = 0.19 (inadequate) | Partial |

---

## 7. Experimental Results as Formulas

### 7.1 Language Modeling (Issue #31)

**Setup**: WikiText-2, GPT-2 tokenizer (50257 vocab), D=768, 2 SSM layers

**Results**:

| Model | Validation PPL | vs Diagonal | Architecture |
|-------|---|---|---|
| **Diagonal** | **469** | -- | Per-dim decay |
| Pascal | 535 | +14% (worse) | Per-dim decay + rotation |
| QuatBlock | 562 | +20% (worse) | 256 quaternion blocks |

**Conclusion**: Rotation is the wrong inductive bias for language. Diagonal dominates.

**CUDA speedup**: 732× over Python implementation

### 7.2 SO(3) Rotation Prediction (Issue #50)

**Setup**: Synthetic SO(3) random walks with Ornstein-Uhlenbeck angular velocity, D=12, 3 seeds

**Metric**: Geodesic angular error in degrees (lower is better)

| Model | Mean Error | vs Diagonal | Block Type |
|-------|---|---|---|
| **Givens** | **2.24°** | -22.2% | Paired SO(2) rotations |
| QuatBlock | 2.30° | -20.1% | Quaternion (SU(2)) blocks |
| Pascal | 2.60° | -9.7% | Grade 1+2 combined |
| Diagonal | 2.88° | -- | Per-dim decay (baseline) |

**Conclusion**: On data with genuine rotational symmetry, rotation models beat Diagonal by 20-22%.

### 7.3 Toy Markov: Dimension Scaling (Issue #38)

**Source**: 2nd-order Markov, 4-token vocabulary, Givens parameterization

| D | Spinor Loss | Diagonal Loss | Gap % | Spinor Params | Diagonal Params |
|---|---|---|---|---|---|
| 3 | 0.6260 | 0.6679 | -6.3% | 10 | 12 |
| 8 | 0.6282 | 0.6627 | -5.2% | 26 | 32 |
| 16 | 0.6238 | 0.6556 | -4.8% | 50 | 64 |

**Trend**: Geometric advantage persists but narrows with increasing D (sparse parameterization effect).

### 7.4 Dense Ablation (Issue #45)

**Question**: Is Spinor advantage due to geometry or mere non-diagonality?

**Setup**: Compare Spinor+Decay vs unconstrained Dense SSM ($D \times D$ transition matrix)

**Critical transition**:
- **D=3**: Dense ≈ Spinor (slight advantage to Dense)
- **D≥9**: Spinor > Dense despite 6-10x fewer parameters

**Interpretation**: The Spin(D) constraint is not just parameter reduction — it encodes structurally relevant information at higher dimensions. Geometry matters beyond non-diagonality.

---

## 8. Scalability & Factorizations

### 8.1 Clifford Algebra Grade Truncation

**Full $Cl(D, 0)$ has dimension $2^D$** — intractable for D ≥ 32.

**Option A: Grade restriction to 0, 1, 2**

Bivector subspace dimension: $\binom{D}{2} = D(D-1)/2$

Cost: $O(D^2)$ parameters for rotation component

Covers full $\mathfrak{so}(D)$ Lie algebra

### 8.2 Factored Clifford Algebra (Quaternion Blocks)

**Decompose**: $\mathbb{R}^D = \bigoplus_{i=1}^{D/3} \mathbb{R}^3$

Apply $Cl(3,0)$ independently to each 3D block

**Spinor group**: $\text{Spin}(3) \cong SU(2)$ (unit quaternions)

**Parameters per block**: 4 real (3 bivector + 1 scalar)

**Total**: $4D/3$ parameters

**Advantage**: Numerically stable, O(D) cost, block-diagonal structure

### 8.3 Low-Rank Bivector Parameterization

**Parameterize**: $B_t = \sum_{i=1}^r v_i \wedge w_i$ (rank-r approximation)

**Parameters**: $O(rD)$ for rank $r$

**Benefit**: For $r \ll D$, substantial compression while maintaining Spin(D) structure

---

## 9. Discrete Spinor Subgroups (Polyhedral Symmetries)

### 9.1 Finite Subgroups of SU(2) (Spin(3))

**ADE classification** (only finite subgroups beyond cyclic and binary dihedral):

| Group | Order | Structure | Reference |
|-------|-------|----------|-----------|
| Binary Tetrahedral | $2T$ | 24 | rotations + reflections |
| Binary Octahedral | $2O$ | 48 | cube symmetries |
| **Binary Icosahedral** | **2I** | **120** | icosahedron symmetries |

**Equation form** (state_definition.md:131-143):

If $u_t \in 2I \subset SU(2)$, then learned spinors are restricted to the 120-element group.

### 9.2 E8 Connection (Dechant 2016)

**Spinor induction**: $H_3$ root system in 3D generates $H_4$ in 4D under spinor map

**Status** (issue #39): Whether discrete polyhedral constraints improve or harm prediction is an open empirical question

---

## 10. Open Theoretical Questions

### 10.1 Sufficient Statistic

Under what conditions does $h_t \in \mathbb{R}^D$ constitute a sufficient statistic for $x_{t+1}$?

**Question** (state_definition.md:166): Is the summary lossless or lossy for prediction?

### 10.2 Stationary Distribution

Does the process on $\mathbb{R}^D$ have a stationary distribution?

**Intuition**: Decay $\lambda_t \in (0,1)$ is contractive → bounded in expectation

**Status**: Formal proof needed

### 10.3 Mixing Time

$$\tau_{\text{mix}} = \min\{t : d_{\text{TV}}(P_t(\cdot|h_0), \pi) < \epsilon\}$$

How many steps to approach stationarity? Determines effective memory length.

### 10.4 Norm Concentration

$$\text{Var}[\log \|h_t\|] \to 0 \text{ as } D \to \infty \text{ ?}$$

**Implication**: If true, the spherical approximation $(S^{D-1})$ becomes asymptotically valid in high D.

---

## 11. Clifford Grade Hierarchy Interpretation

**Empirical pattern** (formalization.md:150-160):

| Grade | Component | Universality | Domain |
|-------|-----------|---|---|
| **0** | Scalar decay | Universal | Present in all models |
| **1** | Per-dimension decay ($\Lambda_t \odot h_t$) | **Universal** | Helps on all domains |
| **2** | Coupled rotation ($u_t h_t u_t^{-1}$) | **Domain-dependent** | Helps on rotation data; hurts on language |

**Grade-1 universality**: The Diagonal SSM (pure grade 1) wins on language.

**Grade-2 domain-dependence**: Pascal (grade 1+2) underperforms both pure models, suggesting competition rather than cooperation.

---

## 12. References & Connections

### 12.1 Key Cited Results

| Result | Authors | Citation |
|--------|---------|----------|
| Tokenization bound: $\epsilon_{\text{tok}} = \frac{\log(1/\delta)}{0.99 \log d}$ | Rajaraman, Jiao, Ramchandran 2024 | arXiv:2404.08335 |
| Curse of memory in SSMs | Wang & Li 2023 | arXiv:2311.14495 |
| Mamba: selective state spaces | Gu & Dao 2023 | arXiv:2312.00752 |
| Gauge equivariance necessity | Cohen et al. 2019 | ICML 2019 |
| Geometric deep learning survey | Bronstein et al. 2021 | arXiv:2104.13478 |
| Clifford algebras in deep learning | Ruhe et al. 2023 | arXiv:2209.04934 |
| Polyhedral subgroups | Dechant 2014 | "Clifford algebras" book |

### 12.2 Adversarial Literature Challenges

| Paper | Challenge | Status |
|-------|-----------|--------|
| Cirone et al. 2024 | Both are log-signature projections; geometry may be irrelevant | **Partially resolved**: Dense ablation (#45) shows Spinor > Dense at D≥9 |
| Movahedi et al. 2024 | Wrong geometric constraint can hurt | **Confirmed**: #31 shows rotation hurts on language |
| Olsson et al. 2022 | Real prediction uses non-Markovian induction heads | **Confirmed**: Language requires non-Markovian mechanisms |
| Merrill et al. 2024 | SSM "state" is illusory; both are TC^0 | **Addressed**: Below-the-ceiling analysis within complexity class |

---

## 13. File Organization

### Core Formulations
- **`framework/state_definition.md`** — Full state space definition, Clifford algebra setup, transition kernel, Markov property, scalability options (Sections 1-8)
- **`framework/formalization.md`** — Revised conjecture, scope, empirical evidence, falsification protocol (Sections 1-8)

### Proofs & Analysis
- **`framework/proofs/k1_case_disproof.md`** — k=1 case disproof, norm leakage (Section 5 here)
- **`framework/proofs/epsilon_bound.md`** — Norm concentration bound derivation (Section 5 here)

### Implementation
- **`framework/models/ssm_cells.py`** — QuatBlockSSM, GivensSSM, DiagonalSSM, GatedRotationSSM (Section 3 here)
  - Lines 61-77: QuatBlock recurrence
  - Lines 79-111: Quaternion sandwich product (Rodrigues)
  - Lines 114-157: Givens rotation
  - Lines 164-188: Diagonal SSM
  - Lines 190-308: Gated rotation with learnable blending

### Experiments & Results
- **`reports/findings_geometric_bias.qmd`** — Comprehensive Quarto report with all experimental results (Section 7 here)

### Bibliography
- **`references.bib`** — 33+ papers in BibTeX format

---

## 14. Quick LaTeX Reference for Copy-Paste

### State Variables
```latex
h_t \in \mathbb{R}^D
s_t = h_t / \|h_t\|
\hat{h}_t = h_t / \|h_t\|
x_t \in \mathbb{R}^D
u_t \in \text{Spin}(D)
\lambda_t \in (0,1)
```

### Core Recurrence
```latex
h_{t+1} = \lambda_t \cdot u_t h_t u_t^{-1} + \overline{B}_t x_t
u_t = \exp(B_t / 2)
```

### Comparison Models
```latex
% Diagonal
h_{t+1} = \text{diag}(\lambda_t) \odot h_t + \overline{B}_t x_t

% Givens
h_{t+1} = \lambda_t R(\theta_t) h_t + \overline{B}_t x_t
% where R(\theta) = [[cos θ, -sin θ], [sin θ, cos θ]]

% Quaternion
h_{t+1} = \lambda_t \cdot q_t h_t q_t^{-1} + \overline{b}_t x_t
```

### Norm Evolution
```latex
\|h_{t+1}\|^2 = \lambda_t^2 \|h_t\|^2 + 2\lambda_t \|h_t\| \langle v_t, \overline{B}_t x_t \rangle + \|\overline{B}_t x_t\|^2

r_{t+1} = \log \|h_{t+1}\|
\epsilon(1, D) \leq C \cdot \frac{\text{Var}[\log \|h_t\|]}{D}
```

### Markov Property
```latex
I(s_{t+1}; s_{t-j} \mid s_t) = \epsilon(j, D)
P(h_{t+1} | h_0, \ldots, h_t) = P(h_{t+1} | h_t)  % Holds by construction
```

### Clifford Algebra
```latex
Cl(p, q)
\text{Spin}(p, q)
\mathfrak{spin}(p, q) \cong \mathfrak{so}(p+q)
\rho_u(v) = u v u^{-1}
```

---

**End of Formula Extraction**  
*Last updated: 2026-04-02*  
*Total unique formulas indexed: 80+*  
*Connection map: fully traced from hypothesis → proof → experiments → results*
