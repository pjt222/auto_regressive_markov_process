# Epsilon Bound: Norm Variance Controls Non-Markovianity

**Status**: DERIVED (proof sketch) + CALIBRATED (empirical)
**Date**: 2026-04-01
**Issue**: #25
**Depends on**: k=1 disproof (#24)

## 1. Setup

The state evolves as:
$$h_{t+1} = \lambda_t \cdot u_t h_t u_t^{-1} + \bar{B}_t x_t$$

where $\lambda_t \in (0,1)$, $u_t \in \text{Spin}(D)$, and $\bar{B}_t x_t$ is the
input injection. The projected state $s_t = h_t / \|h_t\|$ lives on $S^{D-1}$.

The Markov score is $\epsilon = I(s_{t+1}; s_{t-j} \mid s_t)$, the conditional
mutual information measuring departure from Markovianity.

**Goal**: Show $\epsilon(1, D) \leq C \cdot \text{Var}[\log \|h_t\|] / D$.

## 2. Norm Recurrence

From the k=1 disproof, the squared norm evolves as:
$$\|h_{t+1}\|^2 = \lambda_t^2 \|h_t\|^2 + 2\lambda_t \|h_t\| \langle v_t, \bar{B}_t x_t \rangle + \|\bar{B}_t x_t\|^2$$

where $v_t = u_t s_t u_t^{-1} \in S^{D-1}$.

Define $r_t = \log \|h_t\|$. Then:
$$r_{t+1} = r_t + \frac{1}{2}\log\!\left(\lambda_t^2 + 2\lambda_t e^{-r_t} \langle v_t, \bar{B}_t x_t \rangle + e^{-2r_t}\|\bar{B}_t x_t\|^2\right)$$

## 3. AR(1) Approximation for Log-Norm

**Heuristic step.** For large $r_t$ (large $\|h_t\|$), the $e^{-r_t}$ terms are
small corrections. Expand the logarithm to first order:

$$r_{t+1} \approx r_t + \log\lambda_t + \frac{\langle v_t, \bar{B}_t x_t \rangle}{\lambda_t \|h_t\|} + O(e^{-2r_t})$$

Near the stationary point $r^*$ where $\mathbb{E}[r_{t+1}] = r^*$, define
$\delta_t = r_t - r^*$. Then:

$$\delta_{t+1} \approx \delta_t + \xi_t$$

where $\xi_t = \log\lambda_t - \mathbb{E}[\log\lambda_t] + \lambda_t^{-1} e^{-r^*} \langle v_t, \bar{B}_t x_t \rangle$.

More precisely, this is an AR(1) with coefficient close to 1 but mean-reverting
due to the stationary norm. The stationary variance is:

$$\text{Var}[r_t] \approx \frac{\text{Var}[\xi_t]}{1 - \rho^2}$$

where $\rho \approx 1 - (1 - \lambda^2)/2$ is the AR(1) coefficient.

## 4. Dimension Dependence

The key term is $\langle v_t, \bar{B}_t x_t \rangle$ where $v_t \in S^{D-1}$.

By concentration of measure on $S^{D-1}$: for a fixed vector $w \in \mathbb{R}^D$,
$$\text{Var}[\langle v, w \rangle] = \frac{\|w\|^2}{D}$$

when $v$ is uniformly distributed on $S^{D-1}$.

**Result** (heuristic): The norm driving noise has variance:
$$\text{Var}[\xi_t] = \text{Var}[\log\lambda_t] + \frac{\lambda_t^{-2} e^{-2r^*} \|\bar{B} x\|^2}{D} + \text{cross terms}$$

The $1/D$ factor means $\text{Var}[r_t] = O(1/D)$ when $\lambda_t$ is fixed
(input-independent decay). When $\lambda_t$ varies with input, the
$\text{Var}[\log\lambda_t]$ term dominates and is $O(1)$.

## 5. From Norm Variance to Epsilon

**Claim** (proof sketch):
$$\epsilon(1, D) \leq C \cdot \frac{\text{Var}[\log\|h_t\|]}{D}$$

**Argument**:

1. The non-Markovianity of $s_t$ arises entirely from the hidden norm $\|h_t\|$.
   If $\|h_t\|$ were constant, $s_{t+1}$ would depend only on $(s_t, x_t)$.

2. By data processing inequality:
   $$I(s_{t+1}; s_{t-j} \mid s_t) \leq I(s_{t+1}; r_t \mid s_t)$$
   since all history dependence is mediated through $r_t = \log\|h_t\|$.

3. The influence of $r_t$ on $s_{t+1}$ enters through the mixing weight
   $\lambda_t \|h_t\| / \|h_{t+1}\|$. The sensitivity of $s_{t+1}$ to
   perturbations in $r_t$ is:
   $$\left\|\frac{\partial s_{t+1}}{\partial r_t}\right\| \leq \frac{C_1}{\sqrt{D}}$$

   This $1/\sqrt{D}$ arises because the input injection $\bar{B}_t x_t$ occupies
   a $O(1/\sqrt{D})$ fraction of the full state space in high dimensions.

4. Combining: $I(s_{t+1}; r_t \mid s_t)$ is bounded by the mutual information
   between a variable with variance $\text{Var}[r_t]$ and a function with
   sensitivity $O(1/\sqrt{D})$:
   $$I(s_{t+1}; r_t \mid s_t) \leq \frac{C}{2} \cdot \frac{\text{Var}[r_t]}{D}$$

   using the Gaussian upper bound $I(X; f(X,Z)) \leq \frac{1}{2}\|f'\|^2 \text{Var}[X]$
   and the $1/D$ sensitivity.

**Status**: Steps 3-4 are heuristic. A rigorous proof requires bounding the
sensitivity uniformly over the state space.

## 6. Comparison with Rajaraman

Rajaraman et al. (2024, Thm 4.1) give a tokenization bound:
$$\epsilon_{\text{tok}} = \frac{\log(1/\delta)}{0.99 \log d}$$

where $d$ is the alphabet size and $\delta$ is the approximation error.

| Bound | Decreases with | Nature |
|-------|----------------|--------|
| Ours: $C \cdot \text{Var}[\log\|h\|] / D$ | Dimension $D$ | Continuous state, norm concentration |
| Rajaraman: $\log(1/\delta) / (0.99\log d)$ | Alphabet size $d$ | Discrete state, tokenization |

Both bounds share the structure: non-Markovianity decreases with the
"resolution" of the state space (our $D$, their $d$).

## 7. Empirical Calibration

See `epsilon_calibration.py` for the empirical fit of $C$ across D={3, 9, 15}
and three model types (Spinor+Decay, Diagonal, Dense SSM).

The constant $C$ is fitted by linear regression of $\epsilon$ on
$\text{Var}[\log\|h_t\|] / D$. The R² value indicates goodness of fit.

## References

- k=1 disproof: `proofs/k1_case_disproof.md`
- State definition: `state_definition.md` §7, Open Question 5
- Rajaraman, Jiao & Ramchandran (2024). *Toward a Theory of Tokenization in LLMs.* arXiv:2404.08335
