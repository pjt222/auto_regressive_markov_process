# k=1 Special Case: Disproof

**Status**: DISPROVED
**Date**: 2026-03-31
**Issue**: #24

## Claim (from Conjecture 4.1)

When the source $(x_t)$ is first-order Markov, the projected process $(s_t)$ on
$S^{D-1}$ is exactly Markov, i.e., $\epsilon(1, D) = 0$.

## Result: This Is FALSE

The projected process $(s_t)$ on $S^{D-1}$ is NOT exactly Markov even when the
source is 1st-order Markov, because the norm $\|h_t\|$ acts as a hidden state
that leaks history through the mixing weight at each step.

## Proof Sketch

The recurrence gives:
$$h_{t+1} = \lambda_t \|h_t\| \cdot u_t s_t u_t^{-1} + \overline{B}_t x_t$$

So the projected state:
$$s_{t+1} = \frac{\lambda_t \|h_t\| \cdot v_t + \overline{B}_t x_t}{\|\lambda_t \|h_t\| \cdot v_t + \overline{B}_t x_t\|}$$

where $v_t = u_t s_t u_t^{-1} \in S^{D-1}$. The weight $\lambda_t \|h_t\|$
appears explicitly. Two states $s_t = s_t'$ with different norms
$\|h_t\| \neq \|h_t'\|$ produce different distributions for $s_{t+1}$, even
with the same $x_t$.

The norm evolves as:
$$\|h_{t+1}\|^2 = \lambda_t^2 \|h_t\|^2 + 2\lambda_t \|h_t\| \langle v_t, \overline{B}_t x_t \rangle + \|\overline{B}_t x_t\|^2$$

This accumulates history — it records the magnitude of all previous injections,
geometrically decayed.

## Counterexample (D=1)

$D=1$, $u_t = 1$, $\lambda_t = \lambda \in (0,1)$ fixed, $\overline{B}_t = b > 0$,
$x_t \in \{0, 1\}$ i.i.d. Then $h_{t+1} = \lambda h_t + b x_t$ and
$s_t = \text{sign}(h_t)$.

Two histories producing $s_t = +1$:
- (A) all $x_\tau = 1$: $|h_t^{(A)}|$ large
- (B) alternating $1,0,1,0$: $|h_t^{(B)}|$ small

Given $x_{t+1} = 0$: history (A) keeps $s_{t+1} = +1$ (large positive decays
slowly); history (B) may flip. So $P(s_{t+1} \mid s_t) \neq P(s_{t+1} \mid h_{0:t})$.

## Implications

1. **The correct state is $h_t \in \mathbb{R}^D$, not $s_t \in S^{D-1}$.**
   The raw process $(h_t)$ IS exactly Markov by construction. The conjecture's
   quotient to $S^{D-1}$ discards the norm, which carries history.

2. **The norm is the missing piece.** The pair $(s_t, \|h_t\|)$ is Markov.
   This supports Open Question 7: the norm carries Barbour-like information.

3. **Revised conjecture**: Replace $\epsilon(1, D) = 0$ with:
   $$\epsilon(1, D) \leq C \cdot \text{Var}[\log \|h_t\|] / D$$
   The residual is controlled by the norm's variance, which shrinks near
   stationarity and in high dimensions.

4. **Practical implication**: For the toy example and beyond, the state should
   be $h_t \in \mathbb{R}^D$ (including norm), not the projection $s_t$.
   The quotient is only appropriate after the norm has been shown to concentrate.
