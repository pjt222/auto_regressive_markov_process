"""
Discrete Spinor SSM Experiment (Issue #40)

Tests whether restricting the rotation to a finite subgroup of Spin(3)
(binary polyhedral groups) works as well as continuous rotations.

Key idea: Gumbel-softmax discrete quaternion selection
  Instead of learning 3 bivector params -> continuous quaternion,
  learn |G| logits -> Gumbel-softmax -> weighted sum of codebook quaternions.

Four models compared at D=3:
  1. Continuous Spin(3): 3 bivector + 1 decay + 3 input = 7 params/tok
  2. Discrete 2I (120):  120 logits + 1 decay + 3 input = 124 params/tok
  3. Discrete 2T (24):   24 logits + 1 decay + 3 input = 28 params/tok
  4. Diagonal:           3 decay + 3 input = 6 params/tok

Usage:
    cd framework && python3 discrete_spinor.py

Performance: uses numba JIT + parallel candidate evaluation.
"""

import numpy as np
import numba
import time
import sys
import os

# ---------------------------------------------------------------------------
# Import helpers from siblings
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_example import generate_markov_source

# Import group generators from test module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_binary_polyhedral_groups import (
    generate_binary_icosahedral_group,
    generate_binary_tetrahedral_group,
    generate_binary_octahedral_group,
)


def log(msg):
    print(msg, flush=True)


# ============================================================================
# 1. Generate codebooks at module level (computed once)
# ============================================================================

CODEBOOK_2I = generate_binary_icosahedral_group().astype(np.float64)  # (120, 4)
CODEBOOK_2T = generate_binary_tetrahedral_group().astype(np.float64)  # (24, 4)
CODEBOOK_2O = generate_binary_octahedral_group().astype(np.float64)   # (48, 4)

log(f"Codebooks loaded: 2I={CODEBOOK_2I.shape[0]}, "
    f"2T={CODEBOOK_2T.shape[0]}, 2O={CODEBOOK_2O.shape[0]}")


# ============================================================================
# 2. Token embeddings (same as dense_ablation.py)
# ============================================================================

def embed_tokens_general(sequence, dim):
    n_steps = len(sequence)
    embeddings = np.zeros((n_steps, dim))
    embeddings[sequence == 0, 0] = 1.0
    embeddings[sequence == 1, 1] = 1.0
    return embeddings


# ============================================================================
# 3. Numba helpers
# ============================================================================

@numba.njit(cache=True)
def _clipped_sigmoid(x):
    """Numerically stable sigmoid clipped to [-10, 10]."""
    cx = min(max(x, -10.0), 10.0)
    return 1.0 / (1.0 + np.exp(-cx))


@numba.njit(cache=True)
def _quat_rotate_vec(q, v):
    """Apply unit quaternion rotation to a 3D vector via sandwich product.

    q = [w, x, y, z], v = [vx, vy, vz]
    Returns v' = q v q^{-1} using Rodrigues' formula.
    """
    w = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    vx = v[0]
    vy = v[1]
    vz = v[2]

    # t = 2 * (q_vec x v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    # cross(q_vec, t)
    cx = qy * tz - qz * ty
    cy = qz * tx - qx * tz
    cz = qx * ty - qy * tx

    out = np.empty(3)
    out[0] = vx + w * tx + cx
    out[1] = vy + w * ty + cy
    out[2] = vz + w * tz + cz
    return out


@numba.njit(cache=True)
def _quat_from_bivector(alpha, beta, gamma):
    """Construct unit quaternion from bivector components.

    Bivector B = alpha * e12 + beta * e13 + gamma * e23
    Spinor u = exp(B/2) = cos(|B|/2) + sin(|B|/2) * B/|B|

    Returns quaternion [w, x, y, z].
    """
    norm_b = np.sqrt(alpha * alpha + beta * beta + gamma * gamma)
    if norm_b < 1e-10:
        q = np.empty(4)
        q[0] = 1.0
        q[1] = 0.0
        q[2] = 0.0
        q[3] = 0.0
        return q

    half_angle = norm_b / 2.0
    cos_ha = np.cos(half_angle)
    sin_ha = np.sin(half_angle)

    q = np.empty(4)
    q[0] = cos_ha                          # w
    q[1] = sin_ha * gamma / norm_b         # x (e23)
    q[2] = -sin_ha * beta / norm_b         # y (e13, sign convention)
    q[3] = sin_ha * alpha / norm_b         # z (e12)
    return q


# ============================================================================
# 4. Discrete model: batched loss (Gumbel-softmax selection)
# ============================================================================

@numba.njit(parallel=True, cache=True)
def discrete_loss_batch(candidates, sequence, embeddings, codebook,
                        gumbel_noise, tau, n_params_per_token, codebook_size):
    """Evaluate BCE loss for a batch of discrete spinor SSM candidates.

    Parameters
    ----------
    candidates : (n_candidates, n_total_params) float64
        Per-candidate flat parameter vectors.
        Layout per token: [codebook_size logits, 1 decay_raw, 3 input_scales]
    sequence : (n_steps,) int64
    embeddings : (n_steps, 3) float64
    codebook : (codebook_size, 4) float64
        Unit quaternions for the finite group.
    gumbel_noise : (n_candidates, n_steps, codebook_size) float64
        Pre-generated Gumbel(0,1) noise for each candidate and timestep.
    tau : float64
        Gumbel-softmax temperature.
    n_params_per_token : int
        codebook_size + 1 (decay) + 3 (input) = codebook_size + 4
    codebook_size : int
    """
    n_candidates = candidates.shape[0]
    n_steps = sequence.shape[0]
    dim = 3
    losses = np.empty(n_candidates)

    for c in numba.prange(n_candidates):
        params = candidates[c]
        h_t = np.zeros(dim)
        h_t[2] = 1.0  # initial state
        total_loss = 0.0

        for t in range(n_steps):
            tok = sequence[t]
            base = tok * n_params_per_token

            # --- Gumbel-softmax quaternion selection ---
            # logits for this token
            perturbed = np.empty(codebook_size)
            max_val = -1e30
            for i in range(codebook_size):
                perturbed[i] = (params[base + i] + gumbel_noise[c, t, i]) / tau
                if perturbed[i] > max_val:
                    max_val = perturbed[i]

            # Stable softmax
            exp_sum = 0.0
            weights = np.empty(codebook_size)
            for i in range(codebook_size):
                weights[i] = np.exp(perturbed[i] - max_val)
                exp_sum += weights[i]
            for i in range(codebook_size):
                weights[i] /= exp_sum

            # Weighted sum of codebook quaternions
            q = np.zeros(4)
            for i in range(codebook_size):
                w_i = weights[i]
                for j in range(4):
                    q[j] += w_i * codebook[i, j]

            # Normalize to unit quaternion
            q_norm = np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
            if q_norm > 1e-10:
                for j in range(4):
                    q[j] /= q_norm

            # --- Transition ---
            rotated = _quat_rotate_vec(q, h_t)

            decay_raw = params[base + codebook_size]
            lambda_t = _clipped_sigmoid(decay_raw)

            for d in range(dim):
                input_scale = params[base + codebook_size + 1 + d]
                h_t[d] = lambda_t * rotated[d] + input_scale * embeddings[t, d]

            # Normalize state
            norm_sq = 0.0
            for d in range(dim):
                norm_sq += h_t[d] * h_t[d]
            norm = np.sqrt(norm_sq)
            if norm > 1e-10:
                for d in range(dim):
                    h_t[d] /= norm

            # BCE loss: predict next token from current state
            if t < n_steps - 1:
                logit = h_t[0] * 5.0
                prob = _clipped_sigmoid(logit)
                target = float(sequence[t + 1])
                total_loss += -(target * np.log(prob + 1e-7)
                                + (1.0 - target) * np.log(1.0 - prob + 1e-7))

        losses[c] = total_loss / (n_steps - 1)

    return losses


# ============================================================================
# 5. Continuous Spin(3) model: batched loss
# ============================================================================

@numba.njit(parallel=True, cache=True)
def continuous_loss_batch(candidates, sequence, embeddings, n_params_per_token):
    """Evaluate BCE loss for continuous Spin(3) SSM candidates.

    Layout per token: [3 bivector, 1 decay_raw, 3 input_scales] = 7 params
    """
    n_candidates = candidates.shape[0]
    n_steps = sequence.shape[0]
    dim = 3
    losses = np.empty(n_candidates)

    for c in numba.prange(n_candidates):
        params = candidates[c]
        h_t = np.zeros(dim)
        h_t[2] = 1.0
        total_loss = 0.0

        for t in range(n_steps):
            tok = sequence[t]
            base = tok * n_params_per_token

            # Bivector -> quaternion
            alpha = params[base + 0]
            beta = params[base + 1]
            gamma = params[base + 2]
            q = _quat_from_bivector(alpha, beta, gamma)

            # Rotate
            rotated = _quat_rotate_vec(q, h_t)

            # Decay + input
            decay_raw = params[base + 3]
            lambda_t = _clipped_sigmoid(decay_raw)

            for d in range(dim):
                input_scale = params[base + 4 + d]
                h_t[d] = lambda_t * rotated[d] + input_scale * embeddings[t, d]

            # Normalize
            norm_sq = 0.0
            for d in range(dim):
                norm_sq += h_t[d] * h_t[d]
            norm = np.sqrt(norm_sq)
            if norm > 1e-10:
                for d in range(dim):
                    h_t[d] /= norm

            if t < n_steps - 1:
                logit = h_t[0] * 5.0
                prob = _clipped_sigmoid(logit)
                target = float(sequence[t + 1])
                total_loss += -(target * np.log(prob + 1e-7)
                                + (1.0 - target) * np.log(1.0 - prob + 1e-7))

        losses[c] = total_loss / (n_steps - 1)

    return losses


# ============================================================================
# 6. Diagonal model: batched loss
# ============================================================================

@numba.njit(parallel=True, cache=True)
def diagonal_loss_batch(candidates, sequence, embeddings, n_params_per_token):
    """Evaluate BCE loss for diagonal SSM candidates.

    Layout per token: [3 decay_raw, 3 input_scales] = 6 params
    """
    n_candidates = candidates.shape[0]
    n_steps = sequence.shape[0]
    dim = 3
    losses = np.empty(n_candidates)

    for c in numba.prange(n_candidates):
        params = candidates[c]
        h_t = np.zeros(dim)
        h_t[2] = 1.0
        total_loss = 0.0

        for t in range(n_steps):
            tok = sequence[t]
            base = tok * n_params_per_token
            for d in range(dim):
                lam = _clipped_sigmoid(params[base + d])
                inp = params[base + dim + d]
                h_t[d] = lam * h_t[d] + inp * embeddings[t, d]

            norm_sq = 0.0
            for d in range(dim):
                norm_sq += h_t[d] * h_t[d]
            norm = np.sqrt(norm_sq)
            if norm > 1e-10:
                for d in range(dim):
                    h_t[d] /= norm

            if t < n_steps - 1:
                logit = h_t[0] * 5.0
                prob = _clipped_sigmoid(logit)
                target = float(sequence[t + 1])
                total_loss += -(target * np.log(prob + 1e-7)
                                + (1.0 - target) * np.log(1.0 - prob + 1e-7))

        losses[c] = total_loss / (n_steps - 1)

    return losses


# ============================================================================
# 7. Forward functions (single candidate, returns states for Markov score)
# ============================================================================

@numba.njit(cache=True)
def discrete_forward(params, sequence, embeddings, codebook,
                     n_params_per_token, codebook_size):
    """Run discrete spinor SSM forward pass (argmax selection at inference).

    Returns (states, selected_indices) for analysis.
    """
    n_steps = sequence.shape[0]
    dim = 3
    states = np.zeros((n_steps, dim))
    selected = np.zeros(n_steps, dtype=numba.int64)
    h_t = np.zeros(dim)
    h_t[2] = 1.0

    for t in range(n_steps):
        tok = sequence[t]
        base = tok * n_params_per_token

        # Argmax selection (no Gumbel noise at inference)
        best_idx = 0
        best_val = params[base + 0]
        for i in range(1, codebook_size):
            if params[base + i] > best_val:
                best_val = params[base + i]
                best_idx = i
        selected[t] = best_idx

        q = np.empty(4)
        for j in range(4):
            q[j] = codebook[best_idx, j]

        rotated = _quat_rotate_vec(q, h_t)

        decay_raw = params[base + codebook_size]
        lambda_t = _clipped_sigmoid(decay_raw)

        for d in range(dim):
            input_scale = params[base + codebook_size + 1 + d]
            h_t[d] = lambda_t * rotated[d] + input_scale * embeddings[t, d]

        norm_sq = 0.0
        for d in range(dim):
            norm_sq += h_t[d] * h_t[d]
        norm = np.sqrt(norm_sq)
        if norm > 1e-10:
            for d in range(dim):
                h_t[d] /= norm

        for d in range(dim):
            states[t, d] = h_t[d]

    return states, selected


@numba.njit(cache=True)
def continuous_forward(params, sequence, embeddings, n_params_per_token):
    """Run continuous Spin(3) SSM forward pass, return normalized states."""
    n_steps = sequence.shape[0]
    dim = 3
    states = np.zeros((n_steps, dim))
    h_t = np.zeros(dim)
    h_t[2] = 1.0

    for t in range(n_steps):
        tok = sequence[t]
        base = tok * n_params_per_token

        alpha = params[base + 0]
        beta = params[base + 1]
        gamma = params[base + 2]
        q = _quat_from_bivector(alpha, beta, gamma)

        rotated = _quat_rotate_vec(q, h_t)

        decay_raw = params[base + 3]
        lambda_t = _clipped_sigmoid(decay_raw)

        for d in range(dim):
            input_scale = params[base + 4 + d]
            h_t[d] = lambda_t * rotated[d] + input_scale * embeddings[t, d]

        norm_sq = 0.0
        for d in range(dim):
            norm_sq += h_t[d] * h_t[d]
        norm = np.sqrt(norm_sq)
        if norm > 1e-10:
            for d in range(dim):
                h_t[d] /= norm

        for d in range(dim):
            states[t, d] = h_t[d]

    return states


@numba.njit(cache=True)
def diagonal_forward(params, sequence, embeddings, n_params_per_token):
    """Run diagonal SSM forward pass, return normalized states."""
    n_steps = sequence.shape[0]
    dim = 3
    states = np.zeros((n_steps, dim))
    h_t = np.zeros(dim)
    h_t[2] = 1.0

    for t in range(n_steps):
        tok = sequence[t]
        base = tok * n_params_per_token
        for d in range(dim):
            lam = _clipped_sigmoid(params[base + d])
            inp = params[base + dim + d]
            h_t[d] = lam * h_t[d] + inp * embeddings[t, d]

        norm_sq = 0.0
        for d in range(dim):
            norm_sq += h_t[d] * h_t[d]
        norm = np.sqrt(norm_sq)
        if norm > 1e-10:
            for d in range(dim):
                h_t[d] /= norm

        for d in range(dim):
            states[t, d] = h_t[d]

    return states


# ============================================================================
# 8. Model wrapper classes
# ============================================================================

class DiscreteSpinorSSM:
    """Discrete spinor SSM using Gumbel-softmax over a finite group codebook."""

    def __init__(self, codebook, codebook_name, seed=0):
        self.dim = 3
        self.codebook = codebook.copy()
        self.codebook_name = codebook_name
        self.codebook_size = codebook.shape[0]
        self.n_params_per_token = self.codebook_size + 1 + self.dim  # logits + decay + input
        self.n_tokens = 2
        n_total = self.n_tokens * self.n_params_per_token

        rng = np.random.default_rng(seed)
        self.params = rng.normal(0, 0.1, size=n_total)  # small init for logits
        # Initialize decay to high (sigmoid(1.4) ~ 0.8)
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base + self.codebook_size] = 1.4

    def loss_batch(self, candidates, sequence, embeddings, gumbel_noise, tau):
        return discrete_loss_batch(
            candidates, sequence, embeddings,
            self.codebook, gumbel_noise, tau,
            self.n_params_per_token, self.codebook_size)

    def forward(self, sequence, embeddings):
        states, selected = discrete_forward(
            self.params, sequence, embeddings,
            self.codebook, self.n_params_per_token, self.codebook_size)
        return states, selected

    def total_params(self):
        return self.n_tokens * self.n_params_per_token


class ContinuousSpinorSSM:
    """Continuous Spin(3) SSM using bivector -> quaternion."""

    def __init__(self, seed=0):
        self.dim = 3
        self.n_params_per_token = 3 + 1 + 3  # bivector + decay + input = 7
        self.n_tokens = 2
        n_total = self.n_tokens * self.n_params_per_token

        rng = np.random.default_rng(seed)
        self.params = rng.normal(0, 0.3, size=n_total)
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base + 3] = 1.4  # decay init

    def loss_batch(self, candidates, sequence, embeddings):
        return continuous_loss_batch(
            candidates, sequence, embeddings,
            self.n_params_per_token)

    def forward(self, sequence, embeddings):
        return continuous_forward(
            self.params, sequence, embeddings,
            self.n_params_per_token)

    def total_params(self):
        return self.n_tokens * self.n_params_per_token


class DiagonalSSM:
    """Diagonal SSM baseline (no rotation)."""

    def __init__(self, seed=0):
        self.dim = 3
        self.n_params_per_token = 2 * 3  # decay + input = 6
        self.n_tokens = 2
        n_total = self.n_tokens * self.n_params_per_token

        rng = np.random.default_rng(seed)
        self.params = rng.normal(0, 0.3, size=n_total)
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base:base + 3] = 1.4  # decay init

    def loss_batch(self, candidates, sequence, embeddings):
        return diagonal_loss_batch(
            candidates, sequence, embeddings,
            self.n_params_per_token)

    def forward(self, sequence, embeddings):
        return diagonal_forward(
            self.params, sequence, embeddings,
            self.n_params_per_token)

    def total_params(self):
        return self.n_tokens * self.n_params_per_token


# ============================================================================
# 9. Optimizers
# ============================================================================

def optimize_discrete(model, sequence, embeddings, n_iters=150, pop_size=30,
                      sigma=0.1, seed=0, label=""):
    """(1+lambda)-ES optimizer with Gumbel noise generation + tau annealing."""
    rng = np.random.default_rng(seed)
    n_params = len(model.params)
    effective_pop = max(pop_size, 4 + int(3 * np.log(n_params + 1)))
    n_steps = len(sequence)
    codebook_size = model.codebook_size

    best_params = model.params.copy()

    # Initial loss (tau=1.0, single candidate)
    gumbel_init = _generate_gumbel(rng, 1, n_steps, codebook_size)
    init_losses = model.loss_batch(
        best_params.reshape(1, -1), sequence, embeddings, gumbel_init, 1.0)
    best_loss = init_losses[0]
    losses = [best_loss]

    for iteration in range(n_iters):
        tau = max(0.1, 1.0 * (0.99 ** iteration))

        noise = rng.normal(0, sigma, size=(effective_pop, n_params))
        candidates = best_params[None, :] + noise

        # Pre-generate Gumbel noise for all candidates and timesteps
        gumbel = _generate_gumbel(rng, effective_pop, n_steps, codebook_size)

        candidate_losses = model.loss_batch(
            candidates, sequence, embeddings, gumbel, tau)

        best_idx = np.argmin(candidate_losses)
        if candidate_losses[best_idx] < best_loss:
            best_loss = candidate_losses[best_idx]
            best_params = candidates[best_idx].copy()
        losses.append(best_loss)

        if iteration % 50 == 49:
            sigma *= 0.7
            log(f"    {label} iter {iteration+1}/{n_iters}  "
                f"loss={best_loss:.4f}  tau={tau:.3f}")

    model.params = best_params
    return losses


def optimize_standard(model, sequence, embeddings, n_iters=150, pop_size=30,
                      sigma=0.1, seed=0, label=""):
    """(1+lambda)-ES optimizer for continuous/diagonal models (no Gumbel)."""
    rng = np.random.default_rng(seed)
    n_params = len(model.params)
    effective_pop = max(pop_size, 4 + int(3 * np.log(n_params + 1)))

    best_params = model.params.copy()
    init_losses = model.loss_batch(
        best_params.reshape(1, -1), sequence, embeddings)
    best_loss = init_losses[0]
    losses = [best_loss]

    for iteration in range(n_iters):
        noise = rng.normal(0, sigma, size=(effective_pop, n_params))
        candidates = best_params[None, :] + noise

        candidate_losses = model.loss_batch(candidates, sequence, embeddings)

        best_idx = np.argmin(candidate_losses)
        if candidate_losses[best_idx] < best_loss:
            best_loss = candidate_losses[best_idx]
            best_params = candidates[best_idx].copy()
        losses.append(best_loss)

        if iteration % 50 == 49:
            sigma *= 0.7
            log(f"    {label} iter {iteration+1}/{n_iters}  loss={best_loss:.4f}")

    model.params = best_params
    return losses


def _generate_gumbel(rng, n_candidates, n_steps, codebook_size):
    """Generate Gumbel(0,1) noise: -log(-log(U)), U ~ Uniform(0.001, 0.999)."""
    uniform = rng.uniform(0.001, 0.999, size=(n_candidates, n_steps, codebook_size))
    return -np.log(-np.log(uniform))


# ============================================================================
# 10. Markov score computation (reused from dense_ablation pattern)
# ============================================================================

def compute_markov_score(states, max_lag=5, n_bins=15):
    n_steps, dim = states.shape
    use_dims = min(3, dim)
    projected = states[:, :use_dims]

    bin_ids = np.zeros(n_steps, dtype=np.int64)
    multiplier = 1
    for d_idx in range(use_dims):
        col = projected[:, d_idx]
        edges = np.linspace(col.min() - 1e-8, col.max() + 1e-8, n_bins + 1)
        bins = np.digitize(col, edges) - 1
        bins = np.clip(bins, 0, n_bins - 1)
        bin_ids += bins * multiplier
        multiplier *= n_bins

    scores = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        valid_n = n_steps - lag - 1
        if valid_n < 100:
            scores[lag - 1] = np.nan
            continue

        x_arr = bin_ids[lag + 1:n_steps]
        y_arr = bin_ids[1:n_steps - lag]
        z_arr = bin_ids[lag:n_steps - 1]

        h_xz = _joint_entropy(x_arr, z_arr)
        h_yz = _joint_entropy(y_arr, z_arr)
        h_xyz = _triple_entropy(x_arr, y_arr, z_arr)
        h_z = _entropy(z_arr)

        cmi = h_xz + h_yz - h_xyz - h_z
        scores[lag - 1] = max(0, cmi)

    return scores


def _entropy(x):
    _, counts = np.unique(x, return_counts=True)
    probs = counts / len(x)
    return -np.sum(probs * np.log(probs + 1e-12))


def _joint_entropy(x, y):
    pairs = np.column_stack([x, y])
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    probs = counts / len(x)
    return -np.sum(probs * np.log(probs + 1e-12))


def _triple_entropy(x, y, z):
    triples = np.column_stack([x, y, z])
    _, counts = np.unique(triples, axis=0, return_counts=True)
    probs = counts / len(x)
    return -np.sum(probs * np.log(probs + 1e-12))


# ============================================================================
# 11. JIT warm-up
# ============================================================================

def warmup_jit():
    """Compile all numba functions with tiny inputs."""
    log("Warming up numba JIT...")
    seq = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    emb = embed_tokens_general(seq, 3).astype(np.float64)

    # Discrete (use 2T for faster warm-up since smallest codebook)
    cb_tiny = CODEBOOK_2T
    n_ppt_disc = cb_tiny.shape[0] + 1 + 3
    n_total_disc = 2 * n_ppt_disc
    rng = np.random.default_rng(99)
    params_disc = rng.normal(0, 0.1, size=n_total_disc)
    gumbel_tiny = _generate_gumbel(rng, 1, len(seq), cb_tiny.shape[0])
    discrete_loss_batch(
        params_disc.reshape(1, -1), seq, emb,
        cb_tiny, gumbel_tiny, 1.0, n_ppt_disc, cb_tiny.shape[0])
    discrete_forward(params_disc, seq, emb, cb_tiny, n_ppt_disc, cb_tiny.shape[0])

    # Continuous
    n_ppt_cont = 7
    params_cont = rng.normal(0, 0.3, size=2 * n_ppt_cont)
    continuous_loss_batch(params_cont.reshape(1, -1), seq, emb, n_ppt_cont)
    continuous_forward(params_cont, seq, emb, n_ppt_cont)

    # Diagonal
    n_ppt_diag = 6
    params_diag = rng.normal(0, 0.3, size=2 * n_ppt_diag)
    diagonal_loss_batch(params_diag.reshape(1, -1), seq, emb, n_ppt_diag)
    diagonal_forward(params_diag, seq, emb, n_ppt_diag)

    log("JIT warm-up complete.")


# ============================================================================
# 12. Codebook selection analysis
# ============================================================================

def analyze_codebook_selection(model, test_seq, test_emb):
    """Analyze which codebook elements are selected at inference (argmax)."""
    states, selected = model.forward(test_seq, test_emb)
    codebook_size = model.codebook_size

    # Count selections per token type
    for tok in range(2):
        mask = test_seq == tok
        sel_tok = selected[mask]
        unique_vals, counts = np.unique(sel_tok, return_counts=True)
        top_k = min(5, len(unique_vals))
        sort_idx = np.argsort(-counts)[:top_k]

        log(f"    Token {tok}: {len(unique_vals)}/{codebook_size} "
            f"distinct elements used")
        for rank, idx in enumerate(sort_idx):
            elem_idx = unique_vals[idx]
            q = model.codebook[elem_idx]
            log(f"      #{rank+1}: elem {elem_idx:>3d} "
                f"(q=[{q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f}]) "
                f"count={counts[idx]}")

    return states, selected


# ============================================================================
# 13. Main experiment
# ============================================================================

def run_experiment(n_train=2000, n_test=2000, n_iters=150, n_seeds=3,
                   max_lag=5):
    """Run all 4 models (Continuous, 2I, 2T, Diagonal) at D=3."""
    model_specs = [
        ("Continuous", None, None),
        ("Discrete_2I", CODEBOOK_2I, "2I"),
        ("Discrete_2T", CODEBOOK_2T, "2T"),
        ("Diagonal", None, None),
    ]

    results = {name: [] for name, _, _ in model_specs}
    test_losses = {name: [] for name, _, _ in model_specs}
    train_times = {name: [] for name, _, _ in model_specs}
    param_counts = {}

    for seed in range(n_seeds):
        log(f"\n  Seed {seed+1}/{n_seeds}")
        train_seq = generate_markov_source(n_train, order=2, seed=seed).astype(np.int64)
        test_seq = generate_markov_source(n_test, order=2, seed=seed + 100).astype(np.int64)
        train_emb = embed_tokens_general(train_seq, 3)
        test_emb = embed_tokens_general(test_seq, 3)

        for model_name, codebook, cb_name in model_specs:
            if model_name == "Continuous":
                model = ContinuousSpinorSSM(seed=seed)
            elif model_name == "Diagonal":
                model = DiagonalSSM(seed=seed)
            else:
                model = DiscreteSpinorSSM(codebook, cb_name, seed=seed)

            if seed == 0:
                param_counts[model_name] = model.total_params()

            t0 = time.time()
            if isinstance(model, DiscreteSpinorSSM):
                loss_history = optimize_discrete(
                    model, train_seq, train_emb,
                    n_iters=n_iters, seed=seed,
                    label=f"s{seed} {model_name[:10]:>10}")
            else:
                loss_history = optimize_standard(
                    model, train_seq, train_emb,
                    n_iters=n_iters, seed=seed,
                    label=f"s{seed} {model_name[:10]:>10}")
            elapsed = time.time() - t0
            log(f"    {model_name:>12}: "
                f"{loss_history[0]:.4f} -> {loss_history[-1]:.4f} ({elapsed:.1f}s)")

            # Forward pass for Markov score
            if isinstance(model, DiscreteSpinorSSM):
                states, _ = model.forward(test_seq, test_emb)
            else:
                states = model.forward(test_seq, test_emb)

            scores = compute_markov_score(states, max_lag=max_lag)

            # Test loss
            if isinstance(model, DiscreteSpinorSSM):
                # Use low temperature + fresh Gumbel noise for test
                rng_test = np.random.default_rng(seed + 9999)
                gumbel_test = _generate_gumbel(
                    rng_test, 1, len(test_seq), model.codebook_size)
                test_loss = model.loss_batch(
                    model.params.reshape(1, -1),
                    test_seq, test_emb, gumbel_test, 0.1)[0]
            else:
                test_loss = model.loss_batch(
                    model.params.reshape(1, -1), test_seq, test_emb)[0]

            results[model_name].append(scores)
            test_losses[model_name].append(test_loss)
            train_times[model_name].append(elapsed)

    return results, test_losses, train_times, param_counts, model_specs


def main():
    log("=" * 78)
    log("Discrete Spinor SSM Experiment (Issue #40)")
    log("Does restricting to a finite subgroup of Spin(3) preserve")
    log("the prediction advantage of geometric rotations?")
    log("=" * 78)

    n_train = 2000
    n_test = 2000
    n_iters = 150
    n_seeds = 3
    max_lag = 5

    warmup_jit()

    log(f"\nConfig: D=3, n_train={n_train}, n_test={n_test}")
    log(f"        n_iters={n_iters}, n_seeds={n_seeds}, source_order=2")
    log(f"        tau: 1.0 -> 0.1 (annealed, decay=0.99)")
    log(f"        optimizer=(1+lambda)-ES, parallel numba")

    results, test_losses, train_times, param_counts, model_specs = \
        run_experiment(n_train=n_train, n_test=n_test,
                       n_iters=n_iters, n_seeds=n_seeds, max_lag=max_lag)

    # ========================================================================
    # Summary table
    # ========================================================================
    model_names = [name for name, _, _ in model_specs]

    log(f"\n{'='*78}")
    log("PARAMETER COUNTS")
    log(f"{'='*78}")
    for name in model_names:
        count = param_counts[name]
        per_tok = count // 2
        log(f"  {name:>12}: {count:>6} total  ({per_tok} per token)")

    log(f"\n{'='*78}")
    log("SUMMARY TABLE")
    log(f"{'='*78}")
    log(f"  {'Model':>12} | {'Codebook':>8} | {'Params':>6} | "
        f"{'Test Loss':>16} | {'Markov Score':>16}")
    log(f"  {'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*16}-+-{'-'*16}")

    summary = {}
    for name in model_names:
        tl_mean = np.mean(test_losses[name])
        tl_std = np.std(test_losses[name])
        ms_mean = np.nanmean([np.nanmean(s) for s in results[name]])
        ms_std = np.nanstd([np.nanmean(s) for s in results[name]])
        tt_mean = np.mean(train_times[name])

        if "2I" in name:
            cb_label = "120"
        elif "2T" in name:
            cb_label = "24"
        elif "2O" in name:
            cb_label = "48"
        elif name == "Continuous":
            cb_label = "inf"
        else:
            cb_label = "n/a"

        summary[name] = {
            "test_loss": tl_mean,
            "test_loss_std": tl_std,
            "markov_score": ms_mean,
            "markov_score_std": ms_std,
            "train_time": tt_mean,
            "params": param_counts[name],
            "cb_label": cb_label,
        }

        log(f"  {name:>12} | {cb_label:>8} | {param_counts[name]:>6} | "
            f"{tl_mean:.4f} +/- {tl_std:.4f} | "
            f"{ms_mean:.4f} +/- {ms_std:.4f}")

    # ========================================================================
    # Codebook selection analysis (last seed)
    # ========================================================================
    log(f"\n{'='*78}")
    log("CODEBOOK SELECTION ANALYSIS (last seed, test set)")
    log(f"{'='*78}")

    last_seed = n_seeds - 1
    test_seq = generate_markov_source(n_test, order=2, seed=last_seed + 100).astype(np.int64)
    test_emb = embed_tokens_general(test_seq, 3)

    for model_name, codebook, cb_name in model_specs:
        if codebook is not None:
            log(f"\n  {model_name} ({cb_name}, |G|={codebook.shape[0]}):")
            model = DiscreteSpinorSSM(codebook, cb_name, seed=last_seed)
            # Re-optimize to get final params (or we could store them -- but
            # for simplicity, re-train with the last seed)
            train_seq = generate_markov_source(
                n_train, order=2, seed=last_seed).astype(np.int64)
            train_emb = embed_tokens_general(train_seq, 3)
            optimize_discrete(model, train_seq, train_emb,
                              n_iters=n_iters, seed=last_seed,
                              label=f"analysis {model_name[:8]}")
            analyze_codebook_selection(model, test_seq, test_emb)

    # ========================================================================
    # Comparison: discrete vs continuous
    # ========================================================================
    log(f"\n{'='*78}")
    log("COMPARISON: DISCRETE vs CONTINUOUS")
    log(f"{'='*78}")

    cont_loss = summary["Continuous"]["test_loss"]
    diag_loss = summary["Diagonal"]["test_loss"]

    for name in model_names:
        s = summary[name]
        gap_vs_diag = s["test_loss"] - diag_loss
        gap_vs_cont = s["test_loss"] - cont_loss
        log(f"  {name:>12}: loss={s['test_loss']:.4f}  "
            f"vs_Diag={gap_vs_diag:+.4f}  vs_Cont={gap_vs_cont:+.4f}  "
            f"params={s['params']}")

    # Decision
    log(f"\n{'='*78}")
    log("DECISION")
    log(f"{'='*78}")

    cont = summary["Continuous"]["test_loss"]
    d2i = summary["Discrete_2I"]["test_loss"]
    d2t = summary["Discrete_2T"]["test_loss"]
    diag = summary["Diagonal"]["test_loss"]

    cont_std = summary["Continuous"]["test_loss_std"]
    d2i_std = summary["Discrete_2I"]["test_loss_std"]
    d2t_std = summary["Discrete_2T"]["test_loss_std"]
    diag_std = summary["Diagonal"]["test_loss_std"]

    # Is 2I close to continuous?
    overlap_2i_cont = abs(d2i - cont) < (d2i_std + cont_std)
    # Is 2T close to continuous?
    overlap_2t_cont = abs(d2t - cont) < (d2t_std + cont_std)
    # Do discrete models beat diagonal?
    d2i_beats_diag = d2i < diag
    d2t_beats_diag = d2t < diag

    log(f"  Continuous Spin(3): {cont:.4f} +/- {cont_std:.4f}")
    log(f"  Discrete 2I (120): {d2i:.4f} +/- {d2i_std:.4f}")
    log(f"  Discrete 2T (24):  {d2t:.4f} +/- {d2t_std:.4f}")
    log(f"  Diagonal:          {diag:.4f} +/- {diag_std:.4f}")
    log("")

    if d2i_beats_diag and overlap_2i_cont:
        log("  2I (120 elements): MATCHES continuous -- "
            "finite subgroup is sufficient")
    elif d2i_beats_diag:
        log("  2I (120 elements): beats Diagonal but differs from Continuous")
    else:
        log("  2I (120 elements): does NOT beat Diagonal -- "
            "discretization hurts")

    if d2t_beats_diag and overlap_2t_cont:
        log("  2T (24 elements):  MATCHES continuous -- "
            "even small subgroup works")
    elif d2t_beats_diag:
        log("  2T (24 elements):  beats Diagonal but differs from Continuous")
    else:
        log("  2T (24 elements):  does NOT beat Diagonal -- "
            "too few rotations?")

    # Parameter efficiency comparison
    log(f"\n  Parameter efficiency:")
    log(f"    Continuous:  {param_counts['Continuous']:>6} params, "
        f"loss={cont:.4f}")
    log(f"    Discrete 2I: {param_counts['Discrete_2I']:>6} params, "
        f"loss={d2i:.4f} "
        f"({param_counts['Discrete_2I']/param_counts['Continuous']:.1f}x more params)")
    log(f"    Discrete 2T: {param_counts['Discrete_2T']:>6} params, "
        f"loss={d2t:.4f} "
        f"({param_counts['Discrete_2T']/param_counts['Continuous']:.1f}x more params)")
    log(f"    Diagonal:    {param_counts['Diagonal']:>6} params, "
        f"loss={diag:.4f}")

    log(f"\n{'='*78}")
    log("DONE")
    log(f"{'='*78}")


if __name__ == "__main__":
    main()
