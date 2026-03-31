"""
Quaternion-Block Dimension Scaling Experiment (Issue #38)

Tests whether denser rotation coupling (blocks of 3 via quaternion rotations)
improves the geometric prediction advantage over the Givens (block-2) baseline.

Hypothesis: Givens parameterization (pairs only) is too sparse at higher D.
Quaternion blocks couple 3 dimensions at once and should show a larger or
more stable prediction advantage.

Parameter counts per token:
  QuatBlock Spinor+Decay: (D//3)*3 bivector params + 1 decay + D input_scale
  Diagonal:               D decays + D input_scale = 2D

Usage:
    cd framework && python3 quat_block_scaling.py
"""

import sys
import numpy as np
import time
from toy_example import generate_markov_source


def log(msg):
    """Print with immediate flush so output is visible in piped/redirected runs."""
    print(msg, flush=True)


# ============================================================================
# 1. General-D token embeddings
# ============================================================================

def embed_tokens_general(sequence, dim):
    """Map binary tokens to D-dimensional embeddings.

    Token 0 -> [1, 0, 0, ...], Token 1 -> [0, 1, 0, ...]
    """
    n_steps = len(sequence)
    embeddings = np.zeros((n_steps, dim))
    embeddings[sequence == 0, 0] = 1.0
    embeddings[sequence == 1, 1] = 1.0
    return embeddings


# ============================================================================
# 2. Quaternion operations (from toy_example.py, adapted for block use)
# ============================================================================

def quaternion_from_bivector(alpha, beta, gamma):
    """Construct unit quaternion from bivector components.

    Bivector B = alpha * e12 + beta * e13 + gamma * e23
    Spinor u = exp(B/2) = cos(|B|/2) + sin(|B|/2) * B/|B|

    Returns quaternion [w, x, y, z] where u = w + x*i + y*j + z*k
    """
    norm_b = np.sqrt(alpha**2 + beta**2 + gamma**2)
    if norm_b < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])

    half_angle = norm_b / 2
    cos_ha = np.cos(half_angle)
    sin_ha = np.sin(half_angle)

    w = cos_ha
    x = sin_ha * gamma / norm_b    # e23 component
    y = -sin_ha * beta / norm_b    # e13 component (sign from convention)
    z = sin_ha * alpha / norm_b    # e12 component

    return np.array([w, x, y, z])


def quaternion_rotate(q, v):
    """Apply quaternion rotation to a 3D vector: v' = q * v * q^{-1}.

    For unit quaternion q = [w, x, y, z], this is the sandwich product.
    """
    w, qx, qy, qz = q
    vx, vy, vz = v

    t = 2.0 * np.array([
        qy * vz - qz * vy,
        qz * vx - qx * vz,
        qx * vy - qy * vx,
    ])
    return v + w * t + np.cross(np.array([qx, qy, qz]), t)


# ============================================================================
# 3. Block-diagonal quaternion rotation (general D, blocks of 3)
# ============================================================================

def quat_block_rotate(bivector_params, v):
    """Apply block-diagonal quaternion rotations to vector v.

    bivector_params: array of (D//3)*3 values, grouped as [alpha, beta, gamma]
                     for each block of 3 dimensions.
    v: D-dimensional vector

    Groups dimensions (0,1,2), (3,4,5), ... and applies a quaternion (3D)
    rotation to each block. Remainder dimensions are left unchanged.
    """
    result = v.copy()
    dim = len(v)
    n_blocks = dim // 3

    for block_idx in range(n_blocks):
        param_offset = block_idx * 3
        alpha = bivector_params[param_offset]
        beta = bivector_params[param_offset + 1]
        gamma = bivector_params[param_offset + 2]

        dim_start = block_idx * 3
        block_vec = result[dim_start:dim_start + 3]

        q = quaternion_from_bivector(alpha, beta, gamma)
        rotated_block = quaternion_rotate(q, block_vec)
        result[dim_start:dim_start + 3] = rotated_block

    # Remainder dimensions (if D % 3 != 0) are left as identity
    return result


# ============================================================================
# 4. Trainable QuatBlock Spinor+Decay SSM (general D)
# ============================================================================

class QuatBlockSpinorDecaySSM:
    """Spinor+Decay SSM using block-diagonal quaternion rotations for any D.

    Groups dimensions into blocks of 3, applies quaternion rotation per block.
    Remainder dimensions (D % 3) are left unrotated.

    Parameters per token: (D//3)*3 bivector params + 1 decay + D input_scale
    """

    def __init__(self, dim, seed=0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.n_blocks = dim // 3
        self.n_bivector_params = self.n_blocks * 3
        # Per token: n_bivector_params rotation + 1 decay + dim input_scale
        self.n_params_per_token = self.n_bivector_params + 1 + dim
        self.n_tokens = 2
        self.params = self.rng.normal(0, 0.3, size=self.n_tokens * self.n_params_per_token)
        # Initialize decay to ~0.8
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base + self.n_bivector_params] = 1.4

    def _unpack(self, params, token_id):
        base = token_id * self.n_params_per_token
        bivector_params = params[base:base + self.n_bivector_params]
        decay_raw = params[base + self.n_bivector_params]
        input_scale = params[base + self.n_bivector_params + 1:base + self.n_params_per_token]
        return bivector_params, decay_raw, input_scale

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def run_and_predict(self, sequence, embeddings, params=None):
        if params is None:
            params = self.params
        n_steps = len(sequence)
        dim = self.dim
        states = np.zeros((n_steps, dim))
        h_t = np.zeros(dim)
        h_t[min(2, dim - 1)] = 1.0  # initial state

        for t in range(n_steps):
            x_t = sequence[t]
            bivector_params, decay_raw, input_scale = self._unpack(params, x_t)
            rotated = quat_block_rotate(bivector_params, h_t)
            lambda_t = self._sigmoid(decay_raw)
            h_t = lambda_t * rotated + input_scale * embeddings[t]
            norm = np.linalg.norm(h_t)
            states[t] = h_t / norm if norm > 1e-10 else h_t

        # Readout: use first component as logit
        logits = states[:, 0]
        probs = self._sigmoid(logits * 5.0)
        return states, probs

    def loss(self, sequence, embeddings, params=None):
        _, probs = self.run_and_predict(sequence, embeddings, params)
        targets = sequence[1:]
        pred_probs = probs[:-1]
        eps = 1e-7
        bce = -(targets * np.log(pred_probs + eps) + (1 - targets) * np.log(1 - pred_probs + eps))
        return np.mean(bce)


# ============================================================================
# 5. Trainable Diagonal SSM (general D) -- same as dim_scaling.py
# ============================================================================

class DiagonalSSMGeneral:
    """Diagonal SSM for any D."""

    def __init__(self, dim, seed=0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        # Per token: dim lambda_raw + dim input_scale
        self.n_params_per_token = 2 * dim
        self.n_tokens = 2
        self.params = self.rng.normal(0, 0.3, size=self.n_tokens * self.n_params_per_token)
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base:base + dim] = 1.4

    def _unpack(self, params, token_id):
        base = token_id * self.n_params_per_token
        lambda_raw = params[base:base + self.dim]
        input_scale = params[base + self.dim:base + self.n_params_per_token]
        return lambda_raw, input_scale

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def run_and_predict(self, sequence, embeddings, params=None):
        if params is None:
            params = self.params
        n_steps = len(sequence)
        dim = self.dim
        states = np.zeros((n_steps, dim))
        h_t = np.zeros(dim)
        h_t[min(2, dim - 1)] = 1.0

        for t in range(n_steps):
            x_t = sequence[t]
            lambda_raw, input_scale = self._unpack(params, x_t)
            lambdas = self._sigmoid(lambda_raw)
            h_t = lambdas * h_t + input_scale * embeddings[t]
            norm = np.linalg.norm(h_t)
            states[t] = h_t / norm if norm > 1e-10 else h_t

        logits = states[:, 0]
        probs = self._sigmoid(logits * 5.0)
        return states, probs

    def loss(self, sequence, embeddings, params=None):
        _, probs = self.run_and_predict(sequence, embeddings, params)
        targets = sequence[1:]
        pred_probs = probs[:-1]
        eps = 1e-7
        bce = -(targets * np.log(pred_probs + eps) + (1 - targets) * np.log(1 - pred_probs + eps))
        return np.mean(bce)


# ============================================================================
# 6. Markov score computation (general D)
# ============================================================================

def compute_markov_score_general(states, max_lag=5, n_bins=15):
    """Markov score for arbitrary D states.

    Projects to 3 leading dimensions (by variance) before binning.
    """
    n_steps, dim = states.shape

    use_dims = min(3, dim)
    projected = states[:, :use_dims]

    bin_ids = np.zeros(n_steps, dtype=int)
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
# 7. (1+lambda)-ES optimizer
# ============================================================================

def optimize_model(model, sequence, embeddings, n_iters=100, pop_size=20, sigma=0.1, seed=0, label=""):
    rng = np.random.default_rng(seed)
    best_params = model.params.copy()
    best_loss = model.loss(sequence, embeddings, best_params)
    losses = [best_loss]

    for iteration in range(n_iters):
        noise = rng.normal(0, sigma, size=(pop_size, len(best_params)))
        candidates = best_params[None, :] + noise
        candidate_losses = np.array([model.loss(sequence, embeddings, c) for c in candidates])
        best_idx = np.argmin(candidate_losses)
        if candidate_losses[best_idx] < best_loss:
            best_loss = candidate_losses[best_idx]
            best_params = candidates[best_idx].copy()
        losses.append(best_loss)
        if iteration % 50 == 49:
            sigma *= 0.7
        if iteration % 25 == 24:
            log(f"    {label} iter {iteration+1}/{n_iters}  loss={best_loss:.4f}")

    model.params = best_params
    return losses


# ============================================================================
# 8. Main experiment
# ============================================================================

def run_single_dim(dim, n_train=1000, n_test=1000, n_iters=80, n_seeds=3, max_lag=5):
    """Run training experiment at a single dimension."""
    results = {"quat_spinor": [], "diagonal": []}
    test_losses = {"quat_spinor": [], "diagonal": []}
    train_times = {"quat_spinor": [], "diagonal": []}

    for seed in range(n_seeds):
        log(f"  Seed {seed+1}/{n_seeds}")
        train_seq = generate_markov_source(n_train, order=2, seed=seed)
        test_seq = generate_markov_source(n_test, order=2, seed=seed + 100)
        train_emb = embed_tokens_general(train_seq, dim=dim)
        test_emb = embed_tokens_general(test_seq, dim=dim)

        # QuatBlock Spinor+Decay
        t0 = time.time()
        quat_model = QuatBlockSpinorDecaySSM(dim=dim, seed=seed)
        quat_losses = optimize_model(
            quat_model, train_seq, train_emb, n_iters=n_iters, seed=seed,
            label=f"D={dim} s{seed} QuatSpinor")
        quat_time = time.time() - t0
        log(f"    QuatSpinor done: {quat_losses[0]:.4f} -> {quat_losses[-1]:.4f} ({quat_time:.1f}s)")

        # Diagonal
        t0 = time.time()
        diag_model = DiagonalSSMGeneral(dim=dim, seed=seed)
        diag_losses = optimize_model(
            diag_model, train_seq, train_emb, n_iters=n_iters, seed=seed,
            label=f"D={dim} s{seed} Diag")
        diag_time = time.time() - t0
        log(f"    Diagonal done: {diag_losses[0]:.4f} -> {diag_losses[-1]:.4f} ({diag_time:.1f}s)")

        # Evaluate on test set
        quat_states, _ = quat_model.run_and_predict(test_seq, test_emb)
        diag_states, _ = diag_model.run_and_predict(test_seq, test_emb)

        quat_scores = compute_markov_score_general(quat_states, max_lag=max_lag)
        diag_scores = compute_markov_score_general(diag_states, max_lag=max_lag)

        results["quat_spinor"].append(quat_scores)
        results["diagonal"].append(diag_scores)
        test_losses["quat_spinor"].append(quat_model.loss(test_seq, test_emb))
        test_losses["diagonal"].append(diag_model.loss(test_seq, test_emb))
        train_times["quat_spinor"].append(quat_time)
        train_times["diagonal"].append(diag_time)

    return results, test_losses, train_times


def main():
    log("=" * 70)
    log("Quaternion-Block Dimension Scaling Experiment (Issue #38)")
    log("Does denser rotation coupling (blocks of 3) improve prediction?")
    log("=" * 70)

    dimensions = [3, 9, 15]
    n_train = 1000
    n_test = 1000
    n_iters = 80
    n_seeds = 3
    max_lag = 5

    log(f"\nConfig: dims={dimensions}, n_train={n_train}, n_test={n_test}")
    log(f"        n_iters={n_iters}, n_seeds={n_seeds}")
    log(f"        Rotation: block-diagonal quaternion (D//3 blocks of 3)")

    all_results = {}

    for dim_idx, dim in enumerate(dimensions):
        n_blocks = dim // 3
        n_bivector = n_blocks * 3
        quat_params = 2 * (n_bivector + 1 + dim)
        diag_params = 2 * (2 * dim)
        log(f"\n{'='*70}")
        log(f"[{dim_idx+1}/{len(dimensions)}] D = {dim}  "
            f"(QuatSpinor: {quat_params} params [{n_blocks} blocks], "
            f"Diagonal: {diag_params} params)")
        log(f"{'='*70}")

        results, test_losses, train_times = run_single_dim(
            dim, n_train=n_train, n_test=n_test, n_iters=n_iters, n_seeds=n_seeds, max_lag=max_lag
        )

        quat_tl = np.mean(test_losses["quat_spinor"])
        diag_tl = np.mean(test_losses["diagonal"])
        quat_ms = np.nanmean([np.nanmean(s) for s in results["quat_spinor"]])
        diag_ms = np.nanmean([np.nanmean(s) for s in results["diagonal"]])
        quat_time = np.mean(train_times["quat_spinor"])
        diag_time = np.mean(train_times["diagonal"])

        log(f"  --- D={dim} SUMMARY ---")
        log(f"  Test loss:    QuatSpinor={quat_tl:.4f}  Diagonal={diag_tl:.4f}  "
            f"gap={quat_tl - diag_tl:+.4f}")
        log(f"  Markov score: QuatSpinor={quat_ms:.4f}  Diagonal={diag_ms:.4f}  "
            f"gap={quat_ms - diag_ms:+.4f}")
        log(f"  Train time:   QuatSpinor={quat_time:.1f}s  Diagonal={diag_time:.1f}s")

        all_results[dim] = {
            "quat_test_loss": quat_tl,
            "diag_test_loss": diag_tl,
            "quat_markov": quat_ms,
            "diag_markov": diag_ms,
            "quat_test_loss_std": np.std(test_losses["quat_spinor"]),
            "diag_test_loss_std": np.std(test_losses["diagonal"]),
            "quat_markov_std": np.nanstd([np.nanmean(s) for s in results["quat_spinor"]]),
            "diag_markov_std": np.nanstd([np.nanmean(s) for s in results["diagonal"]]),
        }

    # Summary table
    log("\n" + "=" * 70)
    log("SUMMARY: QuatBlock Spinor vs Diagonal across Dimensions")
    log("=" * 70)
    log(f"{'D':>4} | {'QuatSpinor Loss':>15} | {'Diag Loss':>12} | {'Loss Gap':>10} | "
        f"{'Quat MS':>10} | {'Diag MS':>10} | {'MS Gap':>10}")
    log("-" * 85)

    for dim in dimensions:
        r = all_results[dim]
        loss_gap = r["quat_test_loss"] - r["diag_test_loss"]
        ms_gap = r["quat_markov"] - r["diag_markov"]
        log(f"{dim:>4} | {r['quat_test_loss']:>15.4f} | {r['diag_test_loss']:>12.4f} | "
            f"{loss_gap:>+10.4f} | {r['quat_markov']:>10.4f} | {r['diag_markov']:>10.4f} | "
            f"{ms_gap:>+10.4f}")

    log("-" * 85)

    # Analyze trends
    loss_gaps = [all_results[d]["quat_test_loss"] - all_results[d]["diag_test_loss"] for d in dimensions]
    ms_gaps = [all_results[d]["quat_markov"] - all_results[d]["diag_markov"] for d in dimensions]

    log("\nTrend analysis:")
    log(f"  Loss gap (QuatSpinor - Diagonal):   {' -> '.join(f'{g:+.4f}' for g in loss_gaps)}")
    log(f"  Markov gap (QuatSpinor - Diagonal): {' -> '.join(f'{g:+.4f}' for g in ms_gaps)}")

    if all(g < 0 for g in loss_gaps):
        log("\n  >> Loss gap stays NEGATIVE at all D: quaternion spinor advantage persists")
    elif loss_gaps[-1] < loss_gaps[0]:
        log("\n  >> Loss gap NARROWS with D: quaternion spinor advantage shrinks")
    elif loss_gaps[-1] > 0 and loss_gaps[0] < 0:
        log("\n  >> Loss gap turns POSITIVE at higher D: diagonal catches up")
    else:
        log("\n  >> Loss gap WIDENS with D: quaternion spinor advantage grows")

    if all(g < 0 for g in ms_gaps):
        log("  >> Markov gap stays NEGATIVE: quaternion spinor is more Markovian at all D")
    elif ms_gaps[-1] < ms_gaps[0]:
        log("  >> Markov gap NARROWS with D: quaternion spinor becomes relatively more Markovian")
    else:
        log("  >> Markov gap WIDENS with D: quaternion spinor carries more history at higher D")

    # Compare with Givens results if meaningful
    log("\n" + "=" * 70)
    log("COMPARISON NOTE: Givens (block-2) vs Quaternion (block-3)")
    log("=" * 70)
    log("  Givens (D=3):  loss gap ~ -0.042, Markov gap widening")
    log("  Givens (D=16): loss gap ~ -0.032, advantage narrows")
    log(f"  QuatBlock (D=3):  loss gap = {loss_gaps[0]:+.4f}")
    log(f"  QuatBlock (D=15): loss gap = {loss_gaps[-1]:+.4f}")

    if abs(loss_gaps[-1]) > 0.032:
        log("  >> QuatBlock at D=15 shows LARGER advantage than Givens at D=16")
        log("  >> Denser coupling hypothesis SUPPORTED")
    else:
        log("  >> QuatBlock at D=15 shows SIMILAR or SMALLER advantage than Givens at D=16")
        log("  >> Denser coupling hypothesis NOT clearly supported")


if __name__ == "__main__":
    main()
