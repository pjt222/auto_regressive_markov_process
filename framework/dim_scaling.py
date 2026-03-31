"""
Dimension Scaling Experiment (Issue #38)

Tests whether the prediction advantage of Spinor+Decay SSM over Diagonal SSM
persists (or grows) at higher embedding dimensions D = {3, 8, 16, 32}.

Uses block-diagonal Givens rotations (pairs of dimensions) to keep the spinor
parameterization O(D) — matching the factored Cl approach from
state_definition.md §5 Option B.

Parameter counts per token:
  Spinor+Decay: D/2 rotation angles + 1 decay + D input_scale = 3D/2 + 1
  Diagonal:     D decays + D input_scale = 2D

Usage:
    cd framework && python3 dim_scaling.py
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
# 2. Block-diagonal Givens rotation (general D)
# ============================================================================

def givens_rotate(angles, v):
    """Apply block-diagonal Givens rotations to vector v.

    angles: array of D//2 rotation angles
    v: D-dimensional vector

    Pairs dimensions (0,1), (2,3), ... and applies a 2D rotation to each pair.
    If D is odd, the last dimension is left unchanged.
    """
    result = v.copy()
    dim = len(v)
    n_pairs = dim // 2
    for i in range(n_pairs):
        idx_a = 2 * i
        idx_b = 2 * i + 1
        cos_theta = np.cos(angles[i])
        sin_theta = np.sin(angles[i])
        a_val = result[idx_a]
        b_val = result[idx_b]
        result[idx_a] = cos_theta * a_val - sin_theta * b_val
        result[idx_b] = sin_theta * a_val + cos_theta * b_val
    return result


# ============================================================================
# 3. Trainable Spinor+Decay SSM (general D)
# ============================================================================

class SpinorDecaySSMGeneral:
    """Spinor+Decay SSM using block-diagonal Givens rotations for any D."""

    def __init__(self, dim, seed=0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.n_angles = dim // 2
        # Per token: n_angles rotation + 1 decay + dim input_scale
        self.n_params_per_token = self.n_angles + 1 + dim
        self.n_tokens = 2
        self.params = self.rng.normal(0, 0.3, size=self.n_tokens * self.n_params_per_token)
        # Initialize decay to ~0.8
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base + self.n_angles] = 1.4

    def _unpack(self, params, token_id):
        base = token_id * self.n_params_per_token
        angles = params[base:base + self.n_angles]
        decay_raw = params[base + self.n_angles]
        input_scale = params[base + self.n_angles + 1:base + self.n_params_per_token]
        return angles, decay_raw, input_scale

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
            angles, decay_raw, input_scale = self._unpack(params, x_t)
            rotated = givens_rotate(angles, h_t)
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
# 4. Trainable Diagonal SSM (general D)
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
# 5. Markov score computation (general D)
# ============================================================================

def compute_markov_score_general(states, max_lag=5, n_bins=15):
    """Markov score for arbitrary D states.

    Projects to 3 leading dimensions (by variance) before binning.
    """
    n_steps, dim = states.shape

    # Use first 3 dimensions (or fewer) for binning
    use_dims = min(3, dim)
    projected = states[:, :use_dims]

    # Bin each dimension
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
# 6. (1+lambda)-ES optimizer
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
# 7. Main experiment
# ============================================================================

def run_single_dim(dim, n_train=1000, n_test=1000, n_iters=100, n_seeds=3, max_lag=5):
    """Run training experiment at a single dimension."""
    results = {"spinor": [], "diagonal": []}
    test_losses = {"spinor": [], "diagonal": []}
    train_times = {"spinor": [], "diagonal": []}

    for seed in range(n_seeds):
        log(f"  Seed {seed+1}/{n_seeds}")
        train_seq = generate_markov_source(n_train, order=2, seed=seed)
        test_seq = generate_markov_source(n_test, order=2, seed=seed + 100)
        train_emb = embed_tokens_general(train_seq, dim=dim)
        test_emb = embed_tokens_general(test_seq, dim=dim)

        # Spinor+Decay
        t0 = time.time()
        spinor_model = SpinorDecaySSMGeneral(dim=dim, seed=seed)
        spinor_losses = optimize_model(
            spinor_model, train_seq, train_emb, n_iters=n_iters, seed=seed,
            label=f"D={dim} s{seed} Spinor")
        spinor_time = time.time() - t0
        log(f"    Spinor done: {spinor_losses[0]:.4f} -> {spinor_losses[-1]:.4f} ({spinor_time:.1f}s)")

        # Diagonal
        t0 = time.time()
        diag_model = DiagonalSSMGeneral(dim=dim, seed=seed)
        diag_losses = optimize_model(
            diag_model, train_seq, train_emb, n_iters=n_iters, seed=seed,
            label=f"D={dim} s{seed} Diag")
        diag_time = time.time() - t0
        log(f"    Diagonal done: {diag_losses[0]:.4f} -> {diag_losses[-1]:.4f} ({diag_time:.1f}s)")

        # Evaluate
        spinor_states, _ = spinor_model.run_and_predict(test_seq, test_emb)
        diag_states, _ = diag_model.run_and_predict(test_seq, test_emb)

        spinor_scores = compute_markov_score_general(spinor_states, max_lag=max_lag)
        diag_scores = compute_markov_score_general(diag_states, max_lag=max_lag)

        results["spinor"].append(spinor_scores)
        results["diagonal"].append(diag_scores)
        test_losses["spinor"].append(spinor_model.loss(test_seq, test_emb))
        test_losses["diagonal"].append(diag_model.loss(test_seq, test_emb))
        train_times["spinor"].append(spinor_time)
        train_times["diagonal"].append(diag_time)

    return results, test_losses, train_times


def main():
    log("=" * 70)
    log("Dimension Scaling Experiment (Issue #38)")
    log("Does geometric structure help more at higher D?")
    log("=" * 70)

    dimensions = [3, 8, 16]
    n_train = 1000
    n_test = 1000
    n_iters = 80
    n_seeds = 3
    max_lag = 5

    log(f"\nConfig: dims={dimensions}, n_train={n_train}, n_test={n_test}")
    log(f"        n_iters={n_iters}, n_seeds={n_seeds}")
    log(f"        Rotation: block-diagonal Givens (D/2 angles)")

    all_results = {}

    for dim_idx, dim in enumerate(dimensions):
        spinor_params = 2 * (dim // 2 + 1 + dim)
        diag_params = 2 * (2 * dim)
        log(f"\n{'='*70}")
        log(f"[{dim_idx+1}/{len(dimensions)}] D = {dim}  "
            f"(Spinor: {spinor_params} params, Diagonal: {diag_params} params)")
        log(f"{'='*70}")

        results, test_losses, train_times = run_single_dim(
            dim, n_train=n_train, n_test=n_test, n_iters=n_iters, n_seeds=n_seeds, max_lag=max_lag
        )

        spinor_tl = np.mean(test_losses["spinor"])
        diag_tl = np.mean(test_losses["diagonal"])
        spinor_ms = np.nanmean([np.nanmean(s) for s in results["spinor"]])
        diag_ms = np.nanmean([np.nanmean(s) for s in results["diagonal"]])
        spinor_time = np.mean(train_times["spinor"])
        diag_time = np.mean(train_times["diagonal"])

        log(f"  --- D={dim} SUMMARY ---")
        log(f"  Test loss:    Spinor={spinor_tl:.4f}  Diagonal={diag_tl:.4f}  "
            f"gap={spinor_tl - diag_tl:+.4f}")
        log(f"  Markov score: Spinor={spinor_ms:.4f}  Diagonal={diag_ms:.4f}  "
            f"gap={spinor_ms - diag_ms:+.4f}")
        log(f"  Train time:   Spinor={spinor_time:.1f}s  Diagonal={diag_time:.1f}s")

        all_results[dim] = {
            "spinor_test_loss": spinor_tl,
            "diag_test_loss": diag_tl,
            "spinor_markov": spinor_ms,
            "diag_markov": diag_ms,
            "spinor_test_loss_std": np.std(test_losses["spinor"]),
            "diag_test_loss_std": np.std(test_losses["diagonal"]),
            "spinor_markov_std": np.nanstd([np.nanmean(s) for s in results["spinor"]]),
            "diag_markov_std": np.nanstd([np.nanmean(s) for s in results["diagonal"]]),
        }

    # Summary table
    log("\n" + "=" * 70)
    log("SUMMARY: Prediction Gap and Markov Gap across Dimensions")
    log("=" * 70)
    log(f"{'D':>4} | {'Spinor Loss':>12} | {'Diag Loss':>12} | {'Loss Gap':>10} | "
        f"{'Spinor MS':>10} | {'Diag MS':>10} | {'MS Gap':>10}")
    log("-" * 80)

    for dim in dimensions:
        r = all_results[dim]
        loss_gap = r["spinor_test_loss"] - r["diag_test_loss"]
        ms_gap = r["spinor_markov"] - r["diag_markov"]
        log(f"{dim:>4} | {r['spinor_test_loss']:>12.4f} | {r['diag_test_loss']:>12.4f} | "
            f"{loss_gap:>+10.4f} | {r['spinor_markov']:>10.4f} | {r['diag_markov']:>10.4f} | "
            f"{ms_gap:>+10.4f}")

    log("-" * 80)

    # Analyze trends
    loss_gaps = [all_results[d]["spinor_test_loss"] - all_results[d]["diag_test_loss"] for d in dimensions]
    ms_gaps = [all_results[d]["spinor_markov"] - all_results[d]["diag_markov"] for d in dimensions]

    log("\nTrend analysis:")
    log(f"  Loss gap (Spinor - Diagonal):   {' -> '.join(f'{g:+.4f}' for g in loss_gaps)}")
    log(f"  Markov gap (Spinor - Diagonal): {' -> '.join(f'{g:+.4f}' for g in ms_gaps)}")

    if loss_gaps[-1] < loss_gaps[0]:
        log("\n  >> Loss gap NARROWS with D: spinor advantage shrinks at higher dimensions")
    elif loss_gaps[-1] < 0:
        log("\n  >> Loss gap stays NEGATIVE: spinor advantage persists at higher D")
    else:
        log("\n  >> Loss gap turns POSITIVE at higher D: diagonal catches up")

    if ms_gaps[-1] < ms_gaps[0]:
        log("  >> Markov gap NARROWS with D: spinor becomes relatively more Markovian")
    else:
        log("  >> Markov gap WIDENS with D: spinor carries more history at higher D")

    # Check for crossover
    for i, dim in enumerate(dimensions):
        if loss_gaps[i] < 0 and ms_gaps[i] < 0:
            log(f"\n  ** CROSSOVER at D={dim}: Spinor wins on BOTH prediction AND Markovianity!")
            break


if __name__ == "__main__":
    main()
