"""
Full SO(D) Rotation Scaling Experiment (Issue #38)

Tests whether full so(D) coupling (all D(D-1)/2 bivector parameters via matrix
exponential) provides a larger geometric advantage than block-diagonal Givens
rotations.

This is the upper bound on what geometric structure can do — maximum coupling
between all dimensions.

Parameter counts per token:
  FullSO+Decay: D(D-1)/2 rotation params + 1 decay + D input_scale
  Diagonal:     D decays + D input_scale = 2D

Usage:
    cd framework && python3 full_so_scaling.py
"""

import sys
import numpy as np
import time
from scipy.linalg import expm
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
# 2. Antisymmetric matrix -> SO(D) rotation via matrix exponential
# ============================================================================

def antisym_to_rotation(params, dim):
    """Build SO(D) rotation matrix from D(D-1)/2 parameters.

    Constructs an antisymmetric matrix A where A[i,j] = params[k],
    A[j,i] = -params[k] for each unique pair (i < j), then returns
    R = expm(A) which is guaranteed to be in SO(D).
    """
    antisym_matrix = np.zeros((dim, dim))
    param_idx = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            antisym_matrix[i, j] = params[param_idx]
            antisym_matrix[j, i] = -params[param_idx]
            param_idx += 1
    return expm(antisym_matrix)


# ============================================================================
# 3. Trainable Full SO(D) Spinor+Decay SSM
# ============================================================================

class FullSOSpinorDecaySSM:
    """Spinor+Decay SSM using full SO(D) rotations via matrix exponential.

    The rotation is parameterized by D(D-1)/2 values forming an antisymmetric
    matrix A. The rotation matrix R = expm(A) couples ALL dimension pairs,
    providing maximum geometric expressivity.
    """

    def __init__(self, dim, seed=0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.n_rotation_params = dim * (dim - 1) // 2
        # Per token: n_rotation_params + 1 decay + dim input_scale
        self.n_params_per_token = self.n_rotation_params + 1 + dim
        self.n_tokens = 2
        self.params = self.rng.normal(0, 0.3, size=self.n_tokens * self.n_params_per_token)
        # Initialize decay to ~0.8
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base + self.n_rotation_params] = 1.4

    def _unpack(self, params, token_id):
        base = token_id * self.n_params_per_token
        rotation_params = params[base:base + self.n_rotation_params]
        decay_raw = params[base + self.n_rotation_params]
        input_scale = params[base + self.n_rotation_params + 1:base + self.n_params_per_token]
        return rotation_params, decay_raw, input_scale

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

        # Pre-compute rotation matrices for each token
        rotation_matrices = {}
        for tok in range(self.n_tokens):
            rotation_params, _, _ = self._unpack(params, tok)
            rotation_matrices[tok] = antisym_to_rotation(rotation_params, dim)

        for t in range(n_steps):
            x_t = sequence[t]
            _, decay_raw, input_scale = self._unpack(params, x_t)
            rotated = rotation_matrices[x_t] @ h_t
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
    results = {"full_so": [], "diagonal": []}
    test_losses = {"full_so": [], "diagonal": []}
    train_times = {"full_so": [], "diagonal": []}

    for seed in range(n_seeds):
        log(f"  Seed {seed+1}/{n_seeds}")
        train_seq = generate_markov_source(n_train, order=2, seed=seed)
        test_seq = generate_markov_source(n_test, order=2, seed=seed + 100)
        train_emb = embed_tokens_general(train_seq, dim=dim)
        test_emb = embed_tokens_general(test_seq, dim=dim)

        # Full SO(D) Spinor+Decay
        t0 = time.time()
        full_so_model = FullSOSpinorDecaySSM(dim=dim, seed=seed)
        full_so_losses = optimize_model(
            full_so_model, train_seq, train_emb, n_iters=n_iters, seed=seed,
            label=f"D={dim} s{seed} FullSO")
        full_so_time = time.time() - t0
        log(f"    FullSO done: {full_so_losses[0]:.4f} -> {full_so_losses[-1]:.4f} ({full_so_time:.1f}s)")

        # Diagonal
        t0 = time.time()
        diag_model = DiagonalSSMGeneral(dim=dim, seed=seed)
        diag_losses = optimize_model(
            diag_model, train_seq, train_emb, n_iters=n_iters, seed=seed,
            label=f"D={dim} s{seed} Diag")
        diag_time = time.time() - t0
        log(f"    Diagonal done: {diag_losses[0]:.4f} -> {diag_losses[-1]:.4f} ({diag_time:.1f}s)")

        # Evaluate
        full_so_states, _ = full_so_model.run_and_predict(test_seq, test_emb)
        diag_states, _ = diag_model.run_and_predict(test_seq, test_emb)

        full_so_scores = compute_markov_score_general(full_so_states, max_lag=max_lag)
        diag_scores = compute_markov_score_general(diag_states, max_lag=max_lag)

        results["full_so"].append(full_so_scores)
        results["diagonal"].append(diag_scores)
        test_losses["full_so"].append(full_so_model.loss(test_seq, test_emb))
        test_losses["diagonal"].append(diag_model.loss(test_seq, test_emb))
        train_times["full_so"].append(full_so_time)
        train_times["diagonal"].append(diag_time)

    return results, test_losses, train_times


def main():
    log("=" * 70)
    log("Full SO(D) Rotation Scaling Experiment (Issue #38)")
    log("Upper bound: maximum coupling between all dimensions")
    log("=" * 70)

    dimensions = [3, 8]
    n_train = 1000
    n_test = 1000
    n_iters = 80
    n_seeds = 3
    max_lag = 5

    log(f"\nConfig: dims={dimensions}, n_train={n_train}, n_test={n_test}")
    log(f"        n_iters={n_iters}, n_seeds={n_seeds}")
    log(f"        Rotation: full SO(D) via expm of antisymmetric matrix")

    all_results = {}

    for dim_idx, dim in enumerate(dimensions):
        n_rot = dim * (dim - 1) // 2
        full_so_params = 2 * (n_rot + 1 + dim)
        diag_params = 2 * (2 * dim)
        log(f"\n{'='*70}")
        log(f"[{dim_idx+1}/{len(dimensions)}] D = {dim}  "
            f"(FullSO: {full_so_params} params [{n_rot} rotation], "
            f"Diagonal: {diag_params} params)")
        log(f"{'='*70}")

        results, test_losses, train_times = run_single_dim(
            dim, n_train=n_train, n_test=n_test, n_iters=n_iters, n_seeds=n_seeds, max_lag=max_lag
        )

        full_so_tl = np.mean(test_losses["full_so"])
        diag_tl = np.mean(test_losses["diagonal"])
        full_so_ms = np.nanmean([np.nanmean(s) for s in results["full_so"]])
        diag_ms = np.nanmean([np.nanmean(s) for s in results["diagonal"]])
        full_so_time = np.mean(train_times["full_so"])
        diag_time = np.mean(train_times["diagonal"])

        log(f"  --- D={dim} SUMMARY ---")
        log(f"  Test loss:    FullSO={full_so_tl:.4f}  Diagonal={diag_tl:.4f}  "
            f"gap={full_so_tl - diag_tl:+.4f}")
        log(f"  Markov score: FullSO={full_so_ms:.4f}  Diagonal={diag_ms:.4f}  "
            f"gap={full_so_ms - diag_ms:+.4f}")
        log(f"  Train time:   FullSO={full_so_time:.1f}s  Diagonal={diag_time:.1f}s")

        all_results[dim] = {
            "full_so_test_loss": full_so_tl,
            "diag_test_loss": diag_tl,
            "full_so_markov": full_so_ms,
            "diag_markov": diag_ms,
            "full_so_test_loss_std": np.std(test_losses["full_so"]),
            "diag_test_loss_std": np.std(test_losses["diagonal"]),
            "full_so_markov_std": np.nanstd([np.nanmean(s) for s in results["full_so"]]),
            "diag_markov_std": np.nanstd([np.nanmean(s) for s in results["diagonal"]]),
            "full_so_time": full_so_time,
            "diag_time": diag_time,
        }

    # Summary table
    log("\n" + "=" * 70)
    log("SUMMARY: Full SO(D) vs Diagonal across Dimensions")
    log("=" * 70)
    log(f"{'D':>4} | {'FullSO Loss':>12} | {'Diag Loss':>12} | {'Loss Gap':>10} | "
        f"{'FullSO MS':>10} | {'Diag MS':>10} | {'MS Gap':>10}")
    log("-" * 85)

    for dim in dimensions:
        r = all_results[dim]
        loss_gap = r["full_so_test_loss"] - r["diag_test_loss"]
        ms_gap = r["full_so_markov"] - r["diag_markov"]
        log(f"{dim:>4} | {r['full_so_test_loss']:>12.4f} | {r['diag_test_loss']:>12.4f} | "
            f"{loss_gap:>+10.4f} | {r['full_so_markov']:>10.4f} | {r['diag_markov']:>10.4f} | "
            f"{ms_gap:>+10.4f}")

    log("-" * 85)

    # Parameter count comparison
    log("\nParameter counts:")
    for dim in dimensions:
        n_rot = dim * (dim - 1) // 2
        full_so_total = 2 * (n_rot + 1 + dim)
        diag_total = 2 * (2 * dim)
        log(f"  D={dim}: FullSO={full_so_total} ({n_rot} rot + 1 decay + {dim} scale, x2 tokens)  "
            f"Diagonal={diag_total} ({dim} decay + {dim} scale, x2 tokens)")

    # Analyze trends
    loss_gaps = [all_results[d]["full_so_test_loss"] - all_results[d]["diag_test_loss"] for d in dimensions]
    ms_gaps = [all_results[d]["full_so_markov"] - all_results[d]["diag_markov"] for d in dimensions]

    log("\nTrend analysis:")
    log(f"  Loss gap (FullSO - Diagonal):   {' -> '.join(f'{g:+.4f}' for g in loss_gaps)}")
    log(f"  Markov gap (FullSO - Diagonal): {' -> '.join(f'{g:+.4f}' for g in ms_gaps)}")

    if loss_gaps[-1] < loss_gaps[0]:
        log("\n  >> Loss gap NARROWS with D: full SO advantage shrinks at higher dimensions")
    elif all(g < 0 for g in loss_gaps):
        log("\n  >> Loss gap stays NEGATIVE: full SO advantage persists at higher D")
    else:
        log("\n  >> Loss gap turns POSITIVE at higher D: diagonal catches up")

    if ms_gaps[-1] < ms_gaps[0]:
        log("  >> Markov gap NARROWS with D: full SO becomes relatively more Markovian")
    else:
        log("  >> Markov gap WIDENS with D: full SO carries more history at higher D")

    # Time comparison
    log("\nTime comparison:")
    for dim in dimensions:
        r = all_results[dim]
        ratio = r["full_so_time"] / r["diag_time"] if r["diag_time"] > 0 else float('inf')
        log(f"  D={dim}: FullSO={r['full_so_time']:.1f}s  Diagonal={r['diag_time']:.1f}s  "
            f"(FullSO is {ratio:.1f}x slower)")


if __name__ == "__main__":
    main()
