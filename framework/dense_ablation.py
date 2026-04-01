"""
Generic Dense SSM Ablation (Issue #45 — Cirone Challenge)

Tests whether the Spinor+Decay prediction advantage is attributable to
Spin(D) geometry specifically, or merely to non-diagonality of the
transition matrix.

Three models compared at D={3, 9, 15}:
  1. Diagonal SSM:     h_{t+1} = diag(λ_t) ⊙ h_t + B̄_t x_t
  2. Spinor+Decay SSM: h_{t+1} = λ_t · R(θ_t) h_t + B̄_t x_t   (Givens rotation)
  3. Dense SSM:        h_{t+1} = A_t h_t + B̄_t x_t              (unconstrained D×D)

Decision table (from issue #45):
  Spinor > Dense > Diagonal  →  geometry matters beyond non-diagonality ✓
  Spinor ≈ Dense > Diagonal  →  advantage is non-diagonality, not geometry ✗
  Dense > Spinor > Diagonal  →  geometry constrains too much ✗
  Dense ≈ Diagonal           →  non-diagonality doesn't help (data too simple)

Parameter counts per token (natural parameterization):
  Diagonal:     D (decay) + D (input) = 2D
  Spinor+Decay: D//2 (angles) + 1 (decay) + D (input) = 3D/2 + 1
  Dense:        D² (matrix) + D (input) = D² + D

The parameter mismatch is intentional: if Spinor (fewer params) matches or
beats Dense (more params), the geometric constraint is a strong inductive bias.

Usage:
    cd framework && python3 dense_ablation.py

Performance: uses numba JIT + parallel candidate evaluation (~50-100x vs pure Python).
"""

import numpy as np
import numba
import time
import sys
from toy_example import generate_markov_source


def log(msg):
    print(msg, flush=True)


# ============================================================================
# 1. General-D token embeddings
# ============================================================================

def embed_tokens_general(sequence, dim):
    n_steps = len(sequence)
    embeddings = np.zeros((n_steps, dim))
    embeddings[sequence == 0, 0] = 1.0
    embeddings[sequence == 1, 1] = 1.0
    return embeddings


# ============================================================================
# 2. Numba-accelerated forward passes (batched over candidates)
# ============================================================================

@numba.njit(cache=True)
def _clipped_sigmoid(x):
    cx = min(max(x, -10.0), 10.0)
    return 1.0 / (1.0 + np.exp(-cx))


@numba.njit(parallel=True, cache=True)
def spinor_loss_batch(candidates, sequence, embeddings, dim, n_angles,
                      n_params_per_token):
    """Evaluate BCE loss for a batch of Spinor+Decay SSM parameter vectors."""
    n_candidates = candidates.shape[0]
    n_steps = sequence.shape[0]
    losses = np.empty(n_candidates)

    for c in numba.prange(n_candidates):
        params = candidates[c]
        h_t = np.zeros(dim)
        h_t[min(2, dim - 1)] = 1.0
        total_loss = 0.0

        for t in range(n_steps):
            tok = sequence[t]
            base = tok * n_params_per_token
            # Givens rotation
            rotated = h_t.copy()
            for i in range(n_angles):
                idx_a = 2 * i
                idx_b = 2 * i + 1
                angle = params[base + i]
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                a_val = rotated[idx_a]
                b_val = rotated[idx_b]
                rotated[idx_a] = cos_a * a_val - sin_a * b_val
                rotated[idx_b] = sin_a * a_val + cos_a * b_val

            decay_raw = params[base + n_angles]
            lambda_t = _clipped_sigmoid(decay_raw)

            for d in range(dim):
                input_s = params[base + n_angles + 1 + d]
                h_t[d] = lambda_t * rotated[d] + input_s * embeddings[t, d]

            # Normalize
            norm_sq = 0.0
            for d in range(dim):
                norm_sq += h_t[d] * h_t[d]
            norm = np.sqrt(norm_sq)
            if norm > 1e-10:
                for d in range(dim):
                    h_t[d] /= norm

            # Accumulate loss (predict next token from current state)
            if t < n_steps - 1:
                logit = h_t[0] * 5.0
                prob = _clipped_sigmoid(logit)
                target = float(sequence[t + 1])
                total_loss += -(target * np.log(prob + 1e-7)
                                + (1.0 - target) * np.log(1.0 - prob + 1e-7))

        losses[c] = total_loss / (n_steps - 1)

    return losses


@numba.njit(parallel=True, cache=True)
def diagonal_loss_batch(candidates, sequence, embeddings, dim,
                        n_params_per_token):
    """Evaluate BCE loss for a batch of Diagonal SSM parameter vectors."""
    n_candidates = candidates.shape[0]
    n_steps = sequence.shape[0]
    losses = np.empty(n_candidates)

    for c in numba.prange(n_candidates):
        params = candidates[c]
        h_t = np.zeros(dim)
        h_t[min(2, dim - 1)] = 1.0
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


@numba.njit(parallel=True, cache=True)
def dense_loss_batch(candidates, sequence, embeddings, dim,
                     n_matrix_params, n_params_per_token):
    """Evaluate BCE loss for a batch of Dense SSM parameter vectors."""
    n_candidates = candidates.shape[0]
    n_steps = sequence.shape[0]
    losses = np.empty(n_candidates)

    for c in numba.prange(n_candidates):
        params = candidates[c]
        h_t = np.zeros(dim)
        h_t[min(2, dim - 1)] = 1.0
        total_loss = 0.0

        for t in range(n_steps):
            tok = sequence[t]
            base = tok * n_params_per_token
            # Dense matrix multiply: A @ h_t
            new_h = np.zeros(dim)
            for i in range(dim):
                dot_val = 0.0
                for j in range(dim):
                    dot_val += params[base + i * dim + j] * h_t[j]
                input_s = params[base + n_matrix_params + i]
                new_h[i] = dot_val + input_s * embeddings[t, i]

            # Normalize
            norm_sq = 0.0
            for i in range(dim):
                norm_sq += new_h[i] * new_h[i]
            norm = np.sqrt(norm_sq)
            if norm > 1e-10:
                for i in range(dim):
                    new_h[i] /= norm

            for i in range(dim):
                h_t[i] = new_h[i]

            if t < n_steps - 1:
                logit = h_t[0] * 5.0
                prob = _clipped_sigmoid(logit)
                target = float(sequence[t + 1])
                total_loss += -(target * np.log(prob + 1e-7)
                                + (1.0 - target) * np.log(1.0 - prob + 1e-7))

        losses[c] = total_loss / (n_steps - 1)

    return losses


# ============================================================================
# 3. Numba-accelerated forward pass (single params, returns states)
# ============================================================================

@numba.njit(cache=True)
def spinor_forward(params, sequence, embeddings, dim, n_angles,
                   n_params_per_token):
    """Run Spinor+Decay SSM, return normalized states."""
    n_steps = sequence.shape[0]
    states = np.zeros((n_steps, dim))
    h_t = np.zeros(dim)
    h_t[min(2, dim - 1)] = 1.0

    for t in range(n_steps):
        tok = sequence[t]
        base = tok * n_params_per_token
        rotated = h_t.copy()
        for i in range(n_angles):
            idx_a = 2 * i
            idx_b = 2 * i + 1
            angle = params[base + i]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            a_val = rotated[idx_a]
            b_val = rotated[idx_b]
            rotated[idx_a] = cos_a * a_val - sin_a * b_val
            rotated[idx_b] = sin_a * a_val + cos_a * b_val

        decay_raw = params[base + n_angles]
        lambda_t = _clipped_sigmoid(decay_raw)
        for d in range(dim):
            input_s = params[base + n_angles + 1 + d]
            h_t[d] = lambda_t * rotated[d] + input_s * embeddings[t, d]

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
def diagonal_forward(params, sequence, embeddings, dim, n_params_per_token):
    n_steps = sequence.shape[0]
    states = np.zeros((n_steps, dim))
    h_t = np.zeros(dim)
    h_t[min(2, dim - 1)] = 1.0

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


@numba.njit(cache=True)
def dense_forward(params, sequence, embeddings, dim, n_matrix_params,
                  n_params_per_token):
    n_steps = sequence.shape[0]
    states = np.zeros((n_steps, dim))
    h_t = np.zeros(dim)
    h_t[min(2, dim - 1)] = 1.0

    for t in range(n_steps):
        tok = sequence[t]
        base = tok * n_params_per_token
        new_h = np.zeros(dim)
        for i in range(dim):
            dot_val = 0.0
            for j in range(dim):
                dot_val += params[base + i * dim + j] * h_t[j]
            input_s = params[base + n_matrix_params + i]
            new_h[i] = dot_val + input_s * embeddings[t, i]

        norm_sq = 0.0
        for i in range(dim):
            norm_sq += new_h[i] * new_h[i]
        norm = np.sqrt(norm_sq)
        if norm > 1e-10:
            for i in range(dim):
                new_h[i] /= norm
        for i in range(dim):
            h_t[i] = new_h[i]
            states[t, i] = new_h[i]

    return states


# ============================================================================
# 4. Model wrappers (thin shells around numba kernels)
# ============================================================================

class SpinorDecaySSM:
    def __init__(self, dim, seed=0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.n_angles = dim // 2
        self.n_params_per_token = self.n_angles + 1 + dim
        self.n_tokens = 2
        n_total = self.n_tokens * self.n_params_per_token
        self.params = rng.normal(0, 0.3, size=n_total)
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base + self.n_angles] = 1.4

    def loss_batch(self, candidates, sequence, embeddings):
        return spinor_loss_batch(
            candidates, sequence, embeddings,
            self.dim, self.n_angles, self.n_params_per_token)

    def forward(self, sequence, embeddings):
        return spinor_forward(
            self.params, sequence, embeddings,
            self.dim, self.n_angles, self.n_params_per_token)

    def total_params(self):
        return self.n_tokens * self.n_params_per_token


class DiagonalSSM:
    def __init__(self, dim, seed=0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.n_params_per_token = 2 * dim
        self.n_tokens = 2
        n_total = self.n_tokens * self.n_params_per_token
        self.params = rng.normal(0, 0.3, size=n_total)
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base:base + dim] = 1.4

    def loss_batch(self, candidates, sequence, embeddings):
        return diagonal_loss_batch(
            candidates, sequence, embeddings,
            self.dim, self.n_params_per_token)

    def forward(self, sequence, embeddings):
        return diagonal_forward(
            self.params, sequence, embeddings,
            self.dim, self.n_params_per_token)

    def total_params(self):
        return self.n_tokens * self.n_params_per_token


class DenseSSM:
    def __init__(self, dim, seed=0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.n_matrix_params = dim * dim
        self.n_params_per_token = self.n_matrix_params + dim
        self.n_tokens = 2
        n_total = self.n_tokens * self.n_params_per_token
        self.params = np.zeros(n_total)

        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            # A = 0.8 * I + small noise
            A_flat = rng.normal(0, 0.05, size=dim * dim)
            for i in range(dim):
                A_flat[i * dim + i] += 0.8
            self.params[base:base + self.n_matrix_params] = A_flat
            self.params[base + self.n_matrix_params:base + self.n_params_per_token] = (
                rng.normal(0, 0.3, size=dim))

    def loss_batch(self, candidates, sequence, embeddings):
        return dense_loss_batch(
            candidates, sequence, embeddings,
            self.dim, self.n_matrix_params, self.n_params_per_token)

    def forward(self, sequence, embeddings):
        return dense_forward(
            self.params, sequence, embeddings,
            self.dim, self.n_matrix_params, self.n_params_per_token)

    def total_params(self):
        return self.n_tokens * self.n_params_per_token


# ============================================================================
# 5. (1+λ)-ES optimizer using batched loss evaluation
# ============================================================================

def optimize_model(model, sequence, embeddings, n_iters=150, pop_size=30,
                   sigma=0.1, seed=0, label=""):
    rng = np.random.default_rng(seed)
    n_params = len(model.params)
    effective_pop = max(pop_size, 4 + int(3 * np.log(n_params + 1)))

    best_params = model.params.copy()
    # Evaluate initial loss (batch of 1)
    init_losses = model.loss_batch(best_params.reshape(1, -1), sequence, embeddings)
    best_loss = init_losses[0]
    losses = [best_loss]

    for iteration in range(n_iters):
        noise = rng.normal(0, sigma, size=(effective_pop, n_params))
        candidates = best_params[None, :] + noise

        # Batched evaluation (numba parallel)
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


# ============================================================================
# 6. Markov score computation
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
# 7. Warm up numba JIT
# ============================================================================

def warmup_jit():
    """Compile all numba functions with tiny inputs so compilation cost
    is paid once, upfront."""
    log("Warming up numba JIT...", )
    dim = 3
    seq = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    emb = embed_tokens_general(seq, dim).astype(np.float64)

    sp = SpinorDecaySSM(dim, seed=99)
    di = DiagonalSSM(dim, seed=99)
    de = DenseSSM(dim, seed=99)

    sp.loss_batch(sp.params.reshape(1, -1), seq, emb)
    di.loss_batch(di.params.reshape(1, -1), seq, emb)
    de.loss_batch(de.params.reshape(1, -1), seq, emb)
    sp.forward(seq, emb)
    di.forward(seq, emb)
    de.forward(seq, emb)
    log("JIT warm-up complete.")


# ============================================================================
# 8. Run experiment at a single dimension
# ============================================================================

def run_single_dim(dim, n_train=2000, n_test=2000, n_iters=150, n_seeds=3,
                   max_lag=5):
    model_names = ["spinor", "diagonal", "dense"]
    results = {n: [] for n in model_names}
    test_losses = {n: [] for n in model_names}
    train_times = {n: [] for n in model_names}
    param_counts = {}

    for seed in range(n_seeds):
        log(f"  Seed {seed+1}/{n_seeds}")
        train_seq = generate_markov_source(n_train, order=2, seed=seed).astype(np.int64)
        test_seq = generate_markov_source(n_test, order=2, seed=seed + 100).astype(np.int64)
        train_emb = embed_tokens_general(train_seq, dim)
        test_emb = embed_tokens_general(test_seq, dim)

        models = {
            "spinor": SpinorDecaySSM(dim=dim, seed=seed),
            "diagonal": DiagonalSSM(dim=dim, seed=seed),
            "dense": DenseSSM(dim=dim, seed=seed),
        }

        if seed == 0:
            param_counts = {name: m.total_params() for name, m in models.items()}

        for name, model in models.items():
            t0 = time.time()
            losses = optimize_model(
                model, train_seq, train_emb, n_iters=n_iters, seed=seed,
                label=f"D={dim} s{seed} {name[:6]:>6}")
            elapsed = time.time() - t0
            log(f"    {name:>8}: {losses[0]:.4f} -> {losses[-1]:.4f} ({elapsed:.1f}s)")

            states = model.forward(test_seq, test_emb)
            scores = compute_markov_score(states, max_lag=max_lag)

            # Test loss
            tl = model.loss_batch(
                model.params.reshape(1, -1), test_seq, test_emb)[0]

            results[name].append(scores)
            test_losses[name].append(tl)
            train_times[name].append(elapsed)

    return results, test_losses, train_times, param_counts


# ============================================================================
# 9. Main experiment
# ============================================================================

def main():
    log("=" * 78)
    log("Generic Dense SSM Ablation (Issue #45 — Cirone Challenge)")
    log("Does geometry matter, or is non-diagonality sufficient?")
    log("=" * 78)

    warmup_jit()

    dimensions = [3, 9, 15]
    n_train = 2000
    n_test = 2000
    n_iters = 150
    n_seeds = 3
    max_lag = 5

    log(f"\nConfig: dims={dimensions}, n_train={n_train}, n_test={n_test}")
    log(f"        n_iters={n_iters}, n_seeds={n_seeds}, source_order=2")
    log(f"        optimizer=(1+λ)-ES, parallel numba, 8 cores")

    all_results = {}

    for dim_idx, dim in enumerate(dimensions):
        log(f"\n{'='*78}")
        log(f"[{dim_idx+1}/{len(dimensions)}] D = {dim}")
        log(f"{'='*78}")

        results, test_losses, train_times, param_counts = run_single_dim(
            dim, n_train=n_train, n_test=n_test, n_iters=n_iters,
            n_seeds=n_seeds, max_lag=max_lag,
        )

        log(f"\n  Parameter counts (total, 2 tokens):")
        for name, count in param_counts.items():
            log(f"    {name:>8}: {count}")

        summary = {}
        for name in ["spinor", "diagonal", "dense"]:
            tl_mean = np.mean(test_losses[name])
            tl_std = np.std(test_losses[name])
            ms_mean = np.nanmean([np.nanmean(s) for s in results[name]])
            ms_std = np.nanstd([np.nanmean(s) for s in results[name]])
            tt_mean = np.mean(train_times[name])
            summary[name] = {
                "test_loss": tl_mean,
                "test_loss_std": tl_std,
                "markov_score": ms_mean,
                "markov_score_std": ms_std,
                "train_time": tt_mean,
                "params": param_counts[name],
            }

        log(f"\n  --- D={dim} SUMMARY ---")
        log(f"  {'Model':>8} | {'Test Loss':>16} | {'Markov Score':>16} | "
            f"{'Params':>6} | {'Time':>6}")
        log(f"  {'-'*8}-+-{'-'*16}-+-{'-'*16}-+-{'-'*6}-+-{'-'*6}")
        for name in ["spinor", "diagonal", "dense"]:
            s = summary[name]
            log(f"  {name:>8} | {s['test_loss']:.4f} +/- {s['test_loss_std']:.4f} | "
                f"{s['markov_score']:.4f} +/- {s['markov_score_std']:.4f} | "
                f"{s['params']:>6} | {s['train_time']:>5.1f}s")

        sp = summary["spinor"]["test_loss"]
        di = summary["diagonal"]["test_loss"]
        de = summary["dense"]["test_loss"]
        log(f"\n  Loss gaps: Spinor-Diag={sp-di:+.4f}  "
            f"Dense-Diag={de-di:+.4f}  Spinor-Dense={sp-de:+.4f}")

        all_results[dim] = summary

    # ========================================================================
    # Final summary table
    # ========================================================================
    log("\n" + "=" * 78)
    log("SUMMARY TABLE")
    log("=" * 78)
    log(f"{'D':>4} | {'Spinor Loss':>12} | {'Diag Loss':>12} | {'Dense Loss':>12} | "
        f"{'Sp-Di':>7} | {'De-Di':>7} | {'Sp-De':>7}")
    log("-" * 78)

    for dim in dimensions:
        r = all_results[dim]
        sp = r["spinor"]["test_loss"]
        di = r["diagonal"]["test_loss"]
        de = r["dense"]["test_loss"]
        log(f"{dim:>4} | {sp:>12.4f} | {di:>12.4f} | {de:>12.4f} | "
            f"{sp-di:>+7.4f} | {de-di:>+7.4f} | {sp-de:>+7.4f}")

    log("-" * 78)

    log(f"\n{'D':>4} | {'Spinor MS':>12} | {'Diag MS':>12} | {'Dense MS':>12} | "
        f"{'Sp Params':>9} | {'Di Params':>9} | {'De Params':>9}")
    log("-" * 78)

    for dim in dimensions:
        r = all_results[dim]
        log(f"{dim:>4} | {r['spinor']['markov_score']:>12.4f} | "
            f"{r['diagonal']['markov_score']:>12.4f} | "
            f"{r['dense']['markov_score']:>12.4f} | "
            f"{r['spinor']['params']:>9} | "
            f"{r['diagonal']['params']:>9} | "
            f"{r['dense']['params']:>9}")

    log("-" * 78)

    # ========================================================================
    # Decision table evaluation
    # ========================================================================
    log("\n" + "=" * 78)
    log("DECISION TABLE EVALUATION (Issue #45)")
    log("=" * 78)
    log("  Lower test loss = better prediction")
    log("  Comparison: Spinor has FEWER params than Dense (geometric constraint)")
    log("")

    for dim in dimensions:
        r = all_results[dim]
        sp = r["spinor"]["test_loss"]
        di = r["diagonal"]["test_loss"]
        de = r["dense"]["test_loss"]
        sp_std = r["spinor"]["test_loss_std"]
        de_std = r["dense"]["test_loss_std"]
        di_std = r["diagonal"]["test_loss_std"]

        sp_de_overlap = abs(sp - de) < (sp_std + de_std)
        de_di_overlap = abs(de - di) < (de_std + di_std)

        if sp < de < di:
            verdict = "Spinor > Dense > Diagonal --> GEOMETRY MATTERS"
        elif sp < di and de < di and sp_de_overlap:
            verdict = "Spinor ~ Dense > Diagonal --> non-diagonality, not geometry"
        elif de < sp < di:
            verdict = "Dense > Spinor > Diagonal --> geometry over-constrains"
        elif de_di_overlap:
            verdict = "Dense ~ Diagonal --> non-diagonality doesn't help (data too simple)"
        elif sp < di and sp < de:
            verdict = "Spinor > Dense, Diagonal --> GEOMETRY MATTERS"
        elif sp < di and de >= di:
            verdict = "Spinor > Diagonal >= Dense --> geometry helps, dense fails to train"
        else:
            verdict = f"Inconclusive (Sp={sp:.4f}, De={de:.4f}, Di={di:.4f})"

        log(f"  D={dim:>2}: {verdict}")
        log(f"         Spinor={sp:.4f}  Dense={de:.4f}  Diagonal={di:.4f}  "
            f"(Spinor: {r['spinor']['params']}p, Dense: {r['dense']['params']}p)")

    # Parameter efficiency
    log("\n" + "=" * 78)
    log("PARAMETER EFFICIENCY")
    log("=" * 78)
    for dim in dimensions:
        r = all_results[dim]
        sp_params = r["spinor"]["params"]
        de_params = r["dense"]["params"]
        ratio = de_params / sp_params
        sp_loss = r["spinor"]["test_loss"]
        de_loss = r["dense"]["test_loss"]
        log(f"  D={dim:>2}: Dense has {ratio:.1f}x more params than Spinor "
            f"({de_params} vs {sp_params})")
        if sp_loss <= de_loss:
            log(f"         Spinor wins despite {ratio:.1f}x fewer params --> "
                f"strong inductive bias evidence")
        else:
            log(f"         Dense wins with {ratio:.1f}x more params --> "
                f"inconclusive (could be param advantage)")

    log("\n" + "=" * 78)
    log("DONE")
    log("=" * 78)


if __name__ == "__main__":
    main()
