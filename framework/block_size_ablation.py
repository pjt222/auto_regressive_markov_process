"""
Block-Size Ablation (Issue #44 -- Sharp Transition Detection)

Tests whether prediction performance exhibits a sharp transition as the
rotation block size b increases within D=12 dimensional state space.

At D=12, we partition dimensions into n_blocks = D/b blocks of size b,
each block receiving b*(b-1)/2 Givens (plane) rotations to cover SO(b).

Block sizes tested: b in {1, 2, 3, 4, 6, 12}

| Block b | n_blocks | Rot angles/block | Tot rot/tok | Decay | Input | Total/tok |
|---------|----------|------------------|-------------|-------|-------|-----------|
|   1     |   12     |        0         |      0      |  12   |  12   |    24     |
|   2     |    6     |        1         |      6      |   1   |  12   |    19     |
|   3     |    4     |        3         |     12      |   1   |  12   |    25     |
|   4     |    3     |        6         |     18      |   1   |  12   |    31     |
|   6     |    2     |       15         |     30      |   1   |  12   |    43     |
|  12     |    1     |       66         |     66      |   1   |  12   |    79     |

Block-1 = diagonal model (no rotations, per-dimension decay).
Block-b (b>=2) = block Givens rotation + scalar decay + input scaling.

Usage:
    cd framework && python3 block_size_ablation.py

Performance: uses numba JIT + parallel candidate evaluation.
"""

import numpy as np
import numba
import time
import sys

from toy_example import generate_markov_source
from dense_ablation import (
    DiagonalSSM,
    embed_tokens_general,
    optimize_model,
    compute_markov_score,
)


def log(msg):
    print(msg, flush=True)


# ============================================================================
# 1. Numba helpers (must be redefined -- cannot import njit functions)
# ============================================================================

@numba.njit(cache=True)
def _clipped_sigmoid(x):
    cx = min(max(x, -10.0), 10.0)
    return 1.0 / (1.0 + np.exp(-cx))


# ============================================================================
# 2. Block-b Givens rotation loss (batched, b >= 2)
# ============================================================================

@numba.njit(parallel=True, cache=True)
def block_b_loss_batch(candidates, sequence, embeddings, dim, block_size,
                       n_rot_params, n_params_per_token):
    """Evaluate BCE loss for a batch of block-b SSM parameter vectors.

    Parameter layout per token (b >= 2):
        [0 .. n_rot_params-1]           : rotation angles (all blocks)
        [n_rot_params]                  : scalar decay (lambda_raw)
        [n_rot_params+1 .. n_rot_params+dim] : input_scale (D values)

    Total per token = n_rot_params + 1 + dim
    """
    n_candidates = candidates.shape[0]
    n_steps = sequence.shape[0]
    losses = np.empty(n_candidates)
    n_blocks = dim // block_size
    angles_per_block = block_size * (block_size - 1) // 2

    for c in numba.prange(n_candidates):
        params = candidates[c]
        h_t = np.zeros(dim)
        h_t[min(2, dim - 1)] = 1.0
        total_loss = 0.0

        for t in range(n_steps):
            tok = sequence[t]
            base = tok * n_params_per_token

            # Apply block-diagonal Givens rotations
            rotated = h_t.copy()
            angle_offset = 0
            for blk in range(n_blocks):
                start = blk * block_size
                param_idx = 0
                for i in range(block_size):
                    for j in range(i + 1, block_size):
                        angle = params[base + angle_offset + param_idx]
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)
                        a_val = rotated[start + i]
                        b_val = rotated[start + j]
                        rotated[start + i] = cos_a * a_val - sin_a * b_val
                        rotated[start + j] = sin_a * a_val + cos_a * b_val
                        param_idx += 1
                angle_offset += angles_per_block

            # Scalar decay
            decay_raw = params[base + n_rot_params]
            lambda_t = _clipped_sigmoid(decay_raw)

            # State update: h_t = lambda * rotated + input_scale * embedding
            for d in range(dim):
                input_s = params[base + n_rot_params + 1 + d]
                h_t[d] = lambda_t * rotated[d] + input_s * embeddings[t, d]

            # Normalize
            norm_sq = 0.0
            for d in range(dim):
                norm_sq += h_t[d] * h_t[d]
            norm = np.sqrt(norm_sq)
            if norm > 1e-10:
                for d in range(dim):
                    h_t[d] /= norm

            # BCE loss: predict next token from h_t[0]
            if t < n_steps - 1:
                logit = h_t[0] * 5.0
                prob = _clipped_sigmoid(logit)
                target = float(sequence[t + 1])
                total_loss += -(target * np.log(prob + 1e-7)
                                + (1.0 - target) * np.log(1.0 - prob + 1e-7))

        losses[c] = total_loss / (n_steps - 1)

    return losses


# ============================================================================
# 3. Block-b Givens forward pass (single params, returns states)
# ============================================================================

@numba.njit(cache=True)
def block_b_forward(params, sequence, embeddings, dim, block_size,
                    n_rot_params, n_params_per_token):
    """Run block-b SSM forward, return normalized states."""
    n_steps = sequence.shape[0]
    states = np.zeros((n_steps, dim))
    h_t = np.zeros(dim)
    h_t[min(2, dim - 1)] = 1.0
    n_blocks = dim // block_size
    angles_per_block = block_size * (block_size - 1) // 2

    for t in range(n_steps):
        tok = sequence[t]
        base = tok * n_params_per_token

        rotated = h_t.copy()
        angle_offset = 0
        for blk in range(n_blocks):
            start = blk * block_size
            param_idx = 0
            for i in range(block_size):
                for j in range(i + 1, block_size):
                    angle = params[base + angle_offset + param_idx]
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    a_val = rotated[start + i]
                    b_val = rotated[start + j]
                    rotated[start + i] = cos_a * a_val - sin_a * b_val
                    rotated[start + j] = sin_a * a_val + cos_a * b_val
                    param_idx += 1
            angle_offset += angles_per_block

        decay_raw = params[base + n_rot_params]
        lambda_t = _clipped_sigmoid(decay_raw)

        for d in range(dim):
            input_s = params[base + n_rot_params + 1 + d]
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


# ============================================================================
# 4. BlockBSSM wrapper (b >= 2)
# ============================================================================

class BlockBSSM:
    """Block-diagonal Givens rotation SSM for block_size >= 2."""

    def __init__(self, dim, block_size, seed=0):
        assert dim % block_size == 0, f"dim={dim} not divisible by block_size={block_size}"
        assert block_size >= 2, "Use DiagonalSSM for block_size=1"

        self.dim = dim
        self.block_size = block_size
        self.n_blocks = dim // block_size
        self.angles_per_block = block_size * (block_size - 1) // 2
        self.n_rot_params = self.n_blocks * self.angles_per_block
        # Layout per token: rot_angles + 1 decay + D input_scale
        self.n_params_per_token = self.n_rot_params + 1 + dim
        self.n_tokens = 2
        n_total = self.n_tokens * self.n_params_per_token

        rng = np.random.default_rng(seed)
        self.params = rng.normal(0, 0.3, size=n_total)

        # Initialize decay to sigmoid^{-1}(0.8) ~ 1.4
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base + self.n_rot_params] = 1.4

    def loss_batch(self, candidates, sequence, embeddings):
        return block_b_loss_batch(
            candidates, sequence, embeddings,
            self.dim, self.block_size,
            self.n_rot_params, self.n_params_per_token)

    def forward(self, sequence, embeddings):
        return block_b_forward(
            self.params, sequence, embeddings,
            self.dim, self.block_size,
            self.n_rot_params, self.n_params_per_token)

    def total_params(self):
        return self.n_tokens * self.n_params_per_token


# ============================================================================
# 5. JIT warmup
# ============================================================================

def warmup_jit():
    """Compile all numba functions with tiny inputs."""
    log("Warming up numba JIT...")
    dim = 6
    seq = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    emb = embed_tokens_general(seq, dim).astype(np.float64)

    # Warm up diagonal (b=1) via DiagonalSSM
    diag_model = DiagonalSSM(dim, seed=99)
    diag_model.loss_batch(diag_model.params.reshape(1, -1), seq, emb)
    diag_model.forward(seq, emb)

    # Warm up block-b for b=2 and b=3 (covers the code paths)
    for block_size in [2, 3]:
        blk_model = BlockBSSM(dim, block_size, seed=99)
        blk_model.loss_batch(blk_model.params.reshape(1, -1), seq, emb)
        blk_model.forward(seq, emb)

    log("JIT warm-up complete.")


# ============================================================================
# 6. Run experiment for one block size
# ============================================================================

def run_single_block(block_size, dim, n_train, n_test, n_iters, n_seeds,
                     pop_size, max_lag):
    """Train and evaluate a single block-size configuration across seeds."""
    markov_results = []
    test_loss_results = []
    train_time_results = []
    param_count = None

    for seed in range(n_seeds):
        log(f"    Seed {seed+1}/{n_seeds}")
        train_seq = generate_markov_source(n_train, order=2, seed=seed).astype(np.int64)
        test_seq = generate_markov_source(n_test, order=2, seed=seed + 100).astype(np.int64)
        train_emb = embed_tokens_general(train_seq, dim)
        test_emb = embed_tokens_general(test_seq, dim)

        if block_size == 1:
            model = DiagonalSSM(dim=dim, seed=seed)
        else:
            model = BlockBSSM(dim=dim, block_size=block_size, seed=seed)

        if seed == 0:
            param_count = model.total_params()

        label = f"b={block_size} s{seed}"
        t0 = time.time()
        losses = optimize_model(
            model, train_seq, train_emb,
            n_iters=n_iters, pop_size=pop_size,
            seed=seed, label=label)
        elapsed = time.time() - t0
        log(f"      b={block_size}: {losses[0]:.4f} -> {losses[-1]:.4f} ({elapsed:.1f}s)")

        # Test loss
        test_loss = model.loss_batch(
            model.params.reshape(1, -1), test_seq, test_emb)[0]

        # Markov score
        states = model.forward(test_seq, test_emb)
        scores = compute_markov_score(states, max_lag=max_lag)

        test_loss_results.append(test_loss)
        markov_results.append(scores)
        train_time_results.append(elapsed)

    return {
        "block_size": block_size,
        "params": param_count,
        "test_losses": test_loss_results,
        "markov_scores": markov_results,
        "train_times": train_time_results,
    }


# ============================================================================
# 7. Main experiment
# ============================================================================

def main():
    log("=" * 78)
    log("Block-Size Ablation (Issue #44 -- Sharp Transition Detection)")
    log("Does prediction quality exhibit a sharp transition as block size grows?")
    log("=" * 78)

    # Configuration
    dim = 12
    block_sizes = [1, 2, 3, 4, 6, 12]
    n_train = 2000
    n_test = 2000
    n_iters = 150
    base_pop_size = 30
    n_seeds = 3
    source_order = 2
    max_lag = 5

    log(f"\nConfig: D={dim}, block_sizes={block_sizes}")
    log(f"        n_train={n_train}, n_test={n_test}")
    log(f"        n_iters={n_iters}, base_pop_size={base_pop_size}, n_seeds={n_seeds}")
    log(f"        source_order={source_order}, max_lag={max_lag}")
    log(f"        optimizer=(1+lambda)-ES, parallel numba")

    # Show parameter table
    log(f"\nParameter counts per token:")
    log(f"  {'b':>3} | {'n_blocks':>8} | {'Rot/blk':>7} | {'Tot rot':>7} | "
        f"{'Decay':>5} | {'Input':>5} | {'Tot/tok':>7} | {'Total(2tok)':>11}")
    log(f"  {'-'*3}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*11}")
    for block_size in block_sizes:
        n_blocks = dim // block_size
        angles_per_block = block_size * (block_size - 1) // 2
        total_rot = n_blocks * angles_per_block
        if block_size == 1:
            decay_params = dim  # per-dimension
            total_per_tok = dim + dim  # D decay + D input
        else:
            decay_params = 1  # scalar
            total_per_tok = total_rot + 1 + dim
        total_2tok = total_per_tok * 2
        log(f"  {block_size:>3} | {n_blocks:>8} | {angles_per_block:>7} | "
            f"{total_rot:>7} | {decay_params:>5} | {dim:>5} | "
            f"{total_per_tok:>7} | {total_2tok:>11}")

    warmup_jit()

    # Run all block sizes
    all_results = {}
    for idx, block_size in enumerate(block_sizes):
        log(f"\n{'='*78}")
        log(f"[{idx+1}/{len(block_sizes)}] Block size b = {block_size}")
        log(f"{'='*78}")

        # Adaptive pop_size: scale up for high-parameter models
        if block_size == 1:
            model_for_count = DiagonalSSM(dim=dim, seed=0)
        else:
            model_for_count = BlockBSSM(dim=dim, block_size=block_size, seed=0)
        n_params = model_for_count.total_params()
        effective_pop = max(base_pop_size, 4 + int(3 * np.log(n_params + 1)))
        log(f"  Total params: {n_params}, effective pop_size: {effective_pop}")

        result = run_single_block(
            block_size, dim, n_train, n_test, n_iters, n_seeds,
            pop_size=effective_pop, max_lag=max_lag)
        all_results[block_size] = result

        tl_mean = np.mean(result["test_losses"])
        tl_std = np.std(result["test_losses"])
        ms_mean = np.nanmean([np.nanmean(s) for s in result["markov_scores"]])
        ms_std = np.nanstd([np.nanmean(s) for s in result["markov_scores"]])
        tt_mean = np.mean(result["train_times"])

        log(f"\n  b={block_size} summary:")
        log(f"    Test loss:    {tl_mean:.4f} +/- {tl_std:.4f}")
        log(f"    Markov score: {ms_mean:.4f} +/- {ms_std:.4f}")
        log(f"    Params:       {result['params']}")
        log(f"    Avg time:     {tt_mean:.1f}s")

    # ========================================================================
    # Summary table
    # ========================================================================
    log("\n" + "=" * 78)
    log("SUMMARY TABLE")
    log("=" * 78)
    log(f"  {'b':>3} | {'Params':>6} | {'Test Loss':>18} | "
        f"{'Markov Score':>18} | {'Time':>6}")
    log(f"  {'-'*3}-+-{'-'*6}-+-{'-'*18}-+-{'-'*18}-+-{'-'*6}")

    summaries = {}
    for block_size in block_sizes:
        result = all_results[block_size]
        tl_mean = np.mean(result["test_losses"])
        tl_std = np.std(result["test_losses"])
        ms_mean = np.nanmean([np.nanmean(s) for s in result["markov_scores"]])
        ms_std = np.nanstd([np.nanmean(s) for s in result["markov_scores"]])
        tt_mean = np.mean(result["train_times"])

        summaries[block_size] = {
            "test_loss": tl_mean,
            "test_loss_std": tl_std,
            "markov_score": ms_mean,
            "markov_score_std": ms_std,
            "train_time": tt_mean,
            "params": result["params"],
        }

        log(f"  {block_size:>3} | {result['params']:>6} | "
            f"{tl_mean:.4f} +/- {tl_std:.4f} | "
            f"{ms_mean:.4f} +/- {ms_std:.4f} | "
            f"{tt_mean:>5.1f}s")

    log(f"  {'-'*3}-+-{'-'*6}-+-{'-'*18}-+-{'-'*18}-+-{'-'*6}")

    # ========================================================================
    # Pairwise loss gaps (relative to b=1 baseline)
    # ========================================================================
    log("\n" + "=" * 78)
    log("LOSS GAPS vs BASELINE (b=1 diagonal)")
    log("=" * 78)
    baseline_loss = summaries[1]["test_loss"]
    log(f"  Baseline (b=1) test loss: {baseline_loss:.4f}")
    log(f"\n  {'b':>3} | {'Loss':>8} | {'Gap vs b=1':>10} | {'% Improve':>9} | "
        f"{'Extra Params':>12} | {'Loss/Param':>10}")
    log(f"  {'-'*3}-+-{'-'*8}-+-{'-'*10}-+-{'-'*9}-+-{'-'*12}-+-{'-'*10}")

    for block_size in block_sizes:
        s = summaries[block_size]
        gap = s["test_loss"] - baseline_loss
        pct_improve = -gap / baseline_loss * 100 if baseline_loss > 0 else 0.0
        extra_params = s["params"] - summaries[1]["params"]
        loss_per_param = gap / extra_params if extra_params != 0 else 0.0
        log(f"  {block_size:>3} | {s['test_loss']:.4f} | {gap:>+10.4f} | "
            f"{pct_improve:>+8.2f}% | {extra_params:>12} | "
            f"{loss_per_param:>+10.6f}")

    # ========================================================================
    # Sharp transition analysis
    # ========================================================================
    log("\n" + "=" * 78)
    log("SHARP TRANSITION ANALYSIS")
    log("=" * 78)
    log("  Looking for the steepest loss improvement between consecutive block sizes.")
    log("")

    prev_block_size = None
    max_delta = 0.0
    max_delta_pair = (None, None)

    for block_size in block_sizes:
        if prev_block_size is not None:
            delta = summaries[prev_block_size]["test_loss"] - summaries[block_size]["test_loss"]
            marker = " <-- largest" if abs(delta) > abs(max_delta) else ""
            if abs(delta) > abs(max_delta):
                max_delta = delta
                max_delta_pair = (prev_block_size, block_size)
            log(f"  b={prev_block_size} -> b={block_size}: "
                f"delta_loss = {delta:+.4f}{marker}")
        prev_block_size = block_size

    log("")
    if max_delta_pair[0] is not None:
        log(f"  Sharpest transition: b={max_delta_pair[0]} -> b={max_delta_pair[1]} "
            f"(delta = {max_delta:+.4f})")
        if max_delta_pair == (2, 3):
            log("  --> Consistent with hypothesis: sharp jump at b=2 -> b=3")
            log("      (SO(3) rotations unlock qualitatively different mixing)")
        elif max_delta_pair == (1, 2):
            log("  --> Transition at b=1 -> b=2: any rotation helps vs diagonal")
        else:
            log(f"  --> Transition at unexpected point b={max_delta_pair[0]} -> "
                f"b={max_delta_pair[1]}")
    else:
        log("  No transition detected (single block size).")

    # ========================================================================
    # Parameter efficiency analysis
    # ========================================================================
    log("\n" + "=" * 78)
    log("PARAMETER EFFICIENCY")
    log("=" * 78)
    log("  Improvement per extra parameter (relative to b=1 baseline)")
    log("")

    best_efficiency = -np.inf
    best_efficiency_b = None

    for block_size in block_sizes[1:]:  # skip b=1 (baseline)
        s = summaries[block_size]
        improvement = baseline_loss - s["test_loss"]
        extra_params = s["params"] - summaries[1]["params"]
        efficiency = improvement / extra_params if extra_params > 0 else 0.0
        log(f"  b={block_size:>2}: improvement={improvement:+.4f}, "
            f"extra_params={extra_params:>3}, "
            f"efficiency={efficiency:.6f} loss/param")
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_efficiency_b = block_size

    if best_efficiency_b is not None:
        log(f"\n  Most efficient block size: b={best_efficiency_b} "
            f"({best_efficiency:.6f} loss improvement per extra param)")

    # ========================================================================
    # Markov score analysis
    # ========================================================================
    log("\n" + "=" * 78)
    log("MARKOV SCORE vs BLOCK SIZE")
    log("=" * 78)
    log("  Higher Markov score = more history leakage (less Markov-like)")
    log("")
    for block_size in block_sizes:
        s = summaries[block_size]
        log(f"  b={block_size:>2}: Markov score = {s['markov_score']:.4f} "
            f"+/- {s['markov_score_std']:.4f}")

    log("\n" + "=" * 78)
    log("DONE")
    log("=" * 78)


if __name__ == "__main__":
    main()
