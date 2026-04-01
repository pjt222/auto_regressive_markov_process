"""
Epsilon Calibration Experiment (Issue #25)

Calibrates the bound:  epsilon(1, D) <= C * Var[log ||h_t||] / D

where epsilon is the Markov score (conditional mutual information
I(s_{t+1}; s_{t-j} | s_t)) of the normalized projection s_t = h_t / ||h_t||.

The k=1 case was disproved: the norm ||h_t|| leaks history, so s_t is NOT
Markov.  This experiment quantifies how much the norm variance explains the
Markov violation across models (Spinor+Decay, Diagonal, Dense) and
dimensions D = {3, 9, 15}.

Reuses model classes and optimizer from dense_ablation.py; adds
norm-tracking forward passes to measure Var[log ||h_t||].

Usage:
    cd framework && python3 epsilon_calibration.py

Performance: numba JIT + parallel candidate evaluation.
"""

import numpy as np
import numba
import time
import sys
from toy_example import generate_markov_source
from dense_ablation import (
    SpinorDecaySSM,
    DiagonalSSM,
    DenseSSM,
    optimize_model,
    compute_markov_score,
    embed_tokens_general,
)


def log(msg):
    print(msg, flush=True)


# ============================================================================
# 1. Clipped sigmoid (must be redefined for numba -- cannot import across modules)
# ============================================================================

@numba.njit(cache=True)
def _clipped_sigmoid(x):
    cx = min(max(x, -10.0), 10.0)
    return 1.0 / (1.0 + np.exp(-cx))


# ============================================================================
# 2. Norm-tracking forward passes
#
# Identical to the forward functions in dense_ablation.py, but also record
# log(||h_raw||) at each step BEFORE normalization.  The raw norm is what
# "leaks history" per the k=1 disproof.
# ============================================================================

@numba.njit(cache=True)
def spinor_forward_with_norms(params, sequence, embeddings, dim, n_angles,
                              n_params_per_token):
    """Spinor+Decay SSM forward pass returning (states, log_norms)."""
    n_steps = sequence.shape[0]
    states = np.zeros((n_steps, dim))
    log_norms = np.zeros(n_steps)
    h_t = np.zeros(dim)
    h_t[min(2, dim - 1)] = 1.0

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

        # --- Norm tracking (BEFORE normalization) ---
        norm_sq = 0.0
        for d in range(dim):
            norm_sq += h_t[d] * h_t[d]
        log_norms[t] = 0.5 * np.log(norm_sq + 1e-30)

        # Normalize
        norm = np.sqrt(norm_sq)
        if norm > 1e-10:
            for d in range(dim):
                h_t[d] /= norm

        for d in range(dim):
            states[t, d] = h_t[d]

    return states, log_norms


@numba.njit(cache=True)
def diagonal_forward_with_norms(params, sequence, embeddings, dim,
                                n_params_per_token):
    """Diagonal SSM forward pass returning (states, log_norms)."""
    n_steps = sequence.shape[0]
    states = np.zeros((n_steps, dim))
    log_norms = np.zeros(n_steps)
    h_t = np.zeros(dim)
    h_t[min(2, dim - 1)] = 1.0

    for t in range(n_steps):
        tok = sequence[t]
        base = tok * n_params_per_token
        for d in range(dim):
            lam = _clipped_sigmoid(params[base + d])
            inp = params[base + dim + d]
            h_t[d] = lam * h_t[d] + inp * embeddings[t, d]

        # --- Norm tracking (BEFORE normalization) ---
        norm_sq = 0.0
        for d in range(dim):
            norm_sq += h_t[d] * h_t[d]
        log_norms[t] = 0.5 * np.log(norm_sq + 1e-30)

        # Normalize
        norm = np.sqrt(norm_sq)
        if norm > 1e-10:
            for d in range(dim):
                h_t[d] /= norm

        for d in range(dim):
            states[t, d] = h_t[d]

    return states, log_norms


@numba.njit(cache=True)
def dense_forward_with_norms(params, sequence, embeddings, dim,
                             n_matrix_params, n_params_per_token):
    """Dense SSM forward pass returning (states, log_norms)."""
    n_steps = sequence.shape[0]
    states = np.zeros((n_steps, dim))
    log_norms = np.zeros(n_steps)
    h_t = np.zeros(dim)
    h_t[min(2, dim - 1)] = 1.0

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

        # --- Norm tracking (BEFORE normalization) ---
        norm_sq = 0.0
        for i in range(dim):
            norm_sq += new_h[i] * new_h[i]
        log_norms[t] = 0.5 * np.log(norm_sq + 1e-30)

        # Normalize
        norm = np.sqrt(norm_sq)
        if norm > 1e-10:
            for i in range(dim):
                new_h[i] /= norm

        for i in range(dim):
            h_t[i] = new_h[i]
            states[t, i] = new_h[i]

    return states, log_norms


# ============================================================================
# 3. Warm up numba JIT (including norm-tracking variants)
# ============================================================================

def warmup_jit():
    """Compile all numba functions with tiny inputs so compilation cost
    is paid once, upfront."""
    log("Warming up numba JIT...")
    dim = 3
    seq = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    emb = embed_tokens_general(seq, dim).astype(np.float64)

    # Warm up model classes (loss_batch + forward from dense_ablation)
    sp = SpinorDecaySSM(dim, seed=99)
    di = DiagonalSSM(dim, seed=99)
    de = DenseSSM(dim, seed=99)

    sp.loss_batch(sp.params.reshape(1, -1), seq, emb)
    di.loss_batch(di.params.reshape(1, -1), seq, emb)
    de.loss_batch(de.params.reshape(1, -1), seq, emb)
    sp.forward(seq, emb)
    di.forward(seq, emb)
    de.forward(seq, emb)

    # Warm up norm-tracking forward passes
    spinor_forward_with_norms(
        sp.params, seq, emb, dim, sp.n_angles, sp.n_params_per_token)
    diagonal_forward_with_norms(
        di.params, seq, emb, dim, di.n_params_per_token)
    dense_forward_with_norms(
        de.params, seq, emb, dim, de.n_matrix_params, de.n_params_per_token)

    log("JIT warm-up complete.")


# ============================================================================
# 4. Run experiment at a single dimension
# ============================================================================

def run_single_dim(dim, n_train=2000, n_test=2000, n_iters=150, n_seeds=3,
                   max_lag=5):
    """Train all 3 models, then measure norm variance and Markov score."""
    model_names = ["spinor", "diagonal", "dense"]
    results = {name: [] for name in model_names}

    for seed in range(n_seeds):
        log(f"  Seed {seed + 1}/{n_seeds}")
        train_seq = generate_markov_source(
            n_train, order=2, seed=seed).astype(np.int64)
        test_seq = generate_markov_source(
            n_test, order=2, seed=seed + 100).astype(np.int64)
        train_emb = embed_tokens_general(train_seq, dim)
        test_emb = embed_tokens_general(test_seq, dim)

        models = {
            "spinor": SpinorDecaySSM(dim=dim, seed=seed),
            "diagonal": DiagonalSSM(dim=dim, seed=seed),
            "dense": DenseSSM(dim=dim, seed=seed),
        }

        for name, model in models.items():
            # --- Train ---
            t0 = time.time()
            losses = optimize_model(
                model, train_seq, train_emb, n_iters=n_iters, seed=seed,
                label=f"D={dim} s{seed} {name[:6]:>6}")
            elapsed = time.time() - t0
            log(f"    {name:>8}: {losses[0]:.4f} -> {losses[-1]:.4f} "
                f"({elapsed:.1f}s)")

            # --- Test loss ---
            test_loss = model.loss_batch(
                model.params.reshape(1, -1), test_seq, test_emb)[0]

            # --- Norm-tracking forward on test data ---
            if name == "spinor":
                states, log_norms = spinor_forward_with_norms(
                    model.params, test_seq, test_emb,
                    model.dim, model.n_angles, model.n_params_per_token)
            elif name == "diagonal":
                states, log_norms = diagonal_forward_with_norms(
                    model.params, test_seq, test_emb,
                    model.dim, model.n_params_per_token)
            else:  # dense
                states, log_norms = dense_forward_with_norms(
                    model.params, test_seq, test_emb,
                    model.dim, model.n_matrix_params,
                    model.n_params_per_token)

            # --- Markov score ---
            markov_scores = compute_markov_score(states, max_lag=max_lag)
            mean_markov_score = float(np.nanmean(markov_scores))

            # --- Norm statistics ---
            # Discard initial transient (first 50 steps)
            transient = min(50, len(log_norms) // 4)
            stable_log_norms = log_norms[transient:]
            var_log_norm = float(np.var(stable_log_norms))
            mean_log_norm = float(np.mean(stable_log_norms))

            results[name].append({
                "seed": seed,
                "dim": dim,
                "params": model.total_params(),
                "test_loss": test_loss,
                "markov_score": mean_markov_score,
                "var_log_norm": var_log_norm,
                "mean_log_norm": mean_log_norm,
                "train_time": elapsed,
            })

    return results


# ============================================================================
# 5. Main experiment
# ============================================================================

def main():
    log("=" * 78)
    log("Epsilon Calibration (Issue #25)")
    log("Bound: epsilon(1, D) <= C * Var[log ||h_t||] / D")
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
    log(f"        optimizer=(1+lambda)-ES, parallel numba")

    # ------------------------------------------------------------------
    # Collect all results
    # ------------------------------------------------------------------
    all_rows = []  # flat list of per-(model, dim, seed) dicts

    for dim_idx, dim in enumerate(dimensions):
        log(f"\n{'=' * 78}")
        log(f"[{dim_idx + 1}/{len(dimensions)}] D = {dim}")
        log(f"{'=' * 78}")

        dim_results = run_single_dim(
            dim, n_train=n_train, n_test=n_test, n_iters=n_iters,
            n_seeds=n_seeds, max_lag=max_lag)

        for name in ["spinor", "diagonal", "dense"]:
            for row in dim_results[name]:
                row["model"] = name
                all_rows.append(row)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("SUMMARY TABLE")
    log("=" * 78)
    header = (f"{'D':>4} | {'Model':>8} | {'Params':>6} | {'Test Loss':>10} | "
              f"{'Markov(eps)':>11} | {'Var[log||h||]':>14} | {'Var/D':>8} | "
              f"{'Predicted eps':>13}")
    log(header)
    log("-" * len(header))

    # Aggregate by (model, dim): compute means across seeds
    summary_rows = []
    for dim in dimensions:
        for name in ["spinor", "diagonal", "dense"]:
            rows = [r for r in all_rows
                    if r["dim"] == dim and r["model"] == name]
            mean_test_loss = np.mean([r["test_loss"] for r in rows])
            mean_markov = np.mean([r["markov_score"] for r in rows])
            mean_var_log = np.mean([r["var_log_norm"] for r in rows])
            mean_mean_log = np.mean([r["mean_log_norm"] for r in rows])
            params = rows[0]["params"]
            var_over_d = mean_var_log / dim

            summary_rows.append({
                "dim": dim,
                "model": name,
                "params": params,
                "test_loss": mean_test_loss,
                "markov_score": mean_markov,
                "var_log_norm": mean_var_log,
                "mean_log_norm": mean_mean_log,
                "var_over_d": var_over_d,
            })

    # ------------------------------------------------------------------
    # Fit the bound:  epsilon = C * (Var[log||h||] / D) + intercept
    # ------------------------------------------------------------------
    var_over_d_array = np.array([r["var_over_d"] for r in summary_rows])
    epsilon_array = np.array([r["markov_score"] for r in summary_rows])

    # Handle edge case: if all values are identical, polyfit would fail
    if np.std(var_over_d_array) < 1e-15:
        slope_c = 0.0
        intercept = float(np.mean(epsilon_array))
        r_squared = 0.0
        log("\nWARNING: Var/D has no variation -- linear fit is degenerate.")
    else:
        coefficients = np.polyfit(var_over_d_array, epsilon_array, 1)
        slope_c = coefficients[0]
        intercept = coefficients[1]
        predicted = slope_c * var_over_d_array + intercept
        ss_res = np.sum((epsilon_array - predicted) ** 2)
        ss_tot = np.sum((epsilon_array - np.mean(epsilon_array)) ** 2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-30)

    # Compute predicted epsilon for each row
    for row in summary_rows:
        row["predicted_eps"] = slope_c * row["var_over_d"] + intercept

    # Print summary table
    for row in summary_rows:
        log(f"{row['dim']:>4} | {row['model']:>8} | {row['params']:>6} | "
            f"{row['test_loss']:>10.4f} | {row['markov_score']:>11.6f} | "
            f"{row['var_log_norm']:>14.6f} | {row['var_over_d']:>8.6f} | "
            f"{row['predicted_eps']:>13.6f}")

    # ------------------------------------------------------------------
    # Fitted bound report
    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("FITTED BOUND")
    log("=" * 78)
    log(f"  epsilon = C * Var[log||h||] / D + intercept")
    log(f"  C (slope)    = {slope_c:.6f}")
    log(f"  intercept    = {intercept:.6f}")
    log(f"  R-squared    = {r_squared:.6f}")
    log("")

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------
    log("RESIDUALS (epsilon_observed - epsilon_predicted):")
    log(f"{'D':>4} | {'Model':>8} | {'eps_obs':>10} | {'eps_pred':>10} | "
        f"{'residual':>10}")
    log("-" * 52)
    for row in summary_rows:
        residual = row["markov_score"] - row["predicted_eps"]
        log(f"{row['dim']:>4} | {row['model']:>8} | "
            f"{row['markov_score']:>10.6f} | {row['predicted_eps']:>10.6f} | "
            f"{residual:>+10.6f}")

    # ------------------------------------------------------------------
    # Per-seed detail (for reproducibility)
    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("PER-SEED DETAIL")
    log("=" * 78)
    log(f"{'D':>4} | {'Model':>8} | {'Seed':>4} | {'Test Loss':>10} | "
        f"{'eps':>10} | {'Var[log||h||]':>14} | {'Mean[log||h||]':>15}")
    log("-" * 78)
    for row in all_rows:
        log(f"{row['dim']:>4} | {row['model']:>8} | {row['seed']:>4} | "
            f"{row['test_loss']:>10.4f} | {row['markov_score']:>10.6f} | "
            f"{row['var_log_norm']:>14.6f} | {row['mean_log_norm']:>15.6f}")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("INTERPRETATION")
    log("=" * 78)

    if r_squared > 0.7:
        log(f"  R^2 = {r_squared:.3f} -- STRONG fit.")
        log(f"  The norm variance explains most of the Markov violation.")
        log(f"  The bound epsilon <= {slope_c:.4f} * Var[log||h||] / D "
            f"is empirically supported.")
    elif r_squared > 0.3:
        log(f"  R^2 = {r_squared:.3f} -- MODERATE fit.")
        log(f"  Norm variance partially explains the Markov violation,")
        log(f"  but other factors (e.g. directional history) also contribute.")
    else:
        log(f"  R^2 = {r_squared:.3f} -- WEAK fit.")
        log(f"  Norm variance alone does not explain the Markov violation.")
        log(f"  The bound may need additional terms or the relationship")
        log(f"  may be non-linear.")

    log("")
    log("  Comparison to Rajaraman et al. (theoretical):")
    log("  The Rajaraman bound gives epsilon <= O(Var[log||h||] / D) for")
    log("  hidden Markov models.  Our fitted C gives the empirical constant.")
    if slope_c > 0:
        log(f"  Empirical C = {slope_c:.4f} (positive, consistent with bound).")
    else:
        log(f"  Empirical C = {slope_c:.4f} (non-positive, bound is trivially")
        log(f"  satisfied or norm variance is anti-correlated with epsilon).")

    log("\n" + "=" * 78)
    log("DONE")
    log("=" * 78)


if __name__ == "__main__":
    main()
