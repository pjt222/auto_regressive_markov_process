"""
Quaternion Polyhedral Convergence Experiment (Issue #47)

Tests whether learned quaternions in a QuatBlock SSM spontaneously converge
toward elements of binary polyhedral subgroups (2T, 2O, 2I) during training.

At each ES optimizer iteration, extracts the learned quaternions from the
flat parameter vector and measures their geodesic distance to the nearest
element of each group.

Hypothesis: if the geometric inductive bias is effective, trained quaternions
should settle near polyhedral group elements (which represent the most
symmetric finite rotations in 3D).

Usage:
    cd framework && python3 quat_convergence.py
"""

import sys
import os
import numpy as np
import time

# Import group generators and distance functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
from test_binary_polyhedral_groups import (
    generate_binary_icosahedral_group,
    generate_binary_tetrahedral_group,
    generate_binary_octahedral_group,
    nearest_group_element,
    quat_normalize,
)

from toy_example import generate_markov_source


def log(msg, **kwargs):
    """Print with immediate flush so output is visible in piped/redirected runs."""
    print(msg, flush=True, **kwargs)


# ============================================================================
# 1. Fast vectorized QuatBlock SSM (avoid per-step Python overhead)
# ============================================================================

def quaternion_from_bivector(alpha, beta, gamma):
    """Construct unit quaternion from bivector components."""
    norm_b = np.sqrt(alpha**2 + beta**2 + gamma**2)
    if norm_b < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half_angle = norm_b / 2
    cos_ha = np.cos(half_angle)
    sin_ha = np.sin(half_angle)
    w = cos_ha
    x = sin_ha * gamma / norm_b
    y = -sin_ha * beta / norm_b
    z = sin_ha * alpha / norm_b
    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix_3x3(q):
    """Convert unit quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def precompute_token_matrices(params, dim, n_blocks):
    """Precompute the block-diagonal rotation matrix + decay + input_scale
    for each token from the flat parameter vector.

    Returns for each token: (rotation_matrix_DxD, decay_scalar, input_scale_D)
    """
    n_bivector = n_blocks * 3
    n_params_per_token = n_bivector + 1 + dim
    token_data = []

    for token_id in range(2):
        base = token_id * n_params_per_token
        bivector_params = params[base:base + n_bivector]
        decay_raw = params[base + n_bivector]
        input_scale = params[base + n_bivector + 1:base + n_params_per_token]

        # Build block-diagonal rotation matrix
        rotation_matrix = np.eye(dim)
        for block_idx in range(n_blocks):
            offset = block_idx * 3
            alpha = bivector_params[offset]
            beta = bivector_params[offset + 1]
            gamma = bivector_params[offset + 2]
            q = quaternion_from_bivector(alpha, beta, gamma)
            R_block = quaternion_to_rotation_matrix_3x3(q)
            d_start = block_idx * 3
            rotation_matrix[d_start:d_start+3, d_start:d_start+3] = R_block

        decay = sigmoid(decay_raw)
        token_data.append((rotation_matrix, decay, input_scale))

    return token_data


def fast_loss(params, sequence, embeddings, dim, n_blocks):
    """Compute BCE loss using precomputed rotation matrices.

    Much faster than the per-step Python loop because we precompute
    the rotation matrices once and use matrix-vector multiply.
    """
    token_data = precompute_token_matrices(params, dim, n_blocks)
    n_steps = len(sequence)
    h_t = np.zeros(dim)
    h_t[min(2, dim - 1)] = 1.0

    logits = np.zeros(n_steps)
    for t in range(n_steps):
        x_t = sequence[t]
        rot_mat, decay, input_scale = token_data[x_t]
        rotated = rot_mat @ h_t
        h_t = decay * rotated + input_scale * embeddings[t]
        norm = np.linalg.norm(h_t)
        if norm > 1e-10:
            h_t = h_t / norm
        logits[t] = h_t[0]

    probs = sigmoid(logits * 5.0)
    targets = sequence[1:]
    pred_probs = probs[:-1]
    eps = 1e-7
    bce = -(targets * np.log(pred_probs + eps) +
            (1 - targets) * np.log(1 - pred_probs + eps))
    return np.mean(bce)


# ============================================================================
# 2. Quaternion extraction from flat parameter vector
# ============================================================================

def extract_quaternions(params, n_blocks, n_params_per_token, n_tokens=2):
    """Extract all quaternions from a flat parameter vector.

    Returns: array of shape (n_tokens * n_blocks, 4), list of (token, block) labels.
    """
    quaternions = []
    labels = []

    for token_id in range(n_tokens):
        base = token_id * n_params_per_token

        for block_idx in range(n_blocks):
            param_offset = base + block_idx * 3
            alpha = params[param_offset]
            beta = params[param_offset + 1]
            gamma = params[param_offset + 2]

            q = quaternion_from_bivector(alpha, beta, gamma)
            quaternions.append(q)
            labels.append((token_id, block_idx))

    return np.array(quaternions), labels


# ============================================================================
# 3. Distance computation to polyhedral groups
# ============================================================================

def compute_group_distances(quaternions, groups):
    """Compute distance from each quaternion to nearest element of each group.

    Returns: dict mapping group name to array of distances (n_quats,).
    """
    distances = {}
    for group_name, group_elements in groups.items():
        group_dists = []
        for q in quaternions:
            _, dist = nearest_group_element(q, group_elements)
            group_dists.append(dist)
        distances[group_name] = np.array(group_dists)
    return distances


# ============================================================================
# 4. Random baseline
# ============================================================================

def random_baseline(n_samples, groups, seed=42):
    """Compute mean distance from random unit quaternions to each group.

    Returns: dict mapping group name to (mean_distance, std_distance).
    """
    rng = np.random.default_rng(seed)
    random_quats = np.array([
        quat_normalize(rng.standard_normal(4)) for _ in range(n_samples)
    ])

    baseline = {}
    for group_name, group_elements in groups.items():
        dists = []
        for q in random_quats:
            _, dist = nearest_group_element(q, group_elements)
            dists.append(dist)
        dists = np.array(dists)
        baseline[group_name] = (np.mean(dists), np.std(dists))

    return baseline


# ============================================================================
# 5. ES optimizer with distance tracking
# ============================================================================

def optimize_with_tracking(sequence, embeddings, dim, n_blocks, groups,
                           n_iters=150, pop_size=20, sigma=0.1,
                           seed=0, label=""):
    """(1+lambda)-ES optimizer that logs quaternion-to-group distances
    at each iteration using the fast vectorized loss.

    Returns:
        best_params, losses, distance_trajectories, labels
    """
    rng = np.random.default_rng(seed)

    # Initialize params (same logic as QuatBlockSpinorDecaySSM.__init__)
    n_bivector = n_blocks * 3
    n_params_per_token = n_bivector + 1 + dim
    n_total_params = 2 * n_params_per_token
    best_params = rng.normal(0, 0.3, size=n_total_params)
    # Initialize decay to ~0.8
    for tok in range(2):
        base = tok * n_params_per_token
        best_params[base + n_bivector] = 1.4

    best_loss = fast_loss(best_params, sequence, embeddings, dim, n_blocks)
    losses = [best_loss]

    # Extract initial quaternions and compute distances
    quats, labels = extract_quaternions(best_params, n_blocks, n_params_per_token)
    n_quats = len(quats)

    distance_trajectories = {
        group_name: np.zeros((n_iters + 1, n_quats))
        for group_name in groups
    }

    initial_distances = compute_group_distances(quats, groups)
    for group_name in groups:
        distance_trajectories[group_name][0] = initial_distances[group_name]

    for iteration in range(n_iters):
        noise = rng.normal(0, sigma, size=(pop_size, n_total_params))
        candidates = best_params[None, :] + noise
        candidate_losses = np.array([
            fast_loss(c, sequence, embeddings, dim, n_blocks) for c in candidates
        ])
        best_idx = np.argmin(candidate_losses)
        if candidate_losses[best_idx] < best_loss:
            best_loss = candidate_losses[best_idx]
            best_params = candidates[best_idx].copy()
        losses.append(best_loss)

        # Decay sigma periodically
        if iteration % 50 == 49:
            sigma *= 0.7

        # Extract quaternions and compute distances
        quats, _ = extract_quaternions(best_params, n_blocks, n_params_per_token)
        iter_distances = compute_group_distances(quats, groups)
        for group_name in groups:
            distance_trajectories[group_name][iteration + 1] = iter_distances[group_name]

        if iteration % 25 == 24:
            mean_dists = {gn: np.mean(distance_trajectories[gn][iteration + 1])
                          for gn in groups}
            dist_str = "  ".join(f"{gn}={d:.4f}" for gn, d in mean_dists.items())
            log(f"    {label} iter {iteration+1}/{n_iters}  loss={best_loss:.4f}  {dist_str}")

    return best_params, losses, distance_trajectories, labels


# ============================================================================
# 6. Embed tokens (inline, avoids import complexity)
# ============================================================================

def embed_tokens_general(sequence, dim):
    """Map binary tokens to D-dimensional embeddings."""
    n_steps = len(sequence)
    embeddings = np.zeros((n_steps, dim))
    embeddings[sequence == 0, 0] = 1.0
    embeddings[sequence == 1, 1] = 1.0
    return embeddings


# ============================================================================
# 7. Main experiment
# ============================================================================

def main():
    log("=" * 70)
    log("Quaternion Polyhedral Convergence Experiment (Issue #47)")
    log("Do learned quaternions converge toward 2T/2O/2I subgroups?")
    log("=" * 70)

    # Configuration - tuned for reasonable runtime
    dim = 9
    n_train = 2000
    n_test = 2000
    n_iters = 150
    n_seeds = 3
    pop_size = 20
    sigma = 0.1
    n_baseline_samples = 1000

    n_blocks = dim // 3
    n_bivector = n_blocks * 3
    n_params_per_token = n_bivector + 1 + dim
    n_quats_per_model = n_blocks * 2  # 2 tokens

    log(f"\nConfig: D={dim}, n_blocks={n_blocks}, n_quats_per_model={n_quats_per_model}")
    log(f"        n_params_per_token={n_params_per_token}, total_params={2*n_params_per_token}")
    log(f"        n_train={n_train}, n_test={n_test}, n_iters={n_iters}, n_seeds={n_seeds}")
    log(f"        pop_size={pop_size}, sigma={sigma}")

    # Generate polyhedral groups
    log("\nGenerating binary polyhedral groups...")
    t0 = time.time()
    group_2T = generate_binary_tetrahedral_group()
    group_2O = generate_binary_octahedral_group()
    group_2I = generate_binary_icosahedral_group()
    log(f"  2T: {len(group_2T)} elements, 2O: {len(group_2O)} elements, "
        f"2I: {len(group_2I)} elements ({time.time() - t0:.1f}s)")

    groups = {"2T": group_2T, "2O": group_2O, "2I": group_2I}

    # Random baseline
    log(f"\nComputing random baseline ({n_baseline_samples} random unit quaternions)...")
    baseline = random_baseline(n_baseline_samples, groups)
    log(f"  Random baseline distances (mean +/- std):")
    for group_name in ["2T", "2O", "2I"]:
        mean_d, std_d = baseline[group_name]
        log(f"    {group_name}: {mean_d:.4f} +/- {std_d:.4f}")

    # Run training across seeds
    all_trajectories = []
    all_losses = []
    all_test_losses = []

    for seed in range(n_seeds):
        log(f"\n{'='*70}")
        log(f"Seed {seed+1}/{n_seeds}")
        log(f"{'='*70}")

        train_seq = generate_markov_source(n_train, order=2, seed=seed)
        test_seq = generate_markov_source(n_test, order=2, seed=seed + 100)
        train_emb = embed_tokens_general(train_seq, dim=dim)
        test_emb = embed_tokens_general(test_seq, dim=dim)

        t0 = time.time()
        best_params, losses, distance_trajectories, labels = optimize_with_tracking(
            train_seq, train_emb, dim, n_blocks, groups,
            n_iters=n_iters, pop_size=pop_size, sigma=sigma,
            seed=seed, label=f"s{seed}"
        )
        elapsed = time.time() - t0

        test_loss = fast_loss(best_params, test_seq, test_emb, dim, n_blocks)
        log(f"  Done: train loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
            f"test loss {test_loss:.4f} ({elapsed:.1f}s)")

        all_trajectories.append((distance_trajectories, labels))
        all_losses.append(losses)
        all_test_losses.append(test_loss)

    # ========================================================================
    # Analysis
    # ========================================================================

    log("\n" + "=" * 70)
    log("RESULTS: Quaternion Distance to Polyhedral Groups")
    log("=" * 70)

    # Per-seed initial vs final distances
    log(f"\n{'Seed':>4} {'Quat':>6} {'Token':>5} {'Block':>5} | "
        f"{'2T init':>8} {'2T fin':>8} {'delta':>8} | "
        f"{'2O init':>8} {'2O fin':>8} {'delta':>8} | "
        f"{'2I init':>8} {'2I fin':>8} {'delta':>8}")
    log("-" * 115)

    for seed in range(n_seeds):
        distance_trajectories, labels = all_trajectories[seed]
        for q_idx, (token_id, block_idx) in enumerate(labels):
            row_parts = [f"{seed:>4} {q_idx:>6} {token_id:>5} {block_idx:>5} |"]
            for group_name in ["2T", "2O", "2I"]:
                initial_dist = distance_trajectories[group_name][0, q_idx]
                final_dist = distance_trajectories[group_name][-1, q_idx]
                delta = final_dist - initial_dist
                row_parts.append(
                    f" {initial_dist:>7.4f}  {final_dist:>7.4f}  {delta:>+7.4f} |"
                )
            log("".join(row_parts))

    # Aggregate statistics
    log(f"\n{'='*70}")
    log("AGGREGATE: Mean distance across all quaternions and seeds")
    log(f"{'='*70}")

    for group_name in ["2T", "2O", "2I"]:
        all_initial = []
        all_final = []
        for seed in range(n_seeds):
            dt = all_trajectories[seed][0]
            all_initial.extend(dt[group_name][0])
            all_final.extend(dt[group_name][-1])
        all_initial = np.array(all_initial)
        all_final = np.array(all_final)
        baseline_mean, baseline_std = baseline[group_name]

        log(f"\n  {group_name} ({len(groups[group_name])} elements):")
        log(f"    Random baseline:  {baseline_mean:.4f} +/- {baseline_std:.4f}")
        log(f"    Initial (iter 0): {np.mean(all_initial):.4f} +/- {np.std(all_initial):.4f}")
        log(f"    Final (iter {n_iters}): {np.mean(all_final):.4f} +/- {np.std(all_final):.4f}")
        log(f"    Delta (final-init): {np.mean(all_final) - np.mean(all_initial):+.4f}")
        log(f"    vs random baseline: {np.mean(all_final) - baseline_mean:+.4f}")

    # Distance trajectory summary
    log(f"\n{'='*70}")
    log("TRAJECTORY: Mean distance over training (all quats, all seeds)")
    log(f"{'='*70}")

    checkpoints = [0, n_iters // 4, n_iters // 2, 3 * n_iters // 4, n_iters]
    log(f"{'Iter':>6} | {'2T':>8} | {'2O':>8} | {'2I':>8}")
    log("-" * 40)
    for cp in checkpoints:
        means = {}
        for group_name in ["2T", "2O", "2I"]:
            dists_at_cp = []
            for seed in range(n_seeds):
                dt = all_trajectories[seed][0]
                dists_at_cp.extend(dt[group_name][cp])
            means[group_name] = np.mean(dists_at_cp)
        log(f"{cp:>6} | {means['2T']:>8.4f} | {means['2O']:>8.4f} | {means['2I']:>8.4f}")

    # Trend analysis: linear regression
    log(f"\n{'='*70}")
    log("TREND: Linear regression of mean distance over iterations")
    log(f"{'='*70}")

    iterations = np.arange(n_iters + 1)
    for group_name in ["2T", "2O", "2I"]:
        mean_per_iter = np.zeros(n_iters + 1)
        for iteration_idx in range(n_iters + 1):
            dists = []
            for seed in range(n_seeds):
                dt = all_trajectories[seed][0]
                dists.extend(dt[group_name][iteration_idx])
            mean_per_iter[iteration_idx] = np.mean(dists)

        slope, intercept = np.polyfit(iterations, mean_per_iter, 1)
        log(f"  {group_name}: slope = {slope:+.6f}/iter  "
            f"(total drift = {slope * n_iters:+.4f} over {n_iters} iters)")
        if slope < -1e-5:
            log(f"         -> CONVERGING toward {group_name} (distance decreasing)")
        elif slope > 1e-5:
            log(f"         -> DIVERGING from {group_name} (distance increasing)")
        else:
            log(f"         -> STABLE (no significant trend)")

    # Per-token analysis
    log(f"\n{'='*70}")
    log("PER-TOKEN ANALYSIS: Do token 0 and token 1 behave differently?")
    log(f"{'='*70}")

    for token_id in [0, 1]:
        log(f"\n  Token {token_id}:")
        for group_name in ["2T", "2O", "2I"]:
            initial_dists = []
            final_dists = []
            for seed in range(n_seeds):
                dt, labels = all_trajectories[seed]
                for q_idx, (tid, bid) in enumerate(labels):
                    if tid == token_id:
                        initial_dists.append(dt[group_name][0, q_idx])
                        final_dists.append(dt[group_name][-1, q_idx])
            initial_dists = np.array(initial_dists)
            final_dists = np.array(final_dists)
            log(f"    {group_name}: {np.mean(initial_dists):.4f} -> {np.mean(final_dists):.4f} "
                f"(delta = {np.mean(final_dists) - np.mean(initial_dists):+.4f})")

    # Final interpretation
    log(f"\n{'='*70}")
    log("INTERPRETATION")
    log(f"{'='*70}")

    for group_name in ["2T", "2O", "2I"]:
        all_final = []
        for seed in range(n_seeds):
            dt = all_trajectories[seed][0]
            all_final.extend(dt[group_name][-1])
        mean_final = np.mean(all_final)
        baseline_mean, baseline_std = baseline[group_name]

        if mean_final < baseline_mean - baseline_std:
            log(f"  ** {group_name}: Final distance ({mean_final:.4f}) is MORE than 1 sigma "
                f"below random baseline ({baseline_mean:.4f} +/- {baseline_std:.4f})")
            log(f"     -> SIGNIFICANT convergence toward {group_name}")
        elif mean_final < baseline_mean:
            log(f"  {group_name}: Final distance ({mean_final:.4f}) is below random baseline "
                f"({baseline_mean:.4f}) but within 1 sigma")
            log(f"     -> Weak/marginal convergence toward {group_name}")
        else:
            log(f"  {group_name}: Final distance ({mean_final:.4f}) is at or above random "
                f"baseline ({baseline_mean:.4f})")
            log(f"     -> No convergence toward {group_name}")

    # Overall conclusion
    any_converging = False
    for group_name in ["2T", "2O", "2I"]:
        all_final = []
        for seed in range(n_seeds):
            dt = all_trajectories[seed][0]
            all_final.extend(dt[group_name][-1])
        mean_final = np.mean(all_final)
        baseline_mean, baseline_std = baseline[group_name]
        if mean_final < baseline_mean - baseline_std:
            any_converging = True

    log(f"\n  Overall: ", end="")
    if any_converging:
        log("SOME polyhedral convergence detected. Learned quaternions are "
            "closer to discrete subgroup elements than random.")
    else:
        log("No significant polyhedral convergence detected at this scale. "
            "Learned quaternions remain near random baseline distance from "
            "2T/2O/2I elements.")

    log(f"\n  Test losses: {['seed ' + str(s) + '=' + f'{l:.4f}' for s, l in enumerate(all_test_losses)]}")
    log(f"  Mean test loss: {np.mean(all_test_losses):.4f}")


if __name__ == "__main__":
    main()
