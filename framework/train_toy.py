"""
Training Experiment: Spinor+Decay SSM vs Diagonal SSM on 2nd-order Markov Source

Trains both models to predict next tokens from a 2nd-order Markov chain,
then compares Markov scores. Tests whether training resolves the
rotation-forgetting tension (issue #22, #30).

Usage:
    python framework/train_toy.py

Requires: numpy (no PyTorch needed for this minimal version)
"""

import numpy as np
from toy_example import (
    generate_markov_source,
    embed_tokens,
    quaternion_from_bivector,
    quaternion_rotate,
    normalize,
    compute_markov_score,
)


# ============================================================================
# 1. Trainable Spinor+Decay SSM (gradient-free optimization for simplicity)
# ============================================================================

class TrainableSpinorDecaySSM:
    """Spinor+Decay SSM with trainable parameters via CMA-ES-like optimization.

    Uses numerical gradient estimation (finite differences) to keep the
    implementation simple and dependency-free.
    """

    def __init__(self, dim: int = 3, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # Pack all parameters into a flat vector for optimization
        # Per token (2 tokens): 3 bivector + 1 decay + 3 input_scale = 7
        # Total: 14 parameters
        self.n_params_per_token = 7
        self.n_tokens = 2
        self.params = self.rng.normal(0, 0.3, size=self.n_tokens * self.n_params_per_token)
        # Initialize decay raw to ~0.8 (sigmoid(1.4) ≈ 0.8)
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base + 3] = 1.4  # decay_raw

    def _unpack(self, params, token_id):
        """Extract per-token parameters from flat vector."""
        base = token_id * self.n_params_per_token
        bivector = params[base : base + 3]
        decay_raw = params[base + 3]
        input_scale = params[base + 4 : base + 7]
        return bivector, decay_raw, input_scale

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def run_and_predict(self, sequence, embeddings, params=None):
        """Run model, return predicted next-token log-probs."""
        if params is None:
            params = self.params

        n_steps = len(sequence)
        states = np.zeros((n_steps, self.dim))
        h_t = np.array([0.0, 0.0, 1.0])

        for t in range(n_steps):
            x_t = sequence[t]
            bivector, decay_raw, input_scale = self._unpack(params, x_t)

            # Rotation
            u_t = quaternion_from_bivector(*bivector)
            rotated = quaternion_rotate(u_t, h_t)

            # Decay + inject
            lambda_t = self._sigmoid(decay_raw)
            h_t = lambda_t * rotated + input_scale * embeddings[t]

            states[t] = normalize(h_t)

        # Simple linear readout: predict next token from state
        # P(x_{t+1} = 1 | s_t) = sigmoid(w · s_t + b)
        # We include w and b in the optimization implicitly via the state
        # For simplicity: use the first component of s_t as logit
        logits = states[:, 0]  # use first coordinate as decision boundary
        probs = self._sigmoid(logits * 5.0)  # scale for sharper predictions

        return states, probs

    def loss(self, sequence, embeddings, params=None):
        """Cross-entropy loss for next-token prediction."""
        _, probs = self.run_and_predict(sequence, embeddings, params)

        # Next-token prediction: predict sequence[t+1] from state at t
        targets = sequence[1:]
        pred_probs = probs[:-1]

        # Binary cross-entropy
        eps = 1e-7
        bce = -(
            targets * np.log(pred_probs + eps)
            + (1 - targets) * np.log(1 - pred_probs + eps)
        )
        return np.mean(bce)


class TrainableDiagonalSSM:
    """Diagonal SSM with same parameterization for fair comparison."""

    def __init__(self, dim: int = 3, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # Per token: 3 lambda_raw + 3 input_scale = 6
        # Total: 12 parameters
        self.n_params_per_token = 6
        self.n_tokens = 2
        self.params = self.rng.normal(0, 0.3, size=self.n_tokens * self.n_params_per_token)
        # Initialize lambda_raw to ~0.8
        for tok in range(self.n_tokens):
            base = tok * self.n_params_per_token
            self.params[base : base + 3] = 1.4

    def _unpack(self, params, token_id):
        base = token_id * self.n_params_per_token
        lambda_raw = params[base : base + 3]
        input_scale = params[base + 3 : base + 6]
        return lambda_raw, input_scale

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def run_and_predict(self, sequence, embeddings, params=None):
        if params is None:
            params = self.params

        n_steps = len(sequence)
        states = np.zeros((n_steps, self.dim))
        h_t = np.array([0.0, 0.0, 1.0])

        for t in range(n_steps):
            x_t = sequence[t]
            lambda_raw, input_scale = self._unpack(params, x_t)
            lambdas = self._sigmoid(lambda_raw)
            h_t = lambdas * h_t + input_scale * embeddings[t]
            states[t] = normalize(h_t)

        logits = states[:, 0]
        probs = self._sigmoid(logits * 5.0)
        return states, probs

    def loss(self, sequence, embeddings, params=None):
        _, probs = self.run_and_predict(sequence, embeddings, params)
        targets = sequence[1:]
        pred_probs = probs[:-1]
        eps = 1e-7
        bce = -(
            targets * np.log(pred_probs + eps)
            + (1 - targets) * np.log(1 - pred_probs + eps)
        )
        return np.mean(bce)


# ============================================================================
# 2. Simple evolutionary optimization (gradient-free)
# ============================================================================

def optimize_model(model, sequence, embeddings, n_iters=200, pop_size=20, sigma=0.1, seed=0):
    """Simple (1+λ)-ES optimization of model parameters."""
    rng = np.random.default_rng(seed)

    best_params = model.params.copy()
    best_loss = model.loss(sequence, embeddings, best_params)

    losses = [best_loss]

    for iteration in range(n_iters):
        # Generate population of perturbations
        noise = rng.normal(0, sigma, size=(pop_size, len(best_params)))
        candidates = best_params[None, :] + noise

        # Evaluate all candidates
        candidate_losses = np.array([
            model.loss(sequence, embeddings, c) for c in candidates
        ])

        # Select best
        best_idx = np.argmin(candidate_losses)
        if candidate_losses[best_idx] < best_loss:
            best_loss = candidate_losses[best_idx]
            best_params = candidates[best_idx].copy()

        losses.append(best_loss)

        # Adaptive sigma
        if iteration % 50 == 49:
            sigma *= 0.7  # reduce exploration over time

    model.params = best_params
    return losses


# ============================================================================
# 3. Main training experiment
# ============================================================================

def main():
    print("=" * 70)
    print("Training Experiment: Spinor+Decay SSM vs Diagonal SSM")
    print("Issue #30: Does training resolve the rotation-forgetting tension?")
    print("=" * 70)

    # Parameters
    n_train = 5000
    n_test = 5000
    dim = 3
    source_order = 2
    n_iters = 300
    max_lag = 5
    n_seeds = 3

    print(f"\nConfig: D={dim}, source_order={source_order}")
    print(f"        n_train={n_train}, n_test={n_test}, n_iters={n_iters}")
    print(f"        n_seeds={n_seeds}")

    results = {"spinor_decay": [], "diagonal": []}
    losses = {"spinor_decay": [], "diagonal": []}

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")

        # Generate data
        train_seq = generate_markov_source(n_train, order=source_order, seed=seed)
        test_seq = generate_markov_source(n_test, order=source_order, seed=seed + 100)
        train_emb = embed_tokens(train_seq, dim=dim)
        test_emb = embed_tokens(test_seq, dim=dim)

        # Train Spinor+Decay
        print("  Training Spinor+Decay SSM...", end=" ", flush=True)
        spinor_model = TrainableSpinorDecaySSM(dim=dim, seed=seed)
        spinor_losses = optimize_model(
            spinor_model, train_seq, train_emb, n_iters=n_iters, seed=seed
        )
        print(f"loss: {spinor_losses[0]:.4f} -> {spinor_losses[-1]:.4f}")

        # Train Diagonal
        print("  Training Diagonal SSM...", end=" ", flush=True)
        diag_model = TrainableDiagonalSSM(dim=dim, seed=seed)
        diag_losses = optimize_model(
            diag_model, train_seq, train_emb, n_iters=n_iters, seed=seed
        )
        print(f"loss: {diag_losses[0]:.4f} -> {diag_losses[-1]:.4f}")

        # Evaluate on test set
        spinor_states, spinor_probs = spinor_model.run_and_predict(test_seq, test_emb)
        diag_states, diag_probs = diag_model.run_and_predict(test_seq, test_emb)

        # Compute Markov scores on test states
        spinor_scores = compute_markov_score(spinor_states, max_lag=max_lag)
        diag_scores = compute_markov_score(diag_states, max_lag=max_lag)

        results["spinor_decay"].append(spinor_scores)
        results["diagonal"].append(diag_scores)
        losses["spinor_decay"].append(spinor_losses[-1])
        losses["diagonal"].append(diag_losses[-1])

    # Aggregate results
    spinor_mean = np.nanmean(results["spinor_decay"], axis=0)
    diag_mean = np.nanmean(results["diagonal"], axis=0)
    spinor_std = np.nanstd(results["spinor_decay"], axis=0)
    diag_std = np.nanstd(results["diagonal"], axis=0)

    print("\n" + "=" * 70)
    print("RESULTS (TRAINED models)")
    print("=" * 70)

    print(f"\nFinal training losses:")
    print(f"  Spinor+Decay: {np.mean(losses['spinor_decay']):.4f} +/- {np.std(losses['spinor_decay']):.4f}")
    print(f"  Diagonal:     {np.mean(losses['diagonal']):.4f} +/- {np.std(losses['diagonal']):.4f}")

    print(f"\nMarkov Score (TRAINED): I(s_{{t+1}}; s_{{t-j}} | s_t)")
    print("-" * 60)
    print(f"{'Lag j':>6} | {'Spinor+Decay':>18} | {'Diagonal SSM':>18}")
    print("-" * 60)
    for j in range(max_lag):
        print(
            f"{j + 1:>6} | "
            f"{spinor_mean[j]:>7.4f} +/- {spinor_std[j]:.4f} | "
            f"{diag_mean[j]:>7.4f} +/- {diag_std[j]:.4f}"
        )

    ms = np.nanmean(spinor_mean)
    md = np.nanmean(diag_mean)
    print("-" * 60)
    print(f"{'Mean':>6} | {ms:>7.4f}{'':>14} | {md:>7.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON: Untrained vs Trained")
    print("=" * 70)
    print(f"  Untrained Spinor+Decay: 0.3622 (from toy_example.py)")
    print(f"  Trained Spinor+Decay:   {ms:.4f}")
    print(f"  Untrained Diagonal:     0.0178")
    print(f"  Trained Diagonal:       {md:.4f}")

    if ms < md:
        print("\n>> TRAINED Spinor+Decay has LOWER Markov score!")
        print(">> Training resolves the rotation-forgetting tension!")
        print(">> CONSISTENT with conjecture")
    elif ms < 0.3622:
        print(f"\n>> Training IMPROVED Spinor+Decay: 0.3622 -> {ms:.4f}")
        print(f">> But Diagonal still better: {md:.4f}")
        print(">> PARTIAL support for conjecture")
    else:
        print(f"\n>> Training did NOT improve Spinor+Decay Markov score")
        print(">> Inconsistent with conjecture at this scale")


if __name__ == "__main__":
    main()
