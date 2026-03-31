"""
Toy Example: 3D Spinor SSM vs Diagonal SSM on a 2nd-order Markov Source

Tests the Approximate Markov Property conjecture (framework/formalization.md)
at minimal scale: D=3, k=2, state space S^2.

Usage:
    python framework/toy_example.py

Output:
    - Markov scores for spinor SSM vs diagonal SSM
    - Trajectory visualization data
"""

import numpy as np
from typing import Tuple

# ============================================================================
# 1. Second-order Markov source
# ============================================================================

def generate_markov_source(
    n_steps: int,
    order: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """Generate a k-th order binary Markov chain.

    For order=2, P(x_t | x_{t-1}, x_{t-2}) depends on the last 2 symbols.
    Transition probabilities are chosen to make the chain non-trivial.
    """
    rng = np.random.default_rng(seed)

    # Transition table: P(x_t=1 | x_{t-2}, x_{t-1})
    # Index: (x_{t-2}, x_{t-1}) -> P(x_t=1)
    if order == 2:
        transition_probs = {
            (0, 0): 0.8,  # after 00, likely switch to 1
            (0, 1): 0.3,  # after 01, likely stay at 1 -> 0
            (1, 0): 0.7,  # after 10, likely switch to 1
            (1, 1): 0.2,  # after 11, likely switch to 0
        }
    elif order == 1:
        transition_probs = {
            (0,): 0.7,
            (1,): 0.3,
        }
    else:
        raise ValueError(f"Order {order} not implemented")

    # Generate sequence
    sequence = np.zeros(n_steps, dtype=int)
    # Random initial states
    for i in range(order):
        sequence[i] = rng.integers(0, 2)

    for t in range(order, n_steps):
        context = tuple(sequence[t - order : t])
        p_one = transition_probs[context]
        sequence[t] = rng.binomial(1, p_one)

    return sequence


# ============================================================================
# 2. Token embeddings
# ============================================================================

def embed_tokens(sequence: np.ndarray, dim: int = 3) -> np.ndarray:
    """Map binary tokens to D-dimensional embedding vectors."""
    embeddings = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0]),
    }
    if dim != 3:
        raise ValueError("Only D=3 implemented for toy example")
    return np.array([embeddings[x] for x in sequence])


# ============================================================================
# 3. Quaternion operations (Spin(3) ≅ SU(2))
# ============================================================================

def quaternion_from_bivector(alpha: float, beta: float, gamma: float) -> np.ndarray:
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

    # Quaternion components from bivector
    # e12 -> k, e13 -> -j, e23 -> i (standard Cl(3,0) convention)
    w = cos_ha
    x = sin_ha * gamma / norm_b    # e23 component
    y = -sin_ha * beta / norm_b    # e13 component (sign from convention)
    z = sin_ha * alpha / norm_b    # e12 component

    return np.array([w, x, y, z])


def quaternion_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply quaternion rotation to a 3D vector: v' = q * v * q^{-1}.

    For unit quaternion q = [w, x, y, z], this is the sandwich product.
    """
    w, qx, qy, qz = q
    vx, vy, vz = v

    # Quaternion-vector-quaternion product (Rodrigues' rotation formula)
    t = 2.0 * np.array([
        qy * vz - qz * vy,
        qz * vx - qx * vz,
        qx * vy - qy * vx,
    ])
    return v + w * t + np.cross(np.array([qx, qy, qz]), t)


def normalize(v: np.ndarray) -> np.ndarray:
    """Project onto S^{D-1}: v -> v / ||v||"""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


# ============================================================================
# 4. Spinor SSM model
# ============================================================================

class SpinorSSM:
    """Selective SSM with Spin(3) transition (sandwich product).

    Transition: h_{t+1} = u_t h_t u_t^{-1} + B_bar_t x_t
    where u_t = exp(B_t/2) in Spin(3), B_t depends on x_t.
    """

    def __init__(self, dim: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim = dim

        # Bivector parameters: alpha, beta, gamma per input token
        # B_t(x) = alpha(x) * e12 + beta(x) * e13 + gamma(x) * e23
        self.bivector_params = {
            0: rng.normal(0, 0.5, size=3),  # for token 0
            1: rng.normal(0, 0.5, size=3),  # for token 1
        }

        # Input injection: B_bar per token
        self.input_scale = {
            0: rng.normal(0, 0.3, size=dim),
            1: rng.normal(0, 0.3, size=dim),
        }

    def step(self, h_t: np.ndarray, x_t: int, x_embed: np.ndarray) -> np.ndarray:
        """One transition step."""
        # Compute spinor from input-dependent bivector
        alpha, beta, gamma = self.bivector_params[x_t]
        u_t = quaternion_from_bivector(alpha, beta, gamma)

        # Sandwich product: rotate previous state
        rotated = quaternion_rotate(u_t, h_t)

        # Input injection
        injected = self.input_scale[x_t] * x_embed

        # New raw state
        h_new = rotated + injected
        return h_new

    def run(self, sequence: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Run the model on a full sequence. Returns states on S^2."""
        n_steps = len(sequence)
        states = np.zeros((n_steps, self.dim))

        h_t = np.array([0.0, 0.0, 1.0])  # initial state (north pole)

        for t in range(n_steps):
            h_t = self.step(h_t, sequence[t], embeddings[t])
            states[t] = normalize(h_t)

        return states


# ============================================================================
# 4b. Spinor SSM with scalar decay (Resolution 1 from issue #22)
# ============================================================================

class SpinorDecaySSM:
    """Selective SSM with Spin(3) transition AND scalar decay.

    Transition: h_{t+1} = lambda_t * u_t h_t u_t^{-1} + B_bar_t x_t
    where u_t = exp(B_t/2) in Spin(3), lambda_t in (0,1), both depend on x_t.

    This combines geometric structure (rotation) with forgetting (decay).
    """

    def __init__(self, dim: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim = dim

        # Bivector parameters: alpha, beta, gamma per input token
        self.bivector_params = {
            0: rng.normal(0, 0.5, size=3),
            1: rng.normal(0, 0.5, size=3),
        }

        # Scalar decay per token: lambda_t = sigmoid(raw) in (0, 1)
        self.decay_raw = {
            0: rng.normal(1.5, 0.3),   # sigmoid(1.5) ≈ 0.82
            1: rng.normal(1.5, 0.3),
        }

        # Input injection
        self.input_scale = {
            0: rng.normal(0, 0.3, size=dim),
            1: rng.normal(0, 0.3, size=dim),
        }

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def step(self, h_t: np.ndarray, x_t: int, x_embed: np.ndarray) -> np.ndarray:
        """One transition step: lambda * rotate + inject."""
        # Rotation via sandwich product
        alpha, beta, gamma = self.bivector_params[x_t]
        u_t = quaternion_from_bivector(alpha, beta, gamma)
        rotated = quaternion_rotate(u_t, h_t)

        # Scalar decay
        lambda_t = self._sigmoid(self.decay_raw[x_t])
        decayed_rotated = lambda_t * rotated

        # Input injection
        injected = self.input_scale[x_t] * x_embed

        return decayed_rotated + injected

    def run(self, sequence: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Run the model on a full sequence. Returns states on S^2."""
        n_steps = len(sequence)
        states = np.zeros((n_steps, self.dim))

        h_t = np.array([0.0, 0.0, 1.0])

        for t in range(n_steps):
            h_t = self.step(h_t, sequence[t], embeddings[t])
            states[t] = normalize(h_t)

        return states


# ============================================================================
# 5. Diagonal SSM baseline (Mamba-style, no geometric structure)
# ============================================================================

class DiagonalSSM:
    """Standard diagonal SSM: h_{t+1} = diag(lambda) * h_t + B * x_t."""

    def __init__(self, dim: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim = dim

        # Diagonal transition (input-dependent decay)
        self.lambdas = {
            0: np.clip(rng.normal(0.8, 0.1, size=dim), 0.1, 0.99),
            1: np.clip(rng.normal(0.8, 0.1, size=dim), 0.1, 0.99),
        }

        # Input injection
        self.input_scale = {
            0: rng.normal(0, 0.3, size=dim),
            1: rng.normal(0, 0.3, size=dim),
        }

    def step(self, h_t: np.ndarray, x_t: int, x_embed: np.ndarray) -> np.ndarray:
        """One transition step."""
        decayed = self.lambdas[x_t] * h_t
        injected = self.input_scale[x_t] * x_embed
        return decayed + injected

    def run(self, sequence: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Run the model on a full sequence. Returns states on S^2."""
        n_steps = len(sequence)
        states = np.zeros((n_steps, self.dim))

        h_t = np.array([0.0, 0.0, 1.0])

        for t in range(n_steps):
            h_t = self.step(h_t, sequence[t], embeddings[t])
            states[t] = normalize(h_t)

        return states


# ============================================================================
# 6. Markov score computation
# ============================================================================

def compute_markov_score(
    states: np.ndarray,
    max_lag: int = 5,
    n_bins: int = 20,
) -> np.ndarray:
    """Estimate Markov score: I(s_{t+1}; s_{t-j} | s_t) for j=1,...,max_lag.

    Uses discretization of S^2 via spherical coordinates binning.
    Returns array of scores for each lag.
    """
    n_steps = len(states)

    # Discretize states into bins using spherical coordinates
    theta = np.arccos(np.clip(states[:, 2], -1, 1))  # polar angle
    phi = np.arctan2(states[:, 1], states[:, 0])       # azimuthal angle

    # Bin into grid
    theta_bins = np.digitize(theta, np.linspace(0, np.pi, n_bins + 1)) - 1
    phi_bins = np.digitize(phi, np.linspace(-np.pi, np.pi, n_bins + 1)) - 1
    discrete_states = theta_bins * n_bins + phi_bins

    scores = np.zeros(max_lag)

    for lag in range(1, max_lag + 1):
        # Compute I(s_{t+1}; s_{t-lag} | s_t) via histogram-based MI estimation
        # Using the identity: I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z) - H(Z|Z)
        # Simplified: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)

        valid_range = range(lag + 1, n_steps)
        if len(list(valid_range)) < 100:
            scores[lag - 1] = np.nan
            continue

        x = discrete_states[lag + 1 : n_steps]       # s_{t+1}
        y = discrete_states[1 : n_steps - lag]        # s_{t-lag}
        z = discrete_states[lag : n_steps - 1]        # s_t

        # Joint and marginal entropies via histogram counting
        h_xz = _joint_entropy(x, z)
        h_yz = _joint_entropy(y, z)
        h_xyz = _triple_entropy(x, y, z)
        h_z = _entropy(z)

        cmi = h_xz + h_yz - h_xyz - h_z
        scores[lag - 1] = max(0, cmi)  # CMI is non-negative; clip numerical errors

    return scores


def _entropy(x: np.ndarray) -> float:
    """Shannon entropy of discrete variable."""
    _, counts = np.unique(x, return_counts=True)
    probs = counts / len(x)
    return -np.sum(probs * np.log(probs + 1e-12))


def _joint_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """Joint entropy of two discrete variables."""
    pairs = np.column_stack([x, y])
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    probs = counts / len(x)
    return -np.sum(probs * np.log(probs + 1e-12))


def _triple_entropy(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Joint entropy of three discrete variables."""
    triples = np.column_stack([x, y, z])
    _, counts = np.unique(triples, axis=0, return_counts=True)
    probs = counts / len(x)
    return -np.sum(probs * np.log(probs + 1e-12))


# ============================================================================
# 7. Main experiment
# ============================================================================

def main():
    print("=" * 60)
    print("Toy Example: Spinor SSM vs Diagonal SSM")
    print("Testing Approximate Markov Property on S^2")
    print("=" * 60)

    # Parameters
    n_steps = 10000
    dim = 3
    source_order = 2
    max_lag = 5
    n_seeds = 5

    print(f"\nConfig: D={dim}, source_order={source_order}, n_steps={n_steps}")
    print(f"        max_lag={max_lag}, n_seeds={n_seeds}")

    spinor_scores_all = []
    spinor_decay_scores_all = []
    diagonal_scores_all = []

    for seed in range(n_seeds):
        # Generate source
        sequence = generate_markov_source(n_steps, order=source_order, seed=seed)
        embeddings = embed_tokens(sequence, dim=dim)

        # Run all three models
        spinor_model = SpinorSSM(dim=dim, seed=seed)
        spinor_decay_model = SpinorDecaySSM(dim=dim, seed=seed)
        diagonal_model = DiagonalSSM(dim=dim, seed=seed)

        spinor_states = spinor_model.run(sequence, embeddings)
        spinor_decay_states = spinor_decay_model.run(sequence, embeddings)
        diagonal_states = diagonal_model.run(sequence, embeddings)

        # Compute Markov scores
        spinor_scores = compute_markov_score(spinor_states, max_lag=max_lag)
        spinor_decay_scores = compute_markov_score(spinor_decay_states, max_lag=max_lag)
        diagonal_scores = compute_markov_score(diagonal_states, max_lag=max_lag)

        spinor_scores_all.append(spinor_scores)
        spinor_decay_scores_all.append(spinor_decay_scores)
        diagonal_scores_all.append(diagonal_scores)

    spinor_mean = np.nanmean(spinor_scores_all, axis=0)
    spinor_decay_mean = np.nanmean(spinor_decay_scores_all, axis=0)
    diagonal_mean = np.nanmean(diagonal_scores_all, axis=0)
    spinor_std = np.nanstd(spinor_scores_all, axis=0)
    spinor_decay_std = np.nanstd(spinor_decay_scores_all, axis=0)
    diagonal_std = np.nanstd(diagonal_scores_all, axis=0)

    # Report results
    print("\n" + "-" * 78)
    print("Markov Score: I(s_{t+1}; s_{t-j} | s_t)")
    print("Lower is more Markovian (0 = perfectly Markov)")
    print("-" * 78)
    print(f"{'Lag j':>6} | {'Spinor (no decay)':>18} | {'Spinor + Decay':>18} | {'Diagonal SSM':>16}")
    print("-" * 78)
    for j in range(max_lag):
        print(
            f"{j + 1:>6} | "
            f"{spinor_mean[j]:>7.4f} +/- {spinor_std[j]:.4f} | "
            f"{spinor_decay_mean[j]:>7.4f} +/- {spinor_decay_std[j]:.4f} | "
            f"{diagonal_mean[j]:>7.4f} +/- {diagonal_std[j]:.4f}"
        )

    print("-" * 78)
    ms = np.nanmean(spinor_mean)
    msd = np.nanmean(spinor_decay_mean)
    md = np.nanmean(diagonal_mean)
    print(f"{'Mean':>6} | {ms:>7.4f}{'':>14} | {msd:>7.4f}{'':>14} | {md:>7.4f}")

    print("\n" + "=" * 78)
    print("ANALYSIS")
    print("=" * 78)

    if msd < ms:
        print(f"  Decay REDUCES spinor Markov score: {ms:.4f} -> {msd:.4f} "
              f"({(1 - msd/ms)*100:.1f}% reduction)")
    else:
        print(f"  Decay did NOT reduce spinor Markov score: {ms:.4f} -> {msd:.4f}")

    if msd < md:
        print(f"  Spinor+Decay BEATS diagonal: {msd:.4f} < {md:.4f}")
        print("  >> CONSISTENT with conjecture: geometric structure + decay helps")
    elif msd < ms and msd > md:
        print(f"  Spinor+Decay improved but still > diagonal: {msd:.4f} > {md:.4f}")
        print("  >> PARTIAL support: decay helps, but geometry alone doesn't beat diagonal")
    else:
        print(f"  Diagonal still wins: {md:.4f} < {msd:.4f}")
        print("  >> Inconsistent with conjecture at this scale (random params)")

    print("\nNote: Random (untrained) parameters. Training expected to change results.")
    print("See issue #30 for the trained version of this experiment.")


if __name__ == "__main__":
    main()
