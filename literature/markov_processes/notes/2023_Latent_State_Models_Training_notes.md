# Reading Notes: Delays, Detours, and Forks in the Road: Latent State Models of Training Dynamics

**Citation**: Hu, M. Y., Chen, A., Saphra, N., & Cho, K. (2023). Delays, Detours, and Forks in the Road: Latent State Models of Training Dynamics. *Transactions on Machine Learning Research*, 12/2023. arXiv:2308.09543v3.

---

## Summary

This paper fits Gaussian HMMs over sequences of weight-derived metrics (L1/L2 norms, mean, variance of weights/biases, singular value statistics) collected from multiple training runs with different random seeds. The resulting latent state sequence defines a "training map" — a Markov chain over discrete latent states — that provides a low-dimensional discrete representation of training dynamics. The method identifies "detour states": latent states visited only by slow-converging runs that are positively correlated with convergence time. Experiments span grokking tasks (modular arithmetic, sparse parities), image classification (CIFAR-100, MNIST), and masked language modeling, demonstrating that the training map structure shifts with hyperparameter or architecture choices.

---

## Key Claims

1. Training trajectories across random seeds can be faithfully represented as paths through a small discrete Markov chain (the training map), inferred by an HMM fitted on metric sequences.
2. Detour states — latent states visited by a strict subset of training runs with positive regression coefficient for convergence time — are empirically identifiable and mechanistically interpretable.
3. Tasks sensitive to random seed (grokking) produce forking training maps; tasks insensitive (CIFAR-100 standard) produce linear maps. Trajectory dissimilarity (expected Wasserstein distance between empirical state distributions) quantifies this.
4. The linear regression of convergence time on the unigram distribution over latent states achieves R² up to 0.977, confirming that path through the training map is highly predictive of outcome.
5. Architectural choices (batch norm, residual connections, layer norm) directly change which training map topology emerges.

---

## Mathematical Objects Defined

- **Observation sequence**: z_t = [f_1(w_t), ..., f_d(w_t)]^T ∈ ℝ^d, z-score normalized to z̃_t using statistics from first 1000 checkpoints.
- **HMM parameters**: transition matrix P(s_t | s_{t-1}); emission distribution P(z̃_t | s_t = k) ~ N(μ_k, Σ_k). Fitted with Baum-Welch; model order selected by BIC.
- **Training map**: directed graph with hidden states as vertices, pruned transition matrix as edges (edges with zero empirical frequency removed).
- **Unigram featurization**: P̂(s = k) = (Σ_j 𝟙(s_j = k)) / T — the empirical state distribution over a trajectory, used as feature vector for linear regression.
- **Trajectory dissimilarity**: E[W(p, q)] = (2 / N(N-1)) Σ_{i=1}^N Σ_{j=1}^i W(p_i, q_j), where W is the Wasserstein distance between empirical state distributions of two runs.
- **Feature importance**: |∂ log p(s_t = k | z̃_{1:t}) / ∂ z̃_t| — partial derivative of log posterior w.r.t. observation features, used to annotate state transitions.

---

## Relevance to Our Framework

**What transfers directly:**
- The HMM-over-metrics architecture is immediately applicable: our AR steps produce weight/activation trajectories, and we can fit exactly this model to those sequences to discover discrete latent phases.
- The training map concept gives us a principled way to ask whether shape-space trajectories also exhibit forking — i.e., whether different initializations in shape space lead to qualitatively different paths.
- The unigram featurization + linear regression pipeline for outcome prediction is a plug-in component.

**What does not transfer:**
- The paper treats training as a stochastic process driven by random seed, not by input sequence structure. Our interest is in whether the AR *inference* process (not training) is Markov in shape space.
- The latent states here are states of optimizer dynamics, not states of semantic content. Our "states" would need to correspond to geometric positions in shape space.

---

## Key Insight for Our Framework

Training dynamics can themselves be modeled as a first-order Markov chain over discrete latent states, and the HMM successfully recovers interpretable phases (memorization vs. generalization in grokking). This validates the design choice in our framework of treating each AR step as a Markov transition: if even the *parameter update process* is approximately Markov at the right level of abstraction (aggregated metrics), then the *inference process* (where inputs are fixed and the recurrent state evolves) is a more natural target for a Markov assumption, with the added structure of shape space geometry constraining the transition kernel.

---

## Open Questions

1. Their latent states are defined over weight-space metrics. Our analogue would be states defined over position/geometry in shape space — do such states have natural discrete structure, or is a continuous state HMM (switching linear dynamical system) more appropriate?
2. The paper uses N parallel runs with different seeds to fit one HMM. In our single-run AR setting, can we instead use multiple prompts/contexts to generate the ensemble needed for Baum-Welch?
3. Detour states slow convergence during training. Is there an analogous concept for inference: "detour regions" in shape space through which AR chains pass before settling into a stable attractor?
