# Reading Note: Fast and Forgetful Memory (FFM)

**Citation**: Morad, S., Kortvelesy, R., Liwicki, S., & Prorok, A. (2023). Reinforcement Learning with Fast and Forgetful Memory. arXiv:2310.04128.

---

## Summary

FFM is a hybrid memory model for RL in POMDPs, designed to compress trajectory history into a latent Markov state. Unlike RNNs (which apply nonlinear, destructive transforms to recurrent state) or transformers (which store raw history), FFM uses two psychologically-motivated inductive priors: composite memory (exponential decay of traces via learned alpha) and contextual drift (oscillatory temporal context via learned omega). The result is a state S_n in complex matrix form that is a weighted sum of past inputs, parallelizable at training time (O(log n) time, O(n) space) yet recurrent at inference (O(1) time and space). FFM outperforms GRU and all other tested memory models on POPGym and POMD-Baselines without any hyperparameter tuning, and trains two orders of magnitude faster than RNNs.

---

## Key Claims

1. Forgetting is the most important inductive bias for RL memory: the ablation FFM-ND (no decay) underperforms full FFM most severely among all ablations (Table 3).
2. FFM is a universal approximator of temporal convolution (Eq. 24), implemented as a Fourier Series filter over the gated input signal.
3. The two-component architecture separates concerns: the aggregator S maintains exponentially-decaying, context-modulated history; the cell y extracts a real-valued Markov state from S via gating and layer normalization.
4. FFM is interpretable: each dimension of S has a known trace durability t_alpha and contextual period t_omega (Eq. 25), enabling informed initialization.
5. Decay rates and context periods are learned from data, with the model self-organizing into distinct actor and critic memory modes.

---

## Mathematical Objects Defined

| Object | Type | Role |
|---|---|---|
| S_n | C^{m x c} matrix | Recurrent state: decayed, context-modulated summary of trajectory |
| gamma^t | C^{m x c} matrix | Time-decay kernel: exp(-t*alpha) * exp(-t*i*omega), Eq. 12 |
| x_n | R^d vector | Encoded action-observation pair at step n |
| x̃_n | R^m vector | Input-gated trace (Bernoulli approximation via sigmoid) |
| y_n | R^d vector | Output Markov state fed to policy |
| alpha in R^m_+ | Decay rates | Controls trace durability; learned |
| omega in R^c | Context frequencies | Controls oscillatory period; learned |
| t_alpha, t_omega | Interpretability scalars | Trace durability and max contextual period (Eq. 25) |

The aggregator recurrence is: S_n = gamma * S_{n-1} + B_n * x_n (composite memory form, Eq. 7/11), with a closed-form parallel version (Eq. 13-14).

---

## Relevance to Our Project

**What transfers:**
- The decomposition of memory into *what* (decay/forgetting via alpha) and *when* (oscillatory context via omega) is directly applicable to tracking state in shape space. A shape trajectory could be summarized by a similar exponentially-decaying, oscillation-modulated state.
- The interpretability of S — knowing t_alpha and t_omega for each dimension — maps cleanly onto the need to understand *which geometric features are being tracked* over the AR iteration.
- The parallel/recurrent duality (training = parallel sum; inference = recurrent) is the same property we need: fast batch training on shape sequences, O(1) inference per AR step.
- FFM's forgetting prior directly operationalizes selective Markov sufficiency: the state encodes enough history for the downstream task *by construction*, not by brute-force accumulation.

**What doesn't transfer directly:**
- FFM operates in flat R^d observation space; our embeddings live in a geometrically structured (shape) space. The outer-product context mechanism (Eq. 9) assumes Euclidean structure.
- The RL reward signal drives what "enough history" means. In our unsupervised/generative setting, the criterion for sufficient statistic must come from the AR likelihood or a complexity measure (Barbour), not a reward.
- FFM's Bernoulli input gating is motivated by sparsity of sensory experience — less obviously applicable when each AR step is a full geometric embedding update.

---

## Key Insight for Our Framework

FFM provides a concrete mathematical answer to the sufficient-statistic question: a state is Markov-sufficient when it retains only the *relevant* history, where relevance is defined by the downstream task's temporal scale. The learned alpha and omega parameters adapt the memory horizon to the problem. For our framework, this suggests: the AR recurrence should carry a complex-valued state with learnable decay (how long geometric features persist) and learnable frequency (what periodic structure in shape space to track). The Markov property is then preserved not by storing all history but by choosing alpha, omega such that the discarded history is irrelevant to the next shape prediction.

Contrast with Mamba (Paper 7): Mamba's selection is *input-dependent* (the gates A, B, C depend on x_t), making transitions non-stationary. FFM's decay is *time-homogeneous* (alpha, omega are fixed learned parameters), making the transition kernel stationary — closer to a classical homogeneous Markov chain. For shape space, time-homogeneous decay may be more appropriate if the geometric complexity measure is stationary, but input-dependent selection (Mamba) is needed if the relevant memory horizon varies with the shape being processed.

---

## Open Questions

1. Can the FFM aggregator be adapted to non-Euclidean (e.g., Riemannian) input spaces, replacing the outer-product context with a geometric analogue?
2. Does the learned decay rate alpha correspond to anything interpretable in Barbour's complexity landscape — e.g., do low-complexity shapes have longer memory traces?
3. FFM is a universal approximator of convolution; does this universality extend to the *geometric* convolutions that arise in shape-space dynamics?
4. The paper shows forgetting is critical for RL. Is forgetting also critical for AR shape modeling, or does the Markov structure of shape space make longer memory beneficial?
