# Adversarial Literature Review (Issue #15)

**Date**: 2026-03-31
**Purpose**: Steelman the counter-position — papers arguing against the utility of Markov structure and geometric constraints in sequence models.

---

## Summary

Seven papers were reviewed. The three originally specified in issue #15 plus four additional papers identified through search. Together they converge on a single strongest counterargument:

> **The conjecture conflates two distinct mechanisms.** The spinor model carries more history (higher Markov score) and predicts better (lower test loss). But these two facts do not establish a causal connection through geometric inductive bias. An alternative explanation — that any non-diagonal transition projects more of the input's log-signature, retaining more relevant history — is fully consistent with the data. The prediction advantage might be attributable to non-diagonality, not to the specific geometry of Spin(D).

This counterargument motivates a critical ablation: compare Spinor+Decay against a **generic dense SSM** (same parameter count, no Spin(D) constraint). If Spinor+Decay beats Diagonal but Generic Dense also beats Diagonal by the same margin, the geometric constraint adds nothing beyond non-diagonality.

---

## Paper-by-Paper Analysis

### 1. Deletang et al. (2023) — Neural Networks and the Chomsky Hierarchy

**arXiv**: 2207.02098 | **Venue**: ICLR 2023

**Core argument**: The Chomsky hierarchy predicts which architectures generalize OOD, independently of parameter count. Standard RNNs and Transformers fail on all non-regular tasks. Architectural expressivity — measured by computational class — is the binding constraint.

**Challenge**: Both Spinor+Decay and Diagonal SSM are linear recurrences in TC^0. The geometric bias cannot lift the model to a higher Chomsky tier.

**Response**: The revised conjecture claims a *within-class* advantage on finite-Markov-order sources, not a class-crossing improvement. The D=3 result (8.7%) is a constant-factor improvement within TC^0, consistent with Deletang et al.

### 2. Olsson et al. (2022) — In-Context Learning and Induction Heads

**arXiv**: 2209.11895 | **Venue**: Transformer Circuits Thread

**Core argument**: Transformers perform in-context learning through "induction heads" — [A][B]...[A] → [B] pattern completion that is inherently non-local and non-Markovian. This mechanism is causally responsible for the majority of in-context learning.

**Challenge**: If the dominant algorithm for sequence prediction is long-range pattern matching, any SSM with bounded hidden state is solving a different problem. The spinor-vs-diagonal comparison may be comparing two suboptimal approaches.

**Response**: The D=3 experiment uses a 2nd-order Markov source where induction heads are irrelevant. Issue #31 (real language data at D=768) is the test of whether the spinor advantage survives where induction heads dominate. The formalization should acknowledge this as a key extrapolation risk.

### 3. Yun et al. (2020) — Are Transformers Universal Approximators?

**arXiv**: 1912.10077 | **Venue**: ICLR 2020

**Core argument**: Transformers (with positional encodings) are universal approximators of continuous sequence-to-sequence functions. The architecture's representational power is not the binding constraint.

**Challenge**: Since both models are universal approximators (given sufficient parameters), any advantage must lie in optimization dynamics or inductive bias — not expressivity.

**Response**: The revised conjecture already accepts this framing. Condition 3 in formalization.md ("reduces the effective number of free parameters while maintaining expressivity") is an inductive-bias-efficiency claim, not an expressivity claim. The matched-parameter-count design correctly operationalizes this.

### 4. Merrill, Petty, and Sabharwal (2024) — The Illusion of State in State-Space Models

**arXiv**: 2404.08819 | **Venue**: ICML 2024

**Core argument**: SSMs cannot compute outside TC^0 — the same class as transformers. The "state" in an SSM is illusory in that recurrence does not grant access to a richer computational class.

**Challenge**: The spinor SSM cannot be a fundamentally better history encoder than diagonal, because both are equally TC^0-constrained. The higher Markov score might be statistically redundant history.

**Response**: The project works far below the TC^0 ceiling on a regular-language task. The formalization should clarify that the geometric advantage is claimed for finite-Markov-order prediction, with no claim about non-TC^0 tasks.

### 5. Sarrof, Veitsman, and Hahn (2024) — The Expressive Capacity of SSMs

**arXiv**: 2405.17394 | **Venue**: NeurIPS 2024

**Core argument**: SSMs have "overlapping but distinct" capacities vs transformers. A specific design choice — linearity of the recurrence — limits SSM expressive power. The path to more expressive SSMs runs through nonlinearity, not geometric structuring.

**Challenge**: The spinor model preserves linearity of the recurrence. If nonlinearity is what tasks need, neither spinor nor diagonal has it.

**Response**: The project studies prediction quality on a smooth statistical source, not formal-language state tracking. The rotation's norm preservation and full-rank mixing may be advantageous even without nonlinearity. Orthogonal concern, but should be cited.

### 6. Cirone, Orvieto, Walker, Salvi, and Lyons (2024) — Theoretical Foundations of Deep Selective SSMs

**arXiv**: 2402.19047 | **Venue**: NeurIPS 2024

**Core argument**: Using rough path theory, selective SSMs are provably low-dimensional projections of the log-signature of the input path. The non-Markovian behavior of hidden states is a mathematical necessity of this projection.

**Challenge**: **This is the hardest adversarial challenge.** If both models are log-signature projections, the spinor's higher Markov score is not "better history encoding" — it's projecting more of the log-signature by construction. Any non-diagonal transition would achieve similar benefits, without requiring Spin(D) specifically.

**Response**: **Not yet addressed.** Requires the generic dense SSM ablation to distinguish geometric structure from mere non-diagonality. If Spinor+Decay beats Generic Dense at matched parameters, the Spin(D) constraint is carrying the load. The dimension scaling results (issue #38) could help: if the advantage tracks the geometry of Spin(D) rather than the generic benefit of non-diagonal transitions, the geometric hypothesis is supported.

### 7. Movahedi, Orvieto, and Moosavi-Dezfooli (2024) — Geometric Inductive Biases of Deep Networks

**arXiv**: 2410.12025

**Core argument**: Architectural geometric biases *hurt* when they don't align with task geometry. ResNets fail where MLPs succeed when the convolutional structure forces an incorrect geometry.

**Challenge**: The conjecture assumes Spin(D) is the "right" symmetry for language embeddings, but this is never argued from first principles. Cosine similarity suggests positive-rescaling invariance, not full SO(D).

**Response**: Partially addressed by falsification criterion 3. The positive argument for *why* rotation is appropriate should be strengthened — connect to cosine similarity geometry and gauge symmetry discussion in state_definition.md §2.

---

## Recommended Actions

1. **Scope of Claim paragraph** in formalization.md: The prediction advantage is (a) within-class, (b) for finite-Markov-order sources, (c) an inductive-bias-efficiency claim.

2. **Generic Dense SSM ablation** (new issue or extend #38): Compare Spinor+Decay vs Generic Dense SSM to distinguish geometric structure from non-diagonality. This directly answers Cirone et al.

3. **Symmetry-alignment argument**: Explain why Spin(D) is appropriate for embedding spaces (cosine similarity → SO(D) natural symmetry group → Spin(D) double cover).

4. **Induction head caveat**: Note that issue #31 tests whether spinor advantage survives on natural language where induction heads dominate.

---

## Citation Keys Added to references.bib

- `deletang2023chomsky`
- `olsson2022incontext`
- `yun2020transformers`
- `merrill2024illusion`
- `sarrof2024expressive`
- `cirone2024theoretical`
- `movahedi2024geometric`
