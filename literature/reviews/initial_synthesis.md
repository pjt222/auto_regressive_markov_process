# Initial Literature Synthesis

## Overview
This document synthesizes the initial findings from literature searches across four research areas: autoregression, Markov processes, complexity theory, and geometric computation.

## Key Connections Identified

### 1. Autoregression ↔ Markov Processes
- **Tokenization enables higher-order Markov behavior** in language models
- Transformers can model k-th order Markov processes where k depends on context length
- **Hybrid models** (e.g., Markov Transformers) combine strengths of both approaches
- The Markov property is both a limitation and a computational advantage

### 2. Complexity Theory ↔ Embedding Spaces
- Barbour's **shape dynamics** provides a natural framework for thinking about embedding spaces
- **Entaxy** (scale-invariant entropy) could be relevant for understanding information in embeddings
- The Janus Point theory suggests complexity grows from minimal states - analogous to learning from simple to complex representations
- Shape space as a configuration space maps well to high-dimensional embeddings

### 3. Geometric Computation ↔ N-dimensional Operations
- Gray Cuber's **3D rotation mechanics** generalizable to SO(n) for n-dimensional spaces
- **Complex number multiplication** patterns extend to quaternions and higher algebras
- **Force-directed layouts** relate to modern embedding techniques (t-SNE, UMAP)
- Discrete lattice structures (Gaussian integers) provide structured embedding spaces

### 4. Sequential Processing ↔ Cognitive Models
- Barenholtz's work on **sequential information processing** under cognitive load
- Predictive cognition aligns with autoregressive prediction
- Mental models and imagination connect to generative aspects of autoregression

## Emerging Framework

### Core Concept: Autoregressive Markov Processes in Shape Space

1. **State Space**: High-dimensional shape space (à la Barbour)
2. **Transitions**: Autoregressive updates as Markov transitions
3. **Complexity Measure**: Entaxy or similar scale-invariant measure
4. **Geometry**: SO(n) transformations and geometric algebras
5. **Computation**: Discrete lattice embeddings with number-theoretic properties

### Mathematical Components

1. **Markov Property Preservation**:
   - Use tokenization to enable higher-order dependencies
   - Implement memory mechanisms that don't violate Markovianity
   - Consider continuous-time Markov jump processes

2. **Geometric Structure**:
   - Embed in spaces with natural geometric properties
   - Use group-theoretic constraints (SO(n), SU(n))
   - Leverage discrete lattices for structured representations

3. **Complexity Evolution**:
   - Track complexity growth during autoregressive generation
   - Use Barbour's framework to understand information creation
   - Connect to thermodynamic concepts (but inverted)

## Research Questions Emerging

1. Can we formalize autoregressive steps as movements in Barbour's shape space?
2. How does tokenization granularity affect the order of the resulting Markov process?
3. Can geometric constraints improve autoregressive model performance?
4. Is there a connection between entaxy and perplexity in language models?
5. How do discrete lattice embeddings compare to continuous embeddings?

## Next Steps

1. **Deep dive into key papers**:
   - "Are Transformers Markov Chains?" (2024)
   - Barbour's mathematical papers on shape dynamics
   - Neural Markov Jump Processes (2024)

2. **Mathematical formalization**:
   - Define the state space precisely
   - Formalize the autoregressive transition operator
   - Prove Markov property preservation conditions

3. **Prototype experiments**:
   - Simple autoregressive model with geometric constraints
   - Complexity tracking during generation
   - Comparison of embedding strategies

## Open Questions

- How to balance computational tractability with theoretical elegance?
- Which aspects of each framework are essential vs. optional?
- What's the minimal viable experiment to test these ideas?
- How to connect discrete (tokens) and continuous (embeddings) views?

---
*Synthesis created: 2025-01-16*
*Based on initial literature searches in all four topic areas*