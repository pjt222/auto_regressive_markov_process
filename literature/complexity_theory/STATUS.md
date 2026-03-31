# Domain Status: Complexity Theory (Barbour)

**Priority**: SECONDARY (inspirational)
**Updated**: 2026-03-31

## Role in Project

Julian Barbour's shape dynamics, Janus Point theory, and entaxy concept provided
the original conceptual lens for thinking about embedding spaces as relational
objects where "complexity" governs dynamics.

## Current Assessment (from deep review)

All connections between Barbour's framework and the ML embedding context are
currently **analogies, not homomorphisms**:

- Shape space to embedding space: no quotient structure specified
- Entaxy to perplexity: terminological resonance only
- Janus Point to curriculum learning: metaphorical

However, the **relational intuition IS genuinely apt**: embeddings are
translation-invariant (cosine similarity) and often rotation-symmetric, which
is structurally analogous to shape space's relational nature. The N-body/N-token
analogy (pairwise distances vs. pairwise attention scores) has structural legs.

Shape dynamics itself is a **minority research program** in theoretical physics
with open problems. Building on it means two open links in the theoretical chain.

## What This Means

- Existing files are **kept as-is** (valuable intellectual context)
- Barbour's papers are **still read** (for understanding the analogies deeply)
- The framework should be described as **"inspired by"** rather than **"based on"**
  Barbour's work
- If a formal homomorphism between shape space and embedding space is later
  constructed, this domain would be promoted back to PRIMARY
- The specific physics machinery (N-body potentials, vanishing angular momentum)
  does not transport directly to ML

## Key Papers (now in `papers/`)

- Barbour et al. (2013) - Solution to Problem of Time in Shape Dynamics
- Barbour et al. (2014) - Gravitational Arrow of Time
- Barbour et al. (2018) - Quantum Motion on Shape Space

## Books (obtain separately)

- Barbour (2020) - The Janus Point: A New Theory of Time
- Barbour (1999) - The End of Time
