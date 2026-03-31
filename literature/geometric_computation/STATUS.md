# Domain Status: Geometric Computation

**Priority**: SECONDARY (inspirational)
**Updated**: 2026-03-31

## Role in Project

This domain provided the original inspiration for thinking about n-dimensional
shapes and embedding vectors. The Gray Cuber's visualizations of SO(3) rotations,
Gaussian integers, and complex multiplication seeded the idea of geometric
structure in embedding spaces.

## Current Assessment (from deep review)

The geometric computation ideas are **pedagogically valid but research-redundant**.
The established **geometric deep learning** literature (Bronstein et al. 2021,
Cohen & Welling 2016, Clifford algebras) covers the same concepts with full
mathematical rigor. The Gray Cuber analysis remains as historical record of
the project's intellectual origins.

## What This Means

- Existing files are **kept as-is** (historical record)
- Papers in this domain are **still read** (for intellectual context)
- New formalization work should reference the **geometric deep learning literature**
  (now in `papers/`) rather than the Gray Cuber analysis
- The correct nD generalization is **Clifford algebras**, not the division algebra
  hierarchy (which terminates at 8D by Hurwitz's theorem)

## Key Papers (now in `papers/`)

- Bronstein et al. (2021) - Geometric Deep Learning (foundational survey)
- Cohen & Welling (2016) - Group Equivariant CNNs
- Cohen et al. (2019) - Gauge Equivariant CNNs (fiber bundles)
- Ruhe et al. (2023) - Clifford Neural Layers
- Brehmer et al. (2023) - Geometric Algebra Transformers
