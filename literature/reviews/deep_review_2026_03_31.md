# Deep Review: Autoregressive Markov Processes in Shape Space
**Date**: 2026-03-31
**Review team**: 7 agents across 3 phases (4 domain experts, synoptic-mind team, strategy team)

---

## Executive Summary

This project has a **genuine core intuition** surrounded by **borrowed formalism that does not connect to it**. The intuition -- that autoregressive generation is a Markov process moving through a geometrically structured latent space -- is not trivially wrong and connects to active research frontiers (state space models, geometric deep learning, information geometry). The project is worth continuing, but requires a significant pivot: narrowing from four loosely connected domains to one focused thesis.

**Verdict: Pivot, don't abandon.**

---

## Phase 1: Domain-Expert Assessments

### Markov Process Review (markovian agent)
- **"Each AR self-feed = one Markov step"**: Partially defensible, but only under Reading C -- where the state is a fixed-dimensional embedding that acts as a sufficient statistic. This is the most interesting reading and it was never formally stated.
- **"Markov property should not be violated"**: Circular reasoning. Markovianity is a property of the stochastic process, not caused by network updates.
- **Tokenization-Markov connection** (Rajaraman et al. 2024): Strongest claim in the project. Correctly identified and interpreted. But: the project proposes continuous shape-space embeddings that bypass tokenization -- what replaces it?
- **6 formal gaps**: No state space definition, no transition kernel, no sufficient statistic proof, no stationary distribution, no geometry-Markov connection, no computable preservation criterion.
- **Missing literature**: Mamba/SSMs, LSTMs, mixing time theory, sufficient statistics.

### Complexity Theory Review (theoretical-researcher agent)
- **All Barbour connections are analogies, not homomorphisms.** Shape space to embedding space: no quotient structure specified. Entaxy to perplexity: terminological resonance only. Janus Point to curriculum learning: metaphorical.
- **But**: the relational intuition IS genuinely apt. Embeddings are translation-invariant (cosine similarity) and often rotation-symmetric -- structurally analogous to shape space.
- **The N-body/N-token analogy has structural legs**: pairwise distances in N-body systems vs. pairwise attention scores.
- **Shape dynamics is a minority research program** with open problems -- building on it means two open links in the theoretical chain.
- **Recommendation**: Reframe as "inspired by" not "based on." Attempt empirical complexity measure construction first.

### Research Methodology Review (senior-researcher agent)
- **Search strategy**: Weak. No PRISMA, no documented execution, no inclusion/exclusion criteria.
- **Single-source bias**: All 20 Markov papers from HuggingFace Papers only.
- **Confirmation bias**: Papers selected for support, not challenge. No negative results or impossibility theorems. No critical perspectives on Barbour.
- **Execution gap**: Ratio of infrastructure to scholarship is ~100:0. Ten 0-byte PDFs. One unfilled template. Zero papers actually read.
- **Dormant 14 months** since last substantive work.
- **Recommendation**: Actually read one paper. Drop Gray Cuber. Focus thesis narrowly.

### Geometric/Physics Review (physicist agent)
- **SO(3) to SO(n)**: Mathematically valid generalization, but Gray Cuber adds nothing beyond standard Lie group theory.
- **Division algebra hierarchy terminates at 8D** (Hurwitz theorem). Correct nD generalization is Clifford algebras, not octonions.
- **Gaussian integers are too small** -- interesting lattice structures for high dimensions are A_n, D_n, E_8, Leech lattice.
- **"n-dimensional shapes = embedding vectors"**: Needs formalization. Best interpretation: sections of a fiber bundle (gauge-equivariant networks already do this).
- **Geometric deep learning literature conspicuously absent**: Bronstein et al. (2017/2021), Cohen & Welling (2016), Geometric Algebra Transformers (2023).

---

## Phase 2: Synoptic Integration (synoptic-mind team)

### The Gestalt
All four reviewers independently point at the **same structural gap from different angles**: **the state is undefined.**
- Markov theorist: "no state space definition"
- Methodology reviewer: "no papers read to inform it"
- Physicist: "shapes-as-states needs fiber bundle formalization"
- Complexity theorist: "analogies, not homomorphisms"

They are all seeing the same hole.

### Load-Bearing vs. Decorative
| Domain | Status | Reason |
|--------|--------|--------|
| Autoregression + Markov | **KEEP (core)** | This IS the thesis |
| Geometric state space | **KEEP (core)** | Gives the thesis its distinctive angle |
| Barbour's complexity theory | **DEMOTE** | Relational intuition useful; specific physics not load-bearing |
| Gray Cuber | **DROP** | Standard Lie group theory covers everything needed |

### The Convergent Thesis
> An autoregressive model whose state is defined as a point in a quotient embedding space (modulo some symmetry group) can preserve the Markov property under conditions that are formalizable and empirically testable -- and the structure of the quotient determines what "complexity" means for the generated sequence.

### Single Most Important Next Step
**Define the state.** In mathematical notation. What is s_t at autoregressive step t? Everything else is downstream.

### Field Monitor Observations
- The review was rigorous but disproportionate to the project's stage (early exploration, not failed thesis)
- The genuine spark worth protecting: the cross-domain curiosity that connected these four ideas in the first place
- The viable path forward runs through the geometric deep learning literature, which gives formal structure to the relational intuitions
- What the researcher needs: not another devastating critique, but a clear, narrow path to resume with momentum

---

## Phase 3: Strategic Action Plan

### Critical Path (in order)
1. Create `framework/state_definition.md` -- mathematical definition of the state
2. Read 2 papers deeply with actual notes
3. Write 1-page formalization with conjecture (`framework/formalization.md`)
4. Construct a toy example (2D/3D, compute Markov property)
5. Revise conjecture based on results

### Scope Decisions
- **Autoregression + Markov**: KEEP (core thesis)
- **Geometric structure**: KEEP (distinctive angle)
- **Barbour**: DEMOTE (inspiration only, stop collecting papers)
- **Gray Cuber**: DROP (remove from scope entirely)

### First 3 Sessions
| Session | Focus | Deliverable |
|---------|-------|-------------|
| 1 | Read "Markov Property of NAR" (2024). Create `framework/` dir. Draft state definition. | Filled reading notes + `framework/state_definition.md` |
| 2 | Read Bronstein et al. "Geometric Deep Learning" (2021), symmetry chapters. Update state definition with quotient structure. | Revised state definition with symmetry group |
| 3 | Write formalization with conjecture. Build toy example. | `framework/formalization.md` + working toy example |

### Reading List (priority order)
1. "On the Markov Property of Neural Algorithmic Reasoning" (2024) -- when does neural sequence modeling preserve Markov structure?
2. Bronstein et al. "Geometric Deep Learning" (2021) -- which symmetry group is natural for embeddings?
3. Gu & Dao "Mamba" (2023) -- how does state compression relate to Markov order?
4. "Toward a Theory of Tokenization in LLMs" (2024) -- does tokenization control state space dimension?
5. Amari "Information Geometry" (2016), ch. 1-3 -- is information geometry the right metric for your state space?

### Success Criteria
1. **State definition exists and is falsifiable** -- someone else can read it and say "well-defined" or "ambiguous at point X"
2. **Toy example produces a concrete result** -- a number or proof, not a narrative
3. **Conjecture is stated precisely enough to be wrong** -- "If [conditions], then [process on S/G] satisfies Markov property"

### Timeline
- Weeks 1-3: Sessions 1-3 (reading + state definition + toy example)
- Weeks 4-6: Refine formalization, read remaining papers, iterate
- Weeks 7-10: First concrete result worth showing someone
- **10 weeks at 1-2 sessions/week to a result. Do less, finish it.**

---

## Appendix: Missing Literature (consolidated across all reviews)

### Critical (must read)
- Bronstein et al. "Geometric Deep Learning" (2021)
- Gu & Dao "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Cohen & Welling "Group Equivariant Convolutional Networks" (2016)

### Important (should read)
- Amari "Information Geometry and Its Applications" (2016)
- Ruhe et al. "Clifford Neural Layers for PDE Modeling" (2023)
- Brehmer et al. "Geometric Algebra Transformers" (2023)
- Hsu, Kakade & Zhang "Spectral Algorithm for Learning HMMs" (2012)
- Levin, Peres & Wilmer "Markov Chains and Mixing Times" (2009)

### Useful (read when formalizing)
- Cohen et al. "Gauge Equivariant Convolutional Networks" (2019)
- Hochreiter & Schmidhuber "Long Short-Term Memory" (1997)
- Zeng et al. "Are Transformers Effective for Time Series Forecasting?" (2023)
- Viazovska, E_8 sphere packing (2016)

---

*Review conducted by: markovian, theoretical-researcher, senior-researcher, physicist, adaptic (synoptic integrator), contemplative (field monitor), and strategy team. Orchestrated 2026-03-31.*
