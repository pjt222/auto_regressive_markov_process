# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project exploring the combination of:
- Auto-regression techniques (similar to LLM reasoning)
- Hidden Markov processes (where each self-feed of the autoregressive process represents one Markov step)
- Embedding/vector space grounding based on "complexity" concepts from Julian Barbour's work
- Geometric shape-based calculations with n-dimensional embeddings

## Research Hypothesis (revised 2026-03-31, after k=1 disproof + training experiment)

> An autoregressive SSM with spinor-constrained transitions (Spin(D) rotation +
> scalar decay) achieves better next-token prediction than an unconstrained
> diagonal SSM at matched parameter count, because the rotation constraint acts
> as a geometric inductive bias that encodes embedding space structure.

**Note**: The original hypothesis (quotient embedding space preserves Markov
property) was partially disproved -- the k=1 case is false because the norm leaks
history. The revised hypothesis focuses on prediction quality rather than
Markovianity. Partial empirical support at D=3 (issue #30). See issue #12, #38.

## Domain Priorities

| Domain | Priority | Role |
|--------|----------|------|
| Autoregression + Markov processes | **PRIMARY** | The thesis -- does geometric structure improve AR prediction? |
| Geometric state space structure | **PRIMARY** | The formalization -- spinor transitions, Clifford algebras, inductive bias |
| Barbour's complexity theory | SECONDARY | Inspirational -- relational intuition is apt, specific physics not load-bearing |
| Gray Cuber / pedagogical geometry | SECONDARY | Historical origin -- geometric deep learning literature supersedes it |

See `literature/complexity_theory/STATUS.md` and `literature/geometric_computation/STATUS.md` for detailed rationale.

## Project Status

**Phase**: Literature review -- deep reading (31 papers retrieved, reading in progress)

**Deep review completed**: 2026-03-31 (7 agents, 3 phases). See:
- `literature/reviews/deep_review_2026_03_31.md` -- full report
- `literature/reviews/synoptic_integration.md` -- gestalt from synoptic-mind team

**Central finding**: The state is $h_t \in \mathbb{R}^D$ (not the projection to $S^{D-1}$). The k=1 case was disproved: the norm leaks history. Training experiment (D=3) shows spinor transitions improve prediction by 8.7% but carry more history. See `reports/framework_progress.qmd`.

## Literature Organization

The `literature/` directory contains:
- `README.md` - Overview and collection status
- `bibliography.md` - Comprehensive list of papers
- `review_plan.md` - Structured plan for literature review phases
- Subdirectories for each research area:
  - `autoregression/` - Auto-regressive models and LLM reasoning
  - `markov_processes/` - HMM and Markov chain theory
  - `complexity_theory/` - Barbour's work (SECONDARY -- see STATUS.md)
  - `geometric_computation/` - Geometric deep learning (SECONDARY -- see STATUS.md)
  - `reviews/` - Synthesis and review documents

## Quarto

This project uses Quarto for reporting. Configuration: `_quarto.yml`. Bibliography: `references.bib`.

```bash
# Render reading notes (HTML preview)
quarto render reports/reading_notes.qmd --to html

# Render framework progress
quarto render reports/framework_progress.qmd --to html
```

## Key References

- `references.bib` - BibTeX bibliography (33 papers + 4 books)
- Auto-regression research: Dr. Eran Barenholtz (FAU)
- Complexity theory: Julian Barbour and colleagues
- Geometric deep learning: Bronstein et al. (2021), Cohen & Welling (2016)
- State space models: Gu & Dao (2023, Mamba)

## Development Notes

- The central research question: does geometric structure (spinor transitions) on an AR SSM improve next-token prediction?
- The state is $h_t \in \mathbb{R}^D$ (formalized in `framework/state_definition.md` v0.2)
- The quotient to $S^{D-1}$ is invalid — norm leaks history (k=1 disproof)
- All Barbour connections are currently analogies, not homomorphisms -- use "inspired by" framing
- The correct nD geometric generalization is Clifford algebras, not the division algebra hierarchy
- Literature notes should follow the Quarto reading notes format in `reports/reading_notes.qmd`

## Next Steps

1. ~~Read all 31 papers~~ 24/31 done (7 remaining, lower priority -- issue #36)
2. ~~Define the state~~ **DONE** (revised to $h_t \in \mathbb{R}^D$)
3. ~~Write formalization~~ **DONE** (revised to geometric inductive bias conjecture)
4. ~~Construct toy example~~ **DONE** (`framework/toy_example.py`, `framework/train_toy.py`)
5. **Dimension scaling experiment** (issue #38): test prediction advantage across D
6. **Derive norm concentration bound**: when does Var[log ||h_t||] → 0?
7. **Scale to factored quaternion SSM** (issue #31): real language data at D=768

## Team Definitions
Team definitions for multi-agent compositions are at `/mnt/d/dev/p/agent-almanac/teams/<team-name>.md`. Read the definition and orchestrate via `TeamCreate`.
