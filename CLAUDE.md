# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project exploring the combination of:
- Auto-regression techniques (similar to LLM reasoning)
- Hidden Markov processes (where each self-feed of the autoregressive process represents one Markov step)
- Embedding/vector space grounding based on "complexity" concepts from Julian Barbour's work
- Geometric shape-based calculations with n-dimensional embeddings

## Convergent Thesis (from deep review, 2026-03-31)

> An autoregressive model whose state is defined as a point in a quotient
> embedding space (modulo some symmetry group) can preserve the Markov property
> under conditions that are formalizable and empirically testable -- and the
> structure of the quotient determines what "complexity" means for the generated
> sequence.

## Domain Priorities

| Domain | Priority | Role |
|--------|----------|------|
| Autoregression + Markov processes | **PRIMARY** | The thesis -- when does AR preserve Markov property? |
| Geometric state space structure | **PRIMARY** | The formalization -- quotient spaces, fiber bundles, Clifford algebras |
| Barbour's complexity theory | SECONDARY | Inspirational -- relational intuition is apt, specific physics not load-bearing |
| Gray Cuber / pedagogical geometry | SECONDARY | Historical origin -- geometric deep learning literature supersedes it |

See `literature/complexity_theory/STATUS.md` and `literature/geometric_computation/STATUS.md` for detailed rationale.

## Project Status

**Phase**: Literature review -- deep reading (31 papers retrieved, reading in progress)

**Deep review completed**: 2026-03-31 (7 agents, 3 phases). See:
- `literature/reviews/deep_review_2026_03_31.md` -- full report
- `literature/reviews/synoptic_integration.md` -- gestalt from synoptic-mind team

**Central finding**: The state is undefined. All formal claims depend on specifying what the "state" is at each autoregressive step. See the framework progress tracker: `reports/framework_progress.qmd`

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

- The central research question: under what conditions does an AR model preserve the Markov property in a geometrically structured embedding space?
- The state definition (what is s_t?) must be formalized before other claims can be evaluated
- All Barbour connections are currently analogies, not homomorphisms -- use "inspired by" framing
- The correct nD geometric generalization is Clifford algebras, not the division algebra hierarchy
- Literature notes should follow the Quarto reading notes format in `reports/reading_notes.qmd`

## Next Steps

1. Read all 31 papers (organized in 5 phases A-E, see `reports/reading_notes.qmd`)
2. Define the state s_t in mathematical notation (`framework/state_definition.md`)
3. Write formalization with testable conjecture (`framework/formalization.md`)
4. Construct toy example (2D/3D autoregressive process)
5. Revise and iterate

## Team Definitions
Team definitions for multi-agent compositions are at `/mnt/d/dev/p/agent-almanac/teams/<team-name>.md`. Read the definition and orchestrate via `TeamCreate`.
