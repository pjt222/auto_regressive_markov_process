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

**Phase**: Experimental validation (7 experiments completed 2026-04-01)

**Deep review completed**: 2026-03-31 (7 agents, 3 phases). See:
- `literature/reviews/deep_review_2026_03_31.md` -- full report
- `literature/reviews/synoptic_integration.md` -- gestalt from synoptic-mind team

**Central finding**: Geometric (spinor) inductive bias is **domain-specific**. It hurts on language data (14-20% worse, #31) but **helps substantially on data with genuine rotational structure** (20-22% better, #50). The rotation constraint assumes symmetry — when the data has it, the bias pays off; when it doesn't, it's a liability.

**Experiments completed** (2026-04-01):
- #45: Dense ablation -- Spinor beats Dense at D>=9 despite 6-10x fewer params (toy)
- #25: Epsilon bound -- simple norm-variance bound fails; two-channel leakage model needed
- #44: Block size -- sharp transition at diagonal→rotation, flat across rotation types (toy)
- #40: Discrete spinor -- Gumbel-softmax collapses with ES optimizer
- #47: Convergence tracking -- no spontaneous polyhedral convergence
- #31: **Language modeling** -- Diagonal SSM (469 PPL) beats QuatBlock (562) and Pascal grade-hierarchy (535) on WikiText-2 at D=768. Rotation is the wrong inductive bias for language. CUDA kernel: 732x speedup.
- #50: **SO(3) rotation prediction** -- Givens (2.24°) and QuatBlock (2.30°) beat Diagonal (2.88°) by 20-22% on synthetic rotation walks at D=12. Rotation IS the right bias for rotational data. Pascal (2.60°) also beats Diagonal but underperforms tighter constraints.

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
5. ~~Dimension scaling experiment~~ **DONE** (issue #38, #45: geometric advantage confirmed at D>=9)
6. ~~Derive norm concentration bound~~ **PARTIALLY DONE** (issue #25: simple bound fails, two-channel model needed)
7. ~~Scale to factored quaternion SSM~~ **DONE** (issue #31): rotation HURTS on language (469 vs 535-562 PPL). Domain-specific, not universal.
8. ~~Domain-specific applications~~ **DONE** (issue #50): rotation prediction confirms geometric bias helps on SO(3) data (20-22% over Diagonal). Givens slightly beats QuatBlock at D=12.
9. **Scale #50 to D=3 and D=48**: test whether QuatBlock wins at D=3 (exact SO(3) match) where it should have maximum advantage
10. **N-body physics** (#48): established benchmark with genuine SO(3) symmetry
11. **Learnable coupling topology** (#51): gated rotation instead of fixed blocks
12. **Re-run #40 with gradient-based optimizer** for fair discrete spinor comparison
13. **Performance**: CUDA blockDim restructuring (#53), numba-accelerate remaining ES scripts (#54)

## Team Activation (REQUIRED when user requests a team)

When the user asks to activate or use a team: (1) call `ToolSearch("select:TeamCreate")` to load the TeamCreate tool, (2) read the team definition from `/mnt/d/dev/p/agent-almanac/teams/<team-name>.md`, (3) call `TeamCreate` with the team configuration. Do NOT fall back to spawning individual agents via the Agent tool — always use TeamCreate for team requests.

Available teams are listed in `/mnt/d/dev/p/agent-almanac/teams/_registry.yml`.
