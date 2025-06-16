# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project exploring the combination of:
- Auto-regression techniques (similar to LLM reasoning)
- Hidden Markov processes (where each self-feed of the autoregressive process represents one Markov step)
- Embedding/vector space grounding based on "complexity" concepts from Julian Barbour's work
- Geometric shape-based calculations with n-dimensional embeddings

## Project Status

This project is in the **literature review phase**. Current focus is on gathering and reviewing relevant research across four main areas:
1. Auto-regression and LLM reasoning
2. Hidden Markov processes
3. Complexity theory (Barbour's framework)
4. Geometric computation with n-dimensional shapes

See `literature/` directory for collected papers and review progress.

## Key References

- Auto-regression research: Dr. Eran Barenholtz (FAU)
- Complexity theory: Julian Barbour and colleagues
- Shape-based calculations: The Gray Cuber approach (https://thegraycuber.github.io/)

## Literature Organization

The `literature/` directory contains:
- `README.md` - Overview and collection status
- `bibliography.md` - Comprehensive list of papers to find/found
- `review_plan.md` - Structured plan for literature review phases
- Subdirectories for each research area:
  - `autoregression/` - Auto-regressive models and LLM reasoning
  - `markov_processes/` - HMM and Markov chain theory
  - `complexity_theory/` - Barbour's work and related concepts
  - `geometric_computation/` - Shape-based calculations
  - `reviews/` - Synthesis documents

## Development Notes

- The Markov property should be preserved as the autoregression updates the network
- Consider n-dimensional shapes corresponding to n-dimensional embedding vectors
- The project aims to integrate mathematical concepts from complexity theory with practical machine learning approaches
- Literature notes should follow the template in `bibliography.md` for consistency

## Next Steps

1. Collect core references from the sources listed in `initial_scribbles.md`
2. Create reading notes for each paper using the provided template
3. Identify connections between the different research areas
4. Develop a unified mathematical framework