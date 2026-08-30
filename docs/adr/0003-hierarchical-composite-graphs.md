# ADR 0003: Hierarchical Composite Graphs and Progressive Disclosure

## Status
Accepted

## Context
Real transformer architectures like nanoGPT contain dozens of operations per block across multiple layers. Displaying all primitive operations on a flat canvas produces an overwhelming, unusable visual graph.

## Decision
We implement first-class composite modules with explicit interface nodes (`ModuleInput`, `ModuleOutput`). Composite nodes encapsulate subgraphs (e.g. `Input Embeddings`, `Transformer Block`, `Causal Self-Attention`, `MLP`). A `RepeatModule` repeats a block definition $N$ times with independent parameter weights and instance paths (`gpt/blocks[0]`, `gpt/blocks[1]`, ...). Built-in modules can be inspected and forked into editable custom copies.

## Consequences
- The top-level view remains clean and readable (5-6 high-level nodes).
- Users can progressively drill down into blocks, attention, and primitive operations.
- Repeat instances maintain distinct runtime parameter weights while sharing structural definitions.
