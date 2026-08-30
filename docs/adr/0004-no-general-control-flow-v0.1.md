# ADR 0004: No General Dynamic Control Flow in v0.1

## Status
Accepted

## Context
Full general-purpose dynamic control flow (arbitrary while loops, dynamic conditionals, arbitrary Python function nodes) introduces execution security hazards, cycle ambiguities, serialization complexity, and non-deterministic shape propagation.

## Decision
For v0.1, the visual graph is dataflow-driven. Specific conditional behaviors required by nanoGPT (such as inference final-token-only LM-head forward pass, or optional training loss computation) are modeled via constrained semantic nodes (`SelectTimeStep`, `CrossEntropyLoss` with optional target input) and execution modes (`inspection`, `evaluation`, `training-test`) rather than an arbitrary control-flow wire language.

## Consequences
- The graph remains statically analyzable, acyclic, and safely serializable.
- Symbolic shapes can be reliably inferred ahead of execution.
- Extensible to explicit control-flow router nodes in future milestones (e.g. MoE).
