# ADR 0001: Canonical Model Intermediate Representation

## Status
Accepted

## Context
bpTorch requires a single source of truth for neural-network architectures that serves editing, validation, shape inference, PyTorch runtime execution, and project serialization. UI libraries like React Flow maintain internal node/edge structures with visual and presentation state.

## Decision
We define an independent, versioned JSON-serializable Canonical Model IR (`Project`, `ModelDefinition`, `GraphDefinition`, `NodeInstance`, `PortDefinition`, `Edge`, `WeightBinding`). React Flow nodes and edges are strictly UI projections derived from the canonical IR. All architecture mutations operate on the IR, and UI-specific state (positions, viewports) is decoupled from model semantics.

## Consequences
- Clean separation between architecture semantics and canvas layout.
- Enables headless validation, parity testing, and batch execution without browser/React dependencies.
- Prevents UI layout state from corrupting or altering runtime execution.
