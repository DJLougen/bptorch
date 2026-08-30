# Neural Blueprint Studio v0.1

Executable Unreal Engine Blueprint-style neural-network architecture editor with PyTorch runtime, progressive disclosure, symbolic shape propagation, interactive execution tracing, and numerical parity against `karpathy/nanoGPT`.

## Overview

Neural Blueprint Studio treats the visual graph as the canonical executable model specification. A connection on the canvas represents an actual tensor dependency; changing a node property changes the compiled PyTorch model.

```text
Visual Graph → Canonical Model IR → Validation & Shape Inference → PyTorch Runtime → Actual Tensors & Loss
```

## Features

- **Executable Visual Graph**: Visual graph compiles directly into `torch.nn.Module` with real parameters and tensor ops.
- **Hierarchical Composite Graphs**: Click into subgraphs (Transformer Stack, Blocks, Attention, MLP) with breadcrumb navigation and preserved viewports.
- **Symbolic Shape & Type Engine**: Propagates dimensions (`B`, `T`, `C`, `V`, etc.) and detects mismatches before execution.
- **Constrained Controls**: Dropdown-driven bounded options (e.g. valid head counts for `n_embd`) with immediate local diagnostics.
- **Blueprint Execution Tracing**: Watch data traverse the graph via WebSocket events with speed controls (Instant, Fast, Normal, Step).
- **Interactive Breakpoints & Stepping**: Set breakpoints on any node, inspect input/output tensor statistics, step single instructions, continue, or stop.
- **Reference Parity against nanoGPT**: Verified numerical parity across forward logits, loss, intermediate activations, gradients, optimizer step, and parameter counts.
- **Forward-Only Inference Engine**: `InferenceEngine` runs no-grad forward passes on any blueprint (whole-graph or dual-flow interpreter mode) with a `POST /api/v1/sessions/{id}/infer` API.
- **25 Architecture Sample Gallery**: Pre-built blueprints (Transformers, MLPs, training pipelines, etc.) loadable from the hierarchical **Templates** menu in the UI.
- **Resizable Bento Workspace**: Drag handles resize the node palette, canvas, inspector, and diagnostics drawer.
- **Local-First**: Runs entirely on your local machine with zero external cloud dependencies or telemetry.

## System Requirements

- Python 3.10+ (Python 3.12 recommended) with PyTorch 2.x
- Node.js 18+ and npm / pnpm
- CPU, Apple Silicon (MPS), or CUDA GPU

## Quick Start

```bash
# Install backend and frontend dependencies
make setup

# Launch development environment (FastAPI backend on :8000 + Vite frontend on :5173)
make dev

# Run all automated tests (backend unit tests, parity tests, frontend tests)
make test

# Run strict numerical parity test suite against karpathy/nanoGPT
make parity

# Run linters and type checkers
make lint

# Generate example blueprints for the UI gallery
make examples

# Batch-train or batch-infer all 25 architecture samples
make train-samples
make infer-samples
```

## Repository Structure

- `web/`: React 19 + TypeScript + Vite + React Flow (@xyflow/react) + Zustand frontend
- `server/`: Python + FastAPI + PyTorch + Pydantic backend runtime
- `packages/contracts/`: JSON schemas and generated TypeScript type definitions
- `references/`: Pinned reference `karpathy/nanoGPT` and lock file
- `examples/`: 25 architecture sample blueprints plus `training_results.json` / `inference_results.json`
- `scripts/`: Batch training, inference, parity, and example export utilities
- `docs/adr/`: Architecture Decision Records

## Current Limitations (v0.1)

- Arbitrary Python code execution inside graph nodes is intentionally excluded (see ADR 0001 & ADR 0004).
- Full distributed training and multi-node execution are post-v0.1 roadmap items.
- Only forward inspection, single-step debugging, and one-step optimizer parity verification are in v0.1 scope.
