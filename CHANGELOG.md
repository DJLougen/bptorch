# Changelog

All notable changes to bpTorch are documented in this file.

## [0.2.0] - 2026-09-03

Editor, training workspace, Llama Tiny, and PyTorch import/export.

### Highlights
- **Arch 26 Llama Tiny**: RMSNorm, RoPE, SwiGLU, grouped-query attention
- **Train / Pause** in the TopBar with live train+val loss, param L2, per-node grad norms
- **Playground**: prompt templates (raw, ChatML, Alpaca, Llama-3), temperature, KV cache
- **Cook**: standalone zero-dependency 'train.py' from the canvas ('POST /api/v1/cook/export')
- **Import .py**: FX trace of a small 'nn.Module' into a Project ('POST /api/v1/import/pytorch')
- **Editor**: context menu, comments, collapse, edge waypoints, align/fit selection, custom composites, Compile/Save, disable/notes

### Verified
| Suite | Result |
|---|---|
| Backend unit tests | 279 passed |
| Frontend tests | 49 passed |
| nanoGPT parity | All checks pass ('make parity') |
| Catalog training/inference JSON | 25/25 (Arch 26 not in those JSON files) |

### API
- `POST /api/v1/cook/export`
- `POST /api/v1/import/pytorch` ('code', optional 'class_name'; 422 '{message, ops}' on unsupported FX ops)
- existing infer / samples / trace WebSocket unchanged

## [0.1.0] — 2026-08-30

First public release of **bpTorch** — a Blueprint-style visual editor for PyTorch neural architectures.

### Highlights

- **Executable visual graphs** — canvas connections compile to real `torch.nn.Module` parameters and tensor ops
- **25 architecture samples** — Transformers, MLPs, training pipelines, classification heads, and more
- **Forward-only inference engine** — `InferenceEngine` with whole-graph and dual-flow interpreter modes
- **Interactive tracing** — WebSocket execution events, breakpoints, step/continue/stop, tensor inspection
- **nanoGPT numerical parity** — verified against pinned `karpathy/nanoGPT` reference
- **Resizable bento workspace** — drag handles for palette, canvas, inspector, and diagnostics drawer
- **Hierarchical Templates menu** — starter blueprints + nested flyout with all 25 samples by category

### Verified

| Suite | Result |
|---|---|
| Backend unit tests | 247 passed |
| Architecture training matrix | 25/25 converged |
| Architecture inference matrix | 25/25 forward pass |
| nanoGPT parity | All checks pass |
| Frontend tests | 30 passed |
| Batch training script | 25/25 passed |
| Batch inference script | 25/25 passed |

### API

- `POST /api/v1/sessions/{id}/infer` — forward-only inference on a live session
- `GET /api/v1/samples` — architecture sample catalog
- WebSocket trace events at `/ws/api/v1/sessions/{id}/events`

### Known limitations (v0.1)

- No arbitrary Python in graph nodes (by design — see ADR 0004)
- No distributed / multi-node training
- Block instance switching in breadcrumb UI is not yet wired

[0.2.0]: https://github.com/DJLougen/bptorch/releases/tag/v0.2.0
[0.1.0]: https://github.com/DJLougen/bptorch/releases/tag/v0.1.0
