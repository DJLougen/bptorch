# Neural Blueprint Studio Architecture

## 1. System Topology

Neural Blueprint Studio is a local-first application organized around a strictly typed client-server architecture:

```text
┌────────────────────────────────────────────────────────┐
│               Frontend (React / TypeScript)            │
│  - Canvas (React Flow projection)                      │
│  - Property Inspector & Bounded Controls               │
│  - Node Palette & Search                               │
│  - Breadcrumb Hierarchy & Viewports                    │
│  - Trace Visualizer & Debug Controls                   │
│  - Zustand Stores (IR, UI, Selection, Trace, History)  │
└───────────────────▲─────────────────▲──────────────────┘
                    │ REST API        │ WebSocket Events
                    │ (JSON Schema)   │ (Trace Stream)
┌───────────────────▼─────────────────▼──────────────────┐
│               Backend (FastAPI / PyTorch)              │
│  - Node Registry & Contract Validation                 │
│  - 4-Pass Validator & Shape Inference Engine           │
│  - Parameter Accounting & Weight Tying Manager         │
│  - CompiledGraphModule Runtime (PyTorch Module)        │
│  - Execution Engine, Breakpoints & Stepper             │
│  - Project Serializer & Migration Manager              │
│  - nanoGPT Reference Parity Harness                    │
└────────────────────────────────────────────────────────┘
```

## 2. Canonical Model Intermediate Representation (IR)

The Model IR is the single source of truth for the neural network architecture:

- **Project (`Project`)**: Root container with metadata (`id`, `name`), `model` definition, and `ui` state.
- **Model (`ModelDefinition`)**: Contains `root_graph_id`, `config` (model hyperparameters e.g. `n_embd`, `n_head`), `graphs` (map of graph definitions), and `weight_bindings` (explicit weight sharing e.g. tied embedding & LM head).
- **Graph (`GraphDefinition`)**: Represents a flat or composite subgraph. Contains `interface` (inputs and outputs), `nodes` (node instances), and `edges` (connections between node ports).
- **Node (`NodeInstance`)**: References a `definition_id` (e.g. `builtin.linear@1`), specifies `properties` (literals, `config_ref`, or safe expressions), and `metadata` (`breakpoint`, `disabled`).
- **Edge (`Edge`)**: Explicit connection from `source` (`node_id`, `port_id`) to `target` (`node_id`, `port_id`).
- **UI State (`UIState`)**: Decoupled from IR semantics; stores per-graph viewports, node coordinates, and open graph ID.

## 3. Node Definition Registry

Every node type is registered in the backend registry (`NodeRegistry`). The registry provides:
- Versioned type ID (`builtin.linear@1`, `builtin.causal_self_attention@1`, etc.)
- Metadata (display name, category, icon, docstring)
- Dynamic input/output port definitions based on properties
- Property schemas and UI hints (valid options, ranges, dependencies)
- Parameter accounting rules
- Symbolic shape inference logic
- PyTorch runtime constructor or execution instruction

The frontend fetches the node registry via `GET /api/v1/registry/nodes` and never duplicates registry catalog semantics.

## 4. Symbolic Shape Engine & 4-Pass Validation

Validation operates in 4 distinct passes:
1. **Schema Validation**: Validates IDs, recognized types, required fields, and property types.
2. **Structural Graph Validation**: Verifies port multiplicity, connected required inputs, cycle detection, and interface matching for composite nodes.
3. **Tensor Type & Shape Validation**: Propagates structured symbolic shapes (`[B, T, n_embd]`, `[B, NH, T, HD]`), validates dimension equality, head divisibility (`n_embd % n_head == 0`), and dtype compatibility.
4. **Model-Semantic Validation**: Validates high-level invariants such as maximum sequence length $\le$ `block_size`, tied weight compatibility, and repeated block counts.

Diagnostics are returned with exact `node_id`, `port_id`, `edge_id`, `expected`, `actual`, and actionable `suggestions`.

## 5. Runtime & Compilation Architecture

The graph compiles into a real `CompiledGraphModule(torch.nn.Module)`:
- Builds topological execution plan across all nodes.
- Submodules (`Linear`, `LayerNorm`, `Embedding`, `Dropout`, etc.) are registered in `nn.ModuleDict` / `nn.ModuleList` so PyTorch `parameters()`, `state_dict()`, `.train()`, `.eval()`, and `.to(device)` work natively.
- Functional tensor operations (`Add`, `Reshape`, `Transpose`, `Split`, `MatMul`, `Softmax`, etc.) execute inline with runtime instrumentation.
- Composite subgraphs and repeated stacks (`RepeatModule`) are compiled hierarchically, producing clean hierarchical paths: `gpt/blocks[0]/attention/qkv_projection`.
- Weight bindings (such as tying `TokenEmbedding.weight` to `LMHead.weight`) are applied directly to parameter storage so parameter counters count tied tensors only once.

## 6. Execution Tracing, Breakpoints, and Stepping

The execution runtime supports three modes:
- **Inspection Mode**: `eval()` mode, emits node events via WebSocket, captures tensor summary statistics (shape, dtype, min, max, mean, std, norm, nan count, deterministic sample), and supports pausing on breakpoints and single-instruction stepping.
- **Evaluation Mode**: Fast `eval()` execution for performance and parity testing.
- **Training-Test Mode**: `train()` mode with autograd tracking enabled for backward pass and gradient verification.

## 7. Project Persistence

Projects serialize to atomic, human-readable JSON files (`.nbp.json`) containing versioned schemas, canonical IR, module definitions, configurations, and layout viewports. Autosave and manual save use atomic rename patterns to prevent corruption.
