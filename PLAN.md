# bpTorch v0.1

## Executable UE5-Blueprint-Style Neural-Network Architecture Editor

---

# 0. Instructions to Codex

Build a local-first visual neural-network programming environment whose interaction model resembles Unreal Engine Blueprints.

The first complete architecture supported by the application must be Andrej Karpathy’s nanoGPT.

Do not interpret this project as:

* a static neural-network diagram generator;
* a simplified teaching toy;
* a no-code AutoML product;
* a wrapper around configuration files;
* a TensorBoard clone;
* a Netron clone;
* a frontend that generates code but cannot execute it;
* a training dashboard with a decorative architecture graph.

The visual graph must be the canonical executable model specification.

The primary relationship is:

```text
Visual graph
    ↓
Canonical model IR
    ↓
Validation and shape inference
    ↓
PyTorch runtime
    ↓
Actual tensors, logits and loss
```

The first release is successful only when a user can:

1. open a nanoGPT architecture;
2. understand the complete model at a high level;
3. click into progressively lower-level nodes;
4. edit architecture values through constrained controls;
5. see tensor types and dimensions on connections;
6. receive immediate visual errors for incompatible changes;
7. execute a real forward pass;
8. watch data move through the graph;
9. inspect tensor statistics on nodes and edges;
10. verify that the visual implementation numerically matches reference nanoGPT.

Treat this document as the implementation source of truth. Where a choice is not specified, choose the least complex option that preserves the architectural principles in this document.

Do not build later-roadmap features before v0.1 passes all acceptance tests.

---

# 1. Working Product Definition

Working title:

```text
bpTorch
```

The name is temporary and must be isolated in one application metadata/configuration location so it can be changed without refactoring.

One-sentence product definition:

> A Blueprint-style visual programming environment for PyTorch in which nodes are executable neural-network modules or tensor operations, wires are typed tensors, complex nodes contain editable subgraphs, and real execution can be inspected interactively.

The conceptual hierarchy is:

```text
Model
  ↓
Repeated stack
  ↓
Block
  ↓
Module
  ↓
Primitive operation
  ↓
Parameter or tensor
```

A user must be able to remain at the model level or drill down as far as needed.

The visual graph is not an illustration of separately maintained Python code. It is the model.

---

# 2. Why nanoGPT Is the First Target

Use the official `karpathy/nanoGPT` repository as the reference architecture.

At project initialization, pin the exact nanoGPT Git commit used for parity testing. Store it in:

```text
references/nanogpt.lock.json
```

Example structure:

```json
{
  "repository": "karpathy/nanoGPT",
  "commit": "<PINNED_COMMIT_HASH>",
  "reference_file": "model.py",
  "license": "MIT",
  "pinned_at": "<ISO_TIMESTAMP>"
}
```

Do not silently follow the repository’s `master` branch after pinning.

The official repository now describes nanoGPT as deprecated in favour of nanochat, but nanoGPT remains the appropriate first target because its model implementation is compact, explicit and built around a small number of recognizable components.

The reference architecture includes:

* custom LayerNorm with optional bias;
* causal self-attention;
* fused QKV projection;
* attention-head reshape and transpose operations;
* PyTorch scaled-dot-product attention or manual causal attention;
* output projection;
* MLP with `4 × n_embd` hidden dimension;
* GELU activation;
* dropout;
* pre-normalization Transformer blocks;
* residual additions;
* token embeddings;
* learned positional embeddings;
* a repeated Transformer stack;
* final LayerNorm;
* tied token-embedding and language-model-head weights;
* full-sequence logits and cross-entropy loss during training;
* last-token-only language-model head execution during inference.

This is complex enough to prove the editor is real but small enough to model completely.

---

# 3. Non-Negotiable Product Principles

## 3.1 The graph is executable

A connection on the canvas represents an actual tensor dependency.

Deleting a connection changes model execution.

Changing a node property changes the compiled PyTorch model.

There must never be a separately maintained visual graph and Python implementation that can diverge.

## 3.2 Progressive disclosure

At the top level, nanoGPT should look approximately like:

```text
[Token IDs] ─┐
             ├→ [Input Embeddings] → [Transformer Stack × N]
[Positions] ─┘                              ↓
                                      [Final Norm]
                                            ↓
                                        [LM Head]
                                            ↓
                                      [Logits / Loss]
```

Opening `Transformer Stack × N` reveals the repeated block.

Opening a block reveals:

```text
Input
  ├──────────────────────────────┐
  ↓                              │
LayerNorm                        │
  ↓                              │
Causal Self-Attention            │
  ↓                              │
Add ◄────────────────────────────┘
  ├──────────────────────────────┐
  ↓                              │
LayerNorm                        │
  ↓                              │
MLP                              │
  ↓                              │
Add ◄────────────────────────────┘
  ↓
Output
```

Opening attention reveals QKV, heads and attention computation.

Opening the MLP reveals its linear layers, GELU and dropout.

The default view must never expose hundreds of primitive operations at once.

## 3.3 Constrained editing before free-form editing

Use dropdowns, toggles and bounded numeric controls whenever the valid choices are known.

Examples:

```text
Attention implementation
[ PyTorch SDPA ▼ ]

Bias
[ Enabled ▼ ]

Activation
[ GELU ▼ ]

dtype
[ float32 ▼ ]

Number of heads
[ 1 | 2 | 4 | 8 ▼ ]
```

Do not require users to type enum names manually.

Do not permit invalid head counts when `n_embd` is known and the valid choices can be calculated.

## 3.4 Visible data contracts

Every data wire must be able to display:

* tensor dtype;
* symbolic or concrete shape;
* optional device;
* producer port;
* consumer port.

Example:

```text
float32 [B, T, 128]
```

## 3.5 Immediate and local errors

Do not wait for PyTorch to throw a matrix-multiplication exception during execution.

When possible, detect an error while editing and attach the diagnostic to the relevant node, property, port or edge.

## 3.6 Real PyTorch semantics

Do not make a simplified parallel neural-network implementation.

The runtime must construct and execute real PyTorch parameters, modules and tensor operations.

## 3.7 Local-first operation

The first version must:

* run on the user’s machine;
* require no account;
* require no cloud backend;
* send no telemetry by default;
* work without an internet connection after dependencies are installed.

## 3.8 Escape hatches come later

Do not add arbitrary Python code nodes in v0.1.

Custom Python nodes introduce execution-security, introspection and serialization problems. Establish the safe graph and runtime first.

---

# 4. Definition of the v0.1 Product Boundary

## 4.1 Included in v0.1

The first release must include:

* local web application;
* React/TypeScript node editor;
* Python/PyTorch runtime;
* canonical JSON-serializable model IR;
* typed input and output ports;
* symbolic tensor-shape propagation;
* node property inspector;
* dropdown-driven constrained options;
* hierarchical composite nodes;
* reusable subgraphs;
* repeated modules with independent weights;
* nanoGPT architecture template;
* real PyTorch compilation;
* deterministic one-batch execution;
* live node-execution visualization;
* tensor summary inspection;
* project save and load;
* undo and redo;
* nanoGPT numerical parity tests;
* frontend, backend and end-to-end test suites.

## 4.2 Explicitly excluded from v0.1

Do not implement:

* full training dashboard;
* distributed training;
* DDP;
* FSDP;
* Hugging Face model import;
* ONNX import;
* arbitrary PyTorch source import;
* `torch.export` import;
* `torch.fx` import;
* GGUF support;
* safetensors model import;
* LoRA;
* quantization;
* MoE routing;
* conditional expert execution;
* multimodal architectures;
* agent systems;
* cloud execution;
* user accounts;
* collaboration;
* plugin marketplace;
* automated architecture search;
* AutoML;
* code-generation assistant;
* arbitrary Python nodes;
* desktop packaging;
* mobile packaging.

A minimal backward pass may be used for parity tests, but an interactive training workflow is deferred until the architecture editor and runtime are correct.

---

# 5. Target User Experience

## 5.1 Application layout

Use five persistent interface regions:

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Top Bar: project | config | validate | compile | run | save         │
├──────────────┬──────────────────────────────────────┬────────────────┤
│ Node Palette │                                      │ Inspector      │
│              │                                      │                │
│ Inputs       │             Canvas                   │ Node settings  │
│ Layers       │                                      │ Bindings       │
│ Operations   │                                      │ Shape          │
│ Modules      │                                      │ Parameters     │
│ Outputs      │                                      │ Diagnostics    │
│              │                                      │                │
├──────────────┴──────────────────────────────────────┴────────────────┤
│ Bottom Drawer: trace | tensor | errors | logs | parity              │
└──────────────────────────────────────────────────────────────────────┘
```

Add a breadcrumb row immediately above the canvas:

```text
nanoGPT / Transformer Stack / Block / Causal Self-Attention
```

The breadcrumb must be navigable.

## 5.2 Top bar

The top bar should contain:

* project name;
* dirty-state indicator;
* current graph path;
* model configuration selector;
* validation status;
* compile button;
* run button;
* step button while paused;
* stop button while running;
* save button;
* overflow menu.

Do not place every feature in the top bar.

## 5.3 Node palette

The left palette must support:

* category grouping;
* text search;
* drag-to-canvas;
* click-to-add;
* built-in versus custom badge;
* primitive versus composite badge.

Initial categories:

```text
Inputs
Parameters
Layers
Tensor Operations
Attention
Composite Modules
Outputs
Debug
```

## 5.4 Canvas

Use React Flow for the canvas and interaction primitives.

Required canvas behaviours:

* pan;
* zoom;
* zoom-to-fit;
* minimap;
* node selection;
* edge selection;
* marquee selection;
* multi-select;
* node dragging;
* connection dragging;
* deletion;
* copy;
* paste;
* duplicate;
* keyboard navigation;
* context menu;
* undo;
* redo;
* auto-layout button.

React Flow already provides the base mechanics for custom nodes, handles, dragging, selection, panning and zooming; do not reimplement those primitives.

## 5.5 Node appearance

Every collapsed node must show:

* display name;
* node category icon or label;
* input ports;
* output ports;
* compact property summary;
* output shape when known;
* validation status;
* modified/custom indicator;
* breakpoint indicator;
* execution state during a run.

Example:

```text
┌──────────────────────────────┐
│ Causal Self-Attention        │
│ C=128 | heads=4 | SDPA       │
│                              │
● x [B,T,128]    [B,T,128] ●   │
└──────────────────────────────┘
```

Do not display every property on the node.

## 5.6 Edge appearance

A data edge represents a tensor.

When space permits, show:

```text
float32 [B,T,128]
```

When zoomed out, reduce this to:

```text
[B,T,C]
```

Clicking an edge opens its details in the bottom drawer.

Do not use separate UE-style execution wires in v0.1. Neural execution is dataflow-driven: a node executes when all required inputs are available.

Control-flow wires can be introduced later for conditional architectures.

## 5.7 Opening subgraphs

Double-clicking a composite node opens its internal graph.

Also provide:

```text
Open Internals
```

in the context menu and inspector.

Each graph must preserve its own viewport and selection state.

The browser-style back and forward actions should navigate graph history.

## 5.8 Built-in versus custom modules

Built-in module definitions are viewable but initially read-only.

When the user tries to structurally modify a built-in module, offer:

```text
Create editable copy
```

This creates a custom module definition with lineage metadata:

```json
{
  "derived_from": "builtin.nanogpt.mlp@1",
  "modified": true
}
```

Display custom versions with an asterisk or badge:

```text
MLP*
```

Property changes explicitly exposed by the built-in module do not require forking.

Structural rewiring does.

## 5.9 Repeat-node semantics

The Transformer stack must be represented as:

```text
[Transformer Block × n_layer]
```

The repeat node means:

* one architecture definition;
* `n_layer` runtime instances;
* independent parameters for every instance;
* no implicit weight sharing.

The inspector must explicitly state:

```text
Instances: 6
Weights: independent
```

In v0.1, editing the repeated block definition changes all instances structurally.

Provide an instance selector for inspection:

```text
View instance
[ 0 ▼ ]
```

Do not implement per-instance architectural detachment in v0.1.

---

# 6. Recommended Technology Stack

## 6.1 Frontend

Use:

```text
React
TypeScript with strict mode
Vite
@xyflow/react
Zustand
Zod
Vitest
React Testing Library
Playwright
```

Use a component library only if it does not impose a heavy design system or make node editing difficult.

Prefer ordinary accessible React controls and CSS variables.

Do not use a large application framework when Vite is sufficient.

## 6.2 Backend

Use:

```text
Python 3.12
PyTorch
FastAPI
Pydantic v2
Uvicorn
pytest
Hypothesis
Ruff
Pyright or mypy
```

Use `uv` for Python dependency and environment management.

## 6.3 Communication

Use:

* HTTP for registry, validation, compile and project operations;
* WebSocket for execution and trace events.

The frontend and backend run locally.

## 6.4 Initial deployment form

v0.1 is a local web application:

```text
React frontend
    ↕ localhost
FastAPI/PyTorch backend
```

Provide one developer command:

```bash
make dev
```

Desktop packaging with Tauri or another shell is deferred until the application is functionally correct.

Do not begin by solving Python-sidecar packaging.

## 6.5 Version pinning

Pin exact dependency versions in lockfiles.

Do not hard-code version assumptions into the project plan.

At implementation time:

* use the current stable React Flow major supported by the project;
* use a supported PyTorch 2.x version;
* record exact versions in lockfiles;
* ensure CPU-only CI works.

---

# 7. Repository Structure

Create a monorepo approximately like:

```text
bptorch/
├── PLAN.md
├── README.md
├── ARCHITECTURE.md
├── PROGRESS.md
├── TESTING.md
├── THIRD_PARTY_NOTICES.md
├── Makefile
├── pyproject.toml
├── uv.lock
├── package.json
├── pnpm-lock.yaml
├── apps/
│   ├── web/
│   │   ├── src/
│   │   │   ├── app/
│   │   │   ├── canvas/
│   │   │   ├── components/
│   │   │   ├── inspector/
│   │   │   ├── palette/
│   │   │   ├── trace/
│   │   │   ├── stores/
│   │   │   ├── api/
│   │   │   ├── schemas/
│   │   │   └── tests/
│   │   └── ...
│   └── server/
│       ├── neural_blueprint/
│       │   ├── api/
│       │   ├── ir/
│       │   ├── registry/
│       │   ├── validation/
│       │   ├── shapes/
│       │   ├── runtime/
│       │   ├── tracing/
│       │   ├── projects/
│       │   ├── templates/
│       │   └── parity/
│       └── tests/
├── packages/
│   └── contracts/
│       ├── json-schema/
│       └── generated-typescript/
├── references/
│   ├── nanogpt.lock.json
│   ├── nanogpt/
│   └── licenses/
├── examples/
│   ├── linear-mlp/
│   └── nanogpt/
└── scripts/
    ├── pin_nanogpt.py
    ├── generate_contracts.py
    └── run_parity.py
```

The exact directories can change, but responsibilities must remain separated.

Avoid giant files that combine:

* schema definitions;
* runtime execution;
* API routing;
* shape inference;
* node implementations.

---

# 8. Canonical Model Intermediate Representation

## 8.1 Core requirement

The model IR is the source of truth.

The frontend edits it.

The validator consumes it.

The runtime compiles it.

The project serializer saves it.

The parity harness runs it.

Do not use React Flow’s node objects as the canonical architecture representation. React Flow state is a UI projection of the model IR.

## 8.2 Project structure

Define a versioned project schema:

```json
{
  "schema_version": 1,
  "project": {
    "id": "project_uuid",
    "name": "nanoGPT Experiment",
    "created_at": "ISO_TIMESTAMP",
    "updated_at": "ISO_TIMESTAMP"
  },
  "model": {
    "root_graph_id": "graph_gpt",
    "config": {},
    "graphs": {},
    "weight_bindings": []
  },
  "ui": {
    "graph_viewports": {},
    "node_positions": {},
    "open_graph_id": "graph_gpt"
  }
}
```

Keep architecture semantics separate from UI layout state.

## 8.3 Graph definition

```json
{
  "id": "graph_block",
  "name": "Transformer Block",
  "kind": "module",
  "interface": {
    "inputs": [],
    "outputs": []
  },
  "nodes": [],
  "edges": []
}
```

## 8.4 Node instance

```json
{
  "id": "node_uuid",
  "definition_id": "builtin.linear@1",
  "display_name": "QKV Projection",
  "properties": {
    "in_features": {
      "kind": "config_ref",
      "key": "n_embd"
    },
    "out_features": {
      "kind": "expression",
      "expression": {
        "op": "multiply",
        "left": 3,
        "right": {
          "kind": "config_ref",
          "key": "n_embd"
        }
      }
    },
    "bias": {
      "kind": "config_ref",
      "key": "bias"
    }
  },
  "metadata": {
    "breakpoint": false,
    "disabled": false
  }
}
```

## 8.5 Edge

```json
{
  "id": "edge_uuid",
  "source": {
    "node_id": "node_a",
    "port_id": "output"
  },
  "target": {
    "node_id": "node_b",
    "port_id": "input"
  }
}
```

## 8.6 Port definition

```json
{
  "id": "input",
  "display_name": "Input",
  "direction": "input",
  "required": true,
  "multiplicity": "single",
  "tensor_type": {
    "dtype_family": "floating",
    "rank": 3
  }
}
```

## 8.7 Tensor specification

```json
{
  "dtype": "float32",
  "shape": [
    {"kind": "symbol", "name": "B"},
    {"kind": "symbol", "name": "T"},
    {"kind": "config_ref", "key": "n_embd"}
  ],
  "device": "runtime"
}
```

## 8.8 Stable IDs

All graphs, nodes, ports and edges need stable IDs.

Names are not identities.

Renaming a node must not invalidate saved state, traces or weight mappings.

## 8.9 Schema migrations

Every project has `schema_version`.

Implement a migration registry:

```python
migrate_v1_to_v2(project)
```

Loading an unsupported future schema must produce a clear error rather than silently discarding fields.

---

# 9. Model-Level Configuration and Bindings

## 9.1 nanoGPT configuration

Expose these model-level values:

```text
block_size
vocab_size
n_layer
n_head
n_embd
dropout
bias
```

Use nanoGPT-compatible defaults for the template, but provide a small default demonstration configuration to avoid allocating GPT-2-sized tensors accidentally.

Recommended demonstration configuration:

```text
block_size = 32
vocab_size = 128
n_layer = 2
n_head = 4
n_embd = 64
dropout = 0.0
bias = true
```

## 9.2 Derived values

Support safe derived expressions:

```text
head_dim = n_embd // n_head
mlp_hidden = 4 * n_embd
qkv_dim = 3 * n_embd
```

Do not store these as independently editable values when they are intended to remain derived.

## 9.3 Binding types

A property value can be:

```text
Literal
Model configuration reference
Parent-module property reference
Safe expression
```

Do not permit arbitrary Python expressions.

Allowed expression operations in v0.1:

```text
add
subtract
multiply
integer divide
minimum
maximum
```

No function calls, imports, attribute access or code evaluation.

## 9.4 Inspector display

A bound property should appear as:

```text
Input features
[ n_embd ▼ ] = 64
```

A derived property should appear as:

```text
Output features
[ 3 × n_embd ] = 192
```

The user can switch from a binding to a literal value using a dropdown:

```text
Value source
[ Model configuration ▼ ]
```

## 9.5 Constraint propagation

Changing:

```text
n_embd: 64 → 96
```

must update all bound dimensions.

Changing:

```text
n_head: 4 → 8
```

must update `head_dim`.

A head-count dropdown should list only valid divisors of `n_embd`, unless the user explicitly chooses an advanced manual mode.

---

# 10. Symbolic Shape and Type System

## 10.1 Symbolic dimensions

Support:

```text
B = batch size
T = sequence length
C = embedding dimension
V = vocabulary size
H = MLP hidden dimension
NH = number of heads
HD = head dimension
```

Some dimensions are runtime symbols, while others are bound to model configuration.

## 10.2 Shape expressions

Represent shapes as structured expressions, not strings.

Example:

```json
[
  {"kind": "symbol", "name": "B"},
  {"kind": "symbol", "name": "T"},
  {"kind": "config_ref", "key": "n_embd"}
]
```

## 10.3 Unknown dimensions

Support an explicit unknown dimension:

```text
?
```

Do not convert uncertainty into a false validation failure.

A graph can remain partially valid while some dimensions are unknown.

## 10.4 Initial dtype set

Support:

```text
int64
int32
float32
float16
bfloat16
bool
```

The runtime may restrict particular dtypes depending on device.

The initial nanoGPT parity path should use CPU `float32`.

## 10.5 Validation levels

Implement four validation passes.

### Pass 1: schema validation

Check:

* required fields;
* valid IDs;
* recognized node definitions;
* valid property types;
* valid edge structure.

### Pass 2: structural graph validation

Check:

* all required ports connected;
* single-input ports do not receive multiple edges;
* no prohibited cycles;
* no orphan outputs where required;
* root graph has required outputs;
* composite interfaces match internal graphs.

### Pass 3: tensor type and shape validation

Check:

* dtype compatibility;
* rank compatibility;
* dimension constraints;
* add compatibility;
* linear input dimension;
* embedding input integer dtype;
* head divisibility;
* reshape element-count consistency where provable.

### Pass 4: model-semantic validation

Check nanoGPT-specific invariants:

* `n_embd % n_head == 0`;
* sequence length does not exceed `block_size`;
* LM-head input dimension equals `n_embd`;
* LM-head output dimension equals `vocab_size`;
* tied weights have compatible shapes;
* repeated block count equals `n_layer`.

## 10.6 Diagnostic format

Return structured diagnostics:

```json
{
  "code": "E_LINEAR_INPUT_DIM",
  "severity": "error",
  "message": "Linear expects a final dimension of 96 but receives 64.",
  "node_id": "linear_1",
  "port_id": "input",
  "edge_id": "edge_4",
  "expected": "[B,T,96]",
  "actual": "[B,T,64]",
  "suggestions": [
    "Set in_features to 64.",
    "Insert a projection from 64 to 96."
  ]
}
```

Do not expose a raw Python traceback as the primary error.

Provide developer details in an expandable panel.

---

# 11. Node Definition Registry

## 11.1 Registry purpose

Every supported node type must be declared through a backend registry.

The registry is authoritative for:

* display name;
* category;
* input ports;
* output ports;
* property schema;
* property UI hints;
* shape inference;
* parameter counting;
* runtime construction;
* optional subgraph definition;
* documentation.

The frontend requests the registry from the backend.

Do not manually hard-code a second independent node catalogue in React.

## 11.2 Conceptual interface

```python
class NodeDefinition:
    type_id: str
    version: int
    display_name: str
    category: str

    def property_schema(self) -> dict:
        ...

    def input_ports(self, properties) -> list:
        ...

    def output_ports(self, properties) -> list:
        ...

    def infer_shapes(self, inputs, properties, context):
        ...

    def validate(self, inputs, properties, context):
        ...

    def parameter_spec(self, properties, context):
        ...

    def build_runtime(self, properties, context):
        ...
```

## 11.3 Registry contract tests

Every registered node must pass automated contract tests proving that it:

* has a unique versioned type ID;
* returns valid JSON schema;
* declares at least one output unless it is a terminal node;
* can be serialized;
* can be deserialized;
* supplies shape logic;
* supplies runtime logic or explicitly declares itself visualization-only;
* handles invalid properties predictably.

No visualization-only nodes are expected in the nanoGPT execution graph.

---

# 12. Initial Primitive Node Library

Implement the smallest complete set needed for nanoGPT.

## 12.1 Inputs and interfaces

```text
TokenInput
TargetInput
ModuleInput
ModuleOutput
GraphOutput
RuntimeMode
```

## 12.2 Tensor construction

```text
Arange
Constant
CausalMask
```

## 12.3 Parameterized PyTorch modules

```text
Embedding
Linear
LayerNorm
Dropout
GELU
```

## 12.4 Tensor operations

```text
Add
Split
Reshape
Transpose
Contiguous
MatMul
Scale
MaskedFill
Softmax
SelectTimeStep
FlattenDimensions
```

## 12.5 Attention operations

```text
ScaledDotProductAttention
ManualCausalAttention
SplitHeads
MergeHeads
SplitQKV
```

Prefer domain-specific `SplitHeads`, `MergeHeads` and `SplitQKV` nodes in the standard nanoGPT graph because they are more understandable and safer than arbitrary reshape expressions.

Generic reshape and transpose nodes may still exist in the advanced palette.

## 12.6 Loss and outputs

```text
LanguageModelHead
CrossEntropyLoss
LogitsOutput
LossOutput
```

`LanguageModelHead` may internally use `Linear`, but it should exist as a semantic composite node.

## 12.7 Graph/module operations

```text
ModuleCall
RepeatModule
```

## 12.8 Residual connections

A residual is represented explicitly as an `Add` node with two input edges.

Do not hide residual behaviour in an opaque block implementation.

---

# 13. Composite Module System

## 13.1 Composite principle

A composite node contains another graph.

The internal graph has explicit interface nodes:

```text
[Module Input] → internal nodes → [Module Output]
```

## 13.2 Required built-in composite modules

Implement:

```text
nanoGPT Input Embeddings
nanoGPT Causal Self-Attention
nanoGPT MLP
nanoGPT Transformer Block
nanoGPT Transformer Stack
nanoGPT Model
```

## 13.3 Property exposure

A composite module can expose selected internal properties at its parent level.

Example:

```text
Causal Self-Attention
├── n_embd
├── n_head
├── dropout
├── bias
└── implementation
```

These map to properties of internal nodes.

Do not expose every internal property by default.

## 13.4 Internal graph editing

When a user creates an editable copy of a composite module, they can:

* add nodes;
* delete nodes;
* reconnect nodes;
* change internal properties;
* expose new properties;
* rename internal nodes;
* add module inputs and outputs.

## 13.5 Interface compatibility

When an internal graph is changed, validate the parent module interface.

If a user deletes the internal output path, mark every instance of that custom module invalid.

## 13.6 Definition versus instance

Keep these distinct:

```text
Module definition
Module instance
```

Editing the definition changes all instances using it.

Changing an exposed instance property changes only that instance.

The UI must clearly label which is being edited.

---

# 14. Property Inspector

## 14.1 Inspector sections

When a node is selected, show:

```text
Identity
Configuration
Input contract
Output contract
Derived values
Parameter count
Execution
Diagnostics
Documentation
```

## 14.2 Supported control types

Implement:

* text field for names only;
* integer field;
* floating-point field;
* slider plus numeric field;
* toggle;
* enum dropdown;
* searchable dropdown;
* binding selector;
* read-only derived value;
* reset-to-default button.

## 14.3 Conditional fields

Fields must be context-sensitive.

Example:

```text
Attention implementation
[ PyTorch SDPA ▼ ]
```

When `PyTorch SDPA` is selected, manual-mask-specific options remain hidden.

When `Manual` is selected, show:

```text
Attention dropout
Mask type
Scale mode
Softmax dimension
```

## 14.4 Valid option providers

Allow a field’s options to be calculated from graph context.

Example:

```text
n_embd = 96
```

Valid `n_head` options:

```text
1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96
```

## 14.5 Transactional property edits

For a property change:

1. update a draft value;
2. request validation;
3. preview affected shapes;
4. commit if valid or preserve the value with visible errors if invalid;
5. create one undo-history entry.

Do not create an undo entry for every keystroke.

## 14.6 Propagation preview

When changing a model-level value, show a compact impact summary:

```text
Changing n_embd from 64 to 96 will update:

12 Linear dimensions
4 LayerNorm dimensions
2 Embedding tables
1 LM head
Estimated parameters: 121,216 → 262,464
```

Parameter numbers above are illustrative; calculate real values.

---

# 15. Parameter Accounting

## 15.1 Per-node counts

Every parameterized node must report:

```text
trainable parameters
frozen parameters
unique parameters
shared parameters
```

## 15.2 Model totals

Display:

```text
Total unique parameters
Trainable parameters
Frozen parameters
Shared parameter references
```

## 15.3 Weight tying

nanoGPT ties:

```text
TokenEmbedding.weight
LMHead.weight
```

Represent this explicitly as a weight binding:

```json
{
  "source": {
    "node_id": "token_embedding",
    "parameter": "weight"
  },
  "target": {
    "node_id": "lm_head",
    "parameter": "weight"
  },
  "mode": "share"
}
```

The parameter counter must count this tensor once.

Visually display weight sharing separately from tensor-data edges, for example with:

```text
Shared weight
```

badges and an optional dashed relationship line.

Do not route runtime data through a weight-sharing edge.

## 15.4 Repeated modules

A repeat node with `count = N` and independent weights contributes:

```text
N × parameters of module definition
```

Shared architecture does not imply shared parameters.

---

# 16. Runtime and Compilation Architecture

## 16.1 Runtime principle

Compile the IR into an actual `torch.nn.Module`.

Do not use generated source code as the primary runtime.

Readable source export may be added later.

## 16.2 Compiled graph module

Create a runtime similar to:

```python
class CompiledGraphModule(torch.nn.Module):
    def __init__(self, execution_plan, registered_modules, parameter_bindings):
        super().__init__()
        ...

    def forward(self, **inputs):
        ...
```

## 16.3 Execution plan

Compilation produces:

```text
validated graph
    ↓
topological order
    ↓
module registration plan
    ↓
parameter-binding plan
    ↓
execution instructions
```

Each execution instruction contains:

```text
node path
node type
input value references
output value references
runtime callable
trace policy
```

## 16.4 Module registration

Parameterized nodes must be registered under `nn.ModuleDict`, `nn.ModuleList` or ordinary module attributes so that:

* `parameters()` works;
* `state_dict()` works;
* optimizers work;
* `.train()` and `.eval()` work;
* `.to(device)` works.

## 16.5 Functional operations

Nodes such as:

```text
Add
Reshape
Transpose
Split
MatMul
Softmax
```

execute as PyTorch tensor operations in the graph runtime.

Because these are not necessarily `nn.Module` objects, runtime-level instrumentation is required.

## 16.6 Composite compilation

Compile composite modules recursively.

Runtime node paths must remain hierarchical:

```text
gpt/blocks[1]/attention/qkv_projection
```

Do not flatten all paths into opaque generated identifiers.

## 16.7 Repeat compilation

`RepeatModule` must instantiate independent module copies in an `nn.ModuleList`.

Do not accidentally reuse one module object N times.

## 16.8 Graph hash

Calculate a deterministic architecture hash from canonical model IR.

A compiled session stores:

```text
graph_hash
config_hash
runtime_version
```

If the graph changes after compilation, mark the session:

```text
STALE
```

Do not display old execution results as though they came from the current graph.

## 16.9 Runtime modes

Support:

```text
inspection
evaluation
training-test
```

### Inspection

* `model.eval()`;
* no gradient tracking;
* trace every selected node;
* intended for stepping and tensor inspection.

### Evaluation

* `model.eval()`;
* no gradient tracking;
* minimal tracing;
* intended for parity and performance.

### Training-test

* `model.train()`;
* gradient tracking enabled;
* used by tests and later training UI;
* no interactive stepping required in v0.1.

## 16.10 Device support

Initial device choices:

```text
auto
cpu
cuda
mps
```

CPU must always work.

The nanoGPT numerical parity suite must run on CPU in CI.

---

# 17. nanoGPT Architecture Template

## 17.1 Root model graph

Build this graph:

```text
Token IDs [B,T] ─→ Token Embedding [B,T,C] ───────┐
                                                   │
Position Range [T] → Position Embedding [T,C] ─────┤
                                                   ▼
                                                  Add
                                                   ↓
                                                Dropout
                                                   ↓
                                      Transformer Block × n_layer
                                                   ↓
                                             Final LayerNorm
                                                   ↓
                                                LM Head
                                                   ↓
                                              Logits Output
                                                   │
Targets [B,T] ─────────────────────────────────────┤
                                                   ↓
                                           Cross-Entropy Loss
                                                   ↓
                                               Loss Output
```

## 17.2 Input embeddings graph

Internals:

```text
Token IDs → Token Embedding ───────────────┐
                                           ├→ Add → Dropout → Output
Position IDs → Position Embedding ─────────┘
```

Position IDs are generated from sequence length:

```text
0, 1, 2, ..., T-1
```

Validate:

```text
T ≤ block_size
```

## 17.3 Transformer block graph

Use pre-normalization exactly:

```text
Input x
  ├──────────────────────────────────────────────┐
  ↓                                              │
LayerNorm 1                                      │
  ↓                                              │
Causal Self-Attention                            │
  ↓                                              │
Residual Add ◄───────────────────────────────────┘
  │
  ├──────────────────────────────────────────────┐
  ↓                                              │
LayerNorm 2                                      │
  ↓                                              │
MLP                                              │
  ↓                                              │
Residual Add ◄───────────────────────────────────┘
  ↓
Output
```

The first residual output becomes the input to the second residual path.

## 17.4 MLP graph

```text
Input [B,T,C]
    ↓
Linear C → 4C
    ↓
GELU
    ↓
Linear 4C → C
    ↓
Dropout
    ↓
Output [B,T,C]
```

Use PyTorch `nn.GELU()` semantics matching nanoGPT.

## 17.5 Attention graph

Use a fused QKV projection to preserve reference weight structure.

```text
Input [B,T,C]
    ↓
Linear C → 3C
    ↓
Split QKV
  ├──────── Q [B,T,C]
  ├──────── K [B,T,C]
  └──────── V [B,T,C]
             ↓
Split Heads
[B,NH,T,HD]
             ↓
Causal Attention
             ↓
Merge Heads
[B,T,C]
             ↓
Output Projection C → C
             ↓
Residual Dropout
             ↓
Output [B,T,C]
```

Do not implement three independent Q, K and V linear modules in the standard nanoGPT template. The reference implementation uses one fused projection.

## 17.6 Attention implementation dropdown

Expose:

```text
Implementation
[ PyTorch SDPA ▼ ]
```

Options:

```text
PyTorch SDPA
Manual causal attention
```

The manual path should visually expand to:

```text
Q × Kᵀ
   ↓
Scale by 1 / sqrt(HD)
   ↓
Apply causal mask
   ↓
Softmax
   ↓
Attention dropout
   ↓
Multiply by V
```

Use the manual path for strict CPU parity testing when operation-order differences make comparison clearer.

## 17.7 Final output behaviour

During training with targets:

```text
Final hidden state [B,T,C]
    ↓
LM Head
    ↓
Logits [B,T,V]
    ↓
Flatten
    ↓
Cross entropy with targets
```

Use:

```text
ignore_index = -1
```

For inference without targets, nanoGPT applies the LM head only to the final time position. Implement this through a constrained semantic node or execution mode:

```text
Select final time step
    ↓
LM Head
    ↓
Logits [B,1,V]
```

Do not introduce a fully general control-flow language solely for this optimization.

## 17.8 Initialization

Match nanoGPT initialization:

* linear weights: normal mean `0.0`, standard deviation `0.02`;
* linear biases: zeros;
* embedding weights: normal mean `0.0`, standard deviation `0.02`;
* residual projection weights named semantically as `c_proj.weight`: standard deviation scaled by `1 / sqrt(2 × n_layer)`.

Initialization must be part of runtime/model-definition semantics, not a hidden parity-test patch.

---

# 18. Execution Tracing and Blueprint-Style Debugging

## 18.1 Trace events

The runtime must emit structured events:

```json
{
  "sequence": 42,
  "event": "node_finished",
  "session_id": "session_uuid",
  "graph_hash": "hash",
  "node_path": "gpt/blocks[0]/attention/qkv_projection",
  "timestamp_ns": 123456789,
  "duration_ns": 84210,
  "outputs": []
}
```

Event types:

```text
run_started
node_started
node_finished
node_paused
node_failed
run_finished
run_cancelled
```

## 18.2 Canvas execution state

During a run:

* pending nodes remain neutral;
* the current node receives an execution highlight;
* completed nodes receive a temporary completion state;
* failed nodes receive an error state;
* the active edge can animate in the direction of data flow.

Do not make animation so fast that the user cannot understand it.

Provide speed controls:

```text
Instant
Fast
Normal
Step
```

## 18.3 Tensor summaries

Capture by default:

```text
shape
dtype
device
numel
mean
standard deviation
minimum
maximum
L2 norm
zero fraction
NaN count
positive infinity count
negative infinity count
```

Only compute statistics valid for the dtype.

For integer tensors, show:

```text
minimum
maximum
unique-count estimate
sample values
```

## 18.4 Full tensor handling

Do not stream entire large tensors to the browser by default.

Return:

* summary;
* a small deterministic sample;
* optional histogram;
* optional user-requested slice.

Set a configurable maximum number of sampled values.

## 18.5 Edge inspection

Clicking an edge after execution shows:

```text
Producer
Consumer
Shape
dtype
Device
Summary statistics
Sample
Execution timestamp
```

## 18.6 Node inspection

Clicking a completed node shows:

```text
Input tensor summaries
Output tensor summaries
Runtime duration
Parameter summary
Module training/evaluation state
```

## 18.7 Breakpoints

Allow a breakpoint on any executable node.

In inspection mode:

1. execution runs until the breakpoint;
2. the session retains intermediate tensors;
3. the UI enters a paused state;
4. the user can inspect available inputs;
5. `Step` executes one node;
6. `Continue` resumes;
7. `Stop` releases retained tensors.

Implement stepping first for CPU inspection mode.

Do not attempt to pause a compiled fused CUDA graph in v0.1.

## 18.8 Hierarchical trace paths

For repeated blocks, show both module definition and runtime instance:

```text
Transformer Block
Instance: 1 of 6
Path: gpt/blocks[0]
```

A user must be able to switch between block instances and inspect differences in actual activations.

---

# 19. Backend API Contract

Use versioned routes.

## 19.1 Registry

```text
GET /api/v1/registry/nodes
GET /api/v1/registry/modules
```

Returns node definitions, property schemas and UI metadata.

## 19.2 Validation

```text
POST /api/v1/graphs/validate
```

Request:

```json
{
  "project": {}
}
```

Response:

```json
{
  "valid": true,
  "graph_hash": "hash",
  "resolved_shapes": {},
  "parameter_summary": {},
  "diagnostics": []
}
```

## 19.3 Compilation

```text
POST /api/v1/models/compile
```

Response:

```json
{
  "session_id": "uuid",
  "graph_hash": "hash",
  "device": "cpu",
  "parameter_summary": {}
}
```

## 19.4 Run

```text
POST /api/v1/sessions/{session_id}/run
```

Request:

```json
{
  "mode": "inspection",
  "inputs": {
    "token_ids": [[1, 2, 3, 4]],
    "targets": [[2, 3, 4, 5]]
  },
  "trace": {
    "enabled": true,
    "speed": "normal"
  }
}
```

## 19.5 Step and continue

```text
POST /api/v1/sessions/{session_id}/step
POST /api/v1/sessions/{session_id}/continue
POST /api/v1/sessions/{session_id}/stop
```

## 19.6 Trace WebSocket

```text
WS /api/v1/sessions/{session_id}/events
```

## 19.7 Tensor details

```text
GET /api/v1/sessions/{session_id}/tensors/{tensor_id}/summary
POST /api/v1/sessions/{session_id}/tensors/{tensor_id}/slice
```

## 19.8 Session lifecycle

Sessions must be explicitly disposable.

Release model and tensor references after:

* stop;
* completion plus retention timeout;
* graph recompilation;
* user session deletion.

Prevent memory leaks across repeated runs.

---

# 20. Frontend State Architecture

## 20.1 State divisions

Separate stores for:

```text
project IR
canvas/UI layout
selection
validation results
compiled session
trace state
undo/redo history
```

Do not put all application state into one undifferentiated Zustand store.

## 20.2 Canonical versus derived state

Canonical frontend state:

```text
project IR
UI positions
user selection
```

Derived state:

```text
React Flow nodes
React Flow edges
resolved shape labels
parameter labels
diagnostic badges
```

Do not persist derived React Flow objects as the architecture source of truth.

## 20.3 Undo and redo

Commands that create history entries:

* add node;
* remove node;
* connect edge;
* remove edge;
* move nodes as one completed action;
* change property;
* create custom module;
* rename module;
* change model configuration.

Do not put:

* selection changes;
* canvas zoom;
* trace events;
* validation responses

into architecture undo history.

## 20.4 Optimistic editing

The frontend can display immediate structural changes while backend validation is debounced.

The backend remains authoritative for final shape and semantic validation.

Local handle-category checks may reject clearly incompatible connections before sending a request.

---

# 21. Persistence

## 21.1 Project file

Use a human-readable JSON project file during v0.1.

Suggested extension:

```text
.nbp.json
```

The extension is provisional.

## 21.2 Save contents

Save:

* canonical model IR;
* module definitions;
* model configuration;
* node layout;
* per-graph viewport;
* project metadata.

Do not save:

* active runtime session;
* intermediate tensors;
* stale validation caches;
* WebSocket state.

## 21.3 Atomic writes

Write to a temporary file and atomically replace the prior project file.

Do not risk corrupting the project if the process terminates during saving.

## 21.4 Autosave

Autosave may be implemented after manual save is reliable.

Do not make autosave the only persistence mechanism.

## 21.5 Round-trip test

For every sample project:

```text
load
serialize
reload
```

must preserve architecture semantics and stable IDs.

---

# 22. nanoGPT Reference and Numerical Parity Harness

## 22.1 Reference code

Pin a specific nanoGPT commit.

Either:

* vendor the minimal reference implementation under its MIT licence; or
* include a deterministic script that fetches the pinned commit and verify its hash.

CI must not depend on a moving branch.

Add attribution to:

```text
THIRD_PARTY_NOTICES.md
```

## 22.2 Reference configuration

Use a small deterministic configuration:

```text
block_size = 8
vocab_size = 32
n_layer = 2
n_head = 2
n_embd = 16
dropout = 0.0
bias = true
```

Use:

```text
device = CPU
dtype = float32
```

## 22.3 Weight transfer

Implement an explicit semantic weight map between reference nanoGPT and visual-runtime paths.

Example:

```text
transformer.wte.weight
→ gpt/input_embeddings/token_embedding.weight

transformer.h.0.ln_1.weight
→ gpt/blocks[0]/ln_1.weight

transformer.h.0.attn.c_attn.weight
→ gpt/blocks[0]/attention/qkv_projection.weight
```

Do not rely on dictionary iteration order.

## 22.4 Weight-map completeness test

Fail if:

* a reference parameter has no target;
* a visual parameter has no source;
* shapes differ;
* tied weights are not tied;
* one parameter is mapped twice unexpectedly.

## 22.5 Forward parity

With identical weights and input token IDs, compare:

```text
reference logits
visual-runtime logits
reference loss
visual-runtime loss
```

Initial CPU tolerance target:

```python
torch.testing.assert_close(
    actual,
    expected,
    rtol=1e-5,
    atol=1e-6,
)
```

Broaden tolerances only with a documented numerical reason.

## 22.6 Intermediate parity

Compare key intermediate tensors:

```text
token embeddings
position embeddings
first block LayerNorm output
first block QKV projection
first block attention output
first block MLP output
final LayerNorm output
```

This makes parity failures localizable.

## 22.7 Gradient parity

Run:

```text
forward
loss.backward()
```

Compare gradients for:

```text
token embedding / tied LM-head weight
first QKV projection weight
first attention output projection
first MLP input projection
final LayerNorm weight
```

## 22.8 Optimizer-step parity

Using identical optimizer state and one deterministic batch:

1. run forward;
2. run backward;
3. apply one optimizer step;
4. compare selected updated parameters.

This confirms the compiled graph is trainable, even though the v0.1 UI does not yet expose training.

## 22.9 Attention implementation parity

Run separate tests for:

```text
Manual causal attention
PyTorch SDPA
```

Manual attention is the strict semantic reference.

SDPA may use a slightly broader tolerance if required by kernel implementation.

## 22.10 Parameter-count parity

Compare:

```text
reference unique parameter count
visual-runtime unique parameter count
```

Verify weight tying is counted once.

## 22.11 Inference-path parity

Without targets:

* select the final sequence position;
* run the LM head only on that position;
* verify output shape `[B,1,V]`;
* compare logits with reference nanoGPT.

## 22.12 Parity status in UI

The bundled nanoGPT template should display:

```text
Reference parity
✓ Passed for bundled template
```

After user modification:

```text
Reference parity
Not applicable: architecture modified
```

Do not imply an edited architecture still matches nanoGPT.

---

# 23. Testing Strategy

## 23.1 Backend unit tests

Test:

* IR serialization;
* schema validation;
* ID uniqueness;
* expression evaluation;
* shape inference;
* shape unification;
* graph topological ordering;
* cycle detection;
* port multiplicity;
* parameter counting;
* weight tying;
* repeat-module expansion;
* module registration;
* tensor summaries;
* trace event ordering;
* session cleanup.

## 23.2 Property-based tests

Use Hypothesis for:

* valid linear dimensions;
* invalid linear dimensions;
* add-compatible shapes;
* add-incompatible shapes;
* valid head divisors;
* invalid head configurations;
* serialization round trips;
* graph-order independence.

## 23.3 Node contract tests

Automatically enumerate the registry and test every node definition.

## 23.4 Frontend unit tests

Test:

* registry rendering;
* property-control selection;
* dropdown options;
* node selection;
* inspector updates;
* diagnostic display;
* breadcrumb navigation;
* repeat-instance selector;
* stale-session badge;
* trace-state transitions.

## 23.5 Canvas interaction tests

Test:

* drag node onto canvas;
* connect compatible ports;
* reject incompatible ports;
* delete node;
* undo deletion;
* redo deletion;
* copy and paste;
* open composite;
* return through breadcrumb;
* auto-layout;
* preserve node locations.

## 23.6 Integration tests

Test the full frontend/backend contract:

```text
create graph
validate graph
compile graph
run graph
receive trace events
request tensor summary
```

## 23.7 End-to-end Playwright tests

Required end-to-end scenario:

1. start application;
2. create a project from the nanoGPT template;
3. inspect top-level graph;
4. open Transformer Stack;
5. open Transformer Block;
6. open Causal Self-Attention;
7. select QKV projection;
8. inspect its dimensions;
9. return to model configuration;
10. change `n_embd`;
11. observe propagated shapes;
12. run validation;
13. compile;
14. run one batch;
15. observe execution;
16. select an edge;
17. inspect tensor summary;
18. save project;
19. reload project;
20. confirm architecture remains identical.

## 23.8 Regression fixtures

Store small project fixtures:

```text
valid_linear_graph
invalid_linear_graph
valid_residual_graph
invalid_residual_graph
nanogpt_tiny
nanogpt_modified_activation
nanogpt_invalid_head_count
```

## 23.9 CI

CI must run:

```text
frontend lint
frontend typecheck
frontend unit tests
backend lint
backend typecheck
backend unit tests
parity tests
frontend production build
Playwright smoke test
```

Use CPU-only PyTorch in CI.

---

# 24. Error Handling and Logging

## 24.1 User-facing errors

Errors must state:

* what failed;
* where it failed;
* expected value;
* actual value;
* likely correction.

Bad:

```text
Runtime error.
```

Good:

```text
Residual Add cannot combine [B,T,64] and [B,T,96].
Both inputs must have compatible dimensions.
```

## 24.2 Runtime exceptions

Catch runtime exceptions at the node boundary.

Attach:

```text
node path
node type
input summaries
properties
original exception
```

Display a readable message first and expandable technical details second.

## 24.3 Structured backend logs

Use structured logs with:

```text
timestamp
level
request_id
session_id
graph_hash
node_path
event
```

Do not log full tensor values.

## 24.4 Frontend logging

Do not leave uncontrolled `console.log` statements.

Use a small application logger with development and production levels.

---

# 25. Accessibility and Interaction Quality

Implement:

* keyboard focus states;
* keyboard node selection;
* keyboard deletion;
* accessible labels for controls;
* tooltips that are not the only source of information;
* sufficient contrast;
* reduced-motion mode;
* non-colour validation indicators;
* scalable text;
* focus restoration when returning from subgraphs.

Do not make essential information depend only on edge colour.

---

# 26. Performance Requirements

For the bundled tiny nanoGPT configuration:

```text
Initial project load: under 2 seconds on a typical local machine
Graph validation: under 150 ms
Model compilation on CPU: under 2 seconds
Canvas interactions: visually responsive
Trace event latency: under 100 ms locally
```

These are engineering targets, not absolute guarantees.

Additional constraints:

* do not send complete large tensors over WebSocket;
* debounce validation during property edits;
* cache node-registry metadata;
* invalidate only affected derived state where practical;
* release paused-session tensors after timeout;
* virtualize long inspector lists where needed;
* keep the top-level graph collapsed.

---

# 27. Security Requirements

v0.1 project files are data, not executable code.

Do not evaluate:

* Python strings;
* JavaScript strings;
* arbitrary expressions;
* shell commands;
* import paths from project files.

Safe expressions must use the explicit expression AST.

Validate project schema before loading.

Do not allow project-relative path traversal.

Do not automatically load arbitrary `.pt` or pickle files.

Custom Python modules and external model import are future features that require a separate trust model.

---

# 28. Documentation Requirements

Create and maintain:

## `README.md`

Include:

* product description;
* screenshot placeholder until UI exists;
* system requirements;
* installation;
* development commands;
* test commands;
* current limitations.

## `ARCHITECTURE.md`

Document:

* frontend/backend boundary;
* canonical IR;
* node registry;
* shape engine;
* compiler/runtime;
* trace system;
* project persistence.

## `TESTING.md`

Document:

* unit tests;
* parity tests;
* e2e tests;
* deterministic settings;
* how to debug a parity failure.

## `PROGRESS.md`

For each milestone:

```text
status
completed work
tests run
known limitations
next task
```

Update it whenever a milestone is completed.

## `THIRD_PARTY_NOTICES.md`

Include relevant nanoGPT and library attribution.

## Architecture decision records

Create:

```text
docs/adr/0001-canonical-ir.md
docs/adr/0002-local-web-python-runtime.md
docs/adr/0003-hierarchical-composite-graphs.md
docs/adr/0004-no-general-control-flow-v0.1.md
docs/adr/0005-runtime-not-codegen.md
docs/adr/0006-pinned-nanogpt-reference.md
```

---

# 29. Implementation Milestones

Complete milestones sequentially.

Every milestone must leave the repository runnable.

---

## Milestone 0 — Reference Pinning and Technical Skeleton

### Tasks

* inspect current repository state;
* initialize monorepo if needed;
* pin nanoGPT reference commit;
* store MIT attribution;
* create root documentation;
* create frontend and backend skeletons;
* create `make dev`, `make test`, `make lint`;
* add CI skeleton;
* create ADR files;
* add health endpoint;
* render a minimal app shell.

### Exit criteria

* [ ] `make dev` starts frontend and backend.
* [ ] Frontend can reach backend health endpoint.
* [ ] `make test` runs both test suites.
* [ ] nanoGPT reference commit is pinned.
* [ ] lockfiles are committed.
* [ ] no model-editor functionality is faked.

---

## Milestone 1 — Canonical IR and Node Registry

### Tasks

* implement versioned project schema;
* implement graph, node, port and edge models;
* implement safe property-binding representations;
* implement registry interface;
* register initial basic nodes:

  * TensorInput;
  * Linear;
  * GELU;
  * Add;
  * GraphOutput;
* expose registry through API;
* generate TypeScript contracts from JSON Schema;
* implement serialization round-trip tests.

### Exit criteria

* [ ] A graph can be represented without React Flow.
* [ ] The same graph serializes and deserializes without semantic changes.
* [ ] Frontend receives node definitions from backend registry.
* [ ] No architecture semantics are duplicated manually in the frontend.
* [ ] Registry contract tests pass.

---

## Milestone 2 — First Executable Vertical Slice

Build:

```text
TensorInput → Linear → GELU → Linear → GraphOutput
```

### Tasks

* implement basic shape inference;
* implement structural validation;
* implement `CompiledGraphModule`;
* register PyTorch modules correctly;
* execute a real tensor;
* return output shape and summary;
* display the graph on the canvas;
* display node properties in inspector;
* display shapes on edges.

### Exit criteria

* [ ] User can construct or load the two-layer MLP.
* [ ] Validation detects the wrong input dimension.
* [ ] The graph compiles into `nn.Module`.
* [ ] A real forward pass returns a tensor.
* [ ] Output matches an equivalent hand-written PyTorch model.
* [ ] Canvas is not merely displaying mock data.

This milestone proves the core architecture before nanoGPT complexity is introduced.

---

## Milestone 3 — Canvas Editing

### Tasks

* implement palette;
* drag nodes;
* connect ports;
* reject incompatible connections;
* select nodes and edges;
* delete;
* duplicate;
* copy/paste;
* undo/redo;
* node positioning;
* zoom-to-fit;
* minimap;
* auto-layout;
* context menu;
* dirty-state tracking.

### Exit criteria

* [ ] A user can build the MLP from a blank canvas.
* [ ] All edits modify canonical IR.
* [ ] Undo and redo preserve valid stable IDs.
* [ ] Saving and loading preserves the graph and layout.
* [ ] Invalid edges produce local diagnostics.

---

## Milestone 4 — Symbolic Configuration and Shape Propagation

### Tasks

* implement model configuration;
* implement config-reference bindings;
* implement safe expression AST;
* implement symbolic dimensions;
* implement constraint diagnostics;
* implement context-aware dropdown options;
* implement parameter counts;
* show derived values in inspector.

### Exit criteria

* [ ] Changing a model-level dimension propagates to bound nodes.
* [ ] Invalid head divisibility can be detected.
* [ ] Property dropdowns can depend on graph context.
* [ ] Parameter totals update after edits.
* [ ] Weight-sharing counts can be represented, even before nanoGPT uses them.

---

## Milestone 5 — Composite Graphs and Progressive Disclosure

### Tasks

* implement module interface nodes;
* implement composite definitions;
* implement composite instances;
* implement breadcrumb navigation;
* preserve per-graph viewport;
* implement built-in read-only definitions;
* implement “Create editable copy”;
* implement repeat modules with independent weights;
* implement repeat-instance selector.

### Exit criteria

* [ ] A composite MLP can be opened and inspected.
* [ ] Returning to parent graph preserves viewport.
* [ ] Editing a custom definition affects all its instances.
* [ ] Repeated modules create independent parameter instances.
* [ ] UI clearly distinguishes definition and instance editing.

---

## Milestone 6 — nanoGPT Primitive Coverage

### Tasks

Implement and test:

* Embedding;
* LayerNorm with optional bias;
* Dropout;
* Arange;
* SplitQKV;
* SplitHeads;
* MergeHeads;
* Transpose;
* Contiguous;
* MatMul;
* Scale;
* CausalMask;
* MaskedFill;
* Softmax;
* ScaledDotProductAttention;
* ManualCausalAttention;
* SelectTimeStep;
* FlattenDimensions;
* CrossEntropyLoss;
* weight tying;
* nanoGPT initialization rules.

### Exit criteria

* [ ] Every required nanoGPT operation exists in the registry.
* [ ] Every node has shape tests.
* [ ] Every parameterized node registers parameters correctly.
* [ ] Manual attention matches a direct PyTorch reference.
* [ ] SDPA attention produces correct shapes and causal behaviour.

---

## Milestone 7 — Complete nanoGPT Template

### Tasks

Construct:

* input embeddings composite;
* attention composite;
* MLP composite;
* Transformer block composite;
* Transformer stack repeat;
* root GPT model;
* tied embedding/LM-head weight;
* training logits and loss;
* inference final-token path.

Add top-level model configuration UI.

### Exit criteria

* [ ] nanoGPT opens at a readable high-level view.
* [ ] Every composite can be opened.
* [ ] QKV internals are visible.
* [ ] Residual paths are explicit.
* [ ] Tensor shapes resolve throughout the graph.
* [ ] Parameter count is correct.
* [ ] Model compiles and executes.

---

## Milestone 8 — Reference Parity

### Tasks

* implement state-dict weight map;
* implement completeness checks;
* implement forward parity;
* implement intermediate parity;
* implement loss parity;
* implement gradient parity;
* implement one-step optimizer parity;
* implement parameter-count parity;
* implement inference-path parity;
* expose parity test command.

Command:

```bash
make parity
```

### Exit criteria

* [ ] Bundled tiny nanoGPT passes all parity tests.
* [ ] Weight tying is verified by object identity/storage sharing.
* [ ] A mismatch identifies the first divergent node.
* [ ] CI runs parity tests on CPU.
* [ ] UI displays parity status for the unmodified template.

No execution-visualization work should be considered trustworthy until this milestone passes.

---

## Milestone 9 — Execution Trace and Tensor Inspection

### Tasks

* implement WebSocket trace events;
* implement execution highlighting;
* implement tensor summaries;
* implement edge inspector;
* implement node inspector;
* implement trace speed;
* implement run cancellation;
* implement stale-session detection;
* implement session cleanup.

### Exit criteria

* [ ] User can run one batch and watch node execution.
* [ ] User can click an executed edge and inspect a real tensor summary.
* [ ] Trace paths identify repeated block instances.
* [ ] Graph edits invalidate old sessions.
* [ ] Large tensors are not transferred in full.

---

## Milestone 10 — Breakpoints and Stepping

### Tasks

* add breakpoint toggle;
* implement run-until-breakpoint;
* retain intermediate tensors safely;
* implement single-node step;
* implement continue;
* implement stop;
* display paused-state controls;
* enforce session timeout.

### Exit criteria

* [ ] User can break inside block 1 attention.
* [ ] Input tensors are inspectable before continuing.
* [ ] Step executes exactly one graph instruction.
* [ ] Continue completes the run.
* [ ] Stop releases retained tensors.

---

## Milestone 11 — Editing and Recompilation Demo

### Tasks

Support a controlled architecture modification:

```text
GELU → SiLU
```

This requires adding `SiLU` to the registry.

Also support:

```text
MLP hidden multiplier: 4 → 3
n_layer: 2 → 4
n_head: 4 → 8 when valid
```

### Exit criteria

* [ ] Editing GELU to SiLU changes the runtime module.
* [ ] Parameter count changes when hidden multiplier changes.
* [ ] Repeat-instance count changes with `n_layer`.
* [ ] Valid head-count options update with `n_embd`.
* [ ] Modified architecture loses reference-parity badge.
* [ ] Edited model still compiles and runs.

This milestone demonstrates that the tool is an architecture editor rather than a nanoGPT viewer.

---

## Milestone 12 — v0.1 Hardening

### Tasks

* complete documentation;
* run accessibility review;
* resolve memory leaks;
* add corrupted-project tests;
* add migration scaffold;
* improve diagnostics;
* add example projects;
* add production build;
* verify macOS, Linux and WSL development setup;
* record known limitations;
* remove dead code and placeholder controls.

### Exit criteria

* [ ] All v0.1 definition-of-done items pass.
* [ ] No visible button is non-functional without being clearly marked disabled.
* [ ] CI is green.
* [ ] Fresh install instructions work.
* [ ] nanoGPT acceptance demo works from a clean checkout.
* [ ] `PROGRESS.md` accurately reflects the state of the project.

---

# 30. Required Acceptance Demonstration

The final v0.1 demonstration must follow this sequence.

## Step 1: Launch

Run:

```bash
make dev
```

Open the local application.

## Step 2: Create project

Select:

```text
New Project
→ Templates
→ nanoGPT
```

## Step 3: Inspect model

The top-level graph shows:

```text
Input Embeddings
→ Transformer Stack × 2
→ Final LayerNorm
→ LM Head
→ Logits / Loss
```

## Step 4: Drill down

Open:

```text
Transformer Stack
→ Transformer Block
→ Causal Self-Attention
→ QKV Projection
```

The breadcrumb shows the complete path.

## Step 5: Inspect values

The inspector shows:

```text
Input features: n_embd = 64
Output features: 3 × n_embd = 192
Bias: model.bias = true
Parameters: calculated value
Input shape: [B,T,64]
Output shape: [B,T,192]
```

## Step 6: Edit configuration

Change:

```text
n_embd: 64 → 96
n_head: 4 → 8
```

All relevant shapes update.

An invalid head count must either be absent from the dropdown or produce a precise diagnostic.

## Step 7: Validate

Run validation.

The application reports:

```text
Graph valid
Resolved shapes
Parameter count
```

## Step 8: Compile

Compile on CPU.

The app returns a session associated with the current graph hash.

## Step 9: Run

Execute a deterministic token batch.

Nodes illuminate in execution order.

## Step 10: Inspect tensor

Click the edge after the first block’s attention output.

Display real:

```text
shape
dtype
mean
standard deviation
minimum
maximum
norm
NaN count
sample values
```

## Step 11: Breakpoint

Set a breakpoint on the first block MLP.

Run again.

Execution pauses at that node.

Inspect its input and continue.

## Step 12: Modify architecture

Create an editable MLP copy.

Change:

```text
GELU → SiLU
```

Recompile and run.

The runtime output must change.

## Step 13: Save and reload

Save the project.

Reload it.

The graph, configuration, custom MLP, node layout and stable IDs must remain intact.

## Step 14: Reference template test

Open a fresh unmodified nanoGPT template.

Run the parity suite.

Display:

```text
nanoGPT parity passed
```

---

# 31. v0.1 Definition of Done

The release is not complete until every item below is true.

## Architecture

* [ ] Canonical model IR exists independently of the UI.
* [ ] Graphs support stable IDs.
* [ ] Composite modules are first-class.
* [ ] Repeated modules instantiate independent weights.
* [ ] Weight tying is explicit.
* [ ] nanoGPT is represented completely.

## Editing

* [ ] Nodes can be added, deleted and connected.
* [ ] Values can be edited through constrained controls.
* [ ] Model-level configuration propagates.
* [ ] Built-in composites can be forked into custom composites.
* [ ] Undo and redo work.
* [ ] Projects save and reload.

## Validation

* [ ] Structural validation works.
* [ ] Shape validation works.
* [ ] dtype validation works.
* [ ] head-divisibility validation works.
* [ ] residual mismatch validation works.
* [ ] errors identify the responsible visual element.

## Runtime

* [ ] Graph compiles to real PyTorch.
* [ ] Parameters register correctly.
* [ ] State dictionaries work.
* [ ] CPU execution works.
* [ ] CUDA and MPS are detected when available.
* [ ] graph changes invalidate stale sessions.

## nanoGPT parity

* [ ] Weight map is complete.
* [ ] Parameter count matches.
* [ ] Forward logits match.
* [ ] Loss matches.
* [ ] selected intermediate activations match.
* [ ] selected gradients match.
* [ ] one optimizer step matches.
* [ ] inference final-token output matches.

## Inspection

* [ ] Execution is visible on the canvas.
* [ ] Node paths include repeated-instance indices.
* [ ] Edges expose tensor summaries.
* [ ] Nodes expose input/output summaries.
* [ ] Breakpoints work in inspection mode.
* [ ] Step, continue and stop work.

## Engineering

* [ ] Frontend is strictly typed.
* [ ] Backend is typed.
* [ ] Test suite is automated.
* [ ] CI passes.
* [ ] Dependencies are pinned.
* [ ] documentation is current.
* [ ] no cloud service is required.
* [ ] no telemetry is enabled by default.
* [ ] no arbitrary code is loaded from project files.

---

# 32. Post-v0.1 Roadmap

Do not implement these items before v0.1 completion.

## v0.2 — Minimal Training Workspace

Add:

* Tiny Shakespeare preparation;
* train/validation split;
* AdamW;
* learning-rate controls;
* batch size;
* sequence length;
* gradient accumulation;
* training loss chart;
* validation loss chart;
* checkpoint save/load;
* node-level gradient statistics;
* live parameter norms.

The visual model remains the architecture source of truth.

## v0.3 — Code Export

Generate readable PyTorch code from the canonical IR.

Export must be secondary to the runtime, not the canonical representation.

Add round-trip semantic tests where possible.

## v0.4 — Llama/Qwen-Style Architecture

Add:

* RMSNorm;
* RoPE;
* grouped-query attention;
* separate Q/K/V projections;
* SwiGLU;
* KV-head configuration;
* modern causal-mask behaviour.

Use a small Llama-style model before importing a large checkpoint.

## v0.5 — Existing PyTorch Model Import

Investigate:

```text
torch.export
torch.fx
module-tree inspection
```

PyTorch’s exported representation is FX-based, but tracing arbitrary Python models has constraints, so imported graphs must be treated as a separate adapter problem rather than assumed to be automatic.

## v0.6 — Training and Runtime Overlays

Add:

* gradients;
* parameter norms;
* activation memory;
* execution time;
* FLOP estimates;
* actual allocated memory;
* NaN/Inf propagation;
* comparison between runs.

## v0.7 — Conditional Compute and MoE

Add:

* router nodes;
* top-k selection;
* expert pools;
* conditional execution;
* resident versus active parameter accounting;
* hardware profiles;
* expert loading and unloading.

This is where the augmented-dense and hardware-elastic model experiments can be constructed.

## v0.8 — Architecture Experiment Management

Add:

* graph diff;
* architecture versioning;
* experiment variants;
* metric comparison;
* reproducible run manifests;
* checkpoint lineage;
* exportable reports.

---

# 33. Codex Working Rules

Follow these throughout implementation.

## 33.1 Build vertical slices

Do not implement the entire schema before proving one graph can execute.

The first vertical slice is:

```text
Input → Linear → GELU → Linear → Output
```

Then add hierarchy.

Then add nanoGPT.

## 33.2 Keep the application runnable

Every completed milestone must leave:

```text
make dev
make test
```

working.

## 33.3 Do not hide incomplete features

A visible button must:

* work;
* be clearly disabled with an explanation; or
* not exist yet.

Do not create placeholder toolbar actions that silently do nothing.

## 33.4 Avoid duplicate semantics

Do not independently implement neural-network shape logic in both Python and TypeScript.

The backend is authoritative.

The frontend renders backend registry definitions and validation results.

## 33.5 Prefer explicit structures

Use:

```text
structured expression AST
structured diagnostics
structured trace events
structured weight bindings
```

Do not pass architecture semantics as ad hoc strings.

## 33.6 Preserve determinism

Use seeded fixtures for:

* graph IDs where needed;
* model initialization;
* token batches;
* parity tests;
* tensor samples.

## 33.7 Record deviations

When an implementation decision differs from this plan:

1. document the reason;
2. add or update an ADR;
3. update `PROGRESS.md`;
4. preserve the product principle unless explicitly impossible.

## 33.8 Do not overbuild infrastructure

Do not add:

* Kubernetes;
* external databases;
* authentication;
* message queues;
* remote workers;
* microservices;
* cloud storage.

A local React application and Python process are sufficient.

## 33.9 Maintain testable boundaries

Keep separate:

```text
IR
registry
validation
shape inference
runtime
trace collection
API
UI projection
```

## 33.10 No unbounded TODO accumulation

A TODO must either:

* identify a specific post-v0.1 roadmap item; or
* link to a tracked issue.

Remove obsolete TODOs.

---

# 34. First Implementation Actions

Begin with these actions in order.

1. Inspect the existing repository and do not overwrite working code.
2. Create or update `PLAN.md` with this document.
3. Create `PROGRESS.md`.
4. Pin the nanoGPT reference commit.
5. Add third-party attribution.
6. Scaffold React/Vite/TypeScript frontend.
7. Scaffold FastAPI/Pydantic/PyTorch backend.
8. Add one command that launches both.
9. Define canonical IR schemas.
10. Define node registry interface.
11. Register `TensorInput`, `Linear`, `GELU` and `GraphOutput`.
12. Implement the small two-layer MLP graph fixture.
13. Compile it into real PyTorch.
14. Test numerical equivalence with a hand-written PyTorch MLP.
15. Render it with React Flow.
16. Connect the property inspector.
17. Display inferred edge shapes.
18. Only then begin composite modules and nanoGPT nodes.

The first substantial checkpoint report must include:

```text
Files changed
Architecture decisions made
Commands to run
Tests added
Tests passed
Known limitations
Next milestone
```

---

# 35. Final Product Standard

The target experience is not:

> “I generated a diagram of nanoGPT.”

It is:

> “I opened nanoGPT as a live, hierarchical program; clicked from the whole model into a Transformer block, then into attention, then into its fused QKV projection; changed an architectural value through a valid dropdown; saw every affected tensor shape update; executed a real batch; watched the tensor traverse the model; paused on a layer; inspected the real activation; and verified that the unmodified graph matches reference nanoGPT numerically.”

That is the standard against which every implementation decision should be evaluated.
