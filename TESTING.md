# Testing Strategy and Test Suites

Neural Blueprint Studio incorporates a multi-layer testing strategy to ensure mathematical correctness, runtime safety, and UI interaction quality.

## Test Suites

### 1. Backend Unit & Contract Tests
- **Location**: `server/tests/`
- **Commands**:
  ```bash
  pytest server/tests/unit
  ```
- **Coverage**:
  - IR serialization and round-trip fidelity
  - Schema validation and unique ID guarantees
  - Safe expression AST evaluator
  - Symbolic shape inference and unknown dimension propagation
  - 4-pass validator (schema, structure, shape, semantics)
  - Graph topological ordering and cycle detection
  - Parameter accounting (trainable, frozen, unique, shared)
  - `CompiledGraphModule` runtime execution
  - WebSocket trace event stream and breakpoint manager
  - Every registered node definition contract

### 2. Property-Based Tests (Hypothesis)
- **Location**: `server/tests/property/`
- **Commands**:
  ```bash
  pytest server/tests/property
  ```
- **Coverage**:
  - Dimension compatibility across arbitrary linear projections
  - Add-compatible and incompatible shape combinations
  - Head-divisibility invariants (`n_embd % n_head == 0`)
  - Graph serialization round-trip independence from node insertion order

### 3. nanoGPT Numerical Parity Test Suite
- **Location**: `server/tests/parity/`
- **Commands**:
  ```bash
  make parity
  # or
  pytest server/tests/parity
  ```
- **Coverage**:
  - **State-dict mapping**: Complete bidirectional semantic mapping between `karpathy/nanoGPT` and the compiled visual runtime.
  - **Parameter counts**: Exact parameter equality including weight tying (`wte.weight` == `lm_head.weight`).
  - **Forward logits**: Assert `torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)`.
  - **Loss**: Exact cross-entropy loss matching with `ignore_index = -1`.
  - **Intermediate activations**: Token embeddings, position embeddings, block 1 LayerNorm, QKV projection, attention output, MLP output, final LayerNorm.
  - **Gradients**: Backward pass gradient equality on embeddings, projections, and layernorms.
  - **Optimizer step**: Single-step AdamW weight update equality.
  - **Inference mode**: Final-time-step selection `[B, 1, V]` parity without targets.
  - **Attention modes**: Both Manual Causal Attention and PyTorch SDPA paths.

### 4. Frontend Unit & Interaction Tests
- **Location**: `web/src/__tests__/` or `web/tests/`
- **Commands**:
  ```bash
  npm --prefix web run test
  ```
- **Coverage**:
  - Node palette rendering and category grouping
  - Property inspector constrained controls and dynamic dropdown providers
  - Breadcrumb navigation and viewport caching
  - Repeat-instance selector
  - Stale session indicator and trace state transitions

### 5. End-to-End Playwright Tests
- **Location**: `web/e2e/`
- **Commands**:
  ```bash
  npm --prefix web run test:e2e
  ```
- **Coverage**:
  - Complete Section 30 Required Acceptance sequence from project creation to modification and verification.
