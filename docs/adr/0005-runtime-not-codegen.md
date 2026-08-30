# ADR 0005: PyTorch Module Runtime Instead of Source Code Generation

## Status
Accepted

## Context
A visual neural network tool can either generate Python source files for the user to execute, or construct a live in-memory PyTorch module (`nn.Module`) directly from the graph IR.

## Decision
bpTorch compiles the IR into an in-memory `CompiledGraphModule(torch.nn.Module)`. The module registers child modules into `nn.ModuleDict` / `nn.ModuleList` and builds a topological execution plan. Tracing hooks, breakpoints, tensor statistics capture, and state-dict mappings operate directly on the runtime module. Source code export is a post-v0.1 secondary feature.

## Consequences
- The visual graph is the executable model, avoiding code generator divergence.
- Interactive stepping, breakpoints, and live activation inspections are directly hooked into execution.
- Standard PyTorch functions (`parameters()`, `state_dict()`, `.train()`, `.eval()`, optimizers) work seamlessly.
