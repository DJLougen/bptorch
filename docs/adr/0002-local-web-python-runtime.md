# ADR 0002: Local-First Web Frontend and Python Runtime

## Status
Accepted

## Context
Neural networks in research and production are written in Python with PyTorch. Re-implementing PyTorch execution in JavaScript/WebAssembly lacks kernel fidelity, device acceleration (MPS/CUDA), exact operator semantics, and ecosystem compatibility.

## Decision
The system is built as a local-first web application:
1. Frontend: React, TypeScript, Vite, React Flow (@xyflow/react), Zustand.
2. Backend: Local FastAPI / Python process executing real PyTorch modules and tensor operations.
3. Communication: REST API for schema, validation, compilation, and project operations; WebSocket for real-time execution trace streaming.

## Consequences
- PyTorch numerical parity is exact and uses real PyTorch kernels.
- No cloud backend, external database, account, or telemetry is required.
- The user runs the entire application locally on their machine.
