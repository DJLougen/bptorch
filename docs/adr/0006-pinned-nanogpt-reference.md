# ADR 0006: Pinned nanoGPT Reference Architecture

## Status
Accepted

## Context
nanoGPT (`karpathy/nanoGPT`) serves as our target reference architecture to prove correctness, weight structure compatibility, parameter count exactness, and numerical parity. Following a moving Git branch could introduce non-deterministic breakage in parity test suites.

## Decision
We pin the exact nanoGPT Git commit (`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`) in `references/nanogpt.lock.json` and vendor `references/nanogpt/model.py` under the MIT license. Automated parity tests (`make parity`) run directly against this pinned implementation on CPU with tight tolerances ($rtol=10^{-5}, atol=10^{-6}$).

## Consequences
- Deterministic, reproducible CI and local parity test execution.
- Clear isolation of reference code with proper open-source licensing and attribution.
