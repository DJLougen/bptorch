"""Thread-local decode KV cache context for incremental autoregressive generation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Generator, Optional

import torch

_decode_cache_var: ContextVar[Optional["DecodeCache"]] = ContextVar("decode_cache", default=None)


@dataclass
class DecodeCache:
    """Per-generation decode cache shared across attention layers via contextvars."""

    enabled: bool = True
    past_len: int = 0
    attn_index: int = 0
    k: Dict[int, torch.Tensor] = field(default_factory=dict)
    v: Dict[int, torch.Tensor] = field(default_factory=dict)

    def reset_call_order(self) -> None:
        """Reset per-forward attention layer index (layers call update_kv in graph order)."""
        self.attn_index = 0

    def update_kv(
        self, k_heads: torch.Tensor, v_heads: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append or return cached K/V head tensors along sequence dim (-2)."""
        idx = self.attn_index
        self.attn_index += 1

        if idx in self.k:
            k_out = torch.cat([self.k[idx], k_heads], dim=-2)
            v_out = torch.cat([self.v[idx], v_heads], dim=-2)
        else:
            k_out = k_heads
            v_out = v_heads

        self.k[idx] = k_out
        self.v[idx] = v_out
        return k_out, v_out


def get_decode_cache() -> Optional[DecodeCache]:
    """Return the active decode cache for the current context, if any."""
    return _decode_cache_var.get()


@contextmanager
def decode_cache_scope(cache: DecodeCache) -> Generator[DecodeCache, None, None]:
    """Install *cache* for the duration of a generate_tokens call."""
    token = _decode_cache_var.set(cache)
    try:
        yield cache
    finally:
        _decode_cache_var.reset(token)
