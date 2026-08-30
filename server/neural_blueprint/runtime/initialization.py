"""nanoGPT-compatible parameter initialization rules."""

import math

import torch
import torch.nn as nn


def init_nanogpt_weights(module: nn.Module, n_layer: int = 1) -> None:
    """
    Applies nanoGPT / GPT-2 weight initialization across a PyTorch module tree:
    - Linear weights: normal mean 0.0, std 0.02
    - Linear biases: zeros
    - Embedding weights: normal mean 0.0, std 0.02
    - LayerNorm weights: ones, biases: zeros
    - Residual projections ending in 'c_proj.weight': std scaled by 1 / sqrt(2 * n_layer)
    """
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # Special scaled init for residual projections (e.g. attention.c_proj, mlp.c_proj)
    scale = 0.02 / math.sqrt(2.0 * max(1, n_layer))
    for pn, p in module.named_parameters():
        if pn.endswith("c_proj.weight") or "c_proj" in pn and pn.endswith(".weight"):
            torch.nn.init.normal_(p, mean=0.0, std=scale)
