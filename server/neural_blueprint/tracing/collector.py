"""Tensor statistical summarizer and deterministic value sampling."""

import math
from typing import Any, List

import torch

from neural_blueprint.tracing.events import TensorSummary


class TensorSummarizer:
    """Computes exact and statistical summaries for tensors without streaming full large arrays."""

    @staticmethod
    def summarize(tensor: Any, max_samples: int = 16) -> TensorSummary:
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, (int, float)):
                return TensorSummary(
                    shape=[],
                    dtype="float32" if isinstance(tensor, float) else "int64",
                    device="cpu",
                    numel=1,
                    min=float(tensor),
                    max=float(tensor),
                    mean=float(tensor),
                    sample_values=[tensor],
                )
            return TensorSummary(shape=[], dtype="unknown", device="cpu", numel=0)

        t = tensor.detach().cpu()
        shape = list(t.shape)
        dtype_str = str(t.dtype).replace("torch.", "")
        device_str = str(tensor.device)
        numel = t.numel()

        # Deterministic sample values
        if numel == 0:
            return TensorSummary(
                shape=shape,
                dtype=dtype_str,
                device=device_str,
                numel=0,
                sample_values=[],
            )

        flat = t.flatten()
        sample_slice = flat[: min(max_samples, numel)]
        sample_values: List[Any] = []
        for val in sample_slice:
            v_item = val.item()
            if isinstance(v_item, float):
                if math.isnan(v_item) or math.isinf(v_item):
                    sample_values.append(str(v_item))
                else:
                    sample_values.append(round(v_item, 6))
            else:
                sample_values.append(v_item)

        # Boolean / Integer specific statistics
        if t.dtype in (torch.bool, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            t_float = t.float()
            min_val = float(t.min().item())
            max_val = float(t.max().item())
            mean_val = float(t_float.mean().item())
            zero_frac = float((t == 0).sum().item()) / numel

            return TensorSummary(
                shape=shape,
                dtype=dtype_str,
                device=device_str,
                numel=numel,
                min=min_val,
                max=max_val,
                mean=round(mean_val, 4),
                zero_fraction=round(zero_frac, 4),
                sample_values=sample_values,
            )

        # Floating point statistics
        is_finite = torch.isfinite(t)
        finite_t = t[is_finite]

        nan_count = int(torch.isnan(t).sum().item())
        pos_inf_count = int((t == float("inf")).sum().item())
        neg_inf_count = int((t == float("-inf")).sum().item())
        zero_frac = float((t == 0).sum().item()) / numel

        if finite_t.numel() > 0:
            mean_val = float(finite_t.mean().item())
            std_val = float(finite_t.std().item()) if finite_t.numel() > 1 else 0.0
            min_val = float(finite_t.min().item())
            max_val = float(finite_t.max().item())
            l2_norm = float(torch.norm(finite_t, p=2).item())
        else:
            mean_val = std_val = min_val = max_val = l2_norm = None

        return TensorSummary(
            shape=shape,
            dtype=dtype_str,
            device=device_str,
            numel=numel,
            mean=round(mean_val, 6) if mean_val is not None else None,
            std=round(std_val, 6) if std_val is not None else None,
            min=round(min_val, 6) if min_val is not None else None,
            max=round(max_val, 6) if max_val is not None else None,
            l2_norm=round(l2_norm, 6) if l2_norm is not None else None,
            zero_fraction=round(zero_frac, 4),
            nan_count=nan_count,
            pos_inf_count=pos_inf_count,
            neg_inf_count=neg_inf_count,
            sample_values=sample_values,
        )
