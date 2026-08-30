"""Symbolic shape representations, formatting, and unification utilities."""

from typing import Any, Dict, List, Optional, Tuple, Union

from neural_blueprint.ir.models import (
    ConfigRefDim,
    LiteralDim,
    ShapeDim,
    SymbolDim,
    UnknownDim,
)


def format_dim(
    dim: Union[ShapeDim, Dict[str, Any], int, str],
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Formats a single ShapeDim into a readable string (e.g. 'B', 'T', '64', '?')."""
    if config is None:
        config = {}

    if isinstance(dim, int):
        return str(dim)
    if isinstance(dim, str):
        return dim

    if isinstance(dim, dict):
        kind = dim.get("kind")
        if kind == "symbol":
            return str(dim.get("name", "?"))
        elif kind == "config_ref":
            key = dim.get("key", "")
            val = config.get(key)
            return str(val) if val is not None else str(key)
        elif kind == "literal":
            return str(dim.get("value", "?"))
        else:
            return "?"

    if isinstance(dim, SymbolDim):
        return dim.name
    elif isinstance(dim, ConfigRefDim):
        val = config.get(dim.key)
        return str(val) if val is not None else dim.key
    elif isinstance(dim, LiteralDim):
        return str(dim.value)
    elif isinstance(dim, UnknownDim):
        return "?"

    return "?"


def format_shape(shape: List[Any], config: Optional[Dict[str, Any]] = None) -> str:
    """Formats a list of dimensions into standard tensor shape notation (e.g. '[B, T, 64]')."""
    if not shape:
        return "[]"
    dims_str = ", ".join(format_dim(d, config) for d in shape)
    return f"[{dims_str}]"


def resolve_dim_value(
    dim: Union[ShapeDim, Dict[str, Any], int],
    config: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Resolves a dimension to a concrete integer if possible, or None if symbolic/unknown."""
    if config is None:
        config = {}

    if isinstance(dim, int):
        return dim

    if isinstance(dim, dict):
        kind = dim.get("kind")
        if kind == "literal":
            return int(dim.get("value", 0))
        elif kind == "config_ref":
            val = config.get(dim.get("key", ""))
            return int(val) if isinstance(val, (int, float)) else None
        return None

    if isinstance(dim, LiteralDim):
        return dim.value
    elif isinstance(dim, ConfigRefDim):
        val = config.get(dim.key)
        return int(val) if isinstance(val, (int, float)) else None

    return None


def dims_compatible(
    dim_a: Union[ShapeDim, Dict[str, Any], int],
    dim_b: Union[ShapeDim, Dict[str, Any], int],
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Returns True if two dimensions can unify / are structurally compatible."""
    if config is None:
        config = {}

    if isinstance(dim_a, UnknownDim) or isinstance(dim_b, UnknownDim):
        return True
    if isinstance(dim_a, dict) and dim_a.get("kind") == "unknown":
        return True
    if isinstance(dim_b, dict) and dim_b.get("kind") == "unknown":
        return True

    val_a = resolve_dim_value(dim_a, config)
    val_b = resolve_dim_value(dim_b, config)

    # 1 broadcasts with any dimension in PyTorch
    if val_a == 1 or val_b == 1:
        return True

    if val_a is not None and val_b is not None:
        return val_a == val_b

    str_a = format_dim(dim_a, config)
    str_b = format_dim(dim_b, config)
    return str_a == str_b or str_a == "?" or str_b == "?"


def shapes_compatible(
    shape_a: List[Any],
    shape_b: List[Any],
    config: Optional[Dict[str, Any]] = None,
    allow_broadcast: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Checks if two tensor shapes are compatible, optionally supporting PyTorch broadcasting rules.
    Returns (is_compatible, error_message).
    """
    if config is None:
        config = {}

    if not allow_broadcast:
        if len(shape_a) != len(shape_b):
            a_str = format_shape(shape_a, config)
            b_str = format_shape(shape_b, config)
            return False, f"Rank mismatch: {a_str} vs {b_str}"

        for i, (da, db) in enumerate(zip(shape_a, shape_b)):
            if not dims_compatible(da, db, config):
                str_a = format_shape(shape_a, config)
                str_b = format_shape(shape_b, config)
                return False, f"Dimension mismatch at index {i}: {str_a} vs {str_b}"
        return True, None

    # PyTorch broadcasting: align from trailing dimensions
    len_a = len(shape_a)
    len_b = len(shape_b)
    max_len = max(len_a, len_b)

    # Pad shapes on the left with 1
    padded_a = [LiteralDim(value=1)] * (max_len - len_a) + list(shape_a)
    padded_b = [LiteralDim(value=1)] * (max_len - len_b) + list(shape_b)

    for i in range(max_len):
        da = padded_a[i]
        db = padded_b[i]
        if not dims_compatible(da, db, config):
            str_a = format_shape(shape_a, config)
            str_b = format_shape(shape_b, config)
            return False, f"Cannot broadcast dimensions at position {i}: {str_a} vs {str_b}"

    return True, None
