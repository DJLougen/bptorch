"""Shapes and typing package."""

from neural_blueprint.shapes.engine import ShapePropagator
from neural_blueprint.shapes.types import (
    dims_compatible,
    format_dim,
    format_shape,
    resolve_dim_value,
    shapes_compatible,
)

__all__ = [
    "ShapePropagator",
    "dims_compatible",
    "format_dim",
    "format_shape",
    "resolve_dim_value",
    "shapes_compatible",
]
