"""Unit tests for shape inference and propagation engine."""

from neural_blueprint.ir.models import (
    LiteralDim,
    SymbolDim,
    UnknownDim,
)
from neural_blueprint.shapes.engine import ShapePropagator
from neural_blueprint.shapes.types import (
    dims_compatible,
    format_shape,
    shapes_compatible,
)
from tests.unit.test_serialization import create_sample_project


def test_format_shape():
    shape = [SymbolDim(name="B"), SymbolDim(name="T"), LiteralDim(value=64)]
    assert format_shape(shape) == "[B, T, 64]"


def test_dims_compatible():
    dim1 = SymbolDim(name="B")
    dim2 = SymbolDim(name="B")
    dim3 = SymbolDim(name="T")
    dim_unk = UnknownDim()

    assert dims_compatible(dim1, dim2) is True
    assert dims_compatible(dim1, dim3) is False
    assert dims_compatible(dim1, dim_unk) is True


def test_shapes_compatible():
    shape_a = [SymbolDim(name="B"), SymbolDim(name="T"), LiteralDim(value=64)]
    shape_b = [SymbolDim(name="B"), SymbolDim(name="T"), LiteralDim(value=64)]
    shape_c = [SymbolDim(name="B"), SymbolDim(name="T"), LiteralDim(value=128)]

    compat_ab, _ = shapes_compatible(shape_a, shape_b)
    assert compat_ab is True

    compat_ac, err_msg = shapes_compatible(shape_a, shape_c)
    assert compat_ac is False
    assert "broadcast" in str(err_msg).lower() or "mismatch" in str(err_msg).lower()


def test_shape_propagation_on_mlp_graph():
    project = create_sample_project()
    graph = project.model.graphs["graph_mlp"]

    propagator = ShapePropagator()
    resolved = propagator.propagate_graph(graph, config=project.model.config)

    # Input node output: [B, T, 64]
    assert "node_in" in resolved
    assert "output" in resolved["node_in"]

    # FC1 output: [B, T, 256]
    assert "node_fc1" in resolved
    fc1_out = resolved["node_fc1"]["output"]
    assert format_shape(fc1_out.shape, project.model.config) == "[B, T, 256]"

    # GELU output: [B, T, 256]
    assert "node_gelu" in resolved
    gelu_out = resolved["node_gelu"]["output"]
    assert format_shape(gelu_out.shape, project.model.config) == "[B, T, 256]"

    # FC2 output: [B, T, 64]
    assert "node_fc2" in resolved
    fc2_out = resolved["node_fc2"]["output"]
    assert format_shape(fc2_out.shape, project.model.config) == "[B, T, 64]"
