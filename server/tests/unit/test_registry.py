"""Unit tests for Node Registry and node contract guarantees."""

from neural_blueprint.registry.base import NodeValidationContext
from neural_blueprint.registry.registry import global_registry


def test_registry_contains_initial_nodes():
    nodes = global_registry.list_all()
    type_ids = [n.type_id for n in nodes]

    assert "builtin.tensor_input@1" in type_ids
    assert "builtin.linear@1" in type_ids
    assert "builtin.gelu@1" in type_ids
    assert "builtin.silu@1" in type_ids
    assert "builtin.add@1" in type_ids
    assert "builtin.graph_output@1" in type_ids


def test_node_definition_contracts():
    for node_def in global_registry.list_all():
        # Contract 1: Non-empty type_id and valid category
        assert node_def.type_id
        assert node_def.display_name
        assert node_def.category

        # Contract 2: Property schema is a valid dictionary
        prop_schema = node_def.property_schema()
        assert isinstance(prop_schema, dict)

        # Contract 3: Ports generation
        inputs = node_def.input_ports({})
        outputs = node_def.output_ports({})
        assert isinstance(inputs, list)
        assert isinstance(outputs, list)


def test_linear_node_properties_and_runtime():
    linear = global_registry.require("builtin.linear@1")
    ctx = NodeValidationContext(model_config={"n_embd": 64})

    param_spec = linear.parameter_spec({"in_features": 64, "out_features": 128, "bias": True}, ctx)
    assert param_spec.trainable_count == (64 * 128) + 128
    assert param_spec.parameter_shapes["weight"] == [128, 64]
    assert param_spec.parameter_shapes["bias"] == [128]

    runtime_spec = linear.build_runtime({"in_features": 64, "out_features": 128, "bias": True}, ctx)
    assert runtime_spec is not None
    assert runtime_spec.module_type == "nn_module"
    module = runtime_spec.factory()
    assert module.in_features == 64
    assert module.out_features == 128
