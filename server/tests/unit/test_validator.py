"""Unit tests for the 4-Pass Validator."""

from neural_blueprint.ir.models import Edge, PortReference
from neural_blueprint.validation.diagnostics import (
    E_CYCLE_DETECTED,
    E_HEAD_DIVISIBILITY,
    E_LINEAR_INPUT_DIM,
    E_MULTIPLE_INPUTS,
    E_PORT_UNCONNECTED,
)
from neural_blueprint.validation.validator import ProjectValidator
from tests.unit.test_serialization import create_sample_project


def test_valid_mlp_project():
    project = create_sample_project()
    validator = ProjectValidator()
    result = validator.validate(project)

    assert result.valid is True
    assert len(result.errors) == 0


def test_unconnected_required_port_error():
    project = create_sample_project()
    # Remove edge into FC1
    project.model.graphs["graph_mlp"].edges = [
        e for e in project.model.graphs["graph_mlp"].edges if e.target.node_id != "node_fc1"
    ]

    validator = ProjectValidator()
    result = validator.validate(project)

    assert result.valid is False
    err_codes = [e.code for e in result.errors]
    assert E_PORT_UNCONNECTED in err_codes


def test_multiple_input_connection_error():
    project = create_sample_project()
    # Add a duplicate edge into node_fc1's single input port
    extra_edge = Edge(
        id="e_duplicate",
        source=PortReference(node_id="node_in", port_id="output"),
        target=PortReference(node_id="node_fc1", port_id="input"),
    )
    project.model.graphs["graph_mlp"].edges.append(extra_edge)

    validator = ProjectValidator()
    result = validator.validate(project)

    assert result.valid is False
    assert any(e.code == E_MULTIPLE_INPUTS for e in result.errors)


def test_linear_input_dimension_mismatch_diagnostic():
    project = create_sample_project()
    # Change FC1 in_features to 128 (while input provides 64)
    nodes = project.model.graphs["graph_mlp"].nodes
    for n in nodes:
        if n.id == "node_fc1":
            n.properties["in_features"] = 128

    validator = ProjectValidator()
    result = validator.validate(project)

    assert result.valid is False
    linear_errs = [e for e in result.errors if e.code == E_LINEAR_INPUT_DIM]
    assert len(linear_errs) == 1
    diag = linear_errs[0]
    assert diag.node_id == "node_fc1"
    assert "128" in diag.message
    assert len(diag.suggestions) > 0


def test_head_divisibility_semantic_validation():
    project = create_sample_project()
    # Set invalid head divisibility
    project.model.config["n_embd"] = 64
    project.model.config["n_head"] = 7  # 64 % 7 != 0

    validator = ProjectValidator()
    result = validator.validate(project)

    assert result.valid is False
    assert any(e.code == E_HEAD_DIVISIBILITY for e in result.errors)


def test_cycle_detection():
    project = create_sample_project()
    # Introduce cycle: FC2 -> FC1
    cycle_edge = Edge(
        id="e_cycle",
        source=PortReference(node_id="node_fc2", port_id="output"),
        target=PortReference(node_id="node_fc1", port_id="input"),
    )
    project.model.graphs["graph_mlp"].edges.append(cycle_edge)

    validator = ProjectValidator()
    result = validator.validate(project)

    assert result.valid is False
    assert any(e.code == E_CYCLE_DETECTED for e in result.errors)

def test_weight_tying_mismatch_on_nested_nanogpt_binding_emits_error():
    from neural_blueprint.templates.nanogpt import create_nanogpt_template
    from neural_blueprint.validation.diagnostics import E_WEIGHT_TYING_MISMATCH

    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    root_graph = project.model.graphs[project.model.root_graph_id]
    lm_head = next(n for n in root_graph.nodes if n.id == "node_lm_head")
    lm_head.properties["out_features"] = 100

    result = ProjectValidator().validate(project)
    assert result.valid is False
    assert E_WEIGHT_TYING_MISMATCH in [d.code for d in result.errors]

