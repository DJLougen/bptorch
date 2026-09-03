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



def test_custom_forked_composite_validation_and_compilation():
    from fastapi.testclient import TestClient
    from neural_blueprint.api.main import app
    from neural_blueprint.templates.nanogpt import create_nanogpt_template

    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    block_graph = project.model.graphs["graph_block"]
    mlp_node = next(n for n in block_graph.nodes if "mlp" in n.definition_id)
    orig_definition_id = mlp_node.definition_id

    # Fork graph_mlp into custom.graph_custom_mlp_42
    orig_graph = project.model.graphs["graph_mlp"]
    new_graph_id = "graph_custom_mlp_42"
    custom_graph = orig_graph.model_copy(deep=True)
    custom_graph.id = new_graph_id
    custom_graph.name = "Custom MLP"
    custom_graph.derived_from = orig_definition_id
    custom_graph.modified = True

    project.model.graphs[new_graph_id] = custom_graph
    mlp_node.definition_id = f"custom.{new_graph_id}"

    # 1. ProjectValidator asserts valid
    validator = ProjectValidator()
    result = validator.validate(project)
    assert result.valid is True, f"Validation failed with errors: {[d.message for d in result.errors]}"
    assert len(result.errors) == 0

    # 2. POST /api/v1/models/compile returns 200 OK
    client = TestClient(app)
    resp = client.post(
        "/api/v1/models/compile",
        json={"project": project.model_dump(mode="json"), "mode": "training"},
    )
    assert resp.status_code == 200, f"Compile failed: {resp.text}"
    payload = resp.json()
    assert "session_id" in payload
    assert "graph_hash" in payload
