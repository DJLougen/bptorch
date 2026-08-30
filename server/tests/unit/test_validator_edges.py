"""Focused validator tests for edge-level structural validation."""

from neural_blueprint.ir.models import Edge, NodeInstance, PortReference
from neural_blueprint.validation.diagnostics import (
    E_DUPLICATE_EDGE_ID,
    E_EDGE_MISSING_NODE,
    E_EDGE_MISSING_PORT,
    E_EDGE_PORT_KIND_MISMATCH,
    E_EDGE_SELF_CONNECTION,
    E_EDGE_WRONG_PORT_DIRECTION,
    E_MULTIPLE_INPUTS,
    E_PORT_UNCONNECTED,
)
from neural_blueprint.validation.validator import ProjectValidator
from tests.unit.test_serialization import create_sample_project


def _graph(project):
    return project.model.graphs["graph_mlp"]


def _validate(project):
    return ProjectValidator().validate(project)


def test_valid_mlp_edges_remain_valid():
    result = _validate(create_sample_project())
    assert result.valid is True
    assert result.errors == []


def test_duplicate_edge_id_diagnostic():
    project = create_sample_project()
    graph = _graph(project)
    graph.edges.append(
        Edge(
            id="e1",
            source=PortReference(node_id="node_in", port_id="output"),
            target=PortReference(node_id="node_fc1", port_id="input"),
        )
    )

    result = _validate(project)
    diags = [d for d in result.errors if d.code == E_DUPLICATE_EDGE_ID]

    assert result.valid is False
    assert len(diags) == 2
    assert all(d.edge_id == "e1" for d in diags)


def test_edge_missing_source_node_diagnostic():
    project = create_sample_project()
    graph = _graph(project)
    graph.edges.append(
        Edge(
            id="e_missing_src",
            source=PortReference(node_id="node_missing", port_id="output"),
            target=PortReference(node_id="node_fc1", port_id="input"),
        )
    )

    result = _validate(project)
    diag = next(
        d for d in result.errors if d.code == E_EDGE_MISSING_NODE and d.edge_id == "e_missing_src"
    )

    assert diag.node_id == "node_missing"
    assert "missing source node" in diag.message


def test_edge_missing_target_node_diagnostic():
    project = create_sample_project()
    graph = _graph(project)
    graph.edges.append(
        Edge(
            id="e_missing_tgt",
            source=PortReference(node_id="node_in", port_id="output"),
            target=PortReference(node_id="node_missing", port_id="input"),
        )
    )

    result = _validate(project)
    diag = next(
        d for d in result.errors if d.code == E_EDGE_MISSING_NODE and d.edge_id == "e_missing_tgt"
    )

    assert diag.node_id == "node_missing"
    assert "missing target node" in diag.message


def test_edge_missing_source_port_diagnostic():
    project = create_sample_project()
    graph = _graph(project)
    graph.edges.append(
        Edge(
            id="e_missing_src_port",
            source=PortReference(node_id="node_in", port_id="not_a_port"),
            target=PortReference(node_id="node_fc1", port_id="input"),
        )
    )

    result = _validate(project)
    diag = next(
        d
        for d in result.errors
        if d.code == E_EDGE_MISSING_PORT and d.edge_id == "e_missing_src_port"
    )

    assert diag.node_id == "node_in"
    assert diag.port_id == "not_a_port"


def test_edge_missing_target_port_diagnostic():
    project = create_sample_project()
    graph = _graph(project)
    graph.edges.append(
        Edge(
            id="e_missing_tgt_port",
            source=PortReference(node_id="node_in", port_id="output"),
            target=PortReference(node_id="node_fc1", port_id="not_a_port"),
        )
    )

    result = _validate(project)
    diag = next(
        d
        for d in result.errors
        if d.code == E_EDGE_MISSING_PORT and d.edge_id == "e_missing_tgt_port"
    )

    assert diag.node_id == "node_fc1"
    assert diag.port_id == "not_a_port"


def test_edge_source_must_use_output_port():
    project = create_sample_project()
    graph = _graph(project)
    graph.edges.append(
        Edge(
            id="e_src_input",
            source=PortReference(node_id="node_fc1", port_id="input"),
            target=PortReference(node_id="node_gelu", port_id="input"),
        )
    )

    result = _validate(project)
    diag = next(
        d
        for d in result.errors
        if d.code == E_EDGE_WRONG_PORT_DIRECTION and d.edge_id == "e_src_input"
    )

    assert diag.node_id == "node_fc1"
    assert diag.port_id == "input"
    assert diag.expected == "output"
    assert diag.actual == "input"


def test_edge_target_must_use_input_port():
    project = create_sample_project()
    graph = _graph(project)
    graph.edges.append(
        Edge(
            id="e_tgt_output",
            source=PortReference(node_id="node_fc1", port_id="output"),
            target=PortReference(node_id="node_gelu", port_id="output"),
        )
    )

    result = _validate(project)
    diag = next(
        d
        for d in result.errors
        if d.code == E_EDGE_WRONG_PORT_DIRECTION and d.edge_id == "e_tgt_output"
    )

    assert diag.node_id == "node_gelu"
    assert diag.port_id == "output"
    assert diag.expected == "input"
    assert diag.actual == "output"


def test_edge_port_kind_mismatch_diagnostic():
    project = create_sample_project()
    graph = _graph(project)
    graph.nodes.extend(
        [
            NodeInstance(
                id="node_seq",
                definition_id="builtin.training_sequence@1",
                display_name="Sequence",
                properties={"branch_count": 2},
            ),
            NodeInstance(
                id="node_seq_target",
                definition_id="builtin.gelu@1",
                display_name="GELU Target",
            ),
        ]
    )
    graph.edges.append(
        Edge(
            id="e_kind_mismatch",
            source=PortReference(node_id="node_seq", port_id="then_0"),
            target=PortReference(node_id="node_seq_target", port_id="input"),
        )
    )

    result = _validate(project)
    diag = next(d for d in result.errors if d.code == E_EDGE_PORT_KIND_MISMATCH)

    assert diag.edge_id == "e_kind_mismatch"
    assert diag.expected == "exec"
    assert diag.actual == "data"


def test_edge_self_connection_diagnostic():
    project = create_sample_project()
    graph = _graph(project)
    graph.edges.append(
        Edge(
            id="e_self",
            source=PortReference(node_id="node_fc1", port_id="output"),
            target=PortReference(node_id="node_fc1", port_id="input"),
        )
    )

    result = _validate(project)
    diag = next(d for d in result.errors if d.code == E_EDGE_SELF_CONNECTION)

    assert diag.edge_id == "e_self"
    assert diag.node_id == "node_fc1"


def test_dynamic_port_resolution_uses_node_properties():
    project = create_sample_project()
    graph = _graph(project)
    graph.nodes.append(
        NodeInstance(
            id="node_seq",
            definition_id="builtin.training_sequence@1",
            display_name="Sequence",
            properties={"branch_count": 2},
        )
    )
    graph.edges.append(
        Edge(
            id="e_dynamic_missing",
            source=PortReference(node_id="node_seq", port_id="then_2"),
            target=PortReference(node_id="node_gelu", port_id="input"),
        )
    )

    result = _validate(project)
    diag = next(
        d
        for d in result.errors
        if d.code == E_EDGE_MISSING_PORT and d.edge_id == "e_dynamic_missing"
    )

    assert diag.node_id == "node_seq"
    assert diag.port_id == "then_2"


def test_invalid_edges_excluded_from_multiplicity_checks():
    project = create_sample_project()
    graph = _graph(project)
    graph.edges.append(
        Edge(
            id="e1",
            source=PortReference(node_id="node_in", port_id="output"),
            target=PortReference(node_id="node_fc1", port_id="input"),
        )
    )

    result = _validate(project)
    codes = {d.code for d in result.errors}

    assert E_DUPLICATE_EDGE_ID in codes
    assert E_MULTIPLE_INPUTS not in codes


def test_required_port_checks_still_apply_for_valid_edges():
    project = create_sample_project()
    graph = _graph(project)
    graph.edges = [e for e in graph.edges if e.target.node_id != "node_fc1"]

    result = _validate(project)

    assert result.valid is False
    assert any(d.code == E_PORT_UNCONNECTED for d in result.errors)


def test_malformed_edges_do_not_crash_shape_propagation():
    project = create_sample_project()
    graph = _graph(project)
    graph.edges.extend(
        [
            Edge(
                id="e_bad_nodes",
                source=PortReference(node_id="missing_a", port_id="output"),
                target=PortReference(node_id="missing_b", port_id="input"),
            ),
            Edge(
                id="e_bad_ports",
                source=PortReference(node_id="node_in", port_id="missing_port"),
                target=PortReference(node_id="node_fc1", port_id="missing_port"),
            ),
        ]
    )

    result = _validate(project)

    assert result.valid is False
    assert "graph_mlp" in result.resolved_shapes
    assert isinstance(result.resolved_shapes["graph_mlp"], dict)
    assert "node_in" in result.resolved_shapes["graph_mlp"]
