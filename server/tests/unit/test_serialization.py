"""Unit tests for project serialization, atomic persistence, and round-tripping."""

import tempfile
from pathlib import Path

import pytest
from neural_blueprint.ir.models import (
    ConfigRefValue,
    Edge,
    GraphDefinition,
    GraphInterface,
    ModelDefinition,
    NodeInstance,
    NodePosition,
    PortReference,
    Project,
    ProjectMetadata,
    UIState,
    Viewport,
)
from neural_blueprint.ir.serialization import (
    deserialize_project,
    load_project_file,
    save_project_file,
    serialize_project,
)


def create_sample_project() -> Project:
    return Project(
        project=ProjectMetadata(
            id="proj_test",
            name="MLP Test Project",
            created_at="2026-08-18T12:00:00Z",
            updated_at="2026-08-18T12:00:00Z",
        ),
        model=ModelDefinition(
            root_graph_id="graph_mlp",
            config={"n_embd": 64, "n_hidden": 256},
            graphs={
                "graph_mlp": GraphDefinition(
                    id="graph_mlp",
                    name="Two Layer MLP",
                    kind="root",
                    interface=GraphInterface(),
                    nodes=[
                        NodeInstance(
                            id="node_in",
                            definition_id="builtin.tensor_input@1",
                            display_name="Input",
                            properties={"name": "input"},
                        ),
                        NodeInstance(
                            id="node_fc1",
                            definition_id="builtin.linear@1",
                            display_name="FC1",
                            properties={
                                "in_features": ConfigRefValue(key="n_embd"),
                                "out_features": ConfigRefValue(key="n_hidden"),
                            },
                        ),
                        NodeInstance(
                            id="node_gelu",
                            definition_id="builtin.gelu@1",
                            display_name="GELU",
                        ),
                        NodeInstance(
                            id="node_fc2",
                            definition_id="builtin.linear@1",
                            display_name="FC2",
                            properties={
                                "in_features": ConfigRefValue(key="n_hidden"),
                                "out_features": ConfigRefValue(key="n_embd"),
                            },
                        ),
                        NodeInstance(
                            id="node_out",
                            definition_id="builtin.graph_output@1",
                            display_name="Output",
                            properties={"name": "output"},
                        ),
                    ],
                    edges=[
                        Edge(
                            id="e1",
                            source=PortReference(node_id="node_in", port_id="output"),
                            target=PortReference(node_id="node_fc1", port_id="input"),
                        ),
                        Edge(
                            id="e2",
                            source=PortReference(node_id="node_fc1", port_id="output"),
                            target=PortReference(node_id="node_gelu", port_id="input"),
                        ),
                        Edge(
                            id="e3",
                            source=PortReference(node_id="node_gelu", port_id="output"),
                            target=PortReference(node_id="node_fc2", port_id="input"),
                        ),
                        Edge(
                            id="e4",
                            source=PortReference(node_id="node_fc2", port_id="output"),
                            target=PortReference(node_id="node_out", port_id="input"),
                        ),
                    ],
                )
            },
            weight_bindings=[],
        ),
        ui=UIState(
            graph_viewports={"graph_mlp": Viewport(x=0, y=0, zoom=1.0)},
            node_positions={
                "graph_mlp": {
                    "node_in": NodePosition(x=100, y=100),
                    "node_fc1": NodePosition(x=300, y=100),
                    "node_gelu": NodePosition(x=500, y=100),
                    "node_fc2": NodePosition(x=700, y=100),
                    "node_out": NodePosition(x=900, y=100),
                }
            },
            open_graph_id="graph_mlp",
        ),
    )


def test_serialization_round_trip():
    project = create_sample_project()
    serialized = serialize_project(project)

    assert isinstance(serialized, dict)
    assert serialized["schema_version"] == 1
    assert serialized["model"]["config"]["n_embd"] == 64

    deserialized = deserialize_project(serialized)
    assert deserialized.project.id == project.project.id
    assert len(deserialized.model.graphs["graph_mlp"].nodes) == 5
    assert len(deserialized.model.graphs["graph_mlp"].edges) == 4
    assert deserialized.ui.node_positions["graph_mlp"]["node_fc1"].x == 300


def test_atomic_file_persistence():
    project = create_sample_project()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_project.nbp.json"
        save_project_file(project, filepath)

        assert filepath.exists()

        loaded = load_project_file(filepath)
        assert loaded.project.id == "proj_test"
        assert loaded.model.config["n_hidden"] == 256


def test_future_schema_version_rejection():
    data = serialize_project(create_sample_project())
    data["schema_version"] = 999  # Future unsupported version

    with pytest.raises(ValueError, match="newer than maximum supported version"):
        deserialize_project(data)
