"""Unit tests for Canonical Model IR data models."""

from neural_blueprint.ir.models import (
    ConfigRefValue,
    ExpressionOp,
    ExpressionValue,
    GraphDefinition,
    GraphInterface,
    ModelDefinition,
    NodeInstance,
    NodeMetadata,
    NodePosition,
    Project,
    ProjectMetadata,
    SafeExpression,
    UIState,
    Viewport,
    WeightBinding,
    WeightBindingEndpoint,
)


def test_create_project_model():
    project = Project(
        project=ProjectMetadata(
            id="proj_1",
            name="Test Project",
            created_at="2026-08-18T00:00:00Z",
            updated_at="2026-08-18T00:00:00Z",
        ),
        model=ModelDefinition(
            root_graph_id="graph_main",
            config={"n_embd": 64, "n_head": 4},
            graphs={
                "graph_main": GraphDefinition(
                    id="graph_main",
                    name="Main Graph",
                    kind="root",
                    interface=GraphInterface(),
                    nodes=[
                        NodeInstance(
                            id="node_linear",
                            definition_id="builtin.linear@1",
                            display_name="Linear Layer",
                            properties={
                                "in_features": ConfigRefValue(key="n_embd"),
                                "out_features": ExpressionValue(
                                    expression=SafeExpression(
                                        op=ExpressionOp.MULTIPLY,
                                        left=4,
                                        right=ConfigRefValue(key="n_embd"),
                                    )
                                ),
                            },
                            metadata=NodeMetadata(breakpoint=True),
                        )
                    ],
                    edges=[],
                )
            },
            weight_bindings=[
                WeightBinding(
                    source=WeightBindingEndpoint(node_id="node_a", parameter="weight"),
                    target=WeightBindingEndpoint(node_id="node_b", parameter="weight"),
                    mode="share",
                )
            ],
        ),
        ui=UIState(
            graph_viewports={"graph_main": Viewport(x=100.0, y=200.0, zoom=1.5)},
            node_positions={"graph_main": {"node_linear": NodePosition(x=150.0, y=250.0)}},
            open_graph_id="graph_main",
        ),
    )

    assert project.schema_version == 1
    assert project.project.name == "Test Project"
    assert project.model.config["n_embd"] == 64
    assert len(project.model.graphs["graph_main"].nodes) == 1
    assert project.model.graphs["graph_main"].nodes[0].metadata.breakpoint is True
    assert project.ui.node_positions["graph_main"]["node_linear"].x == 150.0
