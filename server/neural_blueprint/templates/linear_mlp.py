"""Two-layer MLP architecture template generator."""

from typing import Any, Dict

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


def create_linear_mlp_template(
    in_features: int = 64,
    hidden_features: int = 256,
    activation: str = "gelu",
) -> Project:
    config: Dict[str, Any] = {
        "n_embd": in_features,
        "n_hidden": hidden_features,
        "activation": activation,
    }

    act_node_id = {
        "gelu": "builtin.gelu@1",
        "relu": "builtin.relu@1",
        "silu": "builtin.silu@1",
    }.get(activation, "builtin.silu@1")

    g_mlp = GraphDefinition(
        id="graph_mlp",
        name="Two-Layer MLP",
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
                id="node_act",
                definition_id=act_node_id,
                display_name=activation.upper(),
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
                target=PortReference(node_id="node_act", port_id="input"),
            ),
            Edge(
                id="e3",
                source=PortReference(node_id="node_act", port_id="output"),
                target=PortReference(node_id="node_fc2", port_id="input"),
            ),
            Edge(
                id="e4",
                source=PortReference(node_id="node_fc2", port_id="output"),
                target=PortReference(node_id="node_out", port_id="input"),
            ),
        ],
    )

    return Project(
        project=ProjectMetadata(
            id="mlp_default",
            name="Two-Layer MLP",
            created_at="2026-08-18T00:00:00Z",
            updated_at="2026-08-18T00:00:00Z",
        ),
        model=ModelDefinition(
            root_graph_id="graph_mlp",
            config=config,
            graphs={"graph_mlp": g_mlp},
            weight_bindings=[],
        ),
        ui=UIState(
            graph_viewports={"graph_mlp": Viewport(x=0, y=0, zoom=1.0)},
            node_positions={
                "graph_mlp": {
                    "node_in": NodePosition(x=100, y=100),
                    "node_fc1": NodePosition(x=350, y=100),
                    "node_act": NodePosition(x=600, y=100),
                    "node_fc2": NodePosition(x=850, y=100),
                    "node_out": NodePosition(x=1100, y=100),
                }
            },
            open_graph_id="graph_mlp",
        ),
    )
