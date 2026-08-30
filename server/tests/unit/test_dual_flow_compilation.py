"""Unit tests verifying Dual-Flow execution wires and topological precedence in GraphCompiler."""

from neural_blueprint.ir.models import (
    Edge,
    GraphDefinition,
    ModelDefinition,
    NodeInstance,
    PortReference,
    Project,
    ProjectMetadata,
    TrainingConfig,
)
from neural_blueprint.runtime.compiler import GraphCompiler


def test_dual_flow_exec_wires_compilation():
    """
    Constructs a project with explicit Exec wires connecting:
    DatasetSource -> DataLoader -> Forward (Linear) -> Loss (CrossEntropy) -> Backward -> ClipGrad -> OptimizerStep
    and verifies that GraphCompiler compiles instructions honoring execution precedence.
    """
    nodes = [
        NodeInstance(
            id="node_dataset",
            definition_id="builtin.dataset_source@1",
            display_name="Dataset Source",
        ),
        NodeInstance(
            id="node_dataloader",
            definition_id="builtin.dataloader@1",
            display_name="DataLoader",
        ),
        NodeInstance(
            id="node_forward",
            definition_id="builtin.linear@1",
            display_name="Forward Linear",
            properties={"in_features": 16, "out_features": 16},
        ),
        NodeInstance(
            id="node_loss",
            definition_id="builtin.cross_entropy_loss@1",
            display_name="Cross Entropy Loss",
        ),
        NodeInstance(
            id="node_backward",
            definition_id="builtin.backward@1",
            display_name="Backward",
        ),
        NodeInstance(
            id="node_clip_grad",
            definition_id="builtin.clip_gradients@1",
            display_name="Clip Gradients",
            properties={"max_norm": 1.0},
        ),
        NodeInstance(
            id="node_opt_step",
            definition_id="builtin.optimizer_step@1",
            display_name="Optimizer Step",
        ),
    ]

    # Full unbroken chain of Exec wires connecting every step in order
    edges = [
        # Exec wire: DatasetSource -> DataLoader
        Edge(
            id="edge_exec_0",
            source=PortReference(node_id="node_dataset", port_id="exec_out"),
            target=PortReference(node_id="node_dataloader", port_id="exec_in"),
        ),
        # Exec wire: DataLoader -> Forward
        Edge(
            id="edge_exec_1",
            source=PortReference(node_id="node_dataloader", port_id="exec_out"),
            target=PortReference(node_id="node_forward", port_id="exec_in"),
        ),
        # Exec wire: Forward -> Loss
        Edge(
            id="edge_exec_2",
            source=PortReference(node_id="node_forward", port_id="exec_out"),
            target=PortReference(node_id="node_loss", port_id="exec_in"),
        ),
        # Exec wire: Loss -> Backward
        Edge(
            id="edge_exec_3",
            source=PortReference(node_id="node_loss", port_id="exec_out"),
            target=PortReference(node_id="node_backward", port_id="exec_in"),
        ),
        # Exec wire: Backward -> ClipGrad
        Edge(
            id="edge_exec_4",
            source=PortReference(node_id="node_backward", port_id="exec_out"),
            target=PortReference(node_id="node_clip_grad", port_id="exec_in"),
        ),
        # Exec wire: ClipGrad -> OptimizerStep
        Edge(
            id="edge_exec_5",
            source=PortReference(node_id="node_clip_grad", port_id="exec_out"),
            target=PortReference(node_id="node_opt_step", port_id="exec_in"),
        ),
        # Data wire: dataset -> dataloader
        Edge(
            id="edge_data_1",
            source=PortReference(node_id="node_dataset", port_id="dataset"),
            target=PortReference(node_id="node_dataloader", port_id="dataset"),
        ),
        # Data wire: dataloader.batch_x -> forward.input
        Edge(
            id="edge_data_2",
            source=PortReference(node_id="node_dataloader", port_id="batch_x"),
            target=PortReference(node_id="node_forward", port_id="input"),
        ),
        # Data wire: forward.output -> loss.logits
        Edge(
            id="edge_data_3",
            source=PortReference(node_id="node_forward", port_id="output"),
            target=PortReference(node_id="node_loss", port_id="logits"),
        ),
        # Data wire: dataloader.batch_y -> loss.targets
        Edge(
            id="edge_data_4",
            source=PortReference(node_id="node_dataloader", port_id="batch_y"),
            target=PortReference(node_id="node_loss", port_id="targets"),
        ),
        # Data wire: loss.loss -> backward.loss
        Edge(
            id="edge_data_5",
            source=PortReference(node_id="node_loss", port_id="loss"),
            target=PortReference(node_id="node_backward", port_id="loss"),
        ),
    ]

    graph = GraphDefinition(
        id="graph_training",
        name="Dual Flow Training Graph",
        kind="training_event",
        nodes=nodes,
        edges=edges,
    )

    project = Project(
        schema_version=1,
        project=ProjectMetadata(
            id="proj_dual_flow",
            name="Dual Flow Test",
            created_at="2026-08-25T00:00:00Z",
            updated_at="2026-08-25T00:00:00Z",
        ),
        model=ModelDefinition(
            root_graph_id="graph_training",
            config={"n_embd": 16},
            training=TrainingConfig(learning_rate=1e-3),
            graphs={"graph_training": graph},
        ),
    )

    compiler = GraphCompiler()
    plan, modules = compiler.compile_plan(project)

    assert plan is not None
    assert len(plan.instructions) == 7

    # Verify instructions honor exact execution precedence in order
    inst_ids = [inst.node_id for inst in plan.instructions]
    expected_order = [
        "node_dataset",
        "node_dataloader",
        "node_forward",
        "node_loss",
        "node_backward",
        "node_clip_grad",
        "node_opt_step",
    ]
    assert inst_ids == expected_order, f"Expected order {expected_order} but got {inst_ids}"

    # Verify all instructions are identified as exec instructions
    assert len(plan.exec_instructions) == 7
