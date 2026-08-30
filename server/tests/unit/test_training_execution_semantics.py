"""Behavioral tests for single-pass architecture and dual-flow training execution."""

from unittest.mock import patch

from neural_blueprint.ir.models import Edge, PortReference
from neural_blueprint.runtime.compiler import GraphCompiler
from neural_blueprint.templates.architectures import create_arch_7_dual_flow_pipeline
from neural_blueprint.templates.nanogpt import create_nanogpt_template
from neural_blueprint.tracing.debugger import TrainingSession

import pytest


@pytest.mark.asyncio
async def test_architecture_training_executes_plan_without_whole_graph_fallback():
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    session = TrainingSession("architecture-plan", project, device="cpu")

    with patch.object(
        session.model,
        "forward",
        side_effect=AssertionError("architecture plan called whole-graph fallback"),
    ) as forward:
        event = await session.step_batch()

    assert event is not None
    assert event.error is None
    assert session.step == 1
    assert len(session.loss_history) == 1
    forward.assert_not_called()


@pytest.mark.asyncio
async def test_dual_flow_training_does_not_call_whole_graph_fallback():
    project = create_arch_7_dual_flow_pipeline()
    session = TrainingSession("exec-only", project, device="cpu")

    with patch.object(
        session.model,
        "forward",
        side_effect=AssertionError("dual-flow execution called whole-graph fallback"),
    ) as forward:
        event = await session.step_batch()

    assert event is not None
    assert event.error is None
    assert session.step == 1
    assert len(session.loss_history) == 1
    forward.assert_not_called()


def test_compiler_rejects_cyclic_graph_without_fallback_order():
    project = create_arch_7_dual_flow_pipeline()
    graph = project.model.graphs[project.model.root_graph_id]
    graph.edges.append(
        Edge(
            id="cycle",
            source=PortReference(node_id="node_zero_grad", port_id="exec_out"),
            target=PortReference(node_id="node_dataset", port_id="exec_in"),
        )
    )

    with pytest.raises(ValueError, match="Cannot compile graph with cycles"):
        GraphCompiler().compile_plan(project)
