"""Regression tests for session manager and sessions API behavior."""

import asyncio
import time
from types import SimpleNamespace
from urllib.parse import quote

import torch
from fastapi.testclient import TestClient
from neural_blueprint.api.main import app
from neural_blueprint.templates.architectures import create_arch_7_dual_flow_pipeline
from neural_blueprint.templates.nanogpt import create_nanogpt_template
from neural_blueprint.tracing.debugger import (
    RuntimeSession,
    SessionManager,
    TrainingSession,
    global_session_manager,
)
from neural_blueprint.tracing.events import TensorSummary

import pytest


@pytest.fixture
def tiny_nanogpt_project():
    return create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )


def test_session_capacity_reclaims_completed_sessions(tiny_nanogpt_project):
    manager = SessionManager()
    for i in range(32):
        fake = SimpleNamespace(
            state="completed",
            last_active_timestamp=time.time(),
            stop=lambda: None,
        )
        manager._sessions[f"s{i}"] = fake

    session = manager.create_session("new", tiny_nanogpt_project)
    assert session is not None
    assert "new" in manager._sessions


def test_register_session_stops_existing():
    manager = SessionManager()
    calls = {"count": 0}

    def stop_a():
        calls["count"] += 1

    fake_a = SimpleNamespace(state="idle", last_active_timestamp=0.0, stop=stop_a)
    fake_b = SimpleNamespace(state="idle", last_active_timestamp=1.0, stop=lambda: None)
    manager._register_session("same", fake_a)
    manager._register_session("same", fake_b)
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_launch_background_cancels_previous(tiny_nanogpt_project):
    session = RuntimeSession("bg", tiny_nanogpt_project, device="cpu")
    first = session._launch_background(asyncio.sleep(3600))
    second = session._launch_background(asyncio.sleep(3600))
    await asyncio.sleep(0)
    assert first.cancelled() or first.done()
    assert not second.done()


def test_prepare_run_moves_inputs_to_device(tiny_nanogpt_project):
    session = RuntimeSession("prep", tiny_nanogpt_project, device="cpu")
    session.device = "meta"
    session.prepare_run({"token_ids": [[1, 2, 3]]})
    stored = session.value_table[("token_ids", "output")]
    assert isinstance(stored, torch.Tensor)
    assert stored.device.type == "meta"


@pytest.mark.asyncio
async def test_step_batch_survives_batch_size_larger_than_dataset(tiny_nanogpt_project):
    session = TrainingSession("big-batch", tiny_nanogpt_project, device="cpu")
    session.batch_size = 2500
    event = await session.step_batch()
    assert event is not None
    assert event.error is None


@pytest.mark.asyncio
async def test_dual_flow_without_backward_nodes_still_updates_parameters():
    project = create_arch_7_dual_flow_pipeline()
    graph = project.model.graphs["graph_dual_flow"]
    remove_ids = {"node_backward", "node_clip_grad", "node_opt_step", "node_lr_sched"}
    graph.nodes = [n for n in graph.nodes if n.id not in remove_ids]
    graph.edges = [
        e
        for e in graph.edges
        if e.source.node_id not in remove_ids and e.target.node_id not in remove_ids
    ]

    session = TrainingSession("dual-flow-fallback", project, device="cpu")
    session.update_hyperparameters(learning_rate=1e-2)
    before = [p.detach().clone() for p in session.model.parameters()]
    event = await session.step_batch()
    max_delta = max(
        (a - b).abs().max().item() for a, b in zip(before, session.model.parameters())
    )
    assert event is not None
    assert event.error is None
    assert max_delta > 0.0
    assert any(p.grad is not None for p in session.model.parameters())


def test_dataloader_batch_size_property_honored():
    project = create_arch_7_dual_flow_pipeline()
    graph = project.model.graphs["graph_dual_flow"]
    dataloader = next(n for n in graph.nodes if n.id == "node_dataloader")
    dataloader.properties["batch_size"] = 7

    session = TrainingSession("dl-batch", project, device="cpu")
    assert session.batch_size == 7
    asyncio.run(session.step_batch())
    batch_x = session.value_table[("dataloader", "batch_x")]
    assert batch_x is not None
    assert batch_x.shape[0] == 7

@pytest.mark.asyncio
async def test_training_resume_cancels_background_task(tiny_nanogpt_project):
    session = TrainingSession("resume-bg", tiny_nanogpt_project, device="cpu")

    async def idle_loop(max_steps=None, speed_delay=0.0):
        await asyncio.sleep(3600)

    session.run_training_loop = idle_loop
    first = session._launch_background(session.run_training_loop())
    await asyncio.sleep(0)
    session.pause()
    session.resume(speed_delay=0.0)
    await asyncio.sleep(0)
    assert first.cancelled()
    assert session._background_task is not None
    assert session._background_task is not first
    assert not session._background_task.done()
    session.stop()

def test_tensor_summary_exact_match_beats_substring(tiny_nanogpt_project):
    session = global_session_manager.create_session(
        "tensor-summary", tiny_nanogpt_project, device="cpu"
    )
    try:
        session.retained_summaries["node_a:out"] = TensorSummary(shape=[1])
        session.retained_summaries["node_ab:out"] = TensorSummary(shape=[2])

        client = TestClient(app)
        response = client.get(
            f"/api/v1/sessions/{session.session_id}/tensors/{quote('node_a:out', safe='')}/summary"
        )
        assert response.status_code == 200
        assert response.json()["shape"] == [1]
    finally:
        global_session_manager.remove_session(session.session_id)


@pytest.mark.asyncio
async def test_infer_route_returns_outputs(tiny_nanogpt_project):
    session = global_session_manager.create_training_session(
        "infer-route", tiny_nanogpt_project, device="cpu"
    )
    try:
        client = TestClient(app)
        response = client.post(
            f"/api/v1/sessions/{session.session_id}/infer",
            json={"inputs": {}},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["outputs"]
    finally:
        global_session_manager.remove_session(session.session_id)
