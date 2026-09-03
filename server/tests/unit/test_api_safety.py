"""Focused API safety tests for cooker, compile validation, path sandboxing, tester threading, and session capacity."""

from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from neural_blueprint.api.main import app
from neural_blueprint.ir.models import NodeInstance, TrainingConfig
from neural_blueprint.ir.serialization import serialize_project
from neural_blueprint.paths import resolve_sandbox_path
from neural_blueprint.runtime.training_capabilities import (
    UnsupportedTrainingConfigError,
    validate_training_capabilities,
)
from neural_blueprint.templates.architectures import create_arch_5_bottleneck_mlp
from neural_blueprint.templates.nanogpt import create_nanogpt_template
from neural_blueprint.tracing.debugger import (
    SessionCapacityError,
    SessionManager,
    TrainingSession,
)
from tests.unit.test_serialization import create_sample_project

import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_cook_export_rejects_unsupported_topology(client):
    project = create_arch_5_bottleneck_mlp()
    project.model.graphs[project.model.root_graph_id].nodes.append(
        NodeInstance(
            id="node_loader",
            definition_id="builtin.dataloader@1",
            display_name="Data Loader",
            properties={},
        )
    )
    response = client.post(
        "/api/v1/cook/export",
        json={"project": serialize_project(project)},
    )
    assert response.status_code == 422
    assert "Unsupported blueprint topology" in response.json()["detail"]


def test_cook_export_rejects_absolute_and_traversal_paths(client):
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    payload = {"project": serialize_project(project)}

    abs_resp = client.post("/api/v1/cook/export", json={**payload, "output_path": "/tmp/train.py"})
    assert abs_resp.status_code == 400

    trav_resp = client.post(
        "/api/v1/cook/export",
        json={**payload, "output_path": "../outside/train.py"},
    )
    assert trav_resp.status_code == 400


def test_compile_rejects_invalid_mode_and_unavailable_device(client):
    project = create_sample_project()
    payload = {"project": serialize_project(project), "device": "cpu"}

    mode_resp = client.post("/api/v1/models/compile", json={**payload, "mode": "invalid-mode"})
    assert mode_resp.status_code == 422

    with patch("neural_blueprint.api.routes.models.torch.cuda.is_available", return_value=False):
        cuda_resp = client.post("/api/v1/models/compile", json={**payload, "device": "cuda"})
    assert cuda_resp.status_code == 422


@pytest.mark.parametrize(
    "training_update,expected_detail",
    [
        ({"ddp_enabled": True}, "ddp_enabled=True"),
        ({"grad_accum_steps": 4}, "grad_accum_steps=4"),
        ({"precision": "fp8"}, "FP8 precision"),
    ],
)
def test_compile_training_rejects_unsupported_capabilities(
    client, training_update, expected_detail
):
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    project.model.training = project.model.training.model_copy(update=training_update)
    payload = {
        "project": serialize_project(project),
        "device": "cpu",
        "mode": "training",
    }

    response = client.post("/api/v1/models/compile", json=payload)
    assert response.status_code == 422
    assert expected_detail in response.json()["detail"]


def test_compile_inference_allows_unsupported_training_flags(client):
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    project.model.training = project.model.training.model_copy(
        update={"ddp_enabled": True, "grad_accum_steps": 8, "precision": "fp8"}
    )
    payload = {
        "project": serialize_project(project),
        "device": "cpu",
        "mode": "inference",
    }

    response = client.post("/api/v1/models/compile", json=payload)
    assert response.status_code == 200


def test_session_promotion_rejects_unsupported_training_config(client):
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    project.model.training = project.model.training.model_copy(update={"ddp_enabled": True})
    compile_resp = client.post(
        "/api/v1/models/compile",
        json={"project": serialize_project(project), "device": "cpu", "mode": "inference"},
    )
    assert compile_resp.status_code == 200
    session_id = compile_resp.json()["session_id"]

    step_resp = client.post(f"/api/v1/sessions/{session_id}/step-batch")
    assert step_resp.status_code == 422
    assert "ddp_enabled=True" in step_resp.json()["detail"]


def test_validate_training_capabilities_rejects_unsupported_settings():
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    project.model.training = TrainingConfig(precision="fp8")

    with pytest.raises(UnsupportedTrainingConfigError, match="FP8 precision"):
        validate_training_capabilities(project)


def test_session_manager_create_training_session_enforces_capabilities():
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    project.model.training = TrainingConfig(grad_accum_steps=2)
    manager = SessionManager()

    with pytest.raises(UnsupportedTrainingConfigError):
        manager.create_training_session("bad-training", project, device="cpu")


def test_checkpoint_paths_are_sandboxed(client):
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    compile_resp = client.post(
        "/api/v1/models/compile",
        json={"project": serialize_project(project), "device": "cpu", "mode": "training"},
    )
    session_id = compile_resp.json()["session_id"]

    bad_save = client.post(
        f"/api/v1/sessions/{session_id}/checkpoint/save",
        json={"path": "/etc/passwd"},
    )
    assert bad_save.status_code == 400

    good_save = client.post(
        f"/api/v1/sessions/{session_id}/checkpoint/save",
        json={"path": "pytest/session_ckpt.pt"},
    )
    assert good_save.status_code == 200
    saved = good_save.json()["path"]
    assert str(resolve_sandbox_path("pytest/session_ckpt.pt", "checkpoints")) == saved


@pytest.mark.asyncio
async def test_checkpoint_sandbox_roundtrip_restores_training_state():
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    session1 = TrainingSession(session_id="ckpt_roundtrip_1", project=project, device="cpu")

    session1.step = 2
    session1.epoch = 1
    session1.best_loss = 0.42
    session1.loss_history = [
        {"step": 0, "loss": 1.0, "lr": 6e-4, "grad_norm": 0.5, "tokens_per_sec": 100.0},
        {"step": 1, "loss": 0.42, "lr": 6e-4, "grad_norm": 0.4, "tokens_per_sec": 110.0},
    ]
    rel_path = "pytest/roundtrip_ckpt.pt"

    sandboxed = resolve_sandbox_path(rel_path, "checkpoints")
    saved_path = session1.save_checkpoint(str(sandboxed))
    assert saved_path == str(sandboxed)
    assert sandboxed.exists()

    session2 = TrainingSession(session_id="ckpt_roundtrip_2", project=project, device="cpu")
    result = session2.load_checkpoint(rel_path)

    assert result["step"] == 2
    assert session2.step == 2
    assert session2.loss_history == session1.loss_history
    assert session2.best_loss == session1.best_loss

    for (name1, p1), (name2, p2) in zip(
        session1.model.named_parameters(), session2.model.named_parameters()
    ):
        assert name1 == name2
        torch.testing.assert_close(p1, p2)

    for (name1, p1), (name2, p2) in zip(
        session1.optimizer.state_dict().items(), session2.optimizer.state_dict().items()
    ):
        assert name1 == name2
        if isinstance(p1, torch.Tensor):
            torch.testing.assert_close(p1, p2)


def test_checkpoint_load_rejects_non_weights_only_payload():
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    session = TrainingSession(session_id="ckpt_security", project=project, device="cpu")

    evil_path = resolve_sandbox_path("pytest/evil_ckpt.pt", "checkpoints")
    torch.save({"model_state_dict": object()}, evil_path)

    with pytest.raises(Exception) as exc_info:
        session.load_checkpoint("pytest/evil_ckpt.pt")

    assert (
        "Weights only load failed" in str(exc_info.value)
        or exc_info.type.__name__ == "UnpicklingError"
    )


def test_session_manager_capacity_prunes_idle_and_rejects_active_only():
    manager = SessionManager()
    project = create_sample_project()

    for idx in range(SessionManager.MAX_SESSIONS):
        manager.create_session(f"sess-{idx}", project, device="cpu")

    oldest_id = "sess-0"
    manager._sessions[oldest_id].state = "idle"
    manager._sessions[oldest_id].last_active_timestamp = 1.0

    manager.create_session("sess-new", project, device="cpu")
    assert oldest_id not in manager._sessions
    assert "sess-new" in manager._sessions

    for session_id, session in manager._sessions.items():
        session.state = "running"

    with pytest.raises(SessionCapacityError):
        manager.create_session("sess-overflow", project, device="cpu")


@pytest.mark.asyncio
async def test_tester_route_runs_suite_in_executor():
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    calls = {"executor": 0}

    async def fake_run_in_executor(executor, func, *args):
        calls["executor"] += 1
        return func(*args)

    with patch("neural_blueprint.api.routes.tester.asyncio.get_running_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = fake_run_in_executor
        from neural_blueprint.api.routes.tester import TestRunRequest, run_testing_suite

        await run_testing_suite(TestRunRequest(project=project, enabled_tests=["shape_sanity"]))

    assert calls["executor"] == 1

def test_cook_export_rejects_invalid_project(client):
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=3,
        n_embd=16,
    )
    payload = serialize_project(project)
    if isinstance(payload, str):
        import json
        payload = json.loads(payload)
    response = client.post(
        "/api/v1/cook/export",
        json={"project": payload},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "diagnostics" in detail

