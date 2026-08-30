"""Unit tests for execution tracing, breakpoints, stepping, and tensor inspection."""

import asyncio

import torch
from fastapi.testclient import TestClient
from neural_blueprint.api.main import app
from neural_blueprint.ir.serialization import serialize_project
from neural_blueprint.templates.nanogpt import create_nanogpt_template
from neural_blueprint.tracing.collector import TensorSummarizer
from neural_blueprint.tracing.debugger import global_session_manager


def test_tensor_summarizer_numeric_statistics():
    # 1. Float tensor
    x_float = torch.tensor([[-1.0, 0.0, 1.0], [2.0, float("nan"), 4.0]])
    s_float = TensorSummarizer.summarize(x_float)
    assert s_float.shape == [2, 3]
    assert s_float.dtype == "float32"
    assert s_float.numel == 6
    assert s_float.nan_count == 1
    assert s_float.min == -1.0
    assert s_float.max == 4.0
    assert len(s_float.sample_values) == 6

    # 2. Integer tensor
    x_int = torch.tensor([[10, 20], [30, 40]], dtype=torch.long)
    s_int = TensorSummarizer.summarize(x_int)
    assert s_int.shape == [2, 2]
    assert s_int.dtype == "int64"
    assert s_int.min == 10.0
    assert s_int.max == 40.0


def test_runtime_session_step_and_breakpoint():
    async def async_runner():
        project = create_nanogpt_template(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=True,
        )

        session = global_session_manager.create_session("test_debug_session", project)

        # Set breakpoint on Final LayerNorm
        session.set_breakpoint("node_ln_f", True)

        # Prepare run with sample token batch
        token_ids = [[1, 2, 3, 4, 5, 6, 7, 8]]
        targets = [[2, 3, 4, 5, 6, 7, 8, 1]]
        session.prepare_run({"token_ids": token_ids, "targets": targets})

        # Run until breakpoint
        await session.run_until_breakpoint_or_end(speed_delay=0.0)

        # Assert paused on breakpoint
        assert session.state == "paused"
        assert len(session.retained_summaries) > 0

        # Step single instruction
        evt = await session.step_single()
        assert evt is not None

        # Continue until completion
        await session.run_until_breakpoint_or_end(speed_delay=0.0)
        assert session.state == "completed"

    asyncio.run(async_runner())


def test_session_api_endpoints():
    client = TestClient(app)
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )

    # 1. Compile model to create session
    compile_res = client.post(
        "/api/v1/models/compile",
        json={"project": serialize_project(project), "device": "cpu"},
    )
    assert compile_res.status_code == 200
    session_id = compile_res.json()["session_id"]

    # 2. Run session in step mode
    run_res = client.post(
        f"/api/v1/sessions/{session_id}/run",
        json={
            "mode": "inspection",
            "inputs": {"token_ids": [[1, 2, 3, 4, 5, 6, 7, 8]]},
            "trace": {"enabled": True, "speed": "step"},
        },
    )
    assert run_res.status_code == 200

    # 3. Step session
    step_res = client.post(f"/api/v1/sessions/{session_id}/step")
    assert step_res.status_code == 200
    assert "event" in step_res.json()

    # 4. Stop session
    stop_res = client.post(f"/api/v1/sessions/{session_id}/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "stopped"
