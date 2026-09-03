"""Unit tests verifying TrainingSession lifecycle, stepping, convergence, and checkpointing."""

import torch
from neural_blueprint.templates.nanogpt import create_nanogpt_template
from neural_blueprint.tracing.debugger import TrainingSession

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


@pytest.mark.asyncio
async def test_training_session_step_batch(tiny_nanogpt_project):
    session = TrainingSession(
        session_id="test_train_1",
        project=tiny_nanogpt_project,
        device="cpu",
    )
    session.learning_rate = 1e-2
    for pg in session.optimizer.param_groups:
        pg["lr"] = 1e-2

    # Repeat fixed batch so gradient descent strictly drops loss monotonically
    session.dataset_x = session.dataset_x[:8].repeat(10, 1)
    session.dataset_y = session.dataset_y[:8].repeat(10, 1)

    losses = []
    # Step 5 batches
    for step in range(5):
        event = await session.step_batch()
        assert event is not None
        assert session.metrics is not None
        losses.append(session.metrics.loss)

    assert len(session.loss_history) == 5
    assert session.step == 5
    assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]}, final={losses[-1]}"
    assert all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1)), (
        f"Losses did not strictly decrease: {losses}"
    )
    assert isinstance(session.parameter_norms, dict)
    assert len(session.parameter_norms) > 0
    assert all(isinstance(v, float) for v in session.parameter_norms.values())

@pytest.mark.asyncio
async def test_training_session_hyperparameter_update(tiny_nanogpt_project):
    session = TrainingSession(
        session_id="test_train_hp",
        project=tiny_nanogpt_project,
        device="cpu",
    )

    session.update_hyperparameters(learning_rate=1e-2, weight_decay=0.05, grad_clip=0.5)

    assert session.learning_rate == 1e-2
    assert session.weight_decay == 0.05
    assert session.grad_clip == 0.5
    assert session.optimizer.param_groups[0]["lr"] == 1e-2
    assert session.optimizer.param_groups[0]["weight_decay"] == 0.05


@pytest.mark.asyncio
async def test_training_session_checkpoint_save_and_continuous_step_parity(
    tiny_nanogpt_project, tmp_path
):
    """
    Plan §5 Checkpoint Save & Resume Verification:
    Train for 3 steps, trigger SaveCheckpointNode, initialize a fresh runtime session,
    load the checkpoint via LoadCheckpointNode, and train for step 4.
    Verify parameter values and optimizer states match continuous 4-step training without interruption.
    """
    # 1. Uninterrupted continuous 4-step training reference
    session_cont = TrainingSession(
        session_id="session_continuous_4step",
        project=tiny_nanogpt_project,
        device="cpu",
    )
    cont_losses = []
    for _ in range(4):
        await session_cont.step_batch()
        cont_losses.append(session_cont.metrics.loss)

    assert session_cont.step == 4

    # 2. Interrupted training: train 3 steps, save checkpoint
    session_interrupted = TrainingSession(
        session_id="session_interrupted_3step",
        project=tiny_nanogpt_project,
        device="cpu",
    )
    for _ in range(3):
        await session_interrupted.step_batch()

    assert session_interrupted.step == 3

    ckpt_file = tmp_path / "ckpt_step3.pt"
    session_interrupted.save_checkpoint(str(ckpt_file))
    assert ckpt_file.exists()

    # 3. Fresh runtime session: restore from checkpoint and train step 4
    session_restored = TrainingSession(
        session_id="session_restored_step4",
        project=tiny_nanogpt_project,
        device="cpu",
    )
    res = session_restored.load_checkpoint(str(ckpt_file))
    assert res["step"] == 3
    assert session_restored.step == 3

    # Execute step 4 on restored session
    await session_restored.step_batch()
    assert session_restored.step == 4

    # 4. Verify parameter values match continuous 4-step training with exact tolerances
    for (name_cont, p_cont), (name_res, p_res) in zip(
        session_cont.model.named_parameters(), session_restored.model.named_parameters()
    ):
        assert name_cont == name_res
        torch.testing.assert_close(p_res, p_cont, rtol=1e-5, atol=1e-6)

    # 5. Verify optimizer states match continuous 4-step training
    for p_cont, p_res in zip(session_cont.model.parameters(), session_restored.model.parameters()):
        state_cont = session_cont.optimizer.state.get(p_cont, {})
        state_res = session_restored.optimizer.state.get(p_res, {})
        for k in ("exp_avg", "exp_avg_sq"):
            if k in state_cont and k in state_res:
                torch.testing.assert_close(state_res[k], state_cont[k], rtol=1e-5, atol=1e-6)

    # 6. Verify final step 4 loss is identical
    assert abs(session_restored.metrics.loss - cont_losses[3]) <= 1e-5

@pytest.mark.asyncio
async def test_shakespeare_dataset_finite_loss():
    import math
    from neural_blueprint.ir.models import NodeInstance
    from neural_blueprint.templates.nanogpt import create_nanogpt_template

    project = create_nanogpt_template(block_size=8, vocab_size=64, n_layer=2, n_head=2, n_embd=16)
    root_g = project.model.graphs[project.model.root_graph_id]
    root_g.nodes.append(
        NodeInstance(
            id="node_dataset_test",
            definition_id="builtin.dataset_source@1",
            display_name="Shakespeare Dataset",
            properties={"dataset_name": "tiny_shakespeare", "synthetic": False},
        )
    )

    session = TrainingSession("sess_shake", project, device="cpu")
    assert len(session.dataset_x) > 0

    await session.step_batch()
    await session.step_batch()

    assert session.metrics is not None
    assert math.isfinite(session.metrics.loss)
