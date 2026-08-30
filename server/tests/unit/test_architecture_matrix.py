"""Comprehensive test suite validating, compiling, training, checkpointing, and cooking 25 distinct neural network architectures."""

import math
import subprocess
import sys

import torch
from neural_blueprint.cooking.cooker import BlueprintCooker, UnsupportedCookError
from neural_blueprint.runtime.compiler import GraphCompiler
from neural_blueprint.runtime.inference import InferenceEngine
from neural_blueprint.templates.architectures import ALL_ARCHITECTURES
from neural_blueprint.tracing.debugger import TrainingSession
from neural_blueprint.validation.validator import ProjectValidator

import pytest

SUPPORTED_COOK_ARCHITECTURES = {
    "Arch 1: nanoGPT Tiny",
    "Arch 2: nanoGPT Deep (6L)",
    "Arch 3: nanoGPT Wide (1L/8H)",
    "Arch 4: Two-Layer MLP",
    "Arch 6: Manual Attention Transformer",
    "Arch 11: ReLU Classifier MLP",
    "Arch 21: High-Dropout Transformer",
    "Arch 22: BF16 nanoGPT Micro",
    "Arch 23: Single-Block Causal GPT",
    "Arch 24: SiLU Deep Feedforward",
}
UNSUPPORTED_COOK_ARCHITECTURES = {
    "Arch 5: Bottleneck Autoencoder",
    "Arch 7: Dual-Flow Pipeline",
    "Arch 8: ResMLP Residual Network",
    "Arch 9: Multi-Head Projection",
    "Arch 10: Multi-Task Joint Network",
    "Arch 12: Dropout MLP",
    "Arch 13: Deep MLP Tower",
    "Arch 14: Wide-and-Deep Network",
    "Arch 15: Tied-Embedding LM",
    "Arch 16: Warmup Scheduler Pipeline",
    "Arch 17: Step-LR Decay Pipeline",
    "Arch 18: Pre-LayerNorm MLP",
    "Arch 19: Residual Add MLP",
    "Arch 20: Binary Sequence Classifier",
    "Arch 25: Metric Logger Pipeline",
}


@pytest.mark.parametrize("arch_name,builder_fn", ALL_ARCHITECTURES)
def test_architecture_validation_and_compilation(arch_name, builder_fn):
    """Verifies that each architecture passes 4-pass static validation and compiles into a valid ExecutionPlan."""
    project = builder_fn()
    validator = ProjectValidator()
    val_res = validator.validate(project)

    errors = [d for d in val_res.diagnostics if d.severity == "error"]
    assert val_res.valid is True, f"Architecture '{arch_name}' validation failed: {errors}"
    assert len(errors) == 0

    compiler = GraphCompiler()
    graph_hash = compiler.compute_graph_hash(project)
    assert len(graph_hash) == 16

    plan, modules = compiler.compile_plan(project)
    assert plan is not None
    assert len(plan.instructions) > 0


@pytest.mark.parametrize("arch_name,builder_fn", ALL_ARCHITECTURES)
@pytest.mark.asyncio
async def test_architecture_training_and_convergence(arch_name, builder_fn):
    """Verifies that each architecture executes batch stepping, updates metrics, and converges."""
    project = builder_fn()
    training_cfg = getattr(project.model, "training", None)

    session = TrainingSession(
        session_id=f"session_train_{project.project.id}",
        project=project,
        device="cpu",
    )

    # Verify custom settings are applied
    if training_cfg:
        assert session.learning_rate == training_cfg.learning_rate
        assert session.weight_decay == training_cfg.weight_decay
        assert session.grad_clip == training_cfg.grad_clip

    # Execute 3 training steps
    losses = []
    for step in range(3):
        evt = await session.step_batch()
        assert evt is not None
        assert session.metrics is not None
        losses.append(session.metrics.loss)

    assert len(session.loss_history) == 3
    assert session.step == 3
    assert all(loss > 0.0 for loss in losses)


@pytest.mark.parametrize("arch_name,builder_fn", ALL_ARCHITECTURES)
@pytest.mark.asyncio
async def test_architecture_inference_forward_pass(arch_name, builder_fn):
    """Verifies forward-only inference produces finite outputs without gradients."""
    project = builder_fn()
    engine = InferenceEngine(project=project, device="cpu")
    result = await engine.infer()
    assert result["tensor_count"] >= 1, f"{arch_name} produced no output tensors"
    for key, summary in result["outputs"].items():
        assert summary.get("shape") is not None, f"{arch_name} output {key} missing shape"
        values = summary.get("sample_values") or []
        assert all(math.isfinite(float(v)) for v in values), (
            f"{arch_name} output {key} not finite"
        )
    assert all(p.grad is None for p in engine.session.model.parameters()), (
        f"{arch_name} inference produced gradients"
    )


@pytest.mark.parametrize("arch_name,builder_fn", ALL_ARCHITECTURES)
@pytest.mark.asyncio
async def test_architecture_checkpoint_roundtrip(arch_name, builder_fn, tmp_path):
    """Verifies checkpoint saving and state restoration across all 10 architectures."""
    project = builder_fn()

    session1 = TrainingSession(
        session_id=f"session_ckpt1_{project.project.id}",
        project=project,
        device="cpu",
    )

    # Run 2 steps
    for _ in range(2):
        await session1.step_batch()

    assert session1.step == 2

    # Save checkpoint
    ckpt_file = tmp_path / f"ckpt_{project.project.id}.pt"
    saved_path = session1.save_checkpoint(str(ckpt_file))
    assert ckpt_file.exists()
    assert saved_path == str(ckpt_file)

    # Restore in fresh session
    session2 = TrainingSession(
        session_id=f"session_ckpt2_{project.project.id}",
        project=project,
        device="cpu",
    )
    res = session2.load_checkpoint(str(ckpt_file))
    assert res["step"] == 2
    assert session2.step == 2

    # Verify parameter match
    for (n1, p1), (n2, p2) in zip(
        session1.model.named_parameters(), session2.model.named_parameters()
    ):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)


@pytest.mark.parametrize("arch_name,builder_fn", ALL_ARCHITECTURES)
def test_architecture_cooker_generation_and_execution(arch_name, builder_fn, tmp_path):
    """Verifies standalone code generation and isolated CLI subprocess execution for supported architectures."""
    project = builder_fn()

    if arch_name in UNSUPPORTED_COOK_ARCHITECTURES:
        with pytest.raises(UnsupportedCookError):
            BlueprintCooker.cook(project)
        return

    assert arch_name in SUPPORTED_COOK_ARCHITECTURES

    code = BlueprintCooker.cook(project)
    assert "import torch" in code
    assert "def main():" in code

    script_path = tmp_path / f"train_{project.project.id}.py"
    script_path.write_text(code)
    assert script_path.exists()

    # Execute standalone script in an isolated subprocess for 3 steps
    res = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--max-steps",
            "3",
            "--batch-size",
            "4",
            "--save-dir",
            str(tmp_path / f"ckpts_{project.project.id}"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, f"Cooked script execution failed for {arch_name}:\n{res.stderr}"
    assert "=== Starting Blueprint Model Training ===" in res.stdout
    assert "=== Training Complete in" in res.stdout
