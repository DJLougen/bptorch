"""Unit tests for nanoGPT architecture template and hierarchical execution."""

import torch
from neural_blueprint.runtime.compiler import GraphCompiler
from neural_blueprint.runtime.initialization import init_nanogpt_weights
from neural_blueprint.runtime.module import CompiledGraphModule
from neural_blueprint.runtime.parameters import ParameterAccounting
from neural_blueprint.templates.nanogpt import create_nanogpt_template
from neural_blueprint.validation.validator import ProjectValidator


def test_nanogpt_template_validation():
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )

    validator = ProjectValidator()
    result = validator.validate(project)

    assert result.valid is True
    assert len(result.errors) == 0


def test_nanogpt_template_parameter_accounting():
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )

    accounting = ParameterAccounting()
    summary = accounting.calculate_summary(project)

    assert summary.total_unique > 0
    assert summary.trainable == summary.total_unique
    assert summary.shared_references == 1  # Tied weights


def test_nanogpt_compiled_forward_execution():
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )

    compiler = GraphCompiler()
    plan, modules = compiler.compile_plan(project)
    model = CompiledGraphModule(plan, modules, project.model.weight_bindings)

    # Initialize weights
    init_nanogpt_weights(model, n_layer=2)

    # Forward pass with discrete token IDs and targets
    token_ids = torch.tensor([[1, 5, 12, 18, 3, 7, 22, 30]], dtype=torch.long)
    targets = torch.tensor([[5, 12, 18, 3, 7, 22, 30, 2]], dtype=torch.long)

    outputs = model(token_ids=token_ids, targets=targets)

    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert "loss" in outputs

    logits = outputs["logits"]
    loss = outputs["loss"]

    assert logits.shape == torch.Size([1, 8, 32])
    assert loss.dim() == 0  # Scalar loss
    assert not torch.isnan(loss).item()
