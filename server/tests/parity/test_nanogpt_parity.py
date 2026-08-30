"""Comprehensive numerical parity test suite against pinned karpathy/nanoGPT reference."""

import pytest
from neural_blueprint.parity.runner import ParityRunner


@pytest.fixture
def manual_parity_runner():
    return ParityRunner(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
        attention_impl="manual",
    )


@pytest.fixture
def sdpa_parity_runner():
    return ParityRunner(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
        attention_impl="sdpa",
    )


def test_weight_map_completeness(manual_parity_runner):
    assert manual_parity_runner.check_weight_map_completeness() is True


def test_parameter_count_parity(manual_parity_runner):
    assert manual_parity_runner.check_parameter_count_parity() is True


def test_forward_logits_and_loss_parity_manual(manual_parity_runner):
    assert manual_parity_runner.check_forward_parity() is True


def test_forward_logits_and_loss_parity_sdpa(sdpa_parity_runner):
    assert sdpa_parity_runner.check_forward_parity() is True


def test_intermediate_activations_parity(manual_parity_runner):
    assert manual_parity_runner.check_intermediate_parity() is True


def test_gradient_parity(manual_parity_runner):
    assert manual_parity_runner.check_gradient_parity() is True


def test_optimizer_step_parity(manual_parity_runner):
    assert manual_parity_runner.check_optimizer_step_parity() is True


def test_inference_path_parity(manual_parity_runner):
    assert manual_parity_runner.check_inference_parity() is True
