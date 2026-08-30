"""Unit tests for safe expression AST evaluator."""

import pytest
from neural_blueprint.ir.evaluator import evaluate_expression, evaluate_value
from neural_blueprint.ir.models import (
    ConfigRefValue,
    ExpressionOp,
    LiteralValue,
    ParentPropertyRefValue,
    SafeExpression,
)


def test_literal_evaluation():
    assert evaluate_value(42) == 42
    assert evaluate_value("hello") == "hello"
    assert evaluate_value(LiteralValue(value=100)) == 100


def test_config_ref_evaluation():
    config = {"n_embd": 64, "n_head": 4}
    ref = ConfigRefValue(key="n_embd")
    assert evaluate_value(ref, config=config) == 64

    with pytest.raises(KeyError):
        evaluate_value(ConfigRefValue(key="non_existent"), config=config)


def test_parent_property_ref_evaluation():
    parent_props = {"features": 128}
    ref = ParentPropertyRefValue(property_name="features")
    assert evaluate_value(ref, parent_properties=parent_props) == 128

    with pytest.raises(KeyError):
        evaluate_value(
            ParentPropertyRefValue(property_name="missing"), parent_properties=parent_props
        )


def test_safe_expression_operations():
    config = {"n_embd": 64, "n_head": 4}

    # head_dim = n_embd // n_head = 16
    expr_head_dim = SafeExpression(
        op=ExpressionOp.INTEGER_DIVIDE,
        left=ConfigRefValue(key="n_embd"),
        right=ConfigRefValue(key="n_head"),
    )
    assert evaluate_expression(expr_head_dim, config=config) == 16

    # mlp_hidden = 4 * n_embd = 256
    expr_mlp = SafeExpression(
        op=ExpressionOp.MULTIPLY,
        left=4,
        right=ConfigRefValue(key="n_embd"),
    )
    assert evaluate_expression(expr_mlp, config=config) == 256

    # nested: (n_embd * 3) + 10 = 202
    expr_nested = SafeExpression(
        op=ExpressionOp.ADD,
        left=SafeExpression(
            op=ExpressionOp.MULTIPLY,
            left=3,
            right=ConfigRefValue(key="n_embd"),
        ),
        right=10,
    )
    assert evaluate_expression(expr_nested, config=config) == 202


def test_division_by_zero():
    expr = SafeExpression(
        op=ExpressionOp.INTEGER_DIVIDE,
        left=10,
        right=0,
    )
    with pytest.raises(ZeroDivisionError):
        evaluate_expression(expr)
