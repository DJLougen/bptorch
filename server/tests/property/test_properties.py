"""Property-based tests using Hypothesis for canonical IR and expression evaluation."""

import random

from hypothesis import given, settings
from hypothesis import strategies as st
from neural_blueprint.ir.evaluator import evaluate_expression
from neural_blueprint.ir.models import ExpressionOp, SafeExpression
from neural_blueprint.ir.serialization import deserialize_project, serialize_project
from tests.unit.test_serialization import create_sample_project


@given(
    a=st.integers(min_value=-10000, max_value=10000),
    b=st.integers(min_value=-10000, max_value=10000),
)
def test_expression_arithmetic_properties(a: int, b: int):
    # Addition commutative
    expr1 = SafeExpression(op=ExpressionOp.ADD, left=a, right=b)
    expr2 = SafeExpression(op=ExpressionOp.ADD, left=b, right=a)
    assert evaluate_expression(expr1) == evaluate_expression(expr2)

    # Multiplication commutative
    expr_m1 = SafeExpression(op=ExpressionOp.MULTIPLY, left=a, right=b)
    expr_m2 = SafeExpression(op=ExpressionOp.MULTIPLY, left=b, right=a)
    assert evaluate_expression(expr_m1) == evaluate_expression(expr_m2)

    # Min / Max bounds
    expr_min = SafeExpression(op=ExpressionOp.MINIMUM, left=a, right=b)
    expr_max = SafeExpression(op=ExpressionOp.MAXIMUM, left=a, right=b)
    assert evaluate_expression(expr_min) <= evaluate_expression(expr_max)


@settings(max_examples=25)
@given(
    n_embd=st.sampled_from([32, 64, 128, 256, 512]),
    n_hidden=st.sampled_from([64, 128, 256, 512, 1024]),
)
def test_project_serialization_round_trip_property(n_embd: int, n_hidden: int):
    project = create_sample_project()
    project.model.config["n_embd"] = n_embd
    project.model.config["n_hidden"] = n_hidden

    # Shuffle node order to test node list order tolerance
    graph = project.model.graphs["graph_mlp"]
    shuffled_nodes = list(graph.nodes)
    random.shuffle(shuffled_nodes)
    graph.nodes = shuffled_nodes

    serialized = serialize_project(project)
    deserialized = deserialize_project(serialized)

    assert deserialized.model.config["n_embd"] == n_embd
    assert deserialized.model.config["n_hidden"] == n_hidden
    assert len(deserialized.model.graphs["graph_mlp"].nodes) == len(shuffled_nodes)
