"""Safe evaluator for model expressions and configuration references."""

from typing import Any, Dict, Optional, Union

from neural_blueprint.ir.models import (
    ConfigRefValue,
    ExpressionOp,
    ExpressionValue,
    LiteralValue,
    ParentPropertyRefValue,
    SafeExpression,
)


def evaluate_value(
    value: Any,
    config: Optional[Dict[str, Any]] = None,
    parent_properties: Optional[Dict[str, Any]] = None,
) -> Any:
    """Recursively resolves a property value against model config and parent properties."""
    if config is None:
        config = {}
    if parent_properties is None:
        parent_properties = {}

    if isinstance(value, ConfigRefValue):
        if value.key not in config:
            raise KeyError(f"Configuration key '{value.key}' not found in model config")
        return evaluate_value(config[value.key], config, parent_properties)

    if isinstance(value, ParentPropertyRefValue):
        if value.property_name not in parent_properties:
            raise KeyError(f"Parent property '{value.property_name}' not found")
        return evaluate_value(parent_properties[value.property_name], config, parent_properties)

    if isinstance(value, LiteralValue):
        return value.value

    if isinstance(value, ExpressionValue):
        return evaluate_expression(value.expression, config, parent_properties)

    if isinstance(value, SafeExpression):
        return evaluate_expression(value, config, parent_properties)

    if isinstance(value, dict):
        if value.get("kind") == "config_ref" and "key" in value:
            key = value["key"]
            if key not in config:
                raise KeyError(f"Configuration key '{key}' not found in model config")
            return evaluate_value(config[key], config, parent_properties)
        elif value.get("kind") == "parent_property_ref" and "property_name" in value:
            prop_name = value["property_name"]
            if prop_name not in parent_properties:
                raise KeyError(f"Parent property '{prop_name}' not found")
            return evaluate_value(parent_properties[prop_name], config, parent_properties)
        elif value.get("kind") == "literal" and "value" in value:
            return value["value"]
        elif value.get("kind") == "expression" and "expression" in value:
            return evaluate_expression(value["expression"], config, parent_properties)

    return value


def evaluate_expression(
    expr: Union[SafeExpression, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    parent_properties: Optional[Dict[str, Any]] = None,
) -> Union[int, float]:
    """Evaluates a safe arithmetic expression AST."""
    if isinstance(expr, dict):
        op_str = expr.get("op")
        left_val = expr.get("left")
        right_val = expr.get("right")
    else:
        op_str = expr.op.value if isinstance(expr.op, ExpressionOp) else str(expr.op)
        left_val = expr.left
        right_val = expr.right

    left = evaluate_value(left_val, config, parent_properties)
    right = evaluate_value(right_val, config, parent_properties)

    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        l_type = type(left).__name__
        r_type = type(right).__name__
        raise TypeError(f"Expression operands must be numbers, got left={l_type}, right={r_type}")

    if op_str == "add":
        return left + right
    elif op_str == "subtract":
        return left - right
    elif op_str == "multiply":
        return left * right
    elif op_str == "integer_divide":
        if right == 0:
            raise ZeroDivisionError("Integer divide by zero in expression")
        return int(left) // int(right)
    elif op_str == "minimum":
        return min(left, right)
    elif op_str == "maximum":
        return max(left, right)
    else:
        raise ValueError(f"Unsupported expression operator: {op_str}")
