"""Structured diagnostic messages and error codes for graph validation."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Diagnostic(BaseModel):
    """Structured diagnostic adhering to Section 10.6 format."""

    code: str
    severity: Literal["error", "warning", "info"] = "error"
    message: str
    node_id: Optional[str] = None
    port_id: Optional[str] = None
    edge_id: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)


# Standard error codes
E_SCHEMA_INVALID = "E_SCHEMA_INVALID"
E_UNKNOWN_NODE_TYPE = "E_UNKNOWN_NODE_TYPE"
E_PORT_UNCONNECTED = "E_PORT_UNCONNECTED"
E_MULTIPLE_INPUTS = "E_MULTIPLE_INPUTS"
E_CYCLE_DETECTED = "E_CYCLE_DETECTED"
E_ORPHAN_OUTPUT = "E_ORPHAN_OUTPUT"
E_SHAPE_MISMATCH = "E_SHAPE_MISMATCH"
E_LINEAR_INPUT_DIM = "E_LINEAR_INPUT_DIM"
E_HEAD_DIVISIBILITY = "E_HEAD_DIVISIBILITY"
E_RESIDUAL_MISMATCH = "E_RESIDUAL_MISMATCH"
E_DTYPE_MISMATCH = "E_DTYPE_MISMATCH"
E_WEIGHT_TYING_MISMATCH = "E_WEIGHT_TYING_MISMATCH"
E_REPEATED_STACK_COUNT = "E_REPEATED_STACK_COUNT"
E_BLOCK_SIZE_EXCEEDED = "E_BLOCK_SIZE_EXCEEDED"
E_DUPLICATE_EDGE_ID = "E_DUPLICATE_EDGE_ID"
E_EDGE_MISSING_NODE = "E_EDGE_MISSING_NODE"
E_EDGE_MISSING_PORT = "E_EDGE_MISSING_PORT"
E_EDGE_WRONG_PORT_DIRECTION = "E_EDGE_WRONG_PORT_DIRECTION"
E_EDGE_PORT_KIND_MISMATCH = "E_EDGE_PORT_KIND_MISMATCH"
E_EDGE_SELF_CONNECTION = "E_EDGE_SELF_CONNECTION"
