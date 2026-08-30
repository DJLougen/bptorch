"""Canonical Model Intermediate Representation (IR) data models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# --- Expression AST ---


class ExpressionOp(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    INTEGER_DIVIDE = "integer_divide"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"


class ConfigRefValue(BaseModel):
    kind: Literal["config_ref"] = "config_ref"
    key: str


class ParentPropertyRefValue(BaseModel):
    kind: Literal["parent_property_ref"] = "parent_property_ref"
    property_name: str


class LiteralValue(BaseModel):
    kind: Literal["literal"] = "literal"
    value: Any


class SafeExpression(BaseModel):
    op: ExpressionOp
    left: Union[int, float, str, ConfigRefValue, ParentPropertyRefValue, SafeExpression]
    right: Union[int, float, str, ConfigRefValue, ParentPropertyRefValue, SafeExpression]


class ExpressionValue(BaseModel):
    kind: Literal["expression"] = "expression"
    expression: SafeExpression


PropertyValue = Union[ConfigRefValue, ParentPropertyRefValue, LiteralValue, ExpressionValue, Any]


# --- Symbolic Shape and Tensor Types ---


class SymbolDim(BaseModel):
    kind: Literal["symbol"] = "symbol"
    name: str  # e.g., "B", "T", "C", "V", "NH", "HD"


class ConfigRefDim(BaseModel):
    kind: Literal["config_ref"] = "config_ref"
    key: str  # e.g., "n_embd", "block_size", "vocab_size"


class LiteralDim(BaseModel):
    kind: Literal["literal"] = "literal"
    value: int


class UnknownDim(BaseModel):
    kind: Literal["unknown"] = "unknown"


ShapeDim = Union[SymbolDim, ConfigRefDim, LiteralDim, UnknownDim]


class TensorType(BaseModel):
    dtype_family: Literal["floating", "integer", "boolean", "any"] = "floating"
    rank: Optional[int] = None


class TensorSpec(BaseModel):
    dtype: str = "float32"  # "float32", "float16", "bfloat16", "int64", "int32", "bool"
    shape: List[ShapeDim] = Field(default_factory=list)
    device: str = "runtime"


# --- Ports and Edges ---


PortKind = Literal["exec", "data"]


class PortDefinition(BaseModel):
    id: str
    display_name: str
    direction: Literal["input", "output"]
    kind: PortKind = "data"
    required: bool = True
    multiplicity: Literal["single", "multiple"] = "single"
    tensor_type: Optional[TensorType] = None
    default_shape: Optional[List[ShapeDim]] = None
    description: Optional[str] = None


class PortReference(BaseModel):
    node_id: str
    port_id: str


class Edge(BaseModel):
    id: str
    source: PortReference
    target: PortReference


# --- Node Instance ---


class NodeMetadata(BaseModel):
    breakpoint: bool = False
    disabled: bool = False
    notes: Optional[str] = None


class NodeInstance(BaseModel):
    id: str
    definition_id: str  # e.g. "builtin.linear@1", "builtin.gpt_block@1"
    display_name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)


# --- Graphs and Interfaces ---


class VariableDefinition(BaseModel):
    id: str
    name: str
    type: str = "float"  # "float", "int", "str", "bool", "tensor"
    default_value: Any = None


GraphKind = Literal[
    "root", "module", "repeat", "architecture", "training_event", "function", "macro"
]


class GraphInterface(BaseModel):
    inputs: List[PortDefinition] = Field(default_factory=list)
    outputs: List[PortDefinition] = Field(default_factory=list)


class GraphDefinition(BaseModel):
    id: str
    name: str
    kind: GraphKind = "module"
    interface: GraphInterface = Field(default_factory=GraphInterface)
    variables: List[VariableDefinition] = Field(default_factory=list)
    nodes: List[NodeInstance] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    # For repeated modules
    repeat_count: Optional[Union[int, ConfigRefValue]] = None
    target_graph_id: Optional[str] = None  # graph_id of the module definition being repeated


# --- Weight Bindings ---


class WeightBindingEndpoint(BaseModel):
    node_id: str
    parameter: str = "weight"  # e.g. "weight", "bias"


class WeightBinding(BaseModel):
    source: WeightBindingEndpoint
    target: WeightBindingEndpoint
    mode: Literal["share", "copy"] = "share"


# --- Model Definition ---


class TrainingConfig(BaseModel):
    device: str = "cpu"  # "cpu", "mps", "cuda"
    precision: Literal["fp32", "fp16", "bf16", "fp8"] = "fp32"
    ddp_enabled: bool = False
    seed: int = 1337
    max_epochs: int = 10
    max_steps: Optional[int] = 1000
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    batch_size: int = 8
    checkpoint_interval: int = 100
    eval_interval: int = 50


class ModelDefinition(BaseModel):
    root_graph_id: str = "graph_root"
    config: Dict[str, Any] = Field(default_factory=dict)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    graphs: Dict[str, GraphDefinition] = Field(default_factory=dict)
    weight_bindings: List[WeightBinding] = Field(default_factory=list)


# --- UI State ---


class NodePosition(BaseModel):
    x: float
    y: float


class Viewport(BaseModel):
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0


class UIState(BaseModel):
    graph_viewports: Dict[str, Viewport] = Field(default_factory=dict)
    node_positions: Dict[str, Dict[str, NodePosition]] = Field(
        default_factory=dict
    )  # graph_id -> node_id -> position
    open_graph_id: str = "graph_root"


# --- Project Container ---


class ProjectMetadata(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str


class Project(BaseModel):
    schema_version: int = 1
    project: ProjectMetadata
    model: ModelDefinition
    ui: UIState = Field(default_factory=UIState)
