"""Initial basic primitive node implementations."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from neural_blueprint.ir.evaluator import evaluate_value
from neural_blueprint.ir.models import (
    ConfigRefDim,
    LiteralDim,
    PortDefinition,
    ShapeDim,
    SymbolDim,
    TensorSpec,
    TensorType,
    UnknownDim,
)
from neural_blueprint.registry.base import (
    NodeDefinition,
    NodeValidationContext,
    ParameterSpec,
    RuntimeModuleSpec,
)
from neural_blueprint.registry.registry import register_node


@register_node
class TensorInputNode(NodeDefinition):
    type_id = "builtin.tensor_input@1"
    version = 1
    display_name = "Tensor Input"
    category = "Inputs"
    description = "Provides an external input tensor to the model graph."
    icon = "LogIn"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "input"},
                "dtype": {
                    "type": "string",
                    "enum": ["float32", "float16", "bfloat16", "int64", "int32", "bool"],
                    "default": "float32",
                },
                "shape": {
                    "type": "array",
                    "items": {"type": "object"},
                    "default": [
                        {"kind": "symbol", "name": "B"},
                        {"kind": "symbol", "name": "T"},
                        {"kind": "config_ref", "key": "n_embd"},
                    ],
                },
            },
            "required": ["name"],
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return []

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        dtype = properties.get("dtype", "float32")
        shape_data = properties.get("shape", [])
        return [
            PortDefinition(
                id="output",
                display_name="Output",
                direction="output",
                tensor_type=TensorType(dtype_family="floating" if "float" in dtype else "integer"),
                default_shape=shape_data if shape_data else None,
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        cfg = context.model_config if context else {}
        dtype = properties.get("dtype", "float32")
        raw_shape = properties.get("shape")

        if not raw_shape:
            # Default shape is [B, T, n_embd]
            raw_shape = [
                {"kind": "symbol", "name": "B"},
                {"kind": "symbol", "name": "T"},
                {"kind": "config_ref", "key": "n_embd"},
            ]

        shape_dims: List[ShapeDim] = []
        for dim in raw_shape:
            if isinstance(dim, dict):
                k = dim.get("kind")
                if k == "symbol":
                    shape_dims.append(SymbolDim(name=dim.get("name", "?")))
                elif k == "config_ref":
                    key = str(dim.get("key"))
                    val = cfg.get(key)
                    if isinstance(val, (int, float)):
                        shape_dims.append(LiteralDim(value=int(val)))
                    else:
                        shape_dims.append(ConfigRefDim(key=key))
                elif k == "literal":
                    shape_dims.append(LiteralDim(value=int(dim.get("value", 0))))
                else:
                    shape_dims.append(UnknownDim())
            elif isinstance(dim, ConfigRefDim):
                val = cfg.get(dim.key)
                if isinstance(val, (int, float)):
                    shape_dims.append(LiteralDim(value=int(val)))
                else:
                    shape_dims.append(dim)
            elif isinstance(dim, ShapeDim):
                shape_dims.append(dim)
            else:
                shape_dims.append(UnknownDim())

        return {"output": TensorSpec(dtype=dtype, shape=shape_dims)}


@register_node
class LinearNode(NodeDefinition):
    type_id = "builtin.linear@1"
    version = 1
    display_name = "Linear"
    category = "Layers"
    description = "Applies an affine linear transformation: y = xA^T + b."
    icon = "GitCommit"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "in_features": {
                    "type": ["integer", "object"],
                    "description": "Input dimension or config reference",
                    "default": 64,
                },
                "out_features": {
                    "type": ["integer", "object"],
                    "description": "Output dimension or config reference",
                    "default": 64,
                },
                "bias": {
                    "type": ["boolean", "object"],
                    "description": "Whether to include a learnable additive bias",
                    "default": True,
                },
            },
            "required": ["in_features", "out_features"],
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="exec_in",
                display_name="Exec",
                direction="input",
                kind="exec",
                required=False,
            ),
            PortDefinition(
                id="input",
                display_name="Input",
                direction="input",
                kind="data",
                required=True,
                tensor_type=TensorType(dtype_family="floating"),
            ),
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="exec_out",
                display_name="Exec",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="output",
                display_name="Output",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="floating"),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        cfg = context.model_config if context else {}
        out_f = evaluate_value(properties.get("out_features", 64), cfg)
        out_dim = LiteralDim(value=int(out_f)) if isinstance(out_f, (int, float)) else UnknownDim()

        in_spec = inputs.get("input")
        if in_spec and in_spec.shape:
            out_shape = list(in_spec.shape[:-1]) + [out_dim]
            dtype = in_spec.dtype
        else:
            out_shape = [SymbolDim(name="B"), SymbolDim(name="T"), out_dim]
            dtype = "float32"

        return {"output": TensorSpec(dtype=dtype, shape=out_shape)}

    def parameter_spec(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> ParameterSpec:
        cfg = context.model_config if context else {}
        in_f = int(evaluate_value(properties.get("in_features", 64), cfg))
        out_f = int(evaluate_value(properties.get("out_features", 64), cfg))
        bias = bool(evaluate_value(properties.get("bias", True), cfg))

        weight_count = in_f * out_f
        bias_count = out_f if bias else 0

        shapes = {"weight": [out_f, in_f]}
        if bias:
            shapes["bias"] = [out_f]

        return ParameterSpec(
            trainable_count=weight_count + bias_count,
            frozen_count=0,
            parameter_shapes=shapes,
            has_bias=bias,
        )

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        in_f = int(evaluate_value(properties.get("in_features", 64), cfg))
        out_f = int(evaluate_value(properties.get("out_features", 64), cfg))
        bias = bool(evaluate_value(properties.get("bias", True), cfg))

        return RuntimeModuleSpec(
            module_type="nn_module",
            factory=lambda: nn.Linear(in_f, out_f, bias=bias),
            kwargs={"in_features": in_f, "out_features": out_f, "bias": bias},
        )


@register_node
class GELUNode(NodeDefinition):
    type_id = "builtin.gelu@1"
    version = 1
    display_name = "GELU"
    category = "Layers"
    description = "Gaussian Error Linear Unit activation function."
    icon = "Activity"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "approximate": {
                    "type": "string",
                    "enum": ["none", "tanh"],
                    "default": "none",
                },
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="input", display_name="Input", direction="input", required=True)]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": TensorSpec(dtype=in_spec.dtype, shape=list(in_spec.shape))}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[
                    SymbolDim(name="B"),
                    SymbolDim(name="T"),
                    SymbolDim(name="C"),
                ],
            )
        }

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        approx = properties.get("approximate", "none")
        return RuntimeModuleSpec(
            module_type="nn_module",
            factory=lambda: nn.GELU(approximate=approx),
            kwargs={"approximate": approx},
        )


@register_node
class SiLUNode(NodeDefinition):
    type_id = "builtin.silu@1"
    version = 1
    display_name = "SiLU"
    category = "Layers"
    description = "Sigmoid Linear Unit (Swish) activation function."
    icon = "Activity"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="input", display_name="Input", direction="input", required=True)]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": TensorSpec(dtype=in_spec.dtype, shape=list(in_spec.shape))}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[
                    SymbolDim(name="B"),
                    SymbolDim(name="T"),
                    SymbolDim(name="C"),
                ],
            )
        }

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        return RuntimeModuleSpec(
            module_type="nn_module",
            factory=lambda: nn.SiLU(),
        )



@register_node
class ReLUNode(NodeDefinition):
    type_id = "builtin.relu@1"
    version = 1
    display_name = "ReLU"
    category = "Layers"
    description = "Rectified Linear Unit activation function."
    icon = "Activity"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="input", display_name="Input", direction="input", required=True)]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": TensorSpec(dtype=in_spec.dtype, shape=list(in_spec.shape))}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[
                    SymbolDim(name="B"),
                    SymbolDim(name="T"),
                    SymbolDim(name="C"),
                ],
            )
        }

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        return RuntimeModuleSpec(
            module_type="nn_module",
            factory=lambda: nn.ReLU(),
        )


@register_node
class AddNode(NodeDefinition):
    type_id = "builtin.add@1"
    version = 1
    display_name = "Add (Residual)"
    category = "Tensor Operations"
    description = "Element-wise addition for tensor combination and residual connections."
    icon = "Plus"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(id="a", display_name="Input A", direction="input", required=True),
            PortDefinition(id="b", display_name="Input B", direction="input", required=True),
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        spec_a = inputs.get("a")
        spec_b = inputs.get("b")
        if spec_a:
            return {"output": TensorSpec(dtype=spec_a.dtype, shape=list(spec_a.shape))}
        elif spec_b:
            return {"output": TensorSpec(dtype=spec_b.dtype, shape=list(spec_b.shape))}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[
                    SymbolDim(name="B"),
                    SymbolDim(name="T"),
                    SymbolDim(name="C"),
                ],
            )
        }

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: lambda a, b: torch.add(a, b),
        )


@register_node
class GraphOutputNode(NodeDefinition):
    type_id = "builtin.graph_output@1"
    version = 1
    display_name = "Graph Output"
    category = "Loss & Outputs"
    description = "Marks a terminal output of the model graph."
    icon = "LogOut"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "output"},
            },
            "required": ["name"],
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="input", display_name="Input", direction="input", required=True)]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return []

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}
