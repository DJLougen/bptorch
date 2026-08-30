"""Tensor transformation and manipulation primitive nodes."""

from typing import Any, Dict, List, Optional

import torch

from neural_blueprint.ir.evaluator import evaluate_value
from neural_blueprint.ir.models import (
    LiteralDim,
    PortDefinition,
    ShapeDim,
    SymbolDim,
    TensorSpec,
    UnknownDim,
)
from neural_blueprint.registry.base import (
    NodeDefinition,
    NodeValidationContext,
    RuntimeModuleSpec,
)
from neural_blueprint.registry.registry import register_node


@register_node
class ReshapeNode(NodeDefinition):
    type_id = "builtin.reshape@1"
    version = 1
    display_name = "Reshape (View)"
    category = "Tensor Operations"
    description = "Reshapes a tensor to target dimensions."
    icon = "Maximize2"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "shape": {
                    "type": "array",
                    "items": {"type": "object"},
                    "default": [],
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
        return [PortDefinition(id="output", display_name="Reshaped", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        raw_shape = properties.get("shape", [])
        cfg = context.model_config if context else {}
        shape_dims: List[ShapeDim] = []
        for d in raw_shape:
            if isinstance(d, dict):
                k = d.get("kind")
                if k == "symbol":
                    shape_dims.append(SymbolDim(name=d.get("name", "?")))
                elif k == "config_ref":
                    val = cfg.get(d.get("key", ""))
                    if isinstance(val, (int, float)):
                        shape_dims.append(LiteralDim(value=int(val)))
                    else:
                        shape_dims.append(SymbolDim(name=str(d.get("key"))))
                elif k == "literal":
                    shape_dims.append(LiteralDim(value=int(d.get("value", 0))))
                else:
                    shape_dims.append(UnknownDim())
            else:
                shape_dims.append(UnknownDim())

        in_spec = inputs.get("input")
        dtype = in_spec.dtype if in_spec else "float32"
        return {"output": TensorSpec(dtype=dtype, shape=shape_dims)}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        raw_shape = properties.get("shape", [])

        def parse_dim(d):
            if isinstance(d, dict):
                if d.get("kind") == "literal":
                    return int(d.get("value", 0))
                elif d.get("kind") == "config_ref":
                    val = cfg.get(d.get("key", ""))
                    return int(val) if isinstance(val, (int, float)) else -1
            return -1

        target_shape = [parse_dim(d) for d in raw_shape]

        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: lambda x: x.view(*target_shape) if target_shape else x,
        )


@register_node
class SplitNode(NodeDefinition):
    type_id = "builtin.split@1"
    version = 1
    display_name = "Split"
    category = "Tensor Operations"
    description = "Splits a tensor into chunks along a specified dimension."
    icon = "Scissors"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "split_size": {"type": ["integer", "object"], "default": 64},
                "dim": {"type": "integer", "default": -1},
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
        return [
            PortDefinition(id="chunk_0", display_name="Chunk 0", direction="output"),
            PortDefinition(id="chunk_1", display_name="Chunk 1", direction="output"),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {
                "chunk_0": TensorSpec(dtype=in_spec.dtype, shape=list(in_spec.shape)),
                "chunk_1": TensorSpec(dtype=in_spec.dtype, shape=list(in_spec.shape)),
            }
        return {}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        split_size = int(evaluate_value(properties.get("split_size", 64), cfg))
        dim = int(properties.get("dim", -1))
        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: lambda x: torch.split(x, split_size, dim=dim),
        )


@register_node
class TransposeNode(NodeDefinition):
    type_id = "builtin.transpose@1"
    version = 1
    display_name = "Transpose"
    category = "Tensor Operations"
    description = "Swaps two dimensions in a tensor: e.g. (B, T, NH, HD) -> (B, NH, T, HD)."
    icon = "RefreshCw"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dim0": {"type": "integer", "default": 1},
                "dim1": {"type": "integer", "default": 2},
            },
            "required": ["dim0", "dim1"],
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
        return [PortDefinition(id="output", display_name="Transposed", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        d0 = int(properties.get("dim0", 1))
        d1 = int(properties.get("dim1", 2))
        in_spec = inputs.get("input")
        if in_spec and in_spec.shape:
            shape = list(in_spec.shape)
            if d0 < len(shape) and d1 < len(shape):
                shape[d0], shape[d1] = shape[d1], shape[d0]
                return {"output": TensorSpec(dtype=in_spec.dtype, shape=shape)}
        return {}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        d0 = int(properties.get("dim0", 1))
        d1 = int(properties.get("dim1", 2))
        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: lambda x: x.transpose(d0, d1),
        )


@register_node
class ContiguousNode(NodeDefinition):
    type_id = "builtin.contiguous@1"
    version = 1
    display_name = "Contiguous"
    category = "Tensor Operations"
    description = "Ensures tensor memory layout is contiguous before reshaping."
    icon = "Box"

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
        return [PortDefinition(id="output", display_name="Contiguous", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: lambda x: x.contiguous(),
        )


@register_node
class MatMulNode(NodeDefinition):
    type_id = "builtin.matmul@1"
    version = 1
    display_name = "Matrix Multiply"
    category = "Tensor Operations"
    description = "Computes matrix multiplication: a @ b."
    icon = "X"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "transpose_b": {"type": "boolean", "default": False},
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(id="a", display_name="Matrix A", direction="input", required=True),
            PortDefinition(id="b", display_name="Matrix B", direction="input", required=True),
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Product", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        spec_a = inputs.get("a")
        spec_b = inputs.get("b")
        if spec_a and spec_b and len(spec_a.shape) >= 2 and len(spec_b.shape) >= 2:
            out_shape = list(spec_a.shape[:-1]) + [spec_b.shape[-1]]
            return {"output": TensorSpec(dtype=spec_a.dtype, shape=out_shape)}
        return {}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        transpose_b = bool(properties.get("transpose_b", False))
        if transpose_b:
            return RuntimeModuleSpec(
                module_type="functional",
                factory=lambda: lambda a, b: a @ b.transpose(-2, -1),
            )
        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: lambda a, b: a @ b,
        )


@register_node
class ScaleNode(NodeDefinition):
    type_id = "builtin.scale@1"
    version = 1
    display_name = "Scale"
    category = "Tensor Operations"
    description = "Scales a tensor by a constant multiplier (e.g. 1 / sqrt(head_dim))."
    icon = "Divide"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scale": {"type": ["number", "object"], "default": 1.0},
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
        return [PortDefinition(id="output", display_name="Scaled", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        scale_val = float(evaluate_value(properties.get("scale", 1.0), cfg))
        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: lambda x: x * scale_val,
        )


@register_node
class MaskedFillNode(NodeDefinition):
    type_id = "builtin.masked_fill@1"
    version = 1
    display_name = "Masked Fill"
    category = "Tensor Operations"
    description = "Replaces tensor elements with a fill value where mask condition is false."
    icon = "Filter"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "number", "default": float("-inf")},
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(id="input", display_name="Scores", direction="input", required=True),
            PortDefinition(id="mask", display_name="Causal Mask", direction="input", required=True),
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Masked Scores", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        fill_val = float(properties.get("value", float("-inf")))

        def apply_masked_fill(scores, mask):
            t = scores.size(-1)
            sub_mask = mask[:, :, :t, :t] if mask.dim() == 4 else mask
            return scores.masked_fill(sub_mask == 0, fill_val)

        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: apply_masked_fill,
        )


@register_node
class SoftmaxNode(NodeDefinition):
    type_id = "builtin.softmax@1"
    version = 1
    display_name = "Softmax"
    category = "Tensor Operations"
    description = "Normalizes logits to probability distribution along specified dimension."
    icon = "PieChart"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dim": {"type": "integer", "default": -1},
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
        return [PortDefinition(id="output", display_name="Probabilities", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        dim = int(properties.get("dim", -1))
        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: lambda x: torch.softmax(x, dim=dim),
        )


@register_node
class SelectTimeStepNode(NodeDefinition):
    type_id = "builtin.select_time_step@1"
    version = 1
    display_name = "Select Time Step"
    category = "Tensor Operations"
    description = "Selects a specific sequence index (e.g. index -1 for inference LM Head optimization [B, 1, C])."
    icon = "Crosshair"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "default": -1},
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="input", display_name="Hidden States [B, T, C]", direction="input", required=True
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(id="output", display_name="Selected Step [B, 1, C]", direction="output")
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec and in_spec.shape:
            out_shape = [in_spec.shape[0], LiteralDim(value=1)] + in_spec.shape[2:]
            return {"output": TensorSpec(dtype=in_spec.dtype, shape=out_shape)}
        return {}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        idx = int(properties.get("index", -1))
        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: lambda x: x[:, [idx], :],
        )


@register_node
class FlattenDimensionsNode(NodeDefinition):
    type_id = "builtin.flatten@1"
    version = 1
    display_name = "Flatten Dimensions"
    category = "Tensor Operations"
    description = "Flattens leading dimensions into a batch dimension (e.g. [B, T, V] -> [B*T, V])."
    icon = "Minimize2"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "start_dim": {"type": "integer", "default": 0},
                "end_dim": {"type": "integer", "default": -2},
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
        return [PortDefinition(id="output", display_name="Flattened", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec and in_spec.shape:
            last_dim = in_spec.shape[-1]
            return {
                "output": TensorSpec(dtype=in_spec.dtype, shape=[SymbolDim(name="B*T"), last_dim])
            }
        return {}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: lambda x: x.view(-1, x.size(-1)) if x.dim() >= 2 else x.view(-1),
        )
