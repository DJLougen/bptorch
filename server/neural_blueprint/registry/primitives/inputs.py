"""Input, interface, and tensor generation primitive nodes."""

from typing import Any, Dict, List, Optional

import torch

from neural_blueprint.ir.evaluator import evaluate_value
from neural_blueprint.ir.models import (
    LiteralDim,
    PortDefinition,
    SymbolDim,
    TensorSpec,
    TensorType,
    UnknownDim,
)
from neural_blueprint.registry.base import (
    NodeDefinition,
    NodeValidationContext,
    RuntimeModuleSpec,
)
from neural_blueprint.registry.registry import register_node


@register_node
class TokenInputNode(NodeDefinition):
    type_id = "builtin.token_input@1"
    version = 1
    display_name = "Token IDs Input"
    category = "Inputs"
    description = "Provides discrete token IDs [B, T] to the language model."
    icon = "Key"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "token_ids"},
                "dtype": {"type": "string", "enum": ["int64", "int32"], "default": "int64"},
            },
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
        return [
            PortDefinition(
                id="output",
                display_name="Token IDs",
                direction="output",
                tensor_type=TensorType(dtype_family="integer", rank=2),
                default_shape=[
                    {"kind": "symbol", "name": "B"},
                    {"kind": "symbol", "name": "T"},
                ],
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        dtype = properties.get("dtype", "int64")
        return {
            "output": TensorSpec(
                dtype=dtype,
                shape=[SymbolDim(name="B"), SymbolDim(name="T")],
            )
        }


@register_node
class TargetInputNode(NodeDefinition):
    type_id = "builtin.target_input@1"
    version = 1
    display_name = "Targets Input"
    category = "Inputs"
    description = "Provides training target token IDs [B, T] for loss calculation."
    icon = "Target"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "targets"},
                "dtype": {"type": "string", "enum": ["int64", "int32"], "default": "int64"},
            },
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
        return [
            PortDefinition(
                id="output",
                display_name="Targets",
                direction="output",
                tensor_type=TensorType(dtype_family="integer", rank=2),
                default_shape=[
                    {"kind": "symbol", "name": "B"},
                    {"kind": "symbol", "name": "T"},
                ],
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        dtype = properties.get("dtype", "int64")
        return {
            "output": TensorSpec(
                dtype=dtype,
                shape=[SymbolDim(name="B"), SymbolDim(name="T")],
            )
        }


@register_node
class ModuleInputNode(NodeDefinition):
    type_id = "builtin.module_input@1"
    version = 1
    display_name = "Module Input"
    category = "Inputs"
    description = "Receives a tensor passed from the parent graph into a composite module."
    icon = "ArrowDownRight"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "input"},
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
        name = properties.get("name", "input")
        return [PortDefinition(id="output", display_name=name.title(), direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        name = properties.get("name", "input").lower()
        in_spec = inputs.get("output") or inputs.get("input") or inputs.get(name)
        if in_spec:
            return {"output": in_spec}

        if "tok" in name or "idx" in name or "target" in name:
            return {
                "output": TensorSpec(
                    dtype="int64",
                    shape=[SymbolDim(name="B"), SymbolDim(name="T")],
                )
            }

        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }


@register_node
class ModuleOutputNode(NodeDefinition):
    type_id = "builtin.module_output@1"
    version = 1
    display_name = "Module Output"
    category = "Loss & Outputs"
    description = "Exports an internal composite module tensor to the parent graph."
    icon = "ArrowUpRight"

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
        name = properties.get("name", "output")
        return [
            PortDefinition(
                id="input",
                display_name=name.title(),
                direction="input",
                required=True,
            )
        ]

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


@register_node
class ArangeNode(NodeDefinition):
    type_id = "builtin.arange@1"
    version = 1
    display_name = "Arange (Positions)"
    category = "Inputs"
    description = (
        "Generates position index range [0, 1, ..., T-1] for learned positional embeddings."
    )
    icon = "ListOrdered"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dtype": {"type": "string", "enum": ["int64", "int32"], "default": "int64"},
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="sequence_tensor",
                display_name="Sequence Reference",
                direction="input",
                required=False,
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="positions",
                display_name="Positions [T]",
                direction="output",
                tensor_type=TensorType(dtype_family="integer", rank=1),
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        dtype = properties.get("dtype", "int64")
        return {"positions": TensorSpec(dtype=dtype, shape=[SymbolDim(name="T")])}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        dtype_str = properties.get("dtype", "int64")
        torch_dtype = torch.int64 if dtype_str == "int64" else torch.int32

        def generate_positions(ref_tensor=None):
            if ref_tensor is not None:
                t = ref_tensor.size(1) if ref_tensor.dim() >= 2 else ref_tensor.size(0)
                device = ref_tensor.device
            else:
                t = 8
                device = torch.device("cpu")
            return torch.arange(0, t, dtype=torch_dtype, device=device)

        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: generate_positions,
        )


@register_node
class CausalMaskNode(NodeDefinition):
    type_id = "builtin.causal_mask@1"
    version = 1
    display_name = "Causal Mask"
    category = "Inputs"
    description = "Generates a lower-triangular causal attention mask ensuring autoregressive token attention."
    icon = "Shield"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "block_size": {
                    "type": ["integer", "object"],
                    "default": {"kind": "config_ref", "key": "block_size"},
                },
            },
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
        return [
            PortDefinition(
                id="mask",
                display_name="Causal Mask",
                direction="output",
                tensor_type=TensorType(dtype_family="boolean", rank=4),
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        cfg = context.model_config if context else {}
        b_size = evaluate_value(properties.get("block_size", 1024), cfg)
        b_dim = LiteralDim(value=int(b_size)) if isinstance(b_size, (int, float)) else UnknownDim()
        return {
            "mask": TensorSpec(
                dtype="bool",
                shape=[LiteralDim(value=1), LiteralDim(value=1), b_dim, b_dim],
            )
        }

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        b_size = int(evaluate_value(properties.get("block_size", 1024), cfg))

        def create_mask():
            return torch.tril(torch.ones(b_size, b_size)).view(1, 1, b_size, b_size)

        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: create_mask,
        )
