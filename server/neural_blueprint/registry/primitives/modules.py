"""Parameterized PyTorch module node definitions for nanoGPT."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from neural_blueprint.ir.evaluator import evaluate_value
from neural_blueprint.ir.models import (
    LiteralDim,
    PortDefinition,
    SymbolDim,
    TensorSpec,
    TensorType,
)
from neural_blueprint.registry.base import (
    NodeDefinition,
    NodeValidationContext,
    ParameterSpec,
    RuntimeModuleSpec,
)
from neural_blueprint.registry.registry import register_node


class NanoGPTLayerNorm(nn.Module):
    """LayerNorm matching karpathy/nanoGPT with optional learnable bias."""

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


@register_node
class EmbeddingNode(NodeDefinition):
    type_id = "builtin.embedding@1"
    version = 1
    display_name = "Embedding"
    category = "Layers"
    description = "Maps discrete indices to dense learned continuous representation vectors."
    icon = "Layers"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "num_embeddings": {
                    "type": ["integer", "object"],
                    "description": "Vocabulary size or block size",
                    "default": {"kind": "config_ref", "key": "vocab_size"},
                },
                "embedding_dim": {
                    "type": ["integer", "object"],
                    "description": "Embedding channel dimensionality",
                    "default": {"kind": "config_ref", "key": "n_embd"},
                },
            },
            "required": ["num_embeddings", "embedding_dim"],
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
                display_name="Indices",
                direction="input",
                kind="data",
                required=True,
                tensor_type=TensorType(dtype_family="integer"),
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
                display_name="Embeddings",
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
        emb_dim = evaluate_value(properties.get("embedding_dim", 64), cfg)
        c_dim = (
            LiteralDim(value=int(emb_dim))
            if isinstance(emb_dim, (int, float))
            else SymbolDim(name="C")
        )

        in_spec = inputs.get("input")
        if in_spec and in_spec.shape:
            # If input rank <= 2 (e.g. [B, T] or [T]), append c_dim -> [B, T, C] or [T, C]
            if len(in_spec.shape) <= 2:
                out_shape = list(in_spec.shape) + [c_dim]
            else:
                out_shape = list(in_spec.shape[:-1]) + [c_dim]
        else:
            out_shape = [SymbolDim(name="B"), SymbolDim(name="T"), c_dim]

        return {"output": TensorSpec(dtype="float32", shape=out_shape)}

    def parameter_spec(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> ParameterSpec:
        cfg = context.model_config if context else {}
        num_emb = int(evaluate_value(properties.get("num_embeddings", 128), cfg))
        emb_dim = int(evaluate_value(properties.get("embedding_dim", 64), cfg))

        return ParameterSpec(
            trainable_count=num_emb * emb_dim,
            frozen_count=0,
            parameter_shapes={"weight": [num_emb, emb_dim]},
        )

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        num_emb = int(evaluate_value(properties.get("num_embeddings", 128), cfg))
        emb_dim = int(evaluate_value(properties.get("embedding_dim", 64), cfg))

        return RuntimeModuleSpec(
            module_type="nn_module",
            factory=lambda: nn.Embedding(num_emb, emb_dim),
            kwargs={"num_embeddings": num_emb, "embedding_dim": emb_dim},
        )


@register_node
class LayerNormNode(NodeDefinition):
    type_id = "builtin.layernorm@1"
    version = 1
    display_name = "LayerNorm"
    category = "Layers"
    description = (
        "Normalizes activations across channels with optional learnable affine gain and bias."
    )
    icon = "Sliders"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "normalized_shape": {
                    "type": ["integer", "object"],
                    "default": {"kind": "config_ref", "key": "n_embd"},
                },
                "bias": {
                    "type": ["boolean", "object"],
                    "default": {"kind": "config_ref", "key": "bias"},
                },
                "eps": {"type": "number", "default": 1e-5},
            },
            "required": ["normalized_shape"],
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="input",
                display_name="Input",
                direction="input",
                required=True,
                tensor_type=TensorType(dtype_family="floating"),
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="output",
                display_name="Normalized",
                direction="output",
                tensor_type=TensorType(dtype_family="floating"),
            )
        ]

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
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }

    def parameter_spec(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> ParameterSpec:
        cfg = context.model_config if context else {}
        ndim = int(evaluate_value(properties.get("normalized_shape", 64), cfg))
        bias = bool(evaluate_value(properties.get("bias", True), cfg))

        shapes = {"weight": [ndim]}
        if bias:
            shapes["bias"] = [ndim]

        return ParameterSpec(
            trainable_count=ndim * (2 if bias else 1),
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
        ndim = int(evaluate_value(properties.get("normalized_shape", 64), cfg))
        bias = bool(evaluate_value(properties.get("bias", True), cfg))
        eps = float(properties.get("eps", 1e-5))

        return RuntimeModuleSpec(
            module_type="nn_module",
            factory=lambda: NanoGPTLayerNorm(ndim, bias=bias, eps=eps),
            kwargs={"ndim": ndim, "bias": bias, "eps": eps},
        )


@register_node
class DropoutNode(NodeDefinition):
    type_id = "builtin.dropout@1"
    version = 1
    display_name = "Dropout"
    category = "Layers"
    description = "Randomly zeroes elements with probability p during training for regularization."
    icon = "Percent"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dropout": {
                    "type": ["number", "object"],
                    "default": {"kind": "config_ref", "key": "dropout"},
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
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        p = float(evaluate_value(properties.get("dropout", 0.0), cfg))

        return RuntimeModuleSpec(
            module_type="nn_module",
            factory=lambda: nn.Dropout(p),
            kwargs={"p": p},
        )
