"""Loss functions and terminal language model output nodes."""

from typing import Any, Dict, List, Optional

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


@register_node
class LanguageModelHeadNode(NodeDefinition):
    type_id = "builtin.lm_head@1"
    version = 1
    display_name = "LM Head"
    category = "Layers"
    description = (
        "Projects final hidden states to vocabulary logits (tied with Token Embeddings in nanoGPT)."
    )
    icon = "Terminal"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "in_features": {
                    "type": ["integer", "object"],
                    "default": {"kind": "config_ref", "key": "n_embd"},
                },
                "out_features": {
                    "type": ["integer", "object"],
                    "default": {"kind": "config_ref", "key": "vocab_size"},
                },
                "bias": {"type": "boolean", "default": False},
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
                id="input",
                display_name="Hidden States [B, T, C]",
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
                id="logits",
                display_name="Logits [B, T, V]",
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
        cfg = context.model_config if context else {}
        v_val = evaluate_value(properties.get("out_features", 128), cfg)
        v_dim = (
            LiteralDim(value=int(v_val)) if isinstance(v_val, (int, float)) else SymbolDim(name="V")
        )

        in_spec = inputs.get("input")
        if in_spec and in_spec.shape:
            out_shape = list(in_spec.shape[:-1]) + [v_dim]
            return {"logits": TensorSpec(dtype=in_spec.dtype, shape=out_shape)}

        return {
            "logits": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), v_dim],
            )
        }

    def parameter_spec(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> ParameterSpec:
        cfg = context.model_config if context else {}
        in_f = int(evaluate_value(properties.get("in_features", 64), cfg))
        out_f = int(evaluate_value(properties.get("out_features", 128), cfg))
        bias = bool(properties.get("bias", False))

        shapes = {"weight": [out_f, in_f]}
        if bias:
            shapes["bias"] = [out_f]

        return ParameterSpec(
            trainable_count=in_f * out_f + (out_f if bias else 0),
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
        out_f = int(evaluate_value(properties.get("out_features", 128), cfg))
        bias = bool(properties.get("bias", False))

        return RuntimeModuleSpec(
            module_type="nn_module",
            factory=lambda: nn.Linear(in_f, out_f, bias=bias),
            kwargs={"in_features": in_f, "out_features": out_f, "bias": bias},
        )


@register_node
class CrossEntropyLossNode(NodeDefinition):
    type_id = "builtin.cross_entropy_loss@1"
    version = 1
    display_name = "Cross-Entropy Loss"
    category = "Loss & Outputs"
    description = "Calculates multi-class classification cross-entropy loss with ignore_index = -1."
    icon = "Flame"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ignore_index": {"type": "integer", "default": -1},
            },
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
                id="logits", display_name="Logits", direction="input", kind="data", required=True
            ),
            PortDefinition(
                id="targets", display_name="Targets", direction="input", kind="data", required=True
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
                id="loss",
                display_name="Scalar Loss",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="floating", rank=0),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"loss": TensorSpec(dtype="float32", shape=[])}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        ignore_idx = int(properties.get("ignore_index", -1))

        def compute_loss(logits=None, targets=None, **kwargs):
            logits_tensor = (
                logits if logits is not None else kwargs.get("logits", kwargs.get("input"))
            )
            target_tensor = (
                targets if targets is not None else kwargs.get("targets", kwargs.get("target"))
            )
            if logits_tensor is None or target_tensor is None:
                return None
            flat_logits = logits_tensor.view(-1, logits_tensor.size(-1))
            flat_targets = target_tensor.view(-1)
            return F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_idx)

        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: compute_loss,
        )


@register_node
class LogitsOutputNode(NodeDefinition):
    type_id = "builtin.logits_output@1"
    version = 1
    display_name = "Logits Output"
    category = "Loss & Outputs"
    description = "Marks the model's vocabulary logits prediction output."
    icon = "LogOut"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="input", display_name="Logits", direction="input", required=True)]

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
class LossOutputNode(NodeDefinition):
    type_id = "builtin.loss_output@1"
    version = 1
    display_name = "Loss Output"
    category = "Loss & Outputs"
    description = "Marks the scalar training loss output."
    icon = "Activity"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="input", display_name="Loss", direction="input", required=True)]

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
