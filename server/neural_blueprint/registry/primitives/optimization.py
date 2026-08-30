"""Optimization Primitives: AdamW, SGD, Lion, ClipGradients, ZeroGrad, OptimizerStep, Backward, AutocastScope, GradScaler."""

from typing import Any, Dict, List, Optional

from neural_blueprint.ir.models import (
    LiteralDim,
    PortDefinition,
    TensorSpec,
    TensorType,
)
from neural_blueprint.registry.base import (
    NodeDefinition,
    NodeValidationContext,
)
from neural_blueprint.registry.registry import register_node


@register_node
class AdamWOptimizerNode(NodeDefinition):
    type_id = "builtin.adamw_optimizer@1"
    version = 1
    display_name = "AdamW Optimizer"
    category = "Optimization"
    description = "Instantiates an AdamW optimizer with decoupled weight decay."
    icon = "Sliders"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "learning_rate": {"type": "number", "default": 6e-4},
                "beta1": {"type": "number", "default": 0.9},
                "beta2": {"type": "number", "default": 0.95},
                "eps": {"type": "number", "default": 1e-8},
                "weight_decay": {"type": "number", "default": 0.1},
                "fused": {"type": "boolean", "default": False},
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="params",
                display_name="Parameters",
                direction="input",
                kind="data",
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
                id="optimizer",
                display_name="Optimizer",
                direction="output",
                kind="data",
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}


@register_node
class SGDOptimizerNode(NodeDefinition):
    type_id = "builtin.sgd_optimizer@1"
    version = 1
    display_name = "SGD Optimizer"
    category = "Optimization"
    description = "Instantiates a Stochastic Gradient Descent optimizer with momentum."
    icon = "Sliders"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "learning_rate": {"type": "number", "default": 0.01},
                "momentum": {"type": "number", "default": 0.9},
                "weight_decay": {"type": "number", "default": 0.0},
                "nesterov": {"type": "boolean", "default": False},
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="params",
                display_name="Parameters",
                direction="input",
                kind="data",
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
                id="optimizer",
                display_name="Optimizer",
                direction="output",
                kind="data",
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}


@register_node
class LionOptimizerNode(NodeDefinition):
    type_id = "builtin.lion_optimizer@1"
    version = 1
    display_name = "Lion Optimizer"
    category = "Optimization"
    description = "Instantiates the EvoLved Sign Momentum (Lion) optimizer."
    icon = "Sliders"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "learning_rate": {"type": "number", "default": 1e-4},
                "beta1": {"type": "number", "default": 0.9},
                "beta2": {"type": "number", "default": 0.99},
                "weight_decay": {"type": "number", "default": 0.1},
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="params",
                display_name="Parameters",
                direction="input",
                kind="data",
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
                id="optimizer",
                display_name="Optimizer",
                direction="output",
                kind="data",
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}


@register_node
class ClipGradientsNode(NodeDefinition):
    type_id = "builtin.clip_gradients@1"
    version = 1
    display_name = "Clip Gradients"
    category = "Optimization"
    description = "Clips gradient norms of model parameters to prevent gradient explosions."
    icon = "Scissors"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_norm": {"type": "number", "minimum": 0.0, "default": 1.0},
                "norm_type": {"type": "number", "default": 2.0},
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
                id="params",
                display_name="Parameters",
                direction="input",
                kind="data",
                required=False,
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
                id="total_norm",
                display_name="Total Norm",
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
        return {"total_norm": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)])}


@register_node
class ZeroGradNode(NodeDefinition):
    type_id = "builtin.zero_grad@1"
    version = 1
    display_name = "Zero Grad"
    category = "Optimization"
    description = "Zeros the gradients of all optimized model parameters."
    icon = "RefreshCw"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "set_to_none": {"type": "boolean", "default": True},
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
                id="optimizer",
                display_name="Optimizer",
                direction="input",
                kind="data",
                required=False,
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
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}


@register_node
class OptimizerStepNode(NodeDefinition):
    type_id = "builtin.optimizer_step@1"
    version = 1
    display_name = "Optimizer Step"
    category = "Optimization"
    description = "Performs a single parameter optimization step."
    icon = "Zap"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

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
                id="optimizer",
                display_name="Optimizer",
                direction="input",
                kind="data",
                required=False,
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
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}


@register_node
class BackwardNode(NodeDefinition):
    type_id = "builtin.backward@1"
    version = 1
    display_name = "Backward"
    category = "Optimization"
    description = "Computes the gradient of current tensor graph w.r.t. graph leaves via Autograd."
    icon = "Flame"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

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
                id="loss",
                display_name="Loss",
                direction="input",
                kind="data",
                required=False,
                tensor_type=TensorType(dtype_family="floating", rank=0),
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
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}


@register_node
class AutocastScopeNode(NodeDefinition):
    type_id = "builtin.autocast_scope@1"
    version = 1
    display_name = "Autocast Scope"
    category = "Optimization"
    description = "Enables automatic mixed precision (AMP) context for enclosed forward pass."
    icon = "Shield"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "device_type": {
                    "type": "string",
                    "enum": ["cpu", "cuda", "mps"],
                    "default": "cuda",
                },
                "dtype": {
                    "type": "string",
                    "enum": ["fp16", "bf16", "fp8"],
                    "default": "bf16",
                },
                "enabled": {"type": "boolean", "default": True},
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
            )
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
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}


@register_node
class GradScalerNode(NodeDefinition):
    type_id = "builtin.grad_scaler@1"
    version = 1
    display_name = "Grad Scaler"
    category = "Optimization"
    description = "Instantiates a PyTorch GradScaler for fp16 loss scaling and dynamic growth."
    icon = "Sliders"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "init_scale": {"type": "number", "default": 65536.0},
                "growth_factor": {"type": "number", "default": 2.0},
                "backoff_factor": {"type": "number", "default": 0.5},
                "growth_interval": {"type": "integer", "default": 2000},
                "enabled": {"type": "boolean", "default": True},
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
                id="scaler",
                display_name="Scaler",
                direction="output",
                kind="data",
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}
