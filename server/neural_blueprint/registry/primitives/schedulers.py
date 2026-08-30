"""Learning Rate Scheduler Primitives: CosineAnnealingLR, LinearWarmupScheduler, ReduceLROnPlateau, StepLR."""

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
class CosineAnnealingLRNode(NodeDefinition):
    type_id = "builtin.cosine_annealing_lr@1"
    version = 1
    display_name = "Cosine Decay with Warmup"
    category = "LR Schedulers"
    description = "Schedules learning rate using linear warmup followed by cosine annealing decay."
    icon = "Activity"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "warmup_steps": {"type": "integer", "default": 20},
                "total_steps": {"type": "integer", "default": 1000},
                "eta_min": {"type": "number", "default": 1e-5},
                "min_lr_ratio": {"type": "number", "default": 0.1},
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
            ),
            PortDefinition(
                id="current_lr",
                display_name="Current LR",
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
        return {"current_lr": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)])}


@register_node
class LinearWarmupSchedulerNode(NodeDefinition):
    type_id = "builtin.linear_warmup_scheduler@1"
    version = 1
    display_name = "Linear Decay"
    category = "LR Schedulers"
    description = "Schedules learning rate using linear warmup followed by linear decay to zero."
    icon = "Activity"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "warmup_steps": {"type": "integer", "default": 50},
                "total_steps": {"type": "integer", "default": 1000},
                "min_lr_ratio": {"type": "number", "default": 0.0},
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
            ),
            PortDefinition(
                id="current_lr",
                display_name="Current LR",
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
        return {"current_lr": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)])}


@register_node
class ReduceLROnPlateauNode(NodeDefinition):
    type_id = "builtin.reduce_lr_on_plateau@1"
    version = 1
    display_name = "Reduce LR On Plateau"
    category = "LR Schedulers"
    description = "Reduces learning rate when a metric (e.g. val_loss) has stopped improving."
    icon = "Activity"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["min", "max"], "default": "min"},
                "factor": {"type": "number", "default": 0.5},
                "patience": {"type": "integer", "default": 3},
                "min_lr": {"type": "number", "default": 1e-6},
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
                id="val_loss",
                display_name="Val Loss",
                direction="input",
                kind="data",
                required=True,
                tensor_type=TensorType(dtype_family="floating"),
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
            ),
            PortDefinition(
                id="current_lr",
                display_name="Current LR",
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
        return {"current_lr": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)])}


@register_node
class StepLRNode(NodeDefinition):
    type_id = "builtin.step_lr@1"
    version = 1
    display_name = "Step LR"
    category = "LR Schedulers"
    description = "Decays learning rate by gamma every step_size epochs."
    icon = "Activity"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "step_size": {"type": "integer", "default": 30},
                "gamma": {"type": "number", "default": 0.1},
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
            ),
            PortDefinition(
                id="current_lr",
                display_name="Current LR",
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
        return {"current_lr": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)])}
