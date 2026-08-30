"""Metric & Evaluation Primitives: LossAggregator, AccuracyMetric, PerplexityMetric, MetricLogger, ValidationLoop."""

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
class LossAggregatorNode(NodeDefinition):
    type_id = "builtin.loss_aggregator@1"
    version = 1
    display_name = "Accumulate Metrics"
    category = "Metrics & Evaluation"
    description = "Computes running moving average of scalar batch losses."
    icon = "PieChart"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "window_size": {"type": "integer", "default": 20},
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="loss",
                display_name="Loss",
                direction="input",
                kind="data",
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
                id="avg_loss",
                display_name="Average Loss",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="floating", rank=0),
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"avg_loss": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)])}


@register_node
class AccuracyMetricNode(NodeDefinition):
    type_id = "builtin.accuracy_metric@1"
    version = 1
    display_name = "Accuracy Metric"
    category = "Metrics & Evaluation"
    description = "Calculates top-1 or top-k categorical prediction accuracy."
    icon = "Target"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "top_k": {"type": "integer", "default": 1},
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="logits",
                display_name="Logits",
                direction="input",
                kind="data",
                required=True,
                tensor_type=TensorType(dtype_family="floating"),
            ),
            PortDefinition(
                id="targets",
                display_name="Targets",
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
                id="accuracy",
                display_name="Accuracy",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="floating", rank=0),
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"accuracy": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)])}


@register_node
class PerplexityMetricNode(NodeDefinition):
    type_id = "builtin.perplexity_metric@1"
    version = 1
    display_name = "Perplexity Metric"
    category = "Metrics & Evaluation"
    description = "Calculates language model perplexity as exp(cross_entropy_loss)."
    icon = "Activity"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="loss",
                display_name="Loss",
                direction="input",
                kind="data",
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
                id="perplexity",
                display_name="Perplexity",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="floating", rank=0),
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"perplexity": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)])}


@register_node
class MetricLoggerNode(NodeDefinition):
    type_id = "builtin.metric_logger@1"
    version = 1
    display_name = "Metric Logger"
    category = "Metrics & Evaluation"
    description = (
        "Logs scalars, learning rates, and losses to console, live graphs, and TensorBoard."
    )
    icon = "Terminal"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "log_interval": {"type": "integer", "default": 10},
                "tag": {"type": "string", "default": "train"},
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
                id="step",
                display_name="Step",
                direction="input",
                kind="data",
                required=False,
                tensor_type=TensorType(dtype_family="integer"),
            ),
            PortDefinition(
                id="loss",
                display_name="Loss",
                direction="input",
                kind="data",
                required=False,
                tensor_type=TensorType(dtype_family="floating"),
            ),
            PortDefinition(
                id="metrics",
                display_name="Metrics",
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
class ValidationLoopNode(NodeDefinition):
    type_id = "builtin.validation_loop@1"
    version = 1
    display_name = "Validation Loop"
    category = "Metrics & Evaluation"
    description = "Executes evaluation loop over validation dataloader with torch.no_grad()."
    icon = "RefreshCw"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_val_batches": {"type": "integer", "default": 20},
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
                id="val_dataloader",
                display_name="Val DataLoader",
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
                id="loop_body",
                display_name="Loop Body",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="completed",
                display_name="Completed",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="batch_x",
                display_name="Batch X",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer"),
            ),
            PortDefinition(
                id="batch_y",
                display_name="Batch Y",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer"),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {
            "batch_x": TensorSpec(dtype="int64", shape=[]),
            "batch_y": TensorSpec(dtype="int64", shape=[]),
        }
