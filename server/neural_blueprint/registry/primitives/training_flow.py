"""Training Flow Control, Loop, Branch, Event, and Variable primitive nodes."""

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

# --- Flow Control Nodes ---


@register_node
class TrainingSequenceNode(NodeDefinition):
    type_id = "builtin.training_sequence@1"
    version = 1
    display_name = "Sequence"
    category = "Flow Control"
    description = "Executes connected execution branches sequentially from top to bottom."
    icon = "ListOrdered"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "branch_count": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 8,
                    "default": 3,
                }
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
        count = int(properties.get("branch_count", 3))
        return [
            PortDefinition(
                id=f"then_{i}",
                display_name=f"Then {i}",
                direction="output",
                kind="exec",
            )
            for i in range(count)
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}


@register_node
class EpochLoopNode(NodeDefinition):
    type_id = "builtin.epoch_loop@1"
    version = 1
    display_name = "Epoch Loop"
    category = "Flow Control"
    description = "Iterates training across a configured number of epochs."
    icon = "RefreshCw"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "num_epochs": {"type": "integer", "minimum": 1, "default": 10},
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
                id="num_epochs",
                display_name="Num Epochs",
                direction="input",
                kind="data",
                required=False,
                tensor_type=TensorType(dtype_family="integer", rank=0),
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
                id="current_epoch",
                display_name="Current Epoch",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=0),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"current_epoch": TensorSpec(dtype="int64", shape=[LiteralDim(value=1)])}


@register_node
class BatchLoopNode(NodeDefinition):
    type_id = "builtin.batch_loop@1"
    version = 1
    display_name = "Batch Loop"
    category = "Flow Control"
    description = "Iterates over batches yielded by a DataLoader."
    icon = "RefreshCw"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_steps": {"type": "integer", "minimum": 0, "default": 0},
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
                id="dataloader",
                display_name="DataLoader",
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
            PortDefinition(
                id="batch_idx",
                display_name="Batch Index",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=0),
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
            "batch_idx": TensorSpec(dtype="int64", shape=[LiteralDim(value=1)]),
        }


@register_node
class ValidationBranchNode(NodeDefinition):
    type_id = "builtin.validation_branch@1"
    version = 1
    display_name = "Validation Branch"
    category = "Flow Control"
    description = "Branches execution periodically based on step/epoch interval."
    icon = "GitFork"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "interval": {"type": "integer", "minimum": 1, "default": 50},
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
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="on_eval",
                display_name="On Eval",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="skip_eval",
                display_name="Skip",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="should_eval",
                display_name="Should Eval",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="boolean"),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"should_eval": TensorSpec(dtype="bool", shape=[LiteralDim(value=1)])}


@register_node
class EarlyStoppingGateNode(NodeDefinition):
    type_id = "builtin.early_stopping_gate@1"
    version = 1
    display_name = "Early Stopping Gate"
    category = "Flow Control"
    description = (
        "Tracks validation loss and signals execution termination if patience is exhausted."
    )
    icon = "Shield"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "patience": {"type": "integer", "minimum": 1, "default": 5},
                "min_delta": {"type": "number", "minimum": 0.0, "default": 0.001},
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
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="continue_exec",
                display_name="Continue",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="stop_exec",
                display_name="Stop",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="is_best",
                display_name="Is Best",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="boolean"),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"is_best": TensorSpec(dtype="bool", shape=[LiteralDim(value=1)])}


@register_node
class BranchNode(NodeDefinition):
    type_id = "builtin.branch@1"
    version = 1
    display_name = "Branch"
    category = "Flow Control"
    description = "Branches execution conditionally based on a boolean condition."
    icon = "GitFork"

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
                id="condition",
                display_name="Condition",
                direction="input",
                kind="data",
                required=True,
                tensor_type=TensorType(dtype_family="boolean"),
            ),
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="true_branch",
                display_name="True",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="false_branch",
                display_name="False",
                direction="output",
                kind="exec",
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}


@register_node
class DoOnceNode(NodeDefinition):
    type_id = "builtin.do_once@1"
    version = 1
    display_name = "Do Once"
    category = "Flow Control"
    description = "Executes the output branch only once until reset."
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
                id="reset",
                display_name="Reset",
                direction="input",
                kind="exec",
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
                display_name="Out",
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
class WhileLoopNode(NodeDefinition):
    type_id = "builtin.while_loop@1"
    version = 1
    display_name = "While Loop"
    category = "Flow Control"
    description = "Executes loop body repeatedly while condition evaluates to true."
    icon = "RefreshCw"

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
                id="condition",
                display_name="Condition",
                direction="input",
                kind="data",
                required=True,
                tensor_type=TensorType(dtype_family="boolean"),
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
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}


@register_node
class ForLoopNode(NodeDefinition):
    type_id = "builtin.for_loop@1"
    version = 1
    display_name = "For Loop"
    category = "Flow Control"
    description = "Iterates over a range from first_index to last_index."
    icon = "RefreshCw"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "first_index": {"type": "integer", "default": 0},
                "last_index": {"type": "integer", "default": 10},
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
                id="first_index",
                display_name="First Index",
                direction="input",
                kind="data",
                required=False,
                tensor_type=TensorType(dtype_family="integer"),
            ),
            PortDefinition(
                id="last_index",
                display_name="Last Index",
                direction="input",
                kind="data",
                required=False,
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
                id="index",
                display_name="Index",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=0),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"index": TensorSpec(dtype="int64", shape=[LiteralDim(value=1)])}


# --- Event Nodes ---


@register_node
class EventOnTrainBeginNode(NodeDefinition):
    type_id = "builtin.event_on_train_begin@1"
    version = 1
    display_name = "Event OnTrainBegin"
    category = "Events"
    description = "Fires at the start of a training session."
    icon = "Flame"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

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
class EventOnEpochStartNode(NodeDefinition):
    type_id = "builtin.event_on_epoch_start@1"
    version = 1
    display_name = "Event OnEpochStart"
    category = "Events"
    description = "Fires at the start of each training epoch."
    icon = "Flame"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

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
                id="exec_out",
                display_name="Exec",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="epoch",
                display_name="Epoch",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=0),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"epoch": TensorSpec(dtype="int64", shape=[LiteralDim(value=1)])}


@register_node
class EventOnBatchEndNode(NodeDefinition):
    type_id = "builtin.event_on_batch_end@1"
    version = 1
    display_name = "Event OnBatchEnd"
    category = "Events"
    description = "Fires after each batch optimization step is completed."
    icon = "Flame"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

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
                id="exec_out",
                display_name="Exec",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="step",
                display_name="Step",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=0),
            ),
            PortDefinition(
                id="loss",
                display_name="Loss",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="floating", rank=0),
            ),
            PortDefinition(
                id="lr",
                display_name="LR",
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
        return {
            "step": TensorSpec(dtype="int64", shape=[LiteralDim(value=1)]),
            "loss": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)]),
            "lr": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)]),
        }


@register_node
class EventOnValidationNode(NodeDefinition):
    type_id = "builtin.event_on_validation@1"
    version = 1
    display_name = "Event OnValidation"
    category = "Events"
    description = "Fires upon completion of a validation pass."
    icon = "Flame"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

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
                id="exec_out",
                display_name="Exec",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="epoch",
                display_name="Epoch",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=0),
            ),
            PortDefinition(
                id="val_loss",
                display_name="Val Loss",
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
        return {
            "epoch": TensorSpec(dtype="int64", shape=[LiteralDim(value=1)]),
            "val_loss": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)]),
        }


@register_node
class EventOnCheckpointNode(NodeDefinition):
    type_id = "builtin.event_on_checkpoint@1"
    version = 1
    display_name = "Event OnCheckpoint"
    category = "Events"
    description = "Fires when a model/optimizer checkpoint is saved to disk."
    icon = "Flame"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

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
                id="exec_out",
                display_name="Exec",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="step",
                display_name="Step",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=0),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"step": TensorSpec(dtype="int64", shape=[LiteralDim(value=1)])}


@register_node
class EventOnAnomalyDetectedNode(NodeDefinition):
    type_id = "builtin.event_on_anomaly@1"
    version = 1
    display_name = "Event OnAnomalyDetected"
    category = "Events"
    description = "Fires when NaN/Inf or exploding gradients are detected during training."
    icon = "Flame"

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

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
                id="exec_out",
                display_name="Exec",
                direction="output",
                kind="exec",
            ),
            PortDefinition(
                id="nan_count",
                display_name="NaN Count",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=0),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"nan_count": TensorSpec(dtype="int64", shape=[LiteralDim(value=1)])}


# --- Variable Nodes ---


@register_node
class GetVariableNode(NodeDefinition):
    type_id = "builtin.get_variable@1"
    version = 1
    display_name = "Get Variable"
    category = "Variables"
    description = "Retrieves the current value of a Blueprint variable."
    icon = "Key"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "variable_name": {"type": "string", "default": "learning_rate"},
            },
            "required": ["variable_name"],
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
        var_name = properties.get("variable_name", "value")
        return [
            PortDefinition(
                id="value",
                display_name=var_name,
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="any"),
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {"value": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)])}


@register_node
class SetVariableNode(NodeDefinition):
    type_id = "builtin.set_variable@1"
    version = 1
    display_name = "Set Variable"
    category = "Variables"
    description = "Assigns a new value to a Blueprint variable and passes execution."
    icon = "Key"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "variable_name": {"type": "string", "default": "learning_rate"},
            },
            "required": ["variable_name"],
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        var_name = properties.get("variable_name", "value")
        return [
            PortDefinition(
                id="exec_in",
                display_name="Exec",
                direction="input",
                kind="exec",
                required=False,
            ),
            PortDefinition(
                id="value",
                display_name=var_name,
                direction="input",
                kind="data",
                required=True,
                tensor_type=TensorType(dtype_family="any"),
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
                id="value",
                display_name="Value",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="any"),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("value")
        if in_spec:
            return {"value": in_spec}
        return {"value": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)])}
