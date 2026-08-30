"""Persistence Primitives: SaveCheckpoint, LoadCheckpoint, ExportModel."""

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
class SaveCheckpointNode(NodeDefinition):
    type_id = "builtin.save_checkpoint@1"
    version = 1
    display_name = "Save Checkpoint"
    category = "Persistence"
    description = "Saves model weights, optimizer state, and training metadata to disk."
    icon = "Box"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "save_dir": {"type": "string", "default": "checkpoints"},
                "filename_prefix": {"type": "string", "default": "ckpt"},
                "save_optimizer": {"type": "boolean", "default": True},
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
                id="model",
                display_name="Model",
                direction="input",
                kind="data",
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
                id="saved_path",
                display_name="Saved Path",
                direction="output",
                kind="data",
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
class LoadCheckpointNode(NodeDefinition):
    type_id = "builtin.load_checkpoint@1"
    version = 1
    display_name = "Restore Checkpoint"
    category = "Persistence"
    description = "Restores model weights and optimizer states from a saved checkpoint file."
    icon = "RefreshCw"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "checkpoint_path": {"type": "string", "default": "checkpoints/ckpt_best.pt"},
            },
            "required": ["checkpoint_path"],
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
                id="model",
                display_name="Model",
                direction="input",
                kind="data",
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
                id="restored_step",
                display_name="Restored Step",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=0),
            ),
            PortDefinition(
                id="restored_loss",
                display_name="Restored Loss",
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
            "restored_step": TensorSpec(dtype="int64", shape=[LiteralDim(value=1)]),
            "restored_loss": TensorSpec(dtype="float32", shape=[LiteralDim(value=1)]),
        }


@register_node
class ExportModelNode(NodeDefinition):
    type_id = "builtin.export_model@1"
    version = 1
    display_name = "Export Model"
    category = "Persistence"
    description = "Exports trained model graph to ONNX, TorchScript, or SafeTensors format."
    icon = "LogOut"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["torchscript", "onnx", "safetensors"],
                    "default": "torchscript",
                },
                "output_path": {"type": "string", "default": "exports/model.pt"},
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
                id="model",
                display_name="Model",
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
                id="exported_path",
                display_name="Exported Path",
                direction="output",
                kind="data",
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {}
