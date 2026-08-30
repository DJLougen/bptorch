"""Data Pipeline primitives: DatasetSource, Tokenizer, DataLoader, BatchSampler, and DataAugmentation."""

from typing import Any, Dict, List, Optional

from neural_blueprint.ir.evaluator import evaluate_value
from neural_blueprint.ir.models import (
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
)
from neural_blueprint.registry.registry import register_node


@register_node
class DatasetSourceNode(NodeDefinition):
    type_id = "builtin.dataset_source@1"
    version = 1
    display_name = "Dataset Source"
    category = "Data Pipelines"
    description = (
        "Provides or generates training/validation dataset (HuggingFace, Disk, or Synthetic)."
    )
    icon = "Box"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "default": "synthetic_shakespeare"},
                "split": {"type": "string", "enum": ["train", "val", "test"], "default": "train"},
                "synthetic": {"type": "boolean", "default": True},
                "num_samples": {"type": "integer", "default": 1000},
                "vocab_size": {
                    "type": "object",
                    "default": {"kind": "config_ref", "key": "vocab_size"},
                },
                "sequence_length": {
                    "type": "object",
                    "default": {"kind": "config_ref", "key": "block_size"},
                },
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
            ),
            PortDefinition(
                id="dataset",
                display_name="Dataset",
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
class BPEPreTokenizerNode(NodeDefinition):
    type_id = "builtin.bpe_tokenizer@1"
    version = 1
    display_name = "BPE Tokenizer"
    category = "Data Pipelines"
    description = "Tokenizes raw string input into token id sequences using Byte-Pair Encoding."
    icon = "Scissors"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "vocab_size": {"type": "integer", "default": 50257},
                "model_type": {
                    "type": "string",
                    "enum": ["gpt2", "tiktoken_cl100k", "character"],
                    "default": "gpt2",
                },
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="raw_text",
                display_name="Raw Text",
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
                id="token_ids",
                display_name="Token IDs",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=2),
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {
            "token_ids": TensorSpec(
                dtype="int64",
                shape=[SymbolDim(name="B"), SymbolDim(name="T")],
            )
        }


@register_node
class BatchSamplerNode(NodeDefinition):
    type_id = "builtin.batch_sampler@1"
    version = 1
    display_name = "Batch Sampler"
    category = "Data Pipelines"
    description = "Samples batches of indices from a dataset with optional shuffling."
    icon = "Columns"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "batch_size": {
                    "type": "object",
                    "default": {"kind": "config_ref", "key": "batch_size"},
                },
                "shuffle": {"type": "boolean", "default": True},
                "drop_last": {"type": "boolean", "default": False},
            },
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="dataset",
                display_name="Dataset",
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
                id="batch_indices",
                display_name="Batch Indices",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=1),
            )
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        cfg = context.model_config if context else {}
        b_val = evaluate_value(properties.get("batch_size", 8), cfg)
        b_dim: ShapeDim = (
            LiteralDim(value=int(b_val)) if isinstance(b_val, (int, float)) else SymbolDim(name="B")
        )
        return {"batch_indices": TensorSpec(dtype="int64", shape=[b_dim])}


@register_node
class DataLoaderNode(NodeDefinition):
    type_id = "builtin.dataloader@1"
    version = 1
    display_name = "DataLoader"
    category = "Data Pipelines"
    description = "Constructs a PyTorch DataLoader yielding input/target batches (X, Y)."
    icon = "Box"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "batch_size": {
                    "type": "object",
                    "default": {"kind": "config_ref", "key": "batch_size"},
                },
                "shuffle": {"type": "boolean", "default": True},
                "pin_memory": {"type": "boolean", "default": True},
                "num_workers": {"type": "integer", "default": 0},
                "drop_last": {"type": "boolean", "default": False},
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
                id="dataset",
                display_name="Dataset",
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
                id="dataloader",
                display_name="DataLoader",
                direction="output",
                kind="data",
            ),
            PortDefinition(
                id="batch_x",
                display_name="Batch X",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=2),
            ),
            PortDefinition(
                id="batch_y",
                display_name="Batch Y",
                direction="output",
                kind="data",
                tensor_type=TensorType(dtype_family="integer", rank=2),
            ),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        cfg = context.model_config if context else {}
        b_val = evaluate_value(properties.get("batch_size", 8), cfg)
        b_dim: ShapeDim = (
            LiteralDim(value=int(b_val)) if isinstance(b_val, (int, float)) else SymbolDim(name="B")
        )
        t_dim: ShapeDim = (
            LiteralDim(value=int(cfg["block_size"]))
            if "block_size" in cfg and isinstance(cfg["block_size"], (int, float))
            else SymbolDim(name="T")
        )
        return {
            "batch_x": TensorSpec(dtype="int64", shape=[b_dim, t_dim]),
            "batch_y": TensorSpec(dtype="int64", shape=[b_dim, t_dim]),
        }


@register_node
class DataAugmentationNode(NodeDefinition):
    type_id = "builtin.data_augmentation@1"
    version = 1
    display_name = "Data Augmentation"
    category = "Data Pipelines"
    description = "Applies stochastic data transformations or token masking."
    icon = "Flame"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "augmentation_type": {
                    "type": "string",
                    "enum": ["random_masking", "random_crop", "gaussian_noise"],
                    "default": "random_masking",
                },
                "p": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.15},
            },
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
                kind="data",
                required=True,
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
                display_name="Output",
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
        if "input" in inputs:
            return {"output": inputs["input"]}
        return {"output": TensorSpec(dtype="float32", shape=[UnknownDim()])}
