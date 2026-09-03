"""Built-in composite module definitions for hierarchical nanoGPT graphs."""

from typing import Any, Dict, List, Optional

from neural_blueprint.ir.models import (
    PortDefinition,
    SymbolDim,
    TensorSpec,
    TensorType,
)
from neural_blueprint.registry.base import (
    NodeDefinition,
    NodeValidationContext,
)
from neural_blueprint.registry.registry import register_node


@register_node
class NanoGPTInputEmbeddingsComposite(NodeDefinition):
    type_id = "builtin.nanogpt_input_embeddings@1"
    version = 1
    display_name = "Input Embeddings"
    category = "Composite Modules"
    description = "Hierarchical composite module containing token and position embeddings with addition and dropout."
    icon = "Layers"
    is_composite = True

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="tokens",
                display_name="Tokens [B, T]",
                direction="input",
                required=True,
                tensor_type=TensorType(dtype_family="integer"),
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
                display_name="Embeddings [B, T, C]",
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
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }


@register_node
class NanoGPTAttentionComposite(NodeDefinition):
    type_id = "builtin.nanogpt_attention@1"
    version = 1
    display_name = "Causal Self-Attention"
    category = "Composite Modules"
    description = "Hierarchical composite module for multi-head causal self-attention with fused QKV and output projection."
    icon = "Zap"
    is_composite = True

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="input", display_name="Input [B, T, C]", direction="input", required=True
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output [B, T, C]", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }


@register_node
class NanoGPTMLPComposite(NodeDefinition):
    type_id = "builtin.nanogpt_mlp@1"
    version = 1
    display_name = "MLP"
    category = "Composite Modules"
    description = "Hierarchical composite module for transformer feed-forward network with 4xC expansion, GELU and projection."
    icon = "Activity"
    is_composite = True

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="input", display_name="Input [B, T, C]", direction="input", required=True
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output [B, T, C]", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }


@register_node
class NanoGPTBlockComposite(NodeDefinition):
    type_id = "builtin.nanogpt_block@1"
    version = 1
    display_name = "Transformer Block"
    category = "Composite Modules"
    description = "Hierarchical composite module containing LayerNorm, Causal Attention, MLP, and dual residual add paths."
    icon = "Box"
    is_composite = True

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="input", display_name="Input [B, T, C]", direction="input", required=True
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output [B, T, C]", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }


@register_node
class NanoGPTStackComposite(NodeDefinition):
    type_id = "builtin.nanogpt_stack@1"
    version = 1
    display_name = "Transformer Stack"
    category = "Composite Modules"
    description = "Repeated stack executing N independent Transformer Block instances sequentially."
    icon = "Layers"
    is_composite = True

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="input", display_name="Input [B, T, C]", direction="input", required=True
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output [B, T, C]", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }


@register_node
class LlamaInputEmbeddingsComposite(NodeDefinition):
    type_id = "builtin.llama_input_embeddings@1"
    version = 1
    display_name = "Llama Input Embeddings"
    category = "Composite Modules"
    description = "Hierarchical composite module containing token embeddings."
    icon = "Layers"
    is_composite = True

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="tokens", display_name="Tokens", direction="input", required=True
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Embeddings", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }


@register_node
class LlamaAttentionComposite(NodeDefinition):
    type_id = "builtin.llama_attention@1"
    version = 1
    display_name = "Llama Attention"
    category = "Composite Modules"
    description = "Hierarchical composite module containing QKV projection, RoPE, Grouped Query Attention, and output linear projection."
    icon = "Layers"
    is_composite = True

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="input", display_name="Input [B, T, C]", direction="input", required=True
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output [B, T, C]", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }


@register_node
class LlamaMLPComposite(NodeDefinition):
    type_id = "builtin.llama_mlp@1"
    version = 1
    display_name = "Llama MLP"
    category = "Composite Modules"
    description = "Hierarchical composite module containing SwiGLU feedforward block."
    icon = "Layers"
    is_composite = True

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="input", display_name="Input [B, T, C]", direction="input", required=True
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output [B, T, C]", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }


@register_node
class LlamaBlockComposite(NodeDefinition):
    type_id = "builtin.llama_block@1"
    version = 1
    display_name = "Llama Transformer Block"
    category = "Composite Modules"
    description = "Complete Llama Transformer Block with pre-RMSNorm attention and SwiGLU residual streams."
    icon = "Layers"
    is_composite = True

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="input", display_name="Input [B, T, C]", direction="input", required=True
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output [B, T, C]", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }


@register_node
class LlamaStackComposite(NodeDefinition):
    type_id = "builtin.llama_stack@1"
    version = 1
    display_name = "Llama Stack"
    category = "Composite Modules"
    description = "Repeated stack executing N independent Llama Transformer Block instances sequentially."
    icon = "Layers"
    is_composite = True

    def property_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="input", display_name="Input [B, T, C]", direction="input", required=True
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Output [B, T, C]", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        in_spec = inputs.get("input")
        if in_spec:
            return {"output": in_spec}
        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), SymbolDim(name="C")],
            )
        }
