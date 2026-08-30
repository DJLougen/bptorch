"""Attention operation primitive nodes matching nanoGPT architecture."""

import math
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
    RuntimeModuleSpec,
)
from neural_blueprint.registry.registry import register_node


@register_node
class SplitQKVNode(NodeDefinition):
    type_id = "builtin.split_qkv@1"
    version = 1
    display_name = "Split QKV"
    category = "Attention"
    description = (
        "Splits fused QKV linear projection [B, T, 3*C] into separate Query, Key and Value tensors."
    )
    icon = "GitFork"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "n_embd": {
                    "type": ["integer", "object"],
                    "default": {"kind": "config_ref", "key": "n_embd"},
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
                id="input",
                display_name="Fused QKV [B, T, 3C]",
                direction="input",
                required=True,
                tensor_type=TensorType(dtype_family="floating", rank=3),
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(id="q", display_name="Query (Q)", direction="output"),
            PortDefinition(id="k", display_name="Key (K)", direction="output"),
            PortDefinition(id="v", display_name="Value (V)", direction="output"),
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        cfg = context.model_config if context else {}
        emb_val = evaluate_value(properties.get("n_embd", 64), cfg)
        c_dim = (
            LiteralDim(value=int(emb_val))
            if isinstance(emb_val, (int, float))
            else SymbolDim(name="C")
        )

        in_spec = inputs.get("input")
        if in_spec and in_spec.shape:
            shape = list(in_spec.shape[:-1]) + [c_dim]
            spec = TensorSpec(dtype=in_spec.dtype, shape=shape)
        else:
            spec = TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), c_dim],
            )

        return {"q": spec, "k": spec, "v": spec}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        n_embd = int(evaluate_value(properties.get("n_embd", 64), cfg))

        def split_qkv(x=None, input=None):
            val = input if input is not None else x
            q, k, v = val.split(n_embd, dim=-1)
            return {"q": q, "k": k, "v": v}

        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: split_qkv,
        )


@register_node
class SplitHeadsNode(NodeDefinition):
    type_id = "builtin.split_heads@1"
    version = 1
    display_name = "Split Heads"
    category = "Attention"
    description = "Reshapes and transposes [B, T, C] -> [B, n_head, T, head_dim]."
    icon = "Columns"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "n_head": {
                    "type": ["integer", "object"],
                    "default": {"kind": "config_ref", "key": "n_head"},
                },
                "n_embd": {
                    "type": ["integer", "object"],
                    "default": {"kind": "config_ref", "key": "n_embd"},
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
                id="input", display_name="Input [B, T, C]", direction="input", required=True
            )
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(id="output", display_name="Heads [B, NH, T, HD]", direction="output")
        ]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        cfg = context.model_config if context else {}
        n_head_val = evaluate_value(properties.get("n_head", 4), cfg)
        n_embd_val = evaluate_value(properties.get("n_embd", 64), cfg)

        nh_dim = (
            LiteralDim(value=int(n_head_val))
            if isinstance(n_head_val, (int, float))
            else SymbolDim(name="NH")
        )
        if (
            isinstance(n_head_val, (int, float))
            and isinstance(n_embd_val, (int, float))
            and n_head_val > 0
        ):
            hd_dim = LiteralDim(value=int(n_embd_val) // int(n_head_val))
        else:
            hd_dim = SymbolDim(name="HD")

        in_spec = inputs.get("input")
        if in_spec and in_spec.shape:
            b_dim = in_spec.shape[0]
            t_dim = in_spec.shape[1] if len(in_spec.shape) >= 2 else SymbolDim(name="T")
            out_shape = [b_dim, nh_dim, t_dim, hd_dim]
            return {"output": TensorSpec(dtype=in_spec.dtype, shape=out_shape)}

        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), nh_dim, SymbolDim(name="T"), hd_dim],
            )
        }

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        n_head = int(evaluate_value(properties.get("n_head", 4), cfg))

        def split_heads_fn(x=None, input=None):
            val = input if input is not None else x
            if val.dim() == 2:
                val = val.unsqueeze(1)
            b, t, c = val.size()
            return val.view(b, t, n_head, c // n_head).transpose(1, 2)

        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: split_heads_fn,
        )


@register_node
class MergeHeadsNode(NodeDefinition):
    type_id = "builtin.merge_heads@1"
    version = 1
    display_name = "Merge Heads"
    category = "Attention"
    description = "Transposes and reassembles multi-head outputs side-by-side: [B, n_head, T, head_dim] -> [B, T, C]."
    icon = "Combine"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "n_embd": {
                    "type": ["integer", "object"],
                    "default": {"kind": "config_ref", "key": "n_embd"},
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
                id="input", display_name="Heads [B, NH, T, HD]", direction="input", required=True
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
        cfg = context.model_config if context else {}
        n_embd_val = evaluate_value(properties.get("n_embd", 64), cfg)
        c_dim = (
            LiteralDim(value=int(n_embd_val))
            if isinstance(n_embd_val, (int, float))
            else SymbolDim(name="C")
        )

        in_spec = inputs.get("input")
        if in_spec and in_spec.shape and len(in_spec.shape) == 4:
            b_dim = in_spec.shape[0]
            t_dim = in_spec.shape[2]
            return {"output": TensorSpec(dtype=in_spec.dtype, shape=[b_dim, t_dim, c_dim])}

        return {
            "output": TensorSpec(
                dtype="float32",
                shape=[SymbolDim(name="B"), SymbolDim(name="T"), c_dim],
            )
        }

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        n_embd = int(evaluate_value(properties.get("n_embd", 64), cfg))

        def merge_heads_fn(x=None, input=None):
            val = input if input is not None else x
            b, nh, t, hs = val.size()
            return val.transpose(1, 2).contiguous().view(b, t, n_embd)

        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: merge_heads_fn,
        )


@register_node
class ScaledDotProductAttentionNode(NodeDefinition):
    type_id = "builtin.sdpa@1"
    version = 1
    display_name = "Scaled Dot-Product Attention"
    category = "Attention"
    description = (
        "PyTorch native F.scaled_dot_product_attention with causal masking and optional dropout."
    )
    icon = "Zap"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "is_causal": {"type": "boolean", "default": True},
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
        return [
            PortDefinition(id="q", display_name="Query (Q)", direction="input", required=True),
            PortDefinition(id="k", display_name="Key (K)", direction="input", required=True),
            PortDefinition(id="v", display_name="Value (V)", direction="input", required=True),
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Attention Output", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        spec_q = inputs.get("q")
        if spec_q:
            return {"output": spec_q}
        return {}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        is_causal = bool(properties.get("is_causal", True))
        dropout_p = float(evaluate_value(properties.get("dropout", 0.0), cfg))

        class SDPAModule(nn.Module):
            def forward(self, q, k, v):
                p = dropout_p if self.training else 0.0
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=p, is_causal=is_causal
                )

        return RuntimeModuleSpec(
            module_type="nn_module",
            factory=lambda: SDPAModule(),
        )


@register_node
class ManualCausalAttentionNode(NodeDefinition):
    type_id = "builtin.manual_causal_attention@1"
    version = 1
    display_name = "Manual Causal Attention"
    category = "Attention"
    description = (
        "Exact semantic attention matching karpathy/nanoGPT manual causal attention implementation."
    )
    icon = "Settings"

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
        return [
            PortDefinition(id="q", display_name="Query (Q)", direction="input", required=True),
            PortDefinition(id="k", display_name="Key (K)", direction="input", required=True),
            PortDefinition(id="v", display_name="Value (V)", direction="input", required=True),
            PortDefinition(
                id="mask", display_name="Causal Mask", direction="input", required=False
            ),
        ]

    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [PortDefinition(id="output", display_name="Attention Output", direction="output")]

    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        spec_q = inputs.get("q")
        if spec_q:
            return {"output": spec_q}
        return {}

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        dropout_p = float(evaluate_value(properties.get("dropout", 0.0), cfg))

        class ManualAttentionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_dropout = nn.Dropout(dropout_p)

            def forward(self, q, k, v, mask=None):
                b, nh, t, hs = q.size()
                # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
                if mask is not None:
                    att = att.masked_fill(mask[:, :, :t, :t] == 0, float("-inf"))
                else:
                    # Create causal mask on the fly if not provided
                    causal_mask = torch.tril(torch.ones(t, t, device=q.device)).view(1, 1, t, t)
                    att = att.masked_fill(causal_mask == 0, float("-inf"))

                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                return y

        return RuntimeModuleSpec(
            module_type="nn_module",
            factory=lambda: ManualAttentionModule(),
        )
