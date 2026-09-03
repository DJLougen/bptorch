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


@register_node
class RoPENode(NodeDefinition):
    type_id = "builtin.rope@1"
    version = 1
    display_name = "Rotary Position Embedding (RoPE)"
    category = "Attention"
    description = "Applies rotary position embeddings to query or key tensor."
    icon = "RotateCw"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "head_dim": {"type": ["integer", "object"], "default": 32},
                "n_head": {
                    "type": ["integer", "object"],
                    "default": {"kind": "config_ref", "key": "n_head"},
                },
                "base": {"type": "number", "default": 10000.0},
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
                display_name="Input (Q or K)",
                direction="input",
                required=True,
                tensor_type=TensorType(dtype_family="floating"),
            ),
            PortDefinition(
                id="positions",
                display_name="Positions",
                direction="input",
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
                id="output",
                display_name="Rotated",
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

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> RuntimeModuleSpec:
        cfg = context.model_config if context else {}
        head_dim_prop = evaluate_value(properties.get("head_dim", 32), cfg)
        head_dim = int(head_dim_prop) if isinstance(head_dim_prop, (int, float)) else 32
        n_head_prop = evaluate_value(properties.get("n_head", 4), cfg)
        n_head = int(n_head_prop) if isinstance(n_head_prop, (int, float)) else 4
        base = float(evaluate_value(properties.get("base", 10000.0), cfg))

        def apply_rope(x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
            orig_shape = x.shape
            orig_dim = x.dim()

            if orig_dim == 3:
                B, T, C = x.shape
                d = C // n_head
                x_4d = x.view(B, T, n_head, d).transpose(1, 2)
            elif orig_dim == 4:
                x_4d = x
            else:
                return x

            D = x_4d.size(-1)
            T = x_4d.size(-2)
            device = x.device

            inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))

            if positions is not None:
                t = positions.to(dtype=torch.float32, device=device)
            else:
                t = torch.arange(T, device=device, dtype=torch.float32)

            freqs = torch.outer(t, inv_freq)
            cos = freqs.cos().unsqueeze(0).unsqueeze(0)
            sin = freqs.sin().unsqueeze(0).unsqueeze(0)

            x1 = x_4d[..., ::2]
            x2 = x_4d[..., 1::2]

            even_rot = x1 * cos - x2 * sin
            odd_rot = x1 * sin + x2 * cos

            rotated = torch.stack([even_rot, odd_rot], dim=-1).flatten(-2)

            if orig_dim == 3:
                return rotated.transpose(1, 2).contiguous().view(orig_shape)
            return rotated

        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: apply_rope,
            kwargs={"head_dim": head_dim, "n_head": n_head, "base": base},
        )


@register_node
class GroupedQueryAttentionNode(NodeDefinition):
    type_id = "builtin.grouped_query_attention@1"
    version = 1
    display_name = "Grouped Query Attention (GQA)"
    category = "Attention"
    description = "Grouped Query Attention with repeated KV heads and causal masking."
    icon = "Zap"

    def property_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "n_head": {
                    "type": ["integer", "object"],
                    "default": {"kind": "config_ref", "key": "n_head"},
                },
                "n_kv_head": {
                    "type": ["integer", "object"],
                    "default": {"kind": "config_ref", "key": "n_kv_head"},
                },
                "dropout": {
                    "type": ["number", "object"],
                    "default": {"kind": "config_ref", "key": "dropout"},
                },
            },
            "required": ["n_head"],
        }

    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        return [
            PortDefinition(
                id="q",
                display_name="Query",
                direction="input",
                required=True,
                tensor_type=TensorType(dtype_family="floating"),
            ),
            PortDefinition(
                id="k",
                display_name="Key",
                direction="input",
                required=True,
                tensor_type=TensorType(dtype_family="floating"),
            ),
            PortDefinition(
                id="v",
                display_name="Value",
                direction="input",
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
                id="output",
                display_name="Attended",
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
        spec_q = inputs.get("q")
        if spec_q:
            return {"output": TensorSpec(dtype=spec_q.dtype, shape=list(spec_q.shape))}
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
        n_head = int(evaluate_value(properties.get("n_head", 4), cfg))
        n_kv_head_val = evaluate_value(properties.get("n_kv_head", n_head), cfg)
        n_kv_head = int(n_kv_head_val) if n_kv_head_val is not None else n_head
        dropout_p = float(evaluate_value(properties.get("dropout", 0.0), cfg))

        def run_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            B, T_q, C_q = q.size(0), q.size(1), q.size(-1)
            head_dim = C_q // n_head

            if q.dim() == 3:
                q_heads = q.view(B, T_q, n_head, head_dim).transpose(1, 2)
            else:
                q_heads = q

            if k.dim() == 3:
                T_k = k.size(1)
                kv_dim = k.size(-1)
                kv_head_dim = kv_dim // n_kv_head
                if kv_head_dim != head_dim and kv_dim == C_q:
                    # If k was projected with out_features=n_embd, slice down to n_kv_head * head_dim
                    k = k[..., : n_kv_head * head_dim]
                    v = v[..., : n_kv_head * head_dim]
                    kv_head_dim = head_dim
                k_heads = k.view(B, T_k, n_kv_head, kv_head_dim).transpose(1, 2)
                v_heads = v.view(B, T_k, n_kv_head, kv_head_dim).transpose(1, 2)
            else:
                k_heads = k
                v_heads = v

            if n_head != n_kv_head:
                reps = n_head // n_kv_head
                k_heads = torch.repeat_interleave(k_heads, reps, dim=1)
                v_heads = torch.repeat_interleave(v_heads, reps, dim=1)
            out = F.scaled_dot_product_attention(
                q_heads, k_heads, v_heads, is_causal=True, dropout_p=dropout_p
            )
            return out.transpose(1, 2).contiguous().view(B, T_q, n_head * head_dim)

        return RuntimeModuleSpec(
            module_type="functional",
            factory=lambda: run_gqa,
            kwargs={"n_head": n_head, "n_kv_head": n_kv_head, "dropout": dropout_p},
        )
