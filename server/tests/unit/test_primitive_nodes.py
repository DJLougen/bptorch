"""Contract tests and runtime verification for all primitive nodes."""

import math
import pytest
import torch
import torch.nn as nn
from neural_blueprint.registry.base import NodeValidationContext
from neural_blueprint.registry.registry import global_registry
from neural_blueprint.runtime.initialization import init_nanogpt_weights


def test_all_nodes_pass_contract_checks():
    nodes = global_registry.list_all()
    assert len(nodes) >= 20

    ctx = NodeValidationContext(
        model_config={
            "block_size": 32,
            "vocab_size": 128,
            "n_layer": 2,
            "n_head": 4,
            "n_embd": 64,
            "dropout": 0.0,
            "bias": True,
        }
    )

    for node_def in nodes:
        assert node_def.type_id.startswith("builtin.")
        assert node_def.version >= 1
        assert node_def.display_name
        assert node_def.category

        # Property schema check
        schema = node_def.property_schema()
        assert isinstance(schema, dict)

        # Dynamic ports
        inputs = node_def.input_ports({}, ctx)
        outputs = node_def.output_ports({}, ctx)
        assert isinstance(inputs, list)
        assert isinstance(outputs, list)

        # Parameter spec
        p_spec = node_def.parameter_spec({}, ctx)
        assert p_spec.trainable_count >= 0
        assert p_spec.frozen_count >= 0

        # Runtime spec
        runtime_spec = node_def.build_runtime({}, ctx)
        if runtime_spec:
            assert runtime_spec.module_type in ("nn_module", "functional")
            assert runtime_spec.factory is not None


def test_embedding_node_runtime():
    emb_def = global_registry.require("builtin.embedding@1")
    ctx = NodeValidationContext(model_config={"vocab_size": 100, "n_embd": 32})
    r_spec = emb_def.build_runtime({"num_embeddings": 100, "embedding_dim": 32}, ctx)
    module = r_spec.factory()

    assert isinstance(module, nn.Embedding)
    assert module.num_embeddings == 100
    assert module.embedding_dim == 32

    # Forward check
    idx = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out = module(idx)
    assert out.shape == torch.Size([2, 3, 32])


def test_layernorm_node_runtime():
    ln_def = global_registry.require("builtin.layernorm@1")
    ctx = NodeValidationContext(model_config={"n_embd": 64, "bias": True})
    r_spec = ln_def.build_runtime({"normalized_shape": 64, "bias": True}, ctx)
    module = r_spec.factory()

    assert isinstance(module, nn.Module)
    assert module.weight.shape == torch.Size([64])
    assert module.bias.shape == torch.Size([64])

    x = torch.randn(2, 4, 64)
    out = module(x)
    assert out.shape == torch.Size([2, 4, 64])


def test_split_qkv_and_heads_roundtrip():
    # 1. SplitQKV
    split_qkv_def = global_registry.require("builtin.split_qkv@1")
    ctx = NodeValidationContext(model_config={"n_embd": 64})
    r_qkv = split_qkv_def.build_runtime({"n_embd": 64}, ctx)
    fn_qkv = r_qkv.factory()

    fused = torch.randn(2, 8, 192)  # [B, T, 3*C]
    res_qkv = fn_qkv(fused)
    assert res_qkv["q"].shape == torch.Size([2, 8, 64])
    assert res_qkv["k"].shape == torch.Size([2, 8, 64])
    assert res_qkv["v"].shape == torch.Size([2, 8, 64])

    # 2. SplitHeads (B=2, T=8, C=64, n_head=4, head_dim=16)
    split_heads_def = global_registry.require("builtin.split_heads@1")
    ctx_heads = NodeValidationContext(model_config={"n_head": 4, "n_embd": 64})
    r_heads = split_heads_def.build_runtime({"n_head": 4, "n_embd": 64}, ctx_heads)
    fn_heads = r_heads.factory()

    q_heads = fn_heads(res_qkv["q"])
    assert q_heads.shape == torch.Size([2, 4, 8, 16])  # [B, NH, T, HD]

    # 3. MergeHeads
    merge_heads_def = global_registry.require("builtin.merge_heads@1")
    r_merge = merge_heads_def.build_runtime({"n_embd": 64}, ctx_heads)
    fn_merge = r_merge.factory()

    q_merged = fn_merge(q_heads)
    assert q_merged.shape == torch.Size([2, 8, 64])

    # Check numerical preservation
    torch.testing.assert_close(q_merged, res_qkv["q"])


def test_attention_modes_match():
    # Test SDPA vs Manual Causal Attention equivalence on CPU
    torch.manual_seed(42)
    B, NH, T, HD = 2, 4, 8, 16
    q = torch.randn(B, NH, T, HD)
    k = torch.randn(B, NH, T, HD)
    v = torch.randn(B, NH, T, HD)

    # SDPA
    sdpa_def = global_registry.require("builtin.sdpa@1")
    r_sdpa = sdpa_def.build_runtime({"is_causal": True, "dropout": 0.0}, None)
    m_sdpa = r_sdpa.factory()
    m_sdpa.eval()
    out_sdpa = m_sdpa(q, k, v)

    # Manual Causal Attention
    manual_def = global_registry.require("builtin.manual_causal_attention@1")
    r_manual = manual_def.build_runtime({"dropout": 0.0}, None)
    m_manual = r_manual.factory()
    m_manual.eval()
    out_manual = m_manual(q, k, v)

    assert out_sdpa.shape == out_manual.shape == torch.Size([B, NH, T, HD])
    torch.testing.assert_close(out_sdpa, out_manual, rtol=1e-5, atol=1e-6)


def test_nanogpt_initialization():
    class MiniGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(100, 64)
            self.ln_1 = nn.LayerNorm(64)
            self.c_attn = nn.Linear(64, 192)
            self.c_proj = nn.Linear(64, 64)

    model = MiniGPT()
    init_nanogpt_weights(model, n_layer=2)

    # Check c_proj std is scaled
    expected_std = 0.02 / math.sqrt(2.0 * 2)
    assert abs(model.c_proj.weight.std().item() - expected_std) < 0.02

def test_rmsnorm_node_runtime():
    rms_def = global_registry.require("builtin.rmsnorm@1")
    ctx = NodeValidationContext(model_config={"n_embd": 32})
    r_spec = rms_def.build_runtime({"normalized_shape": 32}, ctx)
    module = r_spec.factory()

    assert not hasattr(module, "bias") or module.bias is None
    x = torch.randn(2, 4, 32)
    out = module(x)
    assert out.shape == torch.Size([2, 4, 32])


def test_rope_preserves_norm():
    rope_def = global_registry.require("builtin.rope@1")
    ctx = NodeValidationContext(model_config={"n_head": 4, "head_dim": 8})
    r_spec = rope_def.build_runtime({"head_dim": 8, "n_head": 4}, ctx)
    fn = r_spec.factory()

    x = torch.randn(2, 6, 4, 8)
    y = fn(x)
    assert y.shape == x.shape
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), rtol=1e-4)


def test_swiglu_shape():
    swiglu_def = global_registry.require("builtin.swiglu@1")
    ctx = NodeValidationContext(model_config={"n_embd": 32})
    r_spec = swiglu_def.build_runtime({"n_embd": 32}, ctx)
    module = r_spec.factory()

    x = torch.randn(2, 8, 32)
    out = module(x)
    assert out.shape == torch.Size([2, 8, 32])


def test_gqa_shape():
    gqa_def = global_registry.require("builtin.grouped_query_attention@1")
    ctx = NodeValidationContext(model_config={"n_head": 4, "n_kv_head": 2, "dropout": 0.0})
    r_spec = gqa_def.build_runtime({"n_head": 4, "n_kv_head": 2, "dropout": 0.0}, ctx)
    fn = r_spec.factory()

    q = torch.randn(2, 8, 32)
    k = torch.randn(2, 8, 16)
    v = torch.randn(2, 8, 16)
    out = fn(q, k, v)
    assert out.shape == torch.Size([2, 8, 32])


@pytest.mark.asyncio
async def test_llama_tiny_validates_and_runs():
    from neural_blueprint.templates.llama import create_llama_tiny_template
    from neural_blueprint.validation.validator import ProjectValidator
    from neural_blueprint.runtime.inference import InferenceEngine

    p = create_llama_tiny_template()
    val = ProjectValidator().validate(p)
    assert val.valid

    engine = InferenceEngine(project=p)
    res = await engine.infer()
    assert res.get("tensor_count", 0) >= 1


def test_comment_node_registration():
    comment_node = global_registry.get("builtin.comment@1")
    assert comment_node is not None
    assert comment_node.type_id == "builtin.comment@1"
    assert comment_node.category == "Debug"
    assert len(comment_node.input_ports({})) == 0
    assert len(comment_node.output_ports({})) == 0
