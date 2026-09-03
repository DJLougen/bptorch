"""Llama-Tiny Canonical Architecture Template Generator."""

from typing import Any, Dict

from neural_blueprint.ir.models import (
    ConfigRefValue,
    Edge,
    ExpressionOp,
    ExpressionValue,
    GraphDefinition,
    GraphInterface,
    ModelDefinition,
    NodeInstance,
    PortDefinition,
    PortReference,
    Project,
    ProjectMetadata,
    SafeExpression,
    TrainingConfig,
    UIState,
    WeightBinding,
    WeightBindingEndpoint,
)


def create_llama_tiny_template(
    block_size: int = 32,
    vocab_size: int = 64,
    n_layer: int = 2,
    n_head: int = 4,
    n_kv_head: int = 2,
    n_embd: int = 32,
    dropout: float = 0.0,
) -> Project:
    """Create a minimal Llama-style Transformer blueprint project."""
    config: Dict[str, Any] = {
        "block_size": block_size,
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "n_embd": n_embd,
        "dropout": dropout,
    }


    head_dim_expr = ExpressionValue(
        expression=SafeExpression(
            op=ExpressionOp.INTEGER_DIVIDE,
            left=ConfigRefValue(key="n_embd"),
            right=ConfigRefValue(key="n_head"),
        )
    )
    kv_out_expr = ExpressionValue(
        expression=SafeExpression(
            op=ExpressionOp.MULTIPLY,
            left=ConfigRefValue(key="n_kv_head"),
            right=SafeExpression(
                op=ExpressionOp.INTEGER_DIVIDE,
                left=ConfigRefValue(key="n_embd"),
                right=ConfigRefValue(key="n_head"),
            ),
        )
    )

    # 1. Subgraph: Input Embeddings (Token Embeddings only, no learned pos)
    g_input_emb = GraphDefinition(
        id="graph_llama_input_embeddings",
        name="Llama Input Embeddings",
        kind="module",
        interface=GraphInterface(
            inputs=[
                PortDefinition(id="tokens", display_name="Tokens", direction="input", required=True)
            ],
            outputs=[PortDefinition(id="output", display_name="Embeddings", direction="output")],
        ),
        nodes=[
            NodeInstance(
                id="node_in_tokens",
                definition_id="builtin.module_input@1",
                display_name="Tokens Input",
                properties={"name": "tokens"},
            ),
            NodeInstance(
                id="node_tok_emb",
                definition_id="builtin.embedding@1",
                display_name="Token Embedding (wte)",
                properties={
                    "num_embeddings": ConfigRefValue(key="vocab_size"),
                    "embedding_dim": ConfigRefValue(key="n_embd"),
                },
            ),
            NodeInstance(
                id="node_out_emb",
                definition_id="builtin.module_output@1",
                display_name="Embeddings Output",
                properties={"name": "output"},
            ),
        ],
        edges=[
            Edge(
                id="e_in_to_tok",
                source=PortReference(node_id="node_in_tokens", port_id="output"),
                target=PortReference(node_id="node_tok_emb", port_id="input"),
            ),
            Edge(
                id="e_tok_to_out",
                source=PortReference(node_id="node_tok_emb", port_id="output"),
                target=PortReference(node_id="node_out_emb", port_id="input"),
            ),
        ],
    )

    # 2. Subgraph: Grouped Query Attention with RoPE
    g_attention = GraphDefinition(
        id="graph_llama_attention",
        name="Llama Attention (GQA + RoPE)",
        kind="module",
        interface=GraphInterface(
            inputs=[
                PortDefinition(id="input", display_name="Input", direction="input", required=True)
            ],
            outputs=[PortDefinition(id="output", display_name="Output", direction="output")],
        ),
        nodes=[
            NodeInstance(
                id="node_attn_in",
                definition_id="builtin.module_input@1",
                display_name="Attention Input",
                properties={"name": "input"},
            ),
            NodeInstance(
                id="node_q_proj",
                definition_id="builtin.linear@1",
                display_name="Q Projection",
                properties={
                    "in_features": ConfigRefValue(key="n_embd"),
                    "out_features": ConfigRefValue(key="n_embd"),
                    "bias": False,
                },
            ),
            NodeInstance(
                id="node_k_proj",
                definition_id="builtin.linear@1",
                display_name="K Projection",
                properties={
                    "in_features": ConfigRefValue(key="n_embd"),
                    "out_features": kv_out_expr,
                    "bias": False,
                },
            ),
            NodeInstance(
                id="node_v_proj",
                definition_id="builtin.linear@1",
                display_name="V Projection",
                properties={
                    "in_features": ConfigRefValue(key="n_embd"),
                    "out_features": kv_out_expr,
                    "bias": False,
                },
            ),
            NodeInstance(
                id="node_q_rope",
                definition_id="builtin.rope@1",
                display_name="RoPE on Q",
                properties={
                    "head_dim": head_dim_expr,
                    "n_head": ConfigRefValue(key="n_head"),
                },
            ),
            NodeInstance(
                id="node_k_rope",
                definition_id="builtin.rope@1",
                display_name="RoPE on K",
                properties={
                    "head_dim": head_dim_expr,
                    "n_head": ConfigRefValue(key="n_kv_head"),
                },
            ),
            NodeInstance(
                id="node_gqa",
                definition_id="builtin.grouped_query_attention@1",
                display_name="Grouped Query Attention",
                properties={
                    "n_head": ConfigRefValue(key="n_head"),
                    "n_kv_head": ConfigRefValue(key="n_kv_head"),
                    "dropout": ConfigRefValue(key="dropout"),
                },
            ),
            NodeInstance(
                id="node_out_proj",
                definition_id="builtin.linear@1",
                display_name="Output Projection",
                properties={
                    "in_features": ConfigRefValue(key="n_embd"),
                    "out_features": ConfigRefValue(key="n_embd"),
                    "bias": False,
                },
            ),
            NodeInstance(
                id="node_attn_out",
                definition_id="builtin.module_output@1",
                display_name="Attention Output",
                properties={"name": "output"},
            ),
        ],
        edges=[
            Edge(
                id="e_in_to_q",
                source=PortReference(node_id="node_attn_in", port_id="output"),
                target=PortReference(node_id="node_q_proj", port_id="input"),
            ),
            Edge(
                id="e_in_to_k",
                source=PortReference(node_id="node_attn_in", port_id="output"),
                target=PortReference(node_id="node_k_proj", port_id="input"),
            ),
            Edge(
                id="e_in_to_v",
                source=PortReference(node_id="node_attn_in", port_id="output"),
                target=PortReference(node_id="node_v_proj", port_id="input"),
            ),
            Edge(
                id="e_q_to_rope",
                source=PortReference(node_id="node_q_proj", port_id="output"),
                target=PortReference(node_id="node_q_rope", port_id="input"),
            ),
            Edge(
                id="e_k_to_rope",
                source=PortReference(node_id="node_k_proj", port_id="output"),
                target=PortReference(node_id="node_k_rope", port_id="input"),
            ),
            Edge(
                id="e_rope_q_to_gqa",
                source=PortReference(node_id="node_q_rope", port_id="output"),
                target=PortReference(node_id="node_gqa", port_id="q"),
            ),
            Edge(
                id="e_rope_k_to_gqa",
                source=PortReference(node_id="node_k_rope", port_id="output"),
                target=PortReference(node_id="node_gqa", port_id="k"),
            ),
            Edge(
                id="e_v_to_gqa",
                source=PortReference(node_id="node_v_proj", port_id="output"),
                target=PortReference(node_id="node_gqa", port_id="v"),
            ),
            Edge(
                id="e_gqa_to_out_proj",
                source=PortReference(node_id="node_gqa", port_id="output"),
                target=PortReference(node_id="node_out_proj", port_id="input"),
            ),
            Edge(
                id="e_out_proj_to_attn_out",
                source=PortReference(node_id="node_out_proj", port_id="output"),
                target=PortReference(node_id="node_attn_out", port_id="input"),
            ),
        ],
    )

    # 3. Subgraph: SwiGLU MLP
    g_mlp = GraphDefinition(
        id="graph_llama_mlp",
        name="Llama SwiGLU MLP",
        kind="module",
        interface=GraphInterface(
            inputs=[
                PortDefinition(id="input", display_name="Input", direction="input", required=True)
            ],
            outputs=[PortDefinition(id="output", display_name="Output", direction="output")],
        ),
        nodes=[
            NodeInstance(
                id="node_mlp_in",
                definition_id="builtin.module_input@1",
                display_name="MLP Input",
                properties={"name": "input"},
            ),
            NodeInstance(
                id="node_swiglu",
                definition_id="builtin.swiglu@1",
                display_name="SwiGLU",
                properties={
                    "n_embd": ConfigRefValue(key="n_embd"),
                },
            ),
            NodeInstance(
                id="node_mlp_out",
                definition_id="builtin.module_output@1",
                display_name="MLP Output",
                properties={"name": "output"},
            ),
        ],
        edges=[
            Edge(
                id="e_mlp_in_to_swiglu",
                source=PortReference(node_id="node_mlp_in", port_id="output"),
                target=PortReference(node_id="node_swiglu", port_id="input"),
            ),
            Edge(
                id="e_swiglu_to_out",
                source=PortReference(node_id="node_swiglu", port_id="output"),
                target=PortReference(node_id="node_mlp_out", port_id="input"),
            ),
        ],
    )

    # 4. Subgraph: Llama Transformer Block (RMSNorm -> Attn -> Add; RMSNorm -> SwiGLU -> Add)
    g_block = GraphDefinition(
        id="graph_llama_block",
        name="Llama Transformer Block",
        kind="module",
        interface=GraphInterface(
            inputs=[
                PortDefinition(id="input", display_name="Input", direction="input", required=True)
            ],
            outputs=[PortDefinition(id="output", display_name="Output", direction="output")],
        ),
        nodes=[
            NodeInstance(
                id="node_blk_in",
                definition_id="builtin.module_input@1",
                display_name="Block Input",
                properties={"name": "input"},
            ),
            NodeInstance(
                id="node_rms_1",
                definition_id="builtin.rmsnorm@1",
                display_name="RMSNorm 1",
                properties={
                    "normalized_shape": ConfigRefValue(key="n_embd"),
                },
            ),
            NodeInstance(
                id="node_attn_subgraph",
                definition_id="builtin.llama_attention@1",
                display_name="Llama Attention",
                properties={},
            ),
            NodeInstance(
                id="node_resid_add_1",
                definition_id="builtin.add@1",
                display_name="Residual Add 1",
            ),
            NodeInstance(
                id="node_rms_2",
                definition_id="builtin.rmsnorm@1",
                display_name="RMSNorm 2",
                properties={
                    "normalized_shape": ConfigRefValue(key="n_embd"),
                },
            ),
            NodeInstance(
                id="node_mlp_subgraph",
                definition_id="builtin.llama_mlp@1",
                display_name="Llama SwiGLU MLP",
                properties={},
            ),
            NodeInstance(
                id="node_resid_add_2",
                definition_id="builtin.add@1",
                display_name="Residual Add 2",
            ),
            NodeInstance(
                id="node_blk_out",
                definition_id="builtin.module_output@1",
                display_name="Block Output",
                properties={"name": "output"},
            ),
        ],
        edges=[
            Edge(
                id="e_blk_in_to_rms1",
                source=PortReference(node_id="node_blk_in", port_id="output"),
                target=PortReference(node_id="node_rms_1", port_id="input"),
            ),
            Edge(
                id="e_rms1_to_attn",
                source=PortReference(node_id="node_rms_1", port_id="output"),
                target=PortReference(node_id="node_attn_subgraph", port_id="input"),
            ),
            Edge(
                id="e_blk_in_to_add1",
                source=PortReference(node_id="node_blk_in", port_id="output"),
                target=PortReference(node_id="node_resid_add_1", port_id="a"),
            ),
            Edge(
                id="e_attn_to_add1",
                source=PortReference(node_id="node_attn_subgraph", port_id="output"),
                target=PortReference(node_id="node_resid_add_1", port_id="b"),
            ),
            Edge(
                id="e_add1_to_rms2",
                source=PortReference(node_id="node_resid_add_1", port_id="output"),
                target=PortReference(node_id="node_rms_2", port_id="input"),
            ),
            Edge(
                id="e_rms2_to_mlp",
                source=PortReference(node_id="node_rms_2", port_id="output"),
                target=PortReference(node_id="node_mlp_subgraph", port_id="input"),
            ),
            Edge(
                id="e_add1_to_add2",
                source=PortReference(node_id="node_resid_add_1", port_id="output"),
                target=PortReference(node_id="node_resid_add_2", port_id="a"),
            ),
            Edge(
                id="e_mlp_to_add2",
                source=PortReference(node_id="node_mlp_subgraph", port_id="output"),
                target=PortReference(node_id="node_resid_add_2", port_id="b"),
            ),
            Edge(
                id="e_add2_to_blk_out",
                source=PortReference(node_id="node_resid_add_2", port_id="output"),
                target=PortReference(node_id="node_blk_out", port_id="input"),
            ),
        ],
    )

    # 5. Subgraph: Transformer Stack (Repeat Module)
    g_stack = GraphDefinition(
        id="graph_llama_stack",
        name="Llama Transformer Stack",
        kind="repeat",
        repeat_count=ConfigRefValue(key="n_layer"),
        target_graph_id="graph_llama_block",
        interface=GraphInterface(
            inputs=[
                PortDefinition(id="input", display_name="Input", direction="input", required=True)
            ],
            outputs=[PortDefinition(id="output", display_name="Output", direction="output")],
        ),
        nodes=[],
        edges=[],
    )

    # 6. Root Graph: Llama
    g_root = GraphDefinition(
        id="graph_llama",
        name="Llama Tiny",
        kind="root",
        interface=GraphInterface(),
        nodes=[
            NodeInstance(
                id="node_token_ids",
                definition_id="builtin.token_input@1",
                display_name="Token IDs Input [B, T]",
                properties={"name": "token_ids"},
            ),
            NodeInstance(
                id="node_input_embeddings",
                definition_id="builtin.llama_input_embeddings@1",
                display_name="Llama Input Embeddings",
                properties={},
            ),
            NodeInstance(
                id="node_transformer_stack",
                definition_id="builtin.llama_stack@1",
                display_name="Llama Stack (h)",
                properties={},
            ),
            NodeInstance(
                id="node_rms_f",
                definition_id="builtin.rmsnorm@1",
                display_name="Final RMSNorm (rms_f)",
                properties={
                    "normalized_shape": ConfigRefValue(key="n_embd"),
                },
            ),
            NodeInstance(
                id="node_lm_head",
                definition_id="builtin.lm_head@1",
                display_name="LM Head (tied)",
                properties={
                    "in_features": ConfigRefValue(key="n_embd"),
                    "out_features": ConfigRefValue(key="vocab_size"),
                    "bias": False,
                },
            ),
            NodeInstance(
                id="node_logits_out",
                definition_id="builtin.logits_output@1",
                display_name="Logits Output",
                properties={"name": "logits"},
            ),
            NodeInstance(
                id="node_targets",
                definition_id="builtin.target_input@1",
                display_name="Target IDs [B, T]",
                properties={"name": "targets"},
            ),
            NodeInstance(
                id="node_loss",
                definition_id="builtin.cross_entropy_loss@1",
                display_name="CrossEntropy Loss",
            ),
            NodeInstance(
                id="node_loss_out",
                definition_id="builtin.loss_output@1",
                display_name="Loss Output",
                properties={"name": "loss"},
            ),
        ],
        edges=[
            Edge(
                id="e_tok_to_emb",
                source=PortReference(node_id="node_token_ids", port_id="output"),
                target=PortReference(node_id="node_input_embeddings", port_id="tokens"),
            ),
            Edge(
                id="e_emb_to_stack",
                source=PortReference(node_id="node_input_embeddings", port_id="output"),
                target=PortReference(node_id="node_transformer_stack", port_id="input"),
            ),
            Edge(
                id="e_stack_to_rmsf",
                source=PortReference(node_id="node_transformer_stack", port_id="output"),
                target=PortReference(node_id="node_rms_f", port_id="input"),
            ),
            Edge(
                id="e_rmsf_to_head",
                source=PortReference(node_id="node_rms_f", port_id="output"),
                target=PortReference(node_id="node_lm_head", port_id="input"),
            ),
            Edge(
                id="e_head_to_logits",
                source=PortReference(node_id="node_lm_head", port_id="logits"),
                target=PortReference(node_id="node_logits_out", port_id="input"),
            ),
            Edge(
                id="e_logits_to_loss",
                source=PortReference(node_id="node_lm_head", port_id="logits"),
                target=PortReference(node_id="node_loss", port_id="logits"),
            ),
            Edge(
                id="e_targets_to_loss",
                source=PortReference(node_id="node_targets", port_id="output"),
                target=PortReference(node_id="node_loss", port_id="targets"),
            ),
            Edge(
                id="e_loss_to_out",
                source=PortReference(node_id="node_loss", port_id="loss"),
                target=PortReference(node_id="node_loss_out", port_id="input"),
            ),
        ],
    )

    weight_bindings = [
        WeightBinding(
            source=WeightBindingEndpoint(node_id="node_tok_emb", parameter="weight"),
            target=WeightBindingEndpoint(node_id="node_lm_head", parameter="weight"),
            mode="share",
        )
    ]

    project = Project(
        project=ProjectMetadata(
            id="arch_26_llama_tiny",
            name="Arch 26: Llama Tiny",
            created_at="2026-09-02T00:00:00Z",
            updated_at="2026-09-02T00:00:00Z",
        ),
        model=ModelDefinition(
            root_graph_id="graph_llama",
            config=config,
            training=TrainingConfig(
                learning_rate=6e-4,
                weight_decay=0.1,
                grad_clip=1.0,
                batch_size=8,
                seed=1337,
                max_steps=40,
            ),
            graphs={
                "graph_llama_input_embeddings": g_input_emb,
                "graph_llama_attention": g_attention,
                "graph_llama_mlp": g_mlp,
                "graph_llama_block": g_block,
                "graph_llama_stack": g_stack,
                "graph_llama": g_root,
            },
            weight_bindings=weight_bindings,
        ),
        ui=UIState(
            open_graph_id="graph_llama",
        ),
    )

    return project
