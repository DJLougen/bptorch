"""nanoGPT Canonical Architecture Template Generator."""

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
    NodePosition,
    PortDefinition,
    PortReference,
    Project,
    ProjectMetadata,
    SafeExpression,
    UIState,
    Viewport,
    WeightBinding,
    WeightBindingEndpoint,
)


def create_nanogpt_template(
    block_size: int = 32,
    vocab_size: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    n_embd: int = 64,
    dropout: float = 0.0,
    bias: bool = True,
    attention_impl: str = "sdpa",
) -> Project:
    """
    Constructs the authoritative nanoGPT project specification matching karpathy/nanoGPT.
    Includes hierarchical subgraphs: Input Embeddings, Block, Attention, MLP, and Repeat Stack.
    """
    config: Dict[str, Any] = {
        "block_size": block_size,
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": dropout,
        "bias": bias,
        "attention_implementation": attention_impl,
    }

    # 1. Subgraph: Input Embeddings (Tokens + Positions -> Add -> Dropout)
    g_input_emb = GraphDefinition(
        id="graph_input_embeddings",
        name="Input Embeddings",
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
                id="node_pos_arange",
                definition_id="builtin.arange@1",
                display_name="Position Range [T]",
            ),
            NodeInstance(
                id="node_pos_emb",
                definition_id="builtin.embedding@1",
                display_name="Position Embedding (wpe)",
                properties={
                    "num_embeddings": ConfigRefValue(key="block_size"),
                    "embedding_dim": ConfigRefValue(key="n_embd"),
                },
            ),
            NodeInstance(
                id="node_emb_add",
                definition_id="builtin.add@1",
                display_name="Add Embeddings",
            ),
            NodeInstance(
                id="node_emb_drop",
                definition_id="builtin.dropout@1",
                display_name="Embedding Dropout",
                properties={"dropout": ConfigRefValue(key="dropout")},
            ),
            NodeInstance(
                id="node_emb_out",
                definition_id="builtin.module_output@1",
                display_name="Output",
                properties={"name": "output"},
            ),
        ],
        edges=[
            Edge(
                id="e_tok_in",
                source=PortReference(node_id="node_in_tokens", port_id="output"),
                target=PortReference(node_id="node_tok_emb", port_id="input"),
            ),
            Edge(
                id="e_pos_in",
                source=PortReference(node_id="node_in_tokens", port_id="output"),
                target=PortReference(node_id="node_pos_arange", port_id="sequence_tensor"),
            ),
            Edge(
                id="e_pos_emb",
                source=PortReference(node_id="node_pos_arange", port_id="positions"),
                target=PortReference(node_id="node_pos_emb", port_id="input"),
            ),
            Edge(
                id="e_add_tok",
                source=PortReference(node_id="node_tok_emb", port_id="output"),
                target=PortReference(node_id="node_emb_add", port_id="a"),
            ),
            Edge(
                id="e_add_pos",
                source=PortReference(node_id="node_pos_emb", port_id="output"),
                target=PortReference(node_id="node_emb_add", port_id="b"),
            ),
            Edge(
                id="e_drop_in",
                source=PortReference(node_id="node_emb_add", port_id="output"),
                target=PortReference(node_id="node_emb_drop", port_id="input"),
            ),
            Edge(
                id="e_emb_out",
                source=PortReference(node_id="node_emb_drop", port_id="output"),
                target=PortReference(node_id="node_emb_out", port_id="input"),
            ),
        ],
    )

    # 2. Subgraph: Causal Self-Attention
    attn_node_def = (
        "builtin.sdpa@1" if attention_impl == "sdpa" else "builtin.manual_causal_attention@1"
    )
    g_attention = GraphDefinition(
        id="graph_attention",
        name="Causal Self-Attention",
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
                id="node_qkv_proj",
                definition_id="builtin.linear@1",
                display_name="QKV Projection (c_attn)",
                properties={
                    "in_features": ConfigRefValue(key="n_embd"),
                    "out_features": ExpressionValue(
                        expression=SafeExpression(
                            op=ExpressionOp.MULTIPLY,
                            left=3,
                            right=ConfigRefValue(key="n_embd"),
                        )
                    ),
                    "bias": ConfigRefValue(key="bias"),
                },
            ),
            NodeInstance(
                id="node_split_qkv",
                definition_id="builtin.split_qkv@1",
                display_name="Split QKV",
                properties={"n_embd": ConfigRefValue(key="n_embd")},
            ),
            NodeInstance(
                id="node_split_q",
                definition_id="builtin.split_heads@1",
                display_name="Split Heads (Q)",
                properties={
                    "n_head": ConfigRefValue(key="n_head"),
                    "n_embd": ConfigRefValue(key="n_embd"),
                },
            ),
            NodeInstance(
                id="node_split_k",
                definition_id="builtin.split_heads@1",
                display_name="Split Heads (K)",
                properties={
                    "n_head": ConfigRefValue(key="n_head"),
                    "n_embd": ConfigRefValue(key="n_embd"),
                },
            ),
            NodeInstance(
                id="node_split_v",
                definition_id="builtin.split_heads@1",
                display_name="Split Heads (V)",
                properties={
                    "n_head": ConfigRefValue(key="n_head"),
                    "n_embd": ConfigRefValue(key="n_embd"),
                },
            ),
            NodeInstance(
                id="node_causal_attn",
                definition_id=attn_node_def,
                display_name="Causal Attention",
                properties={
                    "is_causal": True,
                    "dropout": ConfigRefValue(key="dropout"),
                },
            ),
            NodeInstance(
                id="node_merge_heads",
                definition_id="builtin.merge_heads@1",
                display_name="Merge Heads",
                properties={"n_embd": ConfigRefValue(key="n_embd")},
            ),
            NodeInstance(
                id="node_attn_c_proj",
                definition_id="builtin.linear@1",
                display_name="Output Projection (c_proj)",
                properties={
                    "in_features": ConfigRefValue(key="n_embd"),
                    "out_features": ConfigRefValue(key="n_embd"),
                    "bias": ConfigRefValue(key="bias"),
                },
            ),
            NodeInstance(
                id="node_attn_drop",
                definition_id="builtin.dropout@1",
                display_name="Residual Dropout",
                properties={"dropout": ConfigRefValue(key="dropout")},
            ),
            NodeInstance(
                id="node_attn_out",
                definition_id="builtin.module_output@1",
                display_name="Output",
                properties={"name": "output"},
            ),
        ],
        edges=[
            Edge(
                id="e_att_qkv",
                source=PortReference(node_id="node_attn_in", port_id="output"),
                target=PortReference(node_id="node_qkv_proj", port_id="input"),
            ),
            Edge(
                id="e_qkv_split",
                source=PortReference(node_id="node_qkv_proj", port_id="output"),
                target=PortReference(node_id="node_split_qkv", port_id="input"),
            ),
            Edge(
                id="e_q_heads",
                source=PortReference(node_id="node_split_qkv", port_id="q"),
                target=PortReference(node_id="node_split_q", port_id="input"),
            ),
            Edge(
                id="e_k_heads",
                source=PortReference(node_id="node_split_qkv", port_id="k"),
                target=PortReference(node_id="node_split_k", port_id="input"),
            ),
            Edge(
                id="e_v_heads",
                source=PortReference(node_id="node_split_qkv", port_id="v"),
                target=PortReference(node_id="node_split_v", port_id="input"),
            ),
            Edge(
                id="e_sdpa_q",
                source=PortReference(node_id="node_split_q", port_id="output"),
                target=PortReference(node_id="node_causal_attn", port_id="q"),
            ),
            Edge(
                id="e_sdpa_k",
                source=PortReference(node_id="node_split_k", port_id="output"),
                target=PortReference(node_id="node_causal_attn", port_id="k"),
            ),
            Edge(
                id="e_sdpa_v",
                source=PortReference(node_id="node_split_v", port_id="output"),
                target=PortReference(node_id="node_causal_attn", port_id="v"),
            ),
            Edge(
                id="e_merge_in",
                source=PortReference(node_id="node_causal_attn", port_id="output"),
                target=PortReference(node_id="node_merge_heads", port_id="input"),
            ),
            Edge(
                id="e_proj_in",
                source=PortReference(node_id="node_merge_heads", port_id="output"),
                target=PortReference(node_id="node_attn_c_proj", port_id="input"),
            ),
            Edge(
                id="e_att_drop",
                source=PortReference(node_id="node_attn_c_proj", port_id="output"),
                target=PortReference(node_id="node_attn_drop", port_id="input"),
            ),
            Edge(
                id="e_att_out",
                source=PortReference(node_id="node_attn_drop", port_id="output"),
                target=PortReference(node_id="node_attn_out", port_id="input"),
            ),
        ],
    )

    # 3. Subgraph: MLP (Linear C->4C -> GELU -> Linear 4C->C -> Dropout)
    g_mlp = GraphDefinition(
        id="graph_mlp",
        name="MLP",
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
                id="node_mlp_c_fc",
                definition_id="builtin.linear@1",
                display_name="Expansion Linear (c_fc)",
                properties={
                    "in_features": ConfigRefValue(key="n_embd"),
                    "out_features": ExpressionValue(
                        expression=SafeExpression(
                            op=ExpressionOp.MULTIPLY,
                            left=4,
                            right=ConfigRefValue(key="n_embd"),
                        )
                    ),
                    "bias": ConfigRefValue(key="bias"),
                },
            ),
            NodeInstance(
                id="node_mlp_gelu",
                definition_id="builtin.gelu@1",
                display_name="GELU",
                properties={"approximate": "none"},
            ),
            NodeInstance(
                id="node_mlp_c_proj",
                definition_id="builtin.linear@1",
                display_name="Projection Linear (c_proj)",
                properties={
                    "in_features": ExpressionValue(
                        expression=SafeExpression(
                            op=ExpressionOp.MULTIPLY,
                            left=4,
                            right=ConfigRefValue(key="n_embd"),
                        )
                    ),
                    "out_features": ConfigRefValue(key="n_embd"),
                    "bias": ConfigRefValue(key="bias"),
                },
            ),
            NodeInstance(
                id="node_mlp_drop",
                definition_id="builtin.dropout@1",
                display_name="MLP Dropout",
                properties={"dropout": ConfigRefValue(key="dropout")},
            ),
            NodeInstance(
                id="node_mlp_out",
                definition_id="builtin.module_output@1",
                display_name="Output",
                properties={"name": "output"},
            ),
        ],
        edges=[
            Edge(
                id="e_mlp_fc",
                source=PortReference(node_id="node_mlp_in", port_id="output"),
                target=PortReference(node_id="node_mlp_c_fc", port_id="input"),
            ),
            Edge(
                id="e_mlp_gelu",
                source=PortReference(node_id="node_mlp_c_fc", port_id="output"),
                target=PortReference(node_id="node_mlp_gelu", port_id="input"),
            ),
            Edge(
                id="e_mlp_proj",
                source=PortReference(node_id="node_mlp_gelu", port_id="output"),
                target=PortReference(node_id="node_mlp_c_proj", port_id="input"),
            ),
            Edge(
                id="e_mlp_drop",
                source=PortReference(node_id="node_mlp_c_proj", port_id="output"),
                target=PortReference(node_id="node_mlp_drop", port_id="input"),
            ),
            Edge(
                id="e_mlp_out",
                source=PortReference(node_id="node_mlp_drop", port_id="output"),
                target=PortReference(node_id="node_mlp_out", port_id="input"),
            ),
        ],
    )

    # 4. Subgraph: Transformer Block (LN1 -> Attn -> Residual Add 1; LN2 -> MLP -> Residual Add 2)
    g_block = GraphDefinition(
        id="graph_block",
        name="Transformer Block",
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
                id="node_ln_1",
                definition_id="builtin.layernorm@1",
                display_name="LayerNorm 1 (ln_1)",
                properties={
                    "normalized_shape": ConfigRefValue(key="n_embd"),
                    "bias": ConfigRefValue(key="bias"),
                },
            ),
            NodeInstance(
                id="node_attn_subgraph",
                definition_id="builtin.nanogpt_attention@1",
                display_name="Causal Self-Attention",
                properties={},
            ),
            NodeInstance(
                id="node_resid_add_1",
                definition_id="builtin.add@1",
                display_name="Residual Add 1",
            ),
            NodeInstance(
                id="node_ln_2",
                definition_id="builtin.layernorm@1",
                display_name="LayerNorm 2 (ln_2)",
                properties={
                    "normalized_shape": ConfigRefValue(key="n_embd"),
                    "bias": ConfigRefValue(key="bias"),
                },
            ),
            NodeInstance(
                id="node_mlp_subgraph",
                definition_id="builtin.nanogpt_mlp@1",
                display_name="MLP",
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
                display_name="Output",
                properties={"name": "output"},
            ),
        ],
        edges=[
            # Residual 1 path: input -> ln_1 -> attn -> add(input, attn)
            Edge(
                id="e_blk_ln1",
                source=PortReference(node_id="node_blk_in", port_id="output"),
                target=PortReference(node_id="node_ln_1", port_id="input"),
            ),
            Edge(
                id="e_blk_attn",
                source=PortReference(node_id="node_ln_1", port_id="output"),
                target=PortReference(node_id="node_attn_subgraph", port_id="input"),
            ),
            Edge(
                id="e_blk_add1_a",
                source=PortReference(node_id="node_blk_in", port_id="output"),
                target=PortReference(node_id="node_resid_add_1", port_id="a"),
            ),
            Edge(
                id="e_blk_add1_b",
                source=PortReference(node_id="node_attn_subgraph", port_id="output"),
                target=PortReference(node_id="node_resid_add_1", port_id="b"),
            ),
            # Residual 2 path: add1 -> ln_2 -> mlp -> add(add1, mlp)
            Edge(
                id="e_blk_ln2",
                source=PortReference(node_id="node_resid_add_1", port_id="output"),
                target=PortReference(node_id="node_ln_2", port_id="input"),
            ),
            Edge(
                id="e_blk_mlp",
                source=PortReference(node_id="node_ln_2", port_id="output"),
                target=PortReference(node_id="node_mlp_subgraph", port_id="input"),
            ),
            Edge(
                id="e_blk_add2_a",
                source=PortReference(node_id="node_resid_add_1", port_id="output"),
                target=PortReference(node_id="node_resid_add_2", port_id="a"),
            ),
            Edge(
                id="e_blk_add2_b",
                source=PortReference(node_id="node_mlp_subgraph", port_id="output"),
                target=PortReference(node_id="node_resid_add_2", port_id="b"),
            ),
            Edge(
                id="e_blk_out",
                source=PortReference(node_id="node_resid_add_2", port_id="output"),
                target=PortReference(node_id="node_blk_out", port_id="input"),
            ),
        ],
    )

    # 5. Subgraph: Transformer Stack (Repeat Module)
    g_stack = GraphDefinition(
        id="graph_stack",
        name="Transformer Stack",
        kind="repeat",
        repeat_count=ConfigRefValue(key="n_layer"),
        target_graph_id="graph_block",
        interface=GraphInterface(
            inputs=[
                PortDefinition(id="input", display_name="Input", direction="input", required=True)
            ],
            outputs=[PortDefinition(id="output", display_name="Output", direction="output")],
        ),
        nodes=[],
        edges=[],
    )

    # 6. Root Graph: nanoGPT
    g_root = GraphDefinition(
        id="graph_gpt",
        name="nanoGPT",
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
                definition_id="builtin.nanogpt_input_embeddings@1",
                display_name="Input Embeddings",
                properties={},
            ),
            NodeInstance(
                id="node_transformer_stack",
                definition_id="builtin.nanogpt_stack@1",
                display_name="Transformer Stack (h)",
                properties={},
            ),
            NodeInstance(
                id="node_ln_f",
                definition_id="builtin.layernorm@1",
                display_name="Final LayerNorm (ln_f)",
                properties={
                    "normalized_shape": ConfigRefValue(key="n_embd"),
                    "bias": ConfigRefValue(key="bias"),
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
                display_name="Targets Input [B, T]",
                properties={"name": "targets"},
            ),
            NodeInstance(
                id="node_cross_entropy",
                definition_id="builtin.cross_entropy_loss@1",
                display_name="Cross-Entropy Loss",
                properties={"ignore_index": -1},
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
                id="e_root_tok_emb",
                source=PortReference(node_id="node_token_ids", port_id="output"),
                target=PortReference(node_id="node_input_embeddings", port_id="tokens"),
            ),
            Edge(
                id="e_root_emb_stack",
                source=PortReference(node_id="node_input_embeddings", port_id="output"),
                target=PortReference(node_id="node_transformer_stack", port_id="input"),
            ),
            Edge(
                id="e_root_stack_lnf",
                source=PortReference(node_id="node_transformer_stack", port_id="output"),
                target=PortReference(node_id="node_ln_f", port_id="input"),
            ),
            Edge(
                id="e_root_lnf_head",
                source=PortReference(node_id="node_ln_f", port_id="output"),
                target=PortReference(node_id="node_lm_head", port_id="input"),
            ),
            Edge(
                id="e_root_head_logits",
                source=PortReference(node_id="node_lm_head", port_id="logits"),
                target=PortReference(node_id="node_logits_out", port_id="input"),
            ),
            Edge(
                id="e_root_loss_logits",
                source=PortReference(node_id="node_lm_head", port_id="logits"),
                target=PortReference(node_id="node_cross_entropy", port_id="logits"),
            ),
            Edge(
                id="e_root_loss_targets",
                source=PortReference(node_id="node_targets", port_id="output"),
                target=PortReference(node_id="node_cross_entropy", port_id="targets"),
            ),
            Edge(
                id="e_root_loss_out",
                source=PortReference(node_id="node_cross_entropy", port_id="loss"),
                target=PortReference(node_id="node_loss_out", port_id="input"),
            ),
        ],
    )
    # 7. Subgraph: Training Event Graph (Dual-Flow Loop & Optimization)
    g_training_event = GraphDefinition(
        id="graph_training_event",
        name="Training Event Graph",
        kind="training_event",
        interface=GraphInterface(inputs=[], outputs=[]),
        nodes=[
            NodeInstance(
                id="node_evt_begin",
                definition_id="builtin.event_on_train_begin@1",
                display_name="Event OnTrainBegin",
            ),
            NodeInstance(
                id="node_train_seq",
                definition_id="builtin.training_sequence@1",
                display_name="Training Sequence",
            ),
            NodeInstance(
                id="node_dataset_src",
                definition_id="builtin.dataset_source@1",
                display_name="Dataset Source",
            ),
            NodeInstance(
                id="node_train_dataloader",
                definition_id="builtin.dataloader@1",
                display_name="DataLoader",
            ),
            NodeInstance(
                id="node_epoch_loop",
                definition_id="builtin.epoch_loop@1",
                display_name="Epoch Loop",
            ),
            NodeInstance(
                id="node_batch_loop",
                definition_id="builtin.batch_loop@1",
                display_name="Batch Loop",
            ),
            NodeInstance(
                id="node_backward",
                definition_id="builtin.backward@1",
                display_name="Backward Autograd",
            ),
            NodeInstance(
                id="node_clip_grad",
                definition_id="builtin.clip_gradients@1",
                display_name="Clip Gradients",
            ),
            NodeInstance(
                id="node_opt_step",
                definition_id="builtin.optimizer_step@1",
                display_name="Optimizer Step",
            ),
            NodeInstance(
                id="node_lr_sched",
                definition_id="builtin.cosine_annealing_lr@1",
                display_name="Cosine Decay LR",
            ),
            NodeInstance(
                id="node_zero_grad",
                definition_id="builtin.zero_grad@1",
                display_name="Zero Grad",
            ),
        ],
        edges=[
            Edge(
                id="e_evt_seq",
                source=PortReference(node_id="node_evt_begin", port_id="exec_out"),
                target=PortReference(node_id="node_train_seq", port_id="exec_in"),
            ),
            Edge(
                id="e_seq_epoch",
                source=PortReference(node_id="node_train_seq", port_id="then_0"),
                target=PortReference(node_id="node_epoch_loop", port_id="exec_in"),
            ),
            Edge(
                id="e_epoch_batch",
                source=PortReference(node_id="node_epoch_loop", port_id="loop_body"),
                target=PortReference(node_id="node_batch_loop", port_id="exec_in"),
            ),
            Edge(
                id="e_batch_back",
                source=PortReference(node_id="node_batch_loop", port_id="loop_body"),
                target=PortReference(node_id="node_backward", port_id="exec_in"),
            ),
            Edge(
                id="e_back_clip",
                source=PortReference(node_id="node_backward", port_id="exec_out"),
                target=PortReference(node_id="node_clip_grad", port_id="exec_in"),
            ),
            Edge(
                id="e_clip_opt",
                source=PortReference(node_id="node_clip_grad", port_id="exec_out"),
                target=PortReference(node_id="node_opt_step", port_id="exec_in"),
            ),
            Edge(
                id="e_opt_sched",
                source=PortReference(node_id="node_opt_step", port_id="exec_out"),
                target=PortReference(node_id="node_lr_sched", port_id="exec_in"),
            ),
            Edge(
                id="e_sched_zero",
                source=PortReference(node_id="node_lr_sched", port_id="exec_out"),
                target=PortReference(node_id="node_zero_grad", port_id="exec_in"),
            ),
            Edge(
                id="e_dataset_dl",
                source=PortReference(node_id="node_dataset_src", port_id="dataset"),
                target=PortReference(node_id="node_train_dataloader", port_id="dataset"),
            ),
        ],
    )

    # Tied weights binding: wte.weight == lm_head.weight
    weight_bindings = [
        WeightBinding(
            source=WeightBindingEndpoint(
                node_id="node_input_embeddings/node_tok_emb", parameter="weight"
            ),
            target=WeightBindingEndpoint(node_id="node_lm_head", parameter="weight"),
            mode="share",
        )
    ]

    project = Project(
        project=ProjectMetadata(
            id="nanogpt_default",
            name="nanoGPT Architecture",
            created_at="2026-08-18T00:00:00Z",
            updated_at="2026-08-18T00:00:00Z",
        ),
        model=ModelDefinition(
            root_graph_id="graph_gpt",
            config=config,
            graphs={
                "graph_gpt": g_root,
                "graph_input_embeddings": g_input_emb,
                "graph_attention": g_attention,
                "graph_mlp": g_mlp,
                "graph_block": g_block,
                "graph_stack": g_stack,
                "graph_training_event": g_training_event,
            },
            weight_bindings=weight_bindings,
        ),
        ui=UIState(
            graph_viewports={
                "graph_gpt": Viewport(x=0, y=0, zoom=1.0),
                "graph_input_embeddings": Viewport(x=0, y=0, zoom=1.0),
                "graph_attention": Viewport(x=0, y=0, zoom=1.0),
                "graph_mlp": Viewport(x=0, y=0, zoom=1.0),
                "graph_block": Viewport(x=0, y=0, zoom=1.0),
                "graph_stack": Viewport(x=0, y=0, zoom=1.0),
                "graph_training_event": Viewport(x=0, y=0, zoom=1.0),
            },
            node_positions={
                "graph_gpt": {
                    "node_token_ids": NodePosition(x=100, y=200),
                    "node_input_embeddings": NodePosition(x=350, y=200),
                    "node_transformer_stack": NodePosition(x=650, y=200),
                    "node_ln_f": NodePosition(x=950, y=200),
                    "node_lm_head": NodePosition(x=1200, y=200),
                    "node_logits_out": NodePosition(x=1450, y=150),
                    "node_targets": NodePosition(x=1200, y=350),
                    "node_cross_entropy": NodePosition(x=1450, y=300),
                    "node_loss_out": NodePosition(x=1700, y=300),
                },
                "graph_training_event": {
                    "node_evt_begin": NodePosition(x=50, y=150),
                    "node_train_seq": NodePosition(x=250, y=150),
                    "node_dataset_src": NodePosition(x=50, y=350),
                    "node_train_dataloader": NodePosition(x=250, y=350),
                    "node_epoch_loop": NodePosition(x=500, y=150),
                    "node_batch_loop": NodePosition(x=750, y=150),
                    "node_backward": NodePosition(x=1000, y=150),
                    "node_clip_grad": NodePosition(x=1250, y=150),
                    "node_opt_step": NodePosition(x=1500, y=150),
                    "node_lr_sched": NodePosition(x=1750, y=150),
                    "node_zero_grad": NodePosition(x=2000, y=150),
                },
            },
            open_graph_id="graph_gpt",
        ),
    )

    return project
