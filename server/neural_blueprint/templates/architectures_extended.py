"""Extended architecture generators (11–25) showcasing blueprint diversity."""

from typing import Any, Dict, List, Tuple

from neural_blueprint.ir.models import (
    ConfigRefValue,
    Edge,
    GraphDefinition,
    GraphInterface,
    ModelDefinition,
    NodeInstance,
    PortReference,
    Project,
    ProjectMetadata,
    TrainingConfig,
    UIState,
    WeightBinding,
    WeightBindingEndpoint,
)
from neural_blueprint.templates.linear_mlp import create_linear_mlp_template
from neural_blueprint.templates.nanogpt import create_nanogpt_template


def _training(**kwargs) -> TrainingConfig:
  defaults = dict(
      device="cpu",
      precision="fp32",
      learning_rate=1e-3,
      weight_decay=0.01,
      grad_clip=1.0,
      batch_size=8,
      seed=1337,
      max_steps=40,
  )
  defaults.update(kwargs)
  return TrainingConfig(**defaults)


def _mlp_graph(
    graph_id: str,
    name: str,
    config: Dict[str, Any],
    nodes: List[NodeInstance],
    edges: List[Edge],
    project_id: str,
    project_name: str,
    training: TrainingConfig,
) -> Project:
    g_root = GraphDefinition(
        id=graph_id,
        name=name,
        kind="root",
        interface=GraphInterface(),
        nodes=nodes,
        edges=edges,
    )
    return Project(
        project=ProjectMetadata(
            id=project_id,
            name=project_name,
            created_at="2026-08-30T00:00:00Z",
            updated_at="2026-08-30T00:00:00Z",
        ),
        model=ModelDefinition(
            root_graph_id=graph_id,
            config=config,
            training=training,
            graphs={graph_id: g_root},
        ),
        ui=UIState(open_graph_id=graph_id),
    )


# --- Architecture 11: ReLU Classifier MLP ---
def create_arch_11_relu_classifier() -> Project:
    p = create_linear_mlp_template(in_features=48, hidden_features=128, activation="relu")
    p.project.id = "arch_11_relu_classifier"
    p.project.name = "Arch 11: ReLU Classifier MLP"
    p.model.training = _training(learning_rate=8e-3, max_steps=45, seed=111)
    return p


# --- Architecture 12: Dropout-Regularized MLP ---
def create_arch_12_dropout_mlp() -> Project:
    config = {"in_dim": 64, "hidden": 128, "out_dim": 32, "dropout": 0.25}
    nodes = [
        NodeInstance(id="node_in", definition_id="builtin.tensor_input@1", display_name="Input", properties={"name": "input"}),
        NodeInstance(id="node_fc1", definition_id="builtin.linear@1", display_name="FC1", properties={"in_features": ConfigRefValue(key="in_dim"), "out_features": ConfigRefValue(key="hidden")}),
        NodeInstance(id="node_gelu", definition_id="builtin.gelu@1", display_name="GELU"),
        NodeInstance(id="node_drop", definition_id="builtin.dropout@1", display_name="Dropout", properties={"p": ConfigRefValue(key="dropout")}),
        NodeInstance(id="node_fc2", definition_id="builtin.linear@1", display_name="FC2", properties={"in_features": ConfigRefValue(key="hidden"), "out_features": ConfigRefValue(key="out_dim")}),
        NodeInstance(id="node_out", definition_id="builtin.graph_output@1", display_name="Output", properties={"name": "output"}),
    ]
    edges = [
        Edge(id="e1", source=PortReference(node_id="node_in", port_id="output"), target=PortReference(node_id="node_fc1", port_id="input")),
        Edge(id="e2", source=PortReference(node_id="node_fc1", port_id="output"), target=PortReference(node_id="node_gelu", port_id="input")),
        Edge(id="e3", source=PortReference(node_id="node_gelu", port_id="output"), target=PortReference(node_id="node_drop", port_id="input")),
        Edge(id="e4", source=PortReference(node_id="node_drop", port_id="output"), target=PortReference(node_id="node_fc2", port_id="input")),
        Edge(id="e5", source=PortReference(node_id="node_fc2", port_id="output"), target=PortReference(node_id="node_out", port_id="input")),
    ]
    return _mlp_graph("graph_dropout_mlp", "Dropout MLP", config, nodes, edges, "arch_12_dropout_mlp", "Arch 12: Dropout MLP", _training(learning_rate=5e-3, seed=212))


# --- Architecture 13: Deep 5-Layer MLP Tower ---
def create_arch_13_deep_mlp_tower() -> Project:
    config = {"width": 48, "depth": 5, "in_features": 48, "n_embd": 48}
    nodes = [
        NodeInstance(id="node_in", definition_id="builtin.tensor_input@1", display_name="Input", properties={"name": "input"}),
    ]
    edges: List[Edge] = []
    prev = "node_in"
    for i in range(1, 6):
        nid = f"node_l{i}"
        aid = f"node_a{i}"
        nodes.append(NodeInstance(id=nid, definition_id="builtin.linear@1", display_name=f"Linear {i}", properties={"in_features": ConfigRefValue(key="width"), "out_features": ConfigRefValue(key="width")}))
        nodes.append(NodeInstance(id=aid, definition_id="builtin.gelu@1", display_name=f"GELU {i}"))
        edges.append(Edge(id=f"e{i}a", source=PortReference(node_id=prev, port_id="output"), target=PortReference(node_id=nid, port_id="input")))
        edges.append(Edge(id=f"e{i}b", source=PortReference(node_id=nid, port_id="output"), target=PortReference(node_id=aid, port_id="input")))
        prev = aid
    nodes.append(NodeInstance(id="node_out", definition_id="builtin.graph_output@1", display_name="Output", properties={"name": "output"}))
    edges.append(Edge(id="e_out", source=PortReference(node_id=prev, port_id="output"), target=PortReference(node_id="node_out", port_id="input")))
    return _mlp_graph("graph_deep_tower", "Deep MLP Tower", config, nodes, edges, "arch_13_deep_tower", "Arch 13: Deep MLP Tower", _training(learning_rate=2e-3, max_steps=50, seed=313))


# --- Architecture 14: Wide-and-Deep Fusion Network ---
def create_arch_14_wide_and_deep() -> Project:
    config = {"in_dim": 32, "wide_dim": 16, "deep_hidden": 64, "out_dim": 16}
    nodes = [
        NodeInstance(id="node_in", definition_id="builtin.tensor_input@1", display_name="Input", properties={"name": "input"}),
        NodeInstance(id="node_wide", definition_id="builtin.linear@1", display_name="Wide Path", properties={"in_features": ConfigRefValue(key="in_dim"), "out_features": ConfigRefValue(key="wide_dim")}),
        NodeInstance(id="node_deep1", definition_id="builtin.linear@1", display_name="Deep FC1", properties={"in_features": ConfigRefValue(key="in_dim"), "out_features": ConfigRefValue(key="deep_hidden")}),
        NodeInstance(id="node_deep_act", definition_id="builtin.silu@1", display_name="SiLU"),
        NodeInstance(id="node_deep2", definition_id="builtin.linear@1", display_name="Deep FC2", properties={"in_features": ConfigRefValue(key="deep_hidden"), "out_features": ConfigRefValue(key="wide_dim")}),
        NodeInstance(id="node_fuse", definition_id="builtin.add@1", display_name="Fuse Wide+Deep"),
        NodeInstance(id="node_out_proj", definition_id="builtin.linear@1", display_name="Output Projection", properties={"in_features": ConfigRefValue(key="wide_dim"), "out_features": ConfigRefValue(key="out_dim")}),
        NodeInstance(id="node_out", definition_id="builtin.graph_output@1", display_name="Output", properties={"name": "output"}),
    ]
    edges = [
        Edge(id="e1", source=PortReference(node_id="node_in", port_id="output"), target=PortReference(node_id="node_wide", port_id="input")),
        Edge(id="e2", source=PortReference(node_id="node_in", port_id="output"), target=PortReference(node_id="node_deep1", port_id="input")),
        Edge(id="e3", source=PortReference(node_id="node_deep1", port_id="output"), target=PortReference(node_id="node_deep_act", port_id="input")),
        Edge(id="e4", source=PortReference(node_id="node_deep_act", port_id="output"), target=PortReference(node_id="node_deep2", port_id="input")),
        Edge(id="e5a", source=PortReference(node_id="node_wide", port_id="output"), target=PortReference(node_id="node_fuse", port_id="a")),
        Edge(id="e5b", source=PortReference(node_id="node_deep2", port_id="output"), target=PortReference(node_id="node_fuse", port_id="b")),
        Edge(id="e6", source=PortReference(node_id="node_fuse", port_id="output"), target=PortReference(node_id="node_out_proj", port_id="input")),
        Edge(id="e7", source=PortReference(node_id="node_out_proj", port_id="output"), target=PortReference(node_id="node_out", port_id="input")),
    ]
    return _mlp_graph("graph_wide_deep", "Wide-and-Deep", config, nodes, edges, "arch_14_wide_deep", "Arch 14: Wide-and-Deep Network", _training(learning_rate=3e-3, seed=414))


# --- Architecture 15: Tiny Tied-Embedding Language Model ---
def create_arch_15_tied_embedding_lm() -> Project:
    config = {"vocab_size": 48, "n_embd": 24}
    nodes = [
        NodeInstance(id="node_tokens", definition_id="builtin.token_input@1", display_name="Tokens", properties={"name": "token_ids"}),
        NodeInstance(id="node_emb", definition_id="builtin.embedding@1", display_name="Token Embedding", properties={"num_embeddings": ConfigRefValue(key="vocab_size"), "embedding_dim": ConfigRefValue(key="n_embd")}),
        NodeInstance(id="node_lm", definition_id="builtin.lm_head@1", display_name="LM Head", properties={"in_features": ConfigRefValue(key="n_embd"), "out_features": ConfigRefValue(key="vocab_size"), "bias": False}),
        NodeInstance(id="node_logits", definition_id="builtin.logits_output@1", display_name="Logits", properties={"name": "logits"}),
    ]
    edges = [
        Edge(id="e1", source=PortReference(node_id="node_tokens", port_id="output"), target=PortReference(node_id="node_emb", port_id="input")),
        Edge(id="e2", source=PortReference(node_id="node_emb", port_id="output"), target=PortReference(node_id="node_lm", port_id="input")),
        Edge(id="e3", source=PortReference(node_id="node_lm", port_id="logits"), target=PortReference(node_id="node_logits", port_id="input")),
    ]
    g = GraphDefinition(id="graph_tied_lm", name="Tied Embedding LM", kind="root", interface=GraphInterface(), nodes=nodes, edges=edges)
    return Project(
        project=ProjectMetadata(id="arch_15_tied_lm", name="Arch 15: Tied-Embedding LM", created_at="2026-08-30T00:00:00Z", updated_at="2026-08-30T00:00:00Z"),
        model=ModelDefinition(
            root_graph_id="graph_tied_lm",
            config=config,
            training=_training(learning_rate=6e-4, max_steps=40, seed=515),
            graphs={"graph_tied_lm": g},
            weight_bindings=[
                WeightBinding(
                    source=WeightBindingEndpoint(node_id="node_emb", parameter="weight"),
                    target=WeightBindingEndpoint(node_id="node_lm", parameter="weight"),
                    mode="share",
                )
            ],
        ),
        ui=UIState(open_graph_id="graph_tied_lm"),
    )


# --- Architecture 16: Warmup Scheduler Training Pipeline ---
def create_arch_16_warmup_pipeline() -> Project:
    return _dual_flow_variant(
        project_id="arch_16_warmup_pipeline",
        project_name="Arch 16: Warmup Scheduler Pipeline",
        scheduler_def="builtin.linear_warmup_scheduler@1",
        scheduler_props={"warmup_steps": 15, "total_steps": 50},
        training=_training(learning_rate=2e-3, max_steps=50, seed=616),
    )


# --- Architecture 17: Step-LR Decay Pipeline ---
def create_arch_17_step_lr_pipeline() -> Project:
    return _dual_flow_variant(
        project_id="arch_17_step_lr_pipeline",
        project_name="Arch 17: Step-LR Decay Pipeline",
        scheduler_def="builtin.step_lr@1",
        scheduler_props={"step_size": 10, "gamma": 0.5},
        training=_training(learning_rate=4e-3, max_steps=45, seed=717),
    )


# --- Architecture 18: Pre-LayerNorm MLP Block ---
def create_arch_18_prenorm_mlp() -> Project:
    config = {"dim": 64, "hidden": 128, "in_features": 64, "n_embd": 64}
    nodes = [
        NodeInstance(id="node_in", definition_id="builtin.tensor_input@1", display_name="Input", properties={"name": "input"}),
        NodeInstance(id="node_ln1", definition_id="builtin.layernorm@1", display_name="Pre-Norm 1", properties={"normalized_shape": ConfigRefValue(key="dim")}),
        NodeInstance(id="node_fc1", definition_id="builtin.linear@1", display_name="FC1", properties={"in_features": ConfigRefValue(key="dim"), "out_features": ConfigRefValue(key="hidden")}),
        NodeInstance(id="node_gelu", definition_id="builtin.gelu@1", display_name="GELU"),
        NodeInstance(id="node_ln2", definition_id="builtin.layernorm@1", display_name="Pre-Norm 2", properties={"normalized_shape": ConfigRefValue(key="hidden")}),
        NodeInstance(id="node_fc2", definition_id="builtin.linear@1", display_name="FC2", properties={"in_features": ConfigRefValue(key="hidden"), "out_features": ConfigRefValue(key="dim")}),
        NodeInstance(id="node_out", definition_id="builtin.graph_output@1", display_name="Output", properties={"name": "output"}),
    ]
    edges = [
        Edge(id="e1", source=PortReference(node_id="node_in", port_id="output"), target=PortReference(node_id="node_ln1", port_id="input")),
        Edge(id="e2", source=PortReference(node_id="node_ln1", port_id="output"), target=PortReference(node_id="node_fc1", port_id="input")),
        Edge(id="e3", source=PortReference(node_id="node_fc1", port_id="output"), target=PortReference(node_id="node_gelu", port_id="input")),
        Edge(id="e4", source=PortReference(node_id="node_gelu", port_id="output"), target=PortReference(node_id="node_ln2", port_id="input")),
        Edge(id="e5", source=PortReference(node_id="node_ln2", port_id="output"), target=PortReference(node_id="node_fc2", port_id="input")),
        Edge(id="e6", source=PortReference(node_id="node_fc2", port_id="output"), target=PortReference(node_id="node_out", port_id="input")),
    ]
    return _mlp_graph("graph_prenorm_mlp", "Pre-Norm MLP", config, nodes, edges, "arch_18_prenorm_mlp", "Arch 18: Pre-LayerNorm MLP", _training(seed=818))


# --- Architecture 19: Explicit Residual Add Block ---
def create_arch_19_residual_add_mlp() -> Project:
    config = {"dim": 48, "in_features": 48, "n_embd": 48}
    nodes = [
        NodeInstance(id="node_in", definition_id="builtin.tensor_input@1", display_name="Input", properties={"name": "input"}),
        NodeInstance(id="node_fc1", definition_id="builtin.linear@1", display_name="Residual FC", properties={"in_features": ConfigRefValue(key="dim"), "out_features": ConfigRefValue(key="dim")}),
        NodeInstance(id="node_gelu", definition_id="builtin.gelu@1", display_name="GELU"),
        NodeInstance(id="node_add", definition_id="builtin.add@1", display_name="Residual Add"),
        NodeInstance(id="node_out", definition_id="builtin.graph_output@1", display_name="Output", properties={"name": "output"}),
    ]
    edges = [
        Edge(id="e1", source=PortReference(node_id="node_in", port_id="output"), target=PortReference(node_id="node_fc1", port_id="input")),
        Edge(id="e2", source=PortReference(node_id="node_fc1", port_id="output"), target=PortReference(node_id="node_gelu", port_id="input")),
        Edge(id="e3a", source=PortReference(node_id="node_in", port_id="output"), target=PortReference(node_id="node_add", port_id="a")),
        Edge(id="e3b", source=PortReference(node_id="node_gelu", port_id="output"), target=PortReference(node_id="node_add", port_id="b")),
        Edge(id="e4", source=PortReference(node_id="node_add", port_id="output"), target=PortReference(node_id="node_out", port_id="input")),
    ]
    return _mlp_graph("graph_res_add", "Residual Add MLP", config, nodes, edges, "arch_19_residual_add", "Arch 19: Residual Add MLP", _training(learning_rate=4e-3, seed=919))


# --- Architecture 20: Binary Sequence Classifier ---
def create_arch_20_binary_classifier() -> Project:
    config = {"vocab_size": 32, "n_embd": 16, "num_classes": 2}
    nodes = [
        NodeInstance(id="node_tokens", definition_id="builtin.token_input@1", display_name="Tokens", properties={"name": "token_ids"}),
        NodeInstance(id="node_emb", definition_id="builtin.embedding@1", display_name="Embedding", properties={"num_embeddings": ConfigRefValue(key="vocab_size"), "embedding_dim": ConfigRefValue(key="n_embd")}),
        NodeInstance(id="node_cls", definition_id="builtin.linear@1", display_name="Classifier Head", properties={"in_features": ConfigRefValue(key="n_embd"), "out_features": ConfigRefValue(key="num_classes")}),
        NodeInstance(id="node_out", definition_id="builtin.graph_output@1", display_name="Class Logits", properties={"name": "cls_logits"}),
    ]
    edges = [
        Edge(id="e1", source=PortReference(node_id="node_tokens", port_id="output"), target=PortReference(node_id="node_emb", port_id="input")),
        Edge(id="e2", source=PortReference(node_id="node_emb", port_id="output"), target=PortReference(node_id="node_cls", port_id="input")),
        Edge(id="e3", source=PortReference(node_id="node_cls", port_id="output"), target=PortReference(node_id="node_out", port_id="input")),
    ]
    return _mlp_graph("graph_binary_cls", "Binary Classifier", config, nodes, edges, "arch_20_binary_cls", "Arch 20: Binary Sequence Classifier", _training(learning_rate=7e-4, seed=1020))


# --- Architecture 21: High-Dropout nanoGPT ---
def create_arch_21_dropout_transformer() -> Project:
    p = create_nanogpt_template(block_size=12, vocab_size=48, n_layer=2, n_head=3, n_embd=24, dropout=0.2, bias=True)
    p.project.id = "arch_21_dropout_gpt"
    p.project.name = "Arch 21: High-Dropout Transformer"
    p.model.training = _training(learning_rate=4e-4, weight_decay=0.05, max_steps=35, seed=1121)
    return p


# --- Architecture 22: BF16 Precision nanoGPT Micro ---
def create_arch_22_bf16_nanogpt() -> Project:
    p = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16, dropout=0.0, bias=True)
    p.project.id = "arch_22_bf16_gpt"
    p.project.name = "Arch 22: BF16 nanoGPT Micro"
    p.model.training = _training(precision="bf16", learning_rate=6e-4, max_steps=30, seed=1222)
    return p


# --- Architecture 23: Single-Block Causal GPT ---
def create_arch_23_single_block_gpt() -> Project:
    p = create_nanogpt_template(block_size=16, vocab_size=64, n_layer=1, n_head=4, n_embd=32, dropout=0.05, bias=False)
    p.project.id = "arch_23_single_block"
    p.project.name = "Arch 23: Single-Block Causal GPT"
    p.model.training = _training(learning_rate=8e-4, max_steps=40, seed=1323)
    return p


# --- Architecture 24: SiLU Deep Feedforward ---
def create_arch_24_silu_deep_ffn() -> Project:
    p = create_linear_mlp_template(in_features=32, hidden_features=96, activation="silu")
    p.project.id = "arch_24_silu_ffn"
    p.project.name = "Arch 24: SiLU Deep Feedforward"
    p.model.training = _training(learning_rate=6e-3, max_steps=45, seed=1424)
    return p


# --- Architecture 25: Metric-Aware Dual-Flow Pipeline ---
def create_arch_25_metric_pipeline() -> Project:
    return _dual_flow_variant(
        project_id="arch_25_metric_pipeline",
        project_name="Arch 25: Metric Logger Pipeline",
        extra_nodes=[
            NodeInstance(
                id="node_metric",
                definition_id="builtin.metric_logger@1",
                display_name="Metric Logger",
                properties={"log_interval": 5},
            ),
        ],
        extra_exec_edges=[
            Edge(
                id="e_exec_metric",
                source=PortReference(node_id="node_lr_sched", port_id="exec_out"),
                target=PortReference(node_id="node_metric", port_id="exec_in"),
            ),
        ],
        training=_training(learning_rate=1e-3, max_steps=50, seed=1525),
    )


def _dual_flow_variant(
    project_id: str,
    project_name: str,
    training: TrainingConfig,
    scheduler_def: str = "builtin.cosine_annealing_lr@1",
    scheduler_props: Dict[str, Any] | None = None,
    extra_nodes: List[NodeInstance] | None = None,
    extra_exec_edges: List[Edge] | None = None,
) -> Project:
    scheduler_props = scheduler_props or {"warmup_steps": 10, "total_steps": 50}
    nodes = [
        NodeInstance(id="node_dataset", definition_id="builtin.dataset_source@1", display_name="Dataset", properties={"num_samples": 400, "vocab_size": 32, "sequence_length": 8}),
        NodeInstance(id="node_dataloader", definition_id="builtin.dataloader@1", display_name="DataLoader", properties={"batch_size": 8}),
        NodeInstance(id="node_emb", definition_id="builtin.embedding@1", display_name="Embedding", properties={"num_embeddings": 32, "embedding_dim": 16}),
        NodeInstance(id="node_forward", definition_id="builtin.linear@1", display_name="Forward", properties={"in_features": 16, "out_features": 32}),
        NodeInstance(id="node_loss", definition_id="builtin.cross_entropy_loss@1", display_name="Loss"),
        NodeInstance(id="node_backward", definition_id="builtin.backward@1", display_name="Backward"),
        NodeInstance(id="node_clip_grad", definition_id="builtin.clip_gradients@1", display_name="Clip", properties={"max_norm": 1.0}),
        NodeInstance(id="node_opt_step", definition_id="builtin.optimizer_step@1", display_name="Optimizer Step"),
        NodeInstance(id="node_lr_sched", definition_id=scheduler_def, display_name="LR Scheduler", properties=scheduler_props),
        NodeInstance(id="node_zero_grad", definition_id="builtin.zero_grad@1", display_name="Zero Grad"),
    ]
    if extra_nodes:
        nodes.extend(extra_nodes)

    edges = [
        Edge(id="e_exec_1", source=PortReference(node_id="node_dataset", port_id="exec_out"), target=PortReference(node_id="node_dataloader", port_id="exec_in")),
        Edge(id="e_exec_2", source=PortReference(node_id="node_dataloader", port_id="exec_out"), target=PortReference(node_id="node_emb", port_id="exec_in")),
        Edge(id="e_exec_2b", source=PortReference(node_id="node_emb", port_id="exec_out"), target=PortReference(node_id="node_forward", port_id="exec_in")),
        Edge(id="e_exec_3", source=PortReference(node_id="node_forward", port_id="exec_out"), target=PortReference(node_id="node_loss", port_id="exec_in")),
        Edge(id="e_exec_4", source=PortReference(node_id="node_loss", port_id="exec_out"), target=PortReference(node_id="node_backward", port_id="exec_in")),
        Edge(id="e_exec_5", source=PortReference(node_id="node_backward", port_id="exec_out"), target=PortReference(node_id="node_clip_grad", port_id="exec_in")),
        Edge(id="e_exec_6", source=PortReference(node_id="node_clip_grad", port_id="exec_out"), target=PortReference(node_id="node_opt_step", port_id="exec_in")),
        Edge(id="e_exec_7", source=PortReference(node_id="node_opt_step", port_id="exec_out"), target=PortReference(node_id="node_lr_sched", port_id="exec_in")),
        Edge(id="e_exec_8", source=PortReference(node_id="node_lr_sched", port_id="exec_out"), target=PortReference(node_id="node_zero_grad", port_id="exec_in")),
        Edge(id="e_data_1", source=PortReference(node_id="node_dataset", port_id="dataset"), target=PortReference(node_id="node_dataloader", port_id="dataset")),
        Edge(id="e_data_2", source=PortReference(node_id="node_dataloader", port_id="batch_x"), target=PortReference(node_id="node_emb", port_id="input")),
        Edge(id="e_data_3", source=PortReference(node_id="node_emb", port_id="output"), target=PortReference(node_id="node_forward", port_id="input")),
        Edge(id="e_data_4", source=PortReference(node_id="node_forward", port_id="output"), target=PortReference(node_id="node_loss", port_id="logits")),
        Edge(id="e_data_5", source=PortReference(node_id="node_dataloader", port_id="batch_y"), target=PortReference(node_id="node_loss", port_id="targets")),
        Edge(id="e_data_6", source=PortReference(node_id="node_loss", port_id="loss"), target=PortReference(node_id="node_backward", port_id="loss")),
    ]
    if extra_exec_edges:
        edges.extend(extra_exec_edges)

    g = GraphDefinition(id="graph_dual_flow_ext", name="Dual Flow Pipeline", kind="training_event", nodes=nodes, edges=edges)
    return Project(
        project=ProjectMetadata(id=project_id, name=project_name, created_at="2026-08-30T00:00:00Z", updated_at="2026-08-30T00:00:00Z"),
        model=ModelDefinition(root_graph_id="graph_dual_flow_ext", config={"n_embd": 16}, training=training, graphs={"graph_dual_flow_ext": g}),
        ui=UIState(open_graph_id="graph_dual_flow_ext"),
    )


EXTENDED_ARCHITECTURES: List[Tuple[str, Any]] = [
    ("Arch 11: ReLU Classifier MLP", create_arch_11_relu_classifier),
    ("Arch 12: Dropout MLP", create_arch_12_dropout_mlp),
    ("Arch 13: Deep MLP Tower", create_arch_13_deep_mlp_tower),
    ("Arch 14: Wide-and-Deep Network", create_arch_14_wide_and_deep),
    ("Arch 15: Tied-Embedding LM", create_arch_15_tied_embedding_lm),
    ("Arch 16: Warmup Scheduler Pipeline", create_arch_16_warmup_pipeline),
    ("Arch 17: Step-LR Decay Pipeline", create_arch_17_step_lr_pipeline),
    ("Arch 18: Pre-LayerNorm MLP", create_arch_18_prenorm_mlp),
    ("Arch 19: Residual Add MLP", create_arch_19_residual_add_mlp),
    ("Arch 20: Binary Sequence Classifier", create_arch_20_binary_classifier),
    ("Arch 21: High-Dropout Transformer", create_arch_21_dropout_transformer),
    ("Arch 22: BF16 nanoGPT Micro", create_arch_22_bf16_nanogpt),
    ("Arch 23: Single-Block Causal GPT", create_arch_23_single_block_gpt),
    ("Arch 24: SiLU Deep Feedforward", create_arch_24_silu_deep_ffn),
    ("Arch 25: Metric Logger Pipeline", create_arch_25_metric_pipeline),
]
