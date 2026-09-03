# Source Generated with Decompyle++
# File: architectures.cpython-312.pyc (Python 3.12)

'''Authoritative Architecture Matrix generators for 10 distinct neural network topologies and configurations.'''
from typing import Any, List, Tuple

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
from neural_blueprint.templates.architectures_extended import EXTENDED_ARCHITECTURES
from neural_blueprint.templates.linear_mlp import create_linear_mlp_template
from neural_blueprint.templates.nanogpt import create_nanogpt_template


def create_arch_1_nanogpt_tiny():
    p = create_nanogpt_template(block_size = 8, vocab_size = 32, n_layer = 2, n_head = 2, n_embd = 16, dropout = 0, bias = True, attention_impl = 'sdpa')
    p.project.id = 'arch_1_nanogpt_tiny'
    p.project.name = 'Arch 1: nanoGPT Tiny'
    p.model.training = TrainingConfig(device = 'cpu', precision = 'fp32', learning_rate = 0.0006, weight_decay = 0.1, grad_clip = 1, batch_size = 8, seed = 1337, max_steps = 50)
    return p


def create_arch_2_nanogpt_deep():
    p = create_nanogpt_template(block_size = 16, vocab_size = 64, n_layer = 6, n_head = 4, n_embd = 32, dropout = 0.1, bias = False, attention_impl = 'sdpa')
    p.project.id = 'arch_2_nanogpt_deep'
    p.project.name = 'Arch 2: nanoGPT Deep (6L)'
    p.model.training = TrainingConfig(device = 'cpu', precision = 'fp32', learning_rate = 0.0003, weight_decay = 0.01, grad_clip = 0.5, batch_size = 4, seed = 42, max_steps = 30)
    return p


def create_arch_3_nanogpt_wide():
    p = create_nanogpt_template(block_size = 32, vocab_size = 128, n_layer = 1, n_head = 8, n_embd = 64, dropout = 0.05, bias = True, attention_impl = 'sdpa')
    p.project.id = 'arch_3_nanogpt_wide'
    p.project.name = 'Arch 3: nanoGPT Wide (1L/8H)'
    p.model.training = TrainingConfig(device = 'cpu', precision = 'fp32', learning_rate = 0.001, weight_decay = 0.05, grad_clip = 1.5, batch_size = 16, seed = 101, max_steps = 40)
    return p


def create_arch_4_twolayer_mlp():
    p = create_linear_mlp_template(in_features = 64, hidden_features = 256, activation = 'gelu')
    p.project.id = 'arch_4_twolayer_mlp'
    p.project.name = 'Arch 4: Two-Layer MLP'
    p.model.training = TrainingConfig(device = 'cpu', precision = 'fp32', learning_rate = 0.01, weight_decay = 0.0001, grad_clip = 2, batch_size = 16, seed = 777, max_steps = 50)
    return p


def create_arch_5_bottleneck_mlp():
    config = {
        'in_features': 128,
        'hidden_1': 64,
        'bottleneck': 16,
        'hidden_2': 64,
        'out_features': 128 }
    g_root = GraphDefinition(id = 'graph_bottleneck', name = 'Bottleneck Autoencoder', kind = 'root', interface = GraphInterface(), nodes = [
        NodeInstance(id = 'node_in', definition_id = 'builtin.tensor_input@1', display_name = 'Input Features', properties = {
            'name': 'input' }),
        NodeInstance(id = 'node_enc1', definition_id = 'builtin.linear@1', display_name = 'Encoder Linear 1', properties = {
            'in_features': ConfigRefValue(key = 'in_features'),
            'out_features': ConfigRefValue(key = 'hidden_1') }),
        NodeInstance(id = 'node_act1', definition_id = 'builtin.silu@1', display_name = 'SiLU 1'),
        NodeInstance(id = 'node_bottleneck', definition_id = 'builtin.linear@1', display_name = 'Bottleneck Linear', properties = {
            'in_features': ConfigRefValue(key = 'hidden_1'),
            'out_features': ConfigRefValue(key = 'bottleneck') }),
        NodeInstance(id = 'node_dec1', definition_id = 'builtin.linear@1', display_name = 'Decoder Linear 1', properties = {
            'in_features': ConfigRefValue(key = 'bottleneck'),
            'out_features': ConfigRefValue(key = 'hidden_2') }),
        NodeInstance(id = 'node_act2', definition_id = 'builtin.silu@1', display_name = 'SiLU 2'),
        NodeInstance(id = 'node_out_proj', definition_id = 'builtin.linear@1', display_name = 'Output Linear', properties = {
            'in_features': ConfigRefValue(key = 'hidden_2'),
            'out_features': ConfigRefValue(key = 'out_features') }),
        NodeInstance(id = 'node_out', definition_id = 'builtin.graph_output@1', display_name = 'Reconstructed', properties = {
            'name': 'output' })], edges = [
        Edge(id = 'e1', source = PortReference(node_id = 'node_in', port_id = 'output'), target = PortReference(node_id = 'node_enc1', port_id = 'input')),
        Edge(id = 'e2', source = PortReference(node_id = 'node_enc1', port_id = 'output'), target = PortReference(node_id = 'node_act1', port_id = 'input')),
        Edge(id = 'e3', source = PortReference(node_id = 'node_act1', port_id = 'output'), target = PortReference(node_id = 'node_bottleneck', port_id = 'input')),
        Edge(id = 'e4', source = PortReference(node_id = 'node_bottleneck', port_id = 'output'), target = PortReference(node_id = 'node_dec1', port_id = 'input')),
        Edge(id = 'e5', source = PortReference(node_id = 'node_dec1', port_id = 'output'), target = PortReference(node_id = 'node_act2', port_id = 'input')),
        Edge(id = 'e6', source = PortReference(node_id = 'node_act2', port_id = 'output'), target = PortReference(node_id = 'node_out_proj', port_id = 'input')),
        Edge(id = 'e7', source = PortReference(node_id = 'node_out_proj', port_id = 'output'), target = PortReference(node_id = 'node_out', port_id = 'input'))])
    return Project(project = ProjectMetadata(id = 'arch_5_bottleneck_mlp', name = 'Arch 5: Bottleneck Autoencoder', created_at = '2026-08-25T00:00:00Z', updated_at = '2026-08-25T00:00:00Z'), model = ModelDefinition(root_graph_id = 'graph_bottleneck', config = config, training = TrainingConfig(learning_rate = 0.002, weight_decay = 1e-05, grad_clip = 1, batch_size = 8, seed = 555, max_steps = 40), graphs = {
        'graph_bottleneck': g_root }), ui = UIState(open_graph_id = 'graph_bottleneck'))


def create_arch_6_manual_attn_transformer():
    p = create_nanogpt_template(block_size = 12, vocab_size = 48, n_layer = 2, n_head = 3, n_embd = 24, dropout = 0, bias = True, attention_impl = 'manual')
    p.project.id = 'arch_6_manual_attn'
    p.project.name = 'Arch 6: Manual Attention Transformer'
    p.model.training = TrainingConfig(device = 'cpu', precision = 'fp32', learning_rate = 0.0005, weight_decay = 0.02, grad_clip = 0.8, batch_size = 6, seed = 888, max_steps = 35)
    return p


def create_arch_7_dual_flow_pipeline():
    nodes = [
        NodeInstance(id = 'node_dataset', definition_id = 'builtin.dataset_source@1', display_name = 'Dataset Source', properties = {
            'num_samples': 500,
            'vocab_size': 32,
            'sequence_length': 8 }),
        NodeInstance(id = 'node_dataloader', definition_id = 'builtin.dataloader@1', display_name = 'DataLoader', properties = {
            'batch_size': 8 }),
        NodeInstance(id = 'node_emb', definition_id = 'builtin.embedding@1', display_name = 'Embedding', properties = {
            'num_embeddings': 32,
            'embedding_dim': 16 }),
        NodeInstance(id = 'node_forward', definition_id = 'builtin.linear@1', display_name = 'Linear Forward', properties = {
            'in_features': 16,
            'out_features': 32 }),
        NodeInstance(id = 'node_loss', definition_id = 'builtin.cross_entropy_loss@1', display_name = 'Cross Entropy Loss'),
        NodeInstance(id = 'node_backward', definition_id = 'builtin.backward@1', display_name = 'Backward Autograd'),
        NodeInstance(id = 'node_clip_grad', definition_id = 'builtin.clip_gradients@1', display_name = 'Clip Gradients', properties = {
            'max_norm': 1 }),
        NodeInstance(id = 'node_opt_step', definition_id = 'builtin.optimizer_step@1', display_name = 'Optimizer Step'),
        NodeInstance(id = 'node_lr_sched', definition_id = 'builtin.cosine_annealing_lr@1', display_name = 'Cosine Decay LR', properties = {
            'warmup_steps': 10,
            'total_steps': 50 }),
        NodeInstance(id = 'node_zero_grad', definition_id = 'builtin.zero_grad@1', display_name = 'Zero Grad')]
    edges = [
        Edge(id = 'e_exec_1', source = PortReference(node_id = 'node_dataset', port_id = 'exec_out'), target = PortReference(node_id = 'node_dataloader', port_id = 'exec_in')),
        Edge(id = 'e_exec_2', source = PortReference(node_id = 'node_dataloader', port_id = 'exec_out'), target = PortReference(node_id = 'node_emb', port_id = 'exec_in')),
        Edge(id = 'e_exec_2b', source = PortReference(node_id = 'node_emb', port_id = 'exec_out'), target = PortReference(node_id = 'node_forward', port_id = 'exec_in')),
        Edge(id = 'e_exec_3', source = PortReference(node_id = 'node_forward', port_id = 'exec_out'), target = PortReference(node_id = 'node_loss', port_id = 'exec_in')),
        Edge(id = 'e_exec_4', source = PortReference(node_id = 'node_loss', port_id = 'exec_out'), target = PortReference(node_id = 'node_backward', port_id = 'exec_in')),
        Edge(id = 'e_exec_5', source = PortReference(node_id = 'node_backward', port_id = 'exec_out'), target = PortReference(node_id = 'node_clip_grad', port_id = 'exec_in')),
        Edge(id = 'e_exec_6', source = PortReference(node_id = 'node_clip_grad', port_id = 'exec_out'), target = PortReference(node_id = 'node_opt_step', port_id = 'exec_in')),
        Edge(id = 'e_exec_7', source = PortReference(node_id = 'node_opt_step', port_id = 'exec_out'), target = PortReference(node_id = 'node_lr_sched', port_id = 'exec_in')),
        Edge(id = 'e_exec_8', source = PortReference(node_id = 'node_lr_sched', port_id = 'exec_out'), target = PortReference(node_id = 'node_zero_grad', port_id = 'exec_in')),
        Edge(id = 'e_data_1', source = PortReference(node_id = 'node_dataset', port_id = 'dataset'), target = PortReference(node_id = 'node_dataloader', port_id = 'dataset')),
        Edge(id = 'e_data_2', source = PortReference(node_id = 'node_dataloader', port_id = 'batch_x'), target = PortReference(node_id = 'node_emb', port_id = 'input')),
        Edge(id = 'e_data_3', source = PortReference(node_id = 'node_emb', port_id = 'output'), target = PortReference(node_id = 'node_forward', port_id = 'input')),
        Edge(id = 'e_data_4', source = PortReference(node_id = 'node_forward', port_id = 'output'), target = PortReference(node_id = 'node_loss', port_id = 'logits')),
        Edge(id = 'e_data_5', source = PortReference(node_id = 'node_dataloader', port_id = 'batch_y'), target = PortReference(node_id = 'node_loss', port_id = 'targets')),
        Edge(id = 'e_data_6', source = PortReference(node_id = 'node_loss', port_id = 'loss'), target = PortReference(node_id = 'node_backward', port_id = 'loss'))]
    g = GraphDefinition(id = 'graph_dual_flow', name = 'Dual Flow Event Pipeline', kind = 'training_event', nodes = nodes, edges = edges)
    return Project(project = ProjectMetadata(id = 'arch_7_dual_flow', name = 'Arch 7: Dual-Flow Pipeline', created_at = '2026-08-25T00:00:00Z', updated_at = '2026-08-25T00:00:00Z'), model = ModelDefinition(root_graph_id = 'graph_dual_flow', config = {
        'n_embd': 16 }, training = TrainingConfig(learning_rate = 0.001, weight_decay = 0.01, grad_clip = 1, batch_size = 8, seed = 999, max_steps = 50), graphs = {
        'graph_dual_flow': g }), ui = UIState(open_graph_id = 'graph_dual_flow'))


def create_arch_8_resmlp():
    config = {
        'in_dim': 32,
        'hidden_dim': 64,
        'num_classes': 4 }
    g_root = GraphDefinition(id = 'graph_resmlp', name = 'ResMLP Network', kind = 'root', interface = GraphInterface(), nodes = [
        NodeInstance(id = 'node_in', definition_id = 'builtin.tensor_input@1', display_name = 'Input Features', properties = {
            'name': 'input' }),
        NodeInstance(id = 'node_in_proj', definition_id = 'builtin.linear@1', display_name = 'Input Projection', properties = {
            'in_features': ConfigRefValue(key = 'in_dim'),
            'out_features': ConfigRefValue(key = 'hidden_dim') }),
        NodeInstance(id = 'node_ln_1', definition_id = 'builtin.layernorm@1', display_name = 'LayerNorm 1', properties = {
            'normalized_shape': ConfigRefValue(key = 'hidden_dim') }),
        NodeInstance(id = 'node_res_fc1', definition_id = 'builtin.linear@1', display_name = 'Res FC1', properties = {
            'in_features': ConfigRefValue(key = 'hidden_dim'),
            'out_features': ConfigRefValue(key = 'hidden_dim') }),
        NodeInstance(id = 'node_gelu', definition_id = 'builtin.gelu@1', display_name = 'GELU'),
        NodeInstance(id = 'node_res_fc2', definition_id = 'builtin.linear@1', display_name = 'Res FC2', properties = {
            'in_features': ConfigRefValue(key = 'hidden_dim'),
            'out_features': ConfigRefValue(key = 'hidden_dim') }),
        NodeInstance(id = 'node_add', definition_id = 'builtin.add@1', display_name = 'Residual Add'),
        NodeInstance(id = 'node_classifier', definition_id = 'builtin.linear@1', display_name = 'Classifier Head', properties = {
            'in_features': ConfigRefValue(key = 'hidden_dim'),
            'out_features': ConfigRefValue(key = 'num_classes') }),
        NodeInstance(id = 'node_out', definition_id = 'builtin.graph_output@1', display_name = 'Class Logits', properties = {
            'name': 'output' })], edges = [
        Edge(id = 'e1', source = PortReference(node_id = 'node_in', port_id = 'output'), target = PortReference(node_id = 'node_in_proj', port_id = 'input')),
        Edge(id = 'e2', source = PortReference(node_id = 'node_in_proj', port_id = 'output'), target = PortReference(node_id = 'node_ln_1', port_id = 'input')),
        Edge(id = 'e_shortcut', source = PortReference(node_id = 'node_in_proj', port_id = 'output'), target = PortReference(node_id = 'node_add', port_id = 'a')),
        Edge(id = 'e3', source = PortReference(node_id = 'node_ln_1', port_id = 'output'), target = PortReference(node_id = 'node_res_fc1', port_id = 'input')),
        Edge(id = 'e4', source = PortReference(node_id = 'node_res_fc1', port_id = 'output'), target = PortReference(node_id = 'node_gelu', port_id = 'input')),
        Edge(id = 'e5', source = PortReference(node_id = 'node_gelu', port_id = 'output'), target = PortReference(node_id = 'node_res_fc2', port_id = 'input')),
        Edge(id = 'e6', source = PortReference(node_id = 'node_res_fc2', port_id = 'output'), target = PortReference(node_id = 'node_add', port_id = 'b')),
        Edge(id = 'e7', source = PortReference(node_id = 'node_add', port_id = 'output'), target = PortReference(node_id = 'node_classifier', port_id = 'input')),
        Edge(id = 'e8', source = PortReference(node_id = 'node_classifier', port_id = 'output'), target = PortReference(node_id = 'node_out', port_id = 'input'))])
    return Project(project = ProjectMetadata(id = 'arch_8_resmlp', name = 'Arch 8: ResMLP Residual Network', created_at = '2026-08-25T00:00:00Z', updated_at = '2026-08-25T00:00:00Z'), model = ModelDefinition(root_graph_id = 'graph_resmlp', config = config, training = TrainingConfig(learning_rate = 0.003, weight_decay = 0.001, grad_clip = 1, batch_size = 12, seed = 123, max_steps = 45), graphs = {
        'graph_resmlp': g_root }), ui = UIState(open_graph_id = 'graph_resmlp'))


def create_arch_9_multihead_projection():
    config = {
        'n_embd': 40,
        'n_head': 5 }
    g_root = GraphDefinition(id = 'graph_mhead_proj', name = 'MultiHead Projection Network', kind = 'root', interface = GraphInterface(), nodes = [
        NodeInstance(id = 'node_in', definition_id = 'builtin.tensor_input@1', display_name = 'Input Embeddings', properties = {
            'name': 'input' }),
        NodeInstance(id = 'node_qkv', definition_id = 'builtin.linear@1', display_name = 'QKV Linear', properties = {
            'in_features': ConfigRefValue(key = 'n_embd'),
            'out_features': 120 }),
        NodeInstance(id = 'node_split_qkv', definition_id = 'builtin.split_qkv@1', display_name = 'Split QKV', properties = {
            'n_embd': ConfigRefValue(key = 'n_embd') }),
        NodeInstance(id = 'node_split_q', definition_id = 'builtin.split_heads@1', display_name = 'Split Q Heads', properties = {
            'n_head': ConfigRefValue(key = 'n_head'),
            'n_embd': ConfigRefValue(key = 'n_embd') }),
        NodeInstance(id = 'node_split_k', definition_id = 'builtin.split_heads@1', display_name = 'Split K Heads', properties = {
            'n_head': ConfigRefValue(key = 'n_head'),
            'n_embd': ConfigRefValue(key = 'n_embd') }),
        NodeInstance(id = 'node_split_v', definition_id = 'builtin.split_heads@1', display_name = 'Split V Heads', properties = {
            'n_head': ConfigRefValue(key = 'n_head'),
            'n_embd': ConfigRefValue(key = 'n_embd') }),
        NodeInstance(id = 'node_sdpa', definition_id = 'builtin.sdpa@1', display_name = 'SDPA Attention', properties = {
            'is_causal': False }),
        NodeInstance(id = 'node_merge', definition_id = 'builtin.merge_heads@1', display_name = 'Merge Heads', properties = {
            'n_embd': ConfigRefValue(key = 'n_embd') }),
        NodeInstance(id = 'node_ln', definition_id = 'builtin.layernorm@1', display_name = 'LayerNorm', properties = {
            'normalized_shape': ConfigRefValue(key = 'n_embd') }),
        NodeInstance(id = 'node_out', definition_id = 'builtin.graph_output@1', display_name = 'Aggregated Output', properties = {
            'name': 'output' })], edges = [
        Edge(id = 'e1', source = PortReference(node_id = 'node_in', port_id = 'output'), target = PortReference(node_id = 'node_qkv', port_id = 'input')),
        Edge(id = 'e2', source = PortReference(node_id = 'node_qkv', port_id = 'output'), target = PortReference(node_id = 'node_split_qkv', port_id = 'input')),
        Edge(id = 'e_q', source = PortReference(node_id = 'node_split_qkv', port_id = 'q'), target = PortReference(node_id = 'node_split_q', port_id = 'input')),
        Edge(id = 'e_k', source = PortReference(node_id = 'node_split_qkv', port_id = 'k'), target = PortReference(node_id = 'node_split_k', port_id = 'input')),
        Edge(id = 'e_v', source = PortReference(node_id = 'node_split_qkv', port_id = 'v'), target = PortReference(node_id = 'node_split_v', port_id = 'input')),
        Edge(id = 'e_sdpa_q', source = PortReference(node_id = 'node_split_q', port_id = 'output'), target = PortReference(node_id = 'node_sdpa', port_id = 'q')),
        Edge(id = 'e_sdpa_k', source = PortReference(node_id = 'node_split_k', port_id = 'output'), target = PortReference(node_id = 'node_sdpa', port_id = 'k')),
        Edge(id = 'e_sdpa_v', source = PortReference(node_id = 'node_split_v', port_id = 'output'), target = PortReference(node_id = 'node_sdpa', port_id = 'v')),
        Edge(id = 'e_merge', source = PortReference(node_id = 'node_sdpa', port_id = 'output'), target = PortReference(node_id = 'node_merge', port_id = 'input')),
        Edge(id = 'e_ln', source = PortReference(node_id = 'node_merge', port_id = 'output'), target = PortReference(node_id = 'node_ln', port_id = 'input')),
        Edge(id = 'e_out', source = PortReference(node_id = 'node_ln', port_id = 'output'), target = PortReference(node_id = 'node_out', port_id = 'input'))])
    return Project(project = ProjectMetadata(id = 'arch_9_multihead', name = 'Arch 9: Multi-Head Projection', created_at = '2026-08-25T00:00:00Z', updated_at = '2026-08-25T00:00:00Z'), model = ModelDefinition(root_graph_id = 'graph_mhead_proj', config = config, training = TrainingConfig(learning_rate = 0.004, weight_decay = 0, grad_clip = 1, batch_size = 10, seed = 314, max_steps = 50), graphs = {
        'graph_mhead_proj': g_root }), ui = UIState(open_graph_id = 'graph_mhead_proj'))


def create_arch_10_multitask_network():
    config = {
        'vocab_size': 64,
        'n_embd': 32,
        'num_classes': 2 }
    g_root = GraphDefinition(id = 'graph_multitask', name = 'Multi-Task Joint Network', kind = 'root', interface = GraphInterface(), nodes = [
        NodeInstance(id = 'node_tokens', definition_id = 'builtin.token_input@1', display_name = 'Token IDs Input', properties = {
            'name': 'token_ids' }),
        NodeInstance(id = 'node_wte', definition_id = 'builtin.embedding@1', display_name = 'Shared Token Embedding', properties = {
            'num_embeddings': ConfigRefValue(key = 'vocab_size'),
            'embedding_dim': ConfigRefValue(key = 'n_embd') }),
        NodeInstance(id = 'node_backbone_linear', definition_id = 'builtin.linear@1', display_name = 'Backbone Linear', properties = {
            'in_features': ConfigRefValue(key = 'n_embd'),
            'out_features': ConfigRefValue(key = 'n_embd') }),
        NodeInstance(id = 'node_backbone_gelu', definition_id = 'builtin.gelu@1', display_name = 'Backbone GELU'),
        NodeInstance(id = 'node_lm_head', definition_id = 'builtin.lm_head@1', display_name = 'LM Head', properties = {
            'in_features': ConfigRefValue(key = 'n_embd'),
            'out_features': ConfigRefValue(key = 'vocab_size'),
            'bias': False }),
        NodeInstance(id = 'node_lm_logits_out', definition_id = 'builtin.logits_output@1', display_name = 'LM Logits', properties = {
            'name': 'lm_logits' }),
        NodeInstance(id = 'node_cls_head', definition_id = 'builtin.linear@1', display_name = 'Classification Head', properties = {
            'in_features': ConfigRefValue(key = 'n_embd'),
            'out_features': ConfigRefValue(key = 'num_classes') }),
        NodeInstance(id = 'node_cls_logits_out', definition_id = 'builtin.graph_output@1', display_name = 'Class Logits', properties = {
            'name': 'cls_logits' })], edges = [
        Edge(id = 'e1', source = PortReference(node_id = 'node_tokens', port_id = 'output'), target = PortReference(node_id = 'node_wte', port_id = 'input')),
        Edge(id = 'e2', source = PortReference(node_id = 'node_wte', port_id = 'output'), target = PortReference(node_id = 'node_backbone_linear', port_id = 'input')),
        Edge(id = 'e3', source = PortReference(node_id = 'node_backbone_linear', port_id = 'output'), target = PortReference(node_id = 'node_backbone_gelu', port_id = 'input')),
        Edge(id = 'e_lm', source = PortReference(node_id = 'node_backbone_gelu', port_id = 'output'), target = PortReference(node_id = 'node_lm_head', port_id = 'input')),
        Edge(id = 'e_lm_out', source = PortReference(node_id = 'node_lm_head', port_id = 'logits'), target = PortReference(node_id = 'node_lm_logits_out', port_id = 'input')),
        Edge(id = 'e_cls', source = PortReference(node_id = 'node_backbone_gelu', port_id = 'output'), target = PortReference(node_id = 'node_cls_head', port_id = 'input')),
        Edge(id = 'e_cls_out', source = PortReference(node_id = 'node_cls_head', port_id = 'output'), target = PortReference(node_id = 'node_cls_logits_out', port_id = 'input'))])
    return Project(project = ProjectMetadata(id = 'arch_10_multitask', name = 'Arch 10: Multi-Task Network', created_at = '2026-08-25T00:00:00Z', updated_at = '2026-08-25T00:00:00Z'), model = ModelDefinition(root_graph_id = 'graph_multitask', config = config, training = TrainingConfig(learning_rate = 0.0005, weight_decay = 0.05, grad_clip = 1, batch_size = 8, seed = 404, max_steps = 50), graphs = {
        'graph_multitask': g_root }, weight_bindings = [
        WeightBinding(source = WeightBindingEndpoint(node_id = 'node_wte', parameter = 'weight'), target = WeightBindingEndpoint(node_id = 'node_lm_head', parameter = 'weight'), mode = 'share')]), ui = UIState(open_graph_id = 'graph_multitask'))

def create_arch_26_llama_tiny() -> Project:
    from neural_blueprint.templates.llama import create_llama_tiny_template

    p = create_llama_tiny_template(
        block_size=32,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        dropout=0.0,
    )
    p.project.id = "arch_26_llama_tiny"
    p.project.name = "Arch 26: Llama Tiny"
    p.model.training = TrainingConfig(max_steps=40, batch_size=8, seed=1337)
    return p


ALL_ARCHITECTURES: List[Tuple[str, Any]] = [
    ('Arch 1: nanoGPT Tiny', create_arch_1_nanogpt_tiny),
    ('Arch 2: nanoGPT Deep (6L)', create_arch_2_nanogpt_deep),
    ('Arch 3: nanoGPT Wide (1L/8H)', create_arch_3_nanogpt_wide),
    ('Arch 4: Two-Layer MLP', create_arch_4_twolayer_mlp),
    ('Arch 5: Bottleneck Autoencoder', create_arch_5_bottleneck_mlp),
    ('Arch 6: Manual Attention Transformer', create_arch_6_manual_attn_transformer),
    ('Arch 7: Dual-Flow Pipeline', create_arch_7_dual_flow_pipeline),
    ('Arch 8: ResMLP Residual Network', create_arch_8_resmlp),
    ('Arch 9: Multi-Head Projection', create_arch_9_multihead_projection),
    ('Arch 10: Multi-Task Joint Network', create_arch_10_multitask_network),
] + EXTENDED_ARCHITECTURES + [
    ('Arch 26: Llama Tiny', create_arch_26_llama_tiny),
]
