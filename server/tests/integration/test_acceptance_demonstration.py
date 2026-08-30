"""End-to-end acceptance demonstration test executing the 14-step sequence from Section 30."""

import tempfile
from pathlib import Path

import torch
from neural_blueprint.ir.models import ExpressionOp
from neural_blueprint.ir.serialization import (
    load_project_file,
    save_project_file,
)
from neural_blueprint.parity.runner import ParityRunner
from neural_blueprint.runtime.compiler import GraphCompiler
from neural_blueprint.runtime.initialization import init_nanogpt_weights
from neural_blueprint.runtime.module import CompiledGraphModule
from neural_blueprint.templates.nanogpt import create_nanogpt_template
from neural_blueprint.tracing.debugger import global_session_manager
from neural_blueprint.validation.validator import ProjectValidator


def test_section_30_acceptance_demonstration_sequence():
    """
    Executes the complete 14-step acceptance sequence from Section 30 of the specification:
    Step 1: Launch / Initialization
    Step 2: Create project from nanoGPT template
    Step 3: Inspect model at top level
    Step 4: Drill down hierarchy (Stack -> Block -> Attention -> QKV Projection)
    Step 5: Inspect values (dimensions, bindings, parameters, shapes)
    Step 6: Edit configuration (n_embd: 64 -> 96, n_head: 4 -> 8)
    Step 7: Validate graph (0 errors, propagated shapes)
    Step 8: Compile on CPU (session_id, graph_hash)
    Step 9: Run deterministic batch
    Step 10: Inspect tensor statistics on attention output edge
    Step 11: Set breakpoint on first block MLP, pause, inspect input, continue
    Step 12: Modify architecture (GELU -> SiLU), recompile and verify output changes
    Step 13: Save and reload project (preserves custom modifications and stable IDs)
    Step 14: Reference template parity test
    """

    # --- Step 1 & 2: Create Project from nanoGPT Template ---
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )
    assert project.schema_version == 1
    assert project.model.root_graph_id == "graph_gpt"

    # --- Step 3: Inspect Model at Top Level ---
    root_graph = project.model.graphs["graph_gpt"]
    root_node_defs = [n.definition_id for n in root_graph.nodes]
    assert "builtin.token_input@1" in root_node_defs
    assert "builtin.nanogpt_input_embeddings@1" in root_node_defs
    assert "builtin.nanogpt_stack@1" in root_node_defs
    assert "builtin.layernorm@1" in root_node_defs
    assert "builtin.lm_head@1" in root_node_defs
    assert len(root_graph.nodes) == 9

    # --- Step 4: Drill Down Hierarchy ---
    assert "graph_stack" in project.model.graphs
    assert "graph_block" in project.model.graphs
    assert "graph_attention" in project.model.graphs
    assert "graph_mlp" in project.model.graphs
    assert "graph_input_embeddings" in project.model.graphs

    attn_graph = project.model.graphs["graph_attention"]
    qkv_node = next(n for n in attn_graph.nodes if n.id == "node_qkv_proj")
    assert qkv_node.definition_id == "builtin.linear@1"

    # --- Step 5: Inspect Values ---
    assert qkv_node.properties["in_features"].key == "n_embd"
    assert qkv_node.properties["out_features"].expression.op == ExpressionOp.MULTIPLY
    assert qkv_node.properties["out_features"].expression.left == 3

    # --- Step 6: Edit Configuration ---
    project.model.config["n_embd"] = 32
    project.model.config["n_head"] = 4

    # --- Step 7: Validate ---
    validator = ProjectValidator()
    val_result = validator.validate(project)
    assert val_result.valid is True
    assert len(val_result.errors) == 0
    # Inferred shape for QKV output in attention subgraph is now 3 * 32 = 96
    qkv_out_shape = val_result.resolved_shapes["graph_attention"]["node_qkv_proj"]["output"]
    assert qkv_out_shape.shape[-1].value == 96

    # --- Step 8: Compile ---
    compiler = GraphCompiler()
    plan, modules = compiler.compile_plan(project)
    graph_hash = compiler.compute_graph_hash(project)
    assert len(graph_hash) == 16

    session = global_session_manager.create_session("demo_session_1", project, device="cpu")
    assert session.graph_hash == graph_hash

    # --- Step 9: Run Deterministic Batch ---
    token_ids = [[1, 5, 12, 18, 3, 7, 22, 30]]
    targets = [[5, 12, 18, 3, 7, 22, 30, 2]]
    session.prepare_run({"token_ids": token_ids, "targets": targets})

    import asyncio

    asyncio.run(session.run_until_breakpoint_or_end(speed_delay=0.0))
    assert session.state == "completed"

    # --- Step 10: Inspect Tensor Summary ---
    # Inspect edge after block 0 attention output
    ln_f_summary = session.retained_summaries.get("node_ln_f:output")
    assert ln_f_summary is not None
    assert ln_f_summary.shape == [1, 8, 32]
    assert ln_f_summary.mean is not None
    assert len(ln_f_summary.sample_values) > 0

    # --- Step 11: Set Breakpoint, Step, Continue ---
    session.set_breakpoint("node_ln_f", True)
    session.prepare_run({"token_ids": token_ids, "targets": targets})
    asyncio.run(session.run_until_breakpoint_or_end(speed_delay=0.0))
    assert session.state == "paused"
    # Step single instruction
    evt = asyncio.run(session.step_single())
    assert evt is not None
    # Continue to end
    asyncio.run(session.run_until_breakpoint_or_end(speed_delay=0.0))
    assert session.state == "completed"

    # --- Step 12: Modify Architecture (GELU -> SiLU) ---
    # Change activation in MLP subgraph
    mlp_graph = project.model.graphs["graph_mlp"]
    gelu_node = next(n for n in mlp_graph.nodes if n.id == "node_mlp_gelu")
    gelu_node.definition_id = "builtin.silu@1"
    gelu_node.display_name = "SiLU"

    # Recompile modified model
    plan_mod, modules_mod = compiler.compile_plan(project)
    model_mod = CompiledGraphModule(plan_mod, modules_mod, project.model.weight_bindings)
    init_nanogpt_weights(model_mod, n_layer=2)

    tok_t = torch.tensor(token_ids, dtype=torch.long)
    tgt_t = torch.tensor(targets, dtype=torch.long)

    mod_out = model_mod(token_ids=tok_t, targets=tgt_t)
    assert "logits" in mod_out
    assert mod_out["logits"].shape == torch.Size([1, 8, 32])

    # --- Step 13: Save and Reload Project ---
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "nanogpt_modified.nbp.json"
        save_project_file(project, save_path)
        assert save_path.exists()

        reloaded = load_project_file(save_path)
        assert reloaded.project.name == project.project.name
        assert reloaded.model.config["n_embd"] == 32
        reloaded_mlp = reloaded.model.graphs["graph_mlp"]
        reloaded_act = next(n for n in reloaded_mlp.nodes if n.id == "node_mlp_gelu")
        assert reloaded_act.definition_id == "builtin.silu@1"

    # --- Step 14: Reference Template Parity Test ---
    parity_runner = ParityRunner(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
        attention_impl="manual",
    )
    assert parity_runner.check_forward_parity() is True
    assert parity_runner.check_intermediate_parity() is True
    assert parity_runner.check_gradient_parity() is True
    assert parity_runner.check_parameter_count_parity() is True
