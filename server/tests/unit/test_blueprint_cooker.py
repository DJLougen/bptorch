"""Unit tests verifying Blueprint Cooker standalone Python code generation and execution."""

import math
import re
import subprocess
import sys

from neural_blueprint.cooking.cooker import BlueprintCooker
from neural_blueprint.templates.linear_mlp import create_linear_mlp_template
from neural_blueprint.templates.nanogpt import create_nanogpt_template


def test_blueprint_cooker_generates_and_runs_finite_training_loss(tmp_path):
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )

    code = BlueprintCooker.cook(project)
    assert "class GPT(nn.Module):" in code
    assert "class CausalSelfAttention(nn.Module):" in code
    assert "class Block(nn.Module):" in code
    assert "def main():" in code

    script_path = BlueprintCooker.cook_to_file(project, "pytest/nanogpt_train.py")
    assert script_path.exists()

    # 1. Execute standalone script in an isolated subprocess
    res = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--max-steps",
            "5",
            "--batch-size",
            "8",
            "--seed",
            "1337",
            "--save-dir",
            str(tmp_path / "ckpts"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, f"Script execution failed with error: {res.stderr}"
    assert "=== Starting Blueprint Model Training ===" in res.stdout
    assert "=== Training Complete in" in res.stdout
    assert "Saved final checkpoint to:" in res.stdout

    # Parse losses from train.py stdout
    cooked_losses = []
    for line in res.stdout.splitlines():
        match = re.search(r"Step\s+(\d+)/\d+\s+\|\s+Loss:\s+([\d\.]+)", line)
        if match:
            cooked_losses.append(float(match.group(2)))
    assert len(cooked_losses) == 5
    assert all(math.isfinite(loss) and loss > 0.0 for loss in cooked_losses)


def test_two_layer_mlp_cooker_preserves_activation_choice():
    gelu = BlueprintCooker.cook(
        create_linear_mlp_template(
            in_features=32,
            hidden_features=48,
            activation="gelu",
        )
    )
    silu = BlueprintCooker.cook(
        create_linear_mlp_template(
            in_features=32,
            hidden_features=48,
            activation="silu",
        )
    )

    assert "nn.GELU()" in gelu
    assert "nn.SiLU()" not in gelu
    assert "nn.SiLU()" in silu
    assert "nn.GELU()" not in silu

def test_cooked_script_handles_large_batch_size(tmp_path):
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )
    project.model.training = project.model.training.model_copy(update={"batch_size": 2500})
    code = BlueprintCooker.cook(project)
    assert "max(1, dataset_x.size(0) // batch_size)" in code

    script_path = tmp_path / "train_large_batch.py"
    script_path.write_text(code, encoding="utf-8")
    res = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--batch-size",
            "2500",
            "--max-steps",
            "2",
            "--save-dir",
            str(tmp_path / "ckpts"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr


def test_linear_mlp_template_relu_uses_relu_node():
    project = create_linear_mlp_template(activation="relu")
    graph = project.model.graphs["graph_mlp"]
    activation_node = next(n for n in graph.nodes if n.definition_id.endswith("relu@1") or "act" in n.id)
    assert activation_node.definition_id == "builtin.relu@1"
    code = BlueprintCooker.cook(project)
    assert "nn.ReLU()" in code


def test_relu_node_registered():
    from neural_blueprint.registry.registry import global_registry

    assert global_registry.get("builtin.relu@1") is not None


def test_cooked_script_applies_precision_flag():
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    code = BlueprintCooker.cook(project)
    assert "torch.autocast" in code
    assert "args.precision" in code

