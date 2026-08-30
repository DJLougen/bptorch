"""Vertical slice test: Two-layer MLP graph execution vs hand-written PyTorch model."""

import torch
import torch.nn as nn
from neural_blueprint.runtime.compiler import GraphCompiler
from neural_blueprint.runtime.module import CompiledGraphModule
from tests.unit.test_serialization import create_sample_project


class ReferenceMLP(nn.Module):
    def __init__(self, in_features: int = 64, hidden_features: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


def test_compiled_graph_vertical_slice_parity():
    # 1. Load project IR
    project = create_sample_project()

    # 2. Compile into real PyTorch CompiledGraphModule
    compiler = GraphCompiler()
    plan, modules = compiler.compile_plan(project)
    model = CompiledGraphModule(plan, modules)

    # 3. Verify PyTorch module registration
    assert isinstance(model, nn.Module)
    assert "node_fc1" in model.module_dict
    assert "node_fc2" in model.module_dict
    assert "node_gelu" in model.module_dict

    # Check state_dict and parameters are discoverable
    params = list(model.parameters())
    assert len(params) == 4  # fc1.weight, fc1.bias, fc2.weight, fc2.bias

    # 4. Instantiate reference model and copy identical weights
    torch.manual_seed(42)
    ref_model = ReferenceMLP(in_features=64, hidden_features=256)

    # Copy weights from ref_model into compiled graph module
    with torch.no_grad():
        model.module_dict["node_fc1"].weight.copy_(ref_model.fc1.weight)
        model.module_dict["node_fc1"].bias.copy_(ref_model.fc1.bias)
        model.module_dict["node_fc2"].weight.copy_(ref_model.fc2.weight)
        model.module_dict["node_fc2"].bias.copy_(ref_model.fc2.bias)

    # 5. Execute forward pass with deterministic input
    torch.manual_seed(123)
    x = torch.randn(2, 8, 64)

    ref_out = ref_model(x)
    compiled_out = model(x)

    # 6. Assert numerical equivalence
    assert isinstance(compiled_out, torch.Tensor)
    assert compiled_out.shape == torch.Size([2, 8, 64])

    torch.testing.assert_close(
        compiled_out,
        ref_out,
        rtol=1e-6,
        atol=1e-7,
    )
