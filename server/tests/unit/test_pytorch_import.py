"""Unit tests verifying PyTorch module-tree importing and API endpoint."""

import pytest
from fastapi.testclient import TestClient

from neural_blueprint.api.main import app
from neural_blueprint.cooking.cooker import BlueprintCooker
from neural_blueprint.importing.pytorch import (
    ImportUnsupportedError,
    import_pytorch_source,
)
from neural_blueprint.runtime.compiler import GraphCompiler


MLP_SOURCE = """
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
"""

RESIDUAL_SOURCE = """
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        return x + self.fc(x)
"""

CONV_SOURCE = """
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3)

    def forward(self, x):
        return self.conv(x)
"""


def test_import_mlp_chain():
    project = import_pytorch_source(MLP_SOURCE)
    root_graph_id = project.model.root_graph_id
    graph = project.model.graphs[root_graph_id]

    def_ids = [n.definition_id for n in graph.nodes]
    assert "builtin.tensor_input@1" in def_ids
    assert def_ids.count("builtin.linear@1") == 2
    assert "builtin.gelu@1" in def_ids
    assert "builtin.graph_output@1" in def_ids

    # Compile plan does not raise
    plan, modules = GraphCompiler().compile_plan(project)
    assert plan is not None
    assert len(plan.instructions) > 0

    # BlueprintCooker.cook generates valid Python code
    code = BlueprintCooker.cook(project)
    assert "class " in code


def test_import_residual_add():
    project = import_pytorch_source(RESIDUAL_SOURCE)
    root_graph_id = project.model.root_graph_id
    graph = project.model.graphs[root_graph_id]

    def_ids = [n.definition_id for n in graph.nodes]
    assert "builtin.add@1" in def_ids
    assert "builtin.linear@1" in def_ids


def test_import_unsupported_conv2d_direct_and_api():
    # Direct import raises ImportUnsupportedError
    with pytest.raises(ImportUnsupportedError) as exc_info:
        import_pytorch_source(CONV_SOURCE)
    assert any("Conv2d" in op for op in exc_info.value.ops)

    # API returns 422 with ops containing Conv2d
    client = TestClient(app)
    resp = client.post("/api/v1/import/pytorch", json={"code": CONV_SOURCE})
    assert resp.status_code == 422
    payload = resp.json()
    assert "ops" in payload["detail"]
    assert any("Conv2d" in op for op in payload["detail"]["ops"])


def test_import_api_success():
    client = TestClient(app)
    resp = client.post("/api/v1/import/pytorch", json={"code": MLP_SOURCE})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert "project" in payload
    assert payload["project"]["model"]["root_graph_id"] == "graph_imported"
