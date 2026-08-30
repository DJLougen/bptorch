import json
from pathlib import Path

from fastapi.testclient import TestClient
from neural_blueprint.api.main import app
from neural_blueprint.ir.serialization import serialize_project
from neural_blueprint.templates.nanogpt import create_nanogpt_template
from tests.unit.test_serialization import create_sample_project


def test_api_validate_graph():
    client = TestClient(app)
    project = create_sample_project()
    payload = {"project": serialize_project(project)}

    response = client.post("/api/v1/graphs/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert "graph_hash" in data
    assert "resolved_shapes" in data
    assert data["parameter_summary"]["trainable"] > 0


def test_api_compile_model():
    client = TestClient(app)
    project = create_sample_project()
    payload = {"project": serialize_project(project), "device": "cpu"}

    response = client.post("/api/v1/models/compile", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "graph_hash" in data
    assert data["device"] == "cpu"


def test_api_compile_training_mode_and_step_batch_flow():
    client = TestClient(app)
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    payload = {"project": serialize_project(project), "device": "cpu", "mode": "training"}

    response = client.post("/api/v1/models/compile", json=payload)
    assert response.status_code == 200
    data = response.json()
    session_id = data["session_id"]

    # Step batch via API
    step_resp = client.post(f"/api/v1/sessions/{session_id}/step-batch")
    assert step_resp.status_code == 200
    step_data = step_resp.json()
    assert step_data["step"] == 1
    assert step_data["metrics"]["loss"] > 0.0

    # Get metrics via API
    metrics_resp = client.get(f"/api/v1/sessions/{session_id}/metrics")
    assert metrics_resp.status_code == 200
    metrics_data = metrics_resp.json()
    assert len(metrics_data["loss_history"]) == 1


def test_api_compile_public_nanogpt_json_fixture():
    client = TestClient(app)
    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "web"
        / "public"
        / "examples"
        / "nanogpt"
        / "nanogpt_tiny.nbp.json"
    )
    with open(fixture_path) as f:
        project_data = json.load(f)

    # 1. Validate
    val_resp = client.post("/api/v1/graphs/validate", json={"project": project_data})
    assert val_resp.status_code == 200
    assert val_resp.json()["valid"] is True

    # 2. Compile in training mode
    comp_resp = client.post(
        "/api/v1/models/compile",
        json={"project": project_data, "device": "cpu", "mode": "training"},
    )
    assert comp_resp.status_code == 200
    session_id = comp_resp.json()["session_id"]

    # 3. Step batch
    step_resp = client.post(f"/api/v1/sessions/{session_id}/step-batch")
    assert step_resp.status_code == 200
    assert step_resp.json()["step"] == 1


def test_api_cook_export():
    client = TestClient(app)
    project = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    payload = {"project": serialize_project(project)}

    response = client.post("/api/v1/cook/export", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "class GPT(nn.Module):" in data["code"]
    assert "def main():" in data["code"]
