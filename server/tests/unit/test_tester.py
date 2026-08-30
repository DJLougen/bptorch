"""Unit tests verifying BlueprintTester engine and API testing endpoints."""

from fastapi.testclient import TestClient
from neural_blueprint.api.main import app
from neural_blueprint.ir.serialization import serialize_project
from neural_blueprint.templates.linear_mlp import create_linear_mlp_template
from neural_blueprint.templates.nanogpt import create_nanogpt_template
from neural_blueprint.testing.tester import BlueprintTester


def test_tester_available_suites_endpoint():
    client = TestClient(app)
    res = client.get("/api/v1/test/suites")
    assert res.status_code == 200
    suites = res.json()
    assert len(suites) == 6
    suite_ids = [s["id"] for s in suites]
    assert "shape_sanity" in suite_ids
    assert "grad_health" in suite_ids
    assert "single_batch_overfit" in suite_ids
    assert "checkpoint_roundtrip" in suite_ids
    assert "cooker_dryrun" in suite_ids
    assert "numerical_stability" in suite_ids


def test_tester_run_suite_nanogpt():
    client = TestClient(app)
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )

    payload = {"project": serialize_project(project)}
    res = client.post("/api/v1/test/run", json=payload)
    assert res.status_code == 200
    result = res.json()

    assert result["total"] == 6
    assert result["passed"] == 6
    assert result["failed"] == 0
    assert len(result["cases"]) == 6


def test_tester_run_suite_linear_mlp():
    project = create_linear_mlp_template(in_features=64, hidden_features=256)
    result = BlueprintTester.run_suite(project)

    assert result.total == 6
    assert result.passed == 6
    assert result.failed == 0
    assert len(result.cases) == 6
    assert all(c.status == "passed" for c in result.cases)


def test_tester_individual_pillar_methods():
    project = create_nanogpt_template(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )

    # Test individual test runners
    r1 = BlueprintTester.test_shape_and_forward_sanity(project)
    assert r1.status == "passed"
    assert r1.duration_ms > 0

    r2 = BlueprintTester.test_gradient_and_autograd_health(project)
    assert r2.status == "passed"
    assert "total_gradient_norm" in r2.metrics

    r3 = BlueprintTester.test_single_batch_overfit(project)
    assert r3.status == "passed"
    assert r3.metrics["final_loss"] < r3.metrics["initial_loss"]

    r4 = BlueprintTester.test_stateful_checkpoint_roundtrip(project)
    assert r4.status == "passed"

    r5 = BlueprintTester.test_standalone_cooking_dryrun(project)
    assert r5.status == "passed"

    r6 = BlueprintTester.test_numerical_stability(project)
    assert r6.status == "passed"
