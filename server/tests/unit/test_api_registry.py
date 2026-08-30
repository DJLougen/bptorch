from fastapi.testclient import TestClient
from neural_blueprint.api.main import app


def test_api_get_node_catalog():
    client = TestClient(app)
    response = client.get("/api/v1/registry/nodes")
    assert response.status_code == 200
    catalog = response.json()
    assert isinstance(catalog, list)
    assert len(catalog) >= 6

    type_ids = [item["type_id"] for item in catalog]
    assert "builtin.linear@1" in type_ids
    assert "builtin.gelu@1" in type_ids
    assert "builtin.tensor_input@1" in type_ids
