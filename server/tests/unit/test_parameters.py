"""Unit tests for parameter accounting and weight tying calculations."""

from neural_blueprint.runtime.parameters import ParameterAccounting
from tests.unit.test_serialization import create_sample_project


def test_mlp_parameter_counting():
    project = create_sample_project()
    accounting = ParameterAccounting()
    summary = accounting.calculate_summary(project)

    # FC1: 64*256 + 256 = 16640
    # FC2: 256*64 + 64 = 16448
    # Total = 33088
    expected_total = (64 * 256 + 256) + (256 * 64 + 64)

    assert summary.total_unique == expected_total
    assert summary.trainable == expected_total
    assert summary.frozen == 0
    assert "node_fc1" in summary.breakdown_by_node
    assert "node_fc2" in summary.breakdown_by_node
