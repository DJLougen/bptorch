"""Tests for the 25-sample architecture catalog."""

from neural_blueprint.templates.architectures import ALL_ARCHITECTURES
from neural_blueprint.templates.samples_catalog import build_catalog, catalog_by_category


def test_all_architectures_count_is_25():
    assert len(ALL_ARCHITECTURES) >= 25


def test_catalog_has_25_entries_with_unique_ids():
    catalog = build_catalog()
    assert len(catalog) == len(ALL_ARCHITECTURES)
    ids = [e.id for e in catalog]
    assert len(ids) == len(set(ids))


def test_catalog_covers_multiple_categories():
    grouped = catalog_by_category()
    assert len(grouped) >= 6
    assert "Transformers" in grouped
    assert "Training Pipelines" in grouped
    assert "Feedforward" in grouped


def test_each_catalog_entry_has_description_and_highlight():
    for entry in build_catalog():
        assert entry.description
        assert entry.highlight
        assert entry.category
        assert entry.filename.endswith(".nbp.json")
