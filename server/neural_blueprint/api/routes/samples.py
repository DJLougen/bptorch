"""Sample gallery API routes."""

from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

from neural_blueprint.templates.samples_catalog import build_catalog, catalog_by_category

router = APIRouter(prefix="/api/v1/samples", tags=["samples"])


class SampleSummary(BaseModel):
    id: str
    name: str
    category: str
    description: str
    highlight: str
    tags: List[str]
    difficulty: str
    filename: str
    path: str


class SampleCatalogResponse(BaseModel):
    count: int
    categories: Dict[str, List[SampleSummary]]
    samples: List[SampleSummary]


@router.get("", response_model=SampleCatalogResponse)
async def list_samples() -> SampleCatalogResponse:
    entries = build_catalog()
    grouped = catalog_by_category()
    categories: Dict[str, List[SampleSummary]] = {}
    for cat, items in grouped.items():
        categories[cat] = [SampleSummary(**e.to_dict()) for e in items]
    return SampleCatalogResponse(
        count=len(entries),
        categories=categories,
        samples=[SampleSummary(**e.to_dict()) for e in entries],
    )


@router.get("/{sample_id}")
async def get_sample_metadata(sample_id: str) -> Dict[str, Any]:
    for entry in build_catalog():
        if entry.id == sample_id:
            return entry.to_dict()
    return {"error": f"Sample '{sample_id}' not found"}
