"""Registry API routes."""

from typing import Any, Dict, List

from fastapi import APIRouter

from neural_blueprint.registry.registry import global_registry

router = APIRouter(prefix="/api/v1/registry", tags=["registry"])


@router.get("/nodes", response_model=List[Dict[str, Any]])
async def get_node_catalog():
    """Returns authoritative node catalog for palette and inspector."""
    return global_registry.export_catalog()


@router.get("/modules", response_model=List[Dict[str, Any]])
async def get_module_catalog():
    """Returns built-in composite module templates."""
    # Will be populated as composite modules are registered
    return []
