"""PyTorch import route for bpTorch."""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from neural_blueprint.importing.pytorch import (
    ImportUnsupportedError,
    import_pytorch_source,
)

router = APIRouter(prefix="/api/v1/import", tags=["import"])


class ImportPytorchRequest(BaseModel):
    code: str
    class_name: Optional[str] = None


@router.post("/pytorch")
def import_pytorch(req: ImportPytorchRequest):
    try:
        project = import_pytorch_source(req.code, req.class_name)
    except ImportUnsupportedError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc), "ops": exc.ops},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "project": project.model_dump(mode="json")}
