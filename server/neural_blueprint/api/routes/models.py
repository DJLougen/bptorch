"""Model validation, compilation, and standalone cooking API routes."""

import uuid
from typing import Any, Dict, List, Literal, Optional

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from neural_blueprint.cooking.cooker import BlueprintCooker, UnsupportedCookError
from neural_blueprint.ir.models import Project
from neural_blueprint.paths import PathValidationError
from neural_blueprint.runtime.compiler import GraphCompiler
from neural_blueprint.runtime.parameters import ParameterAccounting
from neural_blueprint.runtime.training_capabilities import UnsupportedTrainingConfigError
from neural_blueprint.tracing.debugger import SessionCapacityError, global_session_manager
from neural_blueprint.validation.diagnostics import Diagnostic
from neural_blueprint.validation.validator import ProjectValidator

router = APIRouter(prefix="/api/v1", tags=["models"])

CompileMode = Literal["inference", "training"]
CompileDevice = Literal["cpu", "mps", "cuda"]


def _validate_device_available(device: CompileDevice) -> None:
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise HTTPException(status_code=422, detail="MPS device is not available on this host")
    if device == "cuda" and not torch.cuda.is_available():
        raise HTTPException(status_code=422, detail="CUDA device is not available on this host")


class ValidateRequest(BaseModel):
    project: Project


class ParameterSummaryResponse(BaseModel):
    total_unique: int
    trainable: int
    frozen: int
    shared_references: int
    breakdown_by_node: Dict[str, Any] = Field(default_factory=dict)


class ValidateResponse(BaseModel):
    valid: bool
    graph_hash: str
    resolved_shapes: Dict[str, Any]
    parameter_summary: ParameterSummaryResponse
    diagnostics: List[Diagnostic]


class CompileRequest(BaseModel):
    project: Project
    device: CompileDevice = "cpu"
    mode: CompileMode = "inference"


class CompileResponse(BaseModel):
    session_id: str
    graph_hash: str
    device: str
    parameter_summary: ParameterSummaryResponse


class CookRequest(BaseModel):
    project: Project
    output_path: Optional[str] = None


class CookResponse(BaseModel):
    code: str
    output_path: Optional[str] = None


@router.post("/graphs/validate", response_model=ValidateResponse)
async def validate_graph(req: ValidateRequest):
    validator = ProjectValidator()
    result = validator.validate(req.project)

    compiler = GraphCompiler()
    graph_hash = compiler.compute_graph_hash(req.project)

    accounting = ParameterAccounting()
    param_summary = accounting.calculate_summary(req.project)

    return ValidateResponse(
        valid=result.valid,
        graph_hash=graph_hash,
        resolved_shapes=result.resolved_shapes,
        parameter_summary=ParameterSummaryResponse(
            total_unique=param_summary.total_unique,
            trainable=param_summary.trainable,
            frozen=param_summary.frozen,
            shared_references=param_summary.shared_references,
            breakdown_by_node=param_summary.breakdown_by_node,
        ),
        diagnostics=result.diagnostics,
    )


@router.post("/models/compile", response_model=CompileResponse)
async def compile_model(req: CompileRequest):
    validator = ProjectValidator()
    result = validator.validate(req.project)
    if not result.valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Cannot compile invalid model graph",
                "diagnostics": [d.model_dump() for d in result.errors],
            },
        )

    compiler = GraphCompiler()
    graph_hash = compiler.compute_graph_hash(req.project)

    accounting = ParameterAccounting()
    param_summary = accounting.calculate_summary(req.project)

    session_id = str(uuid.uuid4())

    _validate_device_available(req.device)

    try:
        if req.mode == "training":
            global_session_manager.create_training_session(
                session_id=session_id,
                project=req.project,
                device=req.device,
            )
        else:
            global_session_manager.create_session(
                session_id=session_id,
                project=req.project,
                device=req.device,
            )
    except SessionCapacityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnsupportedTrainingConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return CompileResponse(
        session_id=session_id,
        graph_hash=graph_hash,
        device=req.device,
        parameter_summary=ParameterSummaryResponse(
            total_unique=param_summary.total_unique,
            trainable=param_summary.trainable,
            frozen=param_summary.frozen,
            shared_references=param_summary.shared_references,
            breakdown_by_node=param_summary.breakdown_by_node,
        ),
    )


@router.post("/cook/export", response_model=CookResponse)
async def cook_export(req: CookRequest):
    validator = ProjectValidator()
    result = validator.validate(req.project)
    if not result.valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Cannot cook invalid model graph",
                "diagnostics": [d.model_dump() for d in result.errors],
            },
        )

    try:
        code = BlueprintCooker.cook(req.project)
    except UnsupportedCookError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    saved_path = None
    if req.output_path:
        try:
            saved_path = str(BlueprintCooker.cook_to_file(req.project, req.output_path))
        except PathValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except UnsupportedCookError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return CookResponse(code=code, output_path=saved_path)
