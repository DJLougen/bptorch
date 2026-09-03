"""Session execution, debugging, training operations, and WebSocket trace streaming routes."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import torch

from neural_blueprint.paths import PathValidationError, resolve_sandbox_path
from neural_blueprint.runtime.generation import GenerationEngine
from neural_blueprint.runtime.inference import InferenceEngine
from neural_blueprint.runtime.training_capabilities import UnsupportedTrainingConfigError
from neural_blueprint.tracing.debugger import (
    SessionCapacityError,
    TrainingSession,
    global_session_manager,
)
from neural_blueprint.tracing.events import TraceEventType

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


def _promote_to_training_session(session_id: str, project, device: str) -> TrainingSession:
    existing = global_session_manager.get_session(session_id)
    if existing is not None:
        existing.stop()
    try:
        return global_session_manager.create_training_session(
            session_id=session_id,
            project=project,
            device=device,
        )
    except UnsupportedTrainingConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SessionCapacityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class TraceOptions(BaseModel):
    enabled: bool = True
    speed: str = "normal"  # instant, fast, normal, step


class RunSessionRequest(BaseModel):
    mode: str = "inspection"  # inspection, evaluation, training-test
    inputs: Dict[str, Any] = Field(default_factory=dict)
    trace: TraceOptions = Field(default_factory=TraceOptions)


class TrainSessionRequest(BaseModel):
    max_steps: Optional[int] = None
    speed_delay: float = 0.0


class HyperparametersRequest(BaseModel):
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    grad_clip: Optional[float] = None
    batch_size: Optional[int] = None

class SaveCheckpointRequest(BaseModel):
    path: Optional[str] = None


class LoadCheckpointRequest(BaseModel):
    path: str


class InferRequest(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)

class GenerateRequest(BaseModel):
    prompt: str = ""
    prompt_ids: Optional[list[int]] = None
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    template: Literal["raw", "chatml", "alpaca", "llama3"] = "raw"
    use_cache: bool = True
    stream: bool = True
class DatasetRequest(BaseModel):
    name: Literal["synthetic", "tiny_shakespeare"] = "synthetic"
    val_fraction: float = 0.1
class DatasetUploadRequest(BaseModel):
    text: str




@router.post("/{session_id}/run")
async def run_session(session_id: str, req: RunSessionRequest):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session.prepare_run(req.inputs)

    speed_delays = {
        "instant": 0.0,
        "fast": 0.05,
        "normal": 0.15,
        "step": 0.0,
    }
    delay = speed_delays.get(req.trace.speed, 0.15)

    if req.trace.speed == "step":
        session.state = "paused"
        return {"status": "paused", "session_id": session_id}

    session._launch_background(session.run_until_breakpoint_or_end(speed_delay=delay))

    return {
        "status": "started",
        "session_id": session_id,
        "mode": req.mode,
    }


@router.post("/{session_id}/step")
async def step_session(session_id: str):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    event = await session.step_single()
    return {
        "status": session.state,
        "session_id": session_id,
        "event": event.model_dump() if event else None,
    }


@router.post("/{session_id}/continue")
async def continue_session(session_id: str):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session.state = "running"
    session._launch_background(session.run_until_breakpoint_or_end(speed_delay=0.1))
    return {"status": "running", "session_id": session_id}


@router.post("/{session_id}/stop")
async def stop_session(session_id: str):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session.stop()
    return {"status": "stopped", "session_id": session_id}


@router.post("/{session_id}/train")
async def start_training(session_id: str, req: TrainSessionRequest):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if not isinstance(session, TrainingSession):
        session = _promote_to_training_session(
            session_id=session_id,
            project=session.project,
            device=session.device,
        )

    session._launch_background(
        session.run_training_loop(max_steps=req.max_steps, speed_delay=req.speed_delay)
    )

    return {
        "status": "training_started",
        "session_id": session_id,
        "max_steps": req.max_steps or session.max_steps,
    }


@router.post("/{session_id}/pause")
async def pause_training(session_id: str):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if isinstance(session, TrainingSession):
        session.pause()
    else:
        session.state = "paused"

    return {"status": "paused", "session_id": session_id}


@router.post("/{session_id}/resume")
async def resume_training(session_id: str):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if isinstance(session, TrainingSession):
        session.resume()
    else:
        session.state = "running"
        session._launch_background(session.run_until_breakpoint_or_end(speed_delay=0.1))

    return {"status": "resumed", "session_id": session_id}


@router.post("/{session_id}/step-batch")
async def step_batch(session_id: str):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if not isinstance(session, TrainingSession):
        session = _promote_to_training_session(
            session_id=session_id,
            project=session.project,
            device=session.device,
        )

    event = await session.step_batch()
    return {
        "status": session.state,
        "session_id": session_id,
        "step": session.step,
        "event": event.model_dump() if event else None,
        "metrics": session.metrics.model_dump(),
    }


@router.post("/{session_id}/step-epoch")
async def step_epoch(session_id: str):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if not isinstance(session, TrainingSession):
        session = _promote_to_training_session(
            session_id=session_id,
            project=session.project,
            device=session.device,
        )

    events = await session.step_epoch()
    return {
        "status": session.state,
        "session_id": session_id,
        "epoch": session.epoch,
        "step": session.step,
        "events_count": len(events),
        "metrics": session.metrics.model_dump(),
    }


@router.post("/{session_id}/hyperparameters")
async def update_hyperparameters(session_id: str, req: HyperparametersRequest):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if isinstance(session, TrainingSession):
        session.update_hyperparameters(
            learning_rate=req.learning_rate,
            weight_decay=req.weight_decay,
            grad_clip=req.grad_clip,
            batch_size=req.batch_size,
        )
        return {
            "status": "updated",
            "learning_rate": session.learning_rate,
            "weight_decay": session.weight_decay,
            "grad_clip": session.grad_clip,
            "batch_size": session.batch_size,
        }
    return {"status": "ignored", "detail": "Session is not a TrainingSession"}


@router.post("/{session_id}/checkpoint/save")
async def save_checkpoint(session_id: str, req: SaveCheckpointRequest):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if not isinstance(session, TrainingSession):
        raise HTTPException(status_code=400, detail="Session is not a TrainingSession")

    try:
        sandboxed_path = resolve_sandbox_path(req.path, "checkpoints") if req.path else None
        saved_path = session.save_checkpoint(path=str(sandboxed_path) if sandboxed_path else None)
    except PathValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "saved",
        "path": saved_path,
        "step": session.step,
        "epoch": session.epoch,
    }


@router.post("/{session_id}/checkpoint/load")
async def load_checkpoint(session_id: str, req: LoadCheckpointRequest):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if not isinstance(session, TrainingSession):
        session = _promote_to_training_session(
            session_id=session_id,
            project=session.project,
            device=session.device,
        )

    try:
        sandboxed_path = resolve_sandbox_path(req.path, "checkpoints")
        res = session.load_checkpoint(path=str(sandboxed_path))
        return {"status": "loaded", **res}
    except PathValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/{session_id}/checkpoints")
async def list_checkpoints(session_id: str):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    checkpoints = []
    try:
        ckpt_dir = resolve_sandbox_path("dummy.pt", "checkpoints").parent
    except Exception:
        ckpt_dir = Path("checkpoints")

    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True):
            checkpoints.append({
                "name": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "modified": p.stat().st_mtime,
            })
    return {"status": "ok", "checkpoints": checkpoints}


@router.get("/{session_id}/metrics")
async def get_metrics(session_id: str):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if isinstance(session, TrainingSession):
        return {
            "step": session.step,
            "epoch": session.epoch,
            "loss_history": session.loss_history,
            "metrics": session.metrics.model_dump(),
            "node_gradient_norms": session.node_gradient_norms,
            "parameter_norms": session.parameter_norms,
        }

    return {
        "step": 0,
        "epoch": 0,
        "loss_history": [],
        "metrics": None,
        "node_gradient_norms": {},
        "parameter_norms": {},
    }


@router.post("/{session_id}/infer")
async def infer_session(session_id: str, req: InferRequest):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    if not isinstance(session, TrainingSession):
        raise HTTPException(status_code=400, detail="Inference requires a TrainingSession")
    engine = InferenceEngine(session=session)
    result = await engine.infer(req.inputs)
    return {"status": "ok", "session_id": session_id, **result}
@router.post("/{session_id}/generate")
async def generate_session(session_id: str, req: GenerateRequest):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    if not isinstance(session, TrainingSession):
        session = _promote_to_training_session(session_id, session.project, session.device)

    max_new_tokens = max(1, min(256, req.max_new_tokens))
    engine = GenerationEngine(session=session)

    if req.prompt_ids:
        input_ids = torch.tensor([req.prompt_ids], dtype=torch.long, device=session.device)
    else:
        input_ids = engine.encode_prompt(req.prompt, template=req.template)

    if req.stream:
        async def stream_generate():
            try:
                for token_id, token_str in engine.generate_tokens(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    use_cache=req.use_cache,
                ):
                    await session.emit_event(
                        TraceEventType.TOKEN_GENERATED,
                        token=token_str,
                        token_id=token_id,
                    )
                    await asyncio.sleep(0.005)
                await session.emit_event(TraceEventType.GENERATION_FINISHED)
            except Exception as exc:
                await session.emit_event(TraceEventType.NODE_FAILED, error=str(exc))

        session.launch_background(stream_generate())
        return {"status": "started", "session_id": session_id}

    try:
        token_ids = []
        for token_id, _ in engine.generate_tokens(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            use_cache=req.use_cache,
        ):
            token_ids.append(token_id)
        text = engine.tokenizer.decode(token_ids)
        return {"status": "ok", "token_ids": token_ids, "text": text}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
@router.post("/{session_id}/dataset")
async def set_session_dataset(session_id: str, req: DatasetRequest):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    if not isinstance(session, TrainingSession):
        session = _promote_to_training_session(session_id, session.project, session.device)

    val_fraction = max(0.05, min(0.5, float(req.val_fraction)))
    num_samples = session.load_dataset_by_name(req.name, val_fraction=val_fraction)
    return {"status": "ok", "name": req.name, "num_samples": num_samples}
@router.post("/{session_id}/dataset/upload")
async def upload_session_dataset(session_id: str, req: DatasetUploadRequest):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    if not isinstance(session, TrainingSession):
        session = _promote_to_training_session(session_id, session.project, session.device)

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Uploaded dataset cannot be empty")

    from neural_blueprint.data.tiny_shakespeare import load_token_dataset
    from neural_blueprint.runtime.tokenizer import CharacterTokenizer

    tok = CharacterTokenizer(session.vocab_size)
    split_idx = int(len(req.text) * 0.9)
    train_t = req.text[:split_idx] if split_idx > session.block_size else req.text
    val_t = req.text[split_idx:] if len(req.text) - split_idx > session.block_size else req.text
    session.dataset_x, session.dataset_y = load_token_dataset(train_t, tok, session.block_size)
    session.val_x, session.val_y = load_token_dataset(val_t, tok, session.block_size)

    return {"status": "ok", "num_samples": len(session.dataset_x)}


@router.post("/{session_id}/validate")
async def validate_session_eval(session_id: str):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    if not isinstance(session, TrainingSession):
        raise HTTPException(status_code=400, detail="Session is not a TrainingSession")

    val_loss = await session.evaluate_validation()
    return {"status": "ok", "val_loss": val_loss, "val_samples": len(session.val_x)}







@router.get("/{session_id}/tensors/{tensor_id}/summary")
async def get_tensor_summary(session_id: str, tensor_id: str):
    session = global_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    exact = session.retained_summaries.get(tensor_id)
    if exact is not None:
        return exact.model_dump()

    for key, summary in session.retained_summaries.items():
        if key.endswith(tensor_id) or tensor_id in key:
            return summary.model_dump()

    raise HTTPException(status_code=404, detail=f"Tensor '{tensor_id}' not found in session")


ws_router = APIRouter(prefix="/ws/api/v1/sessions", tags=["trace_ws"])


@ws_router.websocket("/{session_id}/events")
async def websocket_trace_events(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = global_session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return

    try:
        while True:
            evt = await session.event_queue.get()
            await websocket.send_text(json.dumps(evt.model_dump()))
    except WebSocketDisconnect:
        return
