"""Trace event models, tensor summary data structures, and training metrics telemetry."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TraceEventType(str, Enum):
    RUN_STARTED = "run_started"
    NODE_STARTED = "node_started"
    NODE_FINISHED = "node_finished"
    NODE_PAUSED = "node_paused"
    NODE_FAILED = "node_failed"
    RUN_FINISHED = "run_finished"
    RUN_CANCELLED = "run_cancelled"

    # Training loop events
    TRAIN_STARTED = "train_started"
    EPOCH_STARTED = "epoch_started"
    BATCH_ENDED = "batch_ended"
    VALIDATION_FINISHED = "validation_finished"
    CHECKPOINT_SAVED = "checkpoint_saved"
    ANOMALY_DETECTED = "anomaly_detected"
    TRAIN_FINISHED = "train_finished"
    HYPERPARAMETER_UPDATED = "hyperparameter_updated"
    TOKEN_GENERATED = "token_generated"
    GENERATION_FINISHED = "generation_finished"


class TensorSummary(BaseModel):
    """Statistical summary of a runtime tensor."""

    shape: List[int] = Field(default_factory=list)
    dtype: str = "float32"
    device: str = "cpu"
    numel: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    l2_norm: Optional[float] = None
    zero_fraction: Optional[float] = None
    nan_count: Optional[int] = 0
    pos_inf_count: Optional[int] = 0
    neg_inf_count: Optional[int] = 0
    sample_values: List[Any] = Field(default_factory=list)


class TrainingMetrics(BaseModel):
    """Real-time training metrics telemetry emitted during training runs."""

    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    avg_loss: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    grad_status: str = "healthy"  # "healthy", "vanishing", "exploding"
    tokens_per_sec: float = 0.0
    step_time_ms: float = 0.0
    vram_mb: float = 0.0
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    best_loss: Optional[float] = None
    custom_metrics: Dict[str, float] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """Structured execution trace event streamed over WebSocket."""

    sequence: int
    event: TraceEventType
    session_id: str
    graph_hash: str
    node_path: str = ""
    timestamp_ns: int = 0
    duration_ns: Optional[int] = None
    inputs: Dict[str, TensorSummary] = Field(default_factory=dict)
    outputs: Dict[str, TensorSummary] = Field(default_factory=dict)
    metrics: Optional[TrainingMetrics] = None
    error: Optional[str] = None
    token: Optional[str] = None
    token_id: Optional[int] = None
