"""Runtime training capability guards for session creation and compile."""

from __future__ import annotations

from neural_blueprint.ir.models import Project, TrainingConfig


class UnsupportedTrainingConfigError(ValueError):
    """Raised when a project requests training features the runtime does not support."""


def validate_training_capabilities(project: Project) -> None:
    """Reject unsupported training settings before creating a TrainingSession."""
    training: TrainingConfig | None = getattr(project.model, "training", None)
    if training is None:
        return

    if training.ddp_enabled:
        raise UnsupportedTrainingConfigError(
            "Distributed training (ddp_enabled=True) is not supported"
        )

    if training.grad_accum_steps != 1:
        raise UnsupportedTrainingConfigError(
            f"Gradient accumulation (grad_accum_steps={training.grad_accum_steps}) is not supported; use 1"
        )

    if training.precision == "fp8":
        raise UnsupportedTrainingConfigError("FP8 precision is not supported")
