"""Batch-train all catalog architecture samples and record results."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

from neural_blueprint.templates.architectures import ALL_ARCHITECTURES
from neural_blueprint.tracing.debugger import TrainingSession
from neural_blueprint.validation.validator import ProjectValidator

RESULTS_PATH = ROOT / "examples" / "training_results.json"


async def train_sample(display_name: str, builder_fn, steps: int = 5) -> dict:
    project = builder_fn()
    validator = ProjectValidator()
    val = validator.validate(project)
    if not val.valid:
        errors = [d.message for d in val.diagnostics if d.severity == "error"]
        return {
            "id": project.project.id,
            "name": display_name,
            "status": "validation_failed",
            "errors": errors,
        }

    session = TrainingSession(
        session_id=f"batch_{project.project.id}",
        project=project,
        device="cpu",
    )
    losses = []
    last_error = None
    for _ in range(steps):
        try:
            evt = await session.step_batch()
            if evt and evt.error:
                last_error = evt.error
                break
            if session.metrics:
                losses.append(session.metrics.loss)
        except Exception as exc:
            last_error = str(exc)
            break

    status = "ok" if losses and not last_error else "train_failed"
    return {
        "id": project.project.id,
        "name": display_name,
        "status": status,
        "steps_completed": len(losses),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_delta": (losses[0] - losses[-1]) if len(losses) >= 2 else None,
        "error": last_error,
    }


async def main() -> int:
    print(f"Training {len(ALL_ARCHITECTURES)} architecture samples...")
    results = []
    failed = 0
    for display_name, builder_fn in ALL_ARCHITECTURES:
        print(f"  Training: {display_name}...", end=" ", flush=True)
        result = await train_sample(display_name, builder_fn, steps=5)
        results.append(result)
        if result["status"] != "ok":
            failed += 1
            print(f"FAILED ({result.get('error') or result.get('errors')})")
        else:
            print(f"OK  loss {result['initial_loss']:.4f} -> {result['final_loss']:.4f}")

    payload = {
        "total": len(results),
        "passed": len(results) - failed,
        "failed": failed,
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote training results -> {RESULTS_PATH}")
    print(f"Summary: {payload['passed']}/{payload['total']} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
