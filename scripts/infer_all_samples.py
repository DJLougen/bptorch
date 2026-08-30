"""Batch inference across all 25 architecture samples and record results."""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

from neural_blueprint.runtime.inference import InferenceEngine
from neural_blueprint.templates.architectures import ALL_ARCHITECTURES
from neural_blueprint.validation.validator import ProjectValidator

RESULTS_PATH = ROOT / "examples" / "inference_results.json"


async def infer_sample(display_name: str, builder_fn) -> dict:
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

    started = time.monotonic()
    try:
        engine = InferenceEngine(project=project, device="cpu")
        result = await engine.infer()
        duration_ms = round((time.monotonic() - started) * 1000.0, 2)
        outputs = []
        all_finite = True
        for key, summary in result.get("outputs", {}).items():
            shape = summary.get("shape") or []
            dtype = summary.get("dtype")
            values = summary.get("sample_values") or []
            finite = all(math.isfinite(float(v)) for v in values) if values else True
            all_finite = all_finite and finite
            outputs.append({"key": key, "shape": shape, "dtype": dtype})
        return {
            "id": project.project.id,
            "name": display_name,
            "status": "ok",
            "mode": result.get("mode"),
            "output_count": result.get("tensor_count", 0),
            "outputs": outputs,
            "finite": all_finite,
            "duration_ms": duration_ms,
            "error": None,
        }
    except Exception as exc:
        duration_ms = round((time.monotonic() - started) * 1000.0, 2)
        return {
            "id": project.project.id,
            "name": display_name,
            "status": "inference_failed",
            "mode": None,
            "output_count": 0,
            "outputs": [],
            "finite": False,
            "duration_ms": duration_ms,
            "error": str(exc),
        }


async def main() -> int:
    print(f"Inferencing {len(ALL_ARCHITECTURES)} architecture samples...")
    results = []
    failed = 0
    for display_name, builder_fn in ALL_ARCHITECTURES:
        print(f"  Inferring: {display_name}...", end=" ", flush=True)
        result = await infer_sample(display_name, builder_fn)
        results.append(result)
        if result["status"] != "ok":
            failed += 1
            print(f"FAILED ({result.get('error') or result.get('errors')})")
        else:
            print(
                f"OK  mode={result['mode']} outputs={result['output_count']} "
                f"({result['duration_ms']}ms)"
            )

    payload = {
        "total": len(results),
        "passed": len(results) - failed,
        "failed": failed,
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote inference results -> {RESULTS_PATH}")
    print(f"Summary: {payload['passed']}/{payload['total']} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
