"""Automated Blueprint architecture testing & evaluation API routes."""

import asyncio
from functools import partial
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from neural_blueprint.ir.models import Project
from neural_blueprint.testing.tester import BlueprintTester, TestSuiteResult

router = APIRouter(prefix="/api/v1/test", tags=["tester"])


class TestRunRequest(BaseModel):
    project: Project
    enabled_tests: Optional[List[str]] = Field(default=None)


class AvailableTestInfo(BaseModel):
    id: str
    name: str
    description: str


@router.get("/suites", response_model=List[AvailableTestInfo])
async def get_available_tests():
    """Returns the list of automated evaluation tests available in the studio."""
    return [
        AvailableTestInfo(
            id="shape_sanity",
            name="Dynamic Shape & Forward Pass Sanity",
            description="Verifies forward passes across representative batch sizes without dimension mismatch.",
        ),
        AvailableTestInfo(
            id="grad_health",
            name="Autograd & Gradient Flow Health",
            description="Performs autograd backpropagation and checks observed parameter gradients for finite norms.",
        ),
        AvailableTestInfo(
            id="single_batch_overfit",
            name="Optimization & Single-Batch Convergence",
            description="Tests whether the model can strictly overfit and monotonically decrease loss over 5 gradient descent steps.",
        ),
        AvailableTestInfo(
            id="checkpoint_roundtrip",
            name="Stateful Checkpoint Save & Restore",
            description="Tests model and optimizer state persistence to disk and exact numerical restoration.",
        ),
        AvailableTestInfo(
            id="cooker_dryrun",
            name="Standalone Cooking & Subprocess Dry-Run",
            description="Cooks zero-dependency train.py script and verifies isolated CLI subprocess execution.",
        ),
        AvailableTestInfo(
            id="numerical_stability",
            name="Static Schema & Numerical Stability",
            description="Runs static project validation and verifies a finite fp32 training loss.",
        ),
    ]


@router.post("/run", response_model=TestSuiteResult)
async def run_testing_suite(req: TestRunRequest):
    """Executes the automated testing battery on the provided visual blueprint project."""
    loop = asyncio.get_running_loop()
    runner = partial(BlueprintTester.run_suite, req.project, enabled_tests=req.enabled_tests)
    return await loop.run_in_executor(None, runner)
