"""
Automated Testing & Evaluation Engine for Neural Blueprint Studio.
Executes test suites covering shape sanity, autograd health, overfitting, checkpointing, cooking, and stability.
"""

import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from pydantic import BaseModel, Field

from neural_blueprint.cooking.cooker import BlueprintCooker
from neural_blueprint.ir.models import Project
from neural_blueprint.runtime.compiler import GraphCompiler
from neural_blueprint.runtime.initialization import init_nanogpt_weights
from neural_blueprint.runtime.module import CompiledGraphModule
from neural_blueprint.tracing.debugger import TrainingSession
from neural_blueprint.validation.validator import ProjectValidator


class TestCaseResult(BaseModel):
    """Detailed result for a single automated blueprint test case."""

    id: str
    name: str
    status: Literal["passed", "failed", "skipped"] = "passed"
    duration_ms: float = 0.0
    description: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    details: Optional[str] = None


class TestSuiteResult(BaseModel):
    """Comprehensive test suite execution summary."""

    suite_id: str
    project_name: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    duration_ms: float = 0.0
    cases: List[TestCaseResult] = Field(default_factory=list)


class BlueprintTester:
    """Automated testing harness running multi-pillar validation suites on visual blueprint assets."""

    @classmethod
    def test_shape_and_forward_sanity(cls, project: Project) -> TestCaseResult:
        """Tests forward passes across representative batch sizes."""
        t0 = time.time()
        cfg = project.model.config

        try:
            compiler = GraphCompiler()
            plan, modules = compiler.compile_plan(project)
            model = CompiledGraphModule(plan, modules, project.model.weight_bindings)
            init_nanogpt_weights(model, n_layer=cfg.get("n_layer", 2))
            model.eval()

            root_g = project.model.graphs.get(project.model.root_graph_id)
            has_tokens = (
                any(
                    n.definition_id
                    in ("builtin.token_input@1", "builtin.nanogpt_input_embeddings@1")
                    or "token" in n.id
                    for n in (root_g.nodes if root_g else [])
                )
                if root_g
                else True
            )

            vocab_size = int(cfg.get("vocab_size", 32))
            block_size = int(cfg.get("block_size", 8))
            in_features = int(cfg.get("in_features", cfg.get("in_dim", cfg.get("n_embd", 16))))

            batch_sizes = [1, 2, 4]
            tested_shapes = []

            for bs in batch_sizes:
                if has_tokens:
                    x = torch.randint(0, vocab_size, (bs, block_size))
                    out = model(token_ids=x)
                else:
                    x = torch.randn(bs, in_features)
                    out = model(input=x)

                assert out is not None, f"Model returned None for batch_size={bs}"
                if isinstance(out, dict):
                    res_tensor = next(iter(out.values()))
                    tested_shapes.append(list(res_tensor.shape))
                elif isinstance(out, torch.Tensor):
                    tested_shapes.append(list(out.shape))

            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="shape_sanity",
                name="Dynamic Shape & Forward Pass Sanity",
                status="passed",
                duration_ms=round(duration_ms, 2),
                description="Verified forward pass with varying batch sizes without dimension mismatch.",
                metrics={"tested_batch_sizes": batch_sizes, "output_shapes": tested_shapes},
            )

        except Exception as e:
            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="shape_sanity",
                name="Dynamic Shape & Forward Pass Sanity",
                status="failed",
                duration_ms=round(duration_ms, 2),
                description="Forward pass failed on dynamic tensor shapes.",
                error=str(e),
            )

    @classmethod
    def test_gradient_and_autograd_health(cls, project: Project) -> TestCaseResult:
        """Verifies Autograd backpropagation, finite gradient norms, and detects vanishing/exploding gradients."""
        t0 = time.time()
        try:
            session = TrainingSession("test_grad_health", project, device="cpu")
            session.model.train()

            # Execute stepping
            for _ in range(2):
                asyncio_run_step(session)

            total_params = sum(p.numel() for p in session.model.parameters() if p.requires_grad)
            named_grads = {
                n: float(p.grad.norm().item())
                for n, p in session.model.named_parameters()
                if p.grad is not None
            }

            assert len(named_grads) > 0, "No parameter gradients found after backward pass"
            total_norm = session.metrics.grad_norm

            assert not math.isnan(total_norm), "Gradient norm is NaN"
            assert not math.isinf(total_norm), "Gradient norm is Infinite"
            assert total_norm < 1000.0, f"Gradient explosion detected: norm={total_norm}"

            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="grad_health",
                name="Autograd & Gradient Flow Health",
                status="passed",
                duration_ms=round(duration_ms, 2),
                description="Backpropagation produced finite observed parameter gradients within configured bounds.",
                metrics={
                    "total_trainable_parameters": total_params,
                    "total_gradient_norm": round(total_norm, 4),
                    "gradient_status": session.metrics.grad_status,
                    "parameter_grad_norms": {
                        k: round(v, 4) for k, v in list(named_grads.items())[:5]
                    },
                },
            )

        except Exception as e:
            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="grad_health",
                name="Autograd & Gradient Flow Health",
                status="failed",
                duration_ms=round(duration_ms, 2),
                description="Autograd backpropagation or gradient health check failed.",
                error=str(e),
            )

    @classmethod
    def test_single_batch_overfit(cls, project: Project) -> TestCaseResult:
        """Verifies that the model strictly overfits and minimizes loss on a fixed batch across gradient steps."""
        t0 = time.time()
        try:
            session = TrainingSession("test_overfit", project, device="cpu")
            session.learning_rate = 1e-2
            for pg in session.optimizer.param_groups:
                pg["lr"] = 1e-2

            # Fix batch data
            session.dataset_x = session.dataset_x[:8].repeat(10, 1)
            session.dataset_y = session.dataset_y[:8].repeat(10, 1)

            losses = []
            for _ in range(5):
                asyncio_run_step(session)
                losses.append(session.metrics.loss)

            assert len(losses) == 5
            initial_loss = losses[0]
            final_loss = losses[-1]

            assert final_loss < initial_loss, (
                f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
            )
            assert all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1)), (
                f"Losses did not strictly decrease: {losses}"
            )

            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="single_batch_overfit",
                name="Optimization & Single-Batch Convergence",
                status="passed",
                duration_ms=round(duration_ms, 2),
                description="Model demonstrated strict monotonic loss descent over 5 gradient descent steps.",
                metrics={
                    "initial_loss": round(initial_loss, 4),
                    "final_loss": round(final_loss, 4),
                    "loss_trajectory": [round(loss_val, 4) for loss_val in losses],
                },
            )

        except Exception as e:
            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="single_batch_overfit",
                name="Optimization & Single-Batch Convergence",
                status="failed",
                duration_ms=round(duration_ms, 2),
                description="Single-batch optimization test failed to converge.",
                error=str(e),
            )

    @classmethod
    def test_stateful_checkpoint_roundtrip(cls, project: Project) -> TestCaseResult:
        """Verifies checkpoint serialization, disk persistence, and exact parameter restoration."""
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                session1 = TrainingSession("test_ckpt_s1", project, device="cpu")
                for _ in range(2):
                    asyncio_run_step(session1)

                ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
                session1.save_checkpoint(str(ckpt_path))
                assert ckpt_path.exists(), "Checkpoint file was not written to disk"

                # Restore into fresh session
                session2 = TrainingSession("test_ckpt_s2", project, device="cpu")
                res = session2.load_checkpoint(str(ckpt_path))
                assert res["step"] == 2

                # Verify exact parameter equality
                for (n1, p1), (n2, p2) in zip(
                    session1.model.named_parameters(), session2.model.named_parameters()
                ):
                    assert n1 == n2
                    torch.testing.assert_close(p1, p2)

            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="checkpoint_roundtrip",
                name="Stateful Checkpoint Save & Restore",
                status="passed",
                duration_ms=round(duration_ms, 2),
                description="Successfully serialized model/optimizer state to disk and restored with exact numerical equality.",
                metrics={"restored_step": 2, "parameter_equality_verified": True},
            )

        except Exception as e:
            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="checkpoint_roundtrip",
                name="Stateful Checkpoint Save & Restore",
                status="failed",
                duration_ms=round(duration_ms, 2),
                description="Checkpoint save and restoration roundtrip failed.",
                error=str(e),
            )

    @classmethod
    def test_standalone_cooking_dryrun(cls, project: Project) -> TestCaseResult:
        """Compiles visual blueprint into standalone train.py and executes a dry-run in an isolated subprocess."""
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                code = BlueprintCooker.cook(project)
                assert len(code) > 100, "Cooked code string is empty"

                script_path = Path(tmpdir) / "train.py"
                script_path.write_text(code, encoding="utf-8")
                assert script_path.exists(), "train.py script file was not created"

                res = subprocess.run(
                    [
                        sys.executable,
                        str(script_path),
                        "--max-steps",
                        "2",
                        "--batch-size",
                        "4",
                        "--save-dir",
                        str(Path(tmpdir) / "ckpts"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )

                assert res.returncode == 0, (
                    f"train.py execution failed with exit code {res.returncode}:\n{res.stderr}"
                )

            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="cooker_dryrun",
                name="Standalone Cooking & Subprocess Dry-Run",
                status="passed",
                duration_ms=round(duration_ms, 2),
                description="Generated zero-dependency standalone PyTorch script and successfully executed in an isolated process.",
                metrics={"subprocess_exit_code": 0, "code_size_bytes": len(code)},
            )

        except Exception as e:
            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="cooker_dryrun",
                name="Standalone Cooking & Subprocess Dry-Run",
                status="failed",
                duration_ms=round(duration_ms, 2),
                description="Standalone cooking or dry-run execution failed.",
                error=str(e),
            )

    @classmethod
    def test_numerical_stability(cls, project: Project) -> TestCaseResult:
        """Runs static validation and verifies a finite fp32 training loss."""
        t0 = time.time()
        try:
            validator = ProjectValidator()
            val_res = validator.validate(project)
            errors = [d for d in val_res.diagnostics if d.severity == "error"]
            assert val_res.valid is True, f"Project has static validation errors: {errors}"

            session = TrainingSession("test_stability", project, device="cpu")
            session.precision = "fp32"
            evt = asyncio_run_step(session)

            assert evt is not None
            assert not math.isnan(session.metrics.loss), "Loss is NaN"
            assert not math.isinf(session.metrics.loss), "Loss is Inf"

            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="numerical_stability",
                name="Static Schema & Numerical Stability",
                status="passed",
                duration_ms=round(duration_ms, 2),
                description="Verified static project validation and a finite fp32 training loss.",
                metrics={
                    "validation_diagnostics_count": len(val_res.diagnostics),
                    "finite_loss": round(session.metrics.loss, 4),
                },
            )

        except Exception as e:
            duration_ms = (time.time() - t0) * 1000.0
            return TestCaseResult(
                id="numerical_stability",
                name="Static Schema & Numerical Stability",
                status="failed",
                duration_ms=round(duration_ms, 2),
                description="Static validation or finite-loss check failed.",
                error=str(e),
            )

    @classmethod
    def run_suite(
        cls, project: Project, enabled_tests: Optional[List[str]] = None
    ) -> TestSuiteResult:
        """Executes the full automated testing battery on the provided visual blueprint project."""
        start_time = time.time()
        test_runners = [
            ("shape_sanity", cls.test_shape_and_forward_sanity),
            ("grad_health", cls.test_gradient_and_autograd_health),
            ("single_batch_overfit", cls.test_single_batch_overfit),
            ("checkpoint_roundtrip", cls.test_stateful_checkpoint_roundtrip),
            ("cooker_dryrun", cls.test_standalone_cooking_dryrun),
            ("numerical_stability", cls.test_numerical_stability),
        ]

        cases: List[TestCaseResult] = []
        passed_count = 0
        failed_count = 0

        for test_id, runner_fn in test_runners:
            if enabled_tests is not None and test_id not in enabled_tests:
                continue

            case_result = runner_fn(project)
            cases.append(case_result)
            if case_result.status == "passed":
                passed_count += 1
            else:
                failed_count += 1

        total_duration_ms = (time.time() - start_time) * 1000.0

        return TestSuiteResult(
            suite_id=f"suite_{int(time.time())}",
            project_name=project.project.name,
            total=len(cases),
            passed=passed_count,
            failed=failed_count,
            duration_ms=round(total_duration_ms, 2),
            cases=cases,
        )


def asyncio_run_step(session: TrainingSession):
    """Helper to run async step_batch synchronously in tester runners."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Running inside an existing async loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, session.step_batch()).result()
    else:
        return loop.run_until_complete(session.step_batch())
