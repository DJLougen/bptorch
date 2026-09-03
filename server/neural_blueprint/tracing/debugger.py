"""Interactive debugging, breakpoint manager, stepping, training lifecycle, and stateful sessions."""

import asyncio
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from neural_blueprint.ir.evaluator import evaluate_value
from neural_blueprint.ir.models import Project
from neural_blueprint.paths import (
    PathValidationError,
    resolve_sandbox_path,
)
from neural_blueprint.runtime.compiler import GraphCompiler
from neural_blueprint.runtime.initialization import init_nanogpt_weights
from neural_blueprint.runtime.module import CompiledGraphModule
from neural_blueprint.runtime.training_capabilities import validate_training_capabilities
from neural_blueprint.tracing.collector import TensorSummarizer
from neural_blueprint.tracing.events import (
    TensorSummary,
    TraceEvent,
    TraceEventType,
    TrainingMetrics,
)


class RuntimeSession:
    """Manages an active compiled runtime session with single-step debugging and breakpoints."""

    def __init__(
        self,
        session_id: str,
        project: Project,
        device: str = "cpu",
    ):
        self.session_id = session_id
        self.project = project
        self.device = device

        compiler = GraphCompiler()
        self.graph_hash = compiler.compute_graph_hash(project)
        self.plan, self.modules = compiler.compile_plan(project)
        self.model = CompiledGraphModule(self.plan, self.modules, project.model.weight_bindings)
        init_nanogpt_weights(self.model, n_layer=project.model.config.get("n_layer", 2))

        self.state: str = "idle"
        self.current_instruction_idx: int = 0
        self.breakpoints: Set[str] = set()

        for graph in project.model.graphs.values():
            for node in graph.nodes:
                if node.metadata.breakpoint:
                    self.breakpoints.add(node.id)

        self.value_table: Dict[Tuple[str, str], Any] = {}
        self.retained_summaries: Dict[str, TensorSummary] = {}
        self.event_queue: asyncio.Queue[TraceEvent] = asyncio.Queue()
        self.sequence_counter: int = 0
        self.last_active_timestamp: float = time.time()
        self._background_task: Optional[asyncio.Task] = None

    def _launch_background(self, coro) -> asyncio.Task:
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
        self._background_task = asyncio.create_task(coro)
        return self._background_task

    def launch_background(self, coro) -> asyncio.Task:
        return self._launch_background(coro)

    def set_breakpoint(self, node_id: str, enabled: bool) -> None:
        if enabled:
            self.breakpoints.add(node_id)
        else:
            self.breakpoints.discard(node_id)

    async def emit_event(
        self,
        event_type: TraceEventType,
        node_path: str = "",
        duration_ns: Optional[int] = None,
        inputs: Optional[Dict[str, TensorSummary]] = None,
        outputs: Optional[Dict[str, TensorSummary]] = None,
        metrics: Optional[TrainingMetrics] = None,
        error: Optional[str] = None,
        token: Optional[str] = None,
        token_id: Optional[int] = None,
    ) -> TraceEvent:
        self.sequence_counter += 1
        evt = TraceEvent(
            sequence=self.sequence_counter,
            event=event_type,
            session_id=self.session_id,
            graph_hash=self.graph_hash,
            node_path=node_path,
            timestamp_ns=time.time_ns(),
            duration_ns=duration_ns,
            inputs=inputs or {},
            outputs=outputs or {},
            metrics=metrics,
            error=error,
            token=token,
            token_id=token_id,
        )
        await self.event_queue.put(evt)
        return evt

    def prepare_run(self, inputs: Dict[str, Any]) -> None:
        self.value_table.clear()
        self.retained_summaries.clear()
        self.current_instruction_idx = 0
        self.state = "running"
        self.last_active_timestamp = time.time()

        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                if isinstance(v, list):
                    v = torch.tensor(
                        v,
                        dtype=(torch.long if "token" in k or "target" in k else torch.float32),
                    )
                elif isinstance(v, (int, float)):
                    v = torch.tensor(v)
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
            self.value_table[(k, "output")] = v

    async def step_single(self) -> Optional[TraceEvent]:
        if self.current_instruction_idx >= len(self.plan.instructions):
            self.state = "completed"
            return await self.emit_event(TraceEventType.RUN_FINISHED)

        instruction = self.plan.instructions[self.current_instruction_idx]
        start_ns = time.time_ns()

        await self.emit_event(TraceEventType.NODE_STARTED, node_path=instruction.node_path)

        input_summaries: Dict[str, TensorSummary] = {}
        inputs: Dict[str, Any] = {}
        for b in instruction.input_bindings:
            key1 = (b.source_node_path, b.source_port_id)
            src_short = b.source_node_path.split("/")[-1]
            key2 = (src_short, b.source_port_id)

            val = self.value_table.get(key1)
            if val is None:
                val = self.value_table.get(key2)

            if val is not None:
                inputs[b.port_id] = val
                input_summaries[b.port_id] = TensorSummarizer.summarize(val)

        try:
            res = None
            if instruction.module_key and instruction.module_key in self.model.module_dict:
                mod = self.model.module_dict[instruction.module_key]
                if instruction.is_repeat and isinstance(mod, nn.ModuleList):
                    in_val = inputs.get("input")
                    if in_val is None and inputs:
                        in_val = next(iter(inputs.values()))
                    curr_x = in_val
                    for blk in mod:
                        curr_x = blk(curr_x)
                    res = curr_x
                elif instruction.is_composite and isinstance(mod, CompiledGraphModule):
                    res = mod(**inputs)
                elif "q" in inputs and "k" in inputs and "v" in inputs:
                    res = mod(inputs["q"], inputs["k"], inputs["v"])
                elif "input" in inputs and len(inputs) == 1:
                    res = mod(inputs["input"])
                elif len(inputs) == 1:
                    res = mod(next(iter(inputs.values())))
                elif inputs:
                    res = mod(**inputs)
            elif instruction.functional_fn:
                if len(inputs) == 2 and "a" in inputs and "b" in inputs:
                    res = instruction.functional_fn(inputs["a"], inputs["b"])
                elif len(inputs) == 2 and "logits" in inputs and "targets" in inputs:
                    res = instruction.functional_fn(inputs["logits"], inputs["targets"])
                elif len(inputs) == 1:
                    res = instruction.functional_fn(next(iter(inputs.values())))
                else:
                    res = instruction.functional_fn(**inputs)
            elif instruction.definition_id in (
                "builtin.tensor_input@1",
                "builtin.token_input@1",
                "builtin.target_input@1",
            ):
                res = self.value_table.get((instruction.node_id, "output"))
                if res is None and instruction.input_name:
                    res = self.value_table.get((instruction.input_name, "output"))

            output_summaries: Dict[str, TensorSummary] = {}
            if isinstance(res, dict):
                for p_id, p_val in res.items():
                    self.value_table[(instruction.node_path, p_id)] = p_val
                    self.value_table[(instruction.node_id, p_id)] = p_val
                    s = TensorSummarizer.summarize(p_val)
                    output_summaries[p_id] = s
                    self.retained_summaries[f"{instruction.node_path}:{p_id}"] = s
            elif res is not None:
                self.value_table[(instruction.node_path, "output")] = res
                self.value_table[(instruction.node_id, "output")] = res
                s = TensorSummarizer.summarize(res)
                output_summaries["output"] = s
                self.retained_summaries[f"{instruction.node_path}:output"] = s
                for p_name in instruction.output_ports:
                    self.value_table[(instruction.node_path, p_name)] = res
                    self.value_table[(instruction.node_id, p_name)] = res
                    self.retained_summaries[f"{instruction.node_path}:{p_name}"] = s

            duration_ns = time.time_ns() - start_ns
            self.current_instruction_idx += 1
            self.last_active_timestamp = time.time()

            finished_evt = await self.emit_event(
                TraceEventType.NODE_FINISHED,
                node_path=instruction.node_path,
                duration_ns=duration_ns,
                inputs=input_summaries,
                outputs=output_summaries,
            )

            if self.current_instruction_idx < len(self.plan.instructions):
                next_inst = self.plan.instructions[self.current_instruction_idx]
                if next_inst.node_id in self.breakpoints or next_inst.node_path in self.breakpoints:
                    self.state = "paused"
                    await self.emit_event(TraceEventType.NODE_PAUSED, node_path=next_inst.node_path)
            else:
                self.state = "completed"
                await self.emit_event(TraceEventType.RUN_FINISHED)

            return finished_evt

        except Exception as e:
            self.state = "error"
            return await self.emit_event(
                TraceEventType.NODE_FAILED,
                node_path=instruction.node_path,
                error=str(e),
            )

    async def run_until_breakpoint_or_end(self, speed_delay: float = 0.0) -> None:
        self.state = "running"
        await self.emit_event(TraceEventType.RUN_STARTED)

        while self.state == "running" and self.current_instruction_idx < len(
            self.plan.instructions
        ):
            await self.step_single()
            if speed_delay > 0 and self.state == "running":
                await asyncio.sleep(speed_delay)

    def stop(self) -> None:
        self.state = "idle"
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
        self.value_table.clear()
        self.retained_summaries.clear()


class TrainingSession(RuntimeSession):
    """Stateful Blueprint Training Session."""

    def __init__(
        self,
        session_id: str,
        project: Project,
        device: str = "cpu",
    ):
        super().__init__(session_id, project, device=device)

        cfg = project.model.config
        training_cfg = getattr(project.model, "training", None)

        self.learning_rate: float = (
            getattr(training_cfg, "learning_rate", 6e-4) if training_cfg else 6e-4
        )
        self.weight_decay: float = (
            getattr(training_cfg, "weight_decay", 0.1) if training_cfg else 0.1
        )
        self.grad_clip: float = getattr(training_cfg, "grad_clip", 1.0) if training_cfg else 1.0
        self.precision: str = getattr(training_cfg, "precision", "fp32") if training_cfg else "fp32"
        self.seed: int = getattr(training_cfg, "seed", 1337) if training_cfg else 1337
        self.max_epochs: int = getattr(training_cfg, "max_epochs", 10) if training_cfg else 10
        self.max_steps: int = getattr(training_cfg, "max_steps", 1000) or 1000
        self.batch_size: int = getattr(training_cfg, "batch_size", 8) if training_cfg else 8
        self.eval_interval: int = getattr(training_cfg, "eval_interval", 50) if training_cfg else 50
        self.checkpoint_interval: int = (
            getattr(training_cfg, "checkpoint_interval", 100) if training_cfg else 100
        )

        self.block_size: int = cfg.get("block_size", 8)
        self.vocab_size: int = cfg.get("vocab_size", 32)

        self.model.to(self.device)
        # Determine input feature type
        root_g = project.model.graphs.get(project.model.root_graph_id)
        self.has_token_in = (
            any(
                n.definition_id
                in (
                    "builtin.token_input@1",
                    "builtin.nanogpt_input_embeddings@1",
                    "builtin.dataset_source@1",
                    "builtin.dataloader@1",
                    "builtin.embedding@1",
                )
                or "token" in n.id
                or "emb" in n.id
                for n in root_g.nodes
            )
            if root_g
            else True
        )
        num_samples = 2000
        if root_g:
            for n in root_g.nodes:
                if n.definition_id == "builtin.dataloader@1":
                    b_val = evaluate_value(n.properties.get("batch_size", self.batch_size), cfg)
                    if isinstance(b_val, (int, float)) and int(b_val) > 0:
                        self.batch_size = int(b_val)
                elif n.definition_id == "builtin.dataset_source@1":
                    n_val = evaluate_value(n.properties.get("num_samples", num_samples), cfg)
                    if isinstance(n_val, (int, float)) and int(n_val) > 0:
                        num_samples = int(n_val)
        in_features = int(cfg.get("in_features", cfg.get("in_dim", cfg.get("n_embd", 16))))
        torch.manual_seed(self.seed)
        init_nanogpt_weights(self.model, n_layer=cfg.get("n_layer", 2))

        is_shakespeare = False
        for g in project.model.graphs.values():
            for n in g.nodes:
                if n.definition_id == "builtin.dataset_source@1":
                    if n.properties.get("synthetic") is False or n.properties.get("dataset_name") == "tiny_shakespeare":
                        is_shakespeare = True
                        break

        if is_shakespeare:
            self.load_dataset_by_name("tiny_shakespeare")
        elif self.has_token_in:
            self.dataset_x = torch.randint(0, self.vocab_size, (num_samples, self.block_size))
            self.dataset_y = torch.randint(0, self.vocab_size, (num_samples, self.block_size))
            val_samples = 200
            self.val_x = torch.randint(0, self.vocab_size, (val_samples, self.block_size))
            self.val_y = torch.randint(0, self.vocab_size, (val_samples, self.block_size))
        else:
            self.dataset_x = torch.randn(num_samples, in_features)
            self.dataset_y = torch.randn(num_samples, in_features)
            val_samples = 200
            self.val_x = torch.randn(val_samples, in_features)
            self.val_y = torch.randn(val_samples, in_features)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )

        def lr_lambda(current_step: int):
            warmup_steps = 20
            total = max(self.max_steps, 100)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if self.precision == "fp16" and "cuda" in self.device
            else None
        )
        self.step = 0
        self.epoch = 0
        self.loss_history: List[Dict[str, Any]] = []
        self.best_loss: Optional[float] = None
        self.metrics = TrainingMetrics()
        self.node_gradient_norms: Dict[str, float] = {}
        self.parameter_norms: Dict[str, float] = {}

    def update_hyperparameters(
        self,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        grad_clip: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        if learning_rate is not None:
            self.learning_rate = learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = learning_rate
        if weight_decay is not None:
            self.weight_decay = weight_decay
            for pg in self.optimizer.param_groups:
                pg["weight_decay"] = weight_decay
        if grad_clip is not None:
            self.grad_clip = grad_clip
        if batch_size is not None:
            self.batch_size = max(1, int(batch_size))
    async def step_batch(self) -> Optional[TraceEvent]:
        if self.step >= self.max_steps:
            self.state = "completed"
            return await self.emit_event(
                TraceEventType.TRAIN_FINISHED,
                metrics=self.metrics,
            )

        start_ns = time.time_ns()
        self.model.train()

        total_batches = max(1, self.dataset_x.size(0) // self.batch_size)
        batch_idx = self.step % total_batches
        b_start = batch_idx * self.batch_size
        b_end = b_start + self.batch_size

        x = self.dataset_x[b_start:b_end].to(self.device)
        y = self.dataset_y[b_start:b_end].to(self.device)

        self.value_table[("dataset", "dataset")] = self.dataset_x
        self.value_table[("dataloader", "batch_x")] = x
        self.value_table[("dataloader", "batch_y")] = y
        self.value_table[("input", "output")] = x
        self.value_table[("target", "output")] = y

        self.optimizer.zero_grad(set_to_none=True)

        device_type = (
            "cuda" if "cuda" in self.device else ("mps" if "mps" in self.device else "cpu")
        )
        amp_dtype = (
            torch.bfloat16
            if self.precision == "bf16"
            else (torch.float16 if self.precision == "fp16" else torch.float32)
        )
        use_amp = self.precision in ("fp16", "bf16") and device_type in ("cuda", "cpu")

        loss_tensor = None
        total_norm_val = 0.0

        try:
            has_exec_plan = bool(self.plan.exec_instructions)
            instructions_to_run = (
                self.plan.exec_instructions if has_exec_plan else self.plan.instructions
            )
            exec_backward_done = False
            exec_optimizer_done = False

            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                for instruction in instructions_to_run:
                    inputs: Dict[str, Any] = {}
                    bindings = instruction.data_input_bindings or [
                        b for b in instruction.input_bindings if b.kind != "exec"
                    ]
                    for b in bindings:
                        key1 = (b.source_node_path, b.source_port_id)
                        src_short = b.source_node_path.split("/")[-1]
                        key2 = (src_short, b.source_port_id)
                        val = self.value_table.get(key1, self.value_table.get(key2))
                        if val is not None:
                            inputs[b.port_id] = val
                    def_id = instruction.definition_id

                    if def_id in {
                        "builtin.tensor_input@1",
                        "builtin.token_input@1",
                        "builtin.module_input@1",
                    }:
                        self.value_table[(instruction.node_path, "output")] = x
                        self.value_table[(instruction.node_id, "output")] = x
                        for port_id in instruction.data_output_ports or instruction.output_ports:
                            self.value_table[(instruction.node_path, port_id)] = x
                            self.value_table[(instruction.node_id, port_id)] = x

                    elif def_id == "builtin.target_input@1":
                        self.value_table[(instruction.node_path, "output")] = y
                        self.value_table[(instruction.node_id, "output")] = y
                        for port_id in instruction.data_output_ports or instruction.output_ports:
                            self.value_table[(instruction.node_path, port_id)] = y
                            self.value_table[(instruction.node_id, port_id)] = y

                    elif def_id == "builtin.dataloader@1":
                        self.value_table[(instruction.node_path, "batch_x")] = x
                        self.value_table[(instruction.node_id, "batch_x")] = x
                        self.value_table[(instruction.node_path, "batch_y")] = y
                        self.value_table[(instruction.node_id, "batch_y")] = y

                    elif def_id == "builtin.dataset_source@1":
                        self.value_table[(instruction.node_path, "dataset")] = self.dataset_x
                        self.value_table[(instruction.node_id, "dataset")] = self.dataset_x

                    elif def_id == "builtin.zero_grad@1":
                        self.optimizer.zero_grad(set_to_none=True)

                    elif def_id == "builtin.backward@1":
                        loss_to_back = inputs.get("loss", loss_tensor)
                        if loss_to_back is not None:
                            if self.scaler is not None:
                                self.scaler.scale(loss_to_back).backward()
                                self.scaler.unscale_(self.optimizer)
                            else:
                                loss_to_back.backward()
                            exec_backward_done = True

                    elif def_id == "builtin.clip_gradients@1":
                        max_n = float(self.grad_clip)
                        norm_res = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=max_n
                        )
                        total_norm_val = (
                            float(norm_res.item())
                            if isinstance(norm_res, torch.Tensor)
                            else float(norm_res)
                        )

                    elif def_id == "builtin.optimizer_step@1":
                        if self.scaler is not None:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        exec_optimizer_done = True

                    elif def_id == "builtin.cosine_annealing_lr@1":
                        self.scheduler.step()

                    elif (
                        instruction.module_key and instruction.module_key in self.model.module_dict
                    ):
                        mod = self.model.module_dict[instruction.module_key]
                        if instruction.is_repeat and isinstance(mod, nn.ModuleList):
                            in_val = inputs.get("input")
                            if in_val is None and inputs:
                                in_val = next(iter(inputs.values()))
                            curr_x = in_val
                            for blk in mod:
                                curr_x = blk(curr_x)
                            res = curr_x
                        elif instruction.is_composite and isinstance(mod, CompiledGraphModule):
                            res = mod(**inputs)
                        elif "q" in inputs and "k" in inputs and "v" in inputs:
                            res = mod(inputs["q"], inputs["k"], inputs["v"])
                        elif "input" in inputs and len(inputs) == 1:
                            res = mod(inputs["input"])
                        elif len(inputs) == 1:
                            res = mod(next(iter(inputs.values())))
                        elif inputs:
                            res = mod(**inputs)
                        else:
                            res = None

                        if isinstance(res, dict):
                            for p_id, p_val in res.items():
                                self.value_table[(instruction.node_path, p_id)] = p_val
                                self.value_table[(instruction.node_id, p_id)] = p_val
                                if p_id == "loss" and isinstance(p_val, torch.Tensor):
                                    loss_tensor = p_val
                        elif res is not None:
                            self.value_table[(instruction.node_path, "output")] = res
                            self.value_table[(instruction.node_id, "output")] = res
                            for port_id in (
                                instruction.data_output_ports or instruction.output_ports
                            ):
                                self.value_table[(instruction.node_path, port_id)] = res
                                self.value_table[(instruction.node_id, port_id)] = res

                    elif instruction.functional_fn:
                        if len(inputs) == 2 and "logits" in inputs and "targets" in inputs:
                            res = instruction.functional_fn(inputs["logits"], inputs["targets"])
                        elif len(inputs) == 2 and "a" in inputs and "b" in inputs:
                            res = instruction.functional_fn(inputs["a"], inputs["b"])
                        elif len(inputs) == 1:
                            res = instruction.functional_fn(next(iter(inputs.values())))
                        else:
                            res = instruction.functional_fn(**inputs)

                        if isinstance(res, dict):
                            for port_id, value in res.items():
                                self.value_table[(instruction.node_path, port_id)] = value
                                self.value_table[(instruction.node_id, port_id)] = value
                        elif res is not None:
                            self.value_table[(instruction.node_path, "output")] = res
                            self.value_table[(instruction.node_id, "output")] = res
                            for port_id in (
                                instruction.data_output_ports or instruction.output_ports
                            ):
                                self.value_table[(instruction.node_path, port_id)] = res
                                self.value_table[(instruction.node_id, port_id)] = res
                            if def_id == "builtin.cross_entropy_loss@1" and isinstance(
                                res, torch.Tensor
                            ):
                                loss_tensor = res
                                self.value_table[(instruction.node_path, "loss")] = res
                                self.value_table[(instruction.node_id, "loss")] = res

            if loss_tensor is None:
                loss_tensor = self.value_table.get(("node_cross_entropy", "loss"))
                if loss_tensor is None:
                    for (_, key), val in self.value_table.items():
                        if key == "loss" and isinstance(val, torch.Tensor):
                            loss_tensor = val
                            break

            if loss_tensor is None:
                terminal_tensor = None
                for instruction in reversed(instructions_to_run):
                    if not instruction.is_terminal_output:
                        continue
                    for binding in instruction.data_input_bindings:
                        terminal_tensor = self.value_table.get(
                            (binding.source_node_path, binding.source_port_id)
                        )
                        if isinstance(terminal_tensor, torch.Tensor):
                            break
                    if isinstance(terminal_tensor, torch.Tensor):
                        break

                if isinstance(terminal_tensor, torch.Tensor):
                    loss_tensor = (
                        torch.nn.functional.mse_loss(terminal_tensor, y)
                        if terminal_tensor.shape == y.shape
                        else terminal_tensor.square().mean()
                    )

            if loss_tensor is None:
                raise RuntimeError("Training plan produced no differentiable loss")

            if not has_exec_plan or not exec_backward_done:
                if self.scaler is not None:
                    self.scaler.scale(loss_tensor).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    loss_tensor.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip,
                )
                total_norm_val = (
                    float(total_norm.item())
                    if isinstance(total_norm, torch.Tensor)
                    else float(total_norm)
                )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
            elif not exec_optimizer_done:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()

            node_grad_norms = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    node_grad_norms[name] = float(param.grad.norm().item())
            self.node_gradient_norms = node_grad_norms
            self.parameter_norms = {
                name: float(p.data.norm().item())
                for name, p in self.model.named_parameters()
            }

            if math.isnan(total_norm_val) or math.isinf(total_norm_val) or total_norm_val > 100.0:
                grad_status = "exploding"
            elif total_norm_val < 1e-4:
                grad_status = "vanishing"
            else:
                grad_status = "healthy"

            current_lr = self.optimizer.param_groups[0]["lr"]
            duration_ns = time.time_ns() - start_ns
            duration_s = max(duration_ns / 1e9, 1e-6)
            tokens_per_sec = (self.batch_size * self.block_size) / duration_s
            loss_val = float(loss_tensor.item()) if loss_tensor is not None else 0.0

            if self.best_loss is None or loss_val < self.best_loss:
                self.best_loss = loss_val

            self.loss_history.append(
                {
                    "step": self.step,
                    "loss": loss_val,
                    "lr": current_lr,
                    "grad_norm": total_norm_val,
                    "tokens_per_sec": tokens_per_sec,
                }
            )

            self.metrics = TrainingMetrics(
                epoch=self.epoch,
                step=self.step,
                loss=loss_val,
                learning_rate=current_lr,
                grad_norm=total_norm_val,
                grad_status=grad_status,
                tokens_per_sec=tokens_per_sec,
                step_time_ms=duration_ns / 1e6,
                best_loss=self.best_loss,
            )

            evt = await self.emit_event(
                TraceEventType.BATCH_ENDED,
                duration_ns=duration_ns,
                metrics=self.metrics,
            )

            self.step += 1
            self.last_active_timestamp = time.time()
            if batch_idx == total_batches - 1:
                self.epoch += 1

            if self.step >= self.max_steps:
                self.state = "completed"
                await self.emit_event(
                    TraceEventType.TRAIN_FINISHED,
                    metrics=self.metrics,
                )

            return evt
        except Exception as e:
            self.state = "error"
            return await self.emit_event(
                TraceEventType.NODE_FAILED,
                error=str(e),
                metrics=self.metrics,
            )

    async def step_epoch(self) -> List[TraceEvent]:
        total_batches = max(1, self.dataset_x.size(0) // self.batch_size)
        events = []
        for _ in range(total_batches):
            if self.state in ("paused", "completed", "error"):
                break
            evt = await self.step_batch()
            if evt:
                events.append(evt)
        return events

    async def run_training_loop(
        self, max_steps: Optional[int] = None, speed_delay: float = 0.0
    ) -> None:
        target_steps = (self.step + max_steps) if max_steps else self.max_steps
        if target_steps > self.max_steps:
            self.max_steps = target_steps
        self.state = "running"
        await self.emit_event(TraceEventType.TRAIN_STARTED, metrics=self.metrics)

        while self.state == "running" and self.step < target_steps:
            await self.step_batch()
            delay = speed_delay if speed_delay > 0 else 0.02
            if self.state == "running":
                await asyncio.sleep(delay)

        if self.state == "running" and self.step >= target_steps:
            self.state = "completed"
            await self.emit_event(TraceEventType.TRAIN_FINISHED, metrics=self.metrics)
    def pause(self) -> None:
        self.state = "paused"

    def resume(self, speed_delay: float = 0.0) -> None:
        self.state = "running"
        self._launch_background(self.run_training_loop(speed_delay=speed_delay))

    def load_dataset_by_name(self, name: str, val_fraction: float = 0.1) -> int:
        if name == "tiny_shakespeare":
            from neural_blueprint.data.tiny_shakespeare import TINY_SHAKESPEARE, load_token_dataset
            from neural_blueprint.runtime.tokenizer import CharacterTokenizer
            tok = CharacterTokenizer(self.vocab_size)
            split_idx = int(len(TINY_SHAKESPEARE) * (1.0 - val_fraction))
            train_text = TINY_SHAKESPEARE[:split_idx]
            val_text = TINY_SHAKESPEARE[split_idx:]
            self.dataset_x, self.dataset_y = load_token_dataset(train_text, tok, self.block_size)
            self.val_x, self.val_y = load_token_dataset(val_text, tok, self.block_size)
        else:
            in_features = int(self.project.model.config.get("in_features", self.project.model.config.get("in_dim", self.project.model.config.get("n_embd", 16))))
            num_samples = 2000
            val_samples = 200
            if self.has_token_in:
                self.dataset_x = torch.randint(0, self.vocab_size, (num_samples, self.block_size))
                self.dataset_y = torch.randint(0, self.vocab_size, (num_samples, self.block_size))
                self.val_x = torch.randint(0, self.vocab_size, (val_samples, self.block_size))
                self.val_y = torch.randint(0, self.vocab_size, (val_samples, self.block_size))
            else:
                self.dataset_x = torch.randn(num_samples, in_features)
                self.dataset_y = torch.randn(num_samples, in_features)
                self.val_x = torch.randn(val_samples, in_features)
                self.val_y = torch.randn(val_samples, in_features)
        return len(self.dataset_x)

    def stop(self) -> None:
        self.state = "idle"
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()

    @torch.no_grad()
    async def evaluate_validation(self) -> float:
        self.model.eval()
        try:
            val_batch_size = min(len(self.val_x), max(1, self.batch_size * 2))
            bx = self.val_x[:val_batch_size].to(self.device)
            by = self.val_y[:val_batch_size].to(self.device)

            input_name = "token_ids" if self.has_token_in else "input"
            out = self.model(**{input_name: bx})
            if isinstance(out, dict):
                logits = out.get("logits", next(iter(out.values())))
            elif isinstance(out, (list, tuple)):
                logits = out[0]
            else:
                logits = out

            if self.has_token_in and logits.dim() == 3:
                val_loss = float(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1)).item())
            else:
                val_loss = float(torch.nn.functional.mse_loss(logits, by).item())

            if self.metrics:
                self.metrics.val_loss = val_loss
            await self.emit_event(TraceEventType.VALIDATION_FINISHED, metrics=self.metrics)
            return val_loss
        finally:
            self.model.train()

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        if path is None:
            ckpt_path = resolve_sandbox_path(f"ckpt_step_{self.step}.pt", "checkpoints")
        else:
            ckpt_path = Path(path)

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "best_loss": self.best_loss,
            "config": self.project.model.config,
        }
        torch.save(payload, str(ckpt_path))
        return str(ckpt_path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        ckpt_path = Path(path)
        if not ckpt_path.is_absolute():
            try:
                ckpt_path = resolve_sandbox_path(path, "checkpoints")
            except PathValidationError:
                raise
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        payload = torch.load(str(ckpt_path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scheduler_state_dict" in payload:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.step = payload.get("step", 0)
        self.epoch = payload.get("epoch", 0)
        self.loss_history = payload.get("loss_history", [])
        self.best_loss = payload.get("best_loss")

        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss_history[-1]["loss"] if self.loss_history else None,
            "best_loss": self.best_loss,
        }


class SessionCapacityError(RuntimeError):
    """Raised when the session manager cannot accept another session."""


class SessionManager:
    """Singleton session registry managing active compiled runtime and training sessions."""

    MAX_SESSIONS = 32
    _PRUNABLE_STATES = frozenset({"idle", "stopped", "completed"})

    def __init__(self):
        self._sessions: Dict[str, Union[RuntimeSession, TrainingSession]] = {}

    def _ensure_capacity(self) -> None:
        if len(self._sessions) < self.MAX_SESSIONS:
            return

        candidates = [
            (session_id, session)
            for session_id, session in self._sessions.items()
            if session.state in self._PRUNABLE_STATES
        ]
        if not candidates:
            raise SessionCapacityError(
                f"Session limit of {self.MAX_SESSIONS} reached and all sessions are active"
            )

        oldest_id = min(candidates, key=lambda item: item[1].last_active_timestamp)[0]
        self.remove_session(oldest_id)

    def _register_session(
        self, session_id: str, session: Union[RuntimeSession, TrainingSession]
    ) -> None:
        self._ensure_capacity()
        existing = self._sessions.get(session_id)
        if existing is not None and existing is not session:
            existing.stop()
        self._sessions[session_id] = session

    def create_session(
        self, session_id: str, project: Project, device: str = "cpu"
    ) -> RuntimeSession:
        session = RuntimeSession(session_id, project, device=device)
        self._register_session(session_id, session)
        return session

    def create_training_session(
        self, session_id: str, project: Project, device: str = "cpu"
    ) -> TrainingSession:
        validate_training_capabilities(project)
        session = TrainingSession(session_id, project, device=device)
        self._register_session(session_id, session)
        return session

    def get_session(self, session_id: str) -> Optional[Union[RuntimeSession, TrainingSession]]:
        return self._sessions.get(session_id)

    def get_training_session(self, session_id: str) -> Optional[TrainingSession]:
        session = self._sessions.get(session_id)
        if isinstance(session, TrainingSession):
            return session
        return None

    def remove_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].stop()
            del self._sessions[session_id]

    def invalidate_stale_sessions(self, current_graph_hash: str) -> None:
        stale_ids = [
            s_id for s_id, s in self._sessions.items() if s.graph_hash != current_graph_hash
        ]
        for s_id in stale_ids:
            self.remove_session(s_id)


global_session_manager = SessionManager()
