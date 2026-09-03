"""Forward-only inference engine for compiled Blueprint projects and live sessions."""

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from neural_blueprint.ir.models import Project
from neural_blueprint.runtime.module import CompiledGraphModule
from neural_blueprint.tracing.collector import TensorSummarizer
from neural_blueprint.tracing.debugger import TrainingSession

# Exact definition IDs that terminate forward-only interpretation (dual-flow
# builders archs 7/16/17/25). Substring matching is unsafe: "lr" matches
# builtin.linear@1.
_STOP_DEFINITION_IDS = frozenset(
    {
        "builtin.cross_entropy_loss@1",
        "builtin.backward@1",
        "builtin.clip_gradients@1",
        "builtin.optimizer_step@1",
        "builtin.zero_grad@1",
        "builtin.metric_logger@1",
        "builtin.cosine_annealing_lr@1",
        "builtin.linear_warmup_scheduler@1",
        "builtin.step_lr@1",
        "builtin.comment@1",
    }
)


class InferenceEngine:
    """Runs a no-grad forward pass on a Blueprint project or live session.

    Exactly one of ``project`` (fresh, seed-reproducible weights via a private
    TrainingSession) or ``session`` (a live registered session's current weights).
    """

    def __init__(
        self,
        project: Optional[Project] = None,
        session: Optional[TrainingSession] = None,
        device: str = "cpu",
    ) -> None:
        if (project is None) == (session is None):
            raise ValueError("Provide exactly one of 'project' or 'session'")
        if session is not None:
            self.session = session
            self.project = session.project
        else:
            seed = getattr(getattr(project.model, "training", None), "seed", None)
            if seed is not None:
                torch.manual_seed(seed)
            self.session = TrainingSession(
                session_id=f"infer_{project.project.id}",
                project=project,
                device=device,
            )
            self.project = project

    def _input_name(self) -> str:
        root_graph = self.project.model.graphs.get(self.project.model.root_graph_id)
        if root_graph is not None:
            for node in root_graph.nodes:
                if node.definition_id == "builtin.token_input@1":
                    return "token_ids"
                if node.definition_id == "builtin.tensor_input@1":
                    return "input"
        if self.session.plan.input_port_names:
            return self.session.plan.input_port_names[0]
        return "input"

    def _normalize_inputs(self, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Converts raw values to device tensors, mirroring prepare_run's conversion."""
        normalized: Dict[str, Any] = {}
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
                v = v.to(self.session.device)
            normalized[k] = v

        input_name = self._input_name()
        if input_name in normalized:
            x = normalized[input_name]
        elif "x" in normalized:
            x = normalized["x"]
        elif normalized:
            x = next(iter(normalized.values()))
        else:
            x = None

        y = None
        for key in ("target", "targets", "y", "label", "labels"):
            if key in normalized:
                y = normalized[key]
                break

        if x is None:
            x = self.session.dataset_x[: self.session.batch_size].to(self.session.device)
        if y is None:
            y = self.session.dataset_y[: self.session.batch_size].to(self.session.device)
        return x, y

    async def infer(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        session = self.session
        root_graph = self.project.model.graphs.get(self.project.model.root_graph_id)
        mode = "interpreter" if root_graph is not None and root_graph.kind == "training_event" else "whole_graph"

        x, y = self._normalize_inputs(inputs or {})

        session.model.eval()
        saved_breakpoints = set(session.breakpoints)
        session.breakpoints.clear()

        try:
            if mode == "whole_graph":
                with torch.no_grad():
                    out = session.model(**{self._input_name(): x})
                if isinstance(out, torch.Tensor):
                    outputs: Dict[str, torch.Tensor] = {"output": out}
                elif isinstance(out, Mapping):
                    outputs = {k: v for k, v in out.items() if isinstance(v, torch.Tensor)}
                elif isinstance(out, (tuple, list)):
                    outputs = {f"output_{i}": t for i, t in enumerate(out) if isinstance(t, torch.Tensor)}
                else:
                    outputs = {}
            else:
                outputs = await self._run_interpreter(x, y)
        finally:
            session.breakpoints = saved_breakpoints
            session.model.train()
            session.state = "idle"

        return {
            "outputs": {k: TensorSummarizer.summarize(t).model_dump() for k, t in outputs.items()},
            "tensor_count": len(outputs),
            "graph_hash": session.graph_hash,
            "mode": mode,
        }

    async def _run_interpreter(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward-only interpretation of a training_event graph.

        Replicates step_batch's value_table seeding and per-instruction execution
        (with exec bindings filtered), stopping before training-mutating nodes.
        """
        session = self.session
        session.prepare_run({})

        # Seeding block replicated from step_batch (debugger.py:404-408).
        session.value_table[("dataset", "dataset")] = session.dataset_x
        session.value_table[("dataloader", "batch_x")] = x
        session.value_table[("dataloader", "batch_y")] = y
        session.value_table[("input", "output")] = x
        session.value_table[("target", "output")] = y

        instructions = (
            session.plan.exec_instructions
            if session.plan.exec_instructions
            else session.plan.instructions
        )
        stopped: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for instruction in instructions:
                def_id = instruction.definition_id
                if def_id in _STOP_DEFINITION_IDS:
                    bindings = instruction.data_input_bindings or [
                        b for b in instruction.input_bindings if b.kind != "exec"
                    ]
                    for binding in bindings:
                        if binding.port_id != "logits":
                            continue
                        key1 = (binding.source_node_path, binding.source_port_id)
                        src_short = binding.source_node_path.split("/")[-1]
                        key2 = (src_short, binding.source_port_id)
                        val = session.value_table.get(key1, session.value_table.get(key2))
                        if isinstance(val, torch.Tensor):
                            stopped["logits"] = val
                            break
                    if "logits" not in stopped:
                        for binding in bindings:
                            key1 = (binding.source_node_path, binding.source_port_id)
                            src_short = binding.source_node_path.split("/")[-1]
                            key2 = (src_short, binding.source_port_id)
                            val = session.value_table.get(key1, session.value_table.get(key2))
                            if isinstance(val, torch.Tensor):
                                stopped["logits"] = val
                                break
                    break

                bindings = instruction.data_input_bindings or [
                    b for b in instruction.input_bindings if b.kind != "exec"
                ]
                inputs: Dict[str, Any] = {}
                for binding in bindings:
                    key1 = (binding.source_node_path, binding.source_port_id)
                    src_short = binding.source_node_path.split("/")[-1]
                    key2 = (src_short, binding.source_port_id)
                    val = session.value_table.get(key1, session.value_table.get(key2))
                    if val is not None:
                        inputs[binding.port_id] = val

                # Input-node interception (debugger.py:448-474).
                if def_id in {
                    "builtin.tensor_input@1",
                    "builtin.token_input@1",
                    "builtin.module_input@1",
                }:
                    session.value_table[(instruction.node_path, "output")] = x
                    session.value_table[(instruction.node_id, "output")] = x
                    for port_id in instruction.data_output_ports or instruction.output_ports:
                        session.value_table[(instruction.node_path, port_id)] = x
                        session.value_table[(instruction.node_id, port_id)] = x
                elif def_id == "builtin.target_input@1":
                    session.value_table[(instruction.node_path, "output")] = y
                    session.value_table[(instruction.node_id, "output")] = y
                    for port_id in instruction.data_output_ports or instruction.output_ports:
                        session.value_table[(instruction.node_path, port_id)] = y
                        session.value_table[(instruction.node_id, port_id)] = y
                elif def_id == "builtin.dataloader@1":
                    session.value_table[(instruction.node_path, "batch_x")] = x
                    session.value_table[(instruction.node_id, "batch_x")] = x
                    session.value_table[(instruction.node_path, "batch_y")] = y
                    session.value_table[(instruction.node_id, "batch_y")] = y
                elif def_id == "builtin.dataset_source@1":
                    session.value_table[(instruction.node_path, "dataset")] = session.dataset_x
                    session.value_table[(instruction.node_id, "dataset")] = session.dataset_x
                elif (
                    instruction.module_key
                    and instruction.module_key in session.model.module_dict
                ):
                    mod = session.model.module_dict[instruction.module_key]
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
                        for port_id, port_val in res.items():
                            session.value_table[(instruction.node_path, port_id)] = port_val
                            session.value_table[(instruction.node_id, port_id)] = port_val
                    elif res is not None:
                        session.value_table[(instruction.node_path, "output")] = res
                        session.value_table[(instruction.node_id, "output")] = res
                        for port_id in instruction.data_output_ports or instruction.output_ports:
                            session.value_table[(instruction.node_path, port_id)] = res
                            session.value_table[(instruction.node_id, port_id)] = res
                elif instruction.functional_fn:
                    if len(inputs) == 2 and "a" in inputs and "b" in inputs:
                        res = instruction.functional_fn(inputs["a"], inputs["b"])
                    elif len(inputs) == 2 and "logits" in inputs and "targets" in inputs:
                        res = instruction.functional_fn(inputs["logits"], inputs["targets"])
                    elif len(inputs) == 1:
                        res = instruction.functional_fn(next(iter(inputs.values())))
                    else:
                        try:
                            res = instruction.functional_fn(**inputs)
                        except TypeError:
                            res = instruction.functional_fn(*list(inputs.values()))
                    if isinstance(res, dict):
                        for port_id, port_val in res.items():
                            session.value_table[(instruction.node_path, port_id)] = port_val
                            session.value_table[(instruction.node_id, port_id)] = port_val
                    elif res is not None:
                        session.value_table[(instruction.node_path, "output")] = res
                        session.value_table[(instruction.node_id, "output")] = res

        outputs: Dict[str, torch.Tensor] = dict(stopped)
        if not outputs:
            for key, val in reversed(list(session.value_table.items())):
                if isinstance(val, torch.Tensor) and key[1] in ("output", "logits"):
                    outputs.setdefault(key[1], val)
                    break

        session.state = "idle"
        return outputs
