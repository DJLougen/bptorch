"""Compiled PyTorch Module executing canonical graph plans with hierarchical submodules."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neural_blueprint.ir.models import WeightBinding
from neural_blueprint.runtime.compiler import ExecutionPlan


class CompiledGraphModule(nn.Module):
    """
    Executable PyTorch module constructed directly from canonical IR.
    Registers submodules into nn.ModuleDict/nn.ModuleList for standard PyTorch parameter management.
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        modules: Dict[str, Union[nn.Module, nn.ModuleList]],
        weight_bindings: Optional[List[WeightBinding]] = None,
    ):
        super().__init__()
        self.plan = plan
        self.module_dict = nn.ModuleDict(modules)

        # Apply weight tying bindings
        if weight_bindings:
            self._apply_weight_bindings(weight_bindings)

    def _resolve_module_by_path(self, path: str) -> Optional[nn.Module]:
        """Resolves a nested submodule path (e.g. 'node_input_embeddings/node_tok_emb')."""
        parts = path.split("/")
        curr: Any = self
        for part in parts:
            sanitized = part.replace("-", "_").replace(".", "_")
            if isinstance(curr, CompiledGraphModule):
                if sanitized in curr.module_dict:
                    curr = curr.module_dict[sanitized]
                else:
                    return None
            elif isinstance(curr, nn.ModuleDict):
                if sanitized in curr:
                    curr = curr[sanitized]
                else:
                    return None
            else:
                return None
        return curr if isinstance(curr, nn.Module) else None

    def _apply_weight_bindings(self, bindings: List[WeightBinding]) -> None:
        """Ties weights across modules by sharing parameter references directly."""
        for b in bindings:
            if b.mode == "share":
                src_mod = self._resolve_module_by_path(b.source.node_id)
                tgt_mod = self._resolve_module_by_path(b.target.node_id)

                if src_mod is not None and tgt_mod is not None:
                    src_p = getattr(src_mod, b.source.parameter, None)
                    tgt_p = getattr(tgt_mod, b.target.parameter, None)
                    if src_p is not None and tgt_p is not None:
                        # Tie weights by sharing storage/parameter reference
                        setattr(tgt_mod, b.target.parameter, src_p)

    def forward(
        self, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor], None]:
        """
        Executes a forward pass over the compiled execution plan.
        Supports tensor inputs, subgraphs, repeated modules, and loss calculation.
        """
        value_table: Dict[Tuple[str, str], Any] = {}
        outputs: Dict[str, torch.Tensor] = {}

        def store_result(inst, res):
            if isinstance(res, dict):
                for p_id, p_val in res.items():
                    value_table[(inst.node_path, p_id)] = p_val
                    value_table[(inst.node_id, p_id)] = p_val
            elif res is not None:
                value_table[(inst.node_path, "output")] = res
                value_table[(inst.node_id, "output")] = res
                for p_name in inst.output_ports:
                    value_table[(inst.node_path, p_name)] = res
                    value_table[(inst.node_id, p_name)] = res

        # Bind positional args if any
        if args and not kwargs:
            for i, arg in enumerate(args):
                if i < len(self.plan.input_port_names):
                    in_name = self.plan.input_port_names[i]
                    kwargs[in_name] = arg

        for instruction in self.plan.instructions:
            # 1. Handle input nodes
            is_input = instruction.definition_id in (
                "builtin.tensor_input@1",
                "builtin.token_input@1",
                "builtin.target_input@1",
                "builtin.module_input@1",
            )
            if is_input:
                val = None
                keys_to_try = [
                    instruction.input_name,
                    instruction.output_name,
                    instruction.node_id,
                    instruction.node_path,
                ]
                for k in keys_to_try:
                    if k and k in kwargs:
                        val = kwargs[k]
                        break

                if val is None and kwargs:
                    # Semantic aliases
                    if "token" in instruction.node_id and "token_ids" in kwargs:
                        val = kwargs["token_ids"]
                    elif "target" in instruction.node_id and "targets" in kwargs:
                        val = kwargs["targets"]
                    elif len(kwargs) == 1:
                        val = next(iter(kwargs.values()))

                if val is not None:
                    store_result(instruction, val)
                continue

            # Gather inputs for this instruction
            inputs: Dict[str, Any] = {}
            for binding in instruction.input_bindings:
                key1 = (binding.source_node_path, binding.source_port_id)
                src_short = binding.source_node_path.split("/")[-1]
                key2 = (src_short, binding.source_port_id)

                if key1 in value_table:
                    inputs[binding.port_id] = value_table[key1]
                elif key2 in value_table:
                    inputs[binding.port_id] = value_table[key2]

            # 2. Repeat module execution (sequential pass through ModuleList)
            if instruction.is_repeat and instruction.module_key:
                block_list = self.module_dict[instruction.module_key]
                in_val = inputs.get("input")
                if in_val is None and inputs:
                    in_val = next(iter(inputs.values()))

                if isinstance(block_list, nn.ModuleList) and in_val is not None:
                    curr_x = in_val
                    for block in block_list:
                        curr_x = block(curr_x)
                    store_result(instruction, curr_x)
                continue

            # 3. Composite module execution (nested CompiledGraphModule)
            if instruction.is_composite and instruction.module_key:
                child_mod = self.module_dict[instruction.module_key]
                child_res = child_mod(**inputs)
                store_result(instruction, child_res)
                continue

            # 4. Standard parameterized submodule execution
            if instruction.module_key and instruction.module_key in self.module_dict:
                mod = self.module_dict[instruction.module_key]
                if "q" in inputs and "k" in inputs and "v" in inputs:
                    res = mod(inputs["q"], inputs["k"], inputs["v"])
                elif "input" in inputs and len(inputs) == 1:
                    res = mod(inputs["input"])
                elif len(inputs) == 1:
                    res = mod(next(iter(inputs.values())))
                elif inputs:
                    try:
                        res = mod(**inputs)
                    except TypeError:
                        res = mod(*list(inputs.values()))
                else:
                    res = None

                if res is not None:
                    store_result(instruction, res)

            # 5. Functional tensor operation
            elif instruction.functional_fn:
                if len(inputs) == 2 and "a" in inputs and "b" in inputs:
                    res = instruction.functional_fn(inputs["a"], inputs["b"])
                elif len(inputs) == 2 and "logits" in inputs and "targets" in inputs:
                    res = instruction.functional_fn(inputs["logits"], inputs["targets"])
                elif len(inputs) == 3 and "q" in inputs and "k" in inputs and "v" in inputs:
                    res = instruction.functional_fn(inputs["q"], inputs["k"], inputs["v"])
                elif len(inputs) == 1:
                    res = instruction.functional_fn(next(iter(inputs.values())))
                else:
                    try:
                        res = instruction.functional_fn(**inputs)
                    except TypeError:
                        res = instruction.functional_fn(*list(inputs.values()))

                store_result(instruction, res)

            # 6. Terminal output node
            elif instruction.is_terminal_output:
                in_val = inputs.get("input")
                if in_val is None:
                    in_val = inputs.get("logits")
                if in_val is None:
                    in_val = inputs.get("loss")
                if in_val is None and inputs:
                    in_val = next(iter(inputs.values()))

                if in_val is not None and instruction.output_name:
                    outputs[instruction.output_name] = in_val

        if len(outputs) == 1:
            return next(iter(outputs.values()))
        elif outputs:
            return outputs
        if value_table:
            return next(reversed(value_table.values()))
        return None
