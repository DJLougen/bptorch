"""Hierarchical Graph Compiler transforming canonical Dual-Flow IR and subgraphs into PyTorch execution plans."""

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch.nn as nn

from neural_blueprint.ir.evaluator import evaluate_value
from neural_blueprint.ir.models import GraphDefinition, Project
from neural_blueprint.ir.serialization import serialize_project
from neural_blueprint.registry.base import NodeValidationContext
from neural_blueprint.registry.registry import NodeRegistry, global_registry
from neural_blueprint.shapes.engine import ShapePropagator

COMPOSITE_TYPE_MAP = {
    "builtin.nanogpt_input_embeddings@1": "graph_input_embeddings",
    "builtin.nanogpt_attention@1": "graph_attention",
    "builtin.nanogpt_mlp@1": "graph_mlp",
    "builtin.nanogpt_block@1": "graph_block",
    "builtin.nanogpt_stack@1": "graph_stack",
}


@dataclass
class InputBinding:
    port_id: str
    source_node_path: str
    source_port_id: str
    kind: str = "data"  # "data" or "exec"


@dataclass
class ExecutionInstruction:
    node_path: str
    node_id: str
    definition_id: str
    display_name: str
    input_bindings: List[InputBinding]
    output_ports: List[str]
    kind: str = "data"  # "data", "exec", "flow_control", "event", "optimization", "scheduler", "metric", "persistence"
    exec_in_bindings: List[InputBinding] = field(default_factory=list)
    exec_out_ports: List[str] = field(default_factory=list)
    data_input_bindings: List[InputBinding] = field(default_factory=list)
    data_output_ports: List[str] = field(default_factory=list)
    input_name: Optional[str] = None
    module_key: Optional[str] = None
    functional_fn: Optional[Callable[..., Any]] = None
    is_composite: bool = False
    is_repeat: bool = False
    is_terminal_output: bool = False
    output_name: Optional[str] = None


@dataclass
class ExecutionPlan:
    graph_hash: str
    instructions: List[ExecutionInstruction] = field(default_factory=list)
    input_port_names: List[str] = field(default_factory=list)
    output_port_names: List[str] = field(default_factory=list)
    exec_instructions: List[ExecutionInstruction] = field(default_factory=list)


class GraphCompiler:
    """Compiles a Project into an ExecutionPlan and instantiates CompiledGraphModule hierarchically."""

    def __init__(self, registry: Optional[NodeRegistry] = None):
        self.registry = registry or global_registry
        self.propagator = ShapePropagator(self.registry)

    def compute_graph_hash(self, project: Project) -> str:
        """Computes a deterministic SHA256 hash of the architecture IR."""
        data = serialize_project(project)
        model_data = data.get("model", {})
        encoded = json.dumps(model_data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _is_exec_port(
        self, port_id: str, node_def: Optional[Any] = None, is_output: bool = False
    ) -> bool:
        """Determines whether a given port is an Exec wire port."""
        if node_def:
            ports = node_def.output_ports({}) if is_output else node_def.input_ports({})
            for p in ports:
                if p.id == port_id:
                    if getattr(p, "kind", None) == "exec":
                        return True
                    if getattr(p, "kind", None) == "data":
                        return False
        # Fallback to port ID naming heuristics
        return (
            port_id.startswith("exec")
            or port_id.startswith("then_")
            or port_id
            in (
                "loop_body",
                "completed",
                "on_eval",
                "skip_eval",
                "continue_exec",
                "stop_exec",
                "true_branch",
                "false_branch",
                "reset",
            )
        )

    def topological_sort_dual_flow(self, graph: GraphDefinition) -> Tuple[List[str], bool]:
        """
        Sorts graph nodes in topological order honoring both Exec wire execution precedence
        and Data wire dependencies.
        """
        in_degree: Dict[str, int] = {node.id: 0 for node in graph.nodes}
        adj: Dict[str, Set[str]] = defaultdict(set)

        # Build precedence graph with higher priority for exec wires
        for edge in graph.edges:
            src_node = edge.source.node_id
            tgt_node = edge.target.node_id
            if src_node in in_degree and tgt_node in in_degree:
                if tgt_node not in adj[src_node]:
                    adj[src_node].add(tgt_node)
                    in_degree[tgt_node] += 1

        node_order = {node.id: index for index, node in enumerate(graph.nodes)}
        queue = deque(
            sorted(
                (node_id for node_id, degree in in_degree.items() if degree == 0),
                key=node_order.__getitem__,
            )
        )
        ordered: List[str] = []

        while queue:
            node_id = queue.popleft()
            ordered.append(node_id)
            for neighbor in sorted(adj[node_id], key=node_order.__getitem__):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return ordered, len(ordered) != len(graph.nodes)

    def compile_plan(
        self, project: Project, root_graph_id: Optional[str] = None
    ) -> Tuple[ExecutionPlan, Dict[str, Union[nn.Module, nn.ModuleList]]]:
        """Compiles the project into an ExecutionPlan and instantiates module tree."""
        from neural_blueprint.runtime.module import CompiledGraphModule

        graph_hash = self.compute_graph_hash(project)
        cfg = project.model.config
        target_root = root_graph_id or project.model.root_graph_id

        def compile_graph_recursive(
            graph_id: str,
            path_prefix: str = "",
        ) -> Tuple[ExecutionPlan, Dict[str, Union[nn.Module, nn.ModuleList]]]:
            graph = project.model.graphs.get(graph_id)
            if not graph:
                raise ValueError(f"Graph '{graph_id}' not found in project")

            ordered_ids, has_cycle = self.topological_sort_dual_flow(graph)
            if has_cycle:
                raise ValueError(f"Cannot compile graph with cycles: {graph_id}")
            nodes_by_id = {node.id: node for node in graph.nodes}
            inbound_edges: Dict[str, Dict[str, Tuple[str, str]]] = {}
            outbound_exec_node_ids: Set[str] = set()

            for edge in graph.edges:
                if edge.target.node_id not in inbound_edges:
                    inbound_edges[edge.target.node_id] = {}
                inbound_edges[edge.target.node_id][edge.target.port_id] = (
                    edge.source.node_id,
                    edge.source.port_id,
                )
                src_node = nodes_by_id.get(edge.source.node_id)
                src_def = self.registry.get(src_node.definition_id) if src_node else None
                if self._is_exec_port(edge.source.port_id, src_def, is_output=True):
                    outbound_exec_node_ids.add(edge.source.node_id)
            instructions: List[ExecutionInstruction] = []
            exec_instructions: List[ExecutionInstruction] = []
            modules: Dict[str, Union[nn.Module, nn.ModuleList]] = {}
            input_names: List[str] = []
            output_names: List[str] = []

            context = NodeValidationContext(
                model_config=cfg,
                graph_definitions=project.model.graphs,
            )

            for node_id in ordered_ids:
                node = nodes_by_id.get(node_id)
                if not node:
                    continue

                node_path = f"{path_prefix}/{node.id}" if path_prefix else node.id
                sanitized_key = node.id.replace("-", "_").replace(".", "_")

                node_def = self.registry.get(node.definition_id)
                subgraph_id = COMPOSITE_TYPE_MAP.get(node.definition_id)

                all_bindings: List[InputBinding] = []
                exec_bindings: List[InputBinding] = []
                data_bindings: List[InputBinding] = []

                for port_id, (src_id, src_port) in inbound_edges.get(node.id, {}).items():
                    src_full_path = f"{path_prefix}/{src_id}" if path_prefix else src_id
                    src_node = nodes_by_id.get(src_id)
                    src_def = self.registry.get(src_node.definition_id) if src_node else None
                    is_exec = self._is_exec_port(
                        port_id, node_def, is_output=False
                    ) or self._is_exec_port(src_port, src_def, is_output=True)

                    binding = InputBinding(
                        port_id=port_id,
                        source_node_path=src_full_path,
                        source_port_id=src_port,
                        kind="exec" if is_exec else "data",
                    )
                    all_bindings.append(binding)
                    if is_exec:
                        exec_bindings.append(binding)
                    else:
                        data_bindings.append(binding)

                raw_out_ports = (
                    [p.id for p in node_def.output_ports(node.properties, context)]
                    if node_def
                    else ["output"]
                )
                exec_out_ports = [
                    p for p in raw_out_ports if self._is_exec_port(p, node_def, is_output=True)
                ]
                data_out_ports = [
                    p for p in raw_out_ports if not self._is_exec_port(p, node_def, is_output=True)
                ]

                category = node_def.category if node_def else "Layers"
                inst_kind = "data"
                if category in ("Flow Control", "Events"):
                    inst_kind = "flow_control" if category == "Flow Control" else "event"
                elif category == "Optimization":
                    inst_kind = "optimization"
                elif category == "LR Schedulers":
                    inst_kind = "scheduler"
                elif category == "Metrics & Evaluation":
                    inst_kind = "metric"
                elif category == "Persistence":
                    inst_kind = "persistence"

                is_terminal = node.definition_id in (
                    "builtin.graph_output@1",
                    "builtin.module_output@1",
                    "builtin.logits_output@1",
                    "builtin.loss_output@1",
                )
                out_name = node.properties.get("name", "output") if is_terminal else None
                if is_terminal and out_name:
                    output_names.append(out_name)

                is_input = node.definition_id in (
                    "builtin.tensor_input@1",
                    "builtin.token_input@1",
                    "builtin.target_input@1",
                    "builtin.module_input@1",
                )
                in_name = node.properties.get("name", "input") if is_input else None
                if is_input and in_name:
                    input_names.append(in_name)

                # Check if this node maps to a composite subgraph
                if subgraph_id and subgraph_id in project.model.graphs:
                    target_subgraph = project.model.graphs[subgraph_id]

                    if target_subgraph.kind == "repeat":
                        count_val = int(evaluate_value(target_subgraph.repeat_count or 1, cfg))
                        repeat_target_id = target_subgraph.target_graph_id or ""

                        # Instantiate N independent block instances in an nn.ModuleList
                        block_list = nn.ModuleList()
                        for i in range(count_val):
                            block_plan, block_mods = compile_graph_recursive(
                                repeat_target_id,
                                path_prefix=f"{node_path}[{i}]",
                            )
                            block_mod = CompiledGraphModule(block_plan, block_mods)
                            block_list.append(block_mod)

                        modules[sanitized_key] = block_list
                        inst = ExecutionInstruction(
                            node_path=node_path,
                            node_id=node.id,
                            definition_id=node.definition_id,
                            display_name=node.display_name,
                            input_bindings=all_bindings,
                            output_ports=raw_out_ports,
                            kind=inst_kind,
                            exec_in_bindings=exec_bindings,
                            exec_out_ports=exec_out_ports,
                            data_input_bindings=data_bindings,
                            data_output_ports=data_out_ports,
                            input_name=in_name,
                            module_key=sanitized_key,
                            is_repeat=True,
                        )
                        instructions.append(inst)
                        if bool(exec_bindings) or (node.id in outbound_exec_node_ids):
                            exec_instructions.append(inst)
                        continue

                    else:
                        # Single composite module
                        child_plan, child_mods = compile_graph_recursive(
                            subgraph_id,
                            path_prefix=node_path,
                        )
                        child_module = CompiledGraphModule(child_plan, child_mods)
                        modules[sanitized_key] = child_module

                        inst = ExecutionInstruction(
                            node_path=node_path,
                            node_id=node.id,
                            definition_id=node.definition_id,
                            display_name=node.display_name,
                            input_bindings=all_bindings,
                            output_ports=raw_out_ports,
                            kind=inst_kind,
                            exec_in_bindings=exec_bindings,
                            exec_out_ports=exec_out_ports,
                            data_input_bindings=data_bindings,
                            data_output_ports=data_out_ports,
                            input_name=in_name,
                            module_key=sanitized_key,
                            is_composite=True,
                        )
                        instructions.append(inst)
                        if bool(exec_bindings) or (node.id in outbound_exec_node_ids):
                            exec_instructions.append(inst)
                        continue

                # Standard primitive node
                module_key: Optional[str] = None
                functional_fn: Optional[Callable[..., Any]] = None

                if node_def:
                    runtime_spec = node_def.build_runtime(node.properties, context)
                    if runtime_spec:
                        if runtime_spec.module_type == "nn_module" and runtime_spec.factory:
                            m = runtime_spec.factory()
                            modules[sanitized_key] = m
                            module_key = sanitized_key
                        elif runtime_spec.module_type == "functional" and runtime_spec.factory:
                            functional_fn = runtime_spec.factory()

                inst = ExecutionInstruction(
                    node_path=node_path,
                    node_id=node.id,
                    definition_id=node.definition_id,
                    display_name=node.display_name,
                    input_bindings=all_bindings,
                    output_ports=raw_out_ports,
                    kind=inst_kind,
                    exec_in_bindings=exec_bindings,
                    exec_out_ports=exec_out_ports,
                    data_input_bindings=data_bindings,
                    data_output_ports=data_out_ports,
                    input_name=in_name,
                    module_key=module_key,
                    functional_fn=functional_fn,
                    is_terminal_output=is_terminal,
                    output_name=out_name,
                )
                instructions.append(inst)
                if bool(exec_bindings) or (node.id in outbound_exec_node_ids):
                    exec_instructions.append(inst)

            plan = ExecutionPlan(
                graph_hash=graph_hash,
                instructions=instructions,
                input_port_names=input_names,
                output_port_names=output_names,
                exec_instructions=exec_instructions,
            )

            return plan, modules

        plan, modules = compile_graph_recursive(target_root)
        return plan, modules
