"""Parameter accounting and weight tying manager."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from neural_blueprint.ir.evaluator import evaluate_value
from neural_blueprint.ir.models import Project
from neural_blueprint.registry.base import NodeValidationContext, ParameterSpec
from neural_blueprint.registry.registry import NodeRegistry, global_registry


@dataclass
class ParameterSummary:
    total_unique: int = 0
    trainable: int = 0
    frozen: int = 0
    shared_references: int = 0
    breakdown_by_node: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ParameterAccounting:
    """Computes exact parameter counts, tracking shared/tied tensors and repeated stacks."""

    def __init__(self, registry: Optional[NodeRegistry] = None):
        self.registry = registry or global_registry

    def calculate_summary(self, project: Project) -> ParameterSummary:
        cfg = project.model.config
        breakdown: Dict[str, Dict[str, Any]] = {}
        total_trainable = 0
        total_frozen = 0
        total_shared_deduction = 0
        shared_references_count = 0

        # Calculate tied parameter size to deduct from duplicates
        tied_nodes: Set[str] = set()
        for binding in project.model.weight_bindings:
            if binding.mode == "share":
                tied_nodes.add(binding.target.node_id)
                shared_references_count += 1

        root_graph = project.model.graphs.get(project.model.root_graph_id)
        if not root_graph:
            return ParameterSummary()

        def process_graph(graph_id: str, path_prefix: str = "", repeat_multiplier: int = 1):
            nonlocal total_trainable, total_frozen, total_shared_deduction
            graph = project.model.graphs.get(graph_id)
            if not graph:
                return

            context = NodeValidationContext(
                model_config=cfg,
                graph_definitions=project.model.graphs,
            )

            for node in graph.nodes:
                node_path = f"{path_prefix}/{node.id}" if path_prefix else node.id
                node_def = self.registry.get(node.definition_id)

                if graph.kind == "repeat" and graph.target_graph_id:
                    continue

                if node_def:
                    spec: ParameterSpec = node_def.parameter_spec(node.properties, context)
                    node_trainable = spec.trainable_count * repeat_multiplier
                    node_frozen = spec.frozen_count * repeat_multiplier

                    is_tied = node.id in tied_nodes or node_path in tied_nodes
                    if is_tied:
                        total_shared_deduction += node_trainable

                    total_trainable += node_trainable
                    total_frozen += node_frozen

                    breakdown[node_path] = {
                        "display_name": node.display_name,
                        "trainable": node_trainable,
                        "frozen": node_frozen,
                        "total": node_trainable + node_frozen,
                        "shapes": spec.parameter_shapes,
                        "is_shared": is_tied,
                    }

        for graph_id, graph in project.model.graphs.items():
            if graph.kind == "repeat":
                count_val = int(evaluate_value(graph.repeat_count or 1, cfg))
                target_id = graph.target_graph_id or ""
                process_graph(target_id, path_prefix=graph_id, repeat_multiplier=count_val)
            elif graph_id == project.model.root_graph_id:
                process_graph(graph_id)

        unique_total = (total_trainable + total_frozen) - total_shared_deduction

        return ParameterSummary(
            total_unique=max(0, unique_total),
            trainable=max(0, total_trainable - total_shared_deduction),
            frozen=total_frozen,
            shared_references=shared_references_count,
            breakdown_by_node=breakdown,
        )
