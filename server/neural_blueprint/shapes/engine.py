"""Topological shape propagation and unification engine for neural blueprint graphs."""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from neural_blueprint.ir.models import GraphDefinition, TensorSpec
from neural_blueprint.registry.base import NodeValidationContext
from neural_blueprint.registry.registry import NodeRegistry, global_registry


class ShapePropagator:
    """Propagates tensor shapes through a graph in topological order."""

    def __init__(self, registry: Optional[NodeRegistry] = None):
        self.registry = registry or global_registry

    def topological_sort(self, graph: GraphDefinition) -> Tuple[List[str], bool]:
        """Returns (sorted_node_ids, has_cycle) using Kahn's algorithm."""
        in_degree: Dict[str, int] = {node.id: 0 for node in graph.nodes}
        adj: Dict[str, Set[str]] = defaultdict(set)

        for edge in graph.edges:
            src_node = edge.source.node_id
            tgt_node = edge.target.node_id
            if src_node in in_degree and tgt_node in in_degree:
                if tgt_node not in adj[src_node]:
                    adj[src_node].add(tgt_node)
                    in_degree[tgt_node] += 1

        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        ordered: List[str] = []

        while queue:
            node_id = queue.popleft()
            ordered.append(node_id)
            for neighbor in adj[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        has_cycle = len(ordered) != len(graph.nodes)
        return ordered, has_cycle

    def propagate_graph(
        self,
        graph: GraphDefinition,
        config: Optional[Dict[str, Any]] = None,
        parent_properties: Optional[Dict[str, Any]] = None,
        graph_definitions: Optional[Dict[str, GraphDefinition]] = None,
        external_inputs: Optional[Dict[str, TensorSpec]] = None,
    ) -> Dict[str, Dict[str, TensorSpec]]:
        """
        Propagates shapes through the graph.
        Returns: { node_id: { port_id: TensorSpec } } for all nodes.
        """
        if config is None:
            config = {}
        if parent_properties is None:
            parent_properties = {}
        if graph_definitions is None:
            graph_definitions = {}
        if external_inputs is None:
            external_inputs = {}

        context = NodeValidationContext(
            model_config=config,
            parent_properties=parent_properties,
            graph_definitions=graph_definitions,
        )

        nodes_by_id = {node.id: node for node in graph.nodes}
        # Inbound edges: target_node_id -> target_port_id -> (src_node_id, src_port_id)
        inbound_edges: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)
        for edge in graph.edges:
            inbound_edges[edge.target.node_id][edge.target.port_id] = (
                edge.source.node_id,
                edge.source.port_id,
            )

        resolved_shapes: Dict[str, Dict[str, TensorSpec]] = defaultdict(dict)
        ordered_ids, has_cycle = self.topological_sort(graph)

        # Fallback to definition order if cycle exists
        execution_order = ordered_ids if not has_cycle else [node.id for node in graph.nodes]

        for node_id in execution_order:
            node = nodes_by_id.get(node_id)
            if not node:
                continue

            node_def = self.registry.get(node.definition_id)
            if not node_def:
                continue

            # Gather input port specs from connected source outputs
            node_inputs: Dict[str, TensorSpec] = {}
            for port_id, (src_id, src_port) in inbound_edges.get(node_id, {}).items():
                if src_id in resolved_shapes and src_port in resolved_shapes[src_id]:
                    node_inputs[port_id] = resolved_shapes[src_id][src_port]

            # If node is ModuleInput or has external inputs passed in
            for port_id, spec in external_inputs.items():
                if node_id == "input" or node.definition_id == "builtin.module_input@1":
                    node_inputs[port_id] = spec

            # Infer output shapes
            try:
                outputs = node_def.infer_shapes(node_inputs, node.properties, context)
                for port_id, spec in outputs.items():
                    resolved_shapes[node_id][port_id] = spec
            except Exception:
                # Shape inference error handled by validator
                pass

        return resolved_shapes
