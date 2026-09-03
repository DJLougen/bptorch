"""4-Pass Validator for bpTorch projects."""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from neural_blueprint.ir.evaluator import evaluate_value
from neural_blueprint.ir.models import (
    GraphDefinition,
    NodeInstance,
    PortDefinition,
    Project,
    TensorSpec,
)
from neural_blueprint.registry.base import NodeValidationContext
from neural_blueprint.registry.registry import NodeRegistry, global_registry
from neural_blueprint.runtime.compiler import COMPOSITE_TYPE_MAP
from neural_blueprint.shapes.engine import ShapePropagator
from neural_blueprint.shapes.types import (
    format_shape,
    resolve_dim_value,
    shapes_compatible,
)
from neural_blueprint.validation.diagnostics import (
    E_CYCLE_DETECTED,
    E_DUPLICATE_EDGE_ID,
    E_EDGE_MISSING_NODE,
    E_EDGE_MISSING_PORT,
    E_EDGE_PORT_KIND_MISMATCH,
    E_EDGE_SELF_CONNECTION,
    E_EDGE_WRONG_PORT_DIRECTION,
    E_HEAD_DIVISIBILITY,
    E_LINEAR_INPUT_DIM,
    E_MULTIPLE_INPUTS,
    E_PORT_UNCONNECTED,
    E_RESIDUAL_MISMATCH,
    E_SCHEMA_INVALID,
    E_SHAPE_MISMATCH,
    E_UNKNOWN_NODE_TYPE,
    E_WEIGHT_TYING_MISMATCH,
    Diagnostic,
)


class ValidationResult:
    def __init__(
        self,
        valid: bool,
        diagnostics: List[Diagnostic],
        resolved_shapes: Dict[str, Dict[str, Dict[str, TensorSpec]]],
    ):
        self.valid = valid
        self.diagnostics = diagnostics
        self.resolved_shapes = resolved_shapes

    @property
    def errors(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "warning"]


class ProjectValidator:
    """Performs 4-pass validation over canonical project IR."""

    def __init__(self, registry: Optional[NodeRegistry] = None):
        self.registry = registry or global_registry
        self.propagator = ShapePropagator(self.registry)

    def validate(self, project: Project) -> ValidationResult:
        diagnostics: List[Diagnostic] = []

        p1_diags = self.validate_pass1_schema(project)
        diagnostics.extend(p1_diags)

        p2_diags = self.validate_pass2_structure(project)
        diagnostics.extend(p2_diags)

        resolved_shapes: Dict[str, Dict[str, Dict[str, TensorSpec]]] = {}
        for graph_id, graph in project.model.graphs.items():
            shapes = self.propagator.propagate_graph(
                graph=graph,
                config=project.model.config,
                graph_definitions=project.model.graphs,
            )
            resolved_shapes[graph_id] = shapes

        p3_diags = self.validate_pass3_shapes(project, resolved_shapes)
        diagnostics.extend(p3_diags)

        p4_diags = self.validate_pass4_semantics(project, resolved_shapes)
        diagnostics.extend(p4_diags)

        has_errors = any(d.severity == "error" for d in diagnostics)
        return ValidationResult(
            valid=not has_errors,
            diagnostics=diagnostics,
            resolved_shapes=resolved_shapes,
        )

    def _resolve_binding_node(self, project: Project, node_id: str) -> Optional[NodeInstance]:
        root_graph = project.model.graphs.get(project.model.root_graph_id)
        if not root_graph:
            return None
        if "/" in node_id:
            root_id, inner_id = node_id.split("/", 1)
            root_node = next((n for n in root_graph.nodes if n.id == root_id), None)
            if root_node:
                subgraph_id = COMPOSITE_TYPE_MAP.get(root_node.definition_id)
                subgraph = project.model.graphs.get(subgraph_id) if subgraph_id else None
                if subgraph:
                    return next((n for n in subgraph.nodes if n.id == inner_id), None)
            return None
        return next((n for n in root_graph.nodes if n.id == node_id), None)

    def validate_pass1_schema(self, project: Project) -> List[Diagnostic]:
        diagnostics: List[Diagnostic] = []

        if not project.project.id:
            diagnostics.append(
                Diagnostic(
                    code=E_SCHEMA_INVALID,
                    severity="error",
                    message="Project ID must not be empty.",
                )
            )

        root_id = project.model.root_graph_id
        if not root_id or root_id not in project.model.graphs:
            diagnostics.append(
                Diagnostic(
                    code=E_SCHEMA_INVALID,
                    severity="error",
                    message=f"Root graph '{root_id}' does not exist in model graphs.",
                )
            )

        for graph_id, graph in project.model.graphs.items():
            seen_node_ids: Set[str] = set()
            for node in graph.nodes:
                if node.id in seen_node_ids:
                    diagnostics.append(
                        Diagnostic(
                            code=E_SCHEMA_INVALID,
                            severity="error",
                            message=f"Duplicate node ID '{node.id}' in graph '{graph_id}'.",
                            node_id=node.id,
                        )
                    )
                seen_node_ids.add(node.id)

                node_def = self.registry.get(node.definition_id)
                if not node_def:
                    if node.definition_id.startswith("custom."):
                        subgraph_id = node.definition_id[len("custom."):]
                        if subgraph_id not in project.model.graphs:
                            diagnostics.append(
                                Diagnostic(
                                    code=E_UNKNOWN_NODE_TYPE,
                                    severity="error",
                                    message=(
                                        f"Custom subgraph '{subgraph_id}' does not exist in model graphs "
                                        f"for node '{node.display_name}'."
                                    ),
                                    node_id=node.id,
                                    suggestions=["Create the custom subgraph or re-link."],
                                )
                            )
                    else:
                        diagnostics.append(
                            Diagnostic(
                                code=E_UNKNOWN_NODE_TYPE,
                                severity="error",
                                message=(
                                    f"Unknown node definition '{node.definition_id}' "
                                    f"on node '{node.display_name}'."
                                ),
                                node_id=node.id,
                                suggestions=["Select a valid node type from palette."],
                            )
                        )
        return diagnostics

    def _resolve_node_ports(
        self,
        node: NodeInstance,
        context: NodeValidationContext,
    ) -> Tuple[Dict[str, PortDefinition], Dict[str, PortDefinition]]:
        node_def = self.registry.get(node.definition_id)
        if node_def:
            input_ports = {port.id: port for port in node_def.input_ports(node.properties, context)}
            output_ports = {port.id: port for port in node_def.output_ports(node.properties, context)}
            return input_ports, output_ports

        if node.definition_id.startswith("custom.") and context.graph_definitions:
            subgraph_id = node.definition_id[len("custom."):]
            if subgraph_id in context.graph_definitions:
                subgraph = context.graph_definitions[subgraph_id]
                input_ports = {port.id: port for port in subgraph.interface.inputs}
                output_ports = {port.id: port for port in subgraph.interface.outputs}
                return input_ports, output_ports

        return {}, {}

    def _validate_graph_edges(
        self,
        graph_id: str,
        graph: GraphDefinition,
        context: NodeValidationContext,
    ) -> Tuple[List[Diagnostic], Set[str]]:
        diagnostics: List[Diagnostic] = []
        invalid_edge_ids: Set[str] = set()
        nodes_by_id = {node.id: node for node in graph.nodes}

        edge_id_counts: Dict[str, int] = defaultdict(int)
        for edge in graph.edges:
            edge_id_counts[edge.id] += 1

        for edge in graph.edges:
            if edge_id_counts[edge.id] > 1:
                diagnostics.append(
                    Diagnostic(
                        code=E_DUPLICATE_EDGE_ID,
                        severity="error",
                        message=(
                            f"Duplicate edge ID '{edge.id}' in graph '{graph.name}' ({graph_id})."
                        ),
                        edge_id=edge.id,
                        suggestions=["Assign a unique edge ID."],
                    )
                )
                invalid_edge_ids.add(edge.id)

        for edge in graph.edges:
            src_node = nodes_by_id.get(edge.source.node_id)
            if not src_node:
                diagnostics.append(
                    Diagnostic(
                        code=E_EDGE_MISSING_NODE,
                        severity="error",
                        message=(
                            f"Edge '{edge.id}' references missing source node "
                            f"'{edge.source.node_id}'."
                        ),
                        edge_id=edge.id,
                        node_id=edge.source.node_id,
                        suggestions=["Connect from an existing node or remove the edge."],
                    )
                )
                invalid_edge_ids.add(edge.id)

            tgt_node = nodes_by_id.get(edge.target.node_id)
            if not tgt_node:
                diagnostics.append(
                    Diagnostic(
                        code=E_EDGE_MISSING_NODE,
                        severity="error",
                        message=(
                            f"Edge '{edge.id}' references missing target node "
                            f"'{edge.target.node_id}'."
                        ),
                        edge_id=edge.id,
                        node_id=edge.target.node_id,
                        suggestions=["Connect to an existing node or remove the edge."],
                    )
                )
                invalid_edge_ids.add(edge.id)

            if edge.source.node_id == edge.target.node_id:
                diagnostics.append(
                    Diagnostic(
                        code=E_EDGE_SELF_CONNECTION,
                        severity="error",
                        message=(
                            f"Edge '{edge.id}' connects node '{edge.source.node_id}' to itself."
                        ),
                        edge_id=edge.id,
                        node_id=edge.source.node_id,
                        suggestions=["Remove the self-connection or route through another node."],
                    )
                )
                invalid_edge_ids.add(edge.id)

            if not src_node or not tgt_node:
                continue

            src_inputs, src_outputs = self._resolve_node_ports(src_node, context)
            tgt_inputs, tgt_outputs = self._resolve_node_ports(tgt_node, context)

            src_port = src_outputs.get(edge.source.port_id)
            if src_port is None:
                wrong_port = src_inputs.get(edge.source.port_id)
                if wrong_port is not None:
                    diagnostics.append(
                        Diagnostic(
                            code=E_EDGE_WRONG_PORT_DIRECTION,
                            severity="error",
                            message=(
                                f"Edge '{edge.id}' connects from input port "
                                f"'{edge.source.port_id}' on "
                                f"'{src_node.display_name}', but source endpoints "
                                "must use output ports."
                            ),
                            edge_id=edge.id,
                            node_id=src_node.id,
                            port_id=edge.source.port_id,
                            expected="output",
                            actual="input",
                            suggestions=["Connect from an output port on the source node."],
                        )
                    )
                    invalid_edge_ids.add(edge.id)
                    continue

                diagnostics.append(
                    Diagnostic(
                        code=E_EDGE_MISSING_PORT,
                        severity="error",
                        message=(
                            f"Edge '{edge.id}' references missing source port "
                            f"'{edge.source.port_id}' on '{src_node.display_name}'."
                        ),
                        edge_id=edge.id,
                        node_id=src_node.id,
                        port_id=edge.source.port_id,
                        suggestions=["Connect from a valid output port on the source node."],
                    )
                )
                invalid_edge_ids.add(edge.id)
                continue

            tgt_port = tgt_inputs.get(edge.target.port_id)
            if tgt_port is None:
                wrong_port = tgt_outputs.get(edge.target.port_id)
                if wrong_port is not None:
                    diagnostics.append(
                        Diagnostic(
                            code=E_EDGE_WRONG_PORT_DIRECTION,
                            severity="error",
                            message=(
                                f"Edge '{edge.id}' connects to output port "
                                f"'{edge.target.port_id}' on '{tgt_node.display_name}', but "
                                "target endpoints must use input ports."
                            ),
                            edge_id=edge.id,
                            node_id=tgt_node.id,
                            port_id=edge.target.port_id,
                            expected="input",
                            actual="output",
                            suggestions=["Connect to an input port on the target node."],
                        )
                    )
                    invalid_edge_ids.add(edge.id)
                    continue

                diagnostics.append(
                    Diagnostic(
                        code=E_EDGE_MISSING_PORT,
                        severity="error",
                        message=(
                            f"Edge '{edge.id}' references missing target port "
                            f"'{edge.target.port_id}' on '{tgt_node.display_name}'."
                        ),
                        edge_id=edge.id,
                        node_id=tgt_node.id,
                        port_id=edge.target.port_id,
                        suggestions=["Connect to a valid input port on the target node."],
                    )
                )
                invalid_edge_ids.add(edge.id)
                continue

            if src_port.kind != tgt_port.kind:
                diagnostics.append(
                    Diagnostic(
                        code=E_EDGE_PORT_KIND_MISMATCH,
                        severity="error",
                        message=(
                            f"Edge '{edge.id}' connects incompatible port kinds: "
                            f"'{src_port.kind}' -> '{tgt_port.kind}'."
                        ),
                        edge_id=edge.id,
                        node_id=tgt_node.id,
                        port_id=edge.target.port_id,
                        expected=src_port.kind,
                        actual=tgt_port.kind,
                        suggestions=["Connect ports of the same kind (data or exec)."],
                    )
                )
                invalid_edge_ids.add(edge.id)

        return diagnostics, invalid_edge_ids

    def _has_data_cycle(self, edges: List) -> bool:
        adjacency: Dict[str, List[str]] = defaultdict(list)
        nodes: Set[str] = set()
        for edge in edges:
            adjacency[edge.source.node_id].append(edge.target.node_id)
            nodes.add(edge.source.node_id)
            nodes.add(edge.target.node_id)

        visited: Set[str] = set()
        stack: Set[str] = set()

        def dfs(node_id: str) -> bool:
            if node_id in stack:
                return True
            if node_id in visited:
                return False
            visited.add(node_id)
            stack.add(node_id)
            for neighbor in adjacency.get(node_id, []):
                if dfs(neighbor):
                    return True
            stack.remove(node_id)
            return False

        return any(dfs(node_id) for node_id in nodes)

    def validate_pass2_structure(self, project: Project) -> List[Diagnostic]:
        """Pass 2: Structural graph validation."""
        diagnostics: List[Diagnostic] = []

        for graph_id, graph in project.model.graphs.items():
            context = NodeValidationContext(
                model_config=project.model.config,
                graph_definitions=project.model.graphs,
            )
            edge_diags, invalid_edge_ids = self._validate_graph_edges(graph_id, graph, context)
            diagnostics.extend(edge_diags)

            nodes_by_id = {node.id: node for node in graph.nodes}
            data_edges = []
            for edge in graph.edges:
                if edge.id in invalid_edge_ids:
                    continue
                src_node = nodes_by_id.get(edge.source.node_id)
                tgt_node = nodes_by_id.get(edge.target.node_id)
                if not src_node or not tgt_node:
                    continue
                _, src_outputs = self._resolve_node_ports(src_node, context)
                tgt_inputs, _ = self._resolve_node_ports(tgt_node, context)
                src_port = src_outputs.get(edge.source.port_id)
                tgt_port = tgt_inputs.get(edge.target.port_id)
                if src_port and tgt_port and src_port.kind == "data" and tgt_port.kind == "data":
                    data_edges.append(edge)

            if self._has_data_cycle(data_edges):
                diagnostics.append(
                    Diagnostic(
                        code=E_CYCLE_DETECTED,
                        severity="error",
                        message="Cycle detected in data-flow graph.",
                        suggestions=["Remove or reroute edges to break the cycle."],
                    )
                )

            incoming: Dict[Tuple[str, str], List[str]] = defaultdict(list)
            for edge in graph.edges:
                if edge.id in invalid_edge_ids:
                    continue
                incoming[(edge.target.node_id, edge.target.port_id)].append(edge.id)

            for node in graph.nodes:
                input_ports, _ = self._resolve_node_ports(node, context)
                for port_id, port in input_ports.items():
                    if not port.required:
                        continue
                    connections = incoming.get((node.id, port_id), [])
                    if not connections:
                        diagnostics.append(
                            Diagnostic(
                                code=E_PORT_UNCONNECTED,
                                severity="error",
                                message=(
                                    f"Required input port '{port.display_name}' on "
                                    f"'{node.display_name}' is not connected."
                                ),
                                node_id=node.id,
                                port_id=port_id,
                                suggestions=["Connect a compatible output port to this input."],
                            )
                        )
                    elif port.multiplicity == "single" and len(connections) > 1:
                        diagnostics.append(
                            Diagnostic(
                                code=E_MULTIPLE_INPUTS,
                                severity="error",
                                message=(
                                    f"Input port '{port.display_name}' on "
                                    f"'{node.display_name}' accepts only one connection, "
                                    f"but {len(connections)} are present."
                                ),
                                node_id=node.id,
                                port_id=port_id,
                                suggestions=["Remove duplicate connections to this input port."],
                            )
                        )

        return diagnostics

    def validate_pass3_shapes(
        self,
        project: Project,
        resolved_shapes: Dict[str, Dict[str, Dict[str, TensorSpec]]],
    ) -> List[Diagnostic]:
        """Pass 3: Tensor type and shape validation."""
        diagnostics: List[Diagnostic] = []

        for graph_id, graph in project.model.graphs.items():
            graph_shapes = resolved_shapes.get(graph_id, {})

            for edge in graph.edges:
                src_shapes = graph_shapes.get(edge.source.node_id, {})
                tgt_shapes = graph_shapes.get(edge.target.node_id, {})
                src_spec = src_shapes.get(edge.source.port_id)
                tgt_spec = tgt_shapes.get(edge.target.port_id)
                if not src_spec or not tgt_spec or not src_spec.shape or not tgt_spec.shape:
                    continue
                compatible, _ = shapes_compatible(
                    src_spec.shape,
                    tgt_spec.shape,
                    project.model.config,
                )
                if not compatible:
                    diagnostics.append(
                        Diagnostic(
                            code=E_SHAPE_MISMATCH,
                            severity="error",
                            message=(
                                f"Shape mismatch on edge '{edge.id}': "
                                f"{format_shape(src_spec.shape)} -> {format_shape(tgt_spec.shape)}"
                            ),
                            edge_id=edge.id,
                            expected=format_shape(tgt_spec.shape),
                            actual=format_shape(src_spec.shape),
                            suggestions=["Insert a reshape or projection node between ports."],
                        )
                    )

            for node in graph.nodes:
                if node.definition_id == "builtin.linear@1":
                    in_features = node.properties.get("in_features")
                    if in_features is None:
                        continue
                    input_edges = [
                        edge
                        for edge in graph.edges
                        if edge.target.node_id == node.id and edge.target.port_id == "input"
                    ]
                    if not input_edges:
                        continue
                    edge = input_edges[0]
                    src_spec = graph_shapes.get(edge.source.node_id, {}).get(edge.source.port_id)
                    if not src_spec or not src_spec.shape:
                        continue
                    expected_in = evaluate_value(in_features, project.model.config)
                    actual_last = resolve_dim_value(src_spec.shape[-1], project.model.config)
                    if (
                        expected_in is not None
                        and actual_last is not None
                        and int(expected_in) != actual_last
                    ):
                        diagnostics.append(
                            Diagnostic(
                                code=E_LINEAR_INPUT_DIM,
                                severity="error",
                                message=(
                                    f"Linear node '{node.display_name}' expects "
                                    f"in_features={expected_in}, but connected input provides "
                                    f"last dimension {actual_last}."
                                ),
                                node_id=node.id,
                                expected=str(expected_in),
                                actual=str(actual_last),
                                suggestions=[
                                    "Update in_features to match the input tensor shape, or "
                                    "insert a projection layer."
                                ],
                            )
                        )

                if node.definition_id == "builtin.residual_add@1":
                    a_spec = graph_shapes.get(node.id, {}).get("a")
                    b_spec = graph_shapes.get(node.id, {}).get("b")
                    if a_spec and b_spec and a_spec.shape and b_spec.shape:
                        compatible, _ = shapes_compatible(
                            a_spec.shape,
                            b_spec.shape,
                            project.model.config,
                        )
                        if not compatible:
                            diagnostics.append(
                                Diagnostic(
                                    code=E_RESIDUAL_MISMATCH,
                                    severity="error",
                                    message=(
                                        "Residual Add cannot combine tensors with incompatible "
                                        "shapes. Both inputs must have compatible dimensions."
                                    ),
                                    node_id=node.id,
                                    suggestions=["Ensure residual stream and module output match."],
                                )
                            )

        return diagnostics

    def _binding_weight_vocab_size(
        self,
        node: NodeInstance,
        config: Dict[str, object],
    ) -> Optional[int]:
        for key in ("num_embeddings", "out_features", "in_features"):
            if key not in node.properties:
                continue
            try:
                value = evaluate_value(node.properties[key], config)
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(value, (int, float)):
                return int(value)
        return None

    def validate_pass4_semantics(
        self,
        project: Project,
        resolved_shapes: Dict[str, Dict[str, Dict[str, TensorSpec]]],
    ) -> List[Diagnostic]:
        """Pass 4: Model-semantic validation."""
        diagnostics: List[Diagnostic] = []
        config = project.model.config

        n_embd = config.get("n_embd")
        n_head = config.get("n_head")
        if n_embd is not None and n_head is not None:
            try:
                n_embd_val = int(n_embd)
                n_head_val = int(n_head)
                if n_head_val > 0 and n_embd_val % n_head_val != 0:
                    remainder = n_embd_val % n_head_val
                    diagnostics.append(
                        Diagnostic(
                            code=E_HEAD_DIVISIBILITY,
                            severity="error",
                            message=(
                                f"Embedding dimension n_embd ({n_embd_val}) must be divisible by "
                                f"number of heads n_head ({n_head_val}). Remainder is {remainder}."
                            ),
                            suggestions=[f"Set n_head to a divisor of {n_embd_val}."],
                        )
                    )
            except (TypeError, ValueError):
                pass

        for binding in project.model.weight_bindings:
            source_node = self._resolve_binding_node(project, binding.source.node_id)
            target_node = self._resolve_binding_node(project, binding.target.node_id)
            if not source_node or not target_node:
                continue

            if binding.source.parameter != "weight" or binding.target.parameter != "weight":
                continue

            source_size = self._binding_weight_vocab_size(source_node, config)
            target_size = self._binding_weight_vocab_size(target_node, config)
            if source_size is None or target_size is None or source_size == target_size:
                continue

            diagnostics.append(
                Diagnostic(
                    code=E_WEIGHT_TYING_MISMATCH,
                    severity="error",
                    message=(
                        f"Weight tying mismatch between '{binding.source.node_id}' and "
                        f"'{binding.target.node_id}': incompatible weight shapes "
                        f"({source_size} vs {target_size})."
                    ),
                    node_id=target_node.id,
                    suggestions=[
                        "Align vocabulary/output dimensions so tied weights share the same shape."
                    ],
                )
            )

        return diagnostics
