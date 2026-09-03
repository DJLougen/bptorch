"""PyTorch module-tree importer into bpTorch Blueprint Project IR."""

from datetime import datetime, timezone
import operator
import re
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_blueprint.ir.models import (
    Edge,
    GraphDefinition,
    ModelDefinition,
    NodeInstance,
    NodeMetadata,
    PortReference,
    Project,
    ProjectMetadata,
    UIState,
)


class ImportUnsupportedError(ValueError):
    def __init__(self, ops: list[str]):
        self.ops = ops
        super().__init__(f"Unsupported FX ops: {', '.join(ops)}")


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def import_pytorch_source(code: str, class_name: Optional[str] = None) -> Project:
    def safe_import(name, *args, **kwargs):
        allowed = {"torch", "torch.nn", "torch.nn.functional", "math", "typing", "operator"}
        if name in allowed or name.startswith("torch"):
            return __import__(name, *args, **kwargs)
        raise ImportUnsupportedError([f"import_{name}"])

    exec_builtins = {
        "__build_class__": __build_class__,
        "__name__": "imported",
        "__import__": safe_import,
        "range": range,
        "len": len,
        "print": print,
        "int": int,
        "float": float,
        "bool": bool,
        "str": str,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "super": super,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "getattr": getattr,
        "hasattr": hasattr,
        "setattr": setattr,
    }
    restricted_namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "__builtins__": exec_builtins,
    }

    try:
        exec(code, restricted_namespace)
    except Exception as exc:
        raise ImportUnsupportedError([f"syntax_or_exec_error: {exc}"]) from exc

    # Find the target nn.Module class
    target_cls = None
    if class_name:
        candidate = restricted_namespace.get(class_name)
        if isinstance(candidate, type) and issubclass(candidate, nn.Module):
            target_cls = candidate
        else:
            raise ImportUnsupportedError([class_name])
    else:
        candidates = [
            v
            for v in restricted_namespace.values()
            if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module
        ]
        if not candidates:
            raise ImportUnsupportedError(["no_module_class_found"])
        target_cls = candidates[-1]

    # Instantiate with no args
    try:
        module = target_cls()
    except Exception as exc:
        raise ImportUnsupportedError(["constructor"]) from exc

    # Symbolic trace
    try:
        gm = torch.fx.symbolic_trace(module)
    except Exception:
        try:
            gm = torch.fx.symbolic_trace(module, concrete_args=None)
        except Exception as exc:
            raise ImportUnsupportedError([f"symbolic_trace: {exc}"]) from exc

    unsupported_ops: List[str] = []
    named_modules = dict(module.named_modules())

    nodes: List[NodeInstance] = []
    fx_to_ir_node: Dict[str, NodeInstance] = {}
    edges: List[Edge] = []

    for fx_node in gm.graph.nodes:
        op = fx_node.op
        name = _sanitize_name(fx_node.name)

        if op == "get_attr":
            continue

        elif op == "placeholder":
            ir_node = NodeInstance(
                id=f"node_in_{name}",
                definition_id="builtin.tensor_input@1",
                display_name=fx_node.name,
                properties={"name": fx_node.name, "dtype": "float32"},
                metadata=NodeMetadata(),
            )
            nodes.append(ir_node)
            fx_to_ir_node[fx_node.name] = ir_node

        elif op == "call_module":
            mod_target = str(fx_node.target)
            submod = named_modules.get(mod_target)
            if submod is None:
                unsupported_ops.append(f"missing_submodule:{mod_target}")
                continue

            if isinstance(submod, nn.Linear):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.linear@1",
                    display_name=f"Linear ({submod.in_features} -> {submod.out_features})",
                    properties={
                        "in_features": submod.in_features,
                        "out_features": submod.out_features,
                        "bias": submod.bias is not None,
                    },
                    metadata=NodeMetadata(),
                )
            elif isinstance(submod, nn.Embedding):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.embedding@1",
                    display_name=f"Embedding ({submod.num_embeddings}, {submod.embedding_dim})",
                    properties={
                        "num_embeddings": submod.num_embeddings,
                        "embedding_dim": submod.embedding_dim,
                    },
                    metadata=NodeMetadata(),
                )
            elif isinstance(submod, nn.LayerNorm):
                normalized_shape = int(
                    submod.normalized_shape[0]
                    if isinstance(submod.normalized_shape, tuple)
                    else submod.normalized_shape
                )
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.layernorm@1",
                    display_name="LayerNorm",
                    properties={
                        "normalized_shape": normalized_shape,
                        "eps": submod.eps,
                        "bias": submod.bias is not None,
                    },
                    metadata=NodeMetadata(),
                )
            elif isinstance(submod, nn.Dropout):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.dropout@1",
                    display_name="Dropout",
                    properties={"dropout": float(submod.p)},
                    metadata=NodeMetadata(),
                )
            elif isinstance(submod, nn.GELU):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.gelu@1",
                    display_name="GELU",
                    properties={},
                    metadata=NodeMetadata(),
                )
            elif isinstance(submod, nn.ReLU):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.relu@1",
                    display_name="ReLU",
                    properties={},
                    metadata=NodeMetadata(),
                )
            elif isinstance(submod, nn.SiLU):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.silu@1",
                    display_name="SiLU",
                    properties={},
                    metadata=NodeMetadata(),
                )
            else:
                unsupported_ops.append(type(submod).__name__)
                continue

            nodes.append(ir_node)
            fx_to_ir_node[fx_node.name] = ir_node

        elif op == "call_function":
            func = fx_node.target
            if func in (operator.add, torch.add):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.add@1",
                    display_name="Add",
                    properties={},
                    metadata=NodeMetadata(),
                )
            elif func in (F.gelu, torch.gelu):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.gelu@1",
                    display_name="GELU",
                    properties={},
                    metadata=NodeMetadata(),
                )
            elif func in (F.relu, torch.relu):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.relu@1",
                    display_name="ReLU",
                    properties={},
                    metadata=NodeMetadata(),
                )
            elif func in (F.silu, torch.silu):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.silu@1",
                    display_name="SiLU",
                    properties={},
                    metadata=NodeMetadata(),
                )
            elif func in (F.softmax, torch.softmax):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.softmax@1",
                    display_name="Softmax",
                    properties={"dim": -1},
                    metadata=NodeMetadata(),
                )
            elif func in (torch.matmul, operator.matmul):
                ir_node = NodeInstance(
                    id=f"node_{name}",
                    definition_id="builtin.matmul@1",
                    display_name="MatMul",
                    properties={},
                    metadata=NodeMetadata(),
                )
            else:
                target_name = getattr(func, "__name__", str(func))
                unsupported_ops.append(target_name)
                continue

            nodes.append(ir_node)
            fx_to_ir_node[fx_node.name] = ir_node

        elif op == "output":
            ir_node = NodeInstance(
                id=f"node_out_{name}",
                definition_id="builtin.graph_output@1",
                display_name="Output",
                properties={"name": "output"},
                metadata=NodeMetadata(),
            )
            nodes.append(ir_node)
            fx_to_ir_node[fx_node.name] = ir_node

        else:
            unsupported_ops.append(f"{op}:{fx_node.target}")

    if unsupported_ops:
        raise ImportUnsupportedError(unsupported_ops)

    # Wire edges
    edge_count = 0
    for fx_node in gm.graph.nodes:
        if fx_node.name not in fx_to_ir_node:
            continue
        tgt_ir = fx_to_ir_node[fx_node.name]

        if fx_node.op == "output":
            out_args = fx_node.args[0] if fx_node.args else None
            if isinstance(out_args, (list, tuple)):
                arg_nodes = [a for a in out_args if isinstance(a, torch.fx.Node)]
            elif isinstance(out_args, torch.fx.Node):
                arg_nodes = [out_args]
            else:
                arg_nodes = []

            for a_node in arg_nodes:
                if a_node.name in fx_to_ir_node:
                    src_ir = fx_to_ir_node[a_node.name]
                    edges.append(
                        Edge(
                            id=f"e_{src_ir.id}_{tgt_ir.id}_{edge_count}",
                            source=PortReference(node_id=src_ir.id, port_id="output"),
                            target=PortReference(node_id=tgt_ir.id, port_id="input"),
                        )
                    )
                    edge_count += 1

        elif tgt_ir.definition_id == "builtin.add@1":
            add_ports = ["a", "b"]
            fx_args = [a for a in fx_node.args if isinstance(a, torch.fx.Node)]
            for port_id, a_node in zip(add_ports, fx_args):
                if a_node.name in fx_to_ir_node:
                    src_ir = fx_to_ir_node[a_node.name]
                    edges.append(
                        Edge(
                            id=f"e_{src_ir.id}_{tgt_ir.id}_{port_id}",
                            source=PortReference(node_id=src_ir.id, port_id="output"),
                            target=PortReference(node_id=tgt_ir.id, port_id=port_id),
                        )
                    )

        elif tgt_ir.definition_id == "builtin.matmul@1":
            matmul_ports = ["a", "b"]
            fx_args = [a for a in fx_node.args if isinstance(a, torch.fx.Node)]
            for port_id, a_node in zip(matmul_ports, fx_args):
                if a_node.name in fx_to_ir_node:
                    src_ir = fx_to_ir_node[a_node.name]
                    edges.append(
                        Edge(
                            id=f"e_{src_ir.id}_{tgt_ir.id}_{port_id}",
                            source=PortReference(node_id=src_ir.id, port_id="output"),
                            target=PortReference(node_id=tgt_ir.id, port_id=port_id),
                        )
                    )

        else:
            fx_args = [a for a in fx_node.args if isinstance(a, torch.fx.Node)]
            if fx_args:
                a_node = fx_args[0]
                if a_node.name in fx_to_ir_node:
                    src_ir = fx_to_ir_node[a_node.name]
                    edges.append(
                        Edge(
                            id=f"e_{src_ir.id}_{tgt_ir.id}_input",
                            source=PortReference(node_id=src_ir.id, port_id="output"),
                            target=PortReference(node_id=tgt_ir.id, port_id="input"),
                        )
                    )

    root_graph = GraphDefinition(
        id="graph_imported",
        name="Imported Module",
        kind="root",
        nodes=nodes,
        edges=edges,
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    project = Project(
        schema_version=1,
        project=ProjectMetadata(
            id="project_imported",
            name="Imported Module",
            created_at=now_iso,
            updated_at=now_iso,
        ),
        model=ModelDefinition(
            root_graph_id="graph_imported",
            config={},
            graphs={"graph_imported": root_graph},
            weight_bindings=[],
        ),
        ui=UIState(
            open_graph_id="graph_imported",
            node_positions={},
        ),
    )

    return project
