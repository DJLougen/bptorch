"""Generic PyTorch standalone code emitter for arbitrary Blueprint DAGs and hierarchical composite/repeat graphs."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from neural_blueprint.ir.evaluator import evaluate_value
from neural_blueprint.ir.models import NodeInstance, Project
from neural_blueprint.runtime.compiler import COMPOSITE_TYPE_MAP, GraphCompiler

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
        "builtin.target_input@1",
        "builtin.loss_output@1",
        "builtin.comment@1",
    }
)

_SKIP_KINDS = frozenset(
    {
        "exec",
        "flow_control",
        "event",
        "optimization",
        "scheduler",
        "metric",
        "persistence",
    }
)

_SUPPORTED_DEFS = frozenset(
    {
        "builtin.tensor_input@1",
        "builtin.token_input@1",
        "builtin.module_input@1",
        "builtin.linear@1",
        "builtin.embedding@1",
        "builtin.layernorm@1",
        "builtin.rmsnorm@1",
        "builtin.swiglu@1",
        "builtin.dropout@1",
        "builtin.gelu@1",
        "builtin.silu@1",
        "builtin.relu@1",
        "builtin.add@1",
        "builtin.matmul@1",
        "builtin.softmax@1",
        "builtin.reshape@1",
        "builtin.flatten@1",
        "builtin.transpose@1",
        "builtin.contiguous@1",
        "builtin.scale@1",
        "builtin.masked_fill@1",
        "builtin.split@1",
        "builtin.split_qkv@1",
        "builtin.split_heads@1",
        "builtin.merge_heads@1",
        "builtin.rope@1",
        "builtin.grouped_query_attention@1",
        "builtin.sdpa@1",
        "builtin.manual_causal_attention@1",
        "builtin.lm_head@1",
        "builtin.graph_output@1",
        "builtin.logits_output@1",
        "builtin.module_output@1",
    }
)


def _sanitize(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("[", "_")
        .replace("]", "")
        .replace(".", "_")
        .replace("-", "_")
        .replace("@", "_")
    )


def cook_generic(project: Project, mode: str = "train") -> str:
    """Compile arbitrary Blueprint projects (flat DAGs or hierarchical composite/repeat graphs)
    into a standalone zero-dependency PyTorch script string.
    """
    from neural_blueprint.cooking.cooker import UnsupportedCookError

    compiler = GraphCompiler()
    cfg = project.model.config or {}

    ir_nodes: Dict[str, NodeInstance] = {}
    for g in project.model.graphs.values():
        for n in g.nodes:
            ir_nodes[n.id] = n

    root_graph = project.model.graphs.get(project.model.root_graph_id)
    if root_graph is not None and root_graph.kind != "training_event":
        target_root_id = project.model.root_graph_id
    else:
        roots = [
            gid
            for gid, g in project.model.graphs.items()
            if g.kind == "root" and g.kind != "training_event"
        ]
        if len(roots) == 1:
            target_root_id = roots[0]
        elif len(roots) > 1 and project.model.root_graph_id in roots:
            target_root_id = project.model.root_graph_id
        elif roots:
            target_root_id = roots[0]
        else:
            raise UnsupportedCookError(
                "Unsupported blueprint topology: No architecture root graph to cook"
            )

    # Post-order dependency traversal of reachable subgraphs
    visited: Set[str] = set()
    order: List[str] = []

    def visit_graph(gid: str) -> None:
        if gid in visited or gid not in project.model.graphs:
            return
        visited.add(gid)
        g = project.model.graphs[gid]
        if g.kind == "repeat" and g.target_graph_id:
            visit_graph(g.target_graph_id)
        for n in g.nodes:
            sub = COMPOSITE_TYPE_MAP.get(n.definition_id)
            if sub:
                visit_graph(sub)
            elif n.definition_id.startswith("custom."):
                visit_graph(n.definition_id[len("custom.") :])
        order.append(gid)

    visit_graph(target_root_id)

    has_rmsnorm = False
    has_swiglu = False
    has_rope = False
    has_gqa = False

    class_defs: List[str] = []

    for gid in order:
        g = project.model.graphs[gid]
        if g.kind == "repeat":
            continue

        is_root = gid == target_root_id
        class_name = "CookedModel" if is_root else f"Submodule_{_sanitize(gid)}"

        plan, _ = compiler.compile_plan(project, root_graph_id=gid)

        init_lines: List[str] = []
        forward_lines: List[str] = []
        registered_keys: Set[str] = set()

        for inst in plan.instructions:
            if inst.kind in _SKIP_KINDS or inst.definition_id in _STOP_DEFINITION_IDS:
                continue

            if getattr(inst, "is_composite", False):
                sub_id = COMPOSITE_TYPE_MAP.get(inst.definition_id) or (
                    inst.definition_id[len("custom.") :]
                    if inst.definition_id.startswith("custom.")
                    else None
                )
                if sub_id:
                    sub_class = f"Submodule_{_sanitize(sub_id)}"
                    init_lines.append(f"        self.{inst.module_key} = {sub_class}()")
                    registered_keys.add(inst.module_key)
                continue

            if getattr(inst, "is_repeat", False):
                sub_id = COMPOSITE_TYPE_MAP.get(inst.definition_id)
                rep_g = project.model.graphs.get(sub_id) if sub_id else None
                if not rep_g:
                    for cand_g in project.model.graphs.values():
                        if cand_g.kind == "repeat":
                            rep_g = cand_g
                            break

                if rep_g and rep_g.target_graph_id:
                    target_sub_class = f"Submodule_{_sanitize(rep_g.target_graph_id)}"
                    count_val = int(evaluate_value(rep_g.repeat_count or 1, cfg))
                    init_lines.append(
                        f"        self.{inst.module_key} = nn.ModuleList([{target_sub_class}() for _ in range({count_val})])"
                    )
                    registered_keys.add(inst.module_key)
                continue

            if inst.definition_id not in _SUPPORTED_DEFS:
                raise UnsupportedCookError(
                    f"Unsupported blueprint topology: Cannot cook node type {inst.definition_id}"
                )

            if not inst.module_key or inst.module_key in registered_keys:
                continue

            ir_node = ir_nodes.get(inst.node_id)
            props = ir_node.properties if ir_node else {}

            if inst.definition_id == "builtin.linear@1":
                in_f = int(evaluate_value(props.get("in_features", 16), cfg))
                out_f = int(evaluate_value(props.get("out_features", 16), cfg))
                bias = bool(evaluate_value(props.get("bias", True), cfg))
                init_lines.append(
                    f"        self.{inst.module_key} = nn.Linear({in_f}, {out_f}, bias={bias})"
                )
                registered_keys.add(inst.module_key)

            elif inst.definition_id == "builtin.embedding@1":
                n_emb = int(evaluate_value(props.get("num_embeddings", 256), cfg))
                e_dim = int(evaluate_value(props.get("embedding_dim", 32), cfg))
                init_lines.append(
                    f"        self.{inst.module_key} = nn.Embedding({n_emb}, {e_dim})"
                )
                registered_keys.add(inst.module_key)

            elif inst.definition_id == "builtin.layernorm@1":
                n_s = int(
                    evaluate_value(
                        props.get("normalized_shape", props.get("n_embd", 32)), cfg
                    )
                )
                eps = float(evaluate_value(props.get("eps", 1e-5), cfg))
                init_lines.append(
                    f"        self.{inst.module_key} = nn.LayerNorm({n_s}, eps={eps})"
                )
                registered_keys.add(inst.module_key)

            elif inst.definition_id == "builtin.rmsnorm@1":
                has_rmsnorm = True
                n_s = int(
                    evaluate_value(
                        props.get("normalized_shape", props.get("n_embd", 32)), cfg
                    )
                )
                eps = float(evaluate_value(props.get("eps", 1e-5), cfg))
                init_lines.append(
                    f"        self.{inst.module_key} = RMSNorm({n_s}, eps={eps})"
                )
                registered_keys.add(inst.module_key)

            elif inst.definition_id == "builtin.swiglu@1":
                has_swiglu = True
                n_e = int(evaluate_value(props.get("n_embd", 32), cfg))
                raw_h = props.get("hidden_dim")
                h_d = (
                    int(evaluate_value(raw_h, cfg)) if raw_h is not None else "None"
                )
                init_lines.append(
                    f"        self.{inst.module_key} = SwiGLU(n_embd={n_e}, hidden_dim={h_d})"
                )
                registered_keys.add(inst.module_key)

            elif inst.definition_id == "builtin.dropout@1":
                p = float(evaluate_value(props.get("p", props.get("dropout", 0.0)), cfg))
                init_lines.append(f"        self.{inst.module_key} = nn.Dropout(p={p})")
                registered_keys.add(inst.module_key)

            elif inst.definition_id == "builtin.lm_head@1":
                in_f = int(
                    evaluate_value(props.get("in_features", props.get("n_embd", 32)), cfg)
                )
                out_f = int(
                    evaluate_value(
                        props.get("out_features", props.get("vocab_size", 32)), cfg
                    )
                )
                bias = bool(evaluate_value(props.get("bias", False), cfg))
                init_lines.append(
                    f"        self.{inst.module_key} = nn.Linear({in_f}, {out_f}, bias={bias})"
                )
                registered_keys.add(inst.module_key)

        # Hierarchical weight tying in root module
        if is_root:
            node_to_attr: Dict[str, str] = {}
            for g_k, g_v in project.model.graphs.items():
                for n_k in g_v.nodes:
                    if g_k == target_root_id:
                        node_to_attr[n_k.id] = f"self.{n_k.id}"
                    else:
                        parent_inst = next(
                            (
                                i
                                for i in plan.instructions
                                if (
                                    COMPOSITE_TYPE_MAP.get(i.definition_id) == g_k
                                    or (
                                        i.definition_id.startswith("custom.")
                                        and i.definition_id[len("custom.") :] == g_k
                                    )
                                )
                            ),
                            None,
                        )
                        if parent_inst:
                            node_to_attr[n_k.id] = f"self.{parent_inst.module_key}.{n_k.id}"
                        else:
                            node_to_attr[n_k.id] = f"self.{n_k.id}"

            for wb in getattr(project.model, "weight_bindings", []):
                if getattr(wb, "mode", None) == "share":
                    src_id = getattr(getattr(wb, "source", None), "node_id", None) or getattr(
                        wb, "source_node_id", None
                    )
                    tgt_id = getattr(getattr(wb, "target", None), "node_id", None) or getattr(
                        wb, "target_node_id", None
                    )
                    src_attr = node_to_attr.get(src_id)
                    tgt_attr = node_to_attr.get(tgt_id)
                    if src_attr and tgt_attr:
                        init_lines.append(f"        {tgt_attr}.weight = {src_attr}.weight")

        if not init_lines:
            init_lines.append("        pass")

        # Forward pass
        last_temp = "x"
        for inst in plan.instructions:
            if inst.kind in _SKIP_KINDS or inst.definition_id in _STOP_DEFINITION_IDS:
                continue

            t_curr = f"t_{_sanitize(inst.node_path)}"
            bindings = inst.data_input_bindings

            def get_src(b) -> str:
                base = f"t_{_sanitize(b.source_node_path)}"
                if b.source_port_id and b.source_port_id not in (
                    "output",
                    "out",
                    "logits",
                ):
                    return f"{base}_{b.source_port_id}"
                return base

            if inst.definition_id in (
                "builtin.tensor_input@1",
                "builtin.token_input@1",
                "builtin.module_input@1",
            ):
                forward_lines.append(f"        {t_curr} = x")
                last_temp = t_curr

            elif inst.definition_id in (
                "builtin.graph_output@1",
                "builtin.logits_output@1",
                "builtin.module_output@1",
            ):
                src = get_src(bindings[0]) if bindings else last_temp
                forward_lines.append(f"        return {src}")
                last_temp = src

            elif getattr(inst, "is_repeat", False):
                src = get_src(bindings[0]) if bindings else last_temp
                forward_lines.append(f"        {t_curr} = {src}")
                forward_lines.append(f"        for block in self.{inst.module_key}:")
                forward_lines.append(f"            {t_curr} = block({t_curr})")
                last_temp = t_curr

            elif getattr(inst, "is_composite", False):
                src = get_src(bindings[0]) if bindings else last_temp
                forward_lines.append(f"        {t_curr} = self.{inst.module_key}({src})")
                last_temp = t_curr

            elif inst.definition_id in (
                "builtin.linear@1",
                "builtin.embedding@1",
                "builtin.layernorm@1",
                "builtin.rmsnorm@1",
                "builtin.swiglu@1",
                "builtin.dropout@1",
                "builtin.lm_head@1",
            ):
                src = get_src(bindings[0]) if bindings else last_temp
                forward_lines.append(f"        {t_curr} = self.{inst.module_key}({src})")
                last_temp = t_curr

            elif inst.definition_id in (
                "builtin.gelu@1",
                "builtin.silu@1",
                "builtin.relu@1",
            ):
                src = get_src(bindings[0]) if bindings else last_temp
                fn = {
                    "builtin.gelu@1": "F.gelu",
                    "builtin.silu@1": "F.silu",
                    "builtin.relu@1": "F.relu",
                }[inst.definition_id]
                forward_lines.append(f"        {t_curr} = {fn}({src})")
                last_temp = t_curr

            elif inst.definition_id == "builtin.add@1":
                if len(bindings) >= 2:
                    a, b = get_src(bindings[0]), get_src(bindings[1])
                    forward_lines.append(f"        {t_curr} = {a} + {b}")
                else:
                    src = get_src(bindings[0]) if bindings else last_temp
                    forward_lines.append(f"        {t_curr} = {src}")
                last_temp = t_curr

            elif inst.definition_id == "builtin.matmul@1":
                if len(bindings) >= 2:
                    a, b = get_src(bindings[0]), get_src(bindings[1])
                    forward_lines.append(f"        {t_curr} = {a} @ {b}")
                else:
                    src = get_src(bindings[0]) if bindings else last_temp
                    forward_lines.append(f"        {t_curr} = {src}")
                last_temp = t_curr

            elif inst.definition_id == "builtin.softmax@1":
                src = get_src(bindings[0]) if bindings else last_temp
                forward_lines.append(f"        {t_curr} = F.softmax({src}, dim=-1)")
                last_temp = t_curr

            elif inst.definition_id == "builtin.contiguous@1":
                src = get_src(bindings[0]) if bindings else last_temp
                forward_lines.append(f"        {t_curr} = {src}.contiguous()")
                last_temp = t_curr

            elif inst.definition_id == "builtin.flatten@1":
                src = get_src(bindings[0]) if bindings else last_temp
                forward_lines.append(f"        {t_curr} = torch.flatten({src}, 1)")
                last_temp = t_curr

            elif inst.definition_id == "builtin.scale@1":
                src = get_src(bindings[0]) if bindings else last_temp
                ir_node = ir_nodes.get(inst.node_id)
                scale_val = (
                    evaluate_value(ir_node.properties.get("scale", 1.0), cfg)
                    if ir_node
                    else 1.0
                )
                forward_lines.append(f"        {t_curr} = {src} * {scale_val}")
                last_temp = t_curr

            elif inst.definition_id == "builtin.split_qkv@1":
                src = get_src(bindings[0]) if bindings else last_temp
                forward_lines.append(
                    f"        {t_curr}_q, {t_curr}_k, {t_curr}_v = torch.chunk({src}, 3, dim=-1)"
                )
                forward_lines.append(f"        {t_curr} = {t_curr}_q")
                last_temp = t_curr

            elif inst.definition_id == "builtin.split_heads@1":
                src = get_src(bindings[0]) if bindings else last_temp
                ir_node = ir_nodes.get(inst.node_id)
                n_head_val = (
                    int(evaluate_value(ir_node.properties.get("n_head", cfg.get("n_head", 1)), cfg))
                    if ir_node
                    else 1
                )
                forward_lines.append(
                    f"        {t_curr} = {src}.view({src}.size(0), {n_head_val}, -1)"
                )
                last_temp = t_curr

            elif inst.definition_id == "builtin.sdpa@1":
                b_q = next((b for b in bindings if b.port_id == "q"), None)
                b_k = next((b for b in bindings if b.port_id == "k"), None)
                b_v = next((b for b in bindings if b.port_id == "v"), None)
                q_var = get_src(b_q) if b_q else last_temp
                k_var = get_src(b_k) if b_k else last_temp
                v_var = get_src(b_v) if b_v else last_temp
                ir_node = ir_nodes.get(inst.node_id)
                is_causal = (
                    bool(evaluate_value(ir_node.properties.get("is_causal", False), cfg))
                    if ir_node
                    else False
                )
                forward_lines.append(
                    f"        {t_curr} = F.scaled_dot_product_attention({q_var}, {k_var}, {v_var}, is_causal={is_causal})"
                )
                last_temp = t_curr

            elif inst.definition_id == "builtin.merge_heads@1":
                src = get_src(bindings[0]) if bindings else last_temp
                forward_lines.append(
                    f"        {t_curr} = {src}.contiguous().view({src}.size(0), -1)"
                )
                last_temp = t_curr

            elif inst.definition_id == "builtin.rope@1":
                has_rope = True
                src = get_src(bindings[0]) if bindings else last_temp
                ir_node = ir_nodes.get(inst.node_id)
                n_h = (
                    int(evaluate_value(ir_node.properties.get("n_head", cfg.get("n_head", 4)), cfg))
                    if ir_node
                    else 4
                )
                h_d = (
                    int(evaluate_value(ir_node.properties.get("head_dim", 8), cfg))
                    if ir_node
                    else 8
                )
                forward_lines.append(
                    f"        {t_curr} = apply_rope({src}, head_dim={h_d}, n_head={n_h})"
                )
                last_temp = t_curr

            elif inst.definition_id == "builtin.grouped_query_attention@1":
                has_gqa = True
                b_q = next((b for b in bindings if b.port_id == "q"), None)
                b_k = next((b for b in bindings if b.port_id == "k"), None)
                b_v = next((b for b in bindings if b.port_id == "v"), None)
                q_v = get_src(b_q) if b_q else last_temp
                k_v = get_src(b_k) if b_k else last_temp
                v_v = get_src(b_v) if b_v else last_temp
                ir_node = ir_nodes.get(inst.node_id)
                n_h = (
                    int(evaluate_value(ir_node.properties.get("n_head", cfg.get("n_head", 4)), cfg))
                    if ir_node
                    else 4
                )
                n_kv = (
                    int(evaluate_value(ir_node.properties.get("n_kv_head", n_h), cfg))
                    if ir_node
                    else n_h
                )
                forward_lines.append(
                    f"        {t_curr} = run_gqa({q_v}, {k_v}, {v_v}, n_head={n_h}, n_kv_head={n_kv})"
                )
                last_temp = t_curr

            else:
                src = get_src(bindings[0]) if bindings else last_temp
                forward_lines.append(f"        {t_curr} = {src}")
                last_temp = t_curr

        if not any("return " in line for line in forward_lines):
            forward_lines.append(f"        return {last_temp}")

        c_init = "\n".join(init_lines)
        c_fwd = "\n".join(forward_lines)
        class_defs.append(
            f"""class {class_name}(nn.Module):
    def __init__(self):
        super().__init__()
{c_init}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
{c_fwd}
"""
        )

    helpers: List[str] = []
    if has_rmsnorm:
        helpers.append(
            """class RMSNorm(nn.Module):
    def __init__(self, ndim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * rms
"""
        )

    if has_swiglu:
        helpers.append(
            """class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            raw_h = int(8 * n_embd / 3)
            hidden_dim = ((raw_h + 7) // 8) * 8
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
"""
        )

    if has_rope:
        helpers.append(
            """def apply_rope(x: torch.Tensor, head_dim: int = 8, n_head: int = 4, base: float = 10000.0) -> torch.Tensor:
    orig_shape = x.shape
    orig_dim = x.dim()
    if orig_dim == 3:
        B, T, C = x.shape
        d = head_dim if head_dim > 0 else (C // n_head)
        num_h = C // d
        x_4d = x.view(B, T, num_h, d).transpose(1, 2)
    elif orig_dim == 4:
        x_4d = x
    else:
        return x
    D = x_4d.size(-1)
    T = x_4d.size(-2)
    device = x.device
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    x1 = x_4d[..., ::2]
    x2 = x_4d[..., 1::2]
    even_rot = x1 * cos - x2 * sin
    odd_rot = x1 * sin + x2 * cos
    rotated = torch.stack([even_rot, odd_rot], dim=-1).flatten(-2)
    if orig_dim == 3:
        return rotated.transpose(1, 2).contiguous().view(orig_shape)
    return rotated
"""
        )

    if has_gqa:
        helpers.append(
            """def run_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_head: int = 4, n_kv_head: int = 2) -> torch.Tensor:
    B, T_q, C_q = q.size(0), q.size(1), q.size(-1)
    head_dim = C_q // n_head
    if q.dim() == 3:
        q_heads = q.view(B, T_q, n_head, head_dim).transpose(1, 2)
    else:
        q_heads = q
    if k.dim() == 3:
        T_k = k.size(1)
        kv_dim = k.size(-1)
        kv_head_dim = kv_dim // n_kv_head
        if kv_head_dim != head_dim and kv_dim == C_q:
            k = k[..., : n_kv_head * head_dim]
            v = v[..., : n_kv_head * head_dim]
            kv_head_dim = head_dim
        k_heads = k.view(B, T_k, n_kv_head, kv_head_dim).transpose(1, 2)
        v_heads = v.view(B, T_k, n_kv_head, kv_head_dim).transpose(1, 2)
    else:
        k_heads = k
        v_heads = v
    if n_head != n_kv_head:
        reps = n_head // n_kv_head
        k_heads = torch.repeat_interleave(k_heads, reps, dim=1)
        v_heads = torch.repeat_interleave(v_heads, reps, dim=1)
    out = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, is_causal=True)
    return out.transpose(1, 2).contiguous().view(B, T_q, n_head * head_dim)
"""
        )

    all_helpers = "\n".join(helpers)
    all_classes = "\n".join(class_defs)

    has_token_in = any(
        n.definition_id == "builtin.token_input@1"
        for g in project.model.graphs.values()
        for n in g.nodes
    )
    vocab_size = int(cfg.get("vocab_size", 32))
    block_size = int(cfg.get("block_size", 8))
    in_features = int(
        evaluate_value(
            cfg.get("in_features", cfg.get("in_dim", cfg.get("n_embd", 32))), cfg
        )
    )

    input_gen = (
        f"dataset_x = torch.randint(0, {vocab_size}, (num_samples, {block_size})).to(device)"
        if has_token_in
        else f"dataset_x = torch.randn(num_samples, {in_features}).to(device)"
    )

    if mode == "inference":
        main_body = f"""def main():
    parser = argparse.ArgumentParser(description="bpTorch Inference Runner")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device")
    args = parser.parse_args()

    device = args.device
    model = CookedModel().to(device)
    model.eval()

    num_samples = args.batch_size
    {input_gen}
    bx = dataset_x[:args.batch_size]

    with torch.no_grad():
        out = model(bx)

    if isinstance(out, dict):
        out = next(iter(out.values()))
    elif isinstance(out, (list, tuple)):
        out = out[0]

    print("Inference successful! Output shape:", out.shape)
"""
    else:
        main_body = f"""def main():
    parser = argparse.ArgumentParser(description="bpTorch Cooked Model Training")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device")
    parser.add_argument("--precision", type=str, default="fp32", help="Precision (fp32, fp16, bf16)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    print("=== Starting Blueprint Model Training ===")
    model = CookedModel().to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_samples = max(20, args.batch_size * 4)
    {input_gen}

    start_time = time.time()
    for step in range(args.max_steps):
        idx = (step * args.batch_size) % (num_samples - args.batch_size + 1)
        bx = dataset_x[idx : idx + args.batch_size]

        optimizer.zero_grad(set_to_none=True)
        out = model(bx)

        if isinstance(out, dict):
            out = next(iter(out.values()))
        elif isinstance(out, (list, tuple)):
            out = out[0]

        if not out.is_floating_point():
            out = out.float()

        target = torch.randn_like(out)
        loss = F.mse_loss(out, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        print(f"Step {{step + 1:4d}}/{{args.max_steps}} | Loss: {{loss.item():.4f}}")

    total_time = time.time() - start_time
    print(f"=== Training Complete in {{total_time:.2f}}s ===")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "ckpt_final.pt"
    torch.save({{"model_state_dict": model.state_dict()}}, ckpt_path)
    print(f"Saved final checkpoint to: {{ckpt_path}}")
"""

    return f"""#!/usr/bin/env python3
\"\"\"
Standalone PyTorch Model & Training Script
Generated by bpTorch Generic Cooker for project: {project.project.name}
\"\"\"

import argparse
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

{all_helpers}

{all_classes}

{main_body}

if __name__ == "__main__":
    main()
"""
