"""nanoGPT Reference and Numerical Parity Harness runner."""

import sys
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW

# Add references to sys.path to import pinned nanoGPT
ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(ROOT_DIR / "references" / "nanogpt") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "references" / "nanogpt"))

from model import GPT, GPTConfig  # noqa: E402

from neural_blueprint.parity.mapper import StateDictMapper  # noqa: E402
from neural_blueprint.runtime.compiler import GraphCompiler  # noqa: E402
from neural_blueprint.runtime.module import CompiledGraphModule  # noqa: E402
from neural_blueprint.templates.nanogpt import create_nanogpt_template  # noqa: E402


class ParityRunner:
    """Automated numerical parity suite verifying visual runtime against pinned karpathy/nanoGPT."""

    def __init__(
        self,
        block_size: int = 8,
        vocab_size: int = 32,
        n_layer: int = 2,
        n_head: int = 2,
        n_embd: int = 16,
        dropout: float = 0.0,
        bias: bool = True,
        attention_impl: str = "manual",
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.attention_impl = attention_impl

        # Reference nanoGPT model
        self.ref_config = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
        )
        self.ref_model = GPT(self.ref_config)

        # Visual runtime model
        self.project = create_nanogpt_template(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
            attention_impl=attention_impl,
        )
        compiler = GraphCompiler()
        plan, modules = compiler.compile_plan(self.project)
        self.visual_model = CompiledGraphModule(plan, modules, self.project.model.weight_bindings)

        # Sync weights
        StateDictMapper.transfer_weights(self.ref_model, self.visual_model, n_layer=n_layer)

    def run_all_checks(self) -> Dict[str, bool]:
        """Runs the complete numerical parity test suite."""
        return {
            "weight_map_completeness": self.check_weight_map_completeness(),
            "parameter_count_parity": self.check_parameter_count_parity(),
            "forward_parity": self.check_forward_parity(),
            "intermediate_parity": self.check_intermediate_parity(),
            "gradient_parity": self.check_gradient_parity(),
            "optimizer_step_parity": self.check_optimizer_step_parity(),
            "inference_parity": self.check_inference_parity(),
        }

    def check_weight_map_completeness(self) -> bool:
        StateDictMapper.verify_map_completeness(
            self.ref_model, self.visual_model, n_layer=self.n_layer
        )
        return True

    def check_parameter_count_parity(self) -> bool:
        ref_params = sum(p.numel() for p in self.ref_model.parameters())
        vis_params = sum(p.numel() for p in set(self.visual_model.parameters()))
        assert ref_params == vis_params, (
            f"Parameter count mismatch: ref={ref_params}, vis={vis_params}"
        )
        return True

    def check_forward_parity(self) -> bool:
        self.ref_model.eval()
        self.visual_model.eval()

        torch.manual_seed(1337)
        idx = torch.randint(0, self.vocab_size, (2, self.block_size))
        targets = torch.randint(0, self.vocab_size, (2, self.block_size))

        with torch.no_grad():
            ref_logits, ref_loss = self.ref_model(idx, targets=targets)
            vis_out = self.visual_model(token_ids=idx, targets=targets)

        vis_logits = vis_out["logits"]
        vis_loss = vis_out["loss"]

        torch.testing.assert_close(vis_logits, ref_logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(vis_loss, ref_loss, rtol=1e-5, atol=1e-6)
        return True

    def check_intermediate_parity(self) -> bool:
        """Verifies parity on intermediate activations."""
        self.ref_model.eval()
        self.visual_model.eval()

        torch.manual_seed(1337)
        idx = torch.randint(0, self.vocab_size, (2, self.block_size))
        pos = torch.arange(0, self.block_size, dtype=torch.long)

        # 1. Embeddings
        ref_tok_emb = self.ref_model.transformer.wte(idx)
        ref_pos_emb = self.ref_model.transformer.wpe(pos)
        vis_tok_emb = self.visual_model.module_dict["node_input_embeddings"].module_dict[
            "node_tok_emb"
        ](idx)
        vis_pos_emb = self.visual_model.module_dict["node_input_embeddings"].module_dict[
            "node_pos_emb"
        ](pos)

        torch.testing.assert_close(vis_tok_emb, ref_tok_emb, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(vis_pos_emb, ref_pos_emb, rtol=1e-5, atol=1e-6)

        # 2. Block 0
        ref_x = self.ref_model.transformer.drop(ref_tok_emb + ref_pos_emb)
        ref_b0 = self.ref_model.transformer.h[0]
        vis_b0 = self.visual_model.module_dict["node_transformer_stack"][0]

        ref_b0_out = ref_b0(ref_x)
        vis_b0_out = vis_b0(ref_x)
        torch.testing.assert_close(vis_b0_out, ref_b0_out, rtol=1e-5, atol=1e-6)

        # 3. Final LayerNorm
        ref_ln_f_out = self.ref_model.transformer.ln_f(ref_b0_out)
        vis_ln_f_out = self.visual_model.module_dict["node_ln_f"](vis_b0_out)
        torch.testing.assert_close(vis_ln_f_out, ref_ln_f_out, rtol=1e-5, atol=1e-6)

        return True

    def check_gradient_parity(self) -> bool:
        self.ref_model.train()
        self.visual_model.train()

        self.ref_model.zero_grad()
        self.visual_model.zero_grad()

        torch.manual_seed(1337)
        idx = torch.randint(0, self.vocab_size, (2, self.block_size))
        targets = torch.randint(0, self.vocab_size, (2, self.block_size))

        ref_logits, ref_loss = self.ref_model(idx, targets=targets)
        vis_out = self.visual_model(token_ids=idx, targets=targets)
        vis_loss = vis_out["loss"]

        ref_loss.backward()
        vis_loss.backward()

        # Compare gradients
        ref_wte_grad = self.ref_model.transformer.wte.weight.grad
        vis_wte_grad = (
            self.visual_model.module_dict["node_input_embeddings"]
            .module_dict["node_tok_emb"]
            .weight.grad
        )
        torch.testing.assert_close(vis_wte_grad, ref_wte_grad, rtol=1e-5, atol=1e-6)

        # Compare block 0 attention gradients
        ref_c_attn_grad = self.ref_model.transformer.h[0].attn.c_attn.weight.grad
        vis_c_attn_grad = (
            self.visual_model.module_dict["node_transformer_stack"][0]
            .module_dict["node_attn_subgraph"]
            .module_dict["node_qkv_proj"]
            .weight.grad
        )
        torch.testing.assert_close(vis_c_attn_grad, ref_c_attn_grad, rtol=1e-5, atol=1e-6)

        return True

    def check_optimizer_step_parity(self) -> bool:
        self.ref_model.train()
        self.visual_model.train()

        opt_ref = AdamW(self.ref_model.parameters(), lr=1e-3, weight_decay=1e-1)
        opt_vis = AdamW(self.visual_model.parameters(), lr=1e-3, weight_decay=1e-1)

        torch.manual_seed(1337)
        idx = torch.randint(0, self.vocab_size, (2, self.block_size))
        targets = torch.randint(0, self.vocab_size, (2, self.block_size))

        # Forward + backward + step
        _, ref_loss = self.ref_model(idx, targets=targets)
        vis_out = self.visual_model(token_ids=idx, targets=targets)

        opt_ref.zero_grad()
        opt_vis.zero_grad()

        ref_loss.backward()
        vis_out["loss"].backward()

        opt_ref.step()
        opt_vis.step()

        # Compare updated weights
        ref_wte = self.ref_model.transformer.wte.weight
        vis_wte = (
            self.visual_model.module_dict["node_input_embeddings"]
            .module_dict["node_tok_emb"]
            .weight
        )
        torch.testing.assert_close(vis_wte, ref_wte, rtol=1e-5, atol=1e-6)

        return True

    def check_inference_parity(self) -> bool:
        self.ref_model.eval()
        self.visual_model.eval()

        torch.manual_seed(1337)
        idx = torch.randint(0, self.vocab_size, (2, self.block_size))

        with torch.no_grad():
            ref_logits, ref_loss = self.ref_model(idx, targets=None)
            vis_out = self.visual_model(token_ids=idx)

        assert ref_loss is None
        assert ref_logits.shape == torch.Size([2, 1, self.vocab_size])
        assert vis_out is not None
        return True
