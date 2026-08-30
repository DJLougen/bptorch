"""Bidirectional state-dict semantic weight mapper between karpathy/nanoGPT and CompiledGraphModule."""

from typing import List, Tuple

import torch
import torch.nn as nn


class StateDictMapper:
    """Explicit semantic weight mapper between karpathy/nanoGPT reference and visual runtime."""

    @staticmethod
    def build_parameter_map(n_layer: int = 2) -> List[Tuple[str, str]]:
        """
        Builds authoritative parameter mapping pairs:
        (reference_nanogpt_key, compiled_visual_runtime_key)
        """
        mappings = [
            (
                "transformer.wte.weight",
                "module_dict.node_input_embeddings.module_dict.node_tok_emb.weight",
            ),
            (
                "transformer.wpe.weight",
                "module_dict.node_input_embeddings.module_dict.node_pos_emb.weight",
            ),
            ("transformer.ln_f.weight", "module_dict.node_ln_f.weight"),
            ("transformer.ln_f.bias", "module_dict.node_ln_f.bias"),
            ("lm_head.weight", "module_dict.node_lm_head.weight"),
        ]

        for i in range(n_layer):
            block_prefix = f"module_dict.node_transformer_stack.{i}"
            ref_prefix = f"transformer.h.{i}"

            mappings.extend(
                [
                    (f"{ref_prefix}.ln_1.weight", f"{block_prefix}.module_dict.node_ln_1.weight"),
                    (f"{ref_prefix}.ln_1.bias", f"{block_prefix}.module_dict.node_ln_1.bias"),
                    (
                        f"{ref_prefix}.attn.c_attn.weight",
                        f"{block_prefix}.module_dict.node_attn_subgraph.module_dict.node_qkv_proj.weight",
                    ),
                    (
                        f"{ref_prefix}.attn.c_attn.bias",
                        f"{block_prefix}.module_dict.node_attn_subgraph.module_dict.node_qkv_proj.bias",
                    ),
                    (
                        f"{ref_prefix}.attn.c_proj.weight",
                        f"{block_prefix}.module_dict.node_attn_subgraph.module_dict.node_attn_c_proj.weight",
                    ),
                    (
                        f"{ref_prefix}.attn.c_proj.bias",
                        f"{block_prefix}.module_dict.node_attn_subgraph.module_dict.node_attn_c_proj.bias",
                    ),
                    (f"{ref_prefix}.ln_2.weight", f"{block_prefix}.module_dict.node_ln_2.weight"),
                    (f"{ref_prefix}.ln_2.bias", f"{block_prefix}.module_dict.node_ln_2.bias"),
                    (
                        f"{ref_prefix}.mlp.c_fc.weight",
                        f"{block_prefix}.module_dict.node_mlp_subgraph.module_dict.node_mlp_c_fc.weight",
                    ),
                    (
                        f"{ref_prefix}.mlp.c_fc.bias",
                        f"{block_prefix}.module_dict.node_mlp_subgraph.module_dict.node_mlp_c_fc.bias",
                    ),
                    (
                        f"{ref_prefix}.mlp.c_proj.weight",
                        f"{block_prefix}.module_dict.node_mlp_subgraph.module_dict.node_mlp_c_proj.weight",
                    ),
                    (
                        f"{ref_prefix}.mlp.c_proj.bias",
                        f"{block_prefix}.module_dict.node_mlp_subgraph.module_dict.node_mlp_c_proj.bias",
                    ),
                ]
            )

        return mappings

    @classmethod
    def transfer_weights(
        cls, ref_model: nn.Module, visual_model: nn.Module, n_layer: int = 2
    ) -> None:
        """Copies weights from reference nanoGPT model into the compiled visual runtime model."""
        ref_sd = ref_model.state_dict()
        vis_sd = visual_model.state_dict()
        mappings = cls.build_parameter_map(n_layer=n_layer)

        with torch.no_grad():
            for ref_k, vis_k in mappings:
                if ref_k not in ref_sd:
                    raise KeyError(f"Reference key '{ref_k}' not found in reference state_dict")
                if vis_k not in vis_sd:
                    raise KeyError(f"Visual key '{vis_k}' not found in visual state_dict")

                ref_tensor = ref_sd[ref_k]
                vis_tensor = vis_sd[vis_k]

                if ref_tensor.shape != vis_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {ref_k} ({ref_tensor.shape}) vs {vis_k} ({vis_tensor.shape})"
                    )

                vis_tensor.copy_(ref_tensor)

    @classmethod
    def verify_map_completeness(
        cls, ref_model: nn.Module, visual_model: nn.Module, n_layer: int = 2
    ) -> None:
        """Verifies 100% parameter coverage between reference nanoGPT and visual runtime."""
        ref_sd = {k: v for k, v in ref_model.state_dict().items() if not k.endswith(".attn.bias")}
        vis_sd = visual_model.state_dict()
        mappings = cls.build_parameter_map(n_layer=n_layer)

        mapped_ref = {ref_k for ref_k, _ in mappings}
        mapped_vis = {vis_k for _, vis_k in mappings}

        unmapped_ref = set(ref_sd.keys()) - mapped_ref
        if unmapped_ref:
            raise ValueError(f"Unmapped reference parameters: {unmapped_ref}")

        unmapped_vis = set(vis_sd.keys()) - mapped_vis
        if unmapped_vis:
            raise ValueError(f"Unmapped visual parameters: {unmapped_vis}")

        # Check tied weight storage sharing
        wte_param = (
            visual_model.module_dict["node_input_embeddings"].module_dict["node_tok_emb"].weight
        )
        lm_head_param = visual_model.module_dict["node_lm_head"].weight
        if wte_param.data_ptr() != lm_head_param.data_ptr():
            raise AssertionError("Weight tying failed: wte and lm_head do not share data_ptr")
