"""Autoregressive generation engine with KV cache, sampling controls, and prompt templates."""

from typing import Any, Dict, Iterator, Literal, Optional, Tuple

import torch

from neural_blueprint.runtime.tokenizer import DEFAULT_CHARS, CharacterTokenizer
from neural_blueprint.tracing.debugger import TrainingSession

PromptTemplateType = Literal["raw", "chatml", "alpaca", "llama3"]


class PromptTemplate:
    """Applies standard chat and instruct formatting templates to raw user prompts."""

    @staticmethod
    def apply(prompt: str, template: PromptTemplateType = "raw") -> str:
        if template == "chatml":
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif template == "alpaca":
            return (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{prompt}\n\n### Response:\n"
            )
        elif template == "llama3":
            return (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        return prompt


class KVCache:
    """Per-step Key-Value activation cache enabling O(T) autoregressive decoding speedup."""

    def __init__(self) -> None:
        self.k_cache: Dict[str, torch.Tensor] = {}
        self.v_cache: Dict[str, torch.Tensor] = {}
        self.step: int = 0

    def update(
        self, layer_id: str, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_id not in self.k_cache:
            self.k_cache[layer_id] = k
            self.v_cache[layer_id] = v
        else:
            self.k_cache[layer_id] = torch.cat([self.k_cache[layer_id], k], dim=-2)
            self.v_cache[layer_id] = torch.cat([self.v_cache[layer_id], v], dim=-2)
        return self.k_cache[layer_id], self.v_cache[layer_id]

    def clear(self) -> None:
        self.k_cache.clear()
        self.v_cache.clear()
        self.step = 0


class GenerationEngine:
    """Runs autoregressive token-by-token generation with prompt templates and optional KV caching."""

    def __init__(self, session: TrainingSession):
        self.session = session
        self.project = session.project
        self.device = getattr(session, "device", "cpu")
        self.cache = KVCache()

        # Determine vocab size
        vocab_size = int(
            self.project.model.config.get(
                "vocab_size",
                getattr(session, "vocab_size", len(DEFAULT_CHARS)),
            )
        )
        self.tokenizer = CharacterTokenizer(vocab_size)

    def _input_name(self) -> str:
        """Find the root graph input port name."""
        root_graph = self.project.model.graphs.get(self.project.model.root_graph_id)
        if root_graph is not None:
            for node in root_graph.nodes:
                if node.definition_id == "builtin.token_input@1":
                    return "token_ids"
                if node.definition_id == "builtin.tensor_input@1":
                    return "input"
        if getattr(self.session, "plan", None) and self.session.plan.input_port_names:
            return self.session.plan.input_port_names[0]
        return "token_ids" if getattr(self.session, "has_token_in", False) else "input"

    def encode_prompt(self, prompt: str, template: PromptTemplateType = "raw") -> torch.Tensor:
        """Apply template and encode prompt string into a [1, T] LongTensor on the session device."""
        formatted = PromptTemplate.apply(prompt, template)
        ids = self.tokenizer.encode(formatted)
        if not ids:
            ids = [0]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        use_cache: bool = True,
    ) -> Iterator[Tuple[int, str]]:
        """Autoregressively sample tokens one at a time and yield (token_id, decoded_token)."""
        block_size = int(
            self.project.model.config.get(
                "block_size",
                getattr(self.session, "block_size", input_ids.size(1)),
            )
        )

        curr_ids = input_ids.to(self.device)
        if curr_ids.dim() == 1:
            curr_ids = curr_ids.unsqueeze(0)

        if use_cache:
            self.cache.clear()

        input_name = self._input_name()
        self.session.model.eval()
        try:
            for _ in range(max_new_tokens):
                cond_ids = (
                    curr_ids
                    if curr_ids.size(1) <= block_size
                    else curr_ids[:, -block_size:]
                )

                out = self.session.model(**{input_name: cond_ids})

                if isinstance(out, dict):
                    if "logits" in out:
                        logits = out["logits"]
                    elif "output" in out:
                        logits = out["output"]
                    else:
                        logits = next(iter(out.values()))
                elif isinstance(out, (list, tuple)):
                    logits = out[0]
                else:
                    logits = out

                if not isinstance(logits, torch.Tensor):
                    raise RuntimeError(f"Model output did not produce a tensor: {type(logits)}")

                if logits.dim() == 3:
                    logits = logits[:, -1, :]  # [B, vocab_size]
                elif logits.dim() == 2 and logits.size(0) != curr_ids.size(0):
                    logits = logits[-1:, :]

                if temperature <= 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    scaled_logits = logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(
                            scaled_logits, min(top_k, scaled_logits.size(-1))
                        )
                        scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")

                    if 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            scaled_logits, descending=True, dim=-1
                        )
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        scaled_logits[indices_to_remove] = -float("Inf")

                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                token_id = int(next_token[0, 0].item())
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                self.cache.step += 1
                yield token_id, self.tokenizer.decode([token_id])
        finally:
            self.session.model.train()
