"""Sample catalog metadata for the 25 bundled architecture demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from neural_blueprint.templates.architectures import ALL_ARCHITECTURES


@dataclass(frozen=True)
class SampleCatalogEntry:
    id: str
    name: str
    category: str
    description: str
    highlight: str
    tags: List[str] = field(default_factory=list)
    difficulty: str = "intermediate"
    filename: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "highlight": self.highlight,
            "tags": self.tags,
            "difficulty": self.difficulty,
            "filename": self.filename or f"{self.id}.nbp.json",
            "path": f"/examples/{self.id}/{self.filename or f'{self.id}.nbp.json'}",
        }


# Rich metadata keyed by builder display name (stable across exports)
SAMPLE_METADATA: Dict[str, Dict[str, Any]] = {
    "Arch 1: nanoGPT Tiny": {
        "category": "Transformers",
        "description": "Canonical tiny GPT with hierarchical subgraphs: embeddings, blocks, attention, and LM head.",
        "highlight": "Best starting point for exploring compositional transformer blueprints.",
        "tags": ["transformer", "nanogpt", "hierarchical", "sdpa"],
        "difficulty": "beginner",
    },
    "Arch 2: nanoGPT Deep (6L)": {
        "category": "Transformers",
        "description": "Six-layer transformer stack for studying depth scaling and gradient flow.",
        "highlight": "Shows how repeat stacks compose deep models from reusable block subgraphs.",
        "tags": ["transformer", "deep", "stack"],
        "difficulty": "intermediate",
    },
    "Arch 3: nanoGPT Wide (1L/8H)": {
        "category": "Transformers",
        "description": "Single-layer, eight-head wide transformer emphasizing parallel attention heads.",
        "highlight": "Demonstrates width-vs-depth tradeoffs in a compact blueprint.",
        "tags": ["transformer", "wide", "multi-head"],
        "difficulty": "intermediate",
    },
    "Arch 4: Two-Layer MLP": {
        "category": "Feedforward",
        "description": "Classic two-layer perceptron with GELU activation for tabular/feature tasks.",
        "highlight": "Simplest trainable blueprint — ideal for first edits and property tuning.",
        "tags": ["mlp", "gelu", "classification"],
        "difficulty": "beginner",
    },
    "Arch 5: Bottleneck Autoencoder": {
        "category": "Autoencoders",
        "description": "Encoder-decoder MLP with a narrow bottleneck for representation learning.",
        "highlight": "Illustrates hourglass topology and reconstruction-style training.",
        "tags": ["autoencoder", "bottleneck", "encoder-decoder"],
        "difficulty": "intermediate",
    },
    "Arch 6: Manual Attention Transformer": {
        "category": "Attention",
        "description": "nanoGPT variant wired with explicit manual causal attention instead of SDPA.",
        "highlight": "Peel back the attention black box — every matmul is visible on the canvas.",
        "tags": ["attention", "manual", "causal", "transformer"],
        "difficulty": "advanced",
    },
    "Arch 7: Dual-Flow Pipeline": {
        "category": "Training Pipelines",
        "description": "Event-driven training graph with separate data and execution flows.",
        "highlight": "Blueprint training as a visual program: dataloader → forward → backward → optimizer.",
        "tags": ["dual-flow", "event-graph", "training-loop"],
        "difficulty": "advanced",
    },
    "Arch 8: ResMLP Residual Network": {
        "category": "Feedforward",
        "description": "Residual MLP block with LayerNorm and skip connections.",
        "highlight": "Residual wiring without transformers — shows skip connections in pure MLP form.",
        "tags": ["residual", "mlp", "layernorm"],
        "difficulty": "intermediate",
    },
    "Arch 9: Multi-Head Projection": {
        "category": "Attention",
        "description": "Explicit QKV split, head split/merge, and SDPA attention wiring.",
        "highlight": "Every attention primitive is a separate node — maximum transparency.",
        "tags": ["attention", "qkv", "sdpa", "multi-head"],
        "difficulty": "advanced",
    },
    "Arch 10: Multi-Task Joint Network": {
        "category": "Multi-Task",
        "description": "Shared backbone with parallel LM and classification heads plus weight tying.",
        "highlight": "Fan-out topology with tied embeddings — one backbone, two objectives.",
        "tags": ["multi-task", "weight-tying", "classification", "lm"],
        "difficulty": "advanced",
    },
    "Arch 11: ReLU Classifier MLP": {
        "category": "Feedforward",
        "description": "Two-layer MLP using ReLU instead of GELU for sharp activation boundaries.",
        "highlight": "Swap activations in one property — compare ReLU vs GELU behavior.",
        "tags": ["mlp", "relu", "activation"],
        "difficulty": "beginner",
    },
    "Arch 12: Dropout MLP": {
        "category": "Regularization",
        "description": "Feedforward network with dropout between hidden layers.",
        "highlight": "Regularization as a first-class blueprint node, not a hidden flag.",
        "tags": ["dropout", "mlp", "regularization"],
        "difficulty": "beginner",
    },
    "Arch 13: Deep MLP Tower": {
        "category": "Feedforward",
        "description": "Five stacked linear+GELU layers without any attention.",
        "highlight": "Pure depth — see how deep feedforward towers compose from primitives.",
        "tags": ["mlp", "deep", "tower"],
        "difficulty": "intermediate",
    },
    "Arch 14: Wide-and-Deep Network": {
        "category": "Hybrid",
        "description": "Parallel wide memorization path and deep generalization path fused with Add.",
        "highlight": "Google-style wide-and-deep architecture built from basic nodes.",
        "tags": ["wide-deep", "fusion", "add", "hybrid"],
        "difficulty": "intermediate",
    },
    "Arch 15: Tied-Embedding LM": {
        "category": "Language Models",
        "description": "Minimal language model: embedding → LM head with shared weights.",
        "highlight": "Smallest possible LM — weight tying visible as an explicit binding.",
        "tags": ["lm", "embedding", "weight-tying", "minimal"],
        "difficulty": "beginner",
    },
    "Arch 16: Warmup Scheduler Pipeline": {
        "category": "Training Pipelines",
        "description": "Dual-flow graph with linear warmup LR scheduler node.",
        "highlight": "Learning-rate schedules as blueprint nodes in the training event graph.",
        "tags": ["scheduler", "warmup", "dual-flow"],
        "difficulty": "advanced",
    },
    "Arch 17: Step-LR Decay Pipeline": {
        "category": "Training Pipelines",
        "description": "Dual-flow graph with step-wise learning rate decay.",
        "highlight": "Compare scheduler strategies side-by-side with Arch 16.",
        "tags": ["scheduler", "step-lr", "dual-flow"],
        "difficulty": "advanced",
    },
    "Arch 18: Pre-LayerNorm MLP": {
        "category": "Normalization",
        "description": "MLP with LayerNorm before each linear layer (pre-norm pattern).",
        "highlight": "Normalization placement is a wiring choice, not framework magic.",
        "tags": ["layernorm", "pre-norm", "mlp"],
        "difficulty": "intermediate",
    },
    "Arch 19: Residual Add MLP": {
        "category": "Feedforward",
        "description": "Single residual block using an explicit Add node for the skip connection.",
        "highlight": "Skip connections drawn as visible edges — not buried in module code.",
        "tags": ["residual", "add", "skip-connection"],
        "difficulty": "intermediate",
    },
    "Arch 20: Binary Sequence Classifier": {
        "category": "Classification",
        "description": "Token embedding → linear head for binary sequence classification.",
        "highlight": "NLP classification without transformers — just embeddings and a head.",
        "tags": ["classification", "embedding", "binary"],
        "difficulty": "beginner",
    },
    "Arch 21: High-Dropout Transformer": {
        "category": "Transformers",
        "description": "nanoGPT with 20% dropout for regularization-heavy training.",
        "highlight": "Tune dropout on the canvas and watch training dynamics change.",
        "tags": ["transformer", "dropout", "regularization"],
        "difficulty": "intermediate",
    },
    "Arch 22: BF16 nanoGPT Micro": {
        "category": "Precision",
        "description": "Tiny nanoGPT configured for bfloat16 mixed-precision training.",
        "highlight": "Mixed precision is a training property — cook and run with --precision bf16.",
        "tags": ["bf16", "mixed-precision", "transformer"],
        "difficulty": "intermediate",
    },
    "Arch 23: Single-Block Causal GPT": {
        "category": "Transformers",
        "description": "Smallest causal transformer: one block, four heads.",
        "highlight": "Minimal transformer for teaching attention without stack complexity.",
        "tags": ["transformer", "minimal", "causal"],
        "difficulty": "beginner",
    },
    "Arch 24: SiLU Deep Feedforward": {
        "category": "Feedforward",
        "description": "Two-layer MLP with SiLU (Swish) activation.",
        "highlight": "Three activation primitives (GELU, ReLU, SiLU) — pick yours on the canvas.",
        "tags": ["mlp", "silu", "activation"],
        "difficulty": "beginner",
    },
    "Arch 25: Metric Logger Pipeline": {
        "category": "Training Pipelines",
        "description": "Dual-flow training graph with a Metric Logger node in the exec chain.",
        "highlight": "Observability wired into the training program — metrics as blueprint nodes.",
        "tags": ["metrics", "logging", "dual-flow", "observability"],
        "difficulty": "advanced",
    },
    "Arch 26: Llama Tiny": {
        "category": "Transformers",
        "description": "Modern Llama-style Transformer with RMSNorm, RoPE, SwiGLU, and Grouped Query Attention (GQA).",
        "highlight": "State-of-the-art open-weights architecture blueprint with rotary embeddings and GQA.",
        "tags": ["llama", "rmsnorm", "rope", "swiglu", "gqa"],
        "difficulty": "advanced",
    },
}


def build_catalog() -> List[SampleCatalogEntry]:
    entries: List[SampleCatalogEntry] = []
    for display_name, builder_fn in ALL_ARCHITECTURES:
        project = builder_fn()
        meta = SAMPLE_METADATA.get(display_name, {})
        entries.append(
            SampleCatalogEntry(
                id=project.project.id,
                name=project.project.name,
                category=meta.get("category", "General"),
                description=meta.get("description", display_name),
                highlight=meta.get("highlight", ""),
                tags=meta.get("tags", []),
                difficulty=meta.get("difficulty", "intermediate"),
                filename=f"{project.project.id}.nbp.json",
            )
        )
    return entries


def catalog_by_category() -> Dict[str, List[SampleCatalogEntry]]:
    grouped: Dict[str, List[SampleCatalogEntry]] = {}
    for entry in build_catalog():
        grouped.setdefault(entry.category, []).append(entry)
    return grouped


def get_builder_by_id(sample_id: str) -> Tuple[str, Callable[[], Any]] | None:
    for display_name, builder_fn in ALL_ARCHITECTURES:
        if builder_fn().project.id == sample_id:
            return display_name, builder_fn
    for display_name, builder_fn in ALL_ARCHITECTURES:
        project = builder_fn()
        if project.project.id == sample_id:
            return display_name, builder_fn
    return None
