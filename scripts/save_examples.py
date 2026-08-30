"""Generate and save all 25 architecture sample project fixtures."""

import json
from pathlib import Path

from neural_blueprint.ir.serialization import save_project_file
from neural_blueprint.templates.architectures import ALL_ARCHITECTURES
from neural_blueprint.templates.samples_catalog import build_catalog
from neural_blueprint.templates.linear_mlp import create_linear_mlp_template
from neural_blueprint.templates.nanogpt import create_nanogpt_template

ROOT_DIR = Path(__file__).parent.parent
EXAMPLES_DIR = ROOT_DIR / "examples"
WEB_PUBLIC_DIR = ROOT_DIR / "web" / "public" / "examples"
WEB_DIST_DIR = ROOT_DIR / "web" / "dist" / "examples"


def _target_dirs(sample_id: str) -> list[Path]:
    return [
        EXAMPLES_DIR / sample_id,
        WEB_PUBLIC_DIR / sample_id,
        WEB_DIST_DIR / sample_id,
    ]


def save_examples() -> None:
    """Export all 25 architecture samples plus a manifest to public/examples."""
    catalog = build_catalog()
    manifest_samples = []

    for display_name, builder_fn in ALL_ARCHITECTURES:
        project = builder_fn()
        sample_id = project.project.id
        filename = f"{sample_id}.nbp.json"

        for d in _target_dirs(sample_id):
            d.mkdir(parents=True, exist_ok=True)
            save_project_file(project, d / filename)

        entry = next((e for e in catalog if e.id == sample_id), None)
        manifest_samples.append(
            {
                "id": sample_id,
                "name": project.project.name,
                "category": entry.category if entry else "General",
                "description": entry.description if entry else display_name,
                "highlight": entry.highlight if entry else "",
                "tags": entry.tags if entry else [],
                "difficulty": entry.difficulty if entry else "intermediate",
                "path": f"/examples/{sample_id}/{filename}",
                "node_count": sum(len(g.nodes) for g in project.model.graphs.values()),
                "graph_count": len(project.model.graphs),
            }
        )
        print(f"Saved {display_name} -> {sample_id}/{filename}")

    # Legacy paths for backward compatibility
    legacy_dirs = [
        (EXAMPLES_DIR, WEB_PUBLIC_DIR, WEB_DIST_DIR),
    ]
    mlp = create_linear_mlp_template(in_features=64, hidden_features=256)
    gpt = create_nanogpt_template(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    for base, pub, dist in legacy_dirs:
        for d, name, proj in [
            (base / "linear-mlp", "linear_mlp.nbp.json", mlp),
            (pub / "linear-mlp", "linear_mlp.nbp.json", mlp),
            (dist / "linear-mlp", "linear_mlp.nbp.json", mlp),
            (base / "nanogpt", "nanogpt_tiny.nbp.json", gpt),
            (pub / "nanogpt", "nanogpt_tiny.nbp.json", gpt),
            (dist / "nanogpt", "nanogpt_tiny.nbp.json", gpt),
        ]:
            d.mkdir(parents=True, exist_ok=True)
            save_project_file(proj, d / name)

    manifest = {
        "version": 1,
        "title": "bpTorch Sample Gallery",
        "description": "25 trainable architecture demonstrations showcasing the diversity of visual blueprint programming.",
        "count": len(manifest_samples),
        "samples": manifest_samples,
    }

    for out_dir in (WEB_PUBLIC_DIR, WEB_DIST_DIR, EXAMPLES_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "samples.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote manifest ({len(manifest_samples)} samples) -> {manifest_path}")


if __name__ == "__main__":
    save_examples()
