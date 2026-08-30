"""Templates package."""

from neural_blueprint.templates.linear_mlp import create_linear_mlp_template
from neural_blueprint.templates.nanogpt import create_nanogpt_template

__all__ = [
    "create_linear_mlp_template",
    "create_nanogpt_template",
]
