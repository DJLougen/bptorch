"""Base classes and interfaces for node definitions in Neural Blueprint Studio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from neural_blueprint.ir.models import PortDefinition, TensorSpec


@dataclass
class NodeValidationContext:
    model_config: Dict[str, Any] = field(default_factory=dict)
    parent_properties: Dict[str, Any] = field(default_factory=dict)
    graph_definitions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterSpec:
    trainable_count: int = 0
    frozen_count: int = 0
    parameter_shapes: Dict[str, List[int]] = field(default_factory=dict)
    has_bias: bool = False

    @property
    def total_count(self) -> int:
        return self.trainable_count + self.frozen_count


@dataclass
class RuntimeModuleSpec:
    module_type: str  # "nn_module" or "functional"
    factory: Optional[Callable[..., Any]] = None  # Function that returns nn.Module or callable
    kwargs: Dict[str, Any] = field(default_factory=dict)


class NodeDefinition(ABC):
    """Authoritative declaration of a node type's ports, properties, and runtime construction."""

    type_id: str
    version: int = 1
    display_name: str
    category: str
    description: str = ""
    icon: Optional[str] = None
    is_composite: bool = False

    @abstractmethod
    def property_schema(self) -> Dict[str, Any]:
        """Returns JSON schema for node properties, including UI hints and enum options."""
        pass

    @abstractmethod
    def input_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        """Returns dynamic input port definitions based on current properties."""
        pass

    @abstractmethod
    def output_ports(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[PortDefinition]:
        """Returns dynamic output port definitions based on current properties."""
        pass

    @abstractmethod
    def infer_shapes(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Dict[str, TensorSpec]:
        """Infers output tensor shapes from input tensor specs and node properties."""
        pass

    def validate(
        self,
        inputs: Dict[str, TensorSpec],
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> List[Any]:
        """Performs node-local validation checks and returns diagnostic objects."""
        return []

    def parameter_spec(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> ParameterSpec:
        """Calculates parameter counts and shapes for this node."""
        return ParameterSpec()

    def build_runtime(
        self,
        properties: Dict[str, Any],
        context: Optional[NodeValidationContext] = None,
    ) -> Optional[RuntimeModuleSpec]:
        """Constructs the PyTorch runtime specification (module or functional callable)."""
        return None
