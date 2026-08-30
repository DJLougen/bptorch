"""Node registry implementation for discovering and managing node definitions."""

from typing import Any, Dict, List, Optional

from neural_blueprint.registry.base import NodeDefinition


class NodeRegistry:
    """Registry maintaining authoritative definitions of all supported nodes."""

    def __init__(self):
        self._definitions: Dict[str, NodeDefinition] = {}

    def register(self, definition: NodeDefinition) -> None:
        """Registers a node definition by its unique type_id."""
        if not definition.type_id:
            raise ValueError("NodeDefinition must have a non-empty type_id")
        self._definitions[definition.type_id] = definition

    def get(self, type_id: str) -> Optional[NodeDefinition]:
        """Looks up a definition by type_id (e.g. 'builtin.linear@1')."""
        return self._definitions.get(type_id)

    def require(self, type_id: str) -> NodeDefinition:
        """Looks up a definition by type_id or raises KeyError."""
        if type_id not in self._definitions:
            raise KeyError(f"Node definition '{type_id}' is not registered")
        return self._definitions[type_id]

    def list_all(self) -> List[NodeDefinition]:
        """Returns all registered node definitions."""
        return list(self._definitions.values())

    def export_catalog(self) -> List[Dict[str, Any]]:
        """Exports the full node catalog formatted for the frontend client."""
        catalog = []
        for defn in self._definitions.values():
            default_inputs = [p.model_dump() for p in defn.input_ports({})]
            default_outputs = [p.model_dump() for p in defn.output_ports({})]

            catalog.append(
                {
                    "type_id": defn.type_id,
                    "version": defn.version,
                    "display_name": defn.display_name,
                    "category": defn.category,
                    "description": defn.description,
                    "icon": defn.icon,
                    "is_composite": defn.is_composite,
                    "property_schema": defn.property_schema(),
                    "default_inputs": default_inputs,
                    "default_outputs": default_outputs,
                }
            )
        return catalog


# Global singleton registry instance
global_registry = NodeRegistry()


def register_node(cls):
    """Class decorator to register a NodeDefinition in the global registry."""
    instance = cls()
    global_registry.register(instance)
    return cls
