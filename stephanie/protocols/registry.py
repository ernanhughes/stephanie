from typing import Callable, Optional, Union

from stephanie.protocols.base import Protocol, ProtocolRecord


class ProtocolRegistry:
    def __init__(self):
        self._registry: dict[str, ProtocolRecord] = {}

    def register(
        self,
        name: str,
        protocol: Protocol,
        description: str = "",
        input_format: Optional[dict[str, any]] = None,
        output_format: Optional[dict[str, any]] = None,
        failure_modes: Optional[list[str]] = None,
        depends_on: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        capability: Optional[str] = None,
        preferred_for: Optional[list[str]] = None,
        avoid_for: Optional[list[str]] = None
    ):
        """
        Register a new protocol with metadata.
        """
        self._registry[name] = {
            "name": name,
            "description": description or f"Protocol {name}",
            "protocol": protocol,
            "metadata": {
                "input_format": input_format or {},
                "output_format": output_format or {},
                "failure_modes": failure_modes or [],
                "depends_on": depends_on or [],
                "tags": tags or [],
                "capability": capability,
                "preferred_for": preferred_for or [],
                "avoid_for": avoid_for or []
            }
        }

    def get(self, name: str) -> Optional[ProtocolRecord]:
        """Get a protocol by name."""
        return self._registry.get(name)

    def get_protocol(self, name: str) -> Optional[Protocol]:
        """Get just the callable protocol object."""
        rec = self._registry.get(name)
        return rec["protocol"] if rec else None

    def list_all(self) -> list[dict[str, any]]:
        """Return a list of all registered protocols (without the objects)."""
        return [
            {
                "name": name,
                "description": rec["description"],
                "metadata": rec["metadata"]
            }
            for name, rec in self._registry.items()
        ]

    def find_by_tag(self, tag: str) -> list[ProtocolRecord]:
        """Find all protocols with a given tag."""
        return [rec for rec in self._registry.values() if tag in rec["metadata"]["tags"]]

    def find_by_tags(self, tags: list[str]) -> list[ProtocolRecord]:
        """Find all protocols with any of the given tags."""
        return [rec for rec in self._registry.values() if set(tags) & set(rec["metadata"]["tags"])]

    def find_by_capability(self, capability: str) -> list[ProtocolRecord]:
        """Find protocols matching a specific capability."""
        return [rec for rec in self._registry.values() if rec["metadata"].get("capability") == capability]

    def find_by_input_output(self, input_format=None, output_format=None) -> list[ProtocolRecord]:
        """
        Find protocols that match both input and output format.
        """
        results = []
        for rec in self._registry.values():
            meta = rec["metadata"]
            if input_format and not self._matches_format(input_format, meta["input_format"]):
                continue
            if output_format and not self._matches_format(output_format, meta["output_format"]):
                continue
            results.append(rec)
        return results

    def _matches_format(self, expected: dict, actual: dict) -> bool:
        """
        Check if actual format matches expected format.
        This could be enhanced with schema validation (e.g., Pydantic).
        """
        for k, v in expected.items():
            if k not in actual or actual[k] != v:
                return False
        return True

    def update_metadata(self, name: str, updates: dict[str, any]) -> bool:
        """Update metadata of an existing protocol."""
        if name not in self._registry:
            return False
        self._registry[name]["metadata"].update(updates)
        return True

    def delete(self, name: str) -> bool:
        """Remove a protocol from the registry."""
        if name in self._registry:
            del self._registry[name]
            return True
        return False

    def exists(self, name: str) -> bool:
        """Check if a protocol is registered."""
        return name in self._registry

    def clear(self):
        """Clear the entire registry (for testing)."""
        self._registry.clear()

    def size(self) -> int:
        """Return number of registered protocols."""
        return len(self._registry)