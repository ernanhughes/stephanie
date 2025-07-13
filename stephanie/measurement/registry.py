# stephanie/measurement/registry.py
from typing import Callable, Dict


class MeasurementStrategy:
    def __init__(self):
        self._strategies = {}

    def register(self, entity_type: str, metric_name: str, func: Callable):
        """Register a measurement function"""
        key = f"{entity_type}.{metric_name}"
        self._strategies[key] = func

    def get_strategies_for_entity(self, entity_type: str) -> Dict[str, Callable]:
        """Get all strategies for an entity type"""
        return {
            key.split(".")[1]: func
            for key, func in self._strategies.items()
            if key.startswith(f"{entity_type}.")
        }


measurement_registry = MeasurementStrategy()
