# stephanie/measurement/decorators.py
from functools import wraps

from stephanie.measurement.registry import measurement_registry


def measure(entity_type: str, metric_name: str):
    """Decorator to register measurement functions"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        measurement_registry.register(entity_type, metric_name, wrapper)
        return wrapper

    return decorator
