# co_ai/registry/registry.py

_registry = {}


def register(key: str, instance):
    """
    Registers a shared component (tracker, controller, agent, etc.) under a key.
    Raises an error if the key already exists to avoid silent overwrites.
    """
    if key in _registry:
        raise KeyError(f"Key '{key}' is already registered.")
    _registry[key] = instance


def get(key: str):
    """
    Retrieves a registered component by key.
    Raises an error if the key is not found.
    """
    if key not in _registry:
        raise KeyError(f"Key '{key}' not found in registry.")
    return _registry[key]


def has(key: str) -> bool:
    """
    Returns True if a key is registered.
    """
    return key in _registry


def clear():
    """
    Clears the registry — useful for tests or controlled resets.
    """
    _registry.clear()


def all_keys() -> list:
    """
    Returns a list of all registered keys.
    """
    return list(_registry.keys())
