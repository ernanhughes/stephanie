# stephanie/registry/registry.py
"""
Component registry for the Supervisor system.

This module provides a simple registry pattern for managing components
that can be shared across different parts of the system.
"""

_component_registry = {}

def register(name: str, component: object) -> None:
    """
    Register a component with the given name.
    
    Args:
        name: The name to register the component under
        component: The component to register
        
    Raises:
        ValueError: If a component with the same name is already registered
    
    Example:
        >>> register("state_tracker", StateTracker(cfg, memory, logger))
    """
    if name in _component_registry:
        raise ValueError(f"Component '{name}' is already registered")
    
    _component_registry[name] = component

def get_registered_component(name: str) -> object:
    """
    Get a registered component by name.
    
    Args:
        name: The name of the component to retrieve
        
    Returns:
        The registered component
        
    Raises:
        ValueError: If the component is not registered
        
    Example:
        >>> state_tracker = get_registered_component("state_tracker")
    """
    if name not in _component_registry:
        available = list(_component_registry.keys())
        raise ValueError(
            f"Component '{name}' is not registered. "
            f"Available components: {available}"
        )
    return _component_registry[name]

def has_registered_component(name: str) -> bool:
    """
    Check if a component is registered.
    
    Args:
        name: The name of the component to check
        
    Returns:
        True if the component is registered, False otherwise
    
    Example:
        >>> if has_registered_component("plan_trace_monitor"):
        ...     monitor = get_registered_component("plan_trace_monitor")
    """
    return name in _component_registry

def clear_registry() -> None:
    """
    Clear all registered components (primarily for testing).
    
    Example:
        >>> clear_registry()
    """
    _component_registry.clear()

def get_all_components() -> dict:
    """ 
    Get a copy of all registered components.
    
    Returns:
        A dictionary of all registered components
    
    Example:
        >>> components = get_all_components()
        >>> for name, component in components.items():
        ...     print(f"{name}: {type(component)}")
    """
    return _component_registry.copy()