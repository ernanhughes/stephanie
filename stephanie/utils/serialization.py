# stephanie/utils/serialization.py
"""
Utilities for handling serialization of complex objects, especially OmegaConf DictConfig objects.

This module solves the critical problem of "Object of type DictConfig is not JSON serializable"
that we encountered when trying to store PlanTrace objects in the database.
"""

from typing import Any
from omegaconf import OmegaConf
import numpy as np
from stephanie.data.plan_trace import PlanTrace, ExecutionStep

def to_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable format, especially handling OmegaConf objects.
    
    Args:
        obj: Any object to convert
        
    Returns:
        JSON-serializable version of the object
        
    Example:
        >>> cfg = OmegaConf.create({"model": {"name": "llama", "layers": 24}})
        >>> serializable = to_serializable(cfg)
        >>> isinstance(serializable, dict)
        True
    """
    if obj is None:
        return None
    elif isinstance(obj, (int, float, bool, str)):
        return obj
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    elif OmegaConf.is_config(obj):
        # This is an OmegaConf object (DictConfig or ListConfig)
        return OmegaConf.to_container(obj, resolve=True, enum_to_str=True)
    elif hasattr(obj, "to_dict"):
        # Custom objects with to_dict method
        return to_serializable(obj.to_dict())
    elif hasattr(obj, "tolist"):
        # Objects with tolist method (like numpy arrays)
        return obj.tolist()
    else:
        try:
            # Try to convert to string as last resort
            return str(obj)
        except:
            return "non-serializable-object"
        
def default_serializer(obj):
    """Handle serialization of complex objects including NumPy types"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, (ExecutionStep, PlanTrace)):
        return obj.to_dict()
    elif hasattr(obj, '_get_node'):  # OmegaConf DictConfig
        return OmegaConf.to_container(obj, resolve=True, enum_to_str=True)
    raise TypeError(f"Type {type(obj)} not serializable")
