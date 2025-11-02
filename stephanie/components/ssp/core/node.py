# stephanie/components/ssp/core/node.py
"""
Node Data Structure for SSP Tree Search

This module defines the canonical Node class used throughout the SSP system.
The Node represents a state in the search tree used by the ATSSolver and other
components that perform tree-based search.

The structure is designed to be compatible with TreeEventEmitter for telemetry
and progress tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Node:
    """
    Represents a node in the search tree used by SSP components.
    
    This is the canonical definition used throughout the SSP system.
    """
    # Basic identification
    id: str
    parent_id: Optional[str]
    root_id: str
    
    # Tree structure
    depth: int
    sibling_index: int
    node_type: str  # 'root' | 'rewrite' | other node types
    
    # Content
    query: str
    score: float
    context: str
    
    # Optional extended properties
    task_description: Optional[str] = None
    summary: Optional[str] = None
    metric: Optional[float] = None
    is_buggy: Optional[bool] = None
    meta: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.meta is None:
            self.meta = {}