# components/elm/reflection/reflection_trace.py

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ReflectionTrace:
    original_trace_id: str
    failed_axes: List[str]
    correction_instructions: Dict[str, str]
    focus_axes: List[str]
    confidence: float
