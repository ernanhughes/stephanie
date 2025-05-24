from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

@dataclass
class PipelineRun:
    id: Optional[int] = None
    run_id: str = ""
    goal_id: int = 0
    pipeline: str = ""
    strategy: Optional[str] = None
    model_name: Optional[str] = None
    run_config: Optional[Dict] = field(default_factory=dict)
    lookahead_context: Optional[Dict] = field(default_factory=dict)
    symbolic_suggestion: Optional[Dict] = field(default_factory=dict)
    metadata: Optional[Dict] = field(default_factory=dict)
    created_at: Optional[datetime] = None
