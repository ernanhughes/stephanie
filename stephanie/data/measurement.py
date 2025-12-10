from dataclasses import dataclass
from datetime import datetime

@dataclass
class Measurement:
    id: str               # universal id
    target_type: str      # "document", "plan_trace", "code_file", ...
    target_id: str
    dimension: str        # "helpfulness", "correctness", "novelty", ...
    source: str           # "LLM", "human", "rule", "SICQL", ...
    value: float          # the score
    extras: dict          # raw_prompt, response, trace refs, etc.
    created_at: datetime
