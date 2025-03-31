# ai_co_scientist/logs/ranking_log.py
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

@dataclass
class EloRankingLog:
    hypothesis: str
    score: int
    run_id: str
    created_at: str = datetime.now(timezone.utc).isoformat()

    def to_dict(self):
        return asdict(self)
