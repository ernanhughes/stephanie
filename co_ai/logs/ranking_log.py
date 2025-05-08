# co_ai/logs/ranking_log.py
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class EloRankingLog:
    hypothesis: str
    score: int
    run_id: str
    created_at: str = datetime.now(timezone.utc).isoformat()

    def to_dict(self):
        return asdict(self)
