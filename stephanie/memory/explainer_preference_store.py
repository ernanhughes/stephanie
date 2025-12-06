from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Iterable, List, Optional
from datetime import datetime
import json
import logging

log = logging.getLogger(__name__)


@dataclass
class ExplainerPreferenceRecord:
    """
    One comparison between two explainer drafts for a single paper.
    """
    paper_id: str
    task_id: str
    judge_type: str          # "human" | "llm"
    winner: str              # "v1" | "v2" | "TIE" | "ERROR"
    v1_draft: str
    v2_draft: str
    v1_scores: Dict[str, float]
    v2_scores: Dict[str, float]
    improvement: str
    created_at: str

    @classmethod
    def from_raw(
        cls,
        *,
        paper_id: str,
        task_id: str,
        judge_type: str,
        v1_draft: str,
        v2_draft: str,
        judge_result: Dict[str, Any],
    ) -> "ExplainerPreferenceRecord":
        now = datetime.utcnow().isoformat()
        return cls(
            paper_id=paper_id,
            task_id=task_id,
            judge_type=judge_type,
            winner=judge_result.get("winner", "ERROR"),
            v1_draft=v1_draft,
            v2_draft=v2_draft,
            v1_scores=judge_result.get("v1_scores", {}),
            v2_scores=judge_result.get("v2_scores", {}),
            improvement=judge_result.get("improvement", ""),
            created_at=now,
        )


class ExplainerPreferenceStore:
    """
    Minimal JSONL-based store for explainer preferences.
    Simple, robust, no DB schema needed for v0.
    """

    def __init__(self, path: str = "results/explainer_prefs.jsonl"):
        self.path = path

    def log_record(self, rec: ExplainerPreferenceRecord) -> None:
        try:
            line = json.dumps(asdict(rec), ensure_ascii=False)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            log.warning(f"ExplainerPreferenceStore: failed to log record: {e}")

    def iter_records(self, limit: Optional[int] = None) -> Iterable[ExplainerPreferenceRecord]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if limit is not None and i >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    yield ExplainerPreferenceRecord(**raw)
        except FileNotFoundError:
            return
