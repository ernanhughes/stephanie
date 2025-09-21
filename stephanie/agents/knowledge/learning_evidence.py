# stephanie/agents/meta/learning_evidence_agent.py
from __future__ import annotations
import os, json, glob
from typing import Dict, Any, List
from stephanie.agents.base_agent import BaseAgent
from stephanie.models.learning_evidence import LearningEvidenceORM

class LearningEvidenceAgent(BaseAgent):
    """
    Meta-agent that proves 'learning from learning':
      • collects Track A/B/C metrics
      • parses strategy evolution events
      • writes LearningEvidenceORM + CSV/MD
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.report_dir = str(cfg.get("report_dir", "reports/learning_evidence"))
        os.makedirs(self.report_dir, exist_ok=True)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        docs = context.get("documents", [])
        evidence_rows: List[Dict[str, Any]] = []

        for doc in docs:
            doc_id = str(doc.get("id") or doc.get("paper_id"))

            # 1) Collect Track A/B/C scores
            summary_v2 = (context.get("summary_v2") or {}).get(doc_id, {})
            metrics_c = summary_v2.get("metrics", {})
            guardrails = summary_v2.get("guardrail_details", {})

            # 2) Extract strategy evolution logs (already persisted in context/logs)
            strategy_events = context.get("strategy_events", {}).get(doc_id, [])

            for ev in strategy_events:
                row = {
                    "doc_id": doc_id,
                    "strategy_version": ev.get("new_version"),
                    "old_threshold": ev.get("old_thr"),
                    "new_threshold": ev.get("new_thr"),
                    "old_weights": ev.get("old_weights"),
                    "new_weights": ev.get("new_weights"),
                    "avg_gain": ev.get("avg_gain"),
                    "metrics": metrics_c,
                    "guardrails": guardrails,
                }
                evidence_rows.append(row)

                # persist to DB
                self.memory.session.add(LearningEvidenceORM(
                    doc_id=row["doc_id"],
                    strategy_version=row["strategy_version"],
                    old_threshold=row["old_threshold"],
                    new_threshold=row["new_threshold"],
                    old_weights=row["old_weights"],
                    new_weights=row["new_weights"],
                    avg_gain=row["avg_gain"],
                    meta={"metrics": metrics_c, "guardrails": guardrails}
                ))

        self.memory.session.commit()

        # 3) Export CSV for dashboards
        csv_path = os.path.join(self.report_dir, "learning_evidence.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("doc_id,strategy_version,old_thr,new_thr,avg_gain\n")
            for r in evidence_rows:
                f.write(f"{r['doc_id']},{r['strategy_version']},{r['old_threshold']},{r['new_threshold']},{r['avg_gain']}\n")

        # 4) Push evidence summary to context for SIS dashboards
        context.setdefault("learning_evidence", [])
        context["learning_evidence"].extend(evidence_rows)

        return context
