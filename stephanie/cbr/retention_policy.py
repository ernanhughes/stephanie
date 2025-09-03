# stephanie/cbr/retention_policy.py
from typing import Dict, List, Optional

from stephanie.constants import GOAL, PIPELINE_RUN_ID


class DefaultRetentionPolicy:
    def __init__(self, cfg, memory, logger, casebook_scope_mgr):
        self.cfg, self.memory, self.logger = cfg, memory, logger
        self.scope_mgr = casebook_scope_mgr
        self.tag = cfg.get("casebook_tag", "default")

    def retain(self, ctx: Dict, ranked: List[Dict], mars: Dict, scores: Dict) -> Optional[int]:
        goal = ctx[GOAL]
        casebook_id = self.scope_mgr.home_casebook_id(ctx, ctx.get("agent_name") or "UnknownAgent", self.tag)

        scorables_payload = []
        for idx, r in enumerate(ranked):
            scorables_payload.append({
                "id": r.get("id"),
                "type": "hypothesis",
                "role": "output",
                "rank": (idx + 1),
                "meta": {"text": r.get("text",""), "mars_confidence": r.get("mars_confidence")},
            })

        try:
            case = self.memory.casebooks.add_case(
                casebook_id=casebook_id,
                goal_id=goal["id"],
                goal_text=goal["goal_text"],
                agent_name=ctx.get("agent_name") or "UnknownAgent",
                mars_summary=mars,
                scores=scores,
                metadata={
                    "pipeline_run_id": ctx.get(PIPELINE_RUN_ID),
                    "casebook_tag": self.tag,
                    "hypothesis_count": len(ranked),
                },
                scorables=scorables_payload,
            )
            return case.id
        except Exception:
            return None
