# stephanie/services/workers/verify_claims_worker.py
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from stephanie.scoring.scorable import ScorableType

_logger = logging.getLogger(__name__)

class VerifyClaimsKGWorker:
    """
    Bus worker: verify a small set of claims against KG + embeddings + sidequest artifacts.
    Publishes sidequest.result.verify_claims.kg with per-claim verdicts and evidence.
    """
    REQUEST_TOPIC = "sidequest.request.verify_claims.kg"
    RESULT_TOPIC  = "sidequest.result.verify_claims.kg"

    def __init__(self, memory: Any, container: Any, logger: Optional[logging.Logger] = None):
        self.memory = memory
        self.container = container
        self.logger = logger or _logger
        self.bus = container.get_service("bus")

    # ---------- runtime ----------

    def start(self):
        self.bus.subscribe(self.REQUEST_TOPIC, self._on_request)
        self.logger.info("VerifyClaimsKGWorker subscribed", extra={"topic": self.REQUEST_TOPIC})

    # ---------- handlers ----------

    def _on_request(self, msg: Dict[str, Any]):
        t0 = time.time()
        try:
            req = msg if isinstance(msg, dict) else {}
            quest_id     = req.get("quest_id")
            parent_id    = req.get("parent_id")
            parent_type  = req.get("parent_type")
            claims       = req.get("args", {}).get("claims", [])[:5]
            if not claims:
                return

            results = []
            for claim in claims:
                verdict, support = self._verify_one(claim)
                results.append({
                    "claim": claim,
                    "verdict": verdict,            # "supported" | "weak" | "contradicted" | "unknown"
                    "support": support,            # list of {kind, id, sim}
                })

            payload = {
                "quest_id": quest_id,
                "status": "ok",
                "elapsed_s": round(time.time() - t0, 3),
                "parent_id": parent_id,
                "parent_type": parent_type,
                "artifacts": [],                 # no blob here; we return structured results
                "meta": {"results": results}
            }
            self.bus.publish(self.RESULT_TOPIC, payload)
        except Exception as e:
            self.logger.warning(f"VerifyClaimsKGWorker failed: {e}", exc_info=True)

    # ---------- verification ----------

    def _verify_one(self, claim: str) -> (str, List[Dict[str, Any]]):
        support: List[Dict[str, Any]] = []
        score = 0.0

        # 1) KG entity/name overlap
        try:
            kg = self.container.get("knowledge_graph")
            ents = getattr(kg, "find_entities_by_text", None)
            if ents:
                hits = ents(claim) or []
                for h in hits[:3]:
                    support.append({"kind": "kg", "id": h.get("id"), "name": h.get("name"), "sim": 0.65})
                    score = max(score, 0.65)
        except Exception:
            pass

        # 2) Embedding retrieval: conversation turns
        try:
            emb = getattr(self.memory, "embedding", None)
            if emb:
                cands = emb.search_related_scorables(
                    claim, ScorableType.CONVERSATION_TURN, include_ner=True, top_k=8
                ) or []
                for c in cands[:3]:
                    sid = c.get("id") or c.get("scorable_id")
                    sim = float(c.get("score") or 0.0)
                    support.append({"kind": "chat", "id": sid, "sim": round(sim,3)})
                    score = max(score, sim)
        except Exception:
            pass

        # 3) Heuristic contradiction flag (very light)
        contradicted = bool(re.search(r"\bnot\s+(true|supported|significant)|fails?\b", claim.lower()))
        if contradicted and score < 0.5:
            return "contradicted", support

        if score >= 0.75:
            return "supported", support
        if 0.55 <= score < 0.75:
            return "weak", support
        return "unknown", support
