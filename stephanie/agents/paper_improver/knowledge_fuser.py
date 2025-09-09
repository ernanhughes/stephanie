from __future__ import annotations
from typing import Dict, Any, List, Tuple
import hashlib, time

from .vpm_controller import VPMController, default_controller
from stephanie.knowledge.chat_knowledge import ChatKnowledgeBuilder, TransientKnowledge, TWENTY_DOMAINS

class KnowledgeFuser:
    """
    Fuses transient knowledge (chat + paper) into a content plan:
      - aligns domains
      - picks overlapping keyphrases (CBR)
      - builds units: claim + evidence + claim_id
      - merges entities (ABBR/REQUIRED) with de-dupe
    """
    def __init__(self):
        self.ctrl: VPMController = default_controller()
        self.builder = ChatKnowledgeBuilder()

    def fuse(self, *, paper_text: str, chat_messages: List[Dict[str,Any]], section_name: str) -> Dict[str, Any]:
        k = self.builder.build(chat_messages=chat_messages, paper_text=paper_text)
        chat, paper = k["chat"], k["paper"]

        # Domain blending (element-wise max favors confident overlap)
        domains = {d: max(chat.domains.get(d,0.0), paper.domains.get(d,0.0)) for d in TWENTY_DOMAINS}
        # Keep top 5 domains as section focus
        top_domains = sorted(domains.items(), key=lambda kv: kv[1], reverse=True)[:5]

        # Phrase overlap = “knowledge intersection”
        overlap = [p for p in paper.phrases if p.lower() in {c.lower():None for c in chat.phrases}]
        # Seed claims from overlap; backfill from paper anchors if needed
        seed = overlap[:5] or [a["span"] for a in paper.anchors[:5]]

        units = []
        for i, phrase in enumerate(seed):
            units.append({
                "claim_id": f"C{i+1}",
                "claim": f"{phrase}.",
                "evidence": "See paper",
            })

        # Merge entities with chat taking precedence for ABBR “habit”
        abbr = dict(paper.entities.get("ABBR", {}))
        abbr.update(chat.entities.get("ABBR", {}))
        required = list(dict.fromkeys(
            (paper.entities.get("REQUIRED", []) or []) +
            (chat.entities.get("REQUIRED", []) or [])
        ))[:12]

        plan = {
            "section_title": section_name,
            "units": units,
            "entities": {"ABBR": abbr, "REQUIRED": required},
            "paper_text": paper_text,
            "domains": [{"domain": d, "score": float(s)} for d, s in top_domains],
            "meta": {
                "knowledge_hash": hashlib.sha256((" ".join(seed)).encode()).hexdigest()[:10],
                "timestamp": time.time(),
                "sources": {"chat_phr_count": len(chat.phrases), "paper_phr_count": len(paper.phrases)}
            }
        }
        return plan
