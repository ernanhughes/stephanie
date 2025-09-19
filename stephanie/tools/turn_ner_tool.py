# stephanie/tools/turn_ner_tool.py
from __future__ import annotations
from typing import Dict, List, Any

import logging
_logger = logging.getLogger(__name__)

def _normalize_ents(raw: List[dict], role: str) -> List[dict]:
    """
    Normalize to: {'text','type','start','end','score?','role'}.
    Accepts dicts with keys produced by EntityDetector.detect_entities.
    """
    out: List[dict] = []
    for e in raw or []:
        if not isinstance(e, dict):
            continue
        txt = e.get("text") or ""
        if not txt:
            continue
        out.append({
            "text": txt,
            "type": e.get("type") or e.get("label") or "ENT",
            "start": int(e.get("start", -1)),
            "end": int(e.get("end", -1)),
            "score": float(e.get("score", 0.0)) if e.get("score") is not None else None,
            "role": role,
        })
    return out

def annotate_conversation_ner(
    memory,
    conversation_id: int,
    *,
    kg: Any,                      # REQUIRED: pass KnowledgeGraphService from agent
    publish_to_kg: bool = True,   # optional bus publish
) -> Dict[str, int]:
    """
    Runs KG EntityDetector.detect_entities() on each turn (user+assistant),
    writes results to chat_turns.ner via memory.chats.set_turn_ner(...).
    Uses get_turn_texts_for_conversation() to avoid detached loads.
    """

    # Grab the detector from the KG service
    detector = kg._entity_detector
    turns = memory.chats.get_turn_texts_for_conversation(conversation_id)
    seen = updated = 0

    for t in turns:
        seen += 1
        user_txt = (t["user_text"] or "").strip()
        asst_txt = (t["assistant_text"] or "").strip()

        # Run detector separately on each side so offsets are local to the side
        u_ents = _normalize_ents(detector.detect_entities(user_txt), role="user") if user_txt else []
        a_ents = _normalize_ents(detector.detect_entities(asst_txt), role="assistant") if asst_txt else []
        ents = u_ents + a_ents

        # Persist
        memory.chats.set_turn_ner(t["id"], ents)
        updated += 1

        # (Optional) publish to KG indexer via bus
        if publish_to_kg:
            try:
                domains_payload = []
                memory.bus.publish("knowledge_graph.index_request", {
                    "scorable_id": str(t["id"]),
                    "scorable_type": "conversation_turn",
                    "text": f"USER: {user_txt}\nASSISTANT: {asst_txt}".strip(),
                    "entities": ents,          # ✅ normalized NER
                    "domains": domains_payload # [] is fine
                })
            except Exception as ex:
                _logger.error(f"KGIndexPublishError turn_id: {t['id']}, error: {str(ex)}")

    return {"seen": seen, "updated": updated}
