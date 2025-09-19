# stephanie/tools/turn_ner_tool.py
from __future__ import annotations

from typing import Any, Callable, List, Optional, Dict
import logging

_logger = logging.getLogger(__name__)

# Keep references to background tasks to avoid premature garbage collection
_background_tasks = set()


def build_ner_backend(
    kg: Any,
):
    detector = kg._entity_detector

    def _kg_ner(text: str) -> List[dict]:
        try:
            ents = detector.detect_entities(text or "") or []
            # normalize keys
            out = []
            for e in ents:
                if isinstance(e, dict) and e.get("text"):
                    out.append(
                        {
                            "text": e["text"],
                            "type": e.get("type") or e.get("label") or "ENT",
                            "start": int(e.get("start", -1)),
                            "end": int(e.get("end", -1)),
                            "score": float(e.get("score", 0.0)),
                            "source_text": e.get("source_text", e["text"]),
                        }
                    )
            return out
        except Exception as ex:
            _logger.error(f"KGNERBackendError error: {str(ex)}")
            return []

    return _kg_ner


def annotate_conversation_ner(
    memory,
    conversation_id: int,
    kg: Any,
    *,
    publish_to_kg: bool = True,
    only_missing: bool = True,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Dict[str, int]:
    logger = getattr(memory, "logger", None)
    ner = build_ner_backend(kg)

    turns = memory.chats.get_turn_texts_for_conversation(
        conversation_id, only_missing=("ner" if only_missing else None)
    )

    seen = updated = 0
    for t in turns:
        seen += 1
        user_txt = t["user_text"] or ""
        asst_txt = t["assistant_text"] or ""
        joined = f"USER: {user_txt}\nASSISTANT: {asst_txt}".strip()
        if not joined:
            progress_cb and progress_cb(1)
            continue

        ents = ner(joined) or []

        # Split by role boundary for display/use later
        split = []
        role_cut = len(f"USER: {user_txt}\n")
        for e in ents:
            role = "assistant" if (e.get("start", -1) >= role_cut) else "user"
            split.append({**e, "role": role})

        # Persist on the turn
        memory.chats.set_turn_ner(t["id"], split)
        updated += 1
        if publish_to_kg and ents:
            progress_cb and progress_cb(1)
            try:
                import asyncio
                payload = {
                    "scorable_id": str(t["id"]),
                    "scorable_type": "conversation_turn",
                    "text": joined,
                    "entities": split,
                    "domains": t.get(
                        "domains", []
                    ),  # if present in the dict
                }
                # schedule fire-and-forget so we don’t block; keep a reference to avoid GC
                task = asyncio.create_task(
                    memory.bus.publish("knowledge_graph.index_request", payload)
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
            except Exception as ex:
                _logger.error(
                    f"KGIndexPublishError turn_id: {t['id']}, error: {str(ex)}"
                )

    return {"seen": seen, "updated": updated}
