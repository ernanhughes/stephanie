# stephanie/tools/turn_ner_tool.py
from __future__ import annotations

from typing import Any, Callable, List, Optional, Dict
import logging
import asyncio

_logger = logging.getLogger(__name__)

# Keep references to background tasks to avoid premature garbage collection
_background_tasks: set[asyncio.Task] = set()


def build_ner_backend(
    kg: Any,
):
    detector = kg._entity_detector

    def _kg_ner(text: str) -> List[dict]:
        try:
            results = detector.detect_entities(text or "") or []
            out: List[dict] = []
            for e in results:
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
            _logger.error("KGNERBackendError: %s", str(ex))
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
    """
    Extract NER for each turn (idempotent if only_missing=True), persist on ChatTurn,
    and optionally publish an index request to the knowledge graph.
    """
    # Build backend (will raise clear error if KG not ready)
    ner = build_ner_backend(kg)

    # Pull pre-fetched texts (safe outside session)
    turns = memory.chats.get_turn_texts_for_conversation(
        conversation_id,
        only_missing=("ner" if only_missing else None),
    )

    seen = updated = 0

    for t in turns:
        seen += 1
        user_txt = t.get("user_text") or ""
        asst_txt = t.get("assistant_text") or ""
        joined = f"USER: {user_txt}\nASSISTANT: {asst_txt}".strip()

        if not joined:
            # count progress even if there was nothing to do for this turn
            if progress_cb:
                progress_cb(1)
            continue

        ents = ner(joined) or []

        # Split by role (offset-based) for downstream clarity
        split: List[dict] = []
        role_cut = len(f"USER: {user_txt}\n")
        for e in ents:
            start = int(e.get("start", -1))
            role = "assistant" if start >= role_cut else "user"
            split.append({**e, "role": role})

        # Persist on the turn
        memory.chats.set_turn_ner(t["id"], split)
        updated += 1

        # Always bump progress ONCE per processed turn
        if progress_cb:
            progress_cb(1)

        # Optionally publish to KG (fire-and-forget), only if we found entities
        if publish_to_kg and split:
            payload = {
                "scorable_id": str(t["id"]),
                "scorable_type": "conversation_turn",
                "text": joined,
                "entities": split,
                # get_turn_texts_for_conversation doesn't include domains; pass [] for now
                "domains": [],
            }
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(memory.bus.publish("knowledge_graph.index_request", payload))
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
            except RuntimeError:
                # No running loop (e.g., sync context). Don't crash; just log.
                _logger.warning(
                    "No running event loop; skipping KG publish for turn_id=%s", t["id"]
                )
            except Exception as ex:
                _logger.error("KGIndexPublishError turn_id=%s error=%s", t["id"], str(ex))

    return {"seen": seen, "updated": updated}
