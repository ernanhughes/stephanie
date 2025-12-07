# dataloaders/conversation_to_casebook.py
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

from stephanie.models.casebook import CaseBookORM, CaseORM, CaseScorableORM
from stephanie.utils.hash_utils import hash_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers (simple, stable)
# ---------------------------------------------------------------------------
def summarize_goal(bundle: list | dict) -> str:
    """Infers a high-level goal from the conversation history."""
    # Accept both list-of-turns or dict bundle
    turns: List[Dict[str, Any]] = []
    if isinstance(bundle, dict):
        if "turns" in bundle and isinstance(bundle["turns"], list):
            turns = bundle["turns"]
        elif "mapping" in bundle and isinstance(bundle["mapping"], dict):
            turns = _extract_turns_from_mapping(bundle["mapping"])
    elif isinstance(bundle, list):
        turns = bundle

    if not turns:
        return "Untitled Conversation"

    first_user_turn = next((t for t in turns if t.get("role") == "user" and t.get("text")), None)
    if first_user_turn:
        return first_user_turn["text"].split("\n")[0][:100] + "..."
    return "Untitled Conversation"


def make_name(bundle: list | dict) -> str:
    """Generates a unique name for the CaseBook."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    goal_slug = re.sub(r"[^a-zA-Z0-9]+", "_", summarize_goal(bundle).strip()).lower()
    return f"conv_{timestamp}_{goal_slug}"


def user_assistant_pairs(bundle: list[dict]) -> Iterable[Tuple[dict, dict]]:
    """Yields (user, assistant) pairs from a flat (role, text) turn list."""
    user_turn = None
    for turn in bundle:
        if turn.get("role") == "user":
            user_turn = turn
        elif turn.get("role") == "assistant" and user_turn:
            yield (user_turn, turn)
            user_turn = None


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def conversation_to_casebook(
    memory,
    bundle: dict,
    context: dict,
    tags: list[str] | None = None,
) -> tuple[CaseBookORM, dict]:
    """
    Convert a parsed conversation bundle into a CaseBook + Cases + Scorables.
    Returns (CaseBookORM, counts).

    Expected `memory` interface:
      - memory.casebooks.ensure_casebook(name, pipeline_run_id, description, tags) -> CaseBookORM
      - memory.casebooks.add_case(casebook_id, goal_id, goal_text, agent_name, prompt_text, scorables, response_texts)
      - memory.goals.get_or_create(dict(goal_text=..., description=...)) -> GoalORM-like (with .to_dict())

    Bundle shapes supported:
      - {"turns": [{"role": "user"/"assistant", "text": "...", "ts": ...}, ...]}
      - {"mapping": { id: { "message": {...}, "parent": "...", "children": [...] }, ...}, "title": "..."}
    """
    counts = {"cases_created": 0, "scorables_created": 0}

    title = bundle.get("title", "Untitled Conversation")
    conversation_id = bundle.get("id") or bundle.get("conversation_id")

    cb = memory.casebooks.ensure_casebook(
        name=f"chat_{conversation_id or title[:200]}",
        pipeline_run_id=context.get("pipeline_run_id"),
        description=f"Imported chat conversation: {title}",
        tags=tags or [],
    )
    logger.info(f"[CaseBook] Using: {cb.name} (id={cb.id})")

    # --- 1) Extract turns (normalized as [{'role': 'user'|'assistant', 'text': str, 'ts': float|None}, ...]) ---
    mapping = bundle.get("mapping", {})
    if mapping:
        turns = _extract_turns_from_mapping(mapping)
    else:
        turns = bundle.get("turns", []) or []
        # normalize super-light turns; tolerate {'content': '...'}
        for t in turns:
            if "text" not in t and "content" in t:
                t["text"] = t.get("content")

    # Filter to text-bearing user/assistant only
    turns = [
        {"role": (t.get("role") or "").lower(), "text": (t.get("text") or "").strip(), "ts": t.get("ts")}
        for t in turns
        if (t.get("role") in ("user", "assistant")) and (t.get("text") or "").strip()
    ]

    if not turns:
        logger.info(f"⚠️ Skipping {title}, no turns found")
        return cb, counts

    # --- 2) Goal setup ---
    goal_obj = memory.goals.get_or_create({"goal_text": f"{cb.name}", "description": f"{cb.description}"})
    goal = goal_obj.to_dict() if hasattr(goal_obj, "to_dict") else {"id": getattr(goal_obj, "id", None), "goal_text": cb.name}

    # --- 3) Uniqueness guard: load existing scorable hashes for this casebook ---
    existing_hashes = {
        sc.meta.get("turn_hash")
        for sc in memory.session.query(CaseScorableORM)
        .join(CaseORM, CaseScorableORM.case_id == CaseORM.id)
        .filter(CaseORM.casebook_id == cb.id)
        .all()
        if sc.meta and sc.meta.get("turn_hash")
    }

    # --- 4) Create cases from (user → assistant) pairs ---
    for u_turn, a_turn in user_assistant_pairs(turns):
        user_text = u_turn["text"].strip()
        assistant_text = a_turn["text"].strip()
        thash = _turn_hash(user_text, assistant_text)

        if thash in existing_hashes:
            logger.debug(f"[Skip] Duplicate scorable hash {thash[:10]}…")
            continue

        case = memory.casebooks.add_case(
            casebook_id=cb.id,
            goal_id=goal.get("id"),
            goal_text=goal.get("goal_text"),
            agent_name="chat_import",
            prompt_text=user_text,
            scorables=[
                {
                    "text": assistant_text,
                    "role": "assistant",
                    "source": "chat",
                    "meta": {"turn_hash": thash, "u_ts": u_turn.get("ts"), "a_ts": a_turn.get("ts")},
                }
            ],
            response_texts=assistant_text,
        )

        counts["cases_created"] += 1
        counts["scorables_created"] += 1  # one scorable per case here
        existing_hashes.add(thash)

        logger.info(f"[Case] Created case {case.id}, +1 case, +1 scorable")

    return cb, counts


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _turn_hash(user_text: str, assistant_text: str) -> str:
    return hash_text(user_text.strip() + "\n---\n" + assistant_text.strip())


def _extract_turns_from_mapping(mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract a *linearized*, ordered list of turns from an OpenAI/ChatGPT-style mapping.

    We attempt to order by:
      1) message.create_time if present
      2) graph traversal from roots (nodes with no parent) to leaves
      3) id sort as ultimate fallback

    Output: [{'role': 'user'|'assistant', 'text': str, 'ts': float|None}, ...]
    """
    nodes: List[Dict[str, Any]] = []

    # Collect nodes with message payloads
    for _id, node in (mapping or {}).items():
        msg = (node or {}).get("message") or {}
        if not msg:
            continue

        role = (msg.get("author", {}) or {}).get("role") or msg.get("role")
        content = msg.get("content")

        # OpenAI export variants:
        # - content can be list of blocks with 'text' or 'parts'
        # - or string in some tools
        text = None
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # concat all text-like parts
            txt_parts = []
            for c in content:
                if isinstance(c, str):
                    txt_parts.append(c)
                elif isinstance(c, dict):
                    # common keys: 'text', 'content', 'parts'
                    if "text" in c and isinstance(c["text"], str):
                        txt_parts.append(c["text"])
                    elif "content" in c and isinstance(c["content"], str):
                        txt_parts.append(c["content"])
                    elif "parts" in c and isinstance(c["parts"], list):
                        txt_parts.extend([p for p in c["parts"] if isinstance(p, str)])
            text = "\n".join([p for p in txt_parts if p])
        elif isinstance(content, dict):
            # Claude-style or other: content: {"text": "..."}
            if isinstance(content.get("text"), str):
                text = content["text"]

        if not text:
            # Some exports store a plain array of strings under message['content']['parts']
            parts = (msg.get("content") or {}).get("parts") if isinstance(msg.get("content"), dict) else None
            if isinstance(parts, list):
                text = "\n".join([p for p in parts if isinstance(p, str)])

        if not text:
            continue

        create_time = msg.get("create_time") or msg.get("created_at")
        try:
            ts = float(create_time) if create_time is not None else None
        except Exception:
            # ISO string case
            try:
                ts = datetime.fromisoformat(str(create_time)).timestamp()
            except Exception:
                ts = None

        nodes.append(
            {
                "id": _id,
                "parent": node.get("parent"),
                "children": node.get("children") or [],
                "role": (role or "").lower(),
                "text": text,
                "ts": ts,
                "has_time": ts is not None,
            }
        )

    if not nodes:
        return []

    # Primary ordering: by timestamp when all (or most) have it
    with_ts = sum(1 for n in nodes if n["has_time"])
    if with_ts >= max(1, int(0.6 * len(nodes))):  # if ≥60% have timestamps, prefer time ordering
        nodes.sort(key=lambda n: (n["ts"] if n["ts"] is not None else float("inf")))
    else:
        # Graph order: from roots to leaves using parent pointers
        by_id = {n["id"]: n for n in nodes}
        roots = [n for n in nodes if not n.get("parent")] or nodes[:]
        visited, ordered = set(), []

        def dfs(n):
            if n["id"] in visited:
                return
            visited.add(n["id"])
            ordered.append(n)
            for cid in n.get("children") or []:
                if cid in by_id:
                    dfs(by_id[cid])

        for r in roots:
            dfs(r)
        # Fallback: include any stragglers not connected
        for n in nodes:
            if n["id"] not in {x["id"] for x in ordered}:
                ordered.append(n)
        nodes = ordered

    # Normalize to (role, text) chats only
    out = [
        {"role": n["role"], "text": n["text"], "ts": n["ts"]}
        for n in nodes
        if n.get("role") in ("user", "assistant") and n.get("text")
    ]

    # Final cleanup: ensure strict alternation when possible
    # If we see consecutive same-role messages, we still keep them (some UIs do that),
    # and the pairing loop above will only take (user -> assistant) sequences.
    return out
