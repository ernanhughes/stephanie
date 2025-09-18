"""
Chat Importer Module

This module handles importing conversations from various chat formats (JSON/HTML)
into the Stephanie system's database and casebook structure. It supports ChatGPT-style
exports and provides deduplication, normalization, and structured storage.

Key Features:
- Supports both JSON and HTML chat exports
- Automatic deduplication using content hashing
- Conversation normalization and turn extraction
- Integration with both chat storage and casebook systems
- Comprehensive logging for import operations

Usage:
    from stephanie.tools.chat_importer import import_conversations
    result = import_conversations(memory, "/path/to/chat/exports", context)
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from bs4 import BeautifulSoup

from stephanie.models.casebook import CaseBookORM, CaseORM, CaseScorableORM

logger = logging.getLogger(__name__)


def _turn_hash(user_text: str, assistant_text: str) -> str:
    """
    Compute a stable hash for a user→assistant turn to ensure uniqueness.
    
    Args:
        user_text: The user's message text
        assistant_text: The assistant's response text
        
    Returns:
        SHA256 hash of the combined turn content
    """
    key = (user_text.strip() + "||" + assistant_text.strip()).encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def file_hash(path: str) -> str:
    """
    Compute SHA256 hash of a file's contents.
    
    Args:
        path: Path to the file to hash
        
    Returns:
        SHA256 hash of file contents
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def safe_timestamp(ts: Optional[float]) -> Optional[datetime]:
    """
    Safely convert a timestamp to datetime object.
    
    Args:
        ts: Timestamp as float or None
        
    Returns:
        datetime object or None if conversion fails
    """
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts))
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert timestamp: {ts}")
        return None


def conversation_to_chat(memory, bundle: dict, context: dict) -> Any:
    """
    Import a conversation bundle into the chat storage system.
    
    Args:
        memory: Memory interface for database operations
        bundle: Conversation data dictionary
        context: Execution context
        
    Returns:
        The created conversation object
    """
    title = bundle.get("title", "Untitled Conversation")
    conversation_id = bundle.get("id") or bundle.get("conversation_id")
    
    logger.info(f"Importing conversation: {title} (ID: {conversation_id})")

    # Create conversation record
    conv = memory.chats.add_conversation(
        {
            "provider": "openai",
            "external_id": conversation_id,
            "title": title,
            "created_at": safe_timestamp(bundle.get("created_at")),
            "meta": {"raw": bundle},
        }
    )

    # Extract and normalize conversation turns
    if "mapping" in bundle:
        raw_turns = _extract_turns_from_mapping(bundle["mapping"])
        logger.debug(f"Extracted {len(raw_turns)} turns from mapping structure")
    elif "turns" in bundle:
        raw_turns = bundle["turns"]
        logger.debug(f"Using {len(raw_turns)} direct turns from bundle")
    else:
        raw_turns = []
        logger.warning("No conversation turns found in bundle")

    turns = normalize_turns(raw_turns)
    if not turns:
        logger.info(f"⚠️ Skipping {title}, no valid messages found after normalization")
        return conv

    # Persist messages to database
    messages = memory.chats.add_messages(conv.id, turns)
    logger.debug(f"Persisted {len(messages)} messages to database")

    # Build Q/A turns
    turn_rows = memory.chats.add_turns(conv.id, [m.to_dict() for m in messages])
    logger.info(f"✅ Imported {title} → messages={len(messages)}, turns={len(turn_rows)}")

    return conv


def import_conversations(memory, path: str, context: dict) -> dict:
    """
    Import all conversations from JSON/HTML files in a directory.
    
    Args:
        memory: Memory interface for database operations
        path: Directory path containing chat export files
        context: Execution context
        
    Returns:
        Dictionary with import statistics
    """
    logger.info(f"Starting chat import from directory: {path}")
    
    total_files = total_skipped = total_convs = 0
    total_messages = total_turns = 0

    # Process all JSON and HTML files in the directory
    for fp in glob.glob(os.path.join(path, "*.json")) + glob.glob(os.path.join(path, "*.html")):
        total_files += 1
        file_basename = os.path.basename(fp)
        
        # Check if file has already been imported
        h = file_hash(fp)
        if memory.chats.exists_conversation(h):
            logger.info(f"⏭️ Skipping {file_basename}, already imported (hash: {h[:8]}...)")
            total_skipped += 1
            continue

        logger.info(f"Processing {file_basename}")
        
        # Parse file based on format
        try:
            if fp.endswith(".html"):
                with open(fp, "r", encoding="utf-8") as f:
                    bundles = [{"title": os.path.basename(fp), "turns": parse_html(f.read())}]
            else:
                with open(fp, "r", encoding="utf-8") as f:
                    bundles = parse_json(json.load(f))
        except Exception as e:
            logger.error(f"Failed to parse {file_basename}: {str(e)}")
            continue

        # Process each conversation bundle in the file
        for bundle in bundles:
            try:
                conv = conversation_to_chat(memory, bundle, context)
                # Add file metadata to conversation
                conv.meta = {"hash": h, "source_file": os.path.basename(fp)}
                memory.session.add(conv)
                memory.commit()
                total_convs += 1
                total_messages += len(bundle.get("turns", []))
            except Exception as e:
                logger.error(f"Failed to import conversation from {file_basename}: {str(e)}")
                memory.session.rollback()

    # Compile and return import statistics
    stats = {
        "files_processed": total_files,
        "files_skipped": total_skipped,
        "conversations_imported": total_convs,
        "messages_seen": total_messages,
        "turns_estimate": total_messages // 2,  # rough estimate
    }
    
    logger.info(f"Import completed: {stats}")
    return stats


def parse_html(html_content: str) -> List[Dict[str, str]]:
    """
    Parse ChatGPT-style HTML export into a list of conversation turns.
    
    Args:
        html_content: HTML string content
        
    Returns:
        List of turns with role and text fields
    """
    logger.debug("Parsing HTML content")
    soup = BeautifulSoup(html_content, "html.parser")
    turns = []
    
    # Look for markdown blocks and infer role from context
    for block in soup.select("div.markdown"):
        parent = block.parent or {}
        parent_classes = getattr(parent, "get", lambda *_: "")("class", []) or str(parent)
        role = "user" if "user" in str(parent_classes).lower() else "assistant"
        text = block.get_text(separator="\n").strip()
        
        if text:
            turns.append({"role": role, "text": text})
            logger.debug(f"Found {role} message: {text[:50]}...")
    
    logger.info(f"Extracted {len(turns)} turns from HTML")
    return turns




def _parse_vendor_conv(conv: dict) -> dict | None:
    title = conv.get("title") or "Conversation"
    conv_id = conv.get("id") or conv.get("conversation_id")
    msgs = _vendor_collect_messages(conv)
    if not msgs:
        return None

    turns = []
    for m in msgs:
        role = m.get("role")
        if role not in {"user", "assistant"}:
            continue
        txt = _vendor_assistant_text(m) if role == "assistant" else _vendor_user_text(m)
        if txt:
            turns.append({"role": role, "text": txt})

    if not turns:
        return None

    # conversation-level created_at if present, else earliest msg time
    created_at = conv.get("created_at")
    if created_at is None:
        # find earliest numeric or ISO ts
        times = []
        for m in msgs:
            ts = m.get("timestamp") or m.get("created_at") or m.get("create_time")
            try:
                times.append(float(ts))
            except Exception:
                try:
                    from dateutil import parser as _parser
                    times.append(_parser.isoparse(str(ts)).timestamp())
                except Exception:
                    pass
        created_at = min(times) if times else None

    return {
        "title": title,
        "created_at": created_at,
        "turns": normalize_turns(turns),
        "conversation_id": conv_id,
    }


def _parse_chatgpt_conv(conv: dict) -> dict | None:
    title = conv.get("title") or "Conversation"
    created_at = conv.get("create_time")
    turns = []

    if "messages" in conv and isinstance(conv["messages"], list):
        for msg in conv["messages"]:
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, dict) and "parts" in content:
                text = "\n".join(p for p in content.get("parts", []) if isinstance(p, str)).strip()
            else:
                text = content.strip() if isinstance(content, str) else ""
            if role in {"user", "assistant"} and text:
                turns.append({"role": role, "text": text})

    elif "mapping" in conv and isinstance(conv["mapping"], dict):
        turns = _extract_turns_from_mapping(conv["mapping"])

    turns = normalize_turns(turns)
    if not turns:
        return None

    return {"title": title, "created_at": created_at, "turns": turns, "conversation_id": conv.get("conversation_id")}


def parse_json(json_obj):
    """
    Always returns List[bundles]. Handles:
    - Top-level array of vendor conversations
    - Single vendor conversation with chat.history/messages
    - ChatGPT exports with 'messages' or 'mapping'
    """
    bundles = []

    if isinstance(json_obj, list):
        for conv in json_obj:
            if not isinstance(conv, dict):
                continue
            parsed = _parse_vendor_conv(conv) if "chat" in conv else _parse_chatgpt_conv(conv)
            if parsed:
                bundles.append(parsed)
        return bundles

    if isinstance(json_obj, dict):
        parsed = _parse_vendor_conv(json_obj) if "chat" in json_obj else _parse_chatgpt_conv(json_obj)
        return [parsed] if parsed else []

    # Unknown shape -> empty list
    return []


def normalize_turns(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Normalize conversation turns by cleaning text and filtering valid roles.
    
    Args:
        turns: List of turn dictionaries with role and text
        
    Returns:
        Normalized list of turns with only user/assistant roles and cleaned text
    """
    logger.debug("Normalizing conversation turns")
    out = []
    
    for t in turns:
        role = t.get("role", "").lower()
        if role not in {"user", "assistant"}:
            logger.debug(f"Skipping turn with invalid role: {role}")
            continue
            
        text = (t.get("text") or "").strip()
        if not text:
            logger.debug("Skipping empty text turn")
            continue
            
        out.append({"role": role, "text": text})
    
    logger.info(f"Normalized {len(out)} of {len(turns)} turns")
    return out


def conversation_to_casebook(memory, bundle: dict, context: dict) -> CaseBookORM:
    """
    Convert a conversation into a casebook with individual turns as cases.
    
    Args:
        memory: Memory interface for database operations
        bundle: Conversation data dictionary
        context: Execution context
        
    Returns:
        The created casebook object
    """
    title = bundle.get("title", "Untitled Conversation")
    conversation_id = bundle.get("id") or bundle.get("conversation_id")
    
    logger.info(f"Converting conversation to casebook: {title}")

    # Create or get casebook
    cb = memory.casebooks.ensure_casebook(
        name=f"chat_{conversation_id or title[:200]}",
        description=f"Imported chat conversation: {title}",
    )
    logger.info(f"CaseBook: {cb.name} (id={cb.id})")

    # Extract turns from conversation
    mapping = bundle.get("mapping", {})
    if mapping:
        turns = _extract_turns_from_mapping(mapping)
        logger.debug(f"Extracted {len(turns)} turns from mapping")
    else:
        turns = bundle.get("turns", [])
        logger.debug(f"Using {len(turns)} direct turns")

    if not turns:
        logger.warning(f"⚠️ Skipping {title}, no turns found")
        return cb

    # Get or create goal for this casebook
    goal = memory.goals.get_or_create(
        {"goal_text": f"{cb.name}", "description": f"{cb.description}"}
    )
    goal = goal.to_dict()

    # Check for existing turns to avoid duplicates
    existing_hashes = {
        sc.meta.get("turn_hash")
        for sc in memory.session.query(CaseScorableORM)
        .join(CaseORM, CaseScorableORM.case_id == CaseORM.id)
        .filter(CaseORM.casebook_id == cb.id)
        .all()
        if sc.meta and sc.meta.get("turn_hash")
    }
    logger.info(f"Found {len(existing_hashes)} existing turns in casebook")

    # Process each user-assistant turn pair
    new_turns_added = 0
    for i in range(len(turns) - 1):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            user_text = turns[i]["text"].strip()
            assistant_text = turns[i + 1]["text"].strip()
            thash = _turn_hash(user_text, assistant_text)

            # Skip duplicates
            if thash in existing_hashes:
                logger.debug(f"Skipping duplicate turn: {thash[:10]}…")
                continue

            # Add turn as a case
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
                        "meta": {"turn_hash": thash},
                    }
                ],
                response_texts=assistant_text,
            )
            logger.debug(f"Added case id={case.id} with hash {thash[:10]}…")
            existing_hashes.add(thash)
            new_turns_added += 1

    logger.info(f"Added {new_turns_added} new turns to casebook {cb.id}")
    return cb


def _extract_turns_from_mapping(mapping: dict) -> list[dict]:
    """
    Extract ordered conversation turns from a ChatGPT-style "mapping" export.
    Robust to nodes without 'parts', non-text content_types (e.g., user_editable_context),
    and missing children arrays. Only returns user/assistant textual turns.
    """

    if not isinstance(mapping, dict) or not mapping:
        return []

    # ---------- helpers ----------
    def _msg_time(node: dict) -> float:
        msg = (node or {}).get("message") or {}
        ts = msg.get("create_time") or msg.get("timestamp") or 0
        try:
            return float(ts)
        except Exception:
            return 0.0

    def _content_to_text(content) -> str:
        """
        Accepts: str or dict. Returns a clean string or ''.
        - If dict, prefer 'parts' when present
        - Ignore content_type that is clearly non-text (e.g., user_editable_context)
        """
        if content is None:
            return ""

        # raw string content
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, dict):
            ctype = content.get("content_type")
            # Drop clearly non-text systemish payloads
            if ctype and ctype not in ("text", "code", "tether_browsing_display", "assistant_response"):
                return ""

            parts = content.get("parts")
            if isinstance(parts, list):
                texts = [p for p in parts if isinstance(p, str) and p.strip()]
                if texts:
                    return "\n".join(texts).strip()

            # fallbacks some exports use
            for key in ("text", "content", "value"):
                v = content.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        # unsupported shape
        return ""

    def _walk(node_id: str) -> list[dict]:
        node = mapping.get(node_id) or {}
        out: list[dict] = []

        msg = node.get("message") or {}
        role = ((msg.get("author") or {}).get("role") or "").lower()
        text = _content_to_text(msg.get("content"))

        if role in ("user", "assistant") and text:
            out.append({"role": role, "text": text})

        children = node.get("children") or []
        # some exports have None; always iterate a list
        for child_id in children:
            out.extend(_walk(child_id))

        return out

    # ---------- roots + traversal ----------
    roots = [nid for nid, node in mapping.items() if (node or {}).get("parent") is None]
    if not roots:
        return []

    # sort roots chronologically; walk each
    turns: list[dict] = []
    for root in sorted(roots, key=lambda rid: _msg_time(mapping.get(rid) or {})):
        turns.extend(_walk(root))

    return turns


def _vendor_assistant_text(msg: dict) -> str:
    """
    For vendor schema where assistant may have content_list with phases.
    Prefer 'answer' phase; ignore 'think'/'reasoning' phases.
    Fallbacks to msg['content'] if no content_list.
    """
    # 1) Prefer content_list with phase == 'answer'
    cl = msg.get("content_list")
    if isinstance(cl, list) and cl:
        # try explicit phase
        answers = [c for c in cl if isinstance(c, dict) and c.get("phase") == "answer" and c.get("content")]
        if answers:
            return "\n".join(c["content"].strip() for c in answers if isinstance(c.get("content"), str))
        # else: take the last non-empty content
        tail = [c.get("content") for c in cl if isinstance(c, dict) and isinstance(c.get("content"), str) and c.get("content").strip()]
        if tail:
            return "\n".join(tail)

    # 2) Fallback to plain content
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()

    # 3) Nothing usable
    return ""


def _vendor_collect_messages(conv: dict) -> list[dict]:
    """
    Supports both:
      conv['chat']['history']['messages'] : dict id->msg
      conv['chat']['messages']            : list[msg]
    De-dupes by id and sorts by timestamp when present.
    """
    chat = (conv.get("chat") or {})
    history = (chat.get("history") or {})
    msgs = []

    mapping = history.get("messages")
    if isinstance(mapping, dict) and mapping:
        msgs.extend(mapping.values())

    arr = chat.get("messages")
    if isinstance(arr, list) and arr:
        msgs.extend(arr)

    # De-dupe by id
    seen, out = set(), []
    for m in msgs:
        mid = m.get("id") or m.get("messageId") or m.get("_id")
        if mid and mid not in seen:
            seen.add(mid)
            out.append(m)

    # Sort by timestamp-ish
    def _ts(m):
        ts = m.get("timestamp") or m.get("created_at") or m.get("create_time")
        # Allow numeric epoch or ISO. If not parseable, push to end.
        try:
            return float(ts)
        except Exception:
            # Try ISO
            try:
                # Keep it lightweight: only split seconds when present
                from dateutil import parser as _parser  # if you have python-dateutil; otherwise skip
                return _parser.isoparse(str(ts)).timestamp()
            except Exception:
                return float("inf")

    out.sort(key=_ts)
    return out


def _vendor_user_text(msg: dict) -> str:
    """
    Extract user text for vendor schema.
    """
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    # Some vendors might bucket parts differently; add minimal fallback here if needed.
    return ""

