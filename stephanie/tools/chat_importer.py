# tools/chat_importer.py
import json
import glob
import os
from bs4 import BeautifulSoup

import hashlib
from sqlalchemy import cast, String
from stephanie.models.casebook import CaseBookORM

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def import_conversations(memory, path: str):
    """
    Walks a directory, parses chat files, and imports them into memory as CaseBooks.
    
    Args:
        memory (MemoryTool): The memory tool instance for persistence.
        path (str): The path to the directory containing chat exports.
    """
    for fp in glob.glob(os.path.join(path, "*.json")) + glob.glob(os.path.join(path, "*.html")):
        h = file_hash(fp)

        existing = (
            memory.session.query(CaseBookORM)
            .filter(cast(CaseBookORM.meta["hash"], String) == h)
            .first()
        )
        if existing:
            print(f"Skipping {fp}, already imported.")
            continue

        # Parse file
        if fp.endswith(".html"):
            with open(fp, "r", encoding="utf-8") as f:
                bundles = parse_html(f.read())
        else:
            with open(fp, "r", encoding="utf-8") as f:
                bundles = parse_json(json.load(f))

        for bundle in bundles:
            cb = conversation_to_casebook(memory, bundle)
            cb.meta = {"hash": h, "source_file": os.path.basename(fp)}
            memory.session.add(cb)
            memory.commit()
            print(f"Imported: {cb.name}")

def parse_html(html_content: str):
    """Parses a ChatGPT-style HTML export."""
    soup = BeautifulSoup(html_content, "html.parser")
    turns = []
    # Simplified parsing logic; needs to be robust to handle different structures.
    for element in soup.select("div.markdown"):
        role = "user" if "user-role" in str(element.parent) else "assistant"
        text = element.get_text(separator="\n").strip()
        turns.append({"role": role, "text": text})
    return turns

def parse_html(html_content: str):
    """Parses a ChatGPT-style HTML export."""
    soup = BeautifulSoup(html_content, "html.parser")
    turns = []
    # Simplified parsing logic; needs to be robust to handle different structures.
    for element in soup.select("div.markdown"):
        role = "user" if "user-role" in str(element.parent) else "assistant"
        text = element.get_text(separator="\n").strip()
        turns.append({"role": role, "text": text})
    return turns

def parse_json(json_obj):
    """
    Parse ChatGPT-style exports.

    Accepts either:
      - a single conversation dict, or
      - a list of conversation dicts

    Returns: List[{"title": str, "created_at": float|None, "turns": List[{"role": str, "text": str}]}]
    """

    def _parse_one(conv):
        if not isinstance(conv, dict):
            return None

        title = conv.get("title") or "Conversation"
        created_at = conv.get("create_time")

        turns = []

        # Format A: explicit "messages"
        if "messages" in conv and isinstance(conv["messages"], list):
            for msg in conv["messages"]:
                role = msg.get("role")
                content = msg.get("content")

                # content could be dict with "parts" or a raw string
                if isinstance(content, dict) and "parts" in content:
                    text = "\n".join(part for part in content.get("parts", []) if isinstance(part, str)).strip()
                else:
                    text = (content or "").strip() if isinstance(content, str) else ""

                if role in {"user", "assistant"} and text:
                    turns.append({"role": role, "text": text})

        # Format B: ChatGPT export with "mapping"
        elif "mapping" in conv and isinstance(conv["mapping"], dict):
            for node in conv["mapping"].values():
                msg = node.get("message")
                if not msg:
                    continue
                role = (msg.get("author") or {}).get("role")
                parts = (msg.get("content") or {}).get("parts") or []
                text = "\n".join(p for p in parts if isinstance(p, str)).strip()
                if role in {"user", "assistant"} and text:
                    turns.append({"role": role, "text": text})

        # ignore "system" or empty messages by design
        if not turns:
            return None

        return {"title": title, "created_at": created_at, "turns": turns}

    if isinstance(json_obj, list):
        bundles = []
        for conv in json_obj:
            parsed = _parse_one(conv)
            if parsed:
                bundles.append(parsed)
        return bundles

    parsed = _parse_one(json_obj)
    return [parsed] if parsed else []

def normalize_turns(turns: list):
    """Normalizes the text content of turns."""
    for turn in turns:
        text = turn["text"]
        # Basic normalization: strip PII, clean whitespace, etc.
        turn["text"] = text.strip()
    return turns

# tools/chat_importer.py

def conversation_to_casebook(memory, bundle: dict):
    """
    Convert a parsed conversation bundle into a CaseBook + Cases + Scorables.

    Args:
        memory: MemoryTool with .casebooks, .cases, .scorables stores
        bundle: One conversation object from ChatGPT JSON export

    Returns:
        CaseBookORM object
    """
    # --- 1. CaseBook creation ---
    title = bundle.get("title", "Untitled Conversation")
    conversation_id = bundle.get("id") or bundle.get("conversation_id")

    cb = memory.casebooks.ensure_casebook(
        name=f"chat_{conversation_id or title[:50]}",  # truncate long names
        description=f"Imported chat conversation: {title}"
    )

    # --- 2. Handle mapping-based format ---
    mapping = bundle.get("mapping", {})
    if mapping:
        turns = _extract_turns_from_mapping(mapping)
    else:
        # --- 3. Fallback: use normalized turns directly ---
        turns = bundle.get("turns", [])

    if not turns:
        print(f"⚠️ Skipping {title}, no turns found")
        return cb

    # --- 4. Pair up user → assistant messages as cases ---
    for i in range(len(turns) - 1):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            user_text = turns[i]["text"]
            assistant_text = turns[i + 1]["text"]

            case = memory.casebooks.add_case(
                casebook_id=cb.id,
                goal_id=None,           # can attach later if tied to a goal
                goal_text=None,
                agent_name="chat_import",
                prompt_text=user_text,
                scorables=[{
                    "text": assistant_text,
                    "role": "assistant",
                    "source": "chat"
                }],
                response_text=assistant_text
            )

    return cb

def normalize_turns(turns: list):
    """Normalizes the text content of turns. Skips empty/system messages."""
    normalized = []
    for turn in turns:
        # Gracefully handle missing text fields
        text = turn.get("text")
        
        # If text is missing but message.content.parts exists
        if not text and "message" in turn:
            content = turn["message"].get("content", {})
            if isinstance(content, dict) and "parts" in content:
                parts = content.get("parts", [])
                if parts and isinstance(parts, list):
                    text = " ".join([str(p) for p in parts if p])

        if not text or not text.strip():
            # skip completely empty/system turns
            continue

        normalized.append({
            "role": turn.get("role", "unknown"),
            "text": text.strip()
        })

    return normalized

def _extract_turns_from_mapping(mapping: dict) -> list[dict]:
    """
    Extract ordered conversation turns from a ChatGPT-style "mapping" export.

    Args:
        mapping (dict): Mapping of node_id → { message, parent, children }

    Returns:
        List of {"role": str, "text": str} turns in conversation order
    """

    # Find roots (nodes with no parent)
    roots = [node_id for node_id, node in mapping.items() if node.get("parent") is None]
    if not roots:
        return []

    def walk(node_id: str) -> list[dict]:
        node = mapping.get(node_id)
        if not node:
            return []

        msg = node.get("message")
        turns = []
        if msg:
            role = (msg.get("author") or {}).get("role")
            content = msg.get("content") or {}
            parts = content.get("parts") if isinstance(content, dict) else []
            text = "\n".join(p for p in parts if isinstance(p, str)).strip()
            if role in {"user", "assistant"} and text:
                turns.append({"role": role, "text": text})

        for child_id in node.get("children", []):
            turns.extend(walk(child_id))

        return turns

    # Usually ChatGPT exports have a single root thread → walk it
    return walk(roots[0])
