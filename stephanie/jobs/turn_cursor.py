# stephanie/jobs/turn_cursor.py
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CursorState:
    """In-file state layout (JSON-serializable)."""
    conv_ids: List[int]
    index: int
    offsets: Dict[str, int]
    version: int = 1
    updated_at: float = 0.0  # epoch seconds


class TurnCursor:
    """
    Durable cursor that iterates through chat turns across many conversations.

    The cursor remembers:
      - a list of conversation IDs (`conv_ids`)
      - which conversation we are currently on (`index`)
      - an offset (how many turns we've advanced) per conversation (`offsets`)

    Typical usage (nightly job):
        cur = TurnCursor("./data/cursors/overnight_turn_cursor.json")
        cur.set_conversations(latest_conv_ids)           # call periodically to refresh
        conv_id, offset = cur.next_batch_hint()          # where to read next
        rows = chat_store.get_turn_texts_for_conversation(conv_id, offset=offset, limit=20)
        # ... process rows ...
        cur.advance(consumed=len(rows))                  # move the cursor forward

    Notes:
      - `advance()` only moves within the current conversation unless you pass
        `total_in_conv` and the batch consumes the remainder; then it moves to the next conversation.
      - The conversation list can be refreshed at any time; the cursor will try
        to keep the current conv_id if still present, otherwise it clamps to the start.
      - All writes are atomic (tempfile + replace) to avoid partial state on crash.
    """

    def __init__(
        self,
        state_path: str | Path,
        *,
        start_conv_id: Optional[int] = None,
        start_offset: int = 0,
        rotation: str = "wrap",   # "wrap" -> cycle at end, "stop" -> clamp at last
    ):
        self.path = Path(state_path)
        self.rotation = rotation if rotation in ("wrap", "stop") else "wrap"
        self._ensure_dirs()

        # default state
        self.state = CursorState(conv_ids=[], index=0, offsets={}, version=1, updated_at=time.time())

        # load existing state if present
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._load_from_dict(data)
            except Exception:
                # keep defaults on parse error (don't crash nightly)
                pass

        # explicit starting position (optional)
        if start_conv_id is not None:
            self.state.conv_ids = [int(start_conv_id)]
            self.state.index = 0
            self.state.offsets[str(int(start_conv_id))] = max(0, int(start_offset))
            self._touch()
            self._save()

    # --------------------------- public API ---------------------------

    def set_conversations(self, conv_ids: List[int]) -> None:
        """
        Replace/refresh the list of conversations (e.g., newest-first snapshot).
        Preserves current index if the active conversation still exists.
        Offsets for conversations that remain are preserved; removed ones are dropped.
        """
        conv_ids = [int(x) for x in conv_ids]
        old_ids = self.state.conv_ids
        old_active = self.current_conv_id()

        # Build new offsets preserving known ones
        new_offsets: Dict[str, int] = {}
        for cid in conv_ids:
            key = str(cid)
            new_offsets[key] = int(self.state.offsets.get(key, 0))

        # Decide new index: keep current conversation if it still exists; else reset to 0
        new_index = 0
        if old_active is not None and old_active in conv_ids:
            new_index = conv_ids.index(old_active)

        self.state.conv_ids = conv_ids
        self.state.index = min(max(0, new_index), max(0, len(conv_ids) - 1))
        self.state.offsets = new_offsets
        self._touch()
        self._save()

    def current_conv_id(self) -> Optional[int]:
        ids = self.state.conv_ids
        if not ids:
            return None
        idx = min(max(0, self.state.index), len(ids) - 1)
        return int(ids[idx])

    def next_batch_hint(self) -> Tuple[Optional[int], int]:
        """
        Returns (conversation_id, offset) where your next DB read should start.
        If there are no conversations, returns (None, 0).
        """
        cid = self.current_conv_id()
        if cid is None:
            return None, 0
        off = int(self.state.offsets.get(str(cid), 0))
        return cid, off

    def advance(self, consumed: int, total_in_conv: Optional[int] = None) -> None:
        """
        Move the cursor forward within the current conversation by `consumed`.
        If `total_in_conv` is provided and we've reached/exceeded it, advance to the next conversation.
        """
        if not self.state.conv_ids:
            return

        idx = min(max(0, self.state.index), len(self.state.conv_ids) - 1)
        cid = int(self.state.conv_ids[idx])
        key = str(cid)
        cur_off = int(self.state.offsets.get(key, 0))
        new_off = max(0, cur_off + max(0, int(consumed)))

        # If we know the total, decide whether to hop to next conversation
        if total_in_conv is not None and new_off >= int(total_in_conv):
            self._goto_next_conv()
        else:
            self.state.offsets[key] = new_off

        self._touch()
        self._save()

    def skip_conversation(self) -> None:
        """Skip the current conversation entirely (go to next, reset its offset to 0 if unseen)."""
        if not self.state.conv_ids:
            return
        self._goto_next_conv()
        self._touch()
        self._save()

    def rewind_current(self) -> None:
        """Reset the offset of the current conversation to 0 (stay on it)."""
        cid = self.current_conv_id()
        if cid is None:
            return
        self.state.offsets[str(cid)] = 0
        self._touch()
        self._save()

    def set_position(self, conv_id: int, offset: int = 0) -> None:
        """Force the cursor to a specific conversation and offset (creates conv_id if missing)."""
        conv_id = int(conv_id)
        if conv_id not in self.state.conv_ids:
            self.state.conv_ids.append(conv_id)
        self.state.index = self.state.conv_ids.index(conv_id)
        self.state.offsets[str(conv_id)] = max(0, int(offset))
        self._touch()
        self._save()

    def reset(self) -> None:
        """Reset to the beginning: index=0, all offsets cleared (but keep conv_ids)."""
        self.state.index = 0
        self.state.offsets = {}
        self._touch()
        self._save()

    def snapshot(self) -> dict:
        """Return a dict view of the current state (for diagnostics)."""
        return {
            "conv_ids": list(self.state.conv_ids),
            "index": int(self.state.index),
            "offsets": dict(self.state.offsets),
            "version": int(self.state.version),
            "updated_at": float(self.state.updated_at),
            "active_conv_id": self.current_conv_id(),
            "active_offset": (self.state.offsets.get(str(self.current_conv_id()), 0)
                              if self.current_conv_id() is not None else 0),
        }

    # -------------------------- internals -----------------------------

    def _goto_next_conv(self) -> None:
        if not self.state.conv_ids:
            return
        idx = self.state.index + 1
        if idx >= len(self.state.conv_ids):
            if self.rotation == "wrap":
                idx = 0
            else:  # "stop"
                idx = len(self.state.conv_ids) - 1
        self.state.index = idx

    def _ensure_dirs(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _touch(self) -> None:
        self.state.updated_at = time.time()

    def _save(self) -> None:
        # atomic write on same filesystem
        payload = {
            "conv_ids": self.state.conv_ids,
            "index": self.state.index,
            "offsets": self.state.offsets,
            "version": self.state.version,
            "updated_at": self.state.updated_at,
        }
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(self.path.parent), encoding="utf-8") as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.path)

    def _load_from_dict(self, data: dict) -> None:
        try:
            conv_ids = [int(x) for x in data.get("conv_ids", [])]
            index = int(data.get("index", 0))
            offsets = {str(int(k)): int(v) for k, v in (data.get("offsets", {}) or {}).items()}
            version = int(data.get("version", 1))
            updated_at = float(data.get("updated_at", time.time()))
        except Exception:
            # fall back to defaults on malformed state
            conv_ids, index, offsets, version, updated_at = [], 0, {}, 1, time.time()

        # clamp index
        if conv_ids:
            index = min(max(0, index), len(conv_ids) - 1)
        else:
            index = 0

        self.state = CursorState(conv_ids=conv_ids, index=index, offsets=offsets, version=version, updated_at=updated_at)

    # -------------------------- dunder --------------------------------

    def __repr__(self) -> str:
        cid = self.current_conv_id()
        off = self.state.offsets.get(str(cid), 0) if cid is not None else 0
        return f"<TurnCursor convs={len(self.state.conv_ids)} index={self.state.index} active={cid} offset={off}>"
