# stephanie/memory/chat_store.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import aliased, selectinload, Query
from sqlalchemy import desc, func, text

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.chat import (ChatConversationORM, ChatMessageORM,
                                   ChatTurnORM)
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType


class ChatStore(BaseSQLAlchemyStore):
    orm_model = ChatConversationORM
    # Use column name so BaseSQLAlchemyStore can apply .desc() reliably
    default_order_by = "created_at"

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "chats"

    def name(self) -> str:
        return self.name

    # ---------- Conversations ----------

    def add_conversation(self, data: dict) -> ChatConversationORM:
        def op(s):
            conv = ChatConversationORM(**data)
            s.add(conv)
            s.flush()
            return conv

        return self._run(op)

    def get_all(self, limit: int = 100) -> List[ChatConversationORM]:
        def op(s):
            return (
                s.query(ChatConversationORM)
                .order_by(
                    desc(
                        getattr(
                            ChatConversationORM,
                            "created_at",
                            ChatConversationORM.id,
                        )
                    )
                )
                .limit(limit)
                .all()
            )

        return self._run(op)

    def exists_conversation(self, file_hash: str) -> bool:
        def op(s):
            return (
                s.query(ChatConversationORM)
                .filter(
                    func.jsonb_extract_path_text(
                        ChatConversationORM.meta, "hash"
                    )
                    == file_hash
                )
                .first()
                is not None
            )

        return self._run(op)

    def get_conversation(self, conv_id: int) -> Optional[ChatConversationORM]:
        def op(s):
            return s.get(ChatConversationORM, conv_id)

        return self._run(op)

    # ---------- Messages ----------

    def add_messages(
        self, conv_id: int, messages: List[dict]
    ) -> List[ChatMessageORM]:
        def op(s):
            objs: List[ChatMessageORM] = []
            for i, msg in enumerate(messages):
                objs.append(
                    ChatMessageORM(
                        conversation_id=conv_id,
                        role=msg["role"],
                        text=msg.get("text", ""),
                        order_index=i,
                        parent_id=msg.get("parent_id"),
                        meta=msg.get("meta", {}),
                    )
                )
            s.add_all(objs)
            s.flush()
            return objs

        return self._run(op)

    def get_messages(self, conv_id: int) -> List[ChatMessageORM]:
        def op(s):
            return (
                s.query(ChatMessageORM)
                .filter_by(conversation_id=conv_id)
                .order_by(ChatMessageORM.order_index)
                .all()
            )

        return self._run(op)

    # ---------- Turns ----------

    def add_turns(
        self, conversation_id: int, messages: List[dict]
    ) -> List[ChatTurnORM]:
        """
        Build Q/A turns from a flat list of messages.
        Assumes messages are in chronological order and include DB ids.
        """

        def op(s):
            turns: List[ChatTurnORM] = []
            for i in range(len(messages) - 1):
                u, a = messages[i], messages[i + 1]
                if u.get("role") == "user" and a.get("role") == "assistant":
                    turn = ChatTurnORM(
                        conversation_id=conversation_id,
                        user_message_id=u["id"],
                        assistant_message_id=a["id"],
                    )
                    s.add(turn)
                    turns.append(turn)
            s.flush()
            return turns

        return self._run(op)

    def get_turn_by_id(self, turn_id: int) -> Optional[ChatTurnORM]:
        def op(s):
            return s.get(ChatTurnORM, turn_id)

        return self._run(op)

    def get_turns_for_conversation(self, conv_id: int) -> List[ChatTurnORM]:
        def op(s):
            return (
                s.query(ChatTurnORM)
                .filter_by(conversation_id=conv_id)
                .order_by(ChatTurnORM.id)
                .all()
            )

        return self._run(op)

    # ---------- Admin / Stats ----------

    def purge_all(self, force: bool = False):
        def op(s):
            if force:
                s.execute(text("TRUNCATE chat_turns RESTART IDENTITY CASCADE"))
                s.execute(
                    text("TRUNCATE chat_messages RESTART IDENTITY CASCADE")
                )
                s.execute(
                    text(
                        "TRUNCATE chat_conversations RESTART IDENTITY CASCADE"
                    )
                )
            else:
                s.query(ChatTurnORM).delete()
                s.query(ChatMessageORM).delete()
                s.query(ChatConversationORM).delete()
            return True

        ok = self._run(op)
        if self.logger and ok:
            self.logger.info(
                "[ChatStore] Purged all conversations, messages, and turns"
            )

    def get_top_conversations(
        self, limit: int = 10, by: str = "turns"
    ) -> List[Tuple[ChatConversationORM, int]]:
        def op(s):
            if by == "messages":
                q = (
                    s.query(
                        ChatConversationORM,
                        func.count(ChatMessageORM.id).label("message_count"),
                    )
                    .join(ChatMessageORM)
                    .group_by(ChatConversationORM.id)
                    .order_by(desc("message_count"))
                    .limit(limit)
                )
            else:
                q = (
                    s.query(
                        ChatConversationORM,
                        func.count(ChatTurnORM.id).label("count"),
                    )
                    .join(ChatTurnORM)
                    .group_by(ChatConversationORM.id)
                    .order_by(func.count(ChatTurnORM.id).desc())
                    .limit(limit)
                )
            return [(conv, int(count)) for conv, count in q.all()]

        return self._run(op)

    def get_message_by_id(self, message_id: int) -> Optional[ChatMessageORM]:
        def op(s):
            return s.get(ChatMessageORM, message_id)

        return self._run(op)

    # ---------- Scorable helpers ----------

    def scorable_from_conversation(
        self, conv: ChatConversationORM
    ) -> Scorable:
        """
        Convert a ChatConversationORM into a Scorable object.
        NOTE: if conv.messages is lazy-loaded, ensure it's loaded before the session closes,
        or fetch messages separately and pass them in.
        """
        text_val = "\n".join(
            [
                f"{m.role}: {m.text}"
                for m in getattr(conv, "messages", [])
                if m.text
            ]
        ).strip()
        return Scorable(
            id=str(conv.id),
            text=text_val,
            target_type=TargetType.CONVERSATION,
            meta=conv.to_dict(include_messages=False),
        )

    def scorable_from_message(self, msg: ChatMessageORM) -> Scorable:
        text_val = msg.text or ""
        return Scorable(
            id=str(msg.id),
            text=f"[{msg.role}] {text_val.strip()}",
            target_type=TargetType.CONVERSATION_MESSAGE,
            meta={"conversation_id": msg.conversation_id},
        )

    def scorable_from_turn(self, turn: ChatTurnORM) -> Scorable:
        user_text = (
            turn.user_message.text
            if getattr(turn, "user_message", None)
            else ""
        )
        assistant_text = (
            turn.assistant_message.text
            if getattr(turn, "assistant_message", None)
            else ""
        )
        text_val = (
            f"USER: {user_text.strip()}\nASSISTANT: {assistant_text.strip()}"
        )
        return Scorable(
            id=str(turn.id),
            text=text_val,
            target_type=TargetType.CONVERSATION_TURN,
            meta={"conversation_id": turn.conversation_id},
        )

    def set_turn_star(self, turn_id: int, star: int) -> ChatTurnORM:
        """
        Clamp to [-5, 5], set the star rating, persist, and return the updated turn.
        Uses the _run(op) session pattern.
        """
        star = max(-5, min(5, int(star)))

        def op(s):
            turn = s.get(ChatTurnORM, turn_id)
            if not turn:
                raise ValueError(f"Turn {turn_id} not found")
            turn.star = star
            s.add(turn)
            s.flush()  # ensure PK/state updated before returning
            # Optionally eager-load the messages if your template needs them immediately:
            # _ = turn.user_message, turn.assistant_message
            return turn

        return self._run(op)

    def rated_progress(self, conv_id: int) -> Tuple[int, int]:
        """
        Returns (rated_turns, total_turns) for a conversation.
        'Rated' means star != 0.
        """

        def op(s):
            q = s.query(ChatTurnORM).filter_by(conversation_id=conv_id)
            total = q.count()
            rated = q.filter(ChatTurnORM.star != 0).count()
            return rated, total

        return self._run(op)

    def list_by_turn_count(
        self, *, limit: int = 200, provider: Optional[str] = None
    ) -> List[Tuple[ChatConversationORM, int]]:
        """
        Returns a list of (conversation, turn_count) sorted by turn_count desc.
        If provider is given, filter by provider.
        """

        def op(s):
            q = (
                s.query(
                    ChatConversationORM,
                    func.count(ChatTurnORM.id).label("turns"),
                )
                .outerjoin(
                    ChatTurnORM,
                    ChatTurnORM.conversation_id == ChatConversationORM.id,
                )
                .group_by(ChatConversationORM.id)
            )
            if provider:
                q = q.filter(ChatConversationORM.provider == provider)
            q = q.order_by(desc("turns")).limit(limit)
            return [(conv, int(turns)) for conv, turns in q.all()]

        return self._run(op)

    def get_turns_eager(self, conv_id: int) -> List[ChatTurnORM]:
        def op(s):
            return (
                s.query(ChatTurnORM)
                .options(
                    selectinload(ChatTurnORM.user_message),
                    selectinload(ChatTurnORM.assistant_message),
                )
                .filter(ChatTurnORM.conversation_id == conv_id)
                .order_by(ChatTurnORM.id)
                .all()
            )
        return self._run(op)

    def set_turn_ner(self, turn_id: int, ner: list[dict]) -> None:
        def op(s):
            t = s.get(ChatTurnORM, turn_id)
            if not t:
                raise ValueError(f"Turn {turn_id} not found")
            t.ner = ner
            s.add(t)
        return self._run(op)

    def set_turn_domains(self, turn_id: int, domains: list[dict]) -> None:
        def op(s):
            t = s.get(ChatTurnORM, turn_id)
            if not t:
                raise ValueError(f"Turn {turn_id} not found")
            t.domains = domains
            s.add(t)
        return self._run(op)

    def scoring_batch(
        self, conv_id: int, *, only_unrated: bool = False, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int, int]:
        def op(s):
            base_q = s.query(ChatTurnORM).filter_by(conversation_id=conv_id)
            total = base_q.count()
            rated = base_q.filter(ChatTurnORM.star != 0).count()

            q = (
                s.query(ChatTurnORM)
                 .options(
                     selectinload(ChatTurnORM.user_message),
                     selectinload(ChatTurnORM.assistant_message),
                 )
                 .filter(ChatTurnORM.conversation_id == conv_id)
                 .order_by(ChatTurnORM.id)
            )
            if only_unrated:
                q = q.filter(ChatTurnORM.star == 0)

            turns = q.offset(offset).limit(limit).all()

            out = []
            for t in turns:
                out.append({
                    "id": t.id,
                    "order_index": getattr(t, "order_index", 0),
                    "star": int(getattr(t, "star", 0) or 0),
                    "user_text": t.user_message.text if t.user_message else "—",
                    "assistant_text": t.assistant_message.text if t.assistant_message else "—",
                    "ner": t.ner or [],                 # NEW
                    "domains": t.domains or [],         # NEW
                })
            return out, rated, total
        return self._run(op)

    def get_turn_texts_for_conversation(
        self,
        conv_id: int,
        *,
        limit: int | None = None,
        offset: int = 0,
        order_by_id: bool = True,
        only_missing: str | None = None,   # <-- NEW
    ) -> list[dict]:
        """
        Returns list of dicts with pre-fetched user/assistant texts (no lazy loads).
        Shape: [{"id": int, "order_index": int|None, "user_text": str, "assistant_text": str}]

        only_missing:
          - "ner"      → only turns where ner is NULL
          - "domains"  → only turns where domains is NULL
          - None/other → no filter
        """
        def op(s):
            U = aliased(ChatMessageORM)
            A = aliased(ChatMessageORM)

            q = (
                s.query(
                    ChatTurnORM.id.label("id"),
                    ChatTurnORM.order_index.label("order_index"),
                    U.text.label("user_text"),
                    A.text.label("assistant_text"),
                )
                .join(U, ChatTurnORM.user_message_id == U.id)
                .join(A, ChatTurnORM.assistant_message_id == A.id)
                .filter(ChatTurnORM.conversation_id == conv_id)
            )

            # NEW: filter for missing annotations
            if only_missing == "ner":
                q = q.filter(ChatTurnORM.ner.is_(None))
            elif only_missing == "domains":
                q = q.filter(ChatTurnORM.domains.is_(None))

            if order_by_id:
                q = q.order_by(ChatTurnORM.id.asc())
            if offset:
                q = q.offset(offset)
            if limit:
                q = q.limit(limit)

            rows = q.all()
            return [
                {
                    "id": r.id,
                    "order_index": r.order_index,
                    "user_text": r.user_text or "",
                    "assistant_text": r.assistant_text or "",
                }
                for r in rows
            ]
        return self._run(op)

    def get_turn_domains(self, turn_id: int) -> list[dict]:
        def op(s):
            t = s.get(ChatTurnORM, turn_id)
            return t.domains or [] if t else []
        return self._run(op)

    def get_turn_ner(self, turn_id: int) -> list[dict]:
        def op(s):
            t = s.get(ChatTurnORM, turn_id)
            return t.ner or [] if t else []
        return self._run(op)

    def get_conversation_dict(
        self,
        conv_id: int,
        *,
        include_messages: bool = True,
        include_turns: bool = True,
        include_turn_message_texts: bool = True,
    ) -> dict | None:
        """
        Return a fully materialized dict for a conversation.
        All needed relationships are eagerly loaded *inside* the session
        so no lazy loads happen after return.
        """
        def op(s):
            q = s.query(ChatConversationORM)

            # Eager-load messages (simple)
            if include_messages:
                q = q.options(selectinload(ChatConversationORM.messages))

            # Eager-load turns (+ nested messages for user/assistant)
            if include_turns:
                q = q.options(
                    selectinload(ChatConversationORM.turns)
                    .selectinload(ChatTurnORM.user_message),
                    selectinload(ChatConversationORM.turns)
                    .selectinload(ChatTurnORM.assistant_message),
                ) 

            conv = q.filter(ChatConversationORM.id == conv_id).one_or_none()
            if not conv:
                return None

            # Build dict *before* session closes
            d = conv.to_dict(
                include_messages=include_messages,
                include_turns=include_turns,
            )

            # Optional: flatten user/assistant texts onto turn dicts for templates
            if include_turns and include_turn_message_texts and d.get("turns"):
                for t in d["turns"]:
                    um = t.get("user_message") or {}
                    am = t.get("assistant_message") or {}
                    t["user_text"] = um.get("text") or "—"
                    t["assistant_text"] = am.get("text") or "—"

            return d

        return self._run(op)


    def list_turns(
        self,
        *,
        min_star: Optional[int] = None,
        max_star: Optional[int] = None,
        goal_id: Optional[str] = None,
        casebook_id: Optional[int] = None,
        domain: Optional[str] = None,
        has_entities: bool = True,
        min_text_len: int = 1,
        limit: int = 10_000,
        order_desc: bool = True,
    ) -> List[ChatTurnORM]:
        """
        Fetch chat turns with common filters. All SQL and dialect-specific handling
        happens inside the session via self._run(...). If the DB cannot enforce
        'entities non-empty' (no JSONB), we Python-filter after fetching.
        """
        def op(s):
            q: Query = (
                s.query(ChatTurnORM)
                .filter(ChatTurnORM.text.isnot(None), ChatTurnORM.text != "")
            )

            if min_text_len and min_text_len > 1:
                try:
                    q = q.filter(func.length(ChatTurnORM.text) >= int(min_text_len))
                except Exception:
                    # some dialects don’t support length(); skip quietly
                    pass

            db_filtered = False
            if has_entities:
                q = q.filter(ChatTurnORM.entities.isnot(None))
                # Try DB-side JSON length check; returns (q, db_filtered)
                try:
                    q, db_filtered = self._apply_has_entities(q)
                except Exception:
                    # if store lacks helper for some reason, fall back to best effort
                    try:
                        q = q.filter(func.jsonb_array_length(ChatTurnORM.entities) > 0)
                        db_filtered = True
                    except Exception:
                        db_filtered = False

            if min_star is not None:
                q = q.filter(ChatTurnORM.star >= int(min_star))
            if max_star is not None:
                q = q.filter(ChatTurnORM.star <= int(max_star))
            if goal_id is not None:
                q = q.filter(ChatTurnORM.goal_id == goal_id)
            if casebook_id is not None:
                q = q.filter(ChatTurnORM.casebook_id == casebook_id)
            if domain is not None:
                q = q.filter(ChatTurnORM.domain == domain)

            q = q.order_by(ChatTurnORM.id.desc() if order_desc else ChatTurnORM.id.asc())
            rows = q.limit(int(limit)).all()

            # Python-side enforcement if DB couldn't filter entities non-empty
            if has_entities and not db_filtered:
                def _non_empty_entities(val) -> bool:
                    if val is None:
                        return False
                    if isinstance(val, str):
                        # allow both JSON string and literal '[]'
                        val = val.strip()
                        if val == "" or val == "[]":
                            return False
                        try:
                            import json
                            arr = json.loads(val)
                            return isinstance(arr, list) and len(arr) > 0
                        except Exception:
                            # if not JSON, treat as truthy string
                            return True
                    # ORM may materialize as list already
                    try:
                        return isinstance(val, list) and len(val) > 0
                    except Exception:
                        return False

                rows = [r for r in rows if _non_empty_entities(getattr(r, "entities", None))]

            return rows

        return self._run(op)


    def count_turns(
        self,
        *,
        min_star: Optional[int] = None,
        max_star: Optional[int] = None,
        goal_id: Optional[str] = None,
        casebook_id: Optional[int] = None,
        domain: Optional[str] = None,
        has_entities: bool = True,
        min_text_len: int = 1,
    ) -> int:
        """
        Count turns matching filters (mirrors list_turns). Executes entirely within
        the session via self._run(...). If the DB can't enforce 'entities non-empty',
        we do a lightweight Python-side count.
        """
        def op(s):
            q: Query = (
                s.query(ChatTurnORM.id, ChatTurnORM.entities)
                .filter(ChatTurnORM.text.isnot(None), ChatTurnORM.text != "")
            )

            if min_text_len and min_text_len > 1:
                try:
                    q = q.filter(func.length(ChatTurnORM.text) >= int(min_text_len))
                except Exception:
                    pass

            db_filtered = False
            if has_entities:
                q = q.filter(ChatTurnORM.entities.isnot(None))
                try:
                    # Try to push JSONB length to DB
                    q = q.filter(func.jsonb_array_length(ChatTurnORM.entities) > 0)
                    db_filtered = True
                except Exception:
                    db_filtered = False  # we'll count client-side below

            if min_star is not None:
                q = q.filter(ChatTurnORM.star >= int(min_star))
            if max_star is not None:
                q = q.filter(ChatTurnORM.star <= int(max_star))
            if goal_id is not None:
                q = q.filter(ChatTurnORM.goal_id == goal_id)
            if casebook_id is not None:
                q = q.filter(ChatTurnORM.casebook_id == casebook_id)
            if domain is not None:
                q = q.filter(ChatTurnORM.domain == domain)

            if db_filtered or not has_entities:
                # DB already enforced non-empty entities, or we don't require it
                return q.count()

            # Python-side count for non-JSONB dialects
            rows = q.all()

            def _non_empty_entities(val) -> bool:
                if val is None:
                    return False
                if isinstance(val, str):
                    val = val.strip()
                    if val == "" or val == "[]":
                        return False
                    try:
                        import json
                        arr = json.loads(val)
                        return isinstance(arr, list) and len(arr) > 0
                    except Exception:
                        return True
                try:
                    return isinstance(val, list) and len(val) > 0
                except Exception:
                    return False

            return sum(1 for (_id, ents) in rows if _non_empty_entities(ents))

        return self._run(op)


    def list_turns_with_texts(
        self,
        *,
        min_star: Optional[int] = None,
        max_star: Optional[int] = None,
        require_assistant_text: bool = True,
        require_nonempty_ner: bool = True,
        min_assistant_len: int = 1,
        limit: int = 10000,
        order_desc: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Returns lightweight dict rows for turns with pre-fetched assistant/user texts:
        { id, conversation_id, order_index, star, user_text, assistant_text, ner, domains }

        All SQL happens inside the session; no lazy loads in the caller.
        """
        def op(s):
            U = aliased(ChatMessageORM)
            A = aliased(ChatMessageORM)

            q = (
                s.query(
                    ChatTurnORM.id.label("id"),
                    ChatTurnORM.conversation_id.label("conversation_id"),
                    ChatTurnORM.order_index.label("order_index"),
                    ChatTurnORM.star.label("star"),
                    ChatTurnORM.ner.label("ner"),
                    ChatTurnORM.domains.label("domains"),
                    U.text.label("user_text"),
                    A.text.label("assistant_text"),
                )
                .join(U, ChatTurnORM.user_message_id == U.id)
                .join(A, ChatTurnORM.assistant_message_id == A.id)
            )

            if min_star is not None:
                q = q.filter(ChatTurnORM.star >= int(min_star))
            if max_star is not None:
                q = q.filter(ChatTurnORM.star <= int(max_star))

            if require_assistant_text:
                q = q.filter(A.text.isnot(None), A.text != "")
                if min_assistant_len and min_assistant_len > 1:
                    try:
                        q = q.filter(func.length(A.text) >= int(min_assistant_len))
                    except Exception:
                        pass

            if require_nonempty_ner:
                # Prefer JSON length on Postgres
                try:
                    q = q.filter(ChatTurnORM.ner.isnot(None))
                    q = q.filter(func.jsonb_array_length(ChatTurnORM.ner) > 0)
                except Exception:
                    # Fallback: at least not NULL; client can recheck if needed
                    q = q.filter(ChatTurnORM.ner.isnot(None))

            q = q.order_by(ChatTurnORM.id.desc() if order_desc else ChatTurnORM.id.asc())
            q = q.limit(int(limit))
            rows = q.all()

            out = []
            for r in rows:
                out.append({
                    "id": r.id,
                    "conversation_id": r.conversation_id,
                    "order_index": int(r.order_index or 0),
                    "star": int(r.star or 0),
                    "user_text": r.user_text or "",
                    "assistant_text": r.assistant_text or "",
                    "ner": r.ner or [],
                    "domains": r.domains or [],
                })
            return out

        return self._run(op)
