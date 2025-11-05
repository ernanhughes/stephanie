"""
Chat Store Module

This module provides the data access layer for chat conversations, messages, and turns
in the Stephanie system. It handles all database operations related to chat data storage,
retrieval, and management using SQLAlchemy ORM.

Key Features:
- Complete CRUD operations for conversations, messages, and turns
- Support for star ratings and annotations (NER, domains) on conversation turns
- Efficient querying with filtering, sorting, and pagination
- Integration with the scoring system through Scorable objects
- Support for batch operations and statistics collection

The store uses a session-based pattern with automatic transaction management
through the BaseSQLAlchemyStore parent class.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc, func, or_, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Query, aliased, selectinload

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.chat import (ChatConversationORM, ChatMessageORM,
                                   ChatTurnORM)
from stephanie.scoring.scorable import Scorable, ScorableType

log = logging.getLogger(__name__)

class ChatStore(BaseSQLAlchemyStore):
    """
    Data access layer for chat conversations, messages, and turns.
    
    This class provides methods to store, retrieve, and manage chat data
    with support for annotations, ratings, and efficient querying.
    """
    # ORM model for the base store operations
    orm_model = ChatConversationORM
    
    # Default ordering for queries
    default_order_by = "created_at"

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "chats"

    # ---------- Conversations ----------

    def add_conversation(self, data: dict) -> ChatConversationORM:
        """
        Create and persist a new conversation.
        
        Args:
            data: Dictionary containing conversation attributes
            
        Returns:
            The newly created ChatConversationORM object
        """
        def op(s):
            conv = ChatConversationORM(**data)
            s.add(conv)
            s.flush()
            return conv

        return self._run(op)

    def get_all(self, limit: int = 100) -> List[ChatConversationORM]:
        """
        Retrieve all conversations with a limit.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation objects ordered by creation date (descending)
        """
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
        """
        Check if a conversation with the given file hash already exists.
        
        Args:
            file_hash: SHA256 hash of the source file
            
        Returns:
            True if a conversation with this hash exists, False otherwise
        """
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
        """
        Retrieve a specific conversation by ID.
        
        Args:
            conv_id: The conversation ID to retrieve
            
        Returns:
            The conversation object or None if not found
        """
        def op(s):
            return s.get(ChatConversationORM, conv_id)

        return self._run(op)

    # ---------- Messages ----------

    def add_messages(
        self, conv_id: int, messages: List[dict]
    ) -> List[ChatMessageORM]:
        """
        Add multiple messages to a conversation.
        
        Args:
            conv_id: ID of the conversation to add messages to
            messages: List of message dictionaries with role, text, and metadata
            
        Returns:
            List of created message objects
        """
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
        """
        Retrieve all messages for a conversation in order.
        
        Args:
            conv_id: ID of the conversation
            
        Returns:
            List of message objects ordered by their index
        """
        def op(s):
            return (
                s.query(ChatMessageORM)
                .filter_by(conversation_id=conv_id)
                .order_by(ChatMessageORM.order_index)
                .all()
            )

        return self._run(op)

    # ---------- Turns ----------
    def add_turns(self, conversation_id: int, messages: List[dict]) -> List[ChatTurnORM]:
        """
        Build Q/A turns from a flat list of messages.
        Assumes messages are chronological and include DB ids & order_index.
        
        Args:
            conversation_id: ID of the conversation
            messages: List of message dictionaries with role and ID
            
        Returns:
            List of created turn objects
        """
        def op(s):
            # find current max for this conversation so appends are monotonic
            cur_max = (
                s.query(func.coalesce(func.max(ChatTurnORM.order_index), -1))
                .filter(ChatTurnORM.conversation_id == conversation_id)
                .scalar()
            )
            next_ix = int(cur_max) + 1

            turns: List[ChatTurnORM] = []
            for i in range(len(messages) - 1):
                u, a = messages[i], messages[i + 1]
                if u.get("role") == "user" and a.get("role") == "assistant":
                    # prefer assistant message order if present, else use running index
                    a_ix = a.get("order_index")
                    oi = int(a_ix) if isinstance(a_ix, int) else next_ix
                    turn = ChatTurnORM(
                        conversation_id=conversation_id,
                        user_message_id=u["id"],
                        assistant_message_id=a["id"],
                        order_index=oi,
                    )
                    s.add(turn)
                    turns.append(turn)
                    next_ix = max(next_ix + 1, oi + 1)

            s.flush()
            return turns
        return self._run(op)

    def get_turn_by_id(self, turn_id: int) -> Optional[ChatTurnORM]:
        """
        Retrieve a specific turn by ID.
        
        Args:
            turn_id: The turn ID to retrieve
            
        Returns:
            The turn object or None if not found
        """
        def op(s):
            return s.get(ChatTurnORM, turn_id)

        return self._run(op)

    def get_turns_for_conversation(self, conv_id: int) -> List[ChatTurnORM]:
        """
        Retrieve all turns for a conversation.

        Args:
            conv_id: ID of the conversation

        Returns:
            List of turn objects ordered by ID
        """
        def op(s):
            try:
                turns = (
                    s.query(ChatTurnORM)
                    .options(
                        selectinload(ChatTurnORM.user_message),
                        selectinload(ChatTurnORM.assistant_message),
                    )
                    .filter(ChatTurnORM.conversation_id == conv_id)
                    .order_by(ChatTurnORM.id)
                    .all()
                )
                if not turns:
                    log.warning(f"No turns found for conversation {conv_id}")
                return turns
            except Exception as e:
                log.error(f"Failed to fetch turns for conversation {conv_id}: {e}", exc_info=True)
                raise

        result = self._run(op)
        if result is None:
            log.error(f"_run() returned None for conversation {conv_id}")
            return []
        return result

    # ---------- Admin / Stats ----------

    def purge_all(self, force: bool = False):
        """
        Delete all chat data from the database.
        
        Args:
            force: If True, uses TRUNCATE for faster deletion (requires privileges)
            
        Returns:
            True if operation was successful
        """
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
        """
        Retrieve conversations with the most turns or messages.
        
        Args:
            limit: Maximum number of conversations to return
            by: Metric to sort by - "turns" or "messages"
            
        Returns:
            List of tuples (conversation, count) ordered by count descending
        """
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
                    .options(
                         selectinload(ChatTurnORM.user_message),
                         selectinload(ChatTurnORM.assistant_message),
                     )
                    .join(ChatTurnORM)
                    .group_by(ChatConversationORM.id)
                    .order_by(func.count(ChatTurnORM.id).desc())
                    .limit(limit)
                )
            return [(conv, int(count)) for conv, count in q.all()]

        return self._run(op)

    def get_message_by_id(self, message_id: int) -> Optional[ChatMessageORM]:
        """
        Retrieve a specific message by ID.
        
        Args:
            message_id: The message ID to retrieve
            
        Returns:
            The message object or None if not found
        """
        def op(s):
            return s.get(ChatMessageORM, message_id)

        return self._run(op)

    # ---------- Scorable helpers ----------

    def scorable_from_conversation(
        self, conv: ChatConversationORM
    ) -> Scorable:
        """
        Convert a ChatConversationORM into a Scorable object.
        
        Note: If conv.messages is lazy-loaded, ensure it's loaded before the session closes,
        or fetch messages separately and pass them in.
        
        Args:
            conv: The conversation object to convert
            
        Returns:
            A Scorable object representing the conversation
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
            target_type=ScorableType.CONVERSATION,
            meta=conv.to_dict(include_messages=False),
        )

    def scorable_from_message(self, msg: ChatMessageORM) -> Scorable:
        """
        Convert a ChatMessageORM into a Scorable object.
        
        Args:
            msg: The message object to convert
            
        Returns:
            A Scorable object representing the message
        """
        text_val = msg.text or ""
        return Scorable(
            id=str(msg.id),
            text=f"[{msg.role}] {text_val.strip()}",
            target_type=ScorableType.CONVERSATION_MESSAGE,
            meta={"conversation_id": msg.conversation_id},
        )

    def scorable_from_turn(self, turn: ChatTurnORM) -> Scorable:
        """
        Convert a ChatTurnORM into a Scorable object.
        
        Args:
            turn: The turn object to convert
            
        Returns:
            A Scorable object representing the turn
        """
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
            target_type=ScorableType.CONVERSATION_TURN,
            meta={"conversation_id": turn.conversation_id},
        )

    def set_turn_star(self, turn_id: int, star: int) -> ChatTurnORM:
        """
        Set the star rating for a turn (clamped to [-5, 5]).
        
        Args:
            turn_id: ID of the turn to update
            star: Star rating value between -5 and 5
            
        Returns:
            The updated turn object
            
        Raises:
            ValueError: If the turn is not found
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
        Get rating progress for a conversation.
        
        Args:
            conv_id: ID of the conversation
            
        Returns:
            Tuple of (rated_turns, total_turns) where rated means star != 0
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
        List conversations sorted by turn count (descending).
        
        Args:
            limit: Maximum number of conversations to return
            provider: Optional provider filter
            
        Returns:
            List of tuples (conversation, turn_count) sorted by turn count
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
        """
        Retrieve turns for a conversation with messages eagerly loaded.
        
        Args:
            conv_id: ID of the conversation
            
        Returns:
            List of turn objects with user and assistant messages loaded
        """
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
        """
        Set NER (Named Entity Recognition) annotations for a turn.
        
        Args:
            turn_id: ID of the turn to update
            ner: List of NER annotation dictionaries
            
        Raises:
            ValueError: If the turn is not found
        """
        def op(s):
            t = s.get(ChatTurnORM, turn_id)
            if not t:
                raise ValueError(f"Turn {turn_id} not found")
            t.ner = ner
            s.add(t)
        return self._run(op)

    def set_turn_domains(self, turn_id: int, domains: list[dict]) -> None:
        """
        Set domain annotations for a turn.
        
        Args:
            turn_id: ID of the turn to update
            domains: List of domain annotation dictionaries
            
        Raises:
            ValueError: If the turn is not found
        """
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
        """
        Get a batch of turns for scoring with progress information.
        
        Args:
            conv_id: ID of the conversation
            only_unrated: If True, only return unrated turns (star == 0)
            limit: Maximum number of turns to return
            offset: Offset for pagination
            
        Returns:
            Tuple of (turns_list, rated_count, total_count)
        """
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
                    "ner": t.ner or [],               
                    "domains": t.domains or [],       
                    "ai_knowledge_score": t.ai_knowledge_score, 
                    "ai_knowledge_score_norm": (
                        None if t.ai_knowledge_score is None
                        else max(0.0, min(1.0, float(t.ai_knowledge_score)/100.0))
                    ),
                    "ai_knowledge_rationale": t.ai_knowledge_rationale or "",

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
        Get turn texts with optional filtering for missing annotations.
        
        Args:
            conv_id: ID of the conversation
            limit: Maximum number of turns to return
            offset: Offset for pagination
            order_by_id: If True, order by turn ID
            only_missing: Filter for missing annotations - "ner" or "domains"
            
        Returns:
            List of dictionaries with turn texts and metadata
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
        """
        Get domain annotations for a turn.
        
        Args:
            turn_id: ID of the turn
            
        Returns:
            List of domain annotations or empty list if not found
        """
        def op(s):
            t = s.get(ChatTurnORM, turn_id)
            return t.domains or [] if t else []
        return self._run(op)

    def get_turn_ner(self, turn_id: int) -> list[dict]:
        """
        Get NER annotations for a turn.
        
        Args:
            turn_id: ID of the turn
            
        Returns:
            List of NER annotations or empty list if not found
        """
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
        Get a fully materialized dictionary representation of a conversation.
        
        Args:
            conv_id: ID of the conversation
            include_messages: Whether to include messages in the result
            include_turns: Whether to include turns in the result
            include_turn_message_texts: Whether to flatten message texts onto turns
            
        Returns:
            Dictionary representation of the conversation or None if not found
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
        goal_id: Optional[str] = None,        # not on ChatTurnORM; kept for API compat (no-op)
        casebook_id: Optional[int] = None,    # not on ChatTurnORM; kept for API compat (no-op)
        domain: Optional[str] = None,
        has_entities: bool = True,
        min_text_len: int = 1,
        limit: int = 10_000,
        order_desc: bool = True,
    ) -> List[ChatTurnORM]:
        """
        Fetch chat turns with filtering options. Text length applies to the *assistant* message.
        Notes:
        - `goal_id` and `casebook_id` are ignored here (not present on ChatTurnORM).
        - `domain` matches if ChatTurnORM.domains includes {"domain": <domain>}.
        """
        warned_missing = {"goal": False, "casebook": False}

        def op(s):
            # Base: join assistant message so we can filter by its text
            q: Query = (
                s.query(ChatTurnORM)
                .options(
                     selectinload(ChatTurnORM.user_message),
                     selectinload(ChatTurnORM.assistant_message),
                )
               .join(ChatMessageORM, ChatTurnORM.assistant_message)  # relationship join
            )

            # Assistant text not null/empty
            q = q.filter(
                ChatMessageORM.text.isnot(None),
                ChatMessageORM.text != ""
            )

            # Min text length (assistant)
            if min_text_len and min_text_len > 1:
                try:
                    q = q.filter(func.length(ChatMessageORM.text) >= int(min_text_len))
                except Exception:
                    # Some dialects may lack length(); skip quietly
                    pass

            # Entities filter
            db_filtered_entities = False
            if has_entities:
                q = q.filter(ChatTurnORM.ner.isnot(None)) 
                try:
                    # If PG + JSONB available, enforce array length > 0 in DB
                    if s.bind.dialect.name == "postgresql": 
                        q = q.filter(func.jsonb_array_length(ChatTurnORM.ner.cast(JSONB)) > 0)
                        db_filtered_entities = True
                    else:
                        # Some SQLite builds expose json_array_length()
                        q = q.filter(func.json_array_length(ChatTurnORM.ner) > 0)
                        db_filtered_entities = True
                except Exception:
                    db_filtered_entities = False  # fallback after fetch

            # Star range
            if min_star is not None:
                q = q.filter(ChatTurnORM.star >= int(min_star))
            if max_star is not None:
                q = q.filter(ChatTurnORM.star <= int(max_star))

            # Domain filter
            db_filtered_domain = False
            if domain:
                try:
                    if s.bind.dialect.name == "postgresql":
                        # domains: JSON array of objects; match any with {"domain": <domain>}
                        q = q.filter(
                            ChatTurnORM.domains.cast(JSONB).contains([{"domain": str(domain)}])
                        )
                        db_filtered_domain = True
                    else:
                        # Some SQLite builds provide json_each/json_extract, but portability is spotty.
                        # We’ll do a Python-side filter after fetch.
                        db_filtered_domain = False
                except Exception:
                    db_filtered_domain = False

            # Unsupported filters on this model (log once)
            if goal_id is not None and not warned_missing["goal"]:
                try:
                    self.logger and log.warning("list_turns: goal_id filter ignored (not on ChatTurnORM)")
                finally:
                    warned_missing["goal"] = True
            if casebook_id is not None and not warned_missing["casebook"]:
                try:
                    self.logger and log.warning("list_turns: casebook_id filter ignored (not on ChatTurnORM)")
                finally:
                    warned_missing["casebook"] = True

            # Ordering & fetch
            q = q.order_by(ChatTurnORM.id.desc() if order_desc else ChatTurnORM.id.asc())
            rows = q.limit(int(limit)).all()

            # Python-side fallbacks

            # has_entities fallback
            if has_entities and not db_filtered_entities:
                def _non_empty_entities(val) -> bool:
                    if val is None:
                        return False
                    if isinstance(val, str):
                        v = val.strip()
                        if not v or v == "[]":
                            return False
                        try:
                            import json
                            arr = json.loads(v)
                            return isinstance(arr, list) and len(arr) > 0
                        except Exception:
                            # If it’s a non-empty string but not JSON, treat as present
                            return True
                    return isinstance(val, list) and len(val) > 0
                rows = [r for r in rows if _non_empty_entities(getattr(r, "ner", None))]

            # domain fallback: keep row if any {'domain': <domain>} in domains
            if domain and not db_filtered_domain:
                dnorm = str(domain).strip().lower()
                def _has_domain(dom):
                    try:
                        if dom is None:
                            return False
                        if isinstance(dom, str):
                            import json
                            dom = json.loads(dom)
                        if isinstance(dom, list):
                            for item in dom:
                                name = (item or {}).get("domain")
                                if isinstance(name, str) and name.strip().lower() == dnorm:
                                    return True
                    except Exception:
                        return False
                    return False
                rows = [r for r in rows if _has_domain(getattr(r, "domains", None))]

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
        Count turns matching filtering criteria.
        
        Args:
            min_star: Minimum star rating
            max_star: Maximum star rating
            goal_id: Filter by goal ID
            casebook_id: Filter by casebook ID
            domain: Filter by domain
            has_entities: Only include turns with entities
            min_text_len: Minimum text length
            
        Returns:
            Count of turns matching the criteria
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
        min_ai_score: Optional[float] = None,
        max_ai_score: Optional[float] = None,
        require_assistant_text: bool = True,
        require_nonempty_ner: bool = True,
        min_assistant_len: int = 1,
        limit: int = 10000,
        order_desc: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get turns with pre-fetched text content and metadata.

        Args:
            min_star: Minimum star rating
            max_star: Maximum star rating
            min_ai_score: Minimum AI knowledge score (0-100)
            max_ai_score: Maximum AI knowledge score (0-100)
            require_assistant_text: Only include turns with assistant text
            require_nonempty_ner: Only include turns with NER annotations
            min_assistant_len: Minimum assistant text length
            limit: Maximum number of turns to return
            order_desc: Order by ID descending if True

        Returns:
            List of dictionaries with turn data including texts and metadata
        """

        def op(s):
            U = aliased(ChatMessageORM)
            A = aliased(ChatMessageORM)
            C = aliased(ChatConversationORM)

            q = (
                s.query(
                    ChatTurnORM.id.label("id"),
                    ChatTurnORM.conversation_id.label("conversation_id"),
                    ChatTurnORM.order_index.label("order_index"),
                    ChatTurnORM.star.label("star"),
                    ChatTurnORM.ner.label("ner"),
                    ChatTurnORM.domains.label("domains"),
                    ChatTurnORM.ai_knowledge_score.label("ai_score"),
                    ChatTurnORM.ai_knowledge_rationale.label("ai_rationale"),
                    ChatTurnORM.user_message_id.label("user_message_id"),   
                    ChatTurnORM.assistant_message_id.label("assistant_message_id"), 
                    U.text.label("user_text"),
                    A.text.label("assistant_text"),
                    C.title.label("goal_text"),  # ← conversation title as goal text
                )
                .join(U, ChatTurnORM.user_message_id == U.id)
                .join(A, ChatTurnORM.assistant_message_id == A.id)
                .join(C, ChatTurnORM.conversation_id == C.id)
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
                try:
                    q = q.filter(ChatTurnORM.ner.isnot(None))
                    q = q.filter(func.jsonb_array_length(ChatTurnORM.ner) > 0)
                except Exception:
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
                    "user_message_id": r.user_message_id,           # ✅ added
                    "assistant_message_id": r.assistant_message_id, # ✅ added
                    "user_text": r.user_text or "",
                    "assistant_text": r.assistant_text or "",
                    "ner": r.ner or [],
                    "domains": r.domains or [],
                    "goal_text": r.goal_text or "",
                    "ai_score": r.ai_score,
                    "ai_rationale": r.ai_rationale or "",
                })
            return out

        return self._run(op)
    

    def set_turn_ai_eval(
        self,
        turn_id: int,
        score: int,
        rationale: str
    ) -> ChatTurnORM:
        """
        Upsert the AI knowledge evaluation on a turn.
        - score: 0..100 (or None to clear)
        - rationale: free text (or None to clear)
        """
        def op(s):
            t = s.get(ChatTurnORM, turn_id)
            if not t:
                raise ValueError(f"Turn {turn_id} not found")
            if score is not None:
                if not (0 <= int(score) <= 100):
                    raise ValueError("ai_knowledge_score must be in 0..100")
                t.ai_knowledge_score = int(score)
            else:
                t.ai_knowledge_score = None
            t.ai_knowledge_rationale = (rationale or None)
            s.add(t)
            s.flush()
            return t
        return self._run(op)

    def get_turn_ai_eval(self, turn_id: int) -> dict:
        """Return {'score': int|None, 'rationale': str|None} for a turn."""
        def op(s):
            t = s.get(ChatTurnORM, turn_id)
            if not t:
                raise ValueError(f"Turn {turn_id} not found")
            return {
                "score": t.ai_knowledge_score,
                "rationale": t.ai_knowledge_rationale,
            }
        return self._run(op)


    def list_turns_for_conversation_with_texts(
        self,
        conversation_id: int,
        *,
        min_star: Optional[int] = None,
        max_star: Optional[int] = None,
        min_ai_score: Optional[float] = None,   # 0..100
        max_ai_score: Optional[float] = None,   # 0..100
        require_assistant_text: bool = True,
        require_nonempty_ner: bool = True,
        min_assistant_len: int = 1,
        limit: int = 10_000,
        order_desc: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        List turns (with texts & metadata) for a single conversation.

        Args:
            conversation_id: Conversation ID to scope results.
            min_star/max_star:   Human star filters.
            min_ai_score/max_ai_score: AI knowledge score filters (0..100).
            require_assistant_text: Require assistant text to be non-empty.
            require_nonempty_ner: Require non-empty NER list.
            min_assistant_len: Minimum assistant text length.
            limit: Max rows to return.
            order_desc: Sort by ChatTurnORM.id desc/asc.

        Returns:
            List[dict] with keys:
                id, conversation_id, order_index, star, user_text, assistant_text,
                ner, domains, goal_text, ai_score, ai_rationale
        """
        def op(s):
            U = aliased(ChatMessageORM)
            A = aliased(ChatMessageORM)
            C = aliased(ChatConversationORM)

            q: Query = (
                s.query(
                    ChatTurnORM.id.label("id"),
                    ChatTurnORM.conversation_id.label("conversation_id"),
                    ChatTurnORM.order_index.label("order_index"),
                    ChatTurnORM.star.label("star"),
                    ChatTurnORM.ner.label("ner"),
                    ChatTurnORM.domains.label("domains"),
                    ChatTurnORM.ai_knowledge_score.label("ai_score"),
                    ChatTurnORM.ai_knowledge_rationale.label("ai_rationale"),
                    U.text.label("user_text"),
                    A.text.label("assistant_text"),
                    C.title.label("goal_text"),  # conversation title as goal
                )
                .join(U, ChatTurnORM.user_message_id == U.id)
                .join(A, ChatTurnORM.assistant_message_id == A.id)
                .join(C, ChatTurnORM.conversation_id == C.id)
                .filter(ChatTurnORM.conversation_id == int(conversation_id))
            )

            # Human star filters
            if min_star is not None:
                q = q.filter(ChatTurnORM.star >= int(min_star))
            if max_star is not None:
                q = q.filter(ChatTurnORM.star <= int(max_star))

            # AI score filters (0..100)
            if min_ai_score is not None:
                q = q.filter(ChatTurnORM.ai_knowledge_score >= float(min_ai_score))
            if max_ai_score is not None:
                q = q.filter(ChatTurnORM.ai_knowledge_score <= float(max_ai_score))

            # Assistant text constraints
            if require_assistant_text:
                q = q.filter(A.text.isnot(None), A.text != "")
                if min_assistant_len and min_assistant_len > 1:
                    try:
                        q = q.filter(func.length(A.text) >= int(min_assistant_len))
                    except Exception:
                        # some dialects lack length(); ignore length filter
                        pass

            # NER non-empty (DB-side when possible)
            db_filtered = False
            if require_nonempty_ner:
                q = q.filter(ChatTurnORM.ner.isnot(None))
                try:
                    q = q.filter(func.jsonb_array_length(ChatTurnORM.ner) > 0)
                    db_filtered = True
                except Exception:
                    # fallback to Python-side below
                    db_filtered = False

            # Order & limit
            q = q.order_by(ChatTurnORM.id.desc() if order_desc else ChatTurnORM.id.asc())
            rows = q.limit(int(limit)).all()

            # Python-side NER non-empty enforcement if DB couldn't
            def _non_empty_ner(val) -> bool:
                if val is None:
                    return False
                if isinstance(val, list):
                    return len(val) > 0
                if isinstance(val, str):
                    s = val.strip()
                    if s == "" or s == "[]":
                        return False
                    try:
                        import json as _json
                        arr = _json.loads(s)
                        return isinstance(arr, list) and len(arr) > 0
                    except Exception:
                        # if it's some other string format, treat as present
                        return True
                return False

            out: List[Dict[str, Any]] = []
            for r in rows:
                if require_nonempty_ner and not db_filtered and not _non_empty_ner(r.ner):
                    continue
                out.append({
                    "id": r.id,
                    "conversation_id": r.conversation_id,
                    "order_index": int(r.order_index or 0),
                    "star": int(r.star or 0),
                    "user_text": r.user_text or "",
                    "assistant_text": r.assistant_text or "",
                    "ner": r.ner or [],
                    "domains": r.domains or [],
                    "goal_text": r.goal_text or "",
                    "ai_score": r.ai_score,
                    "ai_rationale": r.ai_rationale or "",
                })
            return out

        return self._run(op)

    def list_turns_by_ids_with_texts(self, ids: List[int]) -> List[Dict]:
        """
        Snapshot turns by ids. Returns SAME fields as list_turns_for_conversation_with_texts.
        """
        if not ids:
            return []
        def op(s):
            U = aliased(ChatMessageORM)
            A = aliased(ChatMessageORM)
            C = aliased(ChatConversationORM)

            rows = (
                s.query(
                    ChatTurnORM.id.label("id"),
                    ChatTurnORM.conversation_id.label("conversation_id"),
                    ChatTurnORM.order_index.label("order_index"),
                    ChatTurnORM.star.label("star"),
                    ChatTurnORM.ner.label("ner"),
                    ChatTurnORM.domains.label("domains"),
                    ChatTurnORM.ai_knowledge_score.label("ai_score"),
                    ChatTurnORM.ai_knowledge_rationale.label("ai_rationale"),
                    U.text.label("user_text"),
                    A.text.label("assistant_text"),
                    C.title.label("goal_text"),
                    C.tags.label("tags"),
                )
                .join(U, ChatTurnORM.user_message_id == U.id)
                .join(A, ChatTurnORM.assistant_message_id == A.id)
                .join(C, ChatTurnORM.conversation_id == C.id)
                .filter(ChatTurnORM.id.in_([int(x) for x in ids]))
                .all()
            )
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
                    "goal_text": r.goal_text or "",
                    "tags": r.tags or [],
                    "ai_score": r.ai_score,
                    "ai_rationale": r.ai_rationale or "",
                })
            return out
        return self._run(op)


    def get_conversations_by_tags(
        self,
        tags: list[str],
        *,
        match: str = "any",   # "any" = at least one tag; "all" = must contain all
        limit: int = 200,
        include_messages: bool = False,
    ) -> List[ChatConversationORM]:
        """
        Return ChatConversationORM rows filtered by tags.
        
        Args:
            tags: List of tags to filter by.
            match: "any" (default) requires at least one tag to match,
                   "all" requires all tags to be present.
            limit: Max number of results to return.
            include_messages: If True, eager-load messages.
        """
        def op(s):
            q = s.query(ChatConversationORM)

            if tags:
                if match == "all":
                    for t in tags:
                        q = q.filter(ChatConversationORM.tags.contains([t]))
                else:  # match == "any"
                    q = q.filter(or_(*[ChatConversationORM.tags.contains([t]) for t in tags]))

            if include_messages:
                q = q.options(selectinload(ChatConversationORM.messages))

            order_col = getattr(ChatConversationORM, "created_at", None)
            if order_col is not None:
                q = q.order_by(order_col.desc())
            else:
                q = q.order_by(ChatConversationORM.id.desc())

            return q.limit(limit).all()

        return self._run(op)
