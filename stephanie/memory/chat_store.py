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
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from stephanie.orm.scorable_embedding import ScorableEmbeddingORM
from stephanie.orm.embedding import EmbeddingORM

from sqlalchemy import String, desc, func, or_, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Query, aliased, selectinload

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.chat import ChatConversationORM, ChatMessageORM, ChatTurnORM
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
        Append multiple messages to a conversation without reusing message order.
        
        Args:
            conv_id: ID of the conversation to add messages to
            messages: List of message dictionaries with role, text, and metadata
            
        Returns:
            List of created message objects
        """
        if not messages:
            return []

        def op(s):
            # Lock the parent row so concurrent writers to the same conversation
            # serialize their order-index allocation on PostgreSQL.
            conversation = (
                s.query(ChatConversationORM)
                .filter(ChatConversationORM.id == conv_id)
                .with_for_update()
                .one_or_none()
            )
            if conversation is None:
                raise ValueError(f"Conversation {conv_id} not found")

            current_max = (
                s.query(func.max(ChatMessageORM.order_index))
                .filter(ChatMessageORM.conversation_id == conv_id)
                .scalar()
            )
            next_index = int(current_max) + 1 if current_max is not None else 0

            objs: List[ChatMessageORM] = []
            for offset, msg in enumerate(messages):
                objs.append(
                    ChatMessageORM(
                        conversation_id=conv_id,
                        role=msg["role"],
                        text=msg.get("text", ""),
                        order_index=next_index + offset,
                        parent_id=msg.get("parent_id"),
                        meta=msg.get("meta", {}),
                    )
                )
            s.add_all(objs)
            conversation.updated_at = datetime.now()
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

        Returns a list of (ChatConversationORM, count) with messages (and turns)
        eagerly loaded so they can be safely used outside the session.
        """

        def op(s):
            # Base query for conversations with the relationships we care about
            base_q = (
                s.query(ChatConversationORM)
                .options(
                    # Eager-load plain messages so ScorableFactory.from_orm
                    # can iterate conv.messages without hitting a detached lazy load
                    selectinload(ChatConversationORM.messages),
                    # If you also rely on turns + their user/assistant messages,
                    # eager-load those too:
                    selectinload(ChatConversationORM.turns)
                    .selectinload(ChatTurnORM.user_message),
                    selectinload(ChatConversationORM.turns)
                    .selectinload(ChatTurnORM.assistant_message),
                )
            )

            if by == "messages":
                q = (
                    s.query(
                        ChatConversationORM,
                        func.count(ChatMessageORM.id).label("message_count"),
                    )
                    .select_from(ChatConversationORM)
                    .options(
                        selectinload(ChatConversationORM.messages),
                        selectinload(ChatConversationORM.turns)
                        .selectinload(ChatTurnORM.user_message),
                        selectinload(ChatConversationORM.turns)
                        .selectinload(ChatTurnORM.assistant_message),
                    )
                    .join(
                        ChatMessageORM,
                        ChatMessageORM.conversation_id == ChatConversationORM.id,
                    )
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
                    .select_from(ChatConversationORM)
                    .options(
                        selectinload(ChatConversationORM.messages),
                        selectinload(ChatConversationORM.turns)
                        .selectinload(ChatTurnORM.user_message),
                        selectinload(ChatConversationORM.turns)
                        .selectinload(ChatTurnORM.assistant_message),
                    )
                    .join(
                        ChatTurnORM,
                        ChatTurnORM.conversation_id == ChatConversationORM.id,
                    )
                    .group_by(ChatConversationORM.id)
                    .order_by(func.count(ChatTurnORM.id).desc())
                    .limit(limit)
                )

            rows = q.all()
            # rows is List[(ChatConversationORM, count)] with relationships already loaded
            return [(conv, int(count)) for conv, count in rows]

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



    def iter_turns_with_texts(
        self,
        *,
        min_star: Optional[int] = None,
        max_star: Optional[int] = None,
        min_ai_score: Optional[float] = None,
        max_ai_score: Optional[float] = None,
        require_assistant_text: bool = True,
        require_nonempty_ner: bool = True,
        min_assistant_len: int = 1,
        total_limit: int = 100000,            # hard cap across all batches
        batch_size: int = 1000,               # tune: 500–5000 is usually sweet
        start_after_id: Optional[int] = None, # keyset cursor (DESC): start from id < this
        include_texts: bool = True,           # set False to skip loading full texts
        include_goal: bool = True,            # set False to skip conversation title join
        order_desc: bool = True,
    ):
        """
        Yield batches of dict rows efficiently via keyset pagination.
        """
        def op(s):
            U = aliased(ChatMessageORM)
            A = aliased(ChatMessageORM)
            C = aliased(ChatConversationORM)

            produced = 0
            cursor_id = start_after_id
            is_pg = s.bind.dialect.name.startswith("postgre")

            # prebuild base selectable (without limit/cursor)
            base_cols = [
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
            ]

            if include_texts:
                text_cols = [U.text.label("user_text"), A.text.label("assistant_text")]
            else:
                text_cols = [func.cast(None, String).label("user_text"),
                            func.cast(None, String).label("assistant_text")]

            if include_goal:
                goal_cols = [C.title.label("goal_text")]
            else:
                goal_cols = [func.cast(None, String).label("goal_text")]

            sel_cols = base_cols + text_cols + goal_cols

            while True:
                q = (
                    s.query(*sel_cols)
                    .join(U, ChatTurnORM.user_message_id == U.id)
                    .join(A, ChatTurnORM.assistant_message_id == A.id)
                )
                if include_goal:
                    q = q.join(C, ChatTurnORM.conversation_id == C.id)

                # filters
                if min_star is not None:
                    q = q.filter(ChatTurnORM.star >= int(min_star))
                if max_star is not None:
                    q = q.filter(ChatTurnORM.star <= int(max_star))
                if min_ai_score is not None:
                    q = q.filter(ChatTurnORM.ai_knowledge_score >= float(min_ai_score))
                if max_ai_score is not None:
                    q = q.filter(ChatTurnORM.ai_knowledge_score <= float(max_ai_score))

                if require_assistant_text:
                    q = q.filter(A.text.isnot(None), A.text != "")
                    if min_assistant_len and min_assistant_len > 1:
                        # LENGTH() exists on most dialects; silently ignore if not
                        try:
                            q = q.filter(func.length(A.text) >= int(min_assistant_len))
                        except Exception:
                            pass

                if require_nonempty_ner:
                    # Fast + indexed on PG if you create a GIN; portable fallback otherwise.
                    if is_pg:
                        # COALESCE(jsonb_array_length(ner), 0) > 0
                        q = q.filter(ChatTurnORM.ner.isnot(None))
                        q = q.filter(func.jsonb_array_length(ChatTurnORM.ner) > 0)
                    else:
                        # Generic fallback: only check NOT NULL; we'll drop empties in Python
                        q = q.filter(ChatTurnORM.ner.isnot(None))

                # keyset cursor (no OFFSET)
                if cursor_id is not None:
                    q = q.filter(ChatTurnORM.id < int(cursor_id)) if order_desc else q.filter(ChatTurnORM.id > int(cursor_id))

                q = q.order_by(ChatTurnORM.id.desc() if order_desc else ChatTurnORM.id.asc()) \
                    .limit(int(batch_size)) \
                    .execution_options(stream_results=True)

                rows = q.all()
                if not rows:
                    break

                batch = []
                for r in rows:
                    row = {
                        "id": r.id,
                        "conversation_id": r.conversation_id,
                        "order_index": int(r.order_index or 0),
                        "star": int(r.star or 0),
                        "user_message_id": r.user_message_id,
                        "assistant_message_id": r.assistant_message_id,
                        "user_text": (r.user_text or "") if include_texts else "",
                        "assistant_text": (r.assistant_text or "") if include_texts else "",
                        "ner": r.ner or [],
                        "domains": r.domains or [],
                        "goal_text": (r.goal_text or "") if include_goal else "",
                        "ai_score": r.ai_score,
                        "ai_rationale": r.ai_rationale or "",
                    }
                    # dialect-agnostic guard for "nonempty NER" when not PG
                    if require_nonempty_ner and not is_pg:
                        try:
                            if not row["ner"]:
                                continue
                            if isinstance(row["ner"], list) and len(row["ner"]) == 0:
                                continue
                        except Exception:
                            pass
                    batch.append(row)

                if not batch:
                    # advance cursor anyway to break potential loop on empty post-filter
                    cursor_id = rows[-1].id
                    continue

                yield batch

                produced += len(batch)
                if produced >= total_limit:
                    break

                # advance keyset cursor
                cursor_id = rows[-1].id if order_desc else rows[-1].id

        return self._run(op)


    def get_chat_messages_for_embedding(self, force_reembed: bool = False, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get chat messages that need embedding processing as scorable items."""
        def op(s):
            # Get messages that don't have embeddings yet
            query = s.query(
                ChatMessageORM.id.label('message_id'),
                ChatMessageORM.role,
                ChatMessageORM.text,
                ChatMessageORM.chat_id,
                ChatConversationORM.title.label('chat_title'),
                ChatMessageORM.created_at
            ).join(ChatConversationORM, ChatMessageORM.chat_id == ChatConversationORM.id).filter(
                ChatMessageORM.text.isnot(None),
                ChatMessageORM.text != '',
                ChatMessageORM.is_active == True
            )
            
            if not force_reembed:
                # Only get messages without embedding mappings
                query = query.outerjoin(
                    ScorableEmbeddingORM,
                    (func.concat('chat_message:', ChatMessageORM.id) == ScorableEmbeddingORM.scorable_id) &
                    (ScorableEmbeddingORM.scorable_type == 'chat_message')
                ).filter(ScorableEmbeddingORM.id.is_(None))
            
            # Order by creation time and limit results
            query = query.order_by(ChatMessageORM.created_at)
            if limit:
                query = query.limit(limit)
                
            return [dict(row) for row in query.all()]
        
        return self._run(op)

    def find_similar_chat_messages(self, query_embedding: List[float], top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
        """Find chat messages similar to the query embedding using the embedding store."""
        def op(s):
            # Use pgvector cosine similarity search
            results = s.query(
                ScorableEmbeddingORM.scorable_id,
                ScorableEmbeddingORM.scorable_type,
                ChatMessageORM.text.label('message_text'),
                ChatMessageORM.role,
                ChatMessageORM.created_at,
                ChatConversationORM.title.label('chat_title'),
                ChatConversationORM.id.label('chat_id'),
                func.cosine_similarity(EmbeddingORM.embedding, query_embedding).label('similarity')
            ).join(
                EmbeddingORM, ScorableEmbeddingORM.embedding_id == EmbeddingORM.id
            ).join(
                ChatMessageORM, func.replace(ScorableEmbeddingORM.scorable_id, 'chat_message:', '') == ChatMessageORM.id.cast(String)
            ).join(
                ChatConversationORM, ChatMessageORM.chat_id == ChatConversationORM.id
            ).filter(
                ScorableEmbeddingORM.scorable_type == 'chat_message',
                func.cosine_similarity(EmbeddingORM.embedding, query_embedding) >= similarity_threshold
            ).order_by(
                func.cosine_similarity(EmbeddingORM.embedding, query_embedding).desc()
            ).limit(top_k).all()
            
            return [{
                'message_id': str(row.scorable_id).replace('chat_message:', ''),
                'chat_id': row.chat_id,
                'chat_title': row.chat_title,
                'role': row.role,
                'text': row.message_text,
                'created_at': row.created_at,
                'similarity': float(row.similarity)
            } for row in results]
        
        return self._run(op)

    def add_conversation_once(
        self, data: dict, request_id: str
    ) -> ChatConversationORM:
        """Idempotently create an SIS conversation using existing JSON metadata."""
        def op(s):
            existing = (
                s.query(ChatConversationORM)
                .filter(
                    ChatConversationORM.meta["sis_create_request_id"].as_string()
                    == request_id
                )
                .one_or_none()
            )
            if existing is not None:
                if existing.title != data.get("title"):
                    raise ValueError("Idempotency key was already used for another conversation")
                return existing
            conversation_data = dict(data)
            conversation_data["meta"] = {
                **(conversation_data.get("meta") or {}),
                "sis_create_request_id": request_id,
            }
            conversation = ChatConversationORM(**conversation_data)
            s.add(conversation)
            s.flush()
            return conversation

        return self._run(op)

    def get_message_by_request_id(
        self, conv_id: int, request_id: str, *, role: Optional[str] = None
    ) -> Optional[ChatMessageORM]:
        """Find an SIS message carrying a client idempotency key."""
        def op(s):
            query = s.query(ChatMessageORM).filter(
                ChatMessageORM.conversation_id == conv_id,
                ChatMessageORM.meta["sis_request_id"].as_string() == request_id,
            )
            if role:
                query = query.filter(ChatMessageORM.role == role)
            return query.order_by(ChatMessageORM.order_index.desc()).first()

        return self._run(op)

    def get_messages_before(
        self, conv_id: int, before_order_index: int, *, limit: int
    ) -> List[ChatMessageORM]:
        """Read only the newest bounded history preceding a message index."""
        if limit <= 0:
            return []

        def op(s):
            rows = (
                s.query(ChatMessageORM)
                .filter(
                    ChatMessageORM.conversation_id == conv_id,
                    ChatMessageORM.order_index < before_order_index,
                )
                .order_by(ChatMessageORM.order_index.desc())
                .limit(limit)
                .all()
            )
            return list(reversed(rows))

        return self._run(op)

    # ---------- SIS external executions (stored in existing conversation JSON) ----------

    def start_external_execution(
        self,
        conv_id: int,
        submission_key: str,
        request_fingerprint: str,
        task: str,
        record: dict,
        *,
        max_records: int = 100,
    ) -> tuple[ChatMessageORM, dict, bool]:
        """Atomically persist the task message and execution link without a new table."""
        def op(s):
            conversation = (
                s.query(ChatConversationORM)
                .filter(ChatConversationORM.id == conv_id)
                .with_for_update()
                .one_or_none()
            )
            if conversation is None:
                raise ValueError(f"Conversation {conv_id} not found")
            meta = dict(conversation.meta or {})
            executions = dict(meta.get("sis_executions") or {})
            requests = dict(meta.get("sis_execution_requests") or {})
            prior_id = requests.get(submission_key)
            if prior_id:
                prior = executions.get(prior_id)
                if not prior:
                    raise RuntimeError("Execution idempotency index is inconsistent")
                if prior.get("request_fingerprint") != request_fingerprint:
                    raise ValueError("Idempotency-Key was already used for another execution")
                message = next(
                    (
                        item
                        for item in s.query(ChatMessageORM)
                        .filter_by(conversation_id=conv_id, role="user")
                        .all()
                        if (item.meta or {}).get("sis_execution_id") == prior_id
                    ),
                    None,
                )
                if message is None:
                    raise RuntimeError("Execution task message is missing")
                return message, deepcopy(prior), False
            if len(executions) >= max_records:
                raise ValueError("Conversation execution history is full")

            current_max = (
                s.query(func.max(ChatMessageORM.order_index))
                .filter(ChatMessageORM.conversation_id == conv_id)
                .scalar()
            )
            message = ChatMessageORM(
                conversation_id=conv_id,
                role="user",
                text=task,
                order_index=int(current_max) + 1 if current_max is not None else 0,
                meta={
                    "source": "sis_worker",
                    "sis_owner": record.get("username"),
                    "sis_request_id": f"stn03-{record['execution_id']}",
                    "sis_execution_id": record["execution_id"],
                    "sis_execution_dry_run": bool(record.get("dry_run")),
                    "execution_status": "QUEUED",
                },
            )
            s.add(message)
            s.flush()
            stored = {**record, "user_message_id": message.id}
            executions[stored["execution_id"]] = stored
            requests[submission_key] = stored["execution_id"]
            meta["sis_executions"] = executions
            meta["sis_execution_requests"] = requests
            conversation.meta = meta
            conversation.updated_at = datetime.now()
            s.flush()
            return message, deepcopy(stored), True

        return self._run(op)

    def update_external_execution(
        self, conv_id: int, execution_id: str, changes: dict
    ) -> Optional[dict]:
        """Update one bounded execution record in canonical conversation metadata."""
        def op(s):
            conversation = (
                s.query(ChatConversationORM)
                .filter(ChatConversationORM.id == conv_id)
                .with_for_update()
                .one_or_none()
            )
            if conversation is None:
                return None
            # Deep-copy nested JSON before mutation so SQLAlchemy sees a new
            # value rather than an in-place change to the loaded JSON object.
            meta = deepcopy(conversation.meta or {})
            executions = dict(meta.get("sis_executions") or {})
            existing_record = executions.get(execution_id)
            if existing_record is None:
                return None
            record = deepcopy(existing_record)
            record = {**record, **changes}
            executions[execution_id] = record
            meta["sis_executions"] = executions
            conversation.meta = meta
            conversation.updated_at = datetime.now()
            if "status" in changes:
                message = next(
                    (
                        item
                        for item in s.query(ChatMessageORM)
                        .filter_by(conversation_id=conv_id, role="user")
                        .all()
                        if (item.meta or {}).get("sis_execution_id") == execution_id
                    ),
                    None,
                )
                if message is not None:
                    message.meta = {
                        **(message.meta or {}),
                        "execution_status": str(changes["status"]).lower(),
                    }
            s.flush()
            return deepcopy(record)

        return self._run(op)

    def get_external_execution(self, conv_id: int, execution_id: str) -> Optional[dict]:
        def op(s):
            conversation = s.get(ChatConversationORM, conv_id)
            if conversation is None:
                return None
            record = (conversation.meta or {}).get("sis_executions", {}).get(execution_id)
            return deepcopy(record) if record is not None else None

        return self._run(op)

    def record_external_execution_review(
        self,
        conv_id: int,
        execution_id: str,
        *,
        review_id: str,
        reviewer: str,
        decision: str,
        notes: str,
        evidence_refs: list[str],
        accepted_scope: list[str],
        reviewed_at: str,
        max_history: int = 20,
    ) -> Optional[dict]:
        """Append one idempotent human review without changing Worker status."""
        def op(s):
            conversation = (
                s.query(ChatConversationORM)
                .filter(ChatConversationORM.id == conv_id)
                .with_for_update()
                .one_or_none()
            )
            if conversation is None:
                return None
            # JSON columns need a fresh nested object to be marked dirty by
            # SQLAlchemy after appending audit history.
            meta = deepcopy(conversation.meta or {})
            executions = dict(meta.get("sis_executions") or {})
            existing_record = executions.get(execution_id)
            if existing_record is None:
                return None
            record = deepcopy(existing_record)

            review = {
                "review_id": review_id,
                "reviewer": reviewer,
                "decision": decision,
                "notes": notes,
                "evidence_refs": list(evidence_refs),
                "accepted_scope": list(accepted_scope),
                "reviewed_at": reviewed_at,
            }
            history = list(record.get("human_review_history") or [])
            prior = next((item for item in history if item.get("review_id") == review_id), None)
            if prior is not None:
                comparable_prior = {key: value for key, value in prior.items() if key != "reviewed_at"}
                comparable_incoming = {key: value for key, value in review.items() if key != "reviewed_at"}
                if comparable_prior != comparable_incoming:
                    raise ValueError("Review idempotency key was already used for another decision")
                return deepcopy(record)

            current = record.get("human_review") or {}
            if current.get("decision") in {"ACCEPT", "REJECT"}:
                raise ValueError("A final human execution review is already recorded")
            if len(history) >= max_history:
                raise ValueError("Execution review history is full")

            history.append(review)
            record["human_review"] = review
            record["human_review_history"] = history
            record["human_review_status"] = decision
            record["updated_at"] = reviewed_at
            executions[execution_id] = record
            meta["sis_executions"] = executions
            conversation.meta = meta
            conversation.updated_at = datetime.now()
            s.flush()
            return deepcopy(record)

        return self._run(op)

    def get_external_execution_by_submission_key(
        self, conv_id: int, submission_key: str
    ) -> Optional[dict]:
        def op(s):
            conversation = s.get(ChatConversationORM, conv_id)
            if conversation is None:
                return None
            meta = conversation.meta or {}
            execution_id = (meta.get("sis_execution_requests") or {}).get(submission_key)
            record = (meta.get("sis_executions") or {}).get(execution_id)
            return deepcopy(record) if record is not None else None

        return self._run(op)

    def list_external_executions(self, conv_id: int) -> list[dict]:
        def op(s):
            conversation = s.get(ChatConversationORM, conv_id)
            if conversation is None:
                return []
            return deepcopy(list(((conversation.meta or {}).get("sis_executions") or {}).values()))

        return self._run(op)

    def add_user_message_once(
        self,
        conv_id: int,
        request_id: str,
        text: str,
        *,
        meta: Optional[dict] = None,
    ) -> ChatMessageORM:
        """Append one user prompt per request key under the conversation lock."""
        def op(s):
            conversation = (
                s.query(ChatConversationORM)
                .filter(ChatConversationORM.id == conv_id)
                .with_for_update()
                .one_or_none()
            )
            if conversation is None:
                raise ValueError(f"Conversation {conv_id} not found")

            existing = (
                s.query(ChatMessageORM)
                .filter(
                    ChatMessageORM.conversation_id == conv_id,
                    ChatMessageORM.role == "user",
                    ChatMessageORM.meta["sis_request_id"].as_string() == request_id,
                )
                .first()
            )
            if existing is not None:
                if existing.text != text:
                    raise ValueError("Idempotency key was already used for another message")
                return existing

            current_max = (
                s.query(func.max(ChatMessageORM.order_index))
                .filter(ChatMessageORM.conversation_id == conv_id)
                .scalar()
            )
            message = ChatMessageORM(
                conversation_id=conv_id,
                role="user",
                text=text,
                order_index=int(current_max) + 1 if current_max is not None else 0,
                meta={**(meta or {}), "sis_request_id": request_id, "execution_status": "in_progress"},
            )
            conversation.updated_at = datetime.now()
            s.add(message)
            s.flush()
            return message

        return self._run(op)

    def mark_user_message_status(
        self,
        conv_id: int,
        message_id: int,
        status: str,
        *,
        error_type: Optional[str] = None,
    ) -> Optional[ChatMessageORM]:
        """Record a request outcome without inventing an assistant response."""
        def op(s):
            message = (
                s.query(ChatMessageORM)
                .filter_by(id=message_id, conversation_id=conv_id, role="user")
                .with_for_update()
                .one_or_none()
            )
            if message is None:
                return None
            message.meta = {
                **(message.meta or {}),
                "execution_status": status,
                **({"error_type": error_type} if error_type else {}),
            }
            if status == "in_progress":
                message.meta.pop("error_type", None)
            return message

        return self._run(op)

    def complete_assistant_turn(
        self,
        conv_id: int,
        user_message_id: int,
        request_id: str,
        text: str,
        *,
        meta: Optional[dict] = None,
    ) -> tuple[ChatMessageORM, ChatTurnORM]:
        """Persist the genuine assistant message and its ChatTurn atomically."""
        def op(s):
            conversation = (
                s.query(ChatConversationORM)
                .filter(ChatConversationORM.id == conv_id)
                .with_for_update()
                .one_or_none()
            )
            if conversation is None:
                raise ValueError(f"Conversation {conv_id} not found")

            existing = (
                s.query(ChatMessageORM)
                .filter(
                    ChatMessageORM.conversation_id == conv_id,
                    ChatMessageORM.role == "assistant",
                    ChatMessageORM.meta["sis_request_id"].as_string() == request_id,
                )
                .first()
            )
            if existing is not None:
                turn = (
                    s.query(ChatTurnORM)
                    .filter(ChatTurnORM.assistant_message_id == existing.id)
                    .one_or_none()
                )
                if turn is None:
                    raise RuntimeError("Existing assistant response has no persisted turn")
                return existing, turn

            user_message = s.get(ChatMessageORM, user_message_id)
            if (
                user_message is None
                or user_message.conversation_id != conv_id
                or user_message.role != "user"
                or (user_message.meta or {}).get("sis_request_id") != request_id
            ):
                raise ValueError("Request does not match a persisted user message")

            current_max = (
                s.query(func.max(ChatMessageORM.order_index))
                .filter(ChatMessageORM.conversation_id == conv_id)
                .scalar()
            )
            assistant = ChatMessageORM(
                conversation_id=conv_id,
                role="assistant",
                text=text,
                order_index=int(current_max) + 1 if current_max is not None else 0,
                meta={**(meta or {}), "sis_request_id": request_id, "execution_status": "completed"},
            )
            conversation.updated_at = datetime.now()
            s.add(assistant)
            s.flush()

            max_turn_order = (
                s.query(func.coalesce(func.max(ChatTurnORM.order_index), -1))
                .filter(ChatTurnORM.conversation_id == conv_id)
                .scalar()
            )
            turn = ChatTurnORM(
                conversation_id=conv_id,
                user_message_id=user_message.id,
                assistant_message_id=assistant.id,
                order_index=max(int(max_turn_order) + 1, assistant.order_index),
            )
            user_message.meta = {
                **(user_message.meta or {}),
                "execution_status": "completed",
            }
            user_message.meta.pop("error_type", None)
            s.add(turn)
            s.flush()
            return assistant, turn

        return self._run(op)
