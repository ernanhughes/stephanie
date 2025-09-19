# stephanie/memory/chat_store.py
from __future__ import annotations

from typing import Optional, List, Tuple

from sqlalchemy import desc, func, text

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.chat import (
    ChatConversationORM,
    ChatMessageORM,
    ChatTurnORM,
)
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType


class ChatStore(BaseSQLAlchemyStore):
    orm_model = ChatConversationORM
    # Use column name so BaseSQLAlchemyStore can apply .desc() reliably
    default_order_by = "created_at"

    def __init__(self, session, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "chats"

    def name(self) -> str:
        return self.name

    # ---------- Conversations ----------

    def add_conversation(self, data: dict) -> ChatConversationORM:
        def op():
            with self._scope() as s:
                conv = ChatConversationORM(**data)
                s.add(conv)
                s.flush()
                return conv
        return self._run(op)

    def get_all(self, limit: int = 100) -> List[ChatConversationORM]:
        def op():
            with self._scope() as s:
                return (
                    s.query(ChatConversationORM)
                    .order_by(
                        desc(getattr(ChatConversationORM, "created_at", ChatConversationORM.id))
                    ) 
                    .limit(limit)
                    .all()
                )
        return self._run(op)

    def exists_conversation(self, file_hash: str) -> bool:
        def op():
            with self._scope() as s:
                return (
                    s.query(ChatConversationORM)
                    .filter(
                        func.jsonb_extract_path_text(ChatConversationORM.meta, "hash") == file_hash
                    )
                    .first()
                    is not None
                )
        return self._run(op)

    def get_conversation(self, conv_id: int) -> Optional[ChatConversationORM]:
        def op():
            with self._scope() as s:
                return s.get(ChatConversationORM, conv_id)
        return self._run(op)

    # ---------- Messages ----------

    def add_messages(self, conv_id: int, messages: List[dict]) -> List[ChatMessageORM]:
        def op():
            with self._scope() as s:
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
        def op():
            with self._scope() as s:
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
        Assumes messages are in chronological order and include DB ids.
        """
        def op():
            with self._scope() as s:
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
        def op():
            with self._scope() as s:
                return s.get(ChatTurnORM, turn_id)
        return self._run(op)

    def get_turns_for_conversation(self, conv_id: int) -> List[ChatTurnORM]:
        def op():
            with self._scope() as s:
                return (
                    s.query(ChatTurnORM)
                    .filter_by(conversation_id=conv_id)
                    .order_by(ChatTurnORM.id)
                    .all()
                )
        return self._run(op)

    # ---------- Admin / Stats ----------

    def purge_all(self, force: bool = False):
        def op():
            with self._scope() as s:
                if force:
                    s.execute(text("TRUNCATE chat_turns RESTART IDENTITY CASCADE"))
                    s.execute(text("TRUNCATE chat_messages RESTART IDENTITY CASCADE"))
                    s.execute(text("TRUNCATE chat_conversations RESTART IDENTITY CASCADE"))
                else:
                    s.query(ChatTurnORM).delete()
                    s.query(ChatMessageORM).delete()
                    s.query(ChatConversationORM).delete()
                return True
        ok = self._run(op)
        if self.logger and ok:
            self.logger.info("[ChatStore] Purged all conversations, messages, and turns")

    def get_top_conversations(self, limit: int = 10, by: str = "turns") -> List[Tuple[ChatConversationORM, int]]:
        def op():
            with self._scope() as s:
                if by == "messages":
                    q = (
                        s.query(ChatConversationORM, func.count(ChatMessageORM.id).label("message_count"))
                        .join(ChatMessageORM)
                        .group_by(ChatConversationORM.id)
                        .order_by(desc("message_count"))
                        .limit(limit)
                    )
                else:
                    q = (
                        s.query(ChatConversationORM, func.count(ChatTurnORM.id).label("count"))
                        .join(ChatTurnORM)
                        .group_by(ChatConversationORM.id)
                        .order_by(func.count(ChatTurnORM.id).desc())
                        .limit(limit)
                    )
                return [(conv, int(count)) for conv, count in q.all()]
        return self._run(op)

    def get_message_by_id(self, message_id: int) -> Optional[ChatMessageORM]:
        def op():
            with self._scope() as s:
                return s.get(ChatMessageORM, message_id)
        return self._run(op)

    # ---------- Scorable helpers ----------

    def scorable_from_conversation(self, conv: ChatConversationORM) -> Scorable:
        """
        Convert a ChatConversationORM into a Scorable object.
        NOTE: if conv.messages is lazy-loaded, ensure it's loaded before the session closes,
        or fetch messages separately and pass them in.
        """
        text_val = "\n".join([f"{m.role}: {m.text}" for m in getattr(conv, "messages", []) if m.text]).strip()
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
        user_text = turn.user_message.text if getattr(turn, "user_message", None) else ""
        assistant_text = turn.assistant_message.text if getattr(turn, "assistant_message", None) else ""
        text_val = f"USER: {user_text.strip()}\nASSISTANT: {assistant_text.strip()}"
        return Scorable(
            id=str(turn.id),
            text=text_val,
            target_type=TargetType.CONVERSATION_TURN,
            meta={"conversation_id": turn.conversation_id},
        )
