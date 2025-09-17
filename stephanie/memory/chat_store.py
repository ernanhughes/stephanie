# stephanie/memory/chat_store.py
from __future__ import annotations

from sqlalchemy import desc, func, text
from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.chat import (ChatConversationORM, ChatMessageORM,
                                   ChatTurnORM)
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType


class ChatStore(BaseSQLAlchemyStore):
    orm_model = ChatConversationORM
    default_order_by = ChatConversationORM.created_at.desc()

    def __init__(self, session: Session, logger=None):
        super().__init__(session, logger)
        self.name = "chats"

    def name(self) -> str:
        return self.name
    
    def add_conversation(self, data: dict) -> ChatConversationORM:
        try:
            conv = ChatConversationORM(**data)
            self.session.add(conv)
            self.session.commit()
            return conv
        except Exception as e:
            self.session.rollback() 
            raise

    def get_all(self, limit: int = 100) -> list[ChatConversationORM]:
        return self.session.query(ChatConversationORM).order_by(desc(ChatConversationORM.created_at)).limit(limit).all()

    def exists_conversation(self, file_hash: str) -> bool:
        return (
            self.session.query(ChatConversationORM)
            .filter(func.jsonb_extract_path_text(ChatConversationORM.meta, 'hash') == file_hash)
            .first()
            is not None
        )

    def add_messages(self, conv_id: int, messages: list[dict]) -> list[ChatMessageORM]:
        objs = []
        for i, msg in enumerate(messages):
            objs.append(ChatMessageORM(
                conversation_id=conv_id,
                role=msg["role"],
                text=msg.get("text", ""),
                order_index=i,
                parent_id=msg.get("parent_id"),
                meta=msg.get("meta", {})
            ))
        self.session.add_all(objs) 
        self.session.commit()
        return objs

    def get_conversation(self, conv_id: int) -> ChatConversationORM | None:
        return self.session.query(ChatConversationORM).filter_by(id=conv_id).first()

    def get_messages(self, conv_id: int) -> list[ChatMessageORM]:
        return self.session.query(ChatMessageORM).filter_by(conversation_id=conv_id).order_by(ChatMessageORM.order_index).all()


    def add_turns(self, conversation_id: int, messages: list) -> list[ChatTurnORM]:
        """
        Build Q/A turns from a flat list of messages.
        Assumes messages are in chronological order.
        """
        turns = []
        for i in range(len(messages) - 1):
            u, a = messages[i], messages[i + 1]
            if u["role"] == "user" and a["role"] == "assistant":
                turn = ChatTurnORM(
                    conversation_id=conversation_id,
                    user_message_id=u["id"],
                    assistant_message_id=a["id"],
                )
                self.session.add(turn)
                turns.append(turn)
        self.session.commit()
        return turns

    def purge_all(self, force: bool = False):
        """
        Remove all chat data (conversations, messages, turns).
        If force=True, uses TRUNCATE (faster, resets IDs).
        Otherwise, uses DELETE.
        """
        if force:
            # TRUNCATE is faster but may need CASCADE depending on DB
            self.session.execute(text("TRUNCATE chat_turns RESTART IDENTITY CASCADE"))
            self.session.execute(text("TRUNCATE chat_messages RESTART IDENTITY CASCADE"))
            self.session.execute(text("TRUNCATE chat_conversations RESTART IDENTITY CASCADE"))
        else:
            # Safer if TRUNCATE not available (e.g., SQLite)
            self.session.query(ChatTurnORM).delete()
            self.session.query(ChatMessageORM).delete()
            self.session.query(ChatConversationORM).delete()
        self.session.commit()
        if self.logger:
            self.logger.info("[ChatStore] Purged all conversations, messages, and turns")


    def get_top_conversations(self, limit: int = 10, by: str = "turns"):
        if by == "messages":
            q = (
                self.session.query(
                    ChatConversationORM,
                    func.count(ChatMessageORM.id).label("message_count")
                )
                .join(ChatMessageORM)
                .group_by(ChatConversationORM.id)
                .order_by(desc("message_count"))   # <-- order by the alias
                .limit(limit)
            )
        else:  # by turns
            q = (
                self.session.query(ChatConversationORM, func.count(ChatTurnORM.id).label("count"))
                .join(ChatTurnORM)
                .group_by(ChatConversationORM.id)
                .order_by(func.count(ChatTurnORM.id).desc())
                .limit(limit)
            )

        return [(conv, count) for conv, count in q.all()]

    def get_message_by_id(self, message_id: int) -> ChatMessageORM | None:
        """
        Fetch a single ChatMessageORM by its ID.
        """
        return (
            self.session.query(ChatMessageORM)
            .filter_by(id=message_id)
            .first()
        )

    def get_turn_by_id(self, turn_id: int) -> ChatTurnORM | None:
        """
        Fetch a single ChatTurnORM by its ID.
        """
        return (
            self.session.query(ChatTurnORM)
            .filter_by(id=turn_id)
            .first()
        )

    def get_turns_for_conversation(self, conv_id: int) -> list[ChatTurnORM]:
        """
        Fetch all ChatTurnORMs for a given conversation.
        """
        return (
            self.session.query(ChatTurnORM)
            .filter_by(conversation_id=conv_id)
            .order_by(ChatTurnORM.id)
            .all()
        )

    def scorable_from_conversation(self, conv: ChatConversationORM) -> Scorable:
        """
        Convert a ChatConversationORM into a Scorable object.
        Uses the full text of the conversation.
        """
        text = "\n".join(
            [f"{m.role}: {m.text}" for m in conv.messages if m.text]
        ).strip()

        return Scorable(
            id=str(conv.id),
            text=text,
            target_type=TargetType.CONVERSATION,
            meta=conv.to_dict(include_messages=False)
        )

    def scorable_from_message(self, msg: ChatMessageORM) -> Scorable:
        """
        Convert a single ChatMessageORM into a Scorable object.
        """
        text = msg.text or ""
        return Scorable(
            id=str(msg.id),
            text=f"[{msg.role}] {text.strip()}",
            target_type=TargetType.CONVERSATION_MESSAGE,
            meta={"conversation_id": msg.conversation_id}
        )

    def scorable_from_turn(self, turn: ChatTurnORM) -> Scorable:
        """
        Convert a ChatTurnORM (user→assistant pair) into a Scorable object.
        """
        user_text = turn.user_message.text if turn.user_message else ""
        assistant_text = turn.assistant_message.text if turn.assistant_message else ""
        text = f"USER: {user_text.strip()}\nASSISTANT: {assistant_text.strip()}"

        return Scorable(
            id=str(turn.id),
            text=text,
            target_type=TargetType.CONVERSATION_TURN,
            meta={"conversation_id": turn.conversation_id}
        )
