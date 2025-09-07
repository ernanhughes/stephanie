# stephanie/memory/chat_store.py
from sqlalchemy import desc, func, text
from sqlalchemy.orm import Session
from stephanie.models.chat import ChatConversationORM, ChatMessageORM, ChatTurnORM

class ChatStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "chats"

    def add_conversation(self, data: dict) -> ChatConversationORM:
        try:
            conv = ChatConversationORM(**data)
            self.session.add(conv)
            self.session.commit()
            return conv
        except Exception as e:
            self.session.rollback() 
            raise


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
