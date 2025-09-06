# stephanie/memory/chat_store.py
from tokenize import String
from sqlalchemy import cast
from sqlalchemy.orm import Session
from stephanie.models.chat import ChatConversationORM, ChatMessageORM, ChatTurnORM

class ChatStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "chats"

    def add_conversation(self, data: dict) -> ChatConversationORM:
        conv = ChatConversationORM(**data)
        self.session.add(conv)
        self.session.commit()
        return conv

    def exists_conversation(self, h: str) -> bool:
        return self.session.query(ChatConversationORM).filter(
            cast(ChatConversationORM.meta["hash"], String) == h).first() is not None

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
