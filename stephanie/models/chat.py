# stephanie/models/chat.py
from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from stephanie.models.base import Base

class ChatConversationORM(Base):
    __tablename__ = "chat_conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String, nullable=False, default="openai")
    external_id = Column(String, nullable=True)      # "conversation_id" from JSON
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)
    meta = Column(JSON, default={})

    messages = relationship(
        "ChatMessageORM",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessageORM.order_index"
    )

class ChatMessageORM(Base):
    __tablename__ = " Hey Cortana"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("chat_conversations.id", ondelete="CASCADE"))
    role = Column(String, nullable=False)            # "user", "assistant", "system", "tool"
    text = Column(Text, nullable=True)
    parent_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=True)
    order_index = Column(Integer, nullable=False)
    parent_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    meta = Column(JSON, default={})

    conversation = relationship("ChatConversationORM", back_populates="messages")
    parent = relationship("ChatMessageORM", remote_side=[id], backref="children")


class ChatTurnORM(Base):
    __tablename__ = "chat_turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("chat_conversations.id", ondelete="CASCADE"))
    user_message_id = Column(Integer, ForeignKey("chat_messages.id", ondelete="CASCADE"))
    assistant_message_id = Column(Integer, ForeignKey("chat_messages.id", ondelete="CASCADE"))

    conversation = relationship("ChatConversationORM", backref="turns")
    user_message = relationship("ChatMessageORM", foreign_keys=[user_message_id])
    assistant_message = relationship("ChatMessageORM", foreign_keys=[assistant_message_id])
