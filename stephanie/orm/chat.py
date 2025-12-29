# stephanie/orm/chat.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (JSON, Column, DateTime, ForeignKey, Integer, String,
                        Text)
from sqlalchemy.orm import relationship

from stephanie.orm.base import Base


class ChatConversationORM(Base):
    __tablename__ = "chat_conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String, nullable=False, default="openai")
    external_id = Column(String, nullable=True)  # "conversation_id" from JSON
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)
    meta = Column(JSON, default={})
    tags = Column(JSON, default=list) 

    messages = relationship(
        "ChatMessageORM",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessageORM.order_index",
    )

    turns = relationship("ChatTurnORM", back_populates="conversation")

    def to_dict(self, include_messages: bool = False, include_turns: bool = False):
        data = {
            "id": self.id,
            "provider": self.provider,
            "external_id": self.external_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "meta": self.meta or {},
        }
        if include_messages:
            data["messages"] = [m.to_dict() for m in self.messages]
        if include_turns:
            data["turns"] = [t.to_dict() for t in self.turns]
        return data


class ChatMessageORM(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        Integer,
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String, nullable=False)  # "user", "assistant", "system", "tool"
    text = Column(Text, nullable=True)

    parent_id = Column(
        Integer,
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=True,
    )

    order_index = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    meta = Column(JSON, default={})

    conversation = relationship("ChatConversationORM", back_populates="messages")

    parent = relationship(
        "ChatMessageORM",
        remote_side=[id],
        backref="children",
        foreign_keys=[parent_id],
    )

    def to_dict(self, include_children: bool = False):
        data = {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "text": self.text,
            "parent_id": self.parent_id,
            "order_index": self.order_index,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "meta": self.meta or {},
        }
        if include_children:
            data["children"] = [c.to_dict() for c in self.children]
        return data


class ChatTurnORM(Base):
    __tablename__ = "chat_turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        Integer, ForeignKey("chat_conversations.id", ondelete="CASCADE")
    )
    user_message_id = Column(
        Integer, ForeignKey("chat_messages.id", ondelete="CASCADE")
    )
    assistant_message_id = Column(
        Integer, ForeignKey("chat_messages.id", ondelete="CASCADE")
    )

    conversation = relationship("ChatConversationORM", back_populates="turns")
    user_message = relationship("ChatMessageORM", foreign_keys=[user_message_id])
    assistant_message = relationship("ChatMessageORM", foreign_keys=[assistant_message_id])
    order_index = Column(Integer, nullable=False, default=0)
    star = Column(Integer, nullable=False, default=0)
    ai_knowledge_score = Column(Integer, nullable=True)     # 0..100
    ai_knowledge_rationale = Column(Text, nullable=True)
    
    ner = Column(JSON, nullable=True)      # [{"text":"PACS","label":"METHOD","start":12,"end":16}, ...]
    domains = Column(JSON, nullable=True)  # [{"domain":"alignment","score":0.82}, ...]

    def to_dict(self, include_messages: bool = True):
        data = {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "user_message_id": self.user_message_id,
            "assistant_message_id": self.assistant_message_id,
            "order_index": self.order_index,
            "star": self.star,

            "ai_knowledge_score": self.ai_knowledge_score,
            "ai_knowledge_score_norm": (
                None if self.ai_knowledge_score is None
                else max(0.0, min(1.0, float(self.ai_knowledge_score)/100.0))
            ),
            "ai_knowledge_rationale": self.ai_knowledge_rationale or "",
            
            "ner": self.ner or [],
            "domains": self.domains or [],
        }
        if include_messages:
            data["user_message"] = self.user_message.to_dict() if self.user_message else None
            data["assistant_message"] = (
                self.assistant_message.to_dict() if self.assistant_message else None
            )
        return data
