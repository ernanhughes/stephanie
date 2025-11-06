from __future__ import annotations

from sqlalchemy import JSON, Column, Float, Integer, String, Text

from stephanie.models.base import Base


class ScorableFeatureORM(Base):
    __tablename__ = "scorable_features"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # identity
    scorable_id = Column(String, nullable=False)  # note not unique alone
    scorable_type = Column(String, nullable=False) 

    # 🔗 parent (no FK; we stay decoupled and simple)
    parent_scorable_id = Column(String, nullable=True)
    parent_scorable_type = Column(String, nullable=True)
    order_in_parent = Column(Integer, nullable=True)

    # context / temporal
    conversation_id = Column(Integer, nullable=True)
    order_index = Column(Integer, nullable=True)
    chat_id = Column(Integer, nullable=True)
    turn_index = Column(Integer, nullable=True)

    # content
    title = Column(Text, nullable=True)
    text = Column(Text, nullable=True)
    near_identity = Column(JSON, nullable=True)

    # annotations
    domains = Column(JSON, nullable=True)
    ner = Column(JSON, nullable=True)

    # signals
    ai_score = Column(Float, nullable=True)
    star = Column(Float, nullable=True)

    # goal
    goal_ref = Column(JSON, nullable=True)

    # embeddings & metrics
    embeddings = Column(JSON, nullable=True)        # {"global":[...], ...}
    metrics_columns = Column(JSON, nullable=True)
    metrics_values = Column(JSON, nullable=True)
    metrics_vector = Column(JSON, nullable=True)

    # agreement/stability
    agreement = Column(Float, nullable=True)
    stability = Column(Float, nullable=True)

    # artifacts
    vpm_png = Column(Text, nullable=True)
    rollout = Column(JSON, nullable=True)

    def to_dict(self) -> dict:
        return {
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "parent_scorable_id": self.parent_scorable_id,
            "parent_scorable_type": self.parent_scorable_type,
            "order_in_parent": self.order_in_parent,
            "conversation_id": self.conversation_id,
            "order_index": self.order_index,
            "chat_id": self.chat_id,
            "turn_index": self.turn_index,
            "title": self.title,
            "text": self.text,
            "near_identity": self.near_identity,
            "domains": self.domains or [],
            "ner": self.ner or [],
            "ai_score": self.ai_score,
            "star": self.star,
            "goal_ref": self.goal_ref,
            "embeddings": self.embeddings or {},
            "metrics_columns": self.metrics_columns or [],
            "metrics_values": self.metrics_values or [],
            "metrics_vector": self.metrics_vector or {},
            "agreement": self.agreement,
            "stability": self.stability,
            "vpm_png": self.vpm_png,
            "rollout": self.rollout or {},
        }

