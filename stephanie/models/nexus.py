# stephanie/models/nexus.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Index,
                        Integer, String, Text)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


# --- Scorable ---------------------------------------------------------------

class NexusScorableORM(Base):
    __tablename__ = "nexus_scorable"

    id = Column(String, primary_key=True)          # external/idempotent id (e.g., turn id)
    created_ts = Column(DateTime, default=datetime.utcnow, index=True)

    chat_id = Column(String, nullable=True, index=True)
    turn_index = Column(Integer, nullable=True, index=True)

    target_type = Column(String, nullable=True, index=True)  # ScorableType
    text = Column(Text, nullable=True)

    # Optional annotations
    domains = Column(JSONB if JSONB else JSON, nullable=True)     # List[str] or [{"domain": "...", "score": ...}]
    entities = Column(JSONB if JSONB else JSON, nullable=True)    # List[str] or [{"text": "...", "label": "..."}]
    meta = Column(JSONB if JSONB else JSON, nullable=True)        # near_identity, etc.

    embedding = relationship("NexusEmbeddingORM", back_populates="scorable", uselist=False, cascade="all, delete-orphan")
    metrics = relationship("NexusMetricsORM", back_populates="scorable", uselist=False, cascade="all, delete-orphan")

    def to_dict(self, include_vectors: bool = False) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "created_ts": self.created_ts.isoformat() if self.created_ts else None,
            "chat_id": self.chat_id,
            "turn_index": self.turn_index,
            "target_type": self.target_type,
            "text": self.text,
            "domains": self.domains or [],
            "entities": self.entities or [],
            "meta": self.meta or {},
        }
        if include_vectors:
            d["embedding"] = self.embedding.to_dict() if self.embedding else None
            d["metrics"] = self.metrics.to_dict() if self.metrics else None
        return d


# --- Embedding --------------------------------------------------------------

class NexusEmbeddingORM(Base):
    """
    pgvector: store in a real vector column via DDL. For ORM portability we keep JSON too.
    If pgvector is unavailable, we store as JSON list and run Python fallback KNN.
    """
    __tablename__ = "nexus_embedding"

    scorable_id = Column(String, ForeignKey("nexus_scorable.id", ondelete="CASCADE"), primary_key=True)
    # Store as JSON list for full portability; pgvector index created via migration/DDL.
    embed_global = Column(JSONB if JSONB else JSON, nullable=False)   # [float, ...]
    norm_l2 = Column(Float, nullable=True)

    scorable = relationship("NexusScorableORM", back_populates="embedding")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scorable_id": self.scorable_id,
            "norm_l2": self.norm_l2,
            "embed_global": self.embed_global or [],
        }


# --- Metrics ----------------------------------------------------------------

class NexusMetricsORM(Base):
    __tablename__ = "nexus_metrics"

    scorable_id = Column(String, ForeignKey("nexus_scorable.id", ondelete="CASCADE"), primary_key=True)
    columns = Column(JSONB if JSONB else JSON, nullable=False, default=list)  # List[str]
    values = Column(JSONB if JSONB else JSON, nullable=False, default=list)   # List[float] aligned with columns
    vector = Column(JSONB if JSONB else JSON, nullable=True)                  # {name: value}

    scorable = relationship("NexusScorableORM", back_populates="metrics")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scorable_id": self.scorable_id,
            "columns": self.columns or [],
            "values": self.values or [],
            "vector": self.vector or {},
        }


# --- Edge ----------------------------------------------------- OK--------------

class NexusEdgeORM(Base):
    __tablename__ = "nexus_edge"

    # Logical run space. Use "live" for continuous edges, or actual run_id for batch artifacts.
    run_id = Column(String, primary_key=True)

    src = Column(String, primary_key=True) 
    dst = Column(String, primary_key=True)
    type = Column(String, primary_key=True)  # "knn_global" | "temporal_next" | "shared_domain" | ...

    weight = Column(Float, nullable=False, default=0.0)
    channels = Column(JSONB if JSONB else JSON, nullable=True)
    created_ts = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("idx_nexus_edge_src", "run_id", "src"),
        Index("idx_nexus_edge_dst", "run_id", "dst"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "src": self.src,
            "dst": self.dst,
            "type": self.type,
            "weight": float(self.weight or 0.0),
            "channels": self.channels or {},
            "created_ts": self.created_ts.isoformat() if self.created_ts else None,
        }


# --- Pulse ------------------------------------------------------------------

class NexusPulseORM(Base):
    __tablename__ = " You sure she doesn't need to take a the dog yeah she might need to take a **** she might have taken **** because she's expecting to go out Yeah she she she stores up her **** for a walks right well I gotta grace he's in the bathtub Put her out the back she she needs All right I'll put her at the back yeah OK So I've been spelling there is unreal"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)

    scorable_id = Column(String, nullable=False, index=True)
    goal_id = Column(String, nullable=True, index=True)
    score = Column(Float, nullable=True)

    neighbors = Column(JSONB if JSONB else JSON, nullable=True)   # [{"nid": "...", "sim": 0.83}, ...]
    subgraph_size = Column(Integer, nullable=True)
    meta = Column(JSONB if JSONB else JSON, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts.isoformat() if self.ts else None,
            "scorable_id": self.scorable_id,
            "goal_id": self.goal_id,
            "score": self.score,
            "neighbors": self.neighbors or [],
            "subgraph_size": self.subgraph_size,
            "meta": self.meta or {},
        }
