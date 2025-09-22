# stephanie/models/knowledge_pair.py
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, Boolean
from datetime import datetime
from stephanie.models.base import Base

class KnowledgePairORM(Base):
    __tablename__ = "knowledge_pairs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_hash = Column(String(32), unique=True, nullable=False)  # MD5 hash of key components
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Source data
    pos_id = Column(Integer, nullable=False)  # positive turn ID
    neg_id = Column(Integer, nullable=False)  # negative turn ID
    conversation_id = Column(Integer, nullable=False)
    domain = Column(String(100), nullable=True)
    
    # Builder parameters (for cache key)
    params_hash = Column(String(32), nullable=False)  # Hash of builder params
    min_entity_overlap = Column(Integer, default=1)
    min_star_pos = Column(Integer, default=1)
    max_star_neg = Column(Integer, default=-1)
    max_negs_per_pos = Column(Integer, default=3)
    
    # Pair metadata
    label_source = Column(String(20), default="human")  # "human", "ai", "both"
    pair_weight = Column(Float, default=1.0)
    ai_conf = Column(Float, nullable=True)
    has_similar_human = Column(Boolean, default=False)
    
    # Pair content
    prompt = Column(String, nullable=False)
    output_a = Column(String, nullable=False)
    output_b = Column(String, nullable=False)
    meta_a = Column(JSON, nullable=False)
    meta_b = Column(JSON, nullable=False)
    
    # Status
    is_valid = Column(Boolean, default=True)
    invalidated_reason = Column(String(100), nullable=True)


    def __repr__(self):
        return f"<KnowledgePair(id={self.id}, pos_id={self.pos_id}, neg_id={self.neg_id}, label_source={self.label_source}, is_valid={self.is_valid})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "pair_hash": self.pair_hash,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "pos_id": self.pos_id,
            "neg_id": self.neg_id,
            "conversation_id": self.conversation_id,
            "domain": self.domain,
            "params_hash": self.params_hash,
            "min_entity_overlap": self.min_entity_overlap,
            "min_star_pos": self.min_star_pos,
            "max_star_neg": self.max_star_neg,
            "max_negs_per_pos": self.max_negs_per_pos,
            "label_source": self.label_source,
            "pair_weight": self.pair_weight,
            "ai_conf": self.ai_conf,
            "has_similar_human": self.has_similar_human,
            "prompt": self.prompt,
            "output_a": self.output_a,
            "output_b": self.output_b,
            "meta_a": self.meta_a,
            "meta_b": self.meta_b,
            "is_valid": self.is_valid,
            "invalidated_reason": self.invalidated_reason,
        } 
    