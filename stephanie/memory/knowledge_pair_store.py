# stephanie/memory/knowledge_pair_store.py
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from stephanie.orm.knowledge_pair import KnowledgePairORM
from stephanie.utils.hash_utils import hash_text


class KnowledgePairStore:
    """Manages persistent storage and retrieval of knowledge pairs"""
    
    def __init__(self, memory):
        self.memory = memory
    
    def save_pairs(self, pairs: List[Dict[str, Any]], params: Dict[str, Any]) -> int:
        """Save knowledge pairs to database with parameter hash"""
        params_hash = self._hash_params(params)
        saved_count = 0
        
        for pair in pairs:
            # Create ORM object
            pair_orm = KnowledgePairORM(
                pair_hash=pair["pair_hash"],
                params_hash=params_hash,
                min_entity_overlap=params.get("min_entity_overlap", 1),
                min_star_pos=params.get("min_star_pos", 1),
                max_star_neg=params.get("max_star_neg", -1),
                max_negs_per_pos=params.get("max_negs_per_pos", 3),
                # Other fields
                pos_id=pair["pos_id"],
                neg_id=pair["neg_id"],
                conversation_id=pair["conversation_id"],
                domain=pair["domain"],
                label_source=pair["label_source"],
                pair_weight=pair["pair_weight"],
                ai_conf=pair.get("ai_conf"),
                has_similar_human=pair["has_similar_human"],
                prompt=pair["prompt"],
                output_a=pair["output_a"],
                output_b=pair["output_b"],
                meta_a=pair["meta_a"],
                meta_b=pair["meta_b"]
            )
            
            # Save to DB (upsert)
            self.memory.session.merge(pair_orm)
            saved_count += 1
        
        self.memory.session.commit()
        return saved_count
    
    def get_pairs(
        self,
        params: Dict[str, Any],
        limit: int = 50000,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Get knowledge pairs, using cache when possible"""
        if force_refresh:
            return []
            
        params_hash = self._hash_params(params)
        return self._get_cached_pairs(params_hash, limit)
    
    def _get_cached_pairs(self, params_hash: str, limit: int) -> List[Dict[str, Any]]:
        """Get pairs from database with matching params hash"""
        def op(session):
            query = (
                session.query(KnowledgePairORM)
                .filter(KnowledgePairORM.params_hash == params_hash)
                .filter(KnowledgePairORM.is_valid == True)
                .order_by(KnowledgePairORM.created_at.desc())
                .limit(limit)
            )
            results = query.all()
            
            # Convert to dict format
            return [{
                "pair_hash": r.pair_hash,
                "pos_id": r.pos_id,
                "neg_id": r.neg_id,
                "conversation_id": r.conversation_id,
                "domain": r.domain,
                "label_source": r.label_source,
                "pair_weight": r.pair_weight,
                "ai_conf": r.ai_conf,
                "has_similar_human": r.has_similar_human,
                "prompt": r.prompt,
                "output_a": r.output_a,
                "output_b": r.output_b,
                "meta_a": r.meta_a,
                "meta_b": r.meta_b
            } for r in results]
        
        return self.memory._run(op)
    
    def invalidate_pairs(self, params: Dict[str, Any], reason: str):
        """Mark existing pairs as invalid when underlying data changes"""
        params_hash = self._hash_params(params)
        
        def op(session):
            session.query(KnowledgePairORM).filter(
                KnowledgePairORM.params_hash == params_hash
            ).update({
                KnowledgePairORM.is_valid: False,
                KnowledgePairORM.invalidated_reason: reason
            })
            session.commit()
        
        self.memory._run(op)
    
    def _hash_params(self, params: Dict[str, Any]) -> str:
        # include ALL knobs that can change pair selection
        keys = [
            "min_star_pos", "max_star_neg", "min_entity_overlap",
            "max_negs_per_pos", "ai_score_threshold", "ai_confidence_threshold",
            "shuffle", "builder_version"
        ]
        # stable JSON to avoid key ordering problems
        payload = {k: params.get(k) for k in keys}
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hash_text(s)
