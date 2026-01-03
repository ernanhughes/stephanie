# stephanie/services/knowledge_graph/edge_enricher.py
from typing import Dict, Optional


def enrich_relationship(
    rel: Dict,
    *,
    doc_hash: str,
    sentence_ix: Optional[int] = None,
    scorable_id: str,
    scorable_type: str,
    evidence_type: str,
) -> Dict:
    """Standardize evidence metadata onto relationships."""
    base = {
        "doc_hash": doc_hash,
        "scorable_id": scorable_id,
        "scorable_type": scorable_type,
        "evidence_type": evidence_type,
    }
    if sentence_ix is not None:
        base["sentence_ix"] = sentence_ix
    rel.setdefault("properties", {}).update(base)
    return rel