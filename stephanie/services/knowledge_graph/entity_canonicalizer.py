# stephanie/services/knowledge_graph/entity_canonicalizer.py
import re

from stephanie.utils.hash_utils import hash_text


class EntityCanonicalizer:
    @staticmethod
    def normalize_surface(s: str) -> str:
        s = (s or "").strip()
        return re.sub(r"\s+", " ", s).lower()

    @staticmethod
    def canonical_id(entity_type: str, surface: str) -> str:
        et = (entity_type or "ENTITY").strip().upper()
        nh = hash_text(EntityCanonicalizer.normalize_surface(surface))
        return f"ent:{et}:{nh}"