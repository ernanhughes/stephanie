# stephanie/memory/skill_filter_store.py

from typing import Any, Dict, List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from stephanie.models.skill_filter import SkillFilterORM


class SkillFilterStore:
    """
    Store for SkillFilterORM objects.
    Provides CRUD and search utilities for skill filters.
    """

    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger or (lambda msg: print(f"[SkillFilterStore] {msg}"))
        self.name = "skill_filters"

    # --- Create ---
    def add_filter(self, data: Dict[str, Any]) -> SkillFilterORM:
        """
        Add a new SkillFilter record.
        Required keys: id, casebook_id
        """
        sf = SkillFilterORM(**data)
        self.session.add(sf)
        self.session.commit()
        self.logger(f"Added SkillFilter {sf.id}")
        return sf

    def bulk_add_filters(self, filters: List[Dict[str, Any]]) -> List[SkillFilterORM]:
        """Insert multiple SkillFilter records at once."""
        objs = [SkillFilterORM(**f) for f in filters]
        self.session.add_all(objs)
        self.session.commit()
        self.logger(f"Added {len(objs)} SkillFilters")
        return objs

    # --- Read ---
    def get_by_id(self, filter_id: str) -> Optional[SkillFilterORM]:
        return self.session.query(SkillFilterORM).filter_by(id=filter_id).first()

    def get_by_casebook(self, casebook_id: str, limit: int = 50) -> List[SkillFilterORM]:
        return (
            self.session.query(SkillFilterORM)
            .filter_by(casebook_id=casebook_id)
            .order_by(desc(SkillFilterORM.created_at))
            .limit(limit)
            .all()
        )

    def get_all(self, limit: int = 100) -> List[SkillFilterORM]:
        return (
            self.session.query(SkillFilterORM)
            .order_by(desc(SkillFilterORM.created_at))
            .limit(limit)
            .all()
        )

    def get_by_domain(self, domain: str, limit: int = 50) -> List[SkillFilterORM]:
        return (
            self.session.query(SkillFilterORM)
            .filter_by(domain=domain)
            .order_by(desc(SkillFilterORM.created_at))
            .limit(limit)
            .all()
        )

    # --- Update ---
    def update_filter(self, filter_id: str, updates: Dict[str, Any]) -> Optional[SkillFilterORM]:
        sf = self.get_by_id(filter_id)
        if not sf:
            return None
        for k, v in updates.items():
            if hasattr(sf, k):
                setattr(sf, k, v)
        self.session.commit()
        self.logger(f"Updated SkillFilter {sf.id}")
        return sf

    # --- Delete ---
    def delete_by_id(self, filter_id: str) -> bool:
        sf = self.get_by_id(filter_id)
        if not sf:
            return False
        self.session.delete(sf)
        self.session.commit()
        self.logger(f"Deleted SkillFilter {filter_id}")
        return True

    def ensure_filter(self, fid: str, data: dict) -> SkillFilterORM:
        sf = self.get_by_id(fid)
        if sf: 
            return sf
        return self.add_filter(data)

    def delete_by_casebook(self, casebook_id: str) -> int:
        count = (
            self.session.query(SkillFilterORM)
            .filter_by(casebook_id=casebook_id)
            .delete(synchronize_session=False)
        )
        self.session.commit()
        self.logger(f"Deleted {count} SkillFilters from casebook {casebook_id}")
        return count
