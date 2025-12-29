# stephanie/memory/skill_filter_store.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import desc

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.skill_filter import SkillFilterORM


class SkillFilterStore(BaseSQLAlchemyStore):
    """
    Store for SkillFilterORM objects.
    Provides CRUD and search utilities for skill filters.
    """
    orm_model = SkillFilterORM
    default_order_by = SkillFilterORM.created_at

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "skill_filters"

    # --- Create ---
    def add_filter(self, data: Dict[str, Any]) -> SkillFilterORM:
        """Add a new SkillFilter record."""
        def op(s):
            sf = SkillFilterORM(**data)
            s.add(sf)
            s.flush()
            s.refresh(sf)
            return sf

        result = self._run(op, commit=True)
        if self.logger:
            self.logger.log("SkillFilterAdded", {"id": result.id})
        return result

    def bulk_add_filters(self, filters: List[Dict[str, Any]]) -> List[SkillFilterORM]:
        """Insert multiple SkillFilter records at once."""
        def op(s):
            objs = [SkillFilterORM(**f) for f in filters]
            s.add_all(objs)
            s.flush()
            for o in objs:
                s.refresh(o)
            return objs

        result = self._run(op, commit=True)
        if self.logger:
            self.logger.log("SkillFilterBulkAdded", {"count": len(result)})
        return result

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
        def op(s):
            sf = s.query(SkillFilterORM).filter_by(id=filter_id).first()
            if not sf:
                return None
            for k, v in updates.items():
                if hasattr(sf, k):
                    setattr(sf, k, v)
            s.flush()
            s.refresh(sf)
            return sf

        result = self._run(op, commit=True)
        if result and self.logger:
            self.logger.log("SkillFilterUpdated", {"id": result.id, "updates": updates})
        return result

    # --- Delete ---
    def delete_by_id(self, filter_id: str) -> bool:
        def op(s):
            sf = s.query(SkillFilterORM).filter_by(id=filter_id).first()
            if not sf:
                return False
            s.delete(sf)
            return True

        result = self._run(op, commit=True)
        if result and self.logger:
            self.logger.log("SkillFilterDeleted", {"id": filter_id})
        return result

    def ensure_filter(self, fid: str, data: dict) -> SkillFilterORM:
        existing = self.get_by_id(fid)
        if existing:
            return existing
        return self.add_filter(data)

    def delete_by_casebook(self, casebook_id: str) -> int:
        def op(s):
            count = (
                s.query(SkillFilterORM)
                .filter_by(casebook_id=casebook_id)
                .delete(synchronize_session=False)
            )
            return count

        count = self._run(op, commit=True)
        if self.logger:
            self.logger.log("SkillFiltersDeletedByCasebook", {
                "casebook_id": casebook_id,
                "count": count
            })
        return count
