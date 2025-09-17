# stephanie/memory/sqlalchemy_store.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from sqlalchemy import func
from sqlalchemy.orm import Session

from stephanie.memory.base import BaseStore


class BaseSQLAlchemyStore(BaseStore):
    """
    Generic SQLAlchemy store with standard CRUD-ish helpers.
    Subclasses must set:
        - orm_model: the ORM class
        - name:      store name (string)
    Optionally:
        - id_attr:   PK column name (default: "id")
        - default_order_by: a column object or column name string
    """

    orm_model: Type[Any] = None          # override in subclass
    id_attr: str = "id"
    default_order_by: Optional[Any] = None  # column or column name

    def __init__(self, session: Session, logger=None):
        super().__init__(db=session, logger=logger)
        self.session: Session = session
        self.logger = logger
        assert self.orm_model is not None, "Subclasses must set orm_model"

    # -------- Standard APIs --------

    def get_by_id(self, obj_id: Any) -> Any | None:
        # SQLAlchemy 1.4+/2.0: session.get is preferred
        return self.session.get(self.orm_model, obj_id)

    def count(self, **filters) -> int:
        q = self.session.query(func.count("*")).select_from(self.orm_model)
        if filters:
            q = q.filter_by(**filters)
        return int(q.scalar() or 0)

    def exists(self, **filters) -> bool:
        q = self.session.query(self.orm_model)
        if filters:
            q = q.filter_by(**filters)
        return self.session.query(q.exists()).scalar() or False

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[Any] = None,
        desc: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        q = self.session.query(self.orm_model)
        if filters:
            q = q.filter_by(**filters)

        col = order_by or self.default_order_by
        if isinstance(col, str):
            col = getattr(self.orm_model, col, None)
        if col is not None:
            q = q.order_by(col.desc() if desc else col.asc())

        if offset:
            q = q.offset(offset)
        if limit:
            q = q.limit(limit)
        return q.all()

    def delete_by_id(self, obj_id: Any) -> bool:
        obj = self.get_by_id(obj_id)
        if not obj:
            return False
        self.session.delete(obj)
        self.session.commit()
        return True

    def deactivate_by_id(self, obj_id: Any) -> bool:
        """Soft-delete pattern if the model has is_active/updated_at."""
        obj = self.get_by_id(obj_id)
        if not obj:
            return False
        if not hasattr(obj, "is_active"):
            raise AttributeError(f"{self.orm_model.__name__} has no 'is_active' field")
        obj.is_active = False
        if hasattr(obj, "updated_at"):
            obj.updated_at = datetime.now()
        self.session.commit()
        return True

    def bulk_add(self, items: List[Dict[str, Any]]) -> List[Any]:
        objs = [self.orm_model(**item) for item in items]
        self.session.add_all(objs)
        self.session.commit()
        return objs

    # Convenience aliases to match your current naming
    def get_all(self, limit: int = 100) -> List[Any]:
        return self.list(limit=limit, order_by=self.default_order_by, desc=True)

    # Optional low-level helpers
    def save(self, obj: Any) -> Any:
        self.session.add(obj)
        self.session.commit()
        return obj

    def flush(self) -> None:
        self.session.flush()
