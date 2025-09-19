from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Callable

from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from stephanie.utils.db_scope import retry, session_scope


class BaseSQLAlchemyStore:
    """
    Generic SQLAlchemy store with standard CRUD-ish helpers.
    Subclasses must set:
        - orm_model: the ORM class
        - name:      store name (string)

    Optional:
        - id_attr:           PK column name (default: "id")
        - default_order_by:  a column object or column name string

    IMPORTANT:
    - This base uses SHORT-LIVED sessions per call.
    - Pass a sessionmaker (recommended), or a Session (legacy; will be wrapped).
    """

    orm_model: Type[Any] = None
    id_attr: str = "id"
    default_order_by: Optional[Any] = None

    def __init__(self, session_or_maker: sessionmaker, logger=None):
        self.logger = logger
        assert self.orm_model is not None, "Subclasses must set orm_model"
        self.session: sessionmaker = session_or_maker

    def _run(self, fn: Callable[[Any], Any], tries: int = 2):
        def op():
            with session_scope(self.session) as s:
                return fn(s)
        return retry(op, tries=tries)

    # -------- Standard APIs --------

    def get_by_id(self, obj_id: Any) -> Any | None:
        return self._run(lambda s: s.get(self.orm_model, obj_id))

    def count(self, **filters) -> int:
        return self._run(
            lambda s: int(
                (s.query(func.count("*"))
                   .select_from(self.orm_model)
                   .filter_by(**filters) if filters else s.query(func.count("*")).select_from(self.orm_model)
                ).scalar() or 0
            )
        )

    def exists(self, **filters) -> bool:
        return self._run(
            lambda s: bool(
                s.query(self.orm_model).filter_by(**filters).exists().scalar()
                if filters else s.query(self.orm_model).exists().scalar()
            )
        )

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[Any] = None,
        desc: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        def op(s):
            q = s.query(self.orm_model)
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

        return self._run(op)

    def delete_by_id(self, obj_id: Any) -> bool:
        def op(s):
            obj = s.get(self.orm_model, obj_id)
            if not obj:
                return False
            s.delete(obj)
            return True
        return self._run(op)

    def deactivate_by_id(self, obj_id: Any) -> bool:
        def op(s):
            obj = s.get(self.orm_model, obj_id)
            if not obj:
                return False
            if not hasattr(obj, "is_active"):
                raise AttributeError(f"{self.orm_model.__name__} has no 'is_active' field")
            obj.is_active = False
            if hasattr(obj, "updated_at"):
                obj.updated_at = datetime.now()
            return True
        return self._run(op)

    def bulk_add(self, items: List[Dict[str, Any]]) -> List[Any]:
        def op(s):
            objs = [self.orm_model(**item) for item in items]
            s.add_all(objs)
            s.flush()
            return objs
        return self._run(op)

    def get_all(self, limit: int = 100) -> List[Any]:
        return self.list(limit=limit, order_by=self.default_order_by, desc=True)

    def save(self, obj: Any) -> Any:
        def op(s):
            s.add(obj)
            s.flush()
            return obj
        return self._run(op)

    def flush(self) -> None:
        return None  # handled by session_scope
