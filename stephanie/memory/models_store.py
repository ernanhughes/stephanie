from __future__ import annotations
from typing import Optional, List
from datetime import datetime
from sqlalchemy import func
from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.model_artifact import ModelArtifactORM

class ModelsStore(BaseSQLAlchemyStore):
    orm_model = ModelArtifactORM
    default_order_by = ModelArtifactORM.id.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "models"

    # --- helpers ---

    def _next_version(self, s, name: str) -> int:
        """Compute next monotonically increasing version for a given model name."""
        last = (
            s.query(func.max(ModelArtifactORM.version))
            .filter(ModelArtifactORM.name == name)
            .scalar()
        )
        return int(last or 0) + 1

    # --- API ---

    def register(self, *, name: str, path: str, meta: Optional[dict] = None, tag: Optional[str] = None) -> ModelArtifactORM:
        """
        Register a new model artifact. Auto-increments version per name.
        Returns the created ORM row.
        """
        def op(s):
            ver = self._next_version(s, name)
            row = ModelArtifactORM(
                name=name,
                version=ver,
                path=path,
                tag=tag,
                meta=meta or {},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            s.add(row)
            s.flush()
            return row
        row = self._run(op)
        if self.logger:
            self.logger.info(f"[ModelsStore] Registered {name} v{row.version} at {path}")
        return row

    def latest(self, name: str) -> Optional[ModelArtifactORM]:
        """Fetch latest version by name."""
        def op(s):
            return (
                s.query(ModelArtifactORM)
                .filter(ModelArtifactORM.name == name)
                .order_by(ModelArtifactORM.version.desc())
                .first()
            )
        return self._run(op)

    def get(self, name: str, version: int) -> Optional[ModelArtifactORM]:
        def op(s):
            return (
                s.query(ModelArtifactORM)
                .filter(ModelArtifactORM.name == name, ModelArtifactORM.version == int(version))
                .one_or_none()
            )
        return self._run(op)

    def list(self, name: Optional[str] = None, limit: int = 100) -> List[ModelArtifactORM]:
        def op(s):
            q = s.query(ModelArtifactORM)
            if name:
                q = q.filter(ModelArtifactORM.name == name)
            return q.order_by(ModelArtifactORM.name.asc(), ModelArtifactORM.version.desc()).limit(limit).all()
        return self._run(op)

    def update_meta(self, name: str, version: int, meta: dict) -> Optional[ModelArtifactORM]:
        def op(s):
            row = (
                s.query(ModelArtifactORM)
                .filter(ModelArtifactORM.name == name, ModelArtifactORM.version == int(version))
                .one_or_none()
            )
            if not row:
                return None
            row.meta = meta or {}
            row.updated_at = datetime.now()
            s.add(row)
            s.flush()
            return row
        return self._run(op)

    def retag(self, name: str, version: int, tag: Optional[str]) -> Optional[ModelArtifactORM]:
        def op(s):
            row = (
                s.query(ModelArtifactORM)
                .filter(ModelArtifactORM.name == name, ModelArtifactORM.version == int(version))
                .one_or_none()
            )
            if not row:
                return None
            row.tag = tag
            row.updated_at = datetime.now()
            s.add(row)
            s.flush()
            return row
        return self._run(op)
