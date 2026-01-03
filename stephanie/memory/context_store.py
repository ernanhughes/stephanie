# stephanie/memory/context_store.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import yaml

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.context_state import ContextStateORM


class ContextStore(BaseSQLAlchemyStore):
    orm_model = ContextStateORM
    default_order_by = ContextStateORM.timestamp.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "contexts"
        self.dump_dir = logger.log_path if logger else None
        if self.dump_dir:
            self.dump_dir = os.path.dirname(self.dump_dir)

    def save(
        self,
        run_id: str,
        stage: str,
        context: dict,
        preferences: dict = None,
        extra_data: dict = None,
    ):
        """
        Saves the current pipeline context to DB (and optionally to disk).
        Increments version and marks it as current for this stage/run.
        """
        def op(s):
            
                # Deactivate previous versions
                prev_versions = (
                    s.query(ContextStateORM)
                    .filter_by(run_id=run_id, stage_name=stage)
                    .all()
                )
                for state in prev_versions:
                    state.is_current = False

                latest_version = max((s.version for s in prev_versions), default=0)
                new_version = latest_version + 1

                db_context = ContextStateORM(
                    pipeline_run_id=context.get("pipeline_run_id"),
                    goal_id=context.get("goal", {}).get("id"),
                    run_id=run_id,
                    stage_name=stage,
                    version=new_version,
                    is_current=True,
                    context=json.dumps(context),
                    preferences=json.dumps(preferences) if preferences else None,
                    extra_data=json.dumps(extra_data or {}),
                    timestamp=datetime.now(timezone.utc),
                )

                s.add(db_context)
                s.flush()

                if self.dump_dir:
                    self._dump_to_yaml(stage, context)

                if self.logger:
                    self.logger.log(
                        "ContextSaved",
                        {
                            "run_id": run_id,
                            "stage": stage,
                            "version": new_version,
                            "timestamp": db_context.timestamp.isoformat(),
                            "is_current": True,
                        },
                    )
                return db_context
        return self._run(op)

    def has_completed(self, goal_id: int, stage_name: str) -> bool:
        def op(s):
            return (
                s.query(ContextStateORM)
                .filter_by(stage_name=stage_name, goal_id=goal_id)
                .count()
            )
        return self._run(op)

    def load(self, run_id: str, stage: Optional[str] = None) -> dict:
        def op(s):
            if stage:
                states = (
                    s.query(ContextStateORM)
                    .filter_by(stage_name=stage, run_id=run_id)
                    .order_by(ContextStateORM.timestamp.asc())
                    .all()
                )
            else:
                states = s.query(ContextStateORM).filter_by(run_id=run_id).all()

            result = {}
            for state in states:
                result.update(json.loads(state.context))
            return result
        return self._run(op)

    def get_latest(self, run_id: str) -> Optional[ContextStateORM]:
        def op(s):
            return (
                s.query(ContextStateORM)
                .filter(ContextStateORM.run_id == run_id)
                .order_by(ContextStateORM.timestamp.desc())
                .first()
            )
        return self._run(op)

    def get_previous(self, run_id: str) -> Optional[ContextStateORM]:
        def op(s):
            results = (
                s.query(ContextStateORM)
                .filter(ContextStateORM.run_id == run_id)
                .order_by(ContextStateORM.timestamp.desc())
                .limit(2)
                .all()
            )
            return results[1] if len(results) >= 2 else None
        return self._run(op)

    def _dump_to_yaml(self, stage: str, context: dict):
        os.makedirs(self.dump_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{stage}_{timestamp}.yaml"
        path = os.path.join(self.dump_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(context, f, allow_unicode=True, sort_keys=False)
            if self.logger:
                self.logger.log("ContextYAMLDumpSaved", {"path": path})
        except Exception as e:
            if self.logger:
                self.logger.log("ContextYAMLDumpFailed", {"error": str(e)})
