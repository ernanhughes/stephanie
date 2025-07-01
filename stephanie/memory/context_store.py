# stores/context_store.py
import json
import os
from datetime import datetime, timezone
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from stephanie.models.context_state import ContextStateORM


class ContextStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "context"
        self.dump_dir = logger.log_path if logger else None
        if self.dump_dir:
            self.dump_dir = os.path.dirname(self.dump_dir)

    def save(self, run_id: str, stage: str, context: dict, preferences: dict = None, extra_data: dict = None):
        """
        Saves the current pipeline context to database and optionally to disk.
        Increments version and marks it as current for this stage/run.
        """
        try:
            # Deactivate previous versions
            prev_versions = self.session.query(ContextStateORM).filter_by(run_id=run_id, stage_name=stage).all()
            for state in prev_versions:
                state.is_current = False

            # Get latest version number
            latest_version = max((s.version for s in prev_versions), default=0)
            new_version = latest_version + 1

            # Create new context state
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
                timestamp=datetime.now(timezone.utc)
            )

            self.session.add(db_context)
            self.session.flush()  # To get ID immediately

            if self.dump_dir:
                self._dump_to_yaml(stage, context)

            if self.logger:
                self.logger.log("ContextSaved", {
                    "run_id": run_id,
                    "stage": stage,
                    "version": new_version,
                    "timestamp": db_context.timestamp.isoformat(),
                    "is_current": True
                })
            self.session.commit()

        except Exception as e:
            self.session.rollback()
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("ContextSaveFailed", {"error": str(e)})
            raise

    def has_completed(self, goal_id:int, stage_name: str) -> bool:
        """Check if this stage has already been run"""
        count = (
            self.session.query(ContextStateORM)
            .filter_by(stage_name=stage_name, goal_id=goal_id)
            .count()
        )
        return count > 0

    def load(self, goal_id:int, stage: Optional[str] = None) -> dict:
        try:
            session = self.session if self.session.is_active else self.sessionmaker()

            if stage:
                states = (
                    session.query(ContextStateORM)
                    .filter_by(stage_name=stage, goal_id=goal_id)
                    .order_by(ContextStateORM.timestamp.asc())
                    .all()
                )
            else:
                states = session.query(ContextStateORM).filter_by(run_id=run_id).all()

            result = {}
            for state in states:
                result.update(json.loads(state.context))

            return result

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.session.rollback()
            if self.logger:
                self.logger.log("ContextLoadFailed", {"error": str(e)})
            return {}

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
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("ContextYAMLDumpFailed", {"error": str(e)})