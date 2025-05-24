import json

from co_ai.memory.base_store import BaseStore
from co_ai.models.lookahead import Lookahead
from co_ai.constants import PIPELINE


class LookaheadStore(BaseStore):
    def __init__(self, db, logger=None):
        super().__init__(db, logger)
        self.name = "lookahead"

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None}>"

    def name(self) -> str:
        return "lookahead"

    def insert(self, goal_id: int, result: Lookahead):
        query = """
            INSERT INTO lookaheads (
                goal_id,
                agent_name,
                model_name,
                input_pipeline,
                suggested_pipeline,
                rationale,
                reflection,
                backup_plans,
                metadata,
                run_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    query,
                    (
                        goal_id,
                        result.agent_name,
                        result.model_name,
                        result.input_pipeline,
                        result.suggested_pipeline,
                        result.rationale,
                        result.reflection,
                        result.backup_plans,
                        json.dumps(result.metadata or {}),  # <-- Convert dict to JSON string
                        result.run_id,
                    ),
                )
                if self.logger:
                    self.logger.log(
                        "LookaheadInserted",
                        {
                            "goal_id": goal_id,
                            "agent": result.agent_name,
                            "model": result.model_name,
                            PIPELINE: result.input_pipeline,
                            "suggested_pipeline": result.suggested_pipeline,
                            "rationale_snippet": (result.rationale or "")[:100],
                        },
                    )
            return None
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("LookaheadInsertFailed", {"error": str(e)})
            return None

    def list_all(self) -> list[Lookahead]:
        query = "SELECT * FROM lookaheads ORDER BY created_at DESC"
        rows = self.db.fetch_all(query)
        results = [self._row_to_result(row) for row in rows]

        if self.logger:
            self.logger.log("LookaheadsListed", {"count": len(results)})

        return results

    def _row_to_result(self, row) -> Lookahead:
        return Lookahead(
            goal=row["goal_id"],
            agent_name=row["agent_name"],
            model_name=row["model_name"],
            input_pipeline=row["input_pipeline"],
            suggested_pipeline=row["suggested_pipeline"],
            rationale=row["rationale"],
            reflection=row["reflection"],
            backup_plans=row["backup_plans"],
            metadata=row["metadata"],
            run_id=row["run_id"],
            created_at=row["created_at"],
        )
