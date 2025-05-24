from typing import List, Optional

from co_ai.memory.base_store import BaseStore
from co_ai.models.goal import Goal


class GoalStore(BaseStore):
    def __init__(self, db, logger=None):
        super().__init__(db, logger)
        self.table_name = "goals"
        self.name = "goals"

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None}>"

    def name(self) -> str:
        return "goals"


    def get_by_id(self, goal_id: int) -> Optional[Goal]:
        query = f"SELECT * FROM {self.table_name} WHERE id = %s"
        row = self.db.fetch_one(query, (goal_id,))
        goal = self._row_to_goal(row) if row else None
        if self.logger:
            self.logger.log("GoalFetchedById", {
                "goal_id": goal_id,
                "found": goal is not None
            })
        return goal

    def get_by_text(self, goal_text: str) -> Optional[Goal]:
        query = f"SELECT * FROM {self.table_name} WHERE goal_text = %s"
        try:
            with self.db.cursor() as cur:
                cur.execute(query, (goal_text,))
                row = cur.fetchone()
                goal = self._row_to_goal(row) if row else None
                if self.logger:
                    self.logger.log("GoalFetchedByText", {
                        "goal_text": goal_text[:100],
                        "found": goal is not None
                    })
                return goal
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("GoalGetFailed", {"error": str(e)})
            return None

    def get_or_create(self, goal_dict: dict) -> Goal:
        goal = Goal(**goal_dict)
        existing = self.get_by_text(goal.goal_text)
        if existing:
            if self.logger:
                self.logger.log("GoalExists", {
                    "goal_id": existing.id,
                    "goal_text": goal.goal_text[:100]
                })
            return existing
        goal.id = self.insert(goal)
        if self.logger:
            self.logger.log("GoalCreated", {
                "goal_text": goal.goal_text[:100],
                "goal_type": goal.goal_type
            })
        return goal

    def insert(self, goal: Goal) -> int:
        query = f"""
            INSERT INTO {self.table_name}
            (goal_text, goal_type, focus_area, strategy, llm_suggested_strategy, source)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    query,
                    (
                        goal.goal_text,
                        goal.goal_type,
                        goal.focus_area,
                        goal.strategy,
                        goal.llm_suggested_strategy,
                        goal.source,
                    ),
                )
                new_id = cur.fetchone()[0]
                self.db.commit()  # Commit after write
                if self.logger:
                    self.logger.log("GoalInserted", {
                        "goal_id": new_id,
                        "goal_text": goal.goal_text[:100],
                        "goal_type": goal.goal_type,
                        "focus_area": goal.focus_area,
                        "strategy": goal.strategy
                    })
                return new_id
        except Exception as e:
            self.db.rollback()
            print(f"❌ Insert failed: {e}")
            if self.logger:
                self.logger.log("GoalInsertFailed", {"error": str(e)})
            return -1

    def list_all(self) -> List[Goal]:
        query = f"SELECT * FROM {self.table_name} ORDER BY created_at DESC"
        rows = self.db.fetch_all(query)
        goals = [self._row_to_goal(row) for row in rows]
        if self.logger:
            self.logger.log("GoalsListed", {
                "count": len(goals)
            })
        return goals

    def update_strategy(self, goal_id: int, strategy: str):
        query = f"UPDATE {self.table_name} SET strategy = %s WHERE id = %s"
        try:
            self.db.execute(query, (strategy, goal_id))
            self.db.commit()
            if self.logger:
                self.logger.log("GoalStrategyUpdated", {
                    "goal_id": goal_id,
                    "new_strategy": strategy
                })
        except Exception as e:
            self.db.rollback()
            if self.logger:
                self.logger.log("GoalStrategyUpdateFailed", {"error": str(e)})

    def _row_to_goal(self, row) -> Goal:
        return Goal(
            id=row["id"],
            goal_text=row["goal_text"],
            goal_type=row["goal_type"],
            focus_area=row["focus_area"],
            strategy=row["strategy"],
            llm_suggested_strategy=row["llm_suggested_strategy"],
            source=row["source"],
            created_at=row["created_at"],
        )