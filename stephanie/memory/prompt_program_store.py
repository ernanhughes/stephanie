# stephanie/memory/prompt_program_store.py
from __future__ import annotations

from typing import List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.prompt import PromptProgramORM


class PromptProgramStore(BaseSQLAlchemyStore):
    orm_model = PromptProgramORM
    default_order_by = PromptProgramORM.version.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "prompt_programs"
        self.table_name = "prompt_programs"

    # --------------------
    # INSERT / ADD
    # --------------------
    def insert(self, prompt_dict: dict) -> PromptProgramORM:
        def op(s):
            
            prompt = PromptProgramORM(**prompt_dict)
            s.add(prompt)
            s.flush()
            return prompt
        return self._run(op)

    def add_prompt(self, prompt: PromptProgramORM) -> PromptProgramORM:
        def op(s):
            
            s.add(prompt)
            s.flush()
            return prompt
        return self._run(op)

    # --------------------
    # RETRIEVAL
    # --------------------
    def get_by_id(self, prompt_id: str) -> Optional[PromptProgramORM]:
        return self._run(lambda s: s.query(PromptProgramORM).filter_by(id=prompt_id).first())

    def get_all_prompts(self) -> List[PromptProgramORM]:
        return self._run(lambda s: s.query(PromptProgramORM).order_by(PromptProgramORM.version.desc()).all())

    def get_prompts_for_goal(self, goal_text: str) -> List[PromptProgramORM]:
        return self._run(lambda s: (
            s.query(PromptProgramORM)
            .filter(PromptProgramORM.goal == goal_text)
            .order_by(PromptProgramORM.version.desc())
            .all()
        ))

    def get_top_prompts(self, goal_text: str, min_value: float = 0.0, top_k: int = 5) -> List[PromptProgramORM]:
        return self._run(lambda s: (
            s.query(PromptProgramORM)
            .filter(
                PromptProgramORM.goal == goal_text,
                PromptProgramORM.score >= min_value,
            )
            .order_by(PromptProgramORM.score.desc().nullslast())
            .limit(top_k)
            .all()
        ))

    def get_prompt_lineage(self, prompt_id: str) -> List[PromptProgramORM]:
        def op(s):
            lineage: list[PromptProgramORM] = []
            current = self.get_by_id(prompt_id)
            while current:
                lineage.insert(0, current)
                if current.parent_id:
                    current = self.get_by_id(current.parent_id)
                else:
                    break
            return lineage
        return self._run(op)

    def get_latest_prompt(self, goal_text: str) -> Optional[PromptProgramORM]:
        return self._run(lambda s: (
            s.query(PromptProgramORM)
            .filter(PromptProgramORM.goal == goal_text)
            .order_by(PromptProgramORM.version.desc())
            .first()
        ))
