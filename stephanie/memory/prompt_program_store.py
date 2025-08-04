# stephanie/memory/prompt_program_store.py
from typing import List, Optional

from sqlalchemy.orm import Session

from stephanie.models.prompt import PromptProgramORM


class PromptProgramStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "prompt_programs"
        self.table_name = "prompt_programs"

    def insert(self, prompt_dict: dict) -> PromptProgramORM:
        prompt = PromptProgramORM(**prompt_dict)
        self.session.add(prompt)
        self.session.commit()
        self.session.refresh(prompt)
        return prompt

    def add_prompt(self, prompt: PromptProgramORM) -> PromptProgramORM:
        self.session.add(prompt)
        self.session.commit()
        self.session.refresh(prompt)
        return prompt

    def get_by_id(self, prompt_id: str) -> Optional[PromptProgramORM]:
        return self.session.query(PromptProgramORM).filter_by(id=prompt_id).first()

    def get_all_prompts(self) -> List[PromptProgramORM]:
        return (
            self.session.query(PromptProgramORM)
            .order_by(PromptProgramORM.version.desc())
            .all()
        )

    def get_prompts_for_goal(self, goal_text: str) -> List[PromptProgramORM]:
        return (
            self.session.query(PromptProgramORM)
            .filter(PromptProgramORM.goal == goal_text)
            .order_by(PromptProgramORM.version.desc())
            .all()
        )

    def get_top_prompts(
        self, goal_text: str, min_value: float = 0.0, top_k: int = 5
    ) -> List[PromptProgramORM]:
        return (
            self.session.query(PromptProgramORM)
            .filter(
                PromptProgramORM.goal == goal_text,
                PromptProgramORM.score >= min_value,
            )
            .order_by(PromptProgramORM.score.desc().nullslast())
            .limit(top_k)
            .all()
        )

    def get_prompt_lineage(self, prompt_id: str) -> List[PromptProgramORM]:
        prompt = self.get_by_id(prompt_id)
        if not prompt:
            return []
        lineage = [prompt]
        while prompt.parent_id:
            prompt = self.get_by_id(prompt.parent_id)
            if prompt:
                lineage.insert(0, prompt)
            else:
                break
        return lineage

    def get_latest_prompt(self, goal_text: str) -> Optional[PromptProgramORM]:
        return (
            self.session.query(PromptProgramORM)
            .filter(PromptProgramORM.goal == goal_text)
            .order_by(PromptProgramORM.version.desc())
            .first()
        )
