# stephanie/dataloaders/casebook_to_rlvr.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from stephanie.scoring.scorable_factory import ScorableFactory


class RLVRItem:
    def __init__(self, prompt: str, response: str, reward: float, meta: Optional[Dict[str, Any]] = None):
        self.prompt = prompt
        self.response = response
        self.reward = reward
        self.meta = meta or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "reward": self.reward,
            "meta": self.meta,
        }

    def __repr__(self) -> str:
        return f"<RLVRItem prompt={self.prompt[:20]}... reward={self.reward:.2f}>"


class CaseBookToRLVRDataset:
    def __init__(self, memory, casebook_name: str, scoring, dimensions: Optional[List[str]] = None):
        self.memory = memory
        self.casebook_name = casebook_name
        self.scoring = scoring
        self.dimensions = dimensions or ["alignment"]

    def build(self) -> List[RLVRItem]:
        casebook = self.memory.casebooks.get_by_name(self.casebook_name)
        if not casebook:
            raise ValueError(f"CaseBook '{self.casebook_name}' not found")

        dataset: List[RLVRItem] = []

        for case in casebook.cases:
            prompt = case.prompt_text or ""

            goal = self.memory.goals.get_by_id(case.goal_id)
            goal_text = goal.goal_text if goal else ""

            scorables = case.scorables

            if len(scorables) == 1:
                scorable_id = scorables[0].scorable_id
                scorable_type = scorables[0].scorable_type
                scorable = ScorableFactory.from_id(self.memory, scorable_id=scorable_id, scorable_type=scorable_type)
                source = goal_text if goal_text else prompt
                meta = {
                    "goal_id": case.goal_id,
                    "goal_text": goal_text,
                    "case_id": case.id,
                    "scorable_id": scorable.id,
                }
                # Still create RLVRItem with a baseline reward
                reward = self.container.get("scoring").score("sicql", scorable=scorable,
                                            context=_as_context(source),
                                            dimensions=self.dimensions).aggregate()
                dataset.append(RLVRItem(prompt, scorable.text, reward, meta))
                continue
            # Compare each scorable against others (pairwise, no duplicates)
            for i, sc in enumerate(scorables):
                for j, sc_other in enumerate(scorables):
                    if j <= i:  # avoid self-comparison & duplicates
                        continue

                    meta = {
                        "goal_id": case.goal_id,
                        "goal_text": goal_text,
                        "case_id": case.id,
                        "scorable_id": sc.id,
                        "competitor_id": sc_other.id,
                    }

                    source = goal_text if goal_text else prompt

                    scorable_id = sc.scorable_id
                    scorable_type = sc.scorable_type
                    scorable = ScorableFactory.from_id(self.memory, scorable_id, scorable_type)
                    # Score both scorables

                    reward_a = self.container.get("scoring").score("sicql", scorable=scorable, context=_as_context(source), dimensions=self.dimensions).aggregate()
                    scorable_other = ScorableFactory.from_orm(sc_other)
                    reward_b = self.container.get("scoring").score("sicql", scorable=scorable_other, context=_as_context(source), dimensions=self.dimensions).aggregate()

                    # Store two RLVR items with relative reward
                    dataset.append(RLVRItem(prompt, scorable.text, reward_a, meta))
                    dataset.append(RLVRItem(prompt, scorable_other.text, reward_b, meta))

        return dataset

def _as_context(source: str) -> Dict[str, Any]:
    return {"goal": {"goal_text": source}}