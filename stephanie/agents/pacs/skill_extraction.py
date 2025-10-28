# stephanie/agents/pacs/skill_extraction.py
from __future__ import annotations
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, PIPELINE
from stephanie.dataloaders.casebook_to_rlvr import CaseBookToRLVRDataset


class CaseBookPreparationAgent(BaseAgent):
    """Prepares CaseBook data for PACS training (RLVR dataset + verifier)."""

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.casebook_name = cfg.get("casebook_name", "default_casebook")
        self.verifier = cfg.get("verifier", "default_verifier")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get(GOAL, {})
        pipeline_stage = context.get(PIPELINE, "casebook_preparation")

        try:
            # 1. Load CaseBook via memory abstraction
            casebook = self.memory.casebooks.get_by_name(self.casebook_name)
            if not casebook:
                raise ValueError(f"CaseBook '{self.casebook_name}' not found")

            # 2. Build RLVR dataset from CaseBook
            dataset_builder = CaseBookToRLVRDataset(
                memory=self.memory,   # use memory tool, not raw session
                casebook=casebook
            )
            dataset = dataset_builder.build()

            # 3. Pick verifier
            verifier_fn = self._get_verifier()

            # 4. Update context
            context.update({
                "casebook": casebook.to_dict(include_cases=False),
                "rlvr_dataset": dataset,
                "verifier": verifier_fn,
                "casebook_name": self.casebook_name,
            })

            # 5. Persist report
            try:
                self.memory.reports.add_report({
                    "stage": pipeline_stage,
                    "casebook": self.casebook_name,
                    "goal": goal.get("goal_text", ""),
                    "num_items": len(dataset),
                })
            except Exception as e:
                self.logger.log("CaseBookReportFailed", {"error": str(e)})

            self.logger.log("CaseBookPrepared", {
                "stage": pipeline_stage,
                "casebook": self.casebook_name,
                "num_items": len(dataset),
            })

            return context

        except Exception as e:
            self.logger.log("CaseBookPreparationFailed", {
                "stage": pipeline_stage,
                "error": str(e),
            })
            raise

    def _get_verifier(self):
        """Resolve verifier from config or fallback."""
        if callable(self.verifier):
            return self.verifier
        if self.verifier == "math":
            from stephanie.scoring.verifiers import boxed_math_verifier
            return boxed_math_verifier
        elif self.verifier == "code":
            from stephanie.scoring.verifiers import code_verifier
            return code_verifier
        return lambda prompt, response, meta: 0.0
