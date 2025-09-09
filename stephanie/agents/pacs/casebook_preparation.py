# stephanie/pipeline/stages/casebook_preparation.py
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, PIPELINE
from stephanie.dataloaders.casebook_to_rlvr import CaseBookToRLVRDataset


class CaseBookPreparationAgent(BaseAgent):
    """Prepares CaseBook data for PACS training"""

    def __init__(self, cfg, memory, logger, reporter=None):
        super().__init__(cfg, memory, logger)
        self.verifier = cfg.get("verifier", "default_verifier")
        self.reporter = reporter or getattr(self, "reporter", None)
        self.dimensions = cfg.get("dimensions", ["alignment"])

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load and prepare CaseBook for training"""
        goal = context.get(GOAL, {})
        pipeline_stage = context.get(PIPELINE, "casebook_preparation")

        try:
            await self.reporter.emit(
                ctx=context,
                stage=pipeline_stage,
                event="start",
                summary="Preparing CaseBooks for training",
                goal=goal,
            )

            # Load CaseBook from DB
            recent_casebooks = self.memory.casebooks.list_casebooks()

            dataset = []
            for cb in recent_casebooks:
                dataset_builder = CaseBookToRLVRDataset(
                    memory=self.memory,
                    casebook_name=cb.name,
                    scoring=self.scoring,
                    dimensions=self.dimensions,
                )
                dataset.extend(dataset_builder.build() or [])
                msg = f"CaseBook '{cb.name}' not found"
                self.logger.log("CaseBookNotFound", {"stage": pipeline_stage, "error": msg})
                if self.reporter:
                    await self.reporter.emit(
                        ctx=context,
                        stage=pipeline_stage,
                        event="error",
                        error=msg,
                        finalize=True,
                    )
                return context  # graceful exit


            # Add to context
            context.update(
                {
                    "casebooks": recent_casebooks,
                    "rlvr_dataset": dataset,
                    "verifier": self._get_verifier(),
                    "casebook_names": [cb.name for cb in recent_casebooks],
                }
            )

            # --- Stage success ---
            self.logger.log(
                "CaseBookPrepared",
                {
                    "stage": pipeline_stage,
                    "num_items": len(dataset),
                },
            )
            if self.reporter:
                await self.reporter.emit(
                    ctx=context,
                    stage=pipeline_stage,
                    event="done",
                    num_items=len(dataset),
                    finalize=True,
                )

            return context

        except Exception as e:
            self.logger.log(
                "CaseBookPreparationFailed",
                {"stage": pipeline_stage, "error": str(e)},
            )
            if self.reporter:
                await self.reporter.emit(
                    ctx=context,
                    stage=pipeline_stage,
                    event="error",
                    error=str(e),
                    finalize=True,
                )
            raise

    def _get_verifier(self):
        """Factory for verifier functions"""
        if self.verifier == "math":
            from stephanie.scoring.verifiers import boxed_math_verifier
            return boxed_math_verifier
        elif self.verifier == "code":
            from stephanie.scoring.verifiers import code_verifier
            return code_verifier
        else:
            return lambda prompt, response, meta: 0.0

    def _analyze_dataset(self, dataset) -> Dict[str, Any]:
        """Analyze dataset composition and return stats for reporting"""
        try:
            n = len(dataset)
            if n == 0:
                return {"count": 0, "domains": {}, "avg_len": 0}

            domains = {}
            lengths = []
            for item in dataset:
                d = getattr(item, "domain", "unknown")
                domains[d] = domains.get(d, 0) + 1
                lengths.append(len(getattr(item, "text", "")))

            return {
                "count": n,
                "domains": domains,
                "avg_len": float(sum(lengths)) / len(lengths),
                "min_len": min(lengths),
                "max_len": max(lengths),
            }
        except Exception:
            return {"count": len(dataset)}
