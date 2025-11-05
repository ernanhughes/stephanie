# stephanie/agents/knowledge/paper_score.py
from __future__ import annotations

import logging
import time
from typing import Dict

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.scorable import ScorableFactory, ScorableType

log = logging.getLogger(__name__)

class PaperScoreAgent(BaseAgent):
    """
    Scores academic papers (e.g. from Arxiv) across multiple scorers.
    Similar design to DocumentRewardScorer, but specialized for research papers.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.dimensions = cfg.get(
            "dimensions",
            ["novelty", "clarity", "relevance", "implementability", "alignment"],
        )
        self.include_mars = cfg.get("include_mars", True)
        self.enabled_scorers = cfg.get(
            "enabled_scorers",
            ["svm", "mrq", "sicql", "ebt", "hrm", "contrastive_ranker"],
        )

        # Initialize MARS calculator
        self.enabled_scorers = cfg.get("enabled_scorers", [])

        log.debug(
            "PaperScoreAgentInitialized:"
            f"dimensions={self.dimensions}, "
            f"scorers={self.enabled_scorers}, "
            f"include_mars={self.include_mars}"
        )

    async def run(self, context: Dict) -> Dict:
        """Score all papers in the context"""
        start_time = time.time()
        documents = context.get(self.input_key, [])

        if not documents:
            self.logger.log("NoPapersFound", {"source": self.input_key})
            return context

        self.report({"event": "start", "step": "PaperScoring", "details": f"{len(documents)} papers"})

        all_bundles = {}
        results = []

        pbar = tqdm(documents, desc="Scoring Papers", total=len(documents), disable=not self.cfg.get("progress", True))

        for idx, doc in enumerate(pbar):
            try:
                doc_scores, bundle = self._score_paper(context, doc)
                results.append(doc_scores)
                all_bundles[doc["id"]] = bundle
            except Exception as e:
                self.logger.log("PaperScoringError", {"doc_id": doc.get("id"), "error": str(e)})
                continue

        # Run MARS analysis
        if self.include_mars and all_bundles:
            corpus = ScoreCorpus(bundles=all_bundles)
            self.logger.log("ScoreCorpusSummary", {
                "dims": corpus.dimensions,
                "scorers": corpus.scorers,
                "shape_example": corpus.get_dimension_matrix(self.dimensions[0]).shape
            })

            mars_results = self.mars_calculator.calculate(corpus, context=context)
            context["mars_analysis"] = {
                "summary": mars_results,
                "recommendations": self.mars_calculator.generate_recommendations(mars_results),
            }

        context[self.output_key] = results
        context["scoring_time"] = time.time() - start_time
        context["total_documents"] = len(documents)

        self.report({"event": "end", "step": "PaperScoring", "details": f"Scored {len(documents)} papers"})
        return context

    def _score_paper(self, context: dict, doc: dict) -> tuple:
        """Score one paper with all scorers"""
        doc_id = doc["id"]
        goal = context.get("goal", {"goal_text": ""})
        scorable = ScorableFactory.from_dict(doc, ScorableType.DOCUMENT)

        score_results = {}

        for scorer_name in self.enabled_scorers:
            try:
                bundle = self.container.get("scoring").score(
                    scorer_name,
                    context=context,
                    scorable=scorable,
                    dimensions=self.dimensions
                )
                for dim, result in bundle.results.items():
                    # ensure the result carries its dimension and source
                    if not getattr(result, "dimension", None):
                        result.dimension = dim
                    if not getattr(result, "source", None):
                        result.source = scorer_name  # fallback if scorer didn't set it

                    # use a composite key to avoid overwriting, but keep result.dimension == dim
                    key = f"{dim}::{result.source}"
                    score_results[key] = result
            except Exception as e:
                self.logger.log("ScorerError", {"scorer": scorer_name, "doc_id": doc_id, "error": str(e)})
                continue

        bundle = ScoreBundle(results=dict(score_results))

        eval_id = self.memory.evaluations.save_bundle(
            bundle=bundle,
            scorable=scorable,
            context=context,
            cfg=self.cfg,
            source="paper_score",
            agent_name=self.name,
            model_name="ensemble",
            evaluator_name=str(self.enabled_scorers)
        )
        self.logger.log("EvaluationSaved", {"id": eval_id})


        report_scores = {
            dim: {"score": result.score, "rationale": result.rationale, "source": result.source}
            for dim, result in score_results.items()
        }

        return {
            "document_id": doc_id,
            "title": doc.get("title", ""),
            "scores": report_scores,
            "goal_text": goal.get("goal_text", ""),
        }, bundle
