# stephanie/agents/knowledge/paper_score.py

import time
from typing import Dict, Any

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.scoring.svm_scorer import SVMScorer
from stephanie.scoring.mrq_scorer import MRQScorer
from stephanie.scoring.sicql_scorer import SICQLScorer
from stephanie.scoring.ebt_scorer import EBTScorer
from stephanie.scoring.hrm_scorer import HRMScorer
from stephanie.scoring.contrastive_ranker_scorer import ContrastiveRankerScorer
from stephanie.scoring.calculations.mars_calculator import MARSCalculator


class PaperScoreAgent(BaseAgent):
    """
    Scores academic papers (e.g. from Arxiv) across multiple scorers.
    Similar design to DocumentRewardScorer, but specialized for research papers.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get(
            "dimensions",
            ["novelty", "clarity", "relevance", "implementability", "alignment"],
        )
        self.include_mars = cfg.get("include_mars", True)
        self.scorer_types = cfg.get(
            "scorer_types",
            ["svm", "mrq", "sicql", "ebt", "hrm", "contrastive_ranker"],
        )

        # Initialize scorers dynamically
        self.scorers = self._initialize_scorers()

        # Initialize MARS calculator
        dimension_config = cfg.get("dimension_config", {})
        self.mars_calculator = MARSCalculator(dimension_config, self.memory, self.logger)

        self.logger.log(
            "PaperScoreAgentInitialized",
            {
                "dimensions": self.dimensions,
                "scorers": self.scorer_types,
                "include_mars": self.include_mars,
            },
        )

    def _initialize_scorers(self) -> Dict[str, Any]:
        scorers = {}
        if "svm" in self.scorer_types:
            scorers["svm"] = SVMScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "mrq" in self.scorer_types:
            scorers["mrq"] = MRQScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "sicql" in self.scorer_types:
            scorers["sicql"] = SICQLScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "ebt" in self.scorer_types:
            scorers["ebt"] = EBTScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "hrm" in self.scorer_types:
            scorers["hrm"] = HRMScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "contrastive_ranker" in self.scorer_types:
            scorers["contrastive_ranker"] = ContrastiveRankerScorer(self.cfg, memory=self.memory, logger=self.logger)
        return scorers

    async def run(self, context: dict) -> dict:
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
        scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)

        score_results = {}
        for scorer_name, scorer in self.scorers.items():
            try:
                bundle = scorer.score(context, scorable=scorable, dimensions=self.dimensions)
                for dim, result in bundle.results.items():
                    score_results[dim] = result
            except Exception as e:
                self.logger.log("ScorerError", {"scorer": scorer_name, "doc_id": doc_id, "error": str(e)})
                continue

        bundle = ScoreBundle(results=score_results)

        # Save to memory
        ScoringManager.save_score_to_memory(
            bundle,
            scorable,
            context,
            self.cfg,
            self.memory,
            self.logger,
            source="paper_score",
            model_name="ensemble",
        )

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
