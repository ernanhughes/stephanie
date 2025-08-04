# stephanie/agents/knowledge/document_reward_scorer.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.contrastive_ranker_scorer import ContrastiveRankerScorer
from stephanie.scoring.ebt_scorer import EBTScorer
from stephanie.scoring.hrm_scorer import HRMScorer
from stephanie.scoring.mrq_scorer import MRQScorer
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.scoring.sicql_scorer import SICQLScorer
from stephanie.scoring.svm_scorer import SVMScorer
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from typing import Dict, List, Any
import time
import random
from tqdm import tqdm


class ScorerAgent(BaseAgent):
    """
    Scores document sections or full documents to assess reward value
    using configured reward model (e.g., SVM-based or regression-based).

    Enhanced with MARS (Model Agreement and Reasoning Signal) analysis
    to evaluate consistency across scoring models using the tensor-based architecture.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get(
            "dimensions", ["helpfulness", "truthfulness", "reasoning_quality"]
        )
        self.include_mars = cfg.get("include_mars", True)
        self.test_mode = cfg.get("test_mode", False)
        self.test_document_count = cfg.get("test_document_count", 100)
        self.embedding_types = cfg.get(
            "embedding_types", ["hnet", "hf", "mxbai"]
        )
        # Configure which scorers to use
        self.scorer_types = cfg.get(
            "scorer_types",
            ["svm", "mrq", "sicql", "ebt", "hrm", "contrastive_ranker"],
        )

        # Initialize scorers dynamically
        self.scorers = self._initialize_scorers()

        # Initialize MARS calculator with dimension-specific configurations
        dimension_config = cfg.get("dimension_config", {})
        self.mars_calculator = MARSCalculator(dimension_config, self.logger)

        self.logger.log(
            "DocumentRewardScorerInitialized",
            {
                "dimensions": self.dimensions,
                "scorers": self.scorer_types,
                "include_mars": self.include_mars,
                "test_mode": self.test_mode,
            },
        )

    def _initialize_scorers(self) -> Dict[str, Any]:
        """Initialize all configured scorers"""
        scorers = {}

        if "svm" in self.scorer_types:
            scorers["svm"] = SVMScorer(
                self.cfg, memory=self.memory, logger=self.logger
            )
        if "mrq" in self.scorer_types:
            scorers["mrq"] = MRQScorer(
                self.cfg, memory=self.memory, logger=self.logger
            )
        if "sicql" in self.scorer_types:
            scorers["sicql"] = SICQLScorer(
                self.cfg, memory=self.memory, logger=self.logger
            )
        if "ebt" in self.scorer_types:
            scorers["ebt"] = EBTScorer(
                self.cfg, memory=self.memory, logger=self.logger
            )
        if "hrm" in self.scorer_types:
            scorers["hrm"] = HRMScorer(
                self.cfg, memory=self.memory, logger=self.logger
            )
        if "contrastive_ranker" in self.scorer_types:
            scorers["contrastive_ranker"] = ContrastiveRankerScorer(
                self.cfg, memory=self.memory, logger=self.logger
            )

        return scorers

    async def run(self, context: dict) -> dict:
        """Main execution method with optional test mode"""
        start_time = time.time()

        documents = context.get(self.input_key, [])
        documents = documents[:5] if self.test_mode else documents

        if not documents:
            self.logger.log("NoDocumentsFound", {"source": self.input_key})
            return context

        # Process all documents and collect ScoreBundles
        all_bundles = {}  # scorable_id -> ScoreBundle
        results = []
        total_documents = len(documents)

        # Process documents with progress tracking
        pbar = tqdm(
            documents,
            desc="Scoring Documents",
            total=total_documents,
            disable=not self.cfg.get("progress", True),
        )

        for idx, doc in enumerate(pbar):
            try:
                # Score document with all scorers
                scoring_start = time.time()
                doc_scores, bundle = self._score_document(context, doc)
                scoring_time = time.time() - scoring_start

                # Update progress bar
                pbar.set_postfix(
                    {
                        "docs": f"{idx + 1}/{total_documents}",
                        "scorers": len(self.scorers),
                    }
                )

                # Log performance metrics
                if (idx + 1) % 10 == 0 or idx == total_documents - 1:
                    self.logger.log(
                        "DocumentScoringProgress",
                        {
                            "processed": idx + 1,
                            "total": total_documents,
                            "avg_time_per_doc": scoring_time,
                            "scorers": len(self.scorers),
                        },
                    )

                # Store results
                results.append(doc_scores)

                # Save bundle for corpus analysis
                all_bundles[doc["id"]] = bundle

            except Exception as e:
                self.logger.log(
                    "DocumentScoringError",
                    {"document_id": doc.get("id", "unknown"), "error": str(e)},
                )
                continue

        # Create ScoreCorpus for MARS analysis
        corpus = ScoreCorpus(bundles=all_bundles)

        # Save corpus to context for potential future analysis
        context["score_corpus"] = corpus.to_dict()

        # Run MARS analysis if requested
        mars_results = {}
        if self.include_mars and all_bundles:
            mars_results = self.mars_calculator.calculate(corpus)
            context["mars_analysis"] = {
                "summary": mars_results,
                "recommendations": self.mars_calculator.generate_recommendations(
                    mars_results
                ),
            }
            self.logger.log(
                "MARSAnalysisCompleted",
                {
                    "document_count": len(all_bundles),
                    "dimensions": self.dimensions,
                },
            )

        # Save results to context
        context[self.output_key] = results
        context["scoring_time"] = time.time() - start_time
        context["total_documents"] = total_documents
        context["scorers_used"] = list(self.scorers.keys())

        self.logger.log(
            "DocumentScoringComplete",
            {
                "total_documents": total_documents,
                "dimensions": self.dimensions,
                "scorers": len(self.scorers),
                "total_time": context["scoring_time"],
            },
        )

        return context

    def _score_document(self, context: dict, doc: dict) -> tuple:
        """Score a single document with all configured scorers"""
        doc_id = doc["id"]
        goal = context.get("goal", {"goal_text": ""})
        scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)

        # Collect ScoreResults for this document
        score_results = {}

        for scorer_name, scorer in self.scorers.items():
            try:
                # Score with this scorer
                score_bundle = scorer.score(
                    goal=goal,
                    scorable=scorable,
                    dimensions=self.dimensions,
                )

                # Add all results to our collection
                for dim, result in score_bundle.results.items():
                    if dim not in score_results:
                        score_results[dim] = result

            except Exception as e:
                self.logger.log(
                    "ScorerError",
                    {
                        "scorer": scorer_name,
                        "document_id": doc_id,
                        "error": str(e),
                    },
                )
                continue

        # Create ScoreBundle for this document
        bundle = ScoreBundle(results=score_results)

        # Save to memory
        ScoringManager.save_score_to_memory(
            bundle,
            scorable,
            context,
            self.cfg,
            self.memory,
            self.logger,
            source="document_reward",
            model_name="ensemble",
        )

        # Prepare results for reporting
        report_scores = {
            dim: {
                "score": result.score,
                "rationale": result.rationale,
                "source": result.source,
            }
            for dim, result in score_results.items()
        }

        return {
            "document_id": doc_id,
            "title": doc.get("title", ""),
            "scores": report_scores,
            "goal_text": goal.get("goal_text", ""),
        }, bundle
