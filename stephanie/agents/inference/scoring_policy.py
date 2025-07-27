
from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.inference.ebt_inference import EBTInferenceAgent
from stephanie.scoring.ebt.buffer import EBTTrainingBuffer
from stephanie.scoring.ebt_scorer import EBTScorer
from stephanie.scoring.llm_scorer import LLMScorer
from stephanie.scoring.mrq_scorer import MRQScorer
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.sicql_scorer import SICQLScorer
from stephanie.scoring.svm_scorer import SVMScorer


class ScoringPolicyAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.source = cfg.get("source", "mrq")

        self.dimensions = cfg.get("dimensions", [])

        self.ebt_refine_threshold = cfg.get("ebt_refinement_threshold", 0.7)
        self.llm_fallback_threshold = cfg.get("llm_fallback_threshold", 0.9)
        self.steps = cfg.get("optimization_steps", 10)
        self.step_size = cfg.get("step_size", 0.05)

        self.training_buffer_path = cfg.get("training_buffer_path", "ebt_buffer.json")
        self.ebt = EBTInferenceAgent(cfg, memory=memory, logger=logger)
        self.training_buffer = EBTTrainingBuffer(self.logger, self.training_buffer_path)


    async def run(self, context: dict) -> dict:
        goal = context["goal"]
        docs = context[self.input_key]
        results = []
        event_ids = []

        svm_scorer = SVMScorer(self.cfg, memory=self.memory, logger=self.logger)
        mrq_scorer = MRQScorer(self.cfg, memory=self.memory, logger=self.logger)
        sicql_scorer = SICQLScorer(self.cfg, memory=self.memory, logger=self.logger)  
        ebt_scorer = EBTScorer(self.cfg, memory=self.memory, logger=self.logger)
        llm_scorer = LLMScorer(self.cfg, memory=self.memory, logger=self.logger)

        for doc in docs:
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            mrq_scores = mrq_scorer.score(
                goal=goal,
                scorable=scorable,
                dimensions=self.dimensions,
            )

            ebt_result = ebt_scorer.score(
                goal=goal,
                scorable=scorable,
                dimensions=self.dimensions,
            )

            llm_scores = llm_scorer.score(
                context=context,
                scorable=scorable,
                dimensions=self.dimensions,
            )

            self.logger.log("MRQScoresCalculated", {"scorable": scorable.id, "scores": mrq_scores})

            ebt_energy = ebt_result["energy"]
            uncertainty_by_dim = ebt_result["uncertainty_by_dimension"]


            refined = False
            refined_text = None
            if any(u > self.ebt_refine_threshold for u in uncertainty_by_dim.values()):
                refined = True
                refined_result = self.ebt.optimize(goal, scorable.text)
                refined_text = refined_result.get("refined_text")
                refined_scorable = ScorableFactory.from_text(refined_text, scorable.target_type)
                mrq_scores = self.mrq.score(context, refined_scorable)
                self.logger.log("DocumentRefinedWithEBT", {"document_id": scorable.id})
                refined_score = refined_result.get("final_energy")

                self._log_memcubes_for_srft(
                    scorable,
                    refined_text,
                    goal.get("goal_text", ""),
                    mrq_scores,
                    llm_scores,
                    ebt_energy,
                    uncertainty_by_dim,
                    refined=refined,
                )

                self.training_buffer.maybe_add(
                    context=context,
                    candidate=scorable.text,
                    llm_score=refined_score,
                    ebt_score=mrq_scores["alignment"],
                    threshold=self.cfg.get("disagreement_threshold", 0.15),
                    metadata={"dimension": "alignment"},
                )

            if any(u > self.llm_fallback_threshold for u in uncertainty_by_dim.values()):
                llm_scores = self.llm.score(context, scorable)
                final_scores = llm_scores
                source = "llm"
                self.logger.log("LLMFallbackUsed", {"document_id": scorable.id})
            else:
                final_scores = mrq_scores
                source = "mrq"

            result_entry = {
                "document_id": scorable.id,
                "original_text": scorable.text,
                "refined_text": refined_text if refined else scorable.text,
                "scores": final_scores,
                "mrq_scores": mrq_scores,
                "ebt_energy": ebt_energy,
                "uncertainty_by_dimension": uncertainty_by_dim,
                "used_refinement": refined,
                "used_llm_fallback": source == "llm",
                "final_source": source,
            }

            event_id = self._log_to_database(result_entry)
            event_ids.append(event_id)
            self.logger.log("ScoringEvent", result_entry)
            results.append(result_entry)

        context[self.output_key] = results
        context["event_ids"] = event_ids

        self.plot_score_distributions(results)
        self.generate_summary_report(results)
        return context
