# stephanie/agents/dspy/memento.py
import traceback
from stephanie.agents.dspy.mcts_reasoning import MCTSReasoningAgent
from stephanie.memory.casebook_store import CaseBookStore
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker
from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.data.score_corpus import ScoreCorpus

class MementoAgent(MCTSReasoningAgent):
    """
    Case-Based Reasoning LATS Agent
    --------------------------------
    Demonstrates the full Memento process:

    - Retrieve: Find similar past cases from the casebook
    - Reuse: Adapt their solutions as candidate traces/hypotheses
    - Revise: Evaluate with MARS + scorers, refine if needed
    - Retain: Store the new case into the casebook
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.casebook_id = cfg.get("casebook_id")
        self.casebook_store = CaseBookStore(memory, logger)
        self.ranker = ScorableRanker(cfg, memory, logger)
        self.include_mars = cfg.get("include_mars", True)
        if self.include_mars:
            self.mars = MARSCalculator(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context[GOAL]

        try:
            # === 1. RETRIEVE ===
            past_cases = self.memory.casebooks.get_cases_for_goal(goal["id"])
            self.logger.log("CBRRetrieve", {"goal": goal, "cases_found": len(past_cases)})
            self.report(
                {
                    "event": "retrieve",
                    "step": "MementoAgent",
                    "details": f"Retrieved {len(past_cases)} past cases",
                }
            )

            reuse_candidates = []
            for case in past_cases[:5]:
                for sc in case.scorables:
                    reuse_candidates.append(sc.scorable_id)

            # === 2. REUSE ===
            context["reuse_candidates"] = reuse_candidates
            lats_result = await super().run(context)

            best_hypotheses = lats_result.get("hypotheses", [])
            self.logger.log("CBRReuse", {"goal": goal, "count": len(best_hypotheses)})
            self.report(
                {
                    "event": "reuse",
                    "step": "MementoAgent",
                    "details": f"Reused {len(reuse_candidates)} candidates → {len(best_hypotheses)} hypotheses",
                }
            )

            # === 2b. RANK ===

            # Build query scorable from the goal text
            query_scorable = Scorable(
                id=goal["id"],
                text=goal["goal_text"],
                target_type=TargetType.GOAL,   # you may need to ensure GOAL is in your TargetType
            )
            ranked = []
            if best_hypotheses:
                # Convert hypotheses into Scorables
                scorables = [
                    Scorable(
                        id=h.get("id"),
                        text=h.get("text", ""),
                        target_type=TargetType.HYPOTHESIS,
                    )
                    for h in best_hypotheses
                ]

                ranked = self.ranker.rank(
                    query=query_scorable,
                    candidates=scorables,
                    context=context
                )

                self.logger.log("CBRRank", {"ranked_count": len(ranked)})
                self.report(
                    {
                        "event": "rank",
                        "step": "MementoAgent",
                        "details": f"Ranked {len(ranked)} hypotheses",
                    }
                )
                bundles = {}
                for scorable in scorables:
                    scores, bundle = self._score(context=context, scorable=scorable)
                    bundles[scorable.id] = bundle
                corpus = ScoreCorpus(bundles=bundles)
 
            # === 2. MARS Analysis ===
            mars_results = {}
            if self.include_mars:
                mars_results = self.mars.calculate(corpus, context=context)

            # === 3. REVISE ===
            recommendations = self.mars.generate_recommendations(mars_results)
            self.logger.log("CBRRevise", {
                "mars_summary": mars_results,
                "recommendations": recommendations
            })
            self.report(
                {
                    "event": "revise",
                    "step": "MementoAgent",
                    "details": f"MARS analysis complete with {len(mars_results)} results",
                    "recommendations": recommendations,
                }
            )

            # Optionally adjust ranking based on MARS consensus
            for r in ranked:
                r["mars_confidence"] = mars_results.get(r["id"], {}).get("agreement_score")

            # === 4. RETAIN ===
            retained_case_id = None
            if ranked:
                case = self.casebook_store.add_case(
                    casebook_id=self.casebook_id,
                    goal_id=goal["id"],
                    goal_text=goal["goal_text"],
                    agent_name="MementoAgent",
                    mars_summary=mars_results,
                    scores={h["id"]: h.get("scores", {}) for h in best_hypotheses},
                    scorables=[
                        {
                            "id": r["id"],
                            "type": "hypothesis",
                            "role": "output",
                            "rank": r.get("rank"),
                            "mars_confidence": r.get("mars_confidence"),
                        }
                        for r in ranked
                    ],
                )
                retained_case_id = case.id
                self.logger.log("CBRRetain", {"case_id": case.id, "goal": goal})
                self.report(
                    {
                        "event": "retain",
                        "step": "MementoAgent",
                        "details": f"Retained case {case.id} with {len(ranked)} ranked hypotheses",
                    }
                )

            # === Final Conclusion Report ===
            self.report(
                {
                    "event": "conclusion",
                    "step": "MementoAgent",
                    "goal": goal.get("goal_text", ""),
                    "summary": f"Retrieved {len(past_cases)} → Reused {len(best_hypotheses)} → Ranked {len(ranked)} → Retained {retained_case_id or 'none'}",
                    "recommendations": recommendations,
                }
            )

            # Return enriched results
            lats_result["ranked_hypotheses"] = ranked
            lats_result["mars_results"] = mars_results
            lats_result["recommendations"] = recommendations

            context[self.output_key] = ranked

            # Choose one "representative" output for scorable details
            if ranked:
                primary_hypothesis = ranked[0]["text"]
            else:
                primary_hypothesis = "[no hypotheses generated]"

            self.set_scorable_details(
                input_text=goal.get(GOAL_TEXT, ""),
                output_text=primary_hypothesis,   # ✅ single string for scorers
                description=f"CBR hypotheses for goal: {goal.get(GOAL_TEXT, '')}",
                extra={
                    "all_hypotheses": [h["text"] for h in ranked],  # ✅ keep the full list
                    "count": len(ranked),
                }
            )

            self.report(
                {
                    "event": "cbr_hypotheses_extracted",
                    "count": len(ranked),
                    "goal_id": goal.get("id", "unknown"),
                    "primary": primary_hypothesis[:120] + ("..." if len(primary_hypothesis) > 120 else ""),
                    "examples": [h["text"] for h in ranked[:2]],
                }
            )

        except Exception as e:
            self.logger.error(
                "CBRRunFailed",
                {"error": str(e), "trace": traceback.format_exc()}
            )
            self.report(
                {
                    "event": "error",
                    "step": "MementoAgent",
                    "details": str(e),
                }
            )
        return context
