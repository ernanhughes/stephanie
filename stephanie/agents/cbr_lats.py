# stephanie/agents/cbr_lats.py
import traceback
from stephanie.agents.dspy.mcts_reasoning_agent import MCTSReasoningAgent
from stephanie.memory.casebook_store import CaseBookStore
from stephanie.constants import GOAL


class CBRLATSAgent(MCTSReasoningAgent):
    """
    Case-Based Reasoning LATS Agent
    --------------------------------
    Demonstrates the full Memento process:

    - Retrieve: Find similar past cases from the casebook
    - Reuse: Adapt their solutions as candidate traces/hypotheses
    - Revise: Evaluate with MARS + scorers, refine if needed
    - Retain: Store the new case into the casebook

    Compact, end-to-end demonstration of the Memento paper.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.casebook_store = CaseBookStore(memory.session)
        self.casebook_id = cfg.get("casebook_id")  # use a default or auto-create

    async def run(self, context: dict) -> dict:
        goal = context[GOAL]

        try:
            # === 1. RETRIEVE ===
            past_cases = self.casebook_store.get_cases_for_goal(goal["id"])
            self.logger.log("CBRRetrieve", {"goal": goal, "cases_found": len(past_cases)})

            # Optionally seed with past scorables
            reuse_candidates = []
            for case in past_cases[:5]:
                for sc in case.scorables:
                    reuse_candidates.append(sc.scorable_id)

            # === 2. REUSE ===
            # Run LATS (fast variant) using reused scorables as priors
            context["reuse_candidates"] = reuse_candidates
            lats_result = await super().run(context)

            best_hypotheses = lats_result.get("hypotheses", [])
            self.logger.log("CBRReuse", {"goal": goal, "count": len(best_hypotheses)})

            # === 3. REVISE ===
            mars_results = lats_result.get("mars_results", {})
            recommendations = self.mars.generate_recommendations(mars_results)
            self.logger.log("CBRRevise", {"recommendations": recommendations})

            # === 4. RETAIN ===
            if best_hypotheses:
                case = self.casebook_store.add_case(
                    casebook_id=self.casebook_id,
                    goal_id=goal["id"],
                    goal_text=goal["goal_text"],
                    agent_name="CBRLATSAgent",
                    mars_summary=mars_results,
                    scores=dict(mars_results.items()),
                    scorables=[
                        {
                            "id": h["id"],
                            "type": "hypothesis",
                            "role": "output",
                        }
                        for h in best_hypotheses
                    ],
                )
                self.logger.log("CBRRetain", {"case_id": case.id, "goal": goal})

            return lats_result

        except Exception as e:
            self.logger.error("CBRRunFailed", {"error": str(e), "trace": traceback.format_exc()})
            return context
