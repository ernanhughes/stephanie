from datetime import datetime

from sqlalchemy.orm import Session

from stephanie.models.belief import BeliefORM
from stephanie.models.cartridge import CartridgeORM
from stephanie.models.icl_example import ICLExampleORM
from stephanie.models.world_view import WorldviewORM
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.svm_scorer import SVMScorer


class WorldviewEvaluatorAgent:
    def __init__(self, db: Session, scorer: SVMScorer, logger=None):
        self.db = db
        self.scorer = scorer
        self.logger = logger

    def evaluate(self, worldview_id: int, goal: dict, dimensions: list[str] = ["alignment", "utility", "novelty"]) -> dict:
        results = {}

        beliefs = self.db.query(BeliefORM).filter_by(worldview_id=worldview_id).all()
        cartridges = self.db.query(CartridgeORM).filter_by(worldview_id=worldview_id).all()

        belief_results = []
        for belief in beliefs:
            text = belief.summary + "\n" + (belief.rationale or "")
            bundle = self.scorer.score(goal, {"text": text}, dimensions=dimensions)
            belief_results.append((belief, bundle))

            self.logger.log("WorldviewBeliefScored", {
                "belief_id": belief.id,
                "scores": bundle.to_dict()
            })

        cartridge_results = []
        for cartridge in cartridges:
            thesis = cartridge.schema.get("core_thesis", "")
            bundle = self.scorer.score(goal, {"text": thesis}, dimensions=dimensions)
            cartridge_results.append((cartridge, bundle))

            self.logger.log("WorldviewCartridgeScored", {
                "cartridge_id": cartridge.id,
                "scores": bundle.to_dict()
            })

        results["beliefs"] = belief_results
        results["cartridges"] = cartridge_results

        return results

    def generate_report(self, evaluation_results) -> str:
        report_lines = ["## 🧾 Worldview Evaluation Report\n"]
        for belief, bundle in evaluation_results.get("beliefs", []):
            report_lines.append(f"### Belief: {belief.summary}")
            for dim, res in bundle.results.items():
                report_lines.append(f"- {dim.capitalize()}: {res.score:.2f} ({res.rationale})")
            report_lines.append("")

        for cart, bundle in evaluation_results.get("cartridges", []):
            report_lines.append(f"### Cartridge: {cart.goal}")
            for dim, res in bundle.results.items():
                report_lines.append(f"- {dim.capitalize()}: {res.score:.2f} ({res.rationale})")
            report_lines.append("")

        return "\n".join(report_lines)
