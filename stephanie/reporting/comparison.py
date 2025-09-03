from typing import Dict, List

from sqlalchemy.orm import Session

from stephanie.models.evaluation import EvaluationORM
from stephanie.models.pipeline_run import PipelineRunORM
from stephanie.models.score import ScoreORM


class ComparisonReporter:
    def __init__(self, session: Session):
        self.session = session

    def get_pipeline_run(self, id: int) -> Dict:
        run = self.session.query(PipelineRunORM).filter(PipelineRunORM.id == id).first()
        if not run:
            raise ValueError(f"No run found with id={id}")
        return self._pipeline_run_to_dict(run)

    def get_evaluations_for_run(self, id: int) -> List[Dict]:
        evaluations = self.session.query(EvaluationORM).filter(
            EvaluationORM.pipeline_run_id == id
        ).all()
        return [self._evaluation_to_dict(e) for e in evaluations]

    def get_dimension_scores(self, evaluation_id: int) -> Dict[str, float]:
        scores = self.session.query(ScoreORM).filter(ScoreORM.evaluation_id == evaluation_id).all()
        return {s.dimension: s.score for s in scores}

    def compare_runs(self, run_a_id: int, run_b_id: int) -> Dict:
        run_a = self.get_pipeline_run(run_a_id)
        run_b = self.get_pipeline_run(run_b_id)

        goal = run_a["goal"]["goal_text"] or run_b["goal"]["goal_text"]

        # Compare evaluations
        evaluations_a = self.get_evaluations_for_run(run_a_id)
        evaluations_b = self.get_evaluations_for_run(run_b_id)

        # Compare scores
        scores_a = self.get_dimension_scores(evaluations_a[0]["id"]) if evaluations_a else {}
        scores_b = self.get_dimension_scores(evaluations_b[0]["id"]) if evaluations_b else {}

        # Compare scores by stage
        stage_scores = []
        for stage_name in run_a["pipeline"]:
            score_a = scores_a.get(stage_name, 0)
            score_b = scores_b.get(stage_name, 0)
            if score_a or score_b:
                stage_scores.append({
                    "stage": stage_name,
                    "run_a_score": score_a,
                    "run_b_score": score_b,
                    "score_diff": score_a - score_b,
                    "rationale": self._generate_rationale(score_a, score_b, stage_name)
                })

        # Delta logic
        total_score_a = run_a.get("score", 0)
        total_score_b = run_b.get("score", 0)

        delta = {
            "preferred": run_a_id if total_score_a > total_score_b else run_b_id,
            "score_diff": total_score_a - total_score_b,
            "embedding_quality": self._compare_embeddings(run_a, run_b),
            "convergence": self._compare_convergence(run_a, run_b),
            "stage_performance": stage_scores
        }

        return {
            "goal": goal,
            "runs": [run_a, run_b],
            "delta": delta
        }

    def _pipeline_run_to_dict(self, run: PipelineRunORM) -> Dict:
        return {
            "run_id": run.run_id,
            "name": run.name,
            "tag": run.tag,
            "description": run.description,
            "pipeline": run.pipeline,
            "strategy": run.strategy,
            "model_name": run.model_name,
            "embedding_type": run.embedding_type,
            "embedding_dim": run.embedding_dimensions,
            "created_at": run.created_at.isoformat(),
            "goal": run.goal.to_dict() if run.goal else None
        }

    def _evaluation_to_dict(self, evaluation: EvaluationORM) -> Dict:
        return {
            "id": evaluation.id,
            "agent_name": evaluation.agent_name,
            "model_name": evaluation.model_name,
            "scores": evaluation.scores,
            "extra_data": evaluation.extra_data,
            "stage_types": [s["name"] for s in evaluation.scores.get("pipeline", [])]
        }

    def _compare_embeddings(self, run_a: dict, run_b: dict) -> str:
        if run_a["embedding_dim"] > run_b["embedding_dim"]:
            return f"{run_a['embedding_type']} has higher dimension ({run_a['embedding_dim']})"
        else:
            return f"{run_b['embedding_type']} has higher dimension ({run_b['embedding_dim']})"

    def _compare_convergence(self, run_a: dict, run_b: dict) -> str:
        if run_a.get("converged") is None or run_b.get("converged") is None:
            return "unknown convergence status"    
        if run_a["converged"] and not run_b["converged"]:
            return "run_a converged; run_b did not"
        elif not run_a["converged"] and run_b["converged"]:
            return "run_b converged; run_a did not"
        else:
            return "both runs converged" if run_a["converged"] else "both runs failed"

    def _generate_rationale(self, score_a: float, score_b: float, stage: str) -> str:
        diff = score_a - score_b
        if diff > 2:
            return f"{stage} scored higher with {diff:.1f} point lead"
        elif diff < -2:
            return f"{stage} scored lower with {-diff:.1f} point deficit"
        else:
            return f"{stage} performance was similar"