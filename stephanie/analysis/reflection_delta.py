# stephanie/analysis/reflection_delta.py
from datetime import datetime, timezone
from statistics import mean


def compare_pipeline_runs(memory, goal_id):
    runs = memory.pipeline_runs.get_by_goal_id(goal_id)
    if len(runs) < 2:
        return []

    deltas = []
    for i, run_a in enumerate(runs):
        for run_b in runs[i + 1 :]:
            scores_a = memory.evaluations.get_by_run_id(run_a.run_id)
            scores_b = memory.evaluations.get_by_run_id(run_b.run_id)

            if not scores_a or not scores_b:
                continue  # skip if unscored

            delta = compute_pipeline_delta(run_a, run_b, scores_a, scores_b)
            deltas.append(delta)

    return deltas


def average_score(scores):
    numeric_scores = [s["score"] for s in scores if s.get("score") is not None]
    return round(mean(numeric_scores), 4) if numeric_scores else None


def list_diff(list1, list2):
    return {
        "only_in_a": [x for x in list1 if x not in list2],
        "only_in_b": [x for x in list2 if x not in list1],
    }


def compute_pipeline_delta(run_a, run_b, scores_a, scores_b):
    score_a = average_score(scores_a)
    score_b = average_score(scores_b)

    return {
        "goal_id": run_a.goal_id,
        "run_id_a": run_a.run_id,
        "run_id_b": run_b.run_id,
        "score_a": score_a,
        "score_b": score_b,
        "score_delta": round(score_b - score_a, 4)
        if score_a is not None and score_b is not None
        else None,
        "pipeline_a": run_a.pipeline,
        "pipeline_b": run_b.pipeline,
        "pipeline_diff": list_diff(run_a.pipeline, run_b.pipeline),
        "strategy_diff": run_b.strategy != run_a.strategy,
        "model_diff": run_b.model_name != run_a.model_name,
        "rationale_diff": (
            run_a.lookahead_context.get("rationale")
            if run_a.lookahead_context
            else None,
            run_b.lookahead_context.get("rationale")
            if run_b.lookahead_context
            else None,
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
