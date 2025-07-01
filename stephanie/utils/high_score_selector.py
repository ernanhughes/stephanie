import json
from collections import defaultdict

from sqlalchemy.orm import joinedload

from stephanie.models.evaluation import EvaluationORM
from stephanie.models.pipeline_run import PipelineRunORM
from stephanie.models.rule_application import RuleApplicationORM
from stephanie.models.score import ScoreORM


def get_high_scoring_runs(session, dimension: str, threshold: float, min_repeat_count: int = 2):
    """
    Returns a dict of {signature: list of (evaluation, run)} for all runs that score above the threshold
    on a given dimension.
    """
    # Step 1: Query evaluations with score above threshold
    evaluations = (
        session.query(EvaluationORM)
        .join(ScoreORM, EvaluationORM.id == ScoreORM.evaluation_id)
        .filter(ScoreORM.dimension == dimension)
        .filter(ScoreORM.score >= threshold)
        .options(joinedload(EvaluationORM.dimension_scores))
        .all()
    )

    # Step 2: Filter out those with rule applications
    valid_runs = []
    for evaluation in evaluations:
        rule_applied = (
            session.query(RuleApplicationORM)
            .filter_by(hypothesis_id=evaluation.hypothesis_id)
            .first()
        )
        if rule_applied:
            continue
        run = session.get(PipelineRunORM, evaluation.pipeline_run_id)
        if run:
            valid_runs.append((evaluation, run))

    # Step 3: Group by config signature
    grouped = defaultdict(list)
    for evaluation, run in valid_runs:
        sig = make_signature(run.run_config or {})
        grouped[sig].append((evaluation, run))

    # Step 4: Filter by minimum repetition
    return {
        sig: entries
        for sig, entries in grouped.items()
        if len(entries) >= min_repeat_count
    }


def make_signature(config):
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            config = {}

    model = config.get("model", {}).get("name", "unknown")
    agent = config.get("agent", "unknown")
    goal_type = config.get("goal", {}).get("goal_type", "unknown")

    return f"{model}::{agent}::{goal_type}"