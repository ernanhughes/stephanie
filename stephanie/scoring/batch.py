# stephanie/scoring/batch.py
from stephanie.agents.pipeline_judge import PipelineJudgeAgent
from stephanie.constants import GOAL, PIPELINE_RUN_ID, RUN_ID
from stephanie.models import EvaluationORM, HypothesisORM, RuleApplicationORM


def get_unscored_hypotheses(session, run_id: str = None):
    # Get hypotheses with no evaluations for this run
    subquery = session.query(EvaluationORM.hypothesis_id).distinct()
    query = session.query(HypothesisORM).filter(~HypothesisORM.id.in_(subquery))

    if run_id:
        query = query.filter(HypothesisORM.pipeline_run_id == run_id)

    return query.all()


async def score_unscored_hypotheses(memory, logger, config, run_id=None):
    session = memory.session
    unscored = get_unscored_hypotheses(session, run_id)
    agent = PipelineJudgeAgent(cfg=config, memory=memory, logger=logger)

    for hypo in unscored:
        goal = memory.goals.get_by_id(hypo.goal_id)
        rule_apps = memory.rule_effects.get_by_hypothesis(hypo.id)

        context = {
            GOAL: goal.to_dict(),
            "hypotheses": [hypo.to_dict()],
            PIPELINE_RUN_ID: hypo.pipeline_run_id,
            RUN_ID: f"batch-repair-{hypo.id}",
            "rule_applications": [ra.to_dict() for ra in rule_apps],
        }

        logger.log(
            "ScoringUnscoredHypothesis",
            {
                "hypothesis_id": hypo.id,
                "goal_id": goal.id,
                "rule_count": len(rule_apps),
            },
        )

        await agent.run(context)
