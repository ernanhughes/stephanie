from datetime import datetime

from stephanie.agents.world.belief_tuner import BeliefTunerAgent
from stephanie.agents.world.worldview_audit import WorldviewAuditAgent
from stephanie.agents.world.worldview_evaluator import WorldviewEvaluatorAgent
from stephanie.agents.world.worldview_merger import WorldviewMergerAgent
from stephanie.agents.world.worldview_pipeline_runner import \
    WorldviewPipelineRunner
from stephanie.core.knowledge_cartridge import KnowledgeCartridge


class AutonomousWorldviewCycleAgent:
    """
    Drives autonomous self-refinement of a worldview via pipeline execution,
    belief assimilation, evaluation, and tuning.
    """

    def __init__(self, worldview, logger=None, config=None):
        self.worldview = worldview
        self.logger = logger or self._default_logger()
        self.config = config or {}
        self.pipeline_runner = WorldviewPipelineRunner(worldview, logger=self.logger)
        self.evaluator = WorldviewEvaluatorAgent(worldview, logger=self.logger)
        self.belief_tuner = BeliefTunerAgent(worldview, logger=self.logger)
        self.merger = WorldviewMergerAgent(worldview, logger=self.logger)
        self.audit = WorldviewAuditAgent(worldview, logger=self.logger)

    def cycle_once(self):
        """
        Perform one self-improvement loop:
        1. Select goal
        2. Run pipeline
        3. Evaluate results
        4. Generate/update beliefs
        5. Tune beliefs
        6. Audit and log
        """
        goal = self._select_goal()
        if not goal:
            self.logger.log("NoGoalAvailable", {"timestamp": datetime.utcnow().isoformat()})
            return

        self.logger.log("CycleStarted", {"goal_id": goal["id"]})
        result = self.pipeline_runner.run(goal_id=goal["id"])
        evaluation = self.evaluator.evaluate(goal, result)
        cartridge = self._create_cartridge(goal, result, evaluation)

        self.belief_tuner.tune_from_result(goal, result, evaluation)
        self.worldview.add_cartridge(cartridge)
        self.audit.record_cycle(goal, cartridge)

        self.logger.log("CycleCompleted", {
            "goal_id": goal["id"],
            "score": evaluation.aggregate(),
            "timestamp": datetime.utcnow().isoformat()
        })

    def run_forever(self, interval_sec=3600):
        import time
        while True:
            self.cycle_once()
            time.sleep(interval_sec)

    def _select_goal(self):
        goals = self.worldview.list_goals()
        # Prioritize unevaluated, high-potential, or user-specified goals
        for goal in goals:
            if not goal.get("last_evaluated"):
                return goal
        return goals[0] if goals else None

    def _create_cartridge(self, goal, result, evaluation):
        cartridge = KnowledgeCartridge(
            goal=goal["text"],
            domain=goal.get("domain", "general"),
            source="autonomous_cycle",
        )
        cartridge.add_summary(str(result)[:500])
        cartridge.add_score(evaluation.aggregate())
        cartridge.add_timestamp()

        return cartridge

    def _default_logger(self):
        class DummyLogger:
            def log(self, tag, payload): print(f"[{tag}] {payload}")
        return DummyLogger()
