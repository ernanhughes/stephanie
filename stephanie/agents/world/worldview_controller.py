# stephanie/agents/world/worldview_controller.py
from stephanie.agents.base_agent import BaseAgent


class WorldviewControllerAgent(BaseAgent):
    def __init__(self, cfg, memory, logger, tools, pipelines):
        super().__init__(cfg, memory=memory, logger=logger)
        self.tools = (
            tools  # e.g. { "arxiv": ArxivSearcher(), "profiler": ProfilerAgent(), ... }
        )
        self.pipelines = pipelines  # Callable pipeline configs
        self.active_worldview = None

    def load_or_create_worldview(self, goal: dict):
        """Load an existing worldview aligned with the goal, or create a new one."""
        from stephanie.models.world_view import WorldviewORM

        # Use domain + goal hash or embedding similarity
        matched = WorldviewORM.find_nearest(goal)
        if matched:
            self.active_worldview = matched
            self.logger.log("WorldviewReused", {"worldview_id": matched.id})
        else:
            self.active_worldview = WorldviewORM.create_from_goal(goal)
            self.logger.log("WorldviewCreated", {"goal": goal})

    def run_pipeline(self, goal: dict):
        """Run the worldview through a full processing/evaluation cycle."""
        self.load_or_create_worldview(goal)

        for step in self.cfg.get(
            "worldview_steps", ["search", "profile", "score", "ingest", "evaluate"]
        ):
            if step == "search":
                self._search_and_add_sources(goal)
            elif step == "profile":
                self._profile_documents()
            elif step == "score":
                self._score_candidates()
            elif step == "ingest":
                self._ingest_beliefs()
            elif step == "evaluate":
                self._evaluate_and_update()

        self.logger.log("WorldviewCycleComplete", {"goal": goal})

    def _search_and_add_sources(self, goal: dict):
        sources = self.tools["arxiv"].search(goal["query"])
        self.active_worldview.add_sources(sources)

    def _profile_documents(self):
        profiler = self.tools["profiler"]
        self.active_worldview.profiled_docs = profiler.process(
            self.active_worldview.sources
        )

    def _score_candidates(self):
        scorer = self.tools["scorer"]
        self.active_worldview.scored_docs = scorer.score_all(
            self.active_worldview.profiled_docs
        )

    def _ingest_beliefs(self):
        ingestor = self.tools["belief_ingest"]
        cartridges = ingestor.ingest(self.active_worldview.scored_docs)
        for c in cartridges:
            self.active_worldview.add_cartridge(c)

    def _evaluate_and_update(self):
        evaluator = self.tools["evaluator"]
        score = evaluator.evaluate(self.active_worldview)
        self.active_worldview.update_score(score)

    def export(self):
        return self.active_worldview.export_to_markdown()

    def visualize(self):
        # Hook for worldview visualizer
        pass

    def run_autonomous_loop(self):
        """Optionally let worldview self-run repeatedly (e.g. daily update)"""
        if self.cfg.get("autonomous", False):
            import time

            interval = self.cfg.get("autonomous_interval", 86400)  # default 24h
            while True:
                self.run_pipeline(self.active_worldview.goal)
                time.sleep(interval)
