from stephanie.agents import DOTSPlannerAgent, LookaheadAgent


class PipelinePlannerAgent:
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        # Load sub-agents if enabled
        self.dots_enabled = cfg.get("dots_enabled", True)
        self.lookahead_enabled = cfg.get("lookahead_enabled", True)

        if self.dots_enabled:
            self.dots = DOTSPlannerAgent(cfg, memory, logger)
        if self.lookahead_enabled:
            self.lookahead = LookaheadAgent(cfg, memory, logger)


    async def run(self, context):
        if self.dots:
            context = await self.dots.run(context)
        if self.lookahead:
            context = await self.lookahead.run(context)
        return context
