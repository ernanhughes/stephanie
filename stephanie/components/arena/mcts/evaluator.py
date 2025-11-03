# stephanie/arena/mcts/evaluator.py
from __future__ import annotations

from stephanie.components.arena.plugins.interfaces import JobCtx
from stephanie.components.arena.plugins.registry import get_play, list_scorers
from stephanie.components.arena.scoring.aggregate import WeightedAggregator


class PlayEvaluator:
    def __init__(self, aggregator: WeightedAggregator, enabled_scorers: list[str]):
        self.aggr = aggregator
        self.scorers = {name: list_scorers()[name] for name in enabled_scorers}

    def evaluate(self, ctx: JobCtx, play_name: str):
        play = get_play(play_name)
        result = play.run(ctx)
        scored = {n: sc.score(ctx, result.metrics) for n, sc in self.scorers.items()}
        reward = self.aggr.aggregate(scored)
        return reward, result, scored
