# stephanie/agents/analysis/refinement_analysis_agent.py

from stephanie.memcubes.memcube_store import MemCubeStore
from stephanie.scoring.scorable_factory import ScorableFactory
from stephanie.utils.visualization import plot_refinement_traces


class RefinementAnalysisAgent:
    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.store = MemCubeStore(memory)

    def run(self, goal_id=None, dimension_filter=None):
        memcubes = self.store.fetch_memcubes_by_type("refinement", goal_id=goal_id)

        if dimension_filter:
            memcubes = [m for m in memcubes if m.extra_data.get("dimension") == dimension_filter]

        improvements = []
        traces = []

        for memcube in memcubes:
            dim = memcube.extra_data.get("dimension")
            orig = memcube.extra_data.get("original_score")
            refined = memcube.extra_data.get("refined_score")

            if orig is not None and refined is not None:
                delta = refined - orig
                improvements.append((dim, delta))
                trace = memcube.extra_data.get("refinement_trace", [])
                traces.append((dim, trace))

        self._log_summary(improvements)
        self._plot_traces(traces)

    def _log_summary(self, improvements):
        by_dim = {}
        for dim, delta in improvements:
            by_dim.setdefault(dim, []).append(delta)

        for dim, deltas in by_dim.items():
            avg = sum(deltas) / len(deltas)
            print(f"[{dim}] Avg Δ = {avg:.3f} across {len(deltas)} examples")
            self.logger.log("RefinementDeltaSummary", {"dimension": dim, "average_delta": avg})

    def _plot_traces(self, traces):
        for dim, trace in traces:
            if trace:
                plot_refinement_traces(trace, title=f"Refinement Trace: {dim}")
