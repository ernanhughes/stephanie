import uuid
from typing import Any, Callable, Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.core.context.context_manager import ContextManager
from stephanie.data.score_bundle import ScoreBundle


class ContextEnsembleRunner:
    def __init__(
        self,
        context_managers: List[ContextManager],
        inference_agent: BaseAgent,
        scorer_fn: Callable[[str, str], ScoreBundle],
        memory=None,
        logger=None
    ):
        self.context_managers = context_managers
        self.inference_agent = inference_agent
        self.scorer_fn = scorer_fn
        self.memory = memory
        self.logger = logger

    def run(self, query: str) -> Dict[str, Any]:
        results = []

        for idx, ctx_mgr in enumerate(self.context_managers):
            prompt = ctx_mgr.assemble()
            result = self.inference_agent.run({
                "prompt": prompt,
                "query": query,
                "context_dict": ctx_mgr.get_context_dict(),
            })

            score = self.scorer_fn(query, result.get("answer", ""))
            results.append({
                "id": str(uuid.uuid4()),
                "ctx_mgr": ctx_mgr,
                "result": result,
                "score": score,
                "raw_prompt": prompt,
            })

        best = max(results, key=lambda r: r["score"].overall)

        if self.memory:
            self.memory.memcube.store({
                "query": query,
                "answer": best["result"]["answer"],
                "score": best["score"].dict() if hasattr(best["score"], "dict") else str(best["score"]),
                "prompt_used": best["raw_prompt"],
                "trace": best["ctx_mgr"].trace,
            })

        if self.logger:
            self.logger.log("BestICLResult", {
                "query": query,
                "score": best["score"].overall,
                "result": best["result"],
            })

        return {
            "query": query,
            "best_answer": best["result"]["answer"],
            "best_score": best["score"],
            "all_results": results,
        }
