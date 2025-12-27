# stephanie/agents/chain_sampling_agent.py
from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.jitter.lifecycle.chain_sampler import \
    diversified_samples
from stephanie.services.chain_runner_adapter import make_run_chain_fn


class ChainSamplingAgent(BaseAgent):
    """
    Input context:
      { "question": str, "image_path": Optional[str],
        "n_total": 8, "p_interleaved": 0.5, "time_budget_s": 10 }
    Output context:
      { "winner": ChainCandidate, "candidates": [...], "trace_id": ... }
    """
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        q = context["question"]
        img_path = context.get("image_path")
        image = Image.open(img_path) if img_path else None

        # supply your underlying runner (LLM+tools / CoT / PlanTrace executor)
        underlying = self.container.get("chain_runner")  # your callable (q, image, seed)->ChainResult
        run_chain_fn = make_run_chain_fn(underlying)

        selector = self.container.get("chain_selector")  # or import basic selector
        # fallback:
        if selector is None:
            from stephanie.components.jitter.lifecycle.chain_sampler import \
                basic_selector_sicql_hrm_mars
            selector = basic_selector_sicql_hrm_mars

        winner, cands = diversified_samples(
            question=q,
            image=image,
            n_total=int(context.get("n_total", 8)),
            p_interleaved=float(context.get("p_interleaved", 0.5)),
            run_chain_fn=run_chain_fn,
            score_selector_fn=selector,
            time_budget_s=context.get("time_budget_s"),
            seed=int(context.get("seed", 0)),
        )

        # Optional: log to MemCube/CaseBooks/VPM timeline
        try:
            mem = self.container.get("MemCubeService")
            mem.save_chain_candidates(goal=context.get("goal_id"), winner=winner, candidates=cands)
        except Exception:
            pass

        return {"winner": winner, "candidates": cands, "trace_id": context.get("trace_id")}
