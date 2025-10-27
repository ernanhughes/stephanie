from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from omegaconf import DictConfig, OmegaConf

# Optional event emitter (don’t hard-crash if it’s absent)
from stephanie.components.ssp.search.tree_events import TreeEventEmitter
from stephanie.components.ssp.util import PlanTrace_safe, get_trace_logger
# Tree search pieces
from stephanie.components.tree.core import AgenticTreeSearch, SolutionNode
from stephanie.components.tree.tree_grpo import TreeGRPOAdapter, TreeGRPOConfig
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.service_container import ServiceContainer
from stephanie.utils.json_sanitize import sanitize


def _maybe(val, default):
    return val if val is not None else default


# --- LLM adapter that uses the service container's PromptService -------------

class _ContainerLLMAgent:
    """
    Minimal adapter exposing `async_call_llm(prompt, context)` so AgenticTreeSearch
    can remain agnostic. Internally uses container.get("prompt").run_prompt(...)
    """

    def __init__(self, container: ServiceContainer, sys_preamble: str | None = None):
        self.prompt_service = container.get("prompt")
        self.sys_preamble = (sys_preamble or "").strip()

    async def async_call_llm(self, prompt: str, context: dict) -> str:
        # prepend a short preamble to bias to crisp outputs without forcing JSON
        full_prompt = f"{self.sys_preamble}\n{prompt}" if self.sys_preamble else prompt
        return await self.prompt_service.run_prompt(full_prompt, context)


class Solver:
    """
    SSP Solver that delegates search to the Tree component, but:
    - Uses PromptService from ServiceContainer for LLM calls
    - Post-hoc scores candidates via ScoringService to pick the winner
    - Optionally uses Tree-GRPO (kept intact)
    """

    def __init__(self, cfg: DictConfig | dict,  container: ServiceContainer):
        self.root: DictConfig = cfg
        self.sp: DictConfig = self.root.self_play
        self.cfg: DictConfig = self.sp.solver
        self.container = container

        # Trace logging
        self.trace_logger = get_trace_logger()

        # ---- LLM agent: call through the container's PromptService
        sys_preamble = self.cfg.get("llm_preamble") or (
            "You are a careful problem solver. Think step-by-step and keep answers concise."
        )
        self.agent = _ContainerLLMAgent(container=container, sys_preamble=sys_preamble)

        # ---- Build base tree search
        max_iterations = int(_maybe(self.cfg.get("search_iterations"),
                                    max(1, int(self.cfg.get("search_depth", 5)) * 4)))
        time_limit = int(_maybe(self.cfg.get("time_limit_sec"), 60))

        self.base = AgenticTreeSearch(
            agent=self.agent,
            max_iterations=max_iterations,
            time_limit=time_limit,
            N_init=int(_maybe(self.cfg.get("N_init"), 3)),
            C_ucb=float(_maybe(self.cfg.get("C_ucb"), 1.2)),
            H_greedy=float(_maybe(self.cfg.get("H_greedy"), 0.3)),
            H_debug=float(_maybe(self.cfg.get("H_debug"), 0.5)),
            no_improve_patience=int(_maybe(self.cfg.get("no_improve_patience"), 30)),
            progress_every=int(_maybe(self.cfg.get("progress_every"), 5)),
            heartbeat_secs=float(_maybe(self.cfg.get("heartbeat_secs"), 10.0)),
            report_top_k=int(_maybe(self.cfg.get("report_top_k"), 5)),
        )

        if self.cfg.get("metric_policy") in ("minimize", "maximize"):
            self.base.metric_policy = self.cfg.metric_policy

        # Tree events
        self._tree_emitter = TreeEventEmitter() if TreeEventEmitter else None
        self.base.emit_cb = self._emit_tree_event  # async callback

        # ---- Scoring config (used for post-hoc selection)
        self.enabled_scorers: List[str] = (
            list(_maybe(self.cfg.get("scorers"),
                        _maybe(OmegaConf.select(self.sp, "verifier.scorers"), ["tiny"])))
        )
        self.dimensions: List[str] = (
            list(_maybe(self.cfg.get("dimensions"),
                        _maybe(OmegaConf.select(self.sp, "verifier.dimensions"),
                               ["novelty","clarity","relevance","implementability","alignment"])))
        )
        self.posthoc_top_k: int = int(_maybe(self.cfg.get("posthoc_top_k"), 12))

        # ---- Optional Tree-GRPO wrapper
        self.use_grpo: bool = bool(self.cfg.get("use_grpo", False))
        self.adapter: Optional[TreeGRPOAdapter] = None
        if self.use_grpo:
            tcfg = self.cfg.get("tree") or {}
            self.adapter = TreeGRPOAdapter(
                self.base,
                TreeGRPOConfig(
                    M=int(_maybe(tcfg.get("M"), 2)),
                    N=int(_maybe(tcfg.get("N"), 2)),
                    L=int(_maybe(tcfg.get("L"), 1)),
                    scorer_name=str(_maybe(tcfg.get("scorer_name"), "sicql")),
                    dimensions=list(_maybe(tcfg.get("dimensions"), ["alignment"])),
                    use_zscore_intra=bool(_maybe(tcfg.get("use_zscore_intra"), False)),
                    use_zscore_inter=bool(_maybe(tcfg.get("use_zscore_inter"), True)),
                    value_alpha=float(_maybe(tcfg.get("value_alpha"), 0.0)),
                    prefer_non_buggy=bool(_maybe(tcfg.get("prefer_non_buggy"), True)),
                ),
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def solve(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(self.solve_async(proposal))

    async def solve_async(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        goal = str(proposal.get("query") or proposal.get("goal") or "").strip()
        if not goal:
            raise ValueError("Solver requires proposal['query'] (non-empty).")

        ctx = self._build_context_for_tree(proposal)

        # Span for the full solve
        with self.trace_logger.span(role="solver", goal=goal, metadata={"use_grpo": self.use_grpo}) as span:
            try:
                if self.use_grpo and self.adapter:
                    forest = await self.adapter.rollout_forest(ctx)
                    best_node, best_reward = self._pick_best_from_grpo(forest)
                    # If GRPO returns nothing robustly, fall back to post-hoc selection
                    if not best_node:
                        best_node, report = await self._posthoc_pick_best(ctx)
                        result = self._result_payload_from_node(best_node, report,
                                                                extra={"training_batch": forest.get("training_batch")})
                    else:
                        report = self._simple_report_from_grpo(forest, best_node, best_reward)
                        result = self._result_payload_from_node(best_node, report,
                                                                extra={"training_batch": forest.get("training_batch")})
                else:
                    # Vanilla run, then pick winner via scoring service
                    await self.base.run(ctx)
                    best_node, report = await self._posthoc_pick_best(ctx)
                    result = self._result_payload_from_node(best_node, report)

                span.success(output=result.get("answer"), artifacts=result)

                self.trace_logger.log(
                    PlanTrace_safe(
                        trace_id=f"solver-{abs(hash(goal)) % 1_000_000}",
                        role="solver",
                        goal=goal,
                        status="completed",
                        metadata={
                            "nodes": len(self._nodes_iterable()),
                            "best_metric": (self.base.best_metric if getattr(self.base, 'best_metric', None) is not None else None),
                            "use_grpo": self.use_grpo,
                        },
                        input=goal,
                        output=result.get("answer", ""),
                        artifacts=sanitize(result),
                    )
                )
                return sanitize(result)

            except Exception as e:
                span.error(e)
                return sanitize({
                    "answer": "",
                    "error": str(e),
                    "reasoning_path": [],
                    "evidence": [],
                    "search_depth": 0,
                })

    # ------------------------------------------------------------------ #
    # Scoring & selection
    # ------------------------------------------------------------------ #

    async def _posthoc_pick_best(self, ctx: dict) -> tuple[Optional[SolutionNode], Dict[str, Any]]:
        """
        Score a set of candidate nodes (top-K by the tree's own report if present,
        otherwise all leaves) using ScoringService; pick the winner.
        """
        # Gather candidates
        leaderboard = self._try_get_leaderboard()
        if leaderboard:
            # Expect items like {"node_id": ..., "reward": ...}
            node_ids = [it.get("node_id") for it in leaderboard][: self.posthoc_top_k]
            candidates = [self.base.nodes_by_id.get(nid) for nid in node_ids if nid in self.base.nodes_by_id]
        else:
            # Fallback: take recent leaves or all nodes with text
            nodes_all = list(self._nodes_iterable())
            textful = [n for n in nodes_all if n and (getattr(n, "summary", None) or getattr(n, "output", None) or getattr(n, "plan", None))]
            candidates = textful[-self.posthoc_top_k:] if len(textful) > self.posthoc_top_k else textful

        # Score each candidate
        scored: list[tuple[SolutionNode, float, dict]] = []
        for node in candidates:
            if not node:
                continue
            text = (node.summary or "") if getattr(node, "summary", None) else (getattr(node, "output", "") or getattr(node, "plan", ""))
            score, dim_scores = await self._score_text_async(ctx, text)
            scored.append((node, score, dim_scores))

        # Choose best
        if not scored:
            return self.base.best_node, {
                "summary": {
                    "iterations": getattr(self.base, "iteration", None),
                    "tree_size": len(self._nodes_iterable()),
                    "elapsed_sec": None,
                    "best_metric": getattr(self.base, "best_metric", None),
                },
                "leaderboard": [],
            }

        scored.sort(key=lambda t: t[1], reverse=True)
        best_node, best_score, best_dims = scored[0]

        # Build a compact report compatible with your existing shape
        lb = []
        for node, s, _dims in scored[: min(5, len(scored))]:
            lb.append({
                "node_id": getattr(node, "id", None),
                "reward": float(s),
                "preview": (node.summary or node.output or node.plan or "")[:160],
            })

        report = {
            "summary": {
                "iterations": getattr(self.base, "iteration", None),
                "tree_size": len(self._nodes_iterable()),
                "elapsed_sec": None,
                "best_metric": float(best_score),
                "counts": {"nodes": len(self._nodes_iterable())},
                "posthoc_scorers": self.enabled_scorers,
                "dimensions": self.dimensions,
            },
            "best": (best_node.to_dict() if best_node else None),
            "leaderboard": lb,
            "posthoc_best_dims": best_dims,
        }
        return best_node, report

    async def _score_text_async(self, context: dict, text: str) -> tuple[float, dict]:
        """
        Score text using container ScoringService across enabled_scorers & dimensions.
        Returns (aggregate_score, per_dim_avg_dict).
        """
        # Make a scorable for this candidate
        scorable = Scorable(
            id=f"cand-{int(time.time()*1_000)}",
            text=text or "",
            target_type=ScorableType.AGENT_OUTPUT,
            meta={"stage_name": "ssp.tree.candidate", "pipeline_run_id": context.get("pipeline_run_id")},
        )

        try:
            scoring = self.container.get("scoring")
        except Exception:
            # No scoring service available; neutral fallback
            return 0.5, {d: 0.5 for d in self.dimensions}

        dim_bag: dict[str, List[float]] = {d: [] for d in self.dimensions}

        # NOTE: scoring.score is sync in your examples; call straight through.
        for scorer_name in self.enabled_scorers:
            try:
                bundle = scoring.score(
                    scorer_name,
                    context=context,
                    scorable=scorable,
                    dimensions=self.dimensions,
                )
                for d in self.dimensions:
                    if d in bundle.results:
                        dim_bag[d].append(float(bundle.results[d].score))
            except Exception as e:
                self.trace_logger.log(
                    PlanTrace_safe(
                        trace_id=f"scorer-{int(time.time()*1000)%1_000_000}",
                        role="solver",
                        goal="posthoc-scoring",
                        status="warning",
                        input={"scorer": scorer_name, "dims": self.dimensions},
                        output="scorer_error",
                        artifacts={"error": str(e)},
                    )
                )
                continue

        dim_avg = {d: (sum(vals) / len(vals) if vals else 0.5) for d, vals in dim_bag.items()}
        final_score = sum(dim_avg.values()) / max(1, len(dim_avg))
        return float(final_score), {k: float(v) for k, v in dim_avg.items()}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_context_for_tree(self, proposal: Dict[str, Any]) -> dict:
        goal_text = str(proposal.get("query", "")).strip()
        difficulty = float(proposal.get("difficulty", self.sp.qmax.initial_difficulty))
        value = max(0.0, min(1.0, difficulty))

        ctx = {
            "goal": {"goal_text": goal_text},
            "task_type": self.cfg.get("task_type", "prompt_improvement"),
            # these are just hints—the post-hoc scorer is authoritative
            "scorer": self.cfg.get("scorer", (self.enabled_scorers[0] if self.enabled_scorers else "tiny")),
            "dimensions": list(self.dimensions),
            "knowledge": proposal.get("connections") or [],
            "value": value,
            "pipeline_run_id": int(time.time() * 1000) % 10_000,
        }
        return ctx

    async def _emit_tree_event(self, event: str, payload: Dict[str, Any]) -> None:
        try:
            if self._tree_emitter:
                await self._tree_emitter.emit(event, payload)
        except Exception:
            pass
        try:
            self.trace_logger.log({
                "trace_id": f"tree-{int(time.time()*1000)}",
                "role": "tree",
                "goal": payload.get("goal") if isinstance(payload, dict) else "",
                "status": event,
                "metadata": {"event": event},
                "input": None,
                "output": "",
                "artifacts": payload,
            })
        except Exception:
            pass

    def _pick_best_from_grpo(self, forest: Dict[str, Any]) -> tuple[Optional[SolutionNode], float]:
        rewards = forest.get("rewards", {}) or {}
        if not rewards:
            return self.base.best_node, float(self.base.best_metric if self.base.best_metric is not None else 0.0)
        best_id = max(rewards, key=rewards.get)
        node = self.base.nodes_by_id.get(best_id)
        return node, float(rewards[best_id])

    def _simple_report_from_grpo(self, forest: Dict[str, Any], best_node: Optional[SolutionNode], best_reward: float) -> Dict[str, Any]:
        lb = sorted(
            [{"node_id": nid, "reward": r} for nid, r in (forest.get("rewards") or {}).items()],
            key=lambda x: x["reward"],
            reverse=True,
        )[:5]
        return {
            "summary": {
                "iterations": getattr(self.base, "iteration", None),
                "tree_size": len(self._nodes_iterable()),
                "elapsed_sec": None,
                "best_metric": best_reward,
                "counts": {"nodes": len(self._nodes_iterable())},
            },
            "best": (best_node.to_dict() if best_node else None),
            "leaderboard": lb,
        }

    def _result_payload_from_node(self, node: Optional[SolutionNode], report: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        answer = ""
        search_depth = 0
        if node:
            answer = (getattr(node, "summary", None) or "") or (getattr(node, "output", None) or "") or (getattr(node, "plan", None) or "")
            search_depth = int(getattr(node, "depth", 0) or 0)

        result = {
            "answer": answer,
            "reasoning_path": report.get("leaderboard", []),
            "evidence": [],
            "search_depth": search_depth,
            "report": report,
        }
        if extra:
            result.update(extra)
        return result

    # Utility: support dict or list internal tree storage
    def _nodes_iterable(self) -> Iterable[SolutionNode]:
        tree = getattr(self.base, "tree", None)
        if isinstance(tree, dict):
            return tree.values()
        return tree or []

    def _try_get_leaderboard(self) -> Optional[List[Dict[str, Any]]]:
        rep = getattr(self.base, "last_report", None)
        if isinstance(rep, dict) and "leaderboard" in rep:
            return rep["leaderboard"] or []
        # some versions stash report in ctx; try to read from base if exposed
        ctx_report = getattr(self.base, "search_report", None)
        if isinstance(ctx_report, dict) and "leaderboard" in ctx_report:
            return ctx_report["leaderboard"] or []
        return None

    # ------------------------------------------------------------------ #
    # Runner
    # ------------------------------------------------------------------ #
    def _run(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        else:
            loop = loop or asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                loop.close()
