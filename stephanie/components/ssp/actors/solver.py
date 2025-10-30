# stephanie/components/ssp/actors/solver.py
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Iterable, List, Optional

from omegaconf import DictConfig, OmegaConf

# ✅ NEW import path: decoupled, reusable emitter from tree package
from stephanie.components.tree.events import TreeEventEmitter

from stephanie.components.ssp.util import PlanTrace_safe, get_trace_logger
from stephanie.components.tree.core import AgenticTreeSearch, SolutionNode
from stephanie.components.tree.tree_grpo import TreeGRPOAdapter, TreeGRPOConfig
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.service_container import ServiceContainer
from stephanie.utils.json_sanitize import sanitize


def _maybe(val, default):
    return val if val is not None else default


class _ContainerLLMAgent:
    """
    Minimal adapter exposing async_call_llm(prompt, context) so AgenticTreeSearch
    can remain agnostic. Internally uses container.get("prompt").run_prompt(...).
    """
    def __init__(self, container: ServiceContainer, sys_preamble: str | None = None):
        self.prompt_service = container.get("prompt")
        self.container = container
        self.sys_preamble = (sys_preamble or "").strip()

    async def async_call_llm(self, prompt: str, context: dict) -> str:
        full_prompt = f"{self.sys_preamble}\n{prompt}" if self.sys_preamble else prompt
        return await self.prompt_service.run_prompt(full_prompt, context)


class Solver:
    """
    SSP Solver that delegates search to the Tree component, but:
    - Uses PromptService from ServiceContainer for LLM calls
    - Post-hoc scores candidates via ScoringService to pick the winner
    - Optionally uses Tree-GRPO (kept intact)
    """

    def __init__(self, cfg: DictConfig | dict, memory, container: ServiceContainer):

        root = cfg
        self.root: DictConfig = root
        self.sp: DictConfig = root.self_play
        self.cfg: DictConfig = self.sp.solver
        self.memory = memory
        self.container = container

        # Trace logging
        self.trace_logger = get_trace_logger()

        # ---- LLM agent
        sys_preamble = self.cfg.get("llm_preamble") or (
            "You are a careful problem solver. Think step-by-step and keep answers concise."
        )
        self.agent = _ContainerLLMAgent(container=container, sys_preamble=sys_preamble)

        # ---- Build base tree search
        max_iterations = int(_maybe(self.cfg.get("search_iterations"),
                                    max(1, int(self.cfg.get("search_depth", 5)) * 4)))
        time_limit = int(_maybe(self.cfg.get("time_limit_sec"), 60))

        # ---- Event emitter (publisher/memory are optional)
        bus = self.memory.bus

        # Optional tracer that mirrors events into your PlanTrace log
        def _plantrace_tracer(event: str, rec: Dict[str, Any]) -> None:
            try:
                self.trace_logger.log(
                    PlanTrace_safe(
                        trace_id=f"tree-{event}-{int(rec.get('ts', time.time())*1000)%1_000_000}",
                        role="tree",
                        goal_text=str(event),
                        goal_id=0,
                        plan_signature="",
                        status="event",
                        meta={"topic": rec.get("topic", "ssp.tree")},
                        execution_steps=[],
                        input_data={},
                        final_output_text="",
                        extra_data=rec,
                    )
                )
            except Exception:
                pass

        self._tree_emitter = TreeEventEmitter(
            topic="ssp.tree",
            publisher=bus,
            memory=self.memory,
            tracer=_plantrace_tracer,
            throttle_ms=int(_maybe(self.cfg.get("throttle_ms"), 50)),
            keep_recent=int(_maybe(self.cfg.get("emit_keep_recent"), 256)),
        )

        # ---- Adapter from AgenticTreeSearch.emit_cb(event, payload_dict)
        async def tree_emit_adapter(event: str, payload_dict: Dict[str, Any]) -> None:
            """
            Core expects: emit_cb(event, payload_dict)
            Emitter expects kwargs: emit(event, **payload)
            """
            try:
                if self._tree_emitter is not None:
                    await self._tree_emitter.emit(event, **(payload_dict or {}))
                # Mirror a couple of useful events onto the search object for downstream reads
                try:
                    if event == "report":
                        setattr(self.base, "last_report", payload_dict)
                    elif event == "progress":
                        setattr(self.base, "last_progress", payload_dict)
                except Exception:
                    pass
            except Exception as e:
                # Don’t break the tree on telemetry failure
                self.trace_logger.log(
                    PlanTrace_safe(
                        trace_id=f"emit-error-{int(time.time()*1000)}",
                        role="solver",
                        goal_text="event-emission",
                        goal_id=0,
                        plan_signature="",
                        status="error",
                        meta={},
                        execution_steps=[],
                        input_data={"event": event},
                        final_output_text=str(e),
                        extra_data={"payload": sanitize(payload_dict)},
                    )
                )

        # ---- Initialize AgenticTreeSearch WITH THE ADAPTER
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
            emit_cb=tree_emit_adapter,  # ← connects tree → emitter
        )

        if self.cfg.get("metric_policy") in ("minimize", "maximize"):
            self.base.metric_policy = self.cfg.metric_policy

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

    async def solve(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Async entry point. Runs the full solve cycle."""
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
                    if not best_node:
                        best_node, report = await self._posthoc_pick_best(ctx)
                        result = self._result_payload_from_node(
                            best_node,
                            report,
                            extra={"training_batch": forest.get("training_batch")}
                        )
                    else:
                        report = self._simple_report_from_grpo(forest, best_node, best_reward)
                        result = self._result_payload_from_node(
                            best_node,
                            report,
                            extra={"training_batch": forest.get("training_batch")}
                        )
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
                        goal_text=goal,
                        goal_id=0,
                        plan_signature="",
                        status="completed",
                        meta={
                            "nodes": len(self._nodes_iterable()),
                            "best_metric": (self.base.best_metric if getattr(self.base, 'best_metric', None) is not None else None),
                            "use_grpo": self.use_grpo,
                        },
                        execution_steps=[],
                        input_data=goal,
                        final_output_text=result.get("answer", ""),
                        extra_data=sanitize(result),
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
        # Prefer leaderboard from last_report emitted by the tree
        leaderboard = self._try_get_leaderboard()
        if leaderboard:
            node_ids = [it.get("id") or it.get("node_id") for it in leaderboard]
            node_ids = [nid for nid in node_ids if nid is not None][: self.posthoc_top_k]
            candidates = [self.base.nodes_by_id.get(nid) for nid in node_ids if nid in self.base.nodes_by_id]
        else:
            nodes_all = list(self._nodes_iterable())
            textful = [n for n in nodes_all if n and (getattr(n, "summary", None) or getattr(n, "output", None) or getattr(n, "plan", None))]
            candidates = textful[-self.posthoc_top_k:] if len(textful) > self.posthoc_top_k else textful

        scored: list[tuple[SolutionNode, float, dict]] = []
        for node in candidates:
            if not node:
                continue
            text = (node.summary or "") if getattr(node, "summary", None) else (getattr(node, "output", "") or getattr(node, "plan", ""))
            score, dim_scores = await self._score_text_async(ctx, text)
            scored.append((node, score, dim_scores))

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
        scorable = Scorable(
            id=f"cand-{int(time.time()*1_000)}",
            text=text or "",
            target_type=ScorableType.AGENT_OUTPUT,
            meta={"stage_name": "ssp.tree.candidate", "pipeline_run_id": context.get("pipeline_run_id")},
        )

        try:
            scoring = self.container.get("scoring")
        except Exception:
            return 0.5, {d: 0.5 for d in self.dimensions}

        dim_bag: dict[str, List[float]] = {d: [] for d in self.dimensions}

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
                        goal_text="posthoc-scoring",
                        goal_id=0,
                        plan_signature="",
                        status="warning",
                        meta={},
                        execution_steps=[],
                        input_data={"scorer": scorer_name, "dims": self.dimensions},
                        final_output_text="scorer_error",
                        extra_data={"error": str(e)},
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

        return {
            "goal": {"goal_text": goal_text},
            "task_type": self.cfg.get("task_type", "prompt_improvement"),
            "scorer": self.cfg.get("scorer", (self.enabled_scorers[0] if self.enabled_scorers else "tiny")),
            "dimensions": list(self.dimensions),
            "knowledge": proposal.get("connections") or [],
            "value": value,
            "pipeline_run_id": int(time.time() * 1000) % 10_000,
        }

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

    def _nodes_iterable(self) -> Iterable[SolutionNode]:
        tree = getattr(self.base, "tree", None)
        if isinstance(tree, dict):
            return tree.values()
        return tree or []

    def _try_get_leaderboard(self) -> Optional[List[Dict[str, Any]]]:
        rep = getattr(self.base, "last_report", None)
        if isinstance(rep, dict) and "leaderboard" in rep:
            return rep["leaderboard"] or []
        ctx_report = getattr(self.base, "search_report", None)
        if isinstance(ctx_report, dict) and "leaderboard" in ctx_report:
            return ctx_report["leaderboard"] or []
        return None
