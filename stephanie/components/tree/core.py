# stephanie/components/tree/core.py
"""
Agentic Tree Search (MCTS-lite) Core Module.

Fixes & Enhancements (2025-10-30):
- Consistent string IDs (no mixed int/str); stable, sortable, human-readable node IDs
- Correct multi-root initialization (every root gets root_id/path/depth set)
- Result cache typed and enforced (Dict[str, SolutionNode])
- Robust tree_id handling (works with None/str/int)
- SSP hooks to bias action selection using risk/hallucination/novelty/greed signals
- TreeEventEmitter integration for history-first reconstruction:
  root_created, node_added, expand, backprop, best_update, progress, rollout_complete
- Safer fallbacks and small robustness passes
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.tree.events import TreeEventEmitter
from stephanie.components.tree.output_verifier import OutputVerifier
from stephanie.components.tree.plan_generator import PlanGenerator
from stephanie.components.tree.solution_node import SolutionNode
from stephanie.components.tree.task_executor import TaskExecutor
from stephanie.components.tree.task_handler import TaskHandler
from stephanie.utils.hash_utils import hash_text


class AgenticTreeSearch:
    def __init__(
        self,
        agent: BaseAgent,
        max_iterations: int = 500,
        time_limit: int = 24 * 60 * 60,
        N_init: int = 5,
        C_ucb: float = 1.2,
        H_greedy: float = 0.3,
        H_debug: float = 0.5,
        no_improve_patience: int = 50,
        random_seed: Optional[int] = 42,
        metric_fn: Optional[Callable[[Optional[float]], float]] = None,
        emit_cb: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
        progress_every: int = 5,
        heartbeat_secs: float = 20.0,
        report_top_k: int = 5,
        event_emitter: Optional[TreeEventEmitter] = None,
    ):
        # Core agent and component initialization
        self.agent = agent
        self.tree: List[SolutionNode] = []
        self.nodes_by_id: Dict[str, SolutionNode] = {}
        self.parent_map: Dict[str, Optional[str]] = {}
        self.children_map: Dict[Optional[str], List[str]] = {}

        # Search configuration parameters
        self.max_iterations = int(max_iterations)
        self.time_limit = float(time_limit)
        self.N_init = int(N_init)
        self.C_ucb = float(C_ucb)
        self.H_greedy = float(H_greedy)
        self.H_debug = float(H_debug)
        self.no_improve_patience = int(no_improve_patience)
        self.emit_cb = emit_cb
        self.progress_every = max(1, int(progress_every))
        self.heartbeat_secs = float(heartbeat_secs)
        self.report_top_k = max(1, int(report_top_k))
        self._last_progress_ts = time.time()
        self._run_id = self._make_run_id()

        # Modular components
        self.plan_generator = PlanGenerator(agent)
        self.verifier = OutputVerifier()
        self.task_executor = TaskExecutor(agent, agent.container, self.verifier)
        self.task_handler = TaskHandler(
            agent=self,
            task_executor=self.task_executor,
            verifier=self.verifier,
            plan_gen=self.plan_generator,
        )

        # Search state and statistics
        self.iteration = 0
        self.start_time = time.time()
        self.best_node: Optional[SolutionNode] = None
        self.best_metric: float = -float("inf")
        self.last_improve_iter: int = 0

        # MCTS statistics
        self.visits: Dict[str, int] = {}
        self.value: Dict[str, float] = {}

        # Metric processing and caching
        self.metric_fn = metric_fn or (lambda m: -1.0 if m is None else float(m))
        self.count_actions = {"draft": 0, "improve": 0, "debug": 0}
        self.count_buggy = 0
        self.count_nodes = 0
        self.result_cache: Dict[str, SolutionNode] = {}  # plan_hash -> node
        self.metric_policy = "maximize"

        # Event emitter (safe no-op if not provided)
        self.events = event_emitter or TreeEventEmitter(topic="tree")

        if random_seed is not None:
            random.seed(int(random_seed))

    # ---------------------------------------------------------------------- #
    # Entry point
    # ---------------------------------------------------------------------- #

    async def run(self, context: dict) -> dict:
        await self._emit("progress", self._progress_payload(None))
        goal = context.get("goal", {})
        task_description = (goal.get("goal_text") or "").strip()
        if not task_description:
            raise ValueError("Context must contain 'goal.goal_text'")

        await self._emit("start", {"goal": task_description})
        self.events.on_progress({"phase": "start", "goal": task_description})

        # Phase 1: initial drafts
        init_tasks = [
            self.plan_generator.draft_plan(task_description, context)
            for _ in range(self.N_init)
        ]
        for plan in await asyncio.gather(*init_tasks):
            if self._should_stop():
                break
            node = await self._process_plan(
                plan, parent_node=None, node_type="draft",
                task_description=task_description, context=context,
            )
            self._add_node(node)
            self._update_best(node)
            # root or child event is fired inside _add_node
            self.iteration += 1
            self.count_actions["draft"] += 1
            self._maybe_emit_progress(node)

        # Phase 2: main loop
        while not self._should_stop():
            parent = self._select_parent_ucb()
            if parent is not None:
                # Structural signal for UIs/replay: which node we're expanding
                self.events.on_expand(parent)

            action, target = self._choose_action(parent, context)

            if action == "draft" or target is None:
                plan = await self.plan_generator.draft_plan(task_description, context)
                node = await self._process_plan(plan, None, "draft", task_description, context)
            elif action == "improve":
                feedback = (target.summary or "Refine quality.").strip()
                plan = await self.plan_generator.improve_plan(target.plan, feedback, context)
                node = await self._process_plan(plan, target, "improve", task_description, context)
            elif action == "debug":
                error_log = (target.output or target.summary or "Unknown error.").strip()
                plan = await self.plan_generator.debug_plan(target.plan, error_log, context)
                node = await self._process_plan(plan, target, "debug", task_description, context)
            else:
                # dead fallback → draft another root step
                plan = await self.plan_generator.draft_plan(task_description, context)
                node = await self._process_plan(plan, None, "draft", task_description, context)

            self._add_node(node)
            self._backprop(node)
            self._update_best(node)
            self.iteration += 1
            self.count_actions[action] = self.count_actions.get(action, 0) + 1
            self._maybe_emit_progress(node)

        # Phase 3: report
        best = self.best_node.to_dict() if self.best_node else None
        await self._emit("progress", self._progress_payload(None, force_complete=True))
        self.events.on_progress({"phase": "complete", **self._progress_payload(None, force_complete=True)})
        report = self._make_report(self.best_node)
        await self._emit("report", report)
        self.events.on_rollout_complete(report)

        context["search_tree_size"] = len(self.tree)
        context["final_solution"] = best
        context["search_report"] = report
        return context

    # ---------------------------------------------------------------------- #
    # Core processing
    # ---------------------------------------------------------------------- #

    async def _process_plan(
        self,
        plan: str,
        parent_node: Optional[SolutionNode],
        node_type: str,
        task_description: str,
        context: dict,
    ) -> SolutionNode:
        """Execute plan via TaskHandler and package SolutionNode; with dedupe cache."""
        plan_hash = self._hash_or_none(plan)
        if plan_hash and plan_hash in self.result_cache:
            return self.result_cache[plan_hash]

        task = context.get("task") or {"type": context.get("task_type", "code_compile")}
        v = await self.task_handler.handle(task.get("type"), plan, context)

        parent = parent_node
        sibling_idx = self._next_child_index(parent.id if parent else None)
        node_depth = 0 if parent is None else (parent.depth + 1)
        node_path = self._make_path(parent, sibling_idx)

        origin = {
            "action": node_type,
            "task_type": task.get("type"),
            "source_id": parent.id if parent else None,
            "reason": (parent.summary or "") if parent else "initial draft",
        }

        plan_h = self._hash_or_none(plan)
        out_h = self._hash_or_none(v.get("merged_output", ""))

        tree_id_raw = context.get("pipeline_run_id") or self._run_id
        tree_id = self._norm_tree_id(tree_id_raw)
        node_id = self._make_node_id(tree_id, node_depth, sibling_idx, plan_h)

        node = SolutionNode(
            tree_id=str(tree_id),
            id=node_id,
            plan=plan,
            code=None,
            task_description=task_description,
            timestamp=time.time(),
            metric=v.get("metric"),
            output=v.get("merged_output"),
            summary=v.get("summary"),
            parent_id=(parent.id if parent else None),
            is_buggy=bool(v.get("is_bug", False)),
            node_type=node_type,
            root_id=(parent.root_id if parent else None),
            depth=node_depth,
            sibling_index=sibling_idx,
            path=node_path,
            origin=origin,
            lineage=((parent.lineage + [parent.id]) if parent else []),
            plan_sha256=plan_h,
            code_sha256=None,
            output_sha256=out_h,
        )
        if plan_hash:
            self.result_cache[plan_hash] = node

        await self._emit("node", node.to_dict())
        return node

    # ---------------------------------------------------------------------- #
    # MCTS policy
    # ---------------------------------------------------------------------- #

    def _select_parent_ucb(self) -> Optional[SolutionNode]:
        candidates = [n for n in self.tree if not n.is_buggy]
        if not candidates:
            return None
        total_N = sum(self.visits.get(n.id, 1) for n in candidates)

        def ucb(n: SolutionNode) -> float:
            N = self.visits.get(n.id, 1)
            Q = self.value.get(n.id, 0.0)
            return Q + self.C_ucb * ((total_N ** 0.5) / (1 + N))

        if self.iteration < self.N_init * 2 and self.best_node:
            return self.best_node
        return max(candidates, key=ucb)

    def _choose_action(
        self,
        parent: Optional[SolutionNode],
        context: Optional[dict] = None
    ) -> Tuple[str, Optional[SolutionNode]]:
        """
        Action policy with SSP overrides.
        SSP (if enabled) can force 'debug' on high risk/hallucination,
        or encourage 'draft' on high novelty exploration signals.
        """
        # SSP override (if present)
        if context:
            forced = self._ssp_action_override(parent, context)
            if forced is not None:
                if forced == "draft":
                    return "draft", None
                if forced == "debug":
                    # Debug the selected parent; fallback to best if parent is missing
                    return "debug", (parent or self.best_node)
                if forced == "improve":
                    # Prefer improving best; fallback to parent or draft
                    if self.best_node is not None:
                        return "improve", self.best_node
                    if parent is not None:
                        return "improve", parent
                    return "draft", None

        # default heuristic
        if parent is None:
            return "draft", None
        if self.best_node and random.random() < self.H_greedy:
            return "improve", self.best_node
        if parent.is_buggy and random.random() < self.H_debug:
            return "debug", parent
        return "improve", parent

    # ---------------------------------------------------------------------- #
    # Bookkeeping and scoring
    # ---------------------------------------------------------------------- #

    def _add_node(self, node: SolutionNode) -> None:
        """
        Adds node and maintains relationships. Fixed multi-root init:
        any node with parent_id=None is a root and gets proper path/depth,
        and emits root/child structural events.
        """
        is_root = node.parent_id is None
        if is_root:
            # ensure root attributes set per root, not just the first
            node.root_id = node.id
            node.depth = 0
            node.sibling_index = self._next_child_index(None)
            node.path = str(node.sibling_index)
            self.children_map.setdefault(None, []).append(node.id)

        self.tree.append(node)
        self.nodes_by_id[node.id] = node
        self.parent_map[node.id] = node.parent_id

        if node.parent_id:
            self.children_map.setdefault(node.parent_id, []).append(node.id)
            p = self.nodes_by_id.get(node.parent_id)
            if p:
                p.children.append(node.id)
                node.root_id = p.root_id or p.id

        self.visits.setdefault(node.id, 0)
        self.value.setdefault(node.id, 0.0)
        self.count_nodes += 1
        if node.is_buggy:
            self.count_buggy += 1

        # Structural events for history-first UI/replay
        try:
            if is_root:
                self.events.on_root_created(node)
            else:
                parent = self.nodes_by_id.get(node.parent_id) if node.parent_id else None
                self.events.on_node_added(parent, node)
        except Exception:
            # never break the search because of telemetry
            pass

    def _backprop(self, leaf: SolutionNode) -> None:
        reward = self.metric_fn(leaf.metric)
        node_id = leaf.id
        while node_id is not None:
            self.visits[node_id] = self.visits.get(node_id, 0) + 1
            v_prev = self.value.get(node_id, 0.0)
            n = self.visits[node_id]
            self.value[node_id] = v_prev + (reward - v_prev) / n
            node_id = self.parent_map.get(node_id)

        # backprop event anchored at the leaf with computed reward
        try:
            self.events.on_backprop(leaf, delta=float(reward))
        except Exception:
            pass

    def _update_best(self, node: SolutionNode) -> None:
        m = self.metric_fn(node.metric)
        if self.metric_policy == "minimize":
            m = -m
        if m > self.best_metric:
            self.best_metric = m
            self.best_node = node
            self.last_improve_iter = self.iteration
            try:
                self.events.on_best_update(node)
            except Exception:
                pass

    # ---------------------------------------------------------------------- #
    # Utility and progress
    # ---------------------------------------------------------------------- #

    def _should_stop(self) -> bool:
        if self.iteration >= self.max_iterations:
            return True
        if (time.time() - self.start_time) > self.time_limit:
            return True
        if (self.iteration - self.last_improve_iter) >= self.no_improve_patience and self.iteration > 0:
            return True
        return False

    async def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.emit_cb:
            return
        try:
            await self.emit_cb(event, payload)
        except Exception:
            pass

    def _progress_payload(self, last_node: Optional[SolutionNode], force_complete: bool = False) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        frac_iter = self.iteration / max(1, self.max_iterations)
        frac_time = elapsed / max(1e-6, self.time_limit)
        progress = 1.0 if force_complete else min(1.0, max(frac_iter, frac_time))

        last = None
        if last_node:
            last = {
                "id": last_node.id,
                "parent_id": last_node.parent_id,
                "type": last_node.node_type,
                "metric": last_node.metric,
                "buggy": last_node.is_buggy,
            }

        return {
            "progress": round(progress, 4),
            "elapsed_sec": round(elapsed, 3),
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "tree_size": len(self.tree),
            "best_metric": self.best_metric if self.best_metric != -float("inf") else None,
            "last_node": last,
            "counts": {
                "draft": self.count_actions.get("draft", 0),
                "improve": self.count_actions.get("improve", 0),
                "debug": self.count_actions.get("debug", 0),
                "nodes": self.count_nodes,
                "buggy": self.count_buggy,
            },
        }

    def _maybe_emit_progress(self, last_node: Optional[SolutionNode]) -> None:
        now = time.time()
        if not ((self.iteration % self.progress_every) == 0 or (now - self._last_progress_ts) >= self.heartbeat_secs):
            return
        self._last_progress_ts = now
        payload = self._progress_payload(last_node)
        asyncio.create_task(self._emit("progress", payload))
        try:
            self.events.on_progress(payload)
        except Exception:
            pass

    def _make_report(self, best_node: Optional[SolutionNode]) -> Dict[str, Any]:
        leaderboard = sorted(
            (n for n in self.tree if n.metric is not None),
            key=lambda n: n.metric,
            reverse=True,
        )[: self.report_top_k]

        return {
            "summary": {
                "iterations": self.iteration,
                "tree_size": len(self.tree),
                "elapsed_sec": round(time.time() - self.start_time, 3),
                "best_metric": self.best_metric if self.best_metric != -float("inf") else None,
                "counts": {
                    "draft": self.count_actions.get("draft", 0),
                    "improve": self.count_actions.get("improve", 0),
                    "debug": self.count_actions.get("debug", 0),
                    "nodes": self.count_nodes,
                    "buggy": self.count_buggy,
                },
            },
            "best": (best_node.to_dict() if best_node else None),
            "leaderboard": [
                {
                    "id": n.id,
                    "parent_id": n.parent_id,
                    "type": n.node_type,
                    "metric": n.metric,
                    "summary": (n.summary or "")[:300],
                    "ts": n.timestamp,
                }
                for n in leaderboard
            ],
        }

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _hash_or_none(self, s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        return hash_text(s)

    def _short_hash(self, s: Optional[str], n: int = 8) -> str:
        h = self._hash_or_none(s) or uuid.uuid4().hex
        return h[:n]

    def _next_child_index(self, parent_id: Optional[str]) -> int:
        if not parent_id:
            return len(self.children_map.get(None, []))
        return len(self.children_map.get(parent_id, []))

    def _make_path(self, parent_node: Optional[SolutionNode], sibling_index: int) -> str:
        if parent_node is None:
            return str(sibling_index)
        return f"{parent_node.path}.{sibling_index}"

    def _make_run_id(self) -> str:
        # time-based short run id
        return f"run{int(time.time()) % 10_000_000:07d}"

    def _norm_tree_id(self, tree_id: Any) -> str:
        try:
            return str(int(tree_id))
        except Exception:
            return str(tree_id)

    def _make_node_id(self, tree_id: str, node_depth: int, sibling_idx: int, plan_hash: Optional[str]) -> str:
        # Sortable and human-readable: TTTTTTT-DVVV-SIIII-HHHHHHHH
        return f"{tree_id}-{node_depth:03d}-{sibling_idx:04d}-{(plan_hash or '')[:8] or self._short_hash('')}"

    def node_reward(self, node: SolutionNode) -> float:
        m = self.metric_fn(node.metric)
        return -m if self.metric_policy == "minimize" else m

    def candidates(self) -> List[SolutionNode]:
        return [n for n in self.tree if not n.is_buggy]

    # ---------------------------------------------------------------------- #
    # SSP integration
    # ---------------------------------------------------------------------- #

    def _ssp_action_override(
        self, parent: Optional[SolutionNode], context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Minimal SSP policy bridge.
        Expects optional context["ssp"] dict with:
          - risk in [0,1]: high risk → 'debug'
          - hallucination in [0,1]: high → 'debug'
          - novelty in [0,1]: high → 'draft'
          - greed in [0,1]: high → 'improve'
        Thresholds are conservative so it only nudges when signals are clear.
        """
        ssp = context.get("ssp") or {}
        risk = float(ssp.get("risk", 0.0))
        hallu = float(ssp.get("hallucination", 0.0))
        novelty = float(ssp.get("novelty", 0.0))
        greed = float(ssp.get("greed", 0.0))

        # strong risk/hallucination → debug
        if (risk >= 0.75 or hallu >= 0.75) and (parent is not None):
            return "debug"

        # strong novelty → explore a fresh draft
        if novelty >= 0.8:
            return "draft"

        # strong greed/exploitation → improve best
        if greed >= 0.8:
            return "improve"

        return None
