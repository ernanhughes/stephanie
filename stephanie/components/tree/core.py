# stephanie/components/tree/core.py
"""
Agentic Tree Search (MCTS-lite) Core Module.

This module implements a Monte Carlo Tree Search-inspired reasoning system for 
autonomous problem solving. It combines planning, execution, and verification 
in a tree-based search framework with modular components.

Key Components:
- SolutionNode: Data structure for tree nodes
- PlanGenerator: Generates and improves plans
- TaskExecutor: Executes plans and produces outputs  
- OutputVerifier: Validates and scores outputs
- TaskHandler: Dispatches tasks based on type

The system explores solution space through iterative drafting, improvement, 
and debugging cycles while maintaining search statistics and progress tracking.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.tree.output_verifier import OutputVerifier
from stephanie.components.tree.plan_generator import PlanGenerator
from stephanie.components.tree.solution_node import SolutionNode
from stephanie.components.tree.task_executor import TaskExecutor
from stephanie.components.tree.task_handler import TaskHandler


class AgenticTreeSearch:
    """
    Monte-Carlo-lite reasoning tree with pluggable task handlers.
    
    This class implements a tree search algorithm that combines MCTS principles
    with agentic reasoning. It maintains a tree of solution attempts where each
    node represents a plan-execution-verification cycle.
    
    The search proceeds through:
    1. Initial draft generation
    2. Iterative improvement cycles  
    3. Debugging of failed attempts
    4. UCB-based parent selection for expansion
    
    Attributes:
        agent: The base agent used for planning and execution
        tree: List of all solution nodes in search order
        nodes_by_id: Dictionary mapping node IDs to node objects
        parent_map: Dictionary tracking parent-child relationships
        children_map: Dictionary of children for each parent node
        max_iterations: Maximum number of search iterations
        time_limit: Maximum search time in seconds
        N_init: Number of initial draft plans
        C_ucb: UCB exploration constant
        H_greedy: Probability of greedy improvement on best node
        H_debug: Probability of debugging buggy nodes
        no_improve_patience: Iterations without improvement before stopping
        emit_cb: Callback for emitting progress events
        progress_every: Emit progress every N iterations
        heartbeat_secs: Minimum time between progress emissions
        report_top_k: Number of top solutions to include in final report
    """

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
    ):
        # Core agent and component initialization
        self.agent = agent
        self.tree: List[SolutionNode] = []
        self.nodes_by_id: Dict[str, SolutionNode] = {}
        self.parent_map: Dict[str, Optional[str]] = {}
        self.children_map: Dict[str, List[str]] = {}

        # Search configuration parameters
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.N_init = N_init
        self.C_ucb = C_ucb
        self.H_greedy = H_greedy
        self.H_debug = H_debug
        self.no_improve_patience = no_improve_patience
        self.emit_cb = emit_cb
        self.progress_every = max(1, int(progress_every))
        self.heartbeat_secs = float(heartbeat_secs)
        self.report_top_k = max(1, int(report_top_k))
        self._last_progress_ts = time.time()

        # Modular components for different search phases
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

        # MCTS statistics for node selection
        self.visits: Dict[str, int] = {}
        self.value: Dict[str, float] = {}

        # Metric processing and caching
        self.metric_fn = metric_fn or (lambda m: -1.0 if m is None else float(m))
        self.count_actions = {"draft": 0, "improve": 0, "debug": 0}
        self.count_buggy = 0
        self.count_nodes = 0
        self.result_cache: Dict[str, Dict[str, Any]] = {}
        self.metric_policy = "maximize"

        # Random seed for reproducible search
        if random_seed is not None:
            random.seed(random_seed)

    # ---------------------------------------------------------------------- #
    # Entry point
    # ---------------------------------------------------------------------- #

    async def run(self, context: dict) -> dict:
        """
        Execute the complete tree search process.
        
        Args:
            context: Dictionary containing task context including 'goal' with 'goal_text'
            
        Returns:
            Updated context with search results and final solution
            
        Raises:
            ValueError: If context doesn't contain a valid task description
        """
        await self._emit("progress", self._progress_payload(None))
        goal = context.get("goal", {})
        task_description = goal.get("goal_text", "").strip()
        if not task_description:
            raise ValueError("Context must contain 'goal.goal_text'")

        await self._emit("start", {"goal": task_description})

        # Phase 1: Generate initial draft plans
        init_tasks = [
            self.plan_generator.draft_plan(task_description, context)
            for _ in range(self.N_init)
        ]
        for plan in await asyncio.gather(*init_tasks):
            if self._should_stop():
                break
            node = await self._process_plan(
                plan,
                parent_node=None,
                node_type="draft",
                task_description=task_description,
                context=context,
            )
            self._add_node(node)
            self._update_best(node)
            self.iteration += 1
            self.count_actions["draft"] += 1
            self._maybe_emit_progress(node)

        # Phase 2: Main search loop with UCB selection
        while not self._should_stop():
            parent = self._select_parent_ucb()
            action, target = self._choose_action(parent)

            if action == "draft" or target is None:
                plan = await self.plan_generator.draft_plan(task_description, context)
                node = await self._process_plan(plan, None, "draft", task_description, context)
            elif action == "improve":
                feedback = target.summary or "No specific feedback."
                plan = await self.plan_generator.improve_plan(target.plan, feedback, context)
                node = await self._process_plan(plan, target, "improve", task_description, context)
            elif action == "debug":
                error_log = target.output or target.summary or "Unknown error."
                plan = await self.plan_generator.debug_plan(target.plan, error_log, context)
                node = await self._process_plan(plan, target, "debug", task_description, context)
            else:
                raise ValueError(f"Unknown action: {action}")

            self._add_node(node)
            self._backprop(node)
            self._update_best(node)
            self.iteration += 1
            self.count_actions[action] = self.count_actions.get(action, 0) + 1
            self._maybe_emit_progress(node)

        # Phase 3: Final results compilation
        best = self.best_node.to_dict() if self.best_node else None
        await self._emit("progress", self._progress_payload(None, force_complete=True))
        report = self._make_report(self.best_node)
        await self._emit("report", report)

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
        """
        Process a plan through the appropriate task handler.
        
        Delegates execution based on task type and creates a SolutionNode
        with execution results. Uses caching to avoid duplicate processing.
        
        Args:
            plan: The plan string to execute
            parent_node: Parent node in search tree (None for root)
            node_type: Type of node ('draft', 'improve', 'debug')
            task_description: Description of the overall task
            context: Execution context dictionary
            
        Returns:
            SolutionNode containing execution results
        """
        plan_hash = self._hash_or_none(plan)
        if plan_hash in self.result_cache:
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
        tree_id = context.get("pipeline_run_id")
        root_id = parent.root_id if parent else None

        id = self._make_numeric_id(tree_id, node_depth, sibling_idx)
        node = SolutionNode(
            tree_id=tree_id,
            id=id,
            plan=plan,
            code=None,
            task_description=task_description,
            timestamp=time.time(), 
            metric=v.get("metric"),
            output=v.get("merged_output"),
            summary=v.get("summary"),
            parent_id=parent.id if parent else None,
            is_buggy=v.get("is_bug", False),
            node_type=node_type,
            root_id=root_id,
            depth=node_depth,
            sibling_index=sibling_idx,
            path=node_path,
            origin=origin,
            lineage=((parent.lineage + [parent.id]) if parent else []),
            plan_sha256=plan_h,
            code_sha256=None,
            output_sha256=out_h,
        )
        self.result_cache[plan_hash] = node
        await self._emit("node", node.to_dict())
        return node

    # ---------------------------------------------------------------------- #
    # MCTS policy
    # ---------------------------------------------------------------------- #

    def _select_parent_ucb(self) -> Optional[SolutionNode]:
        """
        Select parent node for expansion using UCB formula.
        
        Balances exploration (less-visited nodes) and exploitation (high-value nodes).
        Uses modified UCB: Q + C * sqrt(total_N) / (1 + N)
        
        Returns:
            Selected parent node, or None if no valid candidates
        """
        candidates = [n for n in self.tree if not n.is_buggy]
        if not candidates:
            return None
        total_N = sum(self.visits.get(n.id, 1) for n in candidates)

        def ucb(n: SolutionNode) -> float:
            N = self.visits.get(n.id, 1)
            Q = self.value.get(n.id, 0.0)
            return Q + self.C_ucb * ((total_N**0.5) / (1 + N))

        # Early search bias: favor best node for first iterations
        if self.iteration < self.N_init * 2 and self.best_node:
            return self.best_node
        return max(candidates, key=ucb)

    def _choose_action(self, parent: Optional[SolutionNode]) -> Tuple[str, Optional[SolutionNode]]:
        """
        Choose next action based on current state and heuristics.
        
        Args:
            parent: Selected parent node for expansion
            
        Returns:
            Tuple of (action_type, target_node) where action_type is one of:
            'draft', 'improve', 'debug'
        """
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
        Add a new node to the search tree and update data structures.
        
        Maintains parent-child relationships, tree statistics, and node metadata.
        """
        if node.parent_id is None and not any(n.parent_id is None for n in self.tree):
            node.root_id = node.id
            node.path = "0"
            node.depth = 0
            node.sibling_index = 0
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

    def _backprop(self, leaf: SolutionNode) -> None:
        """
        Backpropagate reward values through the tree.
        
        Updates node values and visit counts from leaf to root using
        incremental average calculation.
        
        Args:
            leaf: Leaf node where the current simulation ended
        """
        reward = self.metric_fn(leaf.metric)
        node_id = leaf.id
        while node_id is not None:
            self.visits[node_id] = self.visits.get(node_id, 0) + 1
            v_prev = self.value.get(node_id, 0.0)
            n = self.visits[node_id]
            self.value[node_id] = v_prev + (reward - v_prev) / n
            node_id = self.parent_map.get(node_id)

    def _update_best(self, node: SolutionNode) -> None:
        """
        Update the best solution found so far.
        
        Args:
            node: Candidate node to compare against current best
        """
        m = self.metric_fn(node.metric)
        if self.metric_policy == "minimize":
            m = -m
        if m > self.best_metric:
            self.best_metric = m
            self.best_node = node
            self.last_improve_iter = self.iteration

    # ---------------------------------------------------------------------- #
    # Utility and progress
    # ---------------------------------------------------------------------- #

    def _should_stop(self) -> bool:
        """Check if search should terminate based on stopping criteria."""
        if self.iteration >= self.max_iterations:
            return True
        if (time.time() - self.start_time) > self.time_limit:
            return True
        if (self.iteration - self.last_improve_iter) >= self.no_improve_patience and self.iteration > 0:
            return True
        return False

    async def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        """
        Emit event via callback if configured.
        
        Args:
            event: Event type ('start', 'progress', 'node', 'report')
            payload: Event-specific data payload
        """
        if not self.emit_cb:
            return
        try:
            await self.emit_cb(event, payload)
        except Exception:
            pass

    def _progress_payload(self, last_node: Optional[SolutionNode], force_complete: bool = False) -> Dict[str, Any]:
        """
        Generate progress report payload.
        
        Args:
            last_node: Most recently processed node
            force_complete: Whether to force progress to 100%
            
        Returns:
            Dictionary containing progress metrics and statistics
        """
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
        """
        Emit progress report if conditions are met.
        
        Conditions: iteration multiple of progress_every OR heartbeat timeout reached.
        """
        now = time.time()
        if not ((self.iteration % self.progress_every) == 0 or (now - self._last_progress_ts) >= self.heartbeat_secs):
            return
        self._last_progress_ts = now
        asyncio.create_task(self._emit("progress", self._progress_payload(last_node)))

    def _make_report(self, best_node: Optional[SolutionNode]) -> Dict[str, Any]:
        """
        Generate final search report.
        
        Args:
            best_node: The best solution node found
            
        Returns:
            Comprehensive report dictionary with summary and leaderboard
        """
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

    def _hash_or_none(self, s: Optional[str]) -> Optional[str]:
        """Generate SHA256 hash of string or return None for empty input."""
        if not s:
            return None
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _next_child_index(self, parent_id: Optional[str]) -> int:
        """Get next available sibling index for a parent node."""
        if not parent_id:
            return len(self.children_map.get(None, []))
        return len(self.children_map.get(parent_id, []))

    def _make_path(self, parent_node: Optional[SolutionNode], sibling_index: int) -> str:
        """Generate hierarchical path string for node positioning."""
        if parent_node is None:
            return str(sibling_index)
        return f"{parent_node.path}.{sibling_index}"

    def _make_numeric_id(self, tree_id: int, node_depth: int, sibling_idx: int) -> int:
        """
        Build a sortable numeric ID from (tree, depth, sibling).
        
        Layout: TTTTDDDSIII  (tree_id up to 9999, depth up to 999, sibling up to 9999)
        Example: tree 12, depth 3, sibling 45 -> 120030045
        """
        return (int(tree_id) * 10**7) + (node_depth * 10**4) + sibling_idx