# stephanie/agents/agentic_tree_search.py
"""
Agentic Tree Search (MCTS‑lite) for Stephanie
--------------------------------------------
Key upgrades vs. your original:
- **SolutionNode** is now a dataclass with stable IDs and parent/child links.
- **MCTS‑lite policy** using UCB1 over candidate parents; backpropagates value.
- **Richer OutputVerifier** extracts more metrics (acc, f1, auc, rmse; % and floats),
  merges stdout/stderr, and summarizes robustly.
- **Safer code execution** in an isolated temp folder with `sys.executable -I`,
  60s timeout, and full cleanup.
- **Deterministic sampling** (seedable) + early‑stop on no‑improve.
- **Async fan‑out** for initial drafts.
- **Extensibility hooks**: custom metric_fn, reward shaping, and emit callbacks.

This module remains framework‑friendly: it only depends on `BaseAgent.llm` and
returns a `final_solution` dict in the passed `context`.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.utils.hash_utils import hash_text

# ----------------------------- Data structures ----------------------------- #


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    has_submission_file: bool = False


@dataclass
class SolutionNode:
    plan: str
    code: Optional[str] = None
    metric: Optional[float] = None
    output: Optional[str] = None
    summary: Optional[str] = None
    parent_id: Optional[str] = None
    is_buggy: bool = False
    node_type: str = "draft"  # 'draft', 'improve', 'debug'
    timestamp: float = field(default_factory=lambda: time.time())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    children: List[str] = field(default_factory=list)

    # --- NEW provenance fields ---
    tree_id: Optional[str] = None         # run_id or external tree key
    root_id: Optional[str] = None         # UUID of root node
    depth: int = 0                        # 0 for root
    sibling_index: int = 0                # position among siblings
    path: str = "0"                       # dot path e.g., "0.2.1"
    origin: Dict[str, Any] = field(default_factory=dict)  # action/source/reason
    lineage: List[str] = field(default_factory=list)      # ordered ids root..parent

    # integrity hashes (content proofs)
    plan_sha256: Optional[str] = None
    code_sha256: Optional[str] = None
    output_sha256: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # aliases for downstream consumers
        d.setdefault("prompt_text", self.plan)
        d.setdefault("compiled_prompt", self.plan)
        return d



# ----------------------------- Plan generator ------------------------------ #


class PlanGenerator:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def draft_plan(self, task_description: str, context: dict) -> str:
        knowledge = context.get("knowledge", [])
        prompt = f"""
You are an expert ML engineer.
Create a detailed solution plan for this task (no code):
{task_description}

Relevant tricks from past solutions: {" ".join(knowledge) if knowledge else "None"}
Return only the plan text.
"""
        response = await self.agent.async_call_llm(prompt, context=context)
        return response.strip()

    async def improve_plan(
        self, previous_plan: str, feedback: str, context: dict
    ) -> str:
        knowledge = context.get("knowledge", [])
        prompt = f"""
Improve this ML solution plan (no code):
{previous_plan}

Feedback: {feedback}
Additional knowledge: {" ".join(knowledge) if knowledge else "None"}
Return only the improved plan text.
"""
        response = await self.agent.async_call_llm(prompt, context=context)
        return response.strip()

    async def debug_plan(
        self, previous_plan: str, error_log: str, context: dict
    ) -> str:
        prompt = f"""
Fix this buggy ML solution plan (no code). Plan:
{previous_plan}

Error log:
{error_log}
Return only the corrected plan text.
"""
        response = await self.agent.async_call_llm(prompt, context=context)
        return response.strip()


# ------------------------------ Verification ------------------------------- #


class OutputVerifier:
    """Extracts signals from program output and stderr."""

    # common metrics: add patterns as needed
    _METRIC_PATTERNS = [
        r"val[_\s]?accuracy[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"accuracy[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"f1[_\s]?(score)?[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"auc[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"rmse[:=]\s*([0-9]*\.?[0-9]+)",
        r"score[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"metric[:=]\s*([0-9]*\.?[0-9]+%?)",
    ]

    def __init__(self, prefer_higher: bool = True):
        self.prefer_higher = prefer_higher

    def verify(
        self, stdout: str, stderr: str, has_submission_file: bool
    ) -> Dict[str, Any]:
        merged = self._merge_streams(stdout, stderr)
        is_bug = any(k in merged for k in ("Traceback", "Exception", "Error"))
        is_overfitting = "val_loss increasing" in merged.lower()
        metric = self.extract_metric(merged)
        summary = self.summarize(merged)
        return {
            "is_bug": is_bug,
            "is_overfitting": is_overfitting,
            "has_csv_submission": has_submission_file,
            "metric": metric,
            "summary": summary,
            "merged_output": merged,
        }

    def extract_metric(self, text: str) -> Optional[float]:
        for pattern in self._METRIC_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            # pick the last numeric group
            g = next(
                (grp for grp in m.groups()[::-1] if grp is not None), None
            )
            if not g:
                continue
            try:
                if g.endswith("%"):
                    return float(g[:-1]) / 100.0
                val = float(g)
                # normalize RMSE (lower is better) by inverting to a bounded reward
                if "rmse" in pattern.lower():
                    return 1.0 / (1.0 + val)
                return val
            except ValueError:
                continue
        return None

    def summarize(self, text: str, tail_lines: int = 8) -> str:
        lines = [ln.strip() for ln in text.strip().splitlines()[-tail_lines:]]
        return " ".join(lines) if lines else "No output."

    @staticmethod
    def _merge_streams(stdout: str, stderr: str) -> str:
        if not stderr:
            return stdout or ""
        if not stdout:
            return stderr or ""
        return stdout + "\n--- STDERR ---\n" + stderr


# ------------------------------ Code executor ------------------------------ #


class CodeExecutor:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def score_complexity(
        self, task_description: str, plan: str, context: dict
    ) -> float:
        prompt = f"""
Rate the complexity of this task and plan on a scale of 1–5 (1 = simple, 5 = very complex).
Task: {task_description}
Plan: {plan}

Respond with a single number between 1 and 5.
"""
        response = await self.agent.async_call_llm(prompt, context=context)
        try:
            score = float(re.findall(r"\d+(?:\.\d+)?", response.strip())[0])
            return max(1.0, min(5.0, score))
        except Exception:
            return 3.0

    @staticmethod
    def _strip_markdown(code: str) -> str:
        # remove triple‑fenced blocks if present
        fence = re.compile(r"^```(?:python)?\n|\n```$", re.IGNORECASE)
        return fence.sub("", code).strip()

    async def one_pass_codegen(self, plan: str, context: dict) -> str:
        prompt = f"""
Generate complete, runnable Python code for the following machine learning plan.
Only output code (no backticks, no explanation).

Plan:
{plan}
"""
        response = await self.agent.async_call_llm(prompt, context=context)
        return self._strip_markdown(response)

    async def stepwise_codegen(self, plan: str, context: dict) -> str:
        # Future: break into steps; for now use same path
        return await self.one_pass_codegen(plan, context=context)


# ------------------------------ Tree Search -------------------------------- #


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
        emit_cb: Optional[
            Callable[[str, Dict[str, Any]], Awaitable[None]]
        ] = None,
        progress_every: int = 5,  # emit progress every N iterations
        heartbeat_secs: float = 20.0,  # or every N seconds (whichever first)
        report_top_k: int = 5,  # include top-K nodes in final rep
    ):
        self.agent = agent
        self.tree: List[SolutionNode] = []
        self.nodes_by_id: Dict[str, SolutionNode] = {}
        self.parent_map: Dict[str, Optional[str]] = {}
        self.children_map: Dict[str, List[str]] = {}

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

        self.plan_generator = PlanGenerator(agent)
        self.code_executor = CodeExecutor(agent)
        self.verifier = OutputVerifier()

        self.iteration = 0
        self.start_time = time.time()
        self.best_node: Optional[SolutionNode] = None
        self.best_metric: float = -float("inf")
        self.last_improve_iter: int = 0

        # MCTS stats
        self.visits: Dict[str, int] = {}  # node_id -> N
        self.value: Dict[str, float] = {}  # node_id -> mean value

        # reward shaping
        self.metric_fn = metric_fn or (
            lambda m: -1.0 if m is None else float(m)
        )

        # counters
        self.count_actions = {"draft": 0, "improve": 0, "debug": 0}
        self.count_buggy = 0
        self.count_nodes = 0

        if random_seed is not None:
            random.seed(random_seed)

    # ----------------------------- Orchestration ---------------------------- #

    async def run(self, context: dict) -> dict:
        await self._emit("progress", self._progress_payload(last_node=None))
        goal = context.get("goal", {})
        task_description = goal.get("goal_text", "").strip()
        if not task_description:
            raise ValueError("Context must contain 'goal.goal_text'")

        await self._emit("start", {"goal": task_description})

        # 1) Initial drafts in parallel
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
            self._maybe_emit_progress(last_node=node)

        # 2) Main loop (MCTS‑lite)
        while not self._should_stop():
            parent = self._select_parent_ucb()
            action, target = self._choose_action(parent)

            if action == "draft" or target is None:
                plan = await self.plan_generator.draft_plan(
                    task_description, context
                )
                node = await self._process_plan(
                    plan, None, "draft", task_description, context
                )

            elif action == "improve":
                feedback = target.summary or "No specific feedback."
                plan = await self.plan_generator.improve_plan(
                    target.plan, feedback, context
                )
                node = await self._process_plan(
                    plan, target, "improve", task_description, context
                )

            elif action == "debug":
                error_log = target.output or target.summary or "Unknown error."
                plan = await self.plan_generator.debug_plan(
                    target.plan, error_log, context
                )
                node = await self._process_plan(
                    plan, target, "debug", task_description, context
                )

            else:
                raise ValueError(f"Unknown action: {action}")

            self._add_node(node)
            self._backprop(node)
            self._update_best(node)
            self.iteration += 1

            self.count_actions[action] = self.count_actions.get(action, 0) + 1
            self._maybe_emit_progress(last_node=node)

        # 3) Final result
        best = self.best_node.to_dict() if self.best_node else None

        await self._emit(
            "progress",
            self._progress_payload(last_node=None, force_complete=True),
        )
        report = self._make_report(best_node=self.best_node)
        await self._emit("report", report)
        context["search_tree_size"] = len(self.tree)
        context["final_solution"] = best
        context["search_report"] = report  # << store for later consumers
        return context

    # ------------------------------- Core steps ----------------------------- #

    async def _process_plan(
        self,
        plan: str,
        parent_node: Optional[SolutionNode],
        node_type: str,
        task_description: str,
        context: dict,
    ) -> SolutionNode:
        # codegen
        complexity = await self.code_executor.score_complexity(
            task_description, plan, context
        )
        code = (
            await self.code_executor.stepwise_codegen(plan, context=context)
            if complexity > 3.5
            else await self.code_executor.one_pass_codegen(plan, context=context)
        )

        # execute (⚠️ sandbox in production)
        result = await self.execute_code(code)

        # verify
        v = self.verifier.verify(
            result.stdout, result.stderr, result.has_submission_file
        )

        parent = parent_node
        sibling_idx = self._next_child_index(parent.id if parent else None)
        node_depth = 0 if parent is None else (parent.depth + 1)
        node_path = self._make_path(parent, sibling_idx)

        # origin record
        origin = {
            "action": node_type,                      # 'draft' | 'improve' | 'debug'
            "source_id": parent.id if parent else None,
            "reason": (parent.summary or "") if parent else "initial draft",
        }

        # integrity hashes
        plan_h = self._hash_or_none(plan)
        code_h = self._hash_or_none(code)
        out_h  = self._hash_or_none(v["merged_output"])

        # tree_id from context (use your run_id)
        tree_id = context.get("pipeline_run_id")

        root_id = (parent.root_id if parent else None)
        if root_id is None:
            # first node of the tree becomes the root
            root_id = None  # temporarily None; we’ll set after instantiation if needed

        node = SolutionNode(
            tree_id=tree_id,
            id = f"{tree_id}.{node_depth}.{sibling_idx}",
            plan=plan,
            code=code,
            metric=v["metric"],
            output=v["merged_output"],
            summary=v["summary"],
            parent_id=parent.id if parent else None,
            is_buggy=v["is_bug"],
            node_type=node_type,

            # provenance
            root_id=root_id,           # may be None for first node; fixed in _add_node
            depth=node_depth,
            sibling_index=sibling_idx,
            path=node_path,
            origin=origin,
            lineage=((parent.lineage + [parent.id]) if parent else []),

            # content proofs
            plan_sha256=plan_h,
            code_sha256=code_h,
            output_sha256=out_h,
        )


        await self._emit("node", {
            "id": node.id,
            "parent_id": node.parent_id,
            "type": node.node_type,
            "metric": node.metric,
            "bug": node.is_buggy,
            "visits": self.visits.get(node.id, 0),
            "value": self.value.get(node.id, 0.0),

            "prompt_text": node.plan,
            "prompt_preview": (node.plan[:160] if node.plan else ""),

            "tree_id": node.tree_id,
            "root_id": node.root_id,
            "node_id": node.id,
            "depth": node.depth,
            "sibling_index": node.sibling_index,
            "path": node.path,
            "origin": node.origin,
            "lineage": node.lineage,

            # content proofs (optional to emit every time)
            "plan_sha256": node.plan_sha256,
            "code_sha256": node.code_sha256,
            "output_sha256": node.output_sha256,
        })
        return node

    # ------------------------------- Policy -------------------------------- #

    def _select_parent_ucb(self) -> Optional[SolutionNode]:
        """Choose a parent node to expand using UCB1 over non‑buggy nodes with any metric signal.
        Falls back to a random valid node or None (to draft anew)."""
        candidates = [n for n in self.tree if not n.is_buggy]
        if not candidates:
            return None
        total_N = sum(self.visits.get(n.id, 1) for n in candidates)

        def ucb(n: SolutionNode) -> float:
            N = self.visits.get(n.id, 1)
            Q = self.value.get(n.id, 0.0)
            return Q + self.C_ucb * ((total_N**0.5) / (1 + N))

        # prefer best or random early on
        if self.iteration < self.N_init * 2 and self.best_node:
            return self.best_node
        return max(candidates, key=ucb)

    def _choose_action(
        self, parent: Optional[SolutionNode]
    ) -> Tuple[str, Optional[SolutionNode]]:
        # No parent → draft a fresh plan
        if parent is None:
            return "draft", None

        # Occasionally exploit the current best (if we have one)
        if self.best_node and random.random() < self.H_greedy:
            return "improve", self.best_node

        # If the selected parent looks buggy, sometimes try a debug plan
        if parent.is_buggy and random.random() < self.H_debug:
            return "debug", parent

        # Default: improve the selected parent
        return "improve", parent

    # ------------------------------- Bookkeeping ---------------------------- #

    def _add_node(self, node: SolutionNode) -> None:
        # assign root_id for the very first node
        if node.parent_id is None and not any(n.parent_id is None for n in self.tree):
            node.root_id = node.id
            node.path = "0"         # ensure root path is "0"
            node.depth = 0
            node.sibling_index = 0
            # track a synthetic bucket for root children counts if you ever add more roots
            self.children_map.setdefault(None, []).append(node.id)

        self.tree.append(node)
        self.nodes_by_id[node.id] = node
        self.parent_map[node.id] = node.parent_id

        # link child into parent & children_map
        if node.parent_id:
            self.children_map.setdefault(node.parent_id, []).append(node.id)
            p = self.nodes_by_id.get(node.parent_id)
            if p:
                p.children.append(node.id)
                # ensure root_id cascades
                node.root_id = p.root_id or p.id
        else:
            # root node already placed; nothing more to do
            pass

        self.visits.setdefault(node.id, 0)
        self.value.setdefault(node.id, 0.0)
        self.count_nodes += 1
        if node.is_buggy:
            self.count_buggy += 1

    def _backprop(self, leaf: SolutionNode) -> None:
        reward = self.metric_fn(leaf.metric)
        node_id = leaf.id
        while node_id is not None:
            self.visits[node_id] = self.visits.get(node_id, 0) + 1
            # incremental mean update
            v_prev = self.value.get(node_id, 0.0)
            n = self.visits[node_id]
            self.value[node_id] = v_prev + (reward - v_prev) / n
            node_id = self.parent_map.get(node_id)

    def _update_best(self, node: SolutionNode) -> None:
        m = self.metric_fn(node.metric)
        if m > self.best_metric:
            self.best_metric = m
            self.best_node = node
            self.last_improve_iter = self.iteration

    # ------------------------------ Stopping -------------------------------- #

    def _should_stop(self) -> bool:
        if self.iteration >= self.max_iterations:
            return True
        if (time.time() - self.start_time) > self.time_limit:
            return True
        if (
            self.iteration - self.last_improve_iter
        ) >= self.no_improve_patience and self.iteration > 0:
            return True
        return False

    # ------------------------------- Exec ----------------------------------- #

    async def execute_code(self, code: str) -> ExecutionResult:
        """
        ⚠ SECURITY: This still runs untrusted code; use Docker/gVisor/Firejail in prod.
        We isolate to a fresh temp dir, use `-I` (isolated mode), and clean up.
        """
        tmpdir = tempfile.mkdtemp(prefix="ats_run_")
        py_path = os.path.join(tmpdir, "main.py")
        try:
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(code)
            # run
            import subprocess

            result = subprocess.run(
                [sys.executable, "-I", py_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            # detect outputs
            has_csv = any(
                fn.lower().endswith(".csv") for fn in os.listdir(tmpdir)
            )
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                has_submission_file=has_csv,
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                returncode=1,
                has_submission_file=False,
            )
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    # ------------------------------- Events -------------------------------- #

    async def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if self.emit_cb is None:
            return
        try:
            await self.emit_cb(event, payload)
        except Exception:
            # never let telemetry kill the run
            pass

    # ------------------------------ Progress -------------------------------- #

    def _progress_payload(
        self, last_node: Optional[SolutionNode], force_complete: bool = False
    ) -> Dict[str, Any]:
        """Compute a time/iteration-based progress snapshot."""
        elapsed = time.time() - self.start_time
        frac_iter = self.iteration / max(1, self.max_iterations)
        frac_time = elapsed / max(1e-6, self.time_limit)
        progress = min(1.0, max(frac_iter, frac_time))
        if force_complete:
            progress = 1.0

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
            "best_metric": self.best_metric
            if self.best_metric != -float("inf")
            else None,
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
        """Emit progress every N iterations OR every heartbeat seconds."""
        now = time.time()
        due_by_iter = (self.iteration % self.progress_every) == 0
        due_by_time = (now - self._last_progress_ts) >= self.heartbeat_secs
        if not (due_by_iter or due_by_time):
            return
        self._last_progress_ts = now
        # fire and forget (we're inside async context callers)
        asyncio.create_task(
            self._emit("progress", self._progress_payload(last_node))
        )

    # ------------------------------- Report --------------------------------- #

    def _make_report(
        self, best_node: Optional[SolutionNode]
    ) -> Dict[str, Any]:
        """Produce a compact end-of-run report, including a leaderboard."""
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
                "best_metric": self.best_metric
                if self.best_metric != -float("inf")
                else None,
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
        if not s:
            return None
        return hash_text(s)

    def _next_child_index(self, parent_id: Optional[str]) -> int:
        if not parent_id:
            # root child index is 0 for the first root; we keep a simple counter
            # but since we only have a single root series, use current root count
            return len(self.children_map.get(None, []))  # will be 0 for first root
        return len(self.children_map.get(parent_id, []))

    def _make_path(self, parent_node: Optional[SolutionNode], sibling_index: int) -> str:
        if parent_node is None:
            # root nodes live under "0", "1", ... if you ever allow multiple roots
            return str(sibling_index)
        return f"{parent_node.path}.{sibling_index}"
