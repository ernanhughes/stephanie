# stephanie/components/ssp/impl/solvers/ats_solver.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.tree.events import TreeEventEmitter
from stephanie.components.ssp.impl.solvers.solution_search import SolutionSearch
from stephanie.utils.progress_mixin import ProgressMixin

@dataclass
class Node:
    # Minimal structure compatible with TreeEventEmitter._node_rec()
    id: str
    parent_id: Optional[str]
    root_id: str
    depth: int
    sibling_index: int
    node_type: str  # 'root' | 'rewrite'
    query: str
    score: float
    context: str
    task_description: Optional[str] = None
    summary: Optional[str] = None
    metric: Optional[float] = None
    is_buggy: Optional[bool] = None

class ATSSolver(ProgressMixin):
    """
    Agentic Tree Search (ultra-small) with history-first events + progress reporting.

    - State = text query, Context = joined docs
    - Expand = simple query rewrite (deterministic), evaluate by overlap score
    - Emits: root_created, expand, node_added, best_update, backprop (score), progress, rollout_complete
    - Progress: pstart/pstage/ptick/pdone via ProgressService (if registered in container)
    """

    def __init__(
        self,
        searcher: SolutionSearch,
        max_depth: int = 2,
        beam_width: int = 3,
        *,
        event_emitter: Optional[TreeEventEmitter] = None,
        topic: str = "ssp.ats",
        container: Any = None,
        logger: Any = None,
    ):
        self.searcher = searcher
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.events = event_emitter or TreeEventEmitter(topic=topic)

        # Progress init (will no-op if no ProgressService in container)
        # Prefer an explicit container if passed; else try to borrow from searcher.
        container = container or getattr(searcher, "container", None)
        self._init_progress(container, logger)

    # ----------------------- static helpers -----------------------

    @staticmethod
    def _rewrite(query: str) -> List[str]:
        # Minimal, deterministic rewrites
        return [
            query,
            query.replace("explain", "describe"),
            query + " in practical terms",
        ]

    @staticmethod
    def _overlap_score(text: str, target: str) -> float:
        a = set([t for t in text.lower().split() if t.isalpha() or t.isalnum()])
        b = set([t for t in target.lower().split() if t.isalpha() or t.isalnum()])
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / max(len(b), 1)

    # -------------------------- main -----------------------------

    async def solve(
        self, question: str, seed_answer: str, context: Dict[str, Any]
    ) -> Tuple[str, List[str], int]:

        # ---------- progress: start ----------
        task_key = f"ATS:{hash(question) & 0xffff:04x}"
        rewrites_per_parent = len(self._rewrite(question))
        total_steps = self._estimate_total_steps(rewrites_per_parent)
        self.pstart(task=task_key, total=total_steps)
        self.pstage(task=task_key, stage="root")

        # Root setup
        root_docs = await self.searcher.search(
            question, seed_answer=seed_answer, context=context
        )
        root_ctx = "\n".join(root_docs)
        root_score = self._overlap_score(root_ctx, seed_answer)
        root = Node(
            id="0",
            parent_id=None,
            root_id="0",
            depth=0,
            sibling_index=0,
            node_type="root",
            query=question,
            score=root_score,
            context=root_ctx,
            task_description=question,
            metric=root_score,
        )
        self.events.on_root_created(root)

        frontier = [root]
        best = root
        steps = 0
        done = 0  # progress ticks

        # Main loop
        for depth in range(1, self.max_depth + 1):
            self.pstage(task=task_key, stage=f"depth_{depth}_expand")
            candidates: List[Node] = []

            for parent in frontier:
                self.events.on_expand(parent)
                rewrites = self._rewrite(parent.query)

                for i, q2 in enumerate(rewrites):
                    ctx_docs = await self.searcher.search(
                        q2, seed_answer=seed_answer, context=context
                    )
                    ctx = "\n".join(ctx_docs)
                    sc = self._overlap_score(ctx, seed_answer)
                    child = Node(
                        id=f"{parent.id}.{i}",
                        parent_id=parent.id,
                        root_id=root.id,
                        depth=depth,
                        sibling_index=i,
                        node_type="rewrite",
                        query=q2,
                        score=sc,
                        context=ctx,
                        task_description=question,
                        metric=sc,
                    )
                    self.events.on_node_added(parent, child)
                    self.events.on_backprop(child, delta=float(sc))
                    candidates.append(child)

                    # progress tick after each child
                    steps += 1
                    done += 1
                    self.ptick(task=task_key, done=min(done, total_steps), total=total_steps)

            # Beam select
            self.pstage(task=task_key, stage=f"depth_{depth}_beam_select")
            candidates.sort(key=lambda n: n.score, reverse=True)
            frontier = candidates[: self.beam_width]

            # Track best
            if frontier and frontier[0].score > best.score:
                best = frontier[0]
                self.events.on_best_update(best)

        # MVP answer extraction (echo hint)
        predicted_answer = seed_answer
        evidence = best.context.splitlines()

        self.events.on_progress(
            {"phase": "ats_solve_complete", "steps": steps, "best_score": best.score}
        )
        self.events.on_rollout_complete(
            {
                "best": {
                    "id": best.id,
                    "score": best.score,
                    "query": best.query,
                    "depth": best.depth,
                },
                "steps": steps,
            }
        )

        # ---------- progress: done ----------
        self.pstage(task=task_key, stage="complete")
        self.pdone(task=task_key)

        return predicted_answer, evidence, steps

    # ------------------------ internals --------------------------

    def _estimate_total_steps(self, rewrites_per_parent: int) -> int:
        """
        Conservative upper bound on child expansions (for progress total):
        depth 1..D: sum( rewrites_per_parent * beam_width^(d-1) )
        """
        total = 0
        for d in range(1, self.max_depth + 1):
            total += rewrites_per_parent * (self.beam_width ** (d - 1))
        # Never return zero to avoid divide-by-zero in any progress UIs
        return max(1, total)
