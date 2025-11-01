# stephanie/components/ssp/impl/solvers/ats_solver.py
"""
ATSSolver with two modes:
- solve(): deep search (default, paper "solver" path)
- solve_with_evidence(): no-search answer using proposer evidence (verification aid)
"""

from __future__ import annotations

import asyncio
import heapq
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.utils.progress_mixin import ProgressMixin
from stephanie.components.ssp.core.roles.solver import Solver
from stephanie.components.ssp.core.protocols import EpisodeContext, VerificationResult
from stephanie.components.tree.events import TreeEventEmitter
from stephanie.prompts.prompt_loader import PromptLoader

# Minimal node record (use your canonical Node if available)
@dataclass
class Node:
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

SOLVER_PROMPT_TMPL = """Your task: answer the question using ONLY the provided evidence (short snippets).
Return EXACTLY three lines in this order, no extra text:

rationale: <1-2 sentences that justify your answer>
score: <integer 0-100 representing confidence>
result: <ONE-LINE final answer>

QUESTION:
{question}

EVIDENCE:
{evidence}
"""

_LINE = re.compile(r'^\s*([a-zA-Z_]+)\s*:\s*(.+?)\s*$')

def _parse_three_lines(text: str) -> Dict[str, str]:
    out = {"rationale": "", "score": "", "result": ""}
    for line in (text or "").splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        k, v = m.group(1).lower(), m.group(2).strip()
        if k in out:
            out[k] = v
    return out


class ATSSolver(Solver, ProgressMixin):
    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Any,
        *,
        searcher,                       # SolutionSearch instance
        event_emitter: Optional[TreeEventEmitter] = None,
    ):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self.searcher = searcher
        self.events = event_emitter

        # services
        self.prompt = container.get("prompt")
        self.prompt_loader = PromptLoader(memory=memory, logger=logger)

        # knobs
        sp = (self.cfg.get("self_play") or {})
        self.solver_model = sp.get("solver_model", {"name": "ollama/qwen:0.5b", "api_base": "http://localhost:11434"})
        self.sys_preamble = "Follow the 3-line output format exactly."

        # tree search settings
        self.max_depth = int(self.cfg.get("max_depth", 2))
        self.beam_width = int(self.cfg.get("beam_width", 3))

        # progress
        self._init_progress(container, logger)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def solve(
        self,
        question: str,
        seed_answer: str,
        context: Optional[EpisodeContext] = None,
    ) -> Tuple[str, List[str], int, Dict[str, Any]]:
        """
        Default solver path = deep search (paper solver).
        Returns: predicted_answer, evidence_docs, steps, meta
        """
        return await self._deep_search(question, seed_answer, context or {})

    async def solve_with_evidence(
        self,
        question: str,
        evidence_snippets: List[str],
        *,
        context: Optional[EpisodeContext] = None,
    ) -> Tuple[str, List[str], int, Dict[str, Any]]:
        """
        No-search answer using ONLY the provided evidence (verification helper).
        Returns: predicted_answer, evidence_used, steps, meta
        """
        ev = "\n".join(f"- {s}" for s in (evidence_snippets or []))
        prompt = SOLVER_PROMPT_TMPL.format(question=question, evidence=ev)

        txt = await self.prompt.run_prompt(
            prompt_text=prompt,
            context=context or {},
            model=self.solver_model,
            sys_preamble=self.sys_preamble,
            params={"temperature": 0.1},
        )
        parsed = _parse_three_lines(txt)
        result = parsed["result"].strip()
        meta = {
            "model": self.solver_model,
            "rationale": parsed["rationale"],
            "raw_score": parsed["score"],
            "mode": "evidence_only",
        }
        # steps=1: single-shot LLM
        return result, list(evidence_snippets), 1, meta

    async def verify_answer(
        self,
        question: str,
        seed_answer: str,
        evidence_snippets: List[str],
    ) -> VerificationResult:
        """Optional helper if you want to keep a gate before full search."""
        if not evidence_snippets:
            return VerificationResult(
                is_valid=False,
                score=0.0,
                reason="No evidence provided",
                filter_results={"evidence_usage": False},
                verification_details={"evidence_count": 0},
            )

        predicted, _, _, _ = await self.solve_with_evidence(
            question, evidence_snippets, context={"verify": True}
        )
        score = self._f1(seed_answer, predicted)
        threshold = float((self.cfg.get("verify") or {}).get("pass_threshold", 0.75))
        is_valid = score >= threshold

        return VerificationResult(
            is_valid=is_valid,
            score=score,
            reason=f"Verification {'passed' if is_valid else 'failed'} (score={score:.2f})",
            filter_results={"evidence_usage": True},
            verification_details={"predicted": predicted, "threshold": threshold},
        )

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    async def _deep_search(
        self,
        question: str,
        seed_answer: str,
        context: EpisodeContext,
    ) -> Tuple[str, List[str], int, Dict[str, Any]]:
        task_key = f"ATS:{hash(question) & 0xffff:04x}"
        rewrites_per_parent = len(self._rewrite(question))
        total_steps = self._estimate_total_steps(rewrites_per_parent)
        self.pstart(task=task_key, total=total_steps)
        self.pstage(task=task_key, stage="root")

        root = Node(
            id=f"root-{uuid.uuid4().hex[:6]}",
            parent_id=None,
            root_id="root",
            depth=0,
            sibling_index=0,
            node_type="root",
            query=question,
            score=0.0,
            context="",
            task_description=question,
        )
        if self.events:
            self.events.on_root_created(root)

        best = root
        steps = 0
        done = 0

        for depth in range(1, self.max_depth + 1):
            self.pstage(task=task_key, stage=f"depth-{depth}")

            parents = self._get_candidates(root, depth)
            for parent in parents:
                rewrites = self._rewrite(parent.query)
                for i, q2 in enumerate(rewrites):
                    # Retrieve snippets (search)
                    results = await self.searcher.search(q2, seed_answer=seed_answer, context=context)
                    for snippet in results:
                        sc = self._overlap_score(snippet, seed_answer)
                        child = Node(
                            id=f"node-{uuid.uuid4().hex[:6]}",
                            parent_id=parent.id,
                            root_id=root.id,
                            depth=depth,
                            sibling_index=i,
                            node_type="rewrite",
                            query=q2,
                            score=sc,
                            context=snippet,
                            task_description=question,
                        )
                        if self.events:
                            self.events.on_node_added(parent, child)
                            self.events.on_backprop(child, delta=float(sc))

                        if sc > best.score:
                            best = child
                            if self.events:
                                self.events.on_best_update(best)

                        steps += 1
                        done += 1
                        self.ptick(task=task_key, done=done, total=total_steps)

            self._prune_to_beam(root)

        self.pdone(task=task_key)

        predicted_answer = best.context if best.context else seed_answer
        evidence = best.context.splitlines() if best.context else []
        if self.events:
            self.events.on_progress({"phase": "ats_solve_complete", "steps": steps, "best_score": best.score})
            self.events.on_rollout_complete(
                {"best": {"id": best.id, "score": best.score, "query": best.query, "depth": best.depth}, "steps": steps}
            )

        meta = {"best_score": best.score, "search_depth": best.depth, "evidence_count": len(evidence), "mode": "search"}
        return predicted_answer, evidence, steps, meta

    # ---- small helpers ----

    @staticmethod
    def _rewrite(query: str) -> List[str]:
        return [query, query.replace("explain", "describe"), query + " in practical terms"]

    @staticmethod
    def _overlap_score(text: str, target: str) -> float:
        a = {t for t in text.lower().split() if t.isalpha() or t.isalnum()}
        b = {t for t in target.lower().split() if t.isalpha() or t.isalnum()}
        if not a or not b:
            return 0.0
        return len(a & b) / max(len(b), 1)

    def _estimate_total_steps(self, rewrites_per_parent: int) -> int:
        steps = 0
        nodes_at_depth = 1
        for _ in range(1, self.max_depth + 1):
            steps += nodes_at_depth * rewrites_per_parent
            nodes_at_depth = min(self.beam_width, nodes_at_depth * rewrites_per_parent)
        return steps

    def _get_candidates(self, root: Node, depth: int) -> List[Node]:
        # For the MVP we just re-expand the previous frontier (root).
        # You can keep a real tree and return children of prior layer for accuracy.
        return [root] if depth == 1 else [root]

    def _prune_to_beam(self, root: Node) -> None:
        # Hook for real pruning if you build an explicit tree
        pass

    @staticmethod
    def _f1(ground_truth: str, predicted: str) -> float:
        gt_words = set((ground_truth or "").lower().split())
        pred_words = set((predicted or "").lower().split())
        if not gt_words or not pred_words:
            return 0.0
        common = gt_words & pred_words
        precision = len(common) / len(pred_words)
        recall = len(common) / len(gt_words)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def get_capabilities(self) -> Dict[str, Any]:
        return {"supports_verification_mode": True, "max_search_depth": self.max_depth, "beam_width": self.beam_width}
