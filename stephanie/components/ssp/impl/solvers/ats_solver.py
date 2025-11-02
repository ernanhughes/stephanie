# stephanie/components/ssp/impl/solvers/ats_solver.py
"""
ATSSolver with two modes:
- solve(..., use_search=True): deep search (paper solver path) + VPM snapshots
- solve_with_evidence(): no-search answer using proposer evidence (verification aid)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from stephanie.components.ssp.core.protocols import (EpisodeContext,
                                                     VerificationResult)
from stephanie.components.ssp.core.roles.solver import Solver
from stephanie.components.ssp.reward.heads.naive_quarkish import \
    NaiveQuarkishReward
from stephanie.components.tree.events import TreeEventEmitter
from stephanie.prompts.prompt_loader import PromptLoader
from stephanie.utils.progress_mixin import ProgressMixin


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

_LINE = re.compile(r"^\s*([a-zA-Z_]+)\s*:\s*(.+?)\s*$", re.IGNORECASE)


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
        searcher,  # SolutionSearch instance
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
        self.vpm = container.get("ssp_vpm_viz")  # for progress snapshots

        # knobs
        sp = self.cfg.get("self_play") or {}
        self.solver_model = sp.get(
            "solver_model",
            {"name": "ollama/qwen:0.5b", "api_base": "http://localhost:11434"},
        )
        self.sys_preamble = "Follow the 3-line output format exactly."

        sv = self.cfg.get("solver") or {}
        self.max_depth = int(sv.get("max_depth", self.cfg.get("max_depth", 2)))
        self.beam_width = int(
            sv.get("beam_width", self.cfg.get("beam_width", 3))
        )
        self.progress_every = int(sv.get("progress_every", 1))
        self.snapshot_early = bool(sv.get("snapshot_early", True))
        self.vpm = container.get("ssp_vpm_viz")

        self.verify_threshold = float(cfg.get("verify_threshold", 0.75))
        self.reward_head = NaiveQuarkishReward()
        self._init_progress(container, logger)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def solve(
        self,
        question: str,
        seed_answer: str,
        *,
        context: Optional[EpisodeContext] = None,
        use_search: bool = True,
        evidence_docs: Optional[List[str]] = None,
    ) -> Tuple[str, List[str], int, Dict[str, Any]]:
        """
        Main entry:
          - If use_search=True: run deep search (paper solver path).
          - Else: answer using ONLY `evidence_docs` (verification helper).
        Returns: predicted_answer, evidence_docs, steps, meta
        """
        if use_search:
            return await self._deep_search(
                question, seed_answer, context or {}
            )
        return await self.solve_with_evidence(
            question=question,
            evidence_docs=evidence_docs or [],
            context=context,
        )

    async def solve_with_evidence(
        self,
        question: str,
        evidence_docs: List[str],
        *,
        context: Optional[EpisodeContext] = None,
    ) -> Tuple[str, List[str], int, Dict[str, Any]]:
        """
        No-search answer using ONLY the provided evidence (verification helper).
        Returns: predicted_answer, evidence_used, steps, meta
        """
        ev = "\n".join(f"- {s}" for s in (evidence_docs or []))
        prompt = SOLVER_PROMPT_TMPL.format(question=question, evidence=ev)

        txt = await self.prompt.run_prompt(
            prompt_text=prompt,
            context=context or {},
            model=self.solver_model,
            sys_preamble=self.sys_preamble,
            params={"temperature": 0.1},
        )
        parsed = _parse_three_lines(txt)
        result = (parsed["result"] or "").strip()
        reward = self.reward_head.compute_reward(
            question=question,
            predicted_answer=result,
            seed_answer=(context or {}).get("seed_answer") or "",
            evidence_docs=list(evidence_docs),
            meta_in={"mode": "evidence_only"},
        )
        verified = bool(reward.get("reward",) >= self.verify_threshold)

        meta = {
            "model": self.solver_model,
            "rationale": parsed.get("rationale", ""),
            "raw_score": parsed.get("score", ""),
            "mode": "evidence_only",
            "reward": reward.get("reward", 0),
            "verified": verified,
            "f1": reward.get("f1", 0),
            "coverage": reward.get("coverage", 0),
            "len_reward": reward.get("len_reward", 0),
            "resp_len": reward.get("resp_len", 0),
        }
        # steps=1: single-shot LLM
        return result, list(evidence_docs), 1, meta

    async def verify_answer(
        self,
        question: str,
        seed_answer: str,
        evidence_docs: List[str],
    ) -> VerificationResult:
        """Optional helper if you want to keep a gate before full search."""
        if not evidence_docs:
            return VerificationResult(
                is_valid=False,
                score=0.0,
                reason="No evidence provided",
                filter_results={"evidence_usage": False},
                verification_details={"evidence_count": 0},
            )

        predicted, _, _, _ = await self.solve_with_evidence(
            question, evidence_docs, context={"verify": True}
        )
        score = self._f1(seed_answer, predicted)
        threshold = float(
            (self.cfg.get("verify") or {}).get("pass_threshold", 0.75)
        )
        is_valid = score >= threshold

        return VerificationResult(
            is_valid=is_valid,
            score=score,
            reason=f"Verification {'passed' if is_valid else 'failed'} (score={score:.2f})",
            filter_results={"evidence_usage": True},
            verification_details={
                "predicted": predicted,
                "threshold": threshold,
            },
        )

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def _dims_for_snapshot(
        self,
        *,
        question: str,
        best_score: float,
        steps: int,
        ev_count: int,
        difficulty: float,
        answer_text: Optional[str] = None,
    ) -> Dict[str, float]:
        # Preview-friendly dims in [0,1]
        q_len = min(128.0, float(len((question or "").split())))
        a_len = min(128.0, float(len((answer_text or "").split())))
        return {
            "reward": float(
                max(0.0, min(1.0, best_score))
            ),  # proxy until judged
            "verified": 0.0,
            "difficulty": float(max(0.0, min(1.0, difficulty))),
            "question_len": q_len / 128.0,
            "answer_len": a_len / 128.0,
            "evidence_count": min(1.0, ev_count / 8.0),
            "solver_steps": min(1.0, steps / 64.0),
        }

    async def _deep_search(
        self,
        question: str,
        seed_answer: str,
        context: EpisodeContext,
    ) -> Tuple[str, List[str], int, Dict[str, Any]]:
        task_key = f"ATS-{hash(question) & 0xFFFF:04x}"
        # stable + filename-safe unit (no colons/slashes)
        raw_unit = context.get("vpm_unit") or f"search-{task_key.lower()}"
        unit = re.sub(r'[^a-zA-Z0-9_.-]+', "_", raw_unit)
        difficulty = float((context or {}).get("difficulty", 0.0))

        rewrites_per_parent = len(self._rewrite(question))
        total_steps = self._estimate_total_steps(rewrites_per_parent)
        self.pstart(task=task_key, total=total_steps)
        self.pstage(task=task_key, stage="root")

        # EARLY snapshot (depth 0)
        if self.vpm and self.snapshot_early:
            dims0 = self._dims_for_snapshot(
                question=question,
                best_score=0.0,
                steps=0,
                ev_count=0,
                difficulty=difficulty,
                answer_text=None,
            )
            try:
                self.vpm.snapshot_progress(
                    unit=unit, dims=dims0, step_idx=0, tag="depth0"
                )
            except Exception:
                pass

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
        ev_count = 0
        last_ev_batch = 0
        for depth in range(1, self.max_depth + 1):
            self.pstage(task=task_key, stage=f"depth-{depth}")

            parents = self._get_candidates(root, depth)
            for parent in parents:
                rewrites = self._rewrite(parent.query)
                for i, q2 in enumerate(rewrites):
                    # Retrieve snippets (search)
                    try:
                        results = await self.searcher.search(
                            q2, seed_answer=seed_answer, context=context
                        )
                    except Exception:
                        results = []
                    last_ev_batch = len(results)
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

                        # --- VPM snapshot for this step ---
                        if self.vpm:
                            # use the stable, sanitized `unit` computed above
                            prev_best = float(best.score)
                            # quick helpers
                            def _n01(x, hi): return max(0.0, min(1.0, (x/hi) if hi else 0.0))
                            def _jac(a: str, b: str) -> float:
                                A, B = set(a.lower().split()), set(b.lower().split())
                                return 1.0 - (len(A & B) / max(len(A | B), 1))
                            dims = {
                                # canonical slots used by earlier code
                                "reward": prev_best,            # proxy until judge; “how good so far”
                                "verified": 0.0,                        # 0 during search
                                "difficulty": float((context or {}).get("difficulty", 0.3)),
                                "question_len": _n01(len(q2.split()), 128),
                                "answer_len":  _n01(len((snippet or '').split()), 128),
                                "evidence_count": _n01(last_ev_batch, 8),
                                "solver_steps": _n01(steps, self._estimate_total_steps(len(self._rewrite(question)))),
                                # extra thinking channels (new, will render now)
                                "score": sc,
                                "best_score": prev_best,
                                "improvement": max(0.0, sc - prev_best),
                                "depth": _n01(depth, self.max_depth),
                                "novelty": _jac(snippet or "", best.context or ""),
                            }
                            self.vpm.snapshot_progress(unit=unit, dims=dims, step_idx=steps, tag=f"depth{depth}")

                        if sc > best.score:
                            best = child
                            if self.events:
                                self.events.on_best_update(best)
                            # snapshot explicit improvements
                            if self.vpm:
                                dims["best_score"] = float(best.score)
                                dims["improvement"] = max(0.0, float(best.score) - float(prev_best))
                                self.vpm.snapshot_progress(unit=unit, dims=dims, step_idx=steps, tag=f"improved_d{depth}")                                
                                try:
                                    dims = self._dims_for_snapshot(
                                        question=question,
                                        best_score=best.score,
                                        steps=steps,
                                        ev_count=last_ev_batch,
                                        difficulty=difficulty,
                                        answer_text=best.context,
                                    )
                                    self.vpm.snapshot_progress(
                                        unit=unit,
                                        dims=dims,
                                        step_idx=steps,
                                        tag=f"improved_d{depth}",
                                    )
                                except Exception:
                                    pass

                        steps += 1
                        ev_count += 1
                        done += 1
                        self.ptick(task=task_key, done=done, total=total_steps)

                        # Snapshot cadence
                        if self.vpm and (
                            steps % max(1, self.progress_every) == 0
                        ):
                            try:
                                dims = self._dims_for_snapshot(
                                    question=question,
                                    best_score=best.score,
                                    steps=steps,
                                    ev_count=last_ev_batch,
                                    difficulty=difficulty,
                                    answer_text=best.context,
                                )
                                self.vpm.snapshot_progress(
                                    unit=unit,
                                    dims=dims,
                                    step_idx=steps,
                                    tag=f"depth{depth}",
                                )
                            except Exception:
                                pass

            self._prune_to_beam(root)

        self.pdone(task=task_key)

        predicted_answer = best.context if best.context else seed_answer
        evidence = best.context.splitlines() if best.context else []
        
        reward_results = self.reward_head.compute_reward(
            question=question,
            predicted_answer=predicted_answer,
            seed_answer=seed_answer,
            evidence_docs=evidence,
        )
        verified = bool(reward_results.get("reward",) >= self.verify_threshold)
        
        if self.events:
            self.events.on_progress(
                {
                    "phase": "ats_solve_complete",
                    "steps": steps,
                    "best_score": best.score,
                }
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

        meta = {
            "best_score": best.score,
            "search_depth": best.depth,
            "evidence_count": len(evidence),
            "mode": "search",
            "vpm_unit": unit,
            "reward": reward_results.get("reward", 0),
            "verified": verified,
            "f1": reward_results.get("f1", 0),
            "coverage": reward_results.get("coverage", 0),
            "len_reward": reward_results.get("len_reward", 0),
            "resp_len": reward_results.get("resp_len", 0),
        }

        def _n01(x, hi): return max(0.0, min(1.0, (x/hi) if hi else 0.0))
        def _jac(a: str, b: str) -> float:
            A, B = set((a or "").lower().split()), set((b or "").lower().split())
            return 1.0 - (len(A & B) / max(len(A | B), 1))
        
        reward_val = reward_results.get("reward", 0)
        if reward_val > best.score:
            best = child
            best.score = reward_val  # let “best_score” track the head’s reward
            if self.events:
                self.events.on_best_update(best)
        dims = {
            "reward": reward_val,      # ← primary channel now
            "verified": verified,                        # during search
            "difficulty": float((context or {}).get("difficulty", 0.3)),
            "score": sc,
            "f1": reward_results.get("f1", 0),
            "coverage": reward_results.get("coverage", 0),
            "len_reward": reward_results.get("len_reward", 0),
            "resp_len": reward_results.get("resp_len", 0),
            "question_len": _n01(len(q2.split()), 128),
            "answer_len":  _n01(len((snippet or '').split()), 128),
            "evidence_count": _n01(len(results), 8),
            "solver_steps": _n01(steps, self._estimate_total_steps(len(self._rewrite(question)))),
            "best_score": float(best.score),
            "improvement": max(0.0, reward_results.get("reward", 0) - float(best.score)),
            "depth": _n01(depth, self.max_depth),
            "novelty": _jac(snippet or "", best.context or ""),
        }
        # IMPORTANT: this is the new sink you added above
        self.vpm.record_step(unit=unit, dims=dims, step_idx=steps)

        self.vpm.generate_raw_vpm_image(unit=unit)
        self.vpm.generate_phos_image(unit=unit)
    
        self.vpm.finalize_progress(unit=unit, gif_name=f"{unit}_progress.gif", fps=2)
        return predicted_answer, evidence, steps, meta

    # ---- VPM snapshot helper ------------------------------------------------

    def _snapshot_vpm(
        self,
        *,
        unit: str,
        score: float,
        verified: bool,
        difficulty: float,
        question: str,
        answer_text: str,
        evidence_lines: int,
        steps: int,
        step_idx: int,
        tag: str,
    ) -> None:
        """
        Push a single progress frame to the VPM viz service.
        Maps ATS-local stats onto the common 7-dim vector used by EpisodeTrace.to_vpm_features().
        """
        try:
            if not self.vpm:
                return
            dims = {
                "reward": max(
                    0.0, min(1.0, float(score))
                ),  # reuse overlap as "score-ish"
                "verified": 1.0 if verified else 0.0,
                "difficulty": float(difficulty or 0.0),
                "question_len": min(
                    1.0, len((question or "").split()) / 128.0
                ),
                "answer_len": min(
                    1.0, len((answer_text or "").split()) / 128.0
                ),
                "evidence_count": min(1.0, float(evidence_lines) / 8.0),
                "solver_steps": min(1.0, float(steps) / 64.0),
            }
            self.vpm.snapshot_progress(
                unit=unit, dims=dims, step_idx=step_idx, tag=tag
            )
        except Exception as e:
            # keep failures non-fatal
            try:
                self.logger.warning(
                    "VPM snapshot failed",
                    extra={"error": str(e), "unit": unit, "tag": tag},
                )
            except Exception:
                pass

    # ---- small helpers ----

    @staticmethod
    def _rewrite(query: str) -> List[str]:
        return [
            query,
            query.replace("explain", "describe"),
            query + " in practical terms",
        ]

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
            nodes_at_depth = min(
                self.beam_width, nodes_at_depth * rewrites_per_parent
            )
        return steps

    def _get_candidates(self, root: Node, depth: int) -> List[Node]:
        # MVP: re-expand root at each depth; swap in a real frontier for full tree search.
        return [root]

    def _prune_to_beam(self, root: Node) -> None:
        # Hook for pruning when you maintain a full tree.
        pass

    @staticmethod
    def _f1(ground_truth: str, predicted: str) -> float:
        gt_words = set((ground_truth or "").lower().split())
        pred_words = set((predicted or "").lower().split())
        if not gt_words or not pred_words:
            return 0.0
        common = gt_words & pred_words
        precision = len(common) / len(pred_words) if pred_words else 0.0
        recall = len(common) / len(gt_words) if gt_words else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_verification_mode": True,
            "max_search_depth": self.max_depth,
            "beam_width": self.beam_width,
        }
