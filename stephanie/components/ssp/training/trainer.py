# stephanie/components/ssp/trainer.py
from __future__ import annotations
import time
import uuid
from typing import Any, Dict, Iterable

from stephanie.components.ssp.utils.trace import EpisodeTrace
from stephanie.components.ssp.impl.proposers.searching_proposer import SearchingProposer
from stephanie.components.ssp.impl.solvers.ats_solver import ATSSolver
from stephanie.components.ssp.impl.solvers.solution_search import SolutionSearch
from stephanie.components.ssp.impl.verifiers.f1_verifier import Verifier
from stephanie.components.tree.events import TreeEventEmitter

import logging
_logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, memory, container, logger):
        self.cfg =cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.difficulty = cfg.get("difficulty", 0.3)
        self.verify = Verifier(cfg=self.cfg, memory=self.memory, container=self.container, logger=self.logger)

    async def run_episode(self, seed_answer: str, context: Dict[str, Any]) -> EpisodeTrace:
        episode_id = f"ssp-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
        emitter = TreeEventEmitter(topic="ssp.ats")
        solution_search = SolutionSearch(cfg=self.cfg, memory=self.memory, container=self.container, logger=self.logger, event_emitter=emitter)
        proposer = SearchingProposer(self.cfg, memory=self.memory, container=self.container, logger=self.logger, solution_search=solution_search)
        question, proposer_evidence, solver_meta = await proposer.propose(seed_answer, context=context)
        if not question:
            _logger.warning(f"⚠️  No question proposed for seed answer: {seed_answer}")
            question = f"What is the question for this answer {seed_answer}?"

        # ATS‑based solver with a tiny local searcher that contains the answer in synthetic docs
        solver = ATSSolver(cfg=self.cfg, memory=self.memory, container=self.container, logger=self.logger,
            searcher=solution_search, event_emitter=emitter)
        predicted, evidence, solver_steps, solver_meta = await solver.solve(question, seed_answer=seed_answer, context=context, evidence_snippets=proposer_evidence)
        _logger.info(f"🧠 Predicted answer: {predicted} | using {len(evidence)} evidence docs.")
        ok, verifier_score = self.verify.verify(ground_truth=seed_answer, predicted=predicted)

        ep = EpisodeTrace(
            episode_id=episode_id,
            seed_answer=seed_answer,
            question=question,
            predicted_answer=predicted,
            verified=ok,
            proposer_evidence=proposer_evidence,
            verifier_score=verifier_score,
            difficulty=self.difficulty,
            solver_steps=solver_steps,
            evidence_docs=evidence,
            solver_meta=solver_meta,
        )
        vpm_control = self.container.get("vpm_control")
        vmeta = await vpm_control.generate_for_episode(ep)
        ep.solver_meta["vpm_png"] = vmeta["file"]
        ep.solver_meta["vpm_features"] = vmeta["features"]
        return ep

    async def run_batch(self, seeds: Iterable[str], context: Dict[str, Any]) -> Dict[str, float]:
        n, r_sum, ok_sum = 0, 0.0, 0
        for s in seeds:
            ep = await self.run_episode(s, context)
            n += 1
            r_sum += ep.reward
            ok_sum += int(ep.verified)
            _logger.info(
                f"EP {ep.episode_id} | ok={ep.verified} r={ep.reward:.3f} | Q: {ep.question}\n"
                f"→ A*: {ep.predicted_answer}\n"
            )
        return {
            "episodes": n,
            "avg_reward": (r_sum / n) if n else 0.0,
            "success_rate": (ok_sum / n) if n else 0.0,
        }
