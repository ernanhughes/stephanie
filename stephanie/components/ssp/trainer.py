from __future__ import annotations
import time
import uuid
from typing import Dict, Iterable

from stephanie.components.ssp.trace import EpisodeTrace
from stephanie.components.ssp.proposer import Proposer
from stephanie.components.ssp.ats_solver import ATSSolver, DummySearch
from stephanie.components.ssp.verifier import Verifier


class Trainer:
    def __init__(self, difficulty: float = 0.3, verify_threshold: float = 0.6):
        self.difficulty = difficulty
        self.verify = Verifier(threshold=verify_threshold)

    def run_episode(self, seed_answer: str) -> EpisodeTrace:
        episode_id = f"ssp-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
        proposer = Proposer()
        question = proposer.propose(seed_answer)

        # ATS‑based solver with a tiny local searcher that contains the answer in synthetic docs
        solver = ATSSolver(
            searcher=DummySearch(seed_answer), max_depth=2, beam_width=3
        )
        predicted, evidence, steps = solver.solve(
            question, target_answer_hint=seed_answer
        )

        ok, reward = self.verify.verify(
            ground_truth=seed_answer, predicted=predicted
        )

        return EpisodeTrace(
            episode_id=episode_id,
            seed_answer=seed_answer,
            question=question,
            predicted_answer=predicted,
            verified=ok,
            reward=reward,
            difficulty=self.difficulty,
            solver_steps=steps,
            evidence_docs=evidence,
            meta={},
        )

    def run_batch(self, seeds: Iterable[str]) -> Dict[str, float]:
        n, r_sum, ok_sum = 0, 0.0, 0
        for s in seeds:
            ep = self.run_episode(s)
            n += 1
            r_sum += ep.reward
            ok_sum += int(ep.verified)
            print(
                f"EP {ep.episode_id} | ok={ep.verified} r={ep.reward:.3f} | Q: {ep.question}\n"
                f"→ A*: {ep.predicted_answer}\n"
            )
        return {
            "episodes": n,
            "avg_reward": (r_sum / n) if n else 0.0,
            "success_rate": (ok_sum / n) if n else 0.0,
        }
