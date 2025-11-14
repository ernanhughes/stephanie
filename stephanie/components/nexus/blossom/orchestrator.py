# stephanie/components/nexus/blossom/orchestrator.py
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BlossomConfig:
    max_depth: int = 6
    beam_size: int = 8
    k_generate: int = 6
    novelty_tau: float = 0.08   # cosine distance or L2 in embed space
    merge_tau: float = 0.04
    success_threshold: float = 0.85
    budget_nodes: int = 512
    diversify_ops: bool = True  # mix DIVERGE/REFINE/RECALL

class BlossomOrchestrator:
    def __init__(self, db, generator, featurizer, evaluator, pruner, renderer, reporter, cfg: BlossomConfig):
        self.db, self.gen, self.feat, self.eval, self.prune, self.rndr, self.rpt = \
            db, generator, featurizer, evaluator, pruner, renderer, reporter
        self.cfg = cfg

    def run(self, run_id: str, seed_text: str, goal: Dict[str, Any]) -> Dict[str, Any]:
        root = self._init_seed(run_id, seed_text, goal)
        frontier = [root]
        archive = [root]
        for depth in range(1, self.cfg.max_depth + 1):
            # 1) expand
            cand = []
            for n in frontier:
                cand += self.gen.expand(n, k=self.cfg.k_generate, diversify=self.cfg.diversify_ops)
            # 2) features + scores
            for c in cand:
                c.features = self.feat.compute(c)
                c.scores = self.eval.score(c, goal=goal)
            # 3) prune + merge
            kept = self.prune.select(cand, beam=self.cfg.beam_size,
                                     novelty_tau=self.cfg.novelty_tau, merge_tau=self.cfg.merge_tau,
                                     archive=archive)
            self.db.persist(run_id, depth, kept)
            archive += kept
            frontier = kept
            if self._success(archive, self.cfg.success_threshold):
                break
            if len(archive) >= self.cfg.budget_nodes:
                break
        # 4) pick best path + render
        best_path = self._select_best_path(archive)
        assets = self.rndr.emit(run_id, best_path, archive)
        report = self.rpt.summarize(run_id, best_path, archive, goal)
        return {"run_id": run_id, "best_path": best_path, "assets": assets, "report": report}
