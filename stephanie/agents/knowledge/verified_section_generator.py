# stephanie/agents/knowledge/verified_section_generator.py
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.paper_improver.goals import GoalScorer


def _hash(x: str) -> str: return hashlib.sha256(x.encode("utf-8")).hexdigest()[:12]
def _toklen(s: str) -> int: return max(1, len(s)//4)

@dataclass
class VConfig:
    beam: int = 6
    max_depth: int = 6
    branch_per_node: int = 6
    target_quality: float = 0.95
    max_bundle_tokens: int = 1800
    keep_top_k_bundles: int = 8
    # weights for committee
    w_goal: float = 0.25
    w_evidence: float = 0.25
    w_traj: float = 0.15
    w_coherence: float = 0.15
    w_style: float = 0.10
    w_novelty: float = 0.10

class VerificationScorer:
    """
    Reuses GoalScorer + simple evidence/trajectory checks + calibration.
    Assumes 'knowledge_tree' is provided.
    """
    def __init__(self, memory, logger, goal_template="blog_section", weights: Dict[str, float] = None):
        self.memory = memory
        self.logger = logger
        self.goal = goal_template
        self.gs = GoalScorer(logger=logger)
        self.w = weights or {}

    def _overlap(self, a: str, b: str) -> float:
        aw = set(a.lower().split()); bw = set(b.lower().split())
        return len(aw & bw)/max(1, len(aw))

    def score(self, text: str, tree: Dict[str, Any], neighbors: Tuple[str, str] = ("","")) -> Tuple[float, Dict[str,float]]:
        claims = tree.get("claims", [])
        insights = tree.get("insights", [])
        conns = tree.get("connections", [])

        # goal fit
        goal_fit = self.gs.score(kind="text", goal=self.goal, vpm_row={
            "correctness": 0.6, "progress": 0.6, "evidence": 0.6, "novelty": 0.6
        }).get("score", 0.6)  # we don’t have per-text vpm here; keep neutral baseline

        # evidence support: claim & insight coverage
        claim_cov = 0.0
        if claims:
            hits = sum(1 for c in claims if self._overlap(text, c["text"]) > 0.05)
            claim_cov = hits / len(claims)
        insight_cov = 0.0
        if insights:
            hits = sum(1 for i in insights if self._overlap(text, i["text"]) > 0.05)
            insight_cov = hits / len(insights)
        evidence = 0.6 * claim_cov + 0.4 * insight_cov

        # trajectory alignment ~ rely on connections density
        traj = min(1.0, 0.5 + 0.5 * evidence)

        # coherence with neighbors
        prev, nxt = neighbors
        coh = 0.5
        if prev: coh = max(coh, 0.5 + 0.5*self._overlap(text, prev))
        if nxt:  coh = max(coh, 0.5 + 0.5*self._overlap(text, nxt))

        # novelty vs raw paper section text: prefer synthesis not copy
        # crude proxy: penalize very high overlap with any single claim
        max_claim_overlap = max([self._overlap(text, c["text"]) for c in claims] or [0.0])
        novelty = max(0.0, 1.0 - max_claim_overlap)

        # style: short, clear, cites figures if relevant
        style = 0.5 + 0.2*("figure" in text.lower() or "table" in text.lower())

        parts = {
            "goal": goal_fit, "evidence": evidence, "traj": traj,
            "coherence": coh, "novelty": novelty, "style": style
        }
        # weighted sum (calibration hook)
        w = self.w or {"goal":0.25,"evidence":0.25,"traj":0.15,"coherence":0.15,"novelty":0.10,"style":0.10}
        raw = sum(parts[k]*w[k] for k in w)
        # simple calibration fallback (identity)
        calibrated = min(1.0, max(0.0, raw))
        return calibrated, parts

class VerifiedSectionGeneratorAgent(BaseAgent):
    """
    LATS/beam search over candidates from tile bundles, verified against knowledge tree.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        vcfg = cfg.get("verified_section", {})
        self.c = VConfig(**vcfg) if vcfg else VConfig()
        self.scorer = VerificationScorer(memory, logger, goal_template="blog_section")

    def _tiles_from_context(self, sec: Dict[str,Any], msgs: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        tiles = []
        # paper tiles: first/middle/last spans
        st = (sec.get("section_text") or "").strip()
        if st:
            cuts = [st[:600], st[len(st)//2: len(st)//2 + 600], st[-600:]]
            for i, ct in enumerate(cuts):
                tiles.append({"id": f"ps_{i}", "kind": "paper", "text": ct.strip()})
        # conversation tiles
        for m in msgs:
            tiles.append({"id": f"cv_{_hash(m['text'][:80])}", "kind": "conv", "text": m["text"][:600].strip()})
        return tiles

    def _bundle(self, tiles: List[Dict[str,Any]]) -> List[List[Dict[str,Any]]]:
        # simple greedy packing by token budget + diversity
        random.shuffle(tiles)
        bundles, cur, cur_tok = [], [], 0
        for t in tiles:
            ln = _toklen(t["text"])
            if cur_tok + ln > self.c.max_bundle_tokens and cur:
                bundles.append(cur); cur, cur_tok = [], 0
            cur.append(t); cur_tok += ln
        if cur: bundles.append(cur)
        # keep top-N diverse by kind balance
        def score_bundle(b): 
            kinds = [x["kind"] for x in b]
            bal = 1.0 - abs(kinds.count("paper") - kinds.count("conv"))/max(1,len(b))
            return 0.7*bal + 0.3*min(1.0, len(b)/10)
        bundles.sort(key=score_bundle, reverse=True)
        return bundles[:self.c.keep_top_k_bundles]

    def _prompt(self, section_name: str, bundle: List[Dict[str,Any]], tree: Dict[str,Any]) -> str:
        ctx = "\n\n".join([f"[{t['kind'].upper()}] {t['text']}" for t in bundle])
        claims = "\n".join([f"- {c['text']}" for c in tree.get("claims",[])[:6]])
        return (
            f"You are drafting a crisp blog section titled: {section_name}.\n"
            f"Ground yourself ONLY in the following context tiles and do not invent facts.\n"
            f"Context:\n{ctx}\n\n"
            f"Key claims to cover:\n{claims}\n\n"
            f"Write 2–4 short paragraphs with inline figure/table references when appropriate."
        )

    def _gen(self, prompt: str) -> str:
        # minimal LLM wrapper using your memory.embedding or existing gen; replace with your generator
        # Here we assume a simple synchronous call via memory.embedding.llm if you have it; otherwise stub.
        # For now, just echo prompt tail as a placeholder (replace in your stack).
        return prompt.split("Context:\n")[-1][:1000]  # <<< REPLACE with real LLM call

    async def run(self, context: Dict[str,Any]) -> Dict[str,Any]:
        sec = context.get("paper_section") or {}
        tree = context.get("knowledge_graph") or {}
        crit = context.get("critical_messages", [])
        if not sec or not tree:
            self.logger.log("VerifiedSectionSkipped", {"reason":"missing_inputs"})
            return context

        tiles = self._tiles_from_context(sec, crit)
        bundles = self._bundle(tiles)
        section_name = sec.get("section_name", "Section")

        # Beam search
        Node = Tuple[str, float, Dict[str,float], List[str]]  # (text, score, parts, bundle_ids)
        beam: List[Node] = []
        # seed from bundles
        for b in bundles:
            prompt = self._prompt(section_name, b, tree)
            cand = self._gen(prompt)
            s, parts = self.scorer.score(cand, tree)
            beam.append((cand, s, parts, [t["id"] for t in b]))
        beam.sort(key=lambda x: x[1], reverse=True)
        beam = beam[:self.c.beam]

        depth = 1
        while depth < self.c.max_depth:
            # stopping if strong enough and stable
            top = beam[0]
            if top[1] >= self.c.target_quality:
                break

            # expand
            cands: List[Node] = []
            for (text, score, parts, used) in beam:
                # propose variations: rewrite / swap bundle / focused improvement
                for b in random.sample(bundles, k=min(self.c.branch_per_node, len(bundles))):
                    prompt = self._prompt(section_name, b, tree) + "\n\nRevise to improve coverage and coherence."
                    cand = self._gen(prompt + "\n\nDraft:\n" + text)
                    s, pr = self.scorer.score(cand, tree)
                    cands.append((cand, s, pr, [t["id"] for t in b]))

            cands.sort(key=lambda x: x[1], reverse=True)
            beam = cands[:self.c.beam]
            depth += 1

        best = max(beam, key=lambda x: x[1])
        context["verified_section"] = best[0]
        context["verification_trace"] = {
            "score": best[1], "parts": best[2], "bundles": best[3], "depth": depth
        }
        context["quality_confidence"] = best[1]
        self.logger.log("VerifiedSectionDone", {"confidence": best[1], "depth": depth})
        return context
