# stephanie/agents/paper_improver/curriculum.py
# CurriculumScheduler — teachability scoring + tiered scheduling with exploration and explainability.

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_WORD = r"(?:^|[^A-Za-z0-9_])"  # left boundary (no lookbehind to keep it portable)
def _wb(term: str) -> re.Pattern:
    # compile a case-insensitive “whole-ish” word pattern (handles plurals via (s)?)
    term = re.escape(term)
    if term.endswith(r"\."):
        term = term[:-2] + r"\."  # keep trailing dot for "Eq."
    return re.compile(fr"{_WORD}({term})(s)?(?=[^A-Za-z0-9_]|$)", re.IGNORECASE)

class CurriculumScheduler:
    """
    Ranks papers by estimated teachability (↑ easier to turn into code/blog).
    Features:
      - Robust keyword detection (regex, case-insensitive, plural tolerant)
      - Signals for pseudocode, algorithms, tables/figures, equations, GitHub/code release
      - Section-aware biasing (list of sections or single section)
      - Smooth normalization (length-aware)
      - Explainable scoring
      - Tiered scheduling with exploration and optional filters
      - Optional calibration hooks
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: int = 0):
        self.config = self._merge_default_config(config or {})
        random.seed(seed)
        self._compile_patterns()

    # --------------------------- config ---------------------------

    def _merge_default_config(self, user: Dict[str, Any]) -> Dict[str, Any]:
        cfg = {
            "bonus_keywords": {
                "algorithm": 1.5,
                "pseudocode": 2.0,
                "procedure": 1.0,
                "table": 0.8, "tab.": 0.8, "table ": 0.8, "algorithm ": 1.5,
                "figure": 0.5, "fig.": 0.5, "fig ": 0.5,
                "equation": 0.5, "eq.": 0.5, "eq ": 0.5,
                "implementation": 1.0, "code": 0.8, "github": 1.2, "open-source": 0.8,
                "reproducible": 0.8, "dataset": 0.4
            },
            "penalty_keywords": {
                "proof": -1.0, "theorem": -1.0, "lemma": -0.8, "corollary": -0.8,
                "induction": -1.2, "derivation": -0.7, "by inspection": -0.5,
                "trivially": -0.3, "without loss of generality": -0.8
            },
            "section_bias": {  # applied when section titles match
                "method": 1.0, "methods": 1.0, "approach": 1.0, "model": 0.8,
                "experimental setup": 0.5, "implementation details": 0.6,
                "results": 0.3, "ablation": 0.4,
                "related work": -0.4, "introduction": 0.0, "conclusion": -0.2
            },
            # direct regex signals
            "signals": {
                "code_fence": {"pattern": r"```(\w+)?\n", "weight": 1.5},
                "algo_env":   {"pattern": r"\\begin\{algorithm\}", "weight": 2.0},
                "equation_env":{"pattern": r"\$(?:[^$]+)\$|\\\[(?:.|\n)+?\\\]", "weight": 0.5},
                "numbered_algo":{"pattern": r"Algorithm\s+\d+", "weight": 1.5},
                "numbered_table":{"pattern": r"Table\s+\d+", "weight": 0.8},
                "numbered_fig":  {"pattern": r"Fig\.?\s*\d+", "weight": 0.5},
            },
            # arXiv category boosts (optional)
            "arxiv_boost": {
                "cs.LG": 0.3, "cs.AI": 0.2, "cs.CL": 0.2, "stat.ML": 0.2
            },
            # normalization
            "norm": {
                "scale": 6.0,    # larger => slower saturation
                "length_coef": 0.00015,  # per character dampener
                "min_clip": 0.0,
                "max_clip": 1.0
            },
            "min_score_threshold": 0.0,
            # scheduling
            "tiers": [0.8, 0.6, 0.4, 0.2],  # boundaries for A/B/C/D tiers
            "explore_ratio": 0.15,          # fraction sampled from lower tiers for diversity
            "filters": { "arxiv_primary_only": False, "allowed_cats": [] },
        }
        # merge shallowly
        for k, v in user.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
        return cfg

    def _compile_patterns(self):
        self._bonus = {k: (_wb(k), w) for k, w in self.config["bonus_keywords"].items()}
        self._penal = {k: (_wb(k), w) for k, w in self.config["penalty_keywords"].items()}
        self._signals = {
            name: (re.compile(info["pattern"], re.IGNORECASE | re.MULTILINE), float(info["weight"]))
            for name, info in self.config["signals"].items()
        }

    # ------------------------ scoring ------------------------

    def compute_teachability_score(self, paper: Dict[str, Any]) -> float:
        """Normalized teachability in [0,1]."""
        raw, _ = self._raw_score(paper)
        return self._normalize(raw, paper)

    def score_with_explain(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Return normalized score + full breakdown for dashboards."""
        raw, breakdown = self._raw_score(paper)
        norm = self._normalize(raw, paper)
        return {
            "teachability_score": round(norm, 3),
            "raw": round(raw, 3),
            "breakdown": breakdown
        }

    def _raw_score(self, paper: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        text = self._extract_relevant_text(paper)
        title = (paper.get("title") or "")
        sects = self._collect_sections(paper)

        raw = 0.0
        detail: Dict[str, Any] = {"bonus": {}, "penalty": {}, "signals": {}, "section_bias": {}, "arxiv_boost": 0.0}

        # keyword bonuses
        for key, (pat, w) in self._bonus.items():
            c = len(pat.findall(text)) + len(pat.findall(title))
            if c:
                raw += c * w
                detail["bonus"][key] = c * w

        # penalties
        for key, (pat, w) in self._penal.items():
            c = len(pat.findall(text)) + len(pat.findall(title))
            if c:
                raw += c * w  # w is negative
                detail["penalty"][key] = c * w

        # strong regex signals
        for name, (pat, w) in self._signals.items():
            c = len(pat.findall(text))
            if c:
                raw += c * w
                detail["signals"][name] = c * w

        # section bias (sum over all sections present)
        for s in sects:
            st = s.lower()
            for sec_key, bias in self.config["section_bias"].items():
                if sec_key in st:
                    raw += bias
                    detail["section_bias"].setdefault(sec_key, 0.0)
                    detail["section_bias"][sec_key] += bias

        # arXiv boost
        cat = (paper.get("arxiv_primary") or paper.get("category") or "").strip()
        if cat in self.config["arxiv_boost"]:
            boost = self.config["arxiv_boost"][cat]
            raw += boost
            detail["arxiv_boost"] = boost

        return raw, detail

    def _normalize(self, raw: float, paper: Dict[str, Any]) -> float:
        # Smooth saturation with optional length damping.
        ncfg = self.config["norm"]
        length = max(1, len(self._extract_relevant_text(paper)))
        scale = float(ncfg["scale"]) + float(ncfg["length_coef"]) * length
        val = 1.0 - math.exp(-max(0.0, raw) / max(0.001, scale))  # maps 0..∞ → 0..1
        return float(max(ncfg["min_clip"], min(ncfg["max_clip"], val)))

    # ------------------------ extraction helpers ------------------------

    def _extract_relevant_text(self, paper: Dict[str, Any]) -> str:
        """Prefer body/abstract; include captions, code fences, and section texts if provided."""
        chunks: List[str] = []
        for key in ("body", "abstract", "text", "content", "caption", "description", "pdf_text"):
            v = paper.get(key)
            if isinstance(v, str) and v.strip():
                chunks.append(v)
        # include sections content
        for sec in paper.get("sections", []) or []:
            if isinstance(sec, dict) and isinstance(sec.get("text"), str):
                chunks.append(sec["text"])
        # include code fences (sometimes in separate field)
        if isinstance(paper.get("code"), str):
            chunks.append(paper["code"])
        return "\n".join(chunks)

    def _collect_sections(self, paper: Dict[str, Any]) -> List[str]:
        names: List[str] = []
        if isinstance(paper.get("sections"), list):
            for sec in paper["sections"]:
                title = sec.get("title") if isinstance(sec, dict) else None
                if isinstance(title, str):
                    names.append(title)
        if isinstance(paper.get("section"), str):
            names.append(paper["section"])
        return names

    # ------------------------ scheduling ------------------------

    def schedule_papers(
        self,
        papers: List[Dict[str, Any]],
        *,
        explore_ratio: Optional[float] = None,
        allowed_cats: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Tiered schedule with exploration:
          - Rank by teachability
          - Bucket into A/B/C/D by configured thresholds
          - Take most from Tier A, then sprinkle lower tiers by explore_ratio
        """
        explore = self.config["explore_ratio"] if explore_ratio is None else explore_ratio
        allow = allowed_cats or self.config["filters"].get("allowed_cats", [])

        # score and filter
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for p in papers:
            if allow and (p.get("arxiv_primary") or p.get("category")) not in allow:
                continue
            s = self.compute_teachability_score(p)
            if s >= self.config["min_score_threshold"]:
                p["teachability_score"] = s
                scored.append((s, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return []

        # bucket by tiers
        tA, tB, tC, tD = self.config["tiers"]
        A, B, C, D = [], [], [], []
        for s, p in scored:
            if s >= tA: A.append(p)
            elif s >= tB: B.append(p)
            elif s >= tC: C.append(p)
            elif s >= tD: D.append(p)
            else: D.append(p)

        # assemble with exploration
        out: List[Dict[str, Any]] = []
        out.extend(A)
        # sample from B/C/D with proportions
        remain = int(len(A) * explore)
        pool = B + C + D
        random.shuffle(pool)
        out.extend(pool[:remain])

        # append the rest by score
        seen = {id(x) for x in out}
        tail = [p for _, p in scored if id(p) not in seen]
        out.extend(tail)

        # attach difficulty_rank
        for i, p in enumerate(out, 1):
            p["difficulty_rank"] = i
        return out

    def tag_and_rank(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Backwards-compatible helper: annotate and sort by teachability (easiest first)."""
        scored = [(self.compute_teachability_score(p), p) for p in papers]
        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (score, p) in enumerate(scored, 1):
            p["teachability_score"] = score
            p["difficulty_rank"] = rank
        return [p for _, p in scored]

    # ------------------------ I/O + stats ------------------------

    def load_from_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load papers from JSON files (e.g., scraped/arxiv)."""
        papers = []
        for path in file_paths:
            p = Path(path)
            if p.exists() and p.suffix == ".json":
                try:
                    paper = json.loads(p.read_text())
                    paper["source_path"] = str(p)
                    papers.append(paper)
                except Exception as e:
                    print(f"⚠️ Failed to load {p}: {e}")
        return papers

    def save_curriculum(self, papers: List[Dict[str, Any]], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(papers, f, indent=2)
        print(f"✅ Curriculum saved to {output_path} ({len(papers)} papers)")

    def get_stats(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not papers:
            return {"count": 0, "avg_score": 0.0, "min_score": 0.0, "max_score": 0.0, "passing_count": 0}
        scores = [p.get("teachability_score", self.compute_teachability_score(p)) for p in papers]
        return {
            "count": len(papers),
            "avg_score": round(sum(scores) / len(scores), 3),
            "min_score": round(min(scores), 3),
            "max_score": round(max(scores), 3),
            "passing_count": len([s for s in scores if s >= self.config["min_score_threshold"]]),
            "tier_counts": self._tier_counts(scores)
        }

    def _tier_counts(self, scores: List[float]) -> Dict[str, int]:
        tA, tB, tC, tD = self.config["tiers"]
        A = sum(1 for s in scores if s >= tA)
        B = sum(1 for s in scores if tB <= s < tA)
        C = sum(1 for s in scores if tC <= s < tB)
        D = sum(1 for s in scores if tD <= s < tC)
        U = sum(1 for s in scores if s < tD)
        return {"A": A, "B": B, "C": C, "D": D, "U": U}
