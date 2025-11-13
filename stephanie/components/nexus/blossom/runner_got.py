# stephanie/components/nexus/blossom_runner_got.py
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from stephanie.components.nexus.plan_prompts import (  # tiny helpers (see below)
    build_thought_prompt)

# If you prefer a very small, self-contained runner without GRPO, this file does that.
# It returns the exact shape that BlossomToScorableAgent expects.

@dataclass
class ThoughtState:
    text: str
    parent_idx: Optional[int] = None
    depth: int = 0
    reward: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    role: str = "candidate"     # 'candidate' | 'winner' | etc.

class BlossomRunnerGoT:
    """
    Lightweight ToT→GoT runner:
      - Expands K candidates per depth up to D
      - Scores each state with a learned/composite metric
      - Applies simple GoT merges to reduce duplicates
      - Picks winners and emits nodes list for persistence
    """
    def __init__(self, *, llm, scorer, sharpener=None,
                 K:int=3, D:int=2, merge_thresh: float=0.92, novelty_floor: float=0.15):
        """
        llm: callable(prompt:str, context:dict)->str
        scorer: callable(state_text:str, goal:dict)->(reward:float, metrics:dict)
        sharpener: optional callable(text, goal)->text
        """
        self.llm = llm
        self.scorer = scorer
        self.sharpener = sharpener
        self.K, self.D = int(K), int(D)
        self.merge_thresh = float(merge_thresh)
        self.novelty_floor = float(novelty_floor)

    async def run_episode(self, source_text: str, goal: dict, context: dict) -> dict:
        t0 = time.time()
        episode_id = int(time.time() * 1000)  # or persist first & use DB id

        # Seed = baseline state (the given item)
        baseline_metrics = self._safe_score(source_text, goal)
        baseline = {"text": source_text,
                    "reward": baseline_metrics[0],
                    "metrics": baseline_metrics[1]}

        # Layer 0
        frontier: List[ThoughtState] = [ThoughtState(text=source_text, parent_idx=None, depth=0,
                                                     reward=baseline["reward"], metrics=baseline["metrics"], role="seed")]
        all_states: List[ThoughtState] = [frontier[0]]

        # Expand by depth
        for depth in range(1, self.D + 1):
            next_frontier: List[ThoughtState] = []
            # Select top-K parents to branch from (by reward, allow some diversity)
            parents = self._select_parents(frontier, self.K)
            for p in parents:
                # Generate K variants
                variants = self._generate_variants(p.text, goal, K=self.K, context=context)
                for v in variants:
                    v2 = self._maybe_sharpen(v, goal)
                    reward, metrics = self._safe_score(v2, goal)
                    st = ThoughtState(text=v2, parent_idx=all_states.index(p), depth=depth,
                                      reward=reward, metrics=metrics, role="candidate")
                    next_frontier.append(st)
            # Merge similar (GoT op) & keep diverse/novel states
            next_frontier = self._merge_and_filter(next_frontier, all_states)
            all_states.extend(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        # Pick winners (top by reward, ensure diversity)
        winners = self._pick_winners(all_states, top=3)

        # Build node list (bn_id assigned by store layer; we return pseudo ids for now)
        nodes = []
        idx_to_pseudo_bn = {}
        for i, st in enumerate(all_states):
            rid = str(uuid.uuid4())[:8]
            idx_to_pseudo_bn[i] = rid
            nodes.append({
                "bn_id": -1,  # resolver fills real ID during persist
                "parent_bn_id": idx_to_pseudo_bn.get(st.parent_idx) if st.parent_idx is not None else None,
                "plan_text": st.text,
                "reward": float(st.reward),
                "metrics": st.metrics,
                "role": st.role,
            })

        # winners mapped
        winner_nodes = []
        for w in winners:
            winner_nodes.append({
                "bn_id": -1,
                "plan_text": w.text,
                "reward": float(w.reward),
                "metrics": w.metrics,
            })

        return {
            "episode_id": episode_id,
            "nodes": nodes,
            "winners": winner_nodes,
            "baseline": baseline,
            "elapsed_s": time.time() - t0,
        }

    # ------------------------- internals -------------------------

    def _generate_variants(self, text: str, goal: dict, *, K: int, context: dict) -> List[str]:
        prompt = build_thought_prompt(goal_text=goal.get("goal_text", ""), state_text=text, k=K)
        out = self.llm(prompt, context=context)  # expect numbered list or blocks
        # simple splitter
        cand = [s.strip("- ").strip() for s in out.split("\n") if s.strip()]
        return cand[:K] if cand else [text]

    def _safe_score(self, text: str, goal: dict) -> Tuple[float, Dict[str, float]]:
        # scorer returns (reward, metrics). Compose SICQL/MRQ/HRM under the hood if you want.
        try:
            r, m = self.scorer(text, goal)
            return float(r), dict(m or {})
        except Exception:
            return 0.0, {"alignment": 0.0, "faithfulness": 0.0, "novelty": 0.0, "usefulness": 0.0}

    def _maybe_sharpen(self, text: str, goal: dict) -> str:
        return self.sharpener(text, goal) if callable(self.sharpener) else text

    def _select_parents(self, frontier: List[ThoughtState], k:int) -> List[ThoughtState]:
        if not frontier: return []
        # reward-sorted; keep at least 1 non-top if available for diversity
        fs = sorted(frontier, key=lambda s: s.reward, reverse=True)
        out = fs[:max(1, min(k, len(fs)))]
        if len(fs) > k:
            out.append(fs[-1])
        # dedup
        seen, uniq = set(), []
        for s in out:
            key = (s.text[:80], s.depth)
            if key in seen: continue
            seen.add(key); uniq.append(s)
        return uniq[:k]

    def _merge_and_filter(self, new_states: List[ThoughtState], all_states: List[ThoughtState]) -> List[ThoughtState]:
        # very light novelty gate + near-duplicate merge by cosine via embed cache if available
        if not new_states: return []
        kept: List[ThoughtState] = []
        for st in sorted(new_states, key=lambda s: s.reward, reverse=True):
            novel = st.metrics.get("novelty", 0.0)
            if novel < self.novelty_floor:
                continue
            if self._near_duplicate(st, kept + all_states):
                continue
            kept.append(st)
        return kept

    def _near_duplicate(self, s: ThoughtState, pool: List[ThoughtState]) -> bool:
        # replace with real embed cosine; placeholder: token overlap
        def sim(a,b):
            A,B=set(a.text.lower().split()), set(b.text.lower().split())
            return len(A & B) / max(1,len(A|B))
        return any(sim(s, p) >= self.merge_thresh for p in pool)

    def _pick_winners(self, states: List[ThoughtState], top:int=3) -> List[ThoughtState]:
        cands = [s for s in states if s.depth>0] or states
        cands = sorted(cands, key=lambda s: s.reward, reverse=True)
        winners = []
        for s in cands:
            if not any(self._near_duplicate(s, [w]) for w in winners):
                s.role = "winner"
                winners.append(s)
            if len(winners) >= top: break
        return winners


# ---- tiny prompt helpers (swap with your prompt lib) ----

def build_thought_prompt(*, goal_text: str, state_text: str, k:int) -> str:
    return f"""
You improve and diversify ideas toward this goal:

GOAL: {goal_text}

CURRENT STATE:
\"\"\"{state_text}\"\"\"

Propose {k} DIVERSE next refinements or directions.
- Each should be self-contained and useful.
- Keep them concise but concrete.
- Prefer factual, actionable steps; avoid fluff.

List them:
""".strip()
