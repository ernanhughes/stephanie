# stephanie/services/chain_sampler.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from stephanie.components.nexus.utils.visual_thought import VisualThoughtOp, VisualThoughtType

__all__ = [
    "ChainStep",
    "ChainResult",
    "ChainCandidate",
    "VISUAL_TRIGGER_WORDS",
    "default_visual_bootstrap_ops",
    "detect_visual_triggers",
    "diversified_samples",
    "basic_selector_sicql_hrm_mars",
]

# ---------- Types you adapt/bridge ----------
@dataclass
class ChainStep:
    text: str
    visual_ops: Optional[List[VisualThoughtOp]] = None

@dataclass
class ChainResult:
    """Minimal interface returned by your chain runner."""
    steps: List[ChainStep]
    answer: str
    scores: Dict[str, float]  # e.g., {"sicql": 0.78, "hrm": 0.71, "mars": 0.69}
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChainCandidate:
    """What we pass to the selector."""
    mode: str  # "text" | "interleaved"
    result: ChainResult
    seed: int
    meta: Dict[str, Any] = field(default_factory=dict)

# ---------- Heuristics you can keep or replace ----------
VISUAL_TRIGGER_WORDS: set[str] = {
    "zoom", "focus", "closely", "examine", "detail",
    "highlight", "box", "region", "area",
    "path", "route", "steps", "direction",
    "bridge", "bottleneck", "cluster", "boundary",
}

def default_visual_bootstrap_ops(img: Image.Image) -> List[VisualThoughtOp]:
    """A simple first-step visual op: center zoom + hint box."""
    w, h = img.size
    cx, cy = int(w * 0.5), int(h * 0.5)
    ops: List[VisualThoughtOp] = [
        VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (cx, cy), "scale": 2.0}),
        VisualThoughtOp(VisualThoughtType.BBOX, {"xyxy": (int(w*0.3), int(h*0.3), int(w*0.7), int(h*0.7)), "width": 2}),
    ]
    return ops

def detect_visual_triggers(text: str, triggers: Optional[Sequence[str]] = None) -> int:
    """
    Returns a simple trigger count based on presence of visual words.
    Use this to bias p_interleaved upward when the question is spatial.
    """
    tokens = text.lower()
    words = VISUAL_TRIGGER_WORDS if triggers is None else set(map(str.lower, triggers))
    return sum(1 for w in words if w in tokens)

# ---------- Main API ----------
def diversified_samples(
    question: str,
    image: Optional[Image.Image],
    *,
    n_total: int = 8,
    p_interleaved: float = 0.5,
    run_chain_fn: Callable[[str, Optional[Image.Image], bool, int], ChainResult],
    score_selector_fn: Callable[[List[ChainCandidate]], ChainCandidate],
    time_budget_s: Optional[float] = None,
    seed: int = 0,
    auto_bias_interleaved: bool = True,
) -> Tuple[ChainCandidate, List[ChainCandidate]]:
    """
    Run a mixed batch of chains and select the best one via your selector.
    - n_total: total samples (text-only + interleaved)
    - p_interleaved: base fraction allocated to interleaved chains
    - auto_bias_interleaved: if True, increase interleaved share when visual triggers detected
    - time_budget_s: optional wall time limit (best-effort)
    - run_chain_fn: your chain executor (should honor `force_visual`)
    - score_selector_fn: your existing SICQL/HRM/MARS selector
    """
    rng = random.Random(seed)

    # Optional: bias toward interleaved if the query looks spatial
    if auto_bias_interleaved:
        trig = detect_visual_triggers(question)
        if trig > 0:
            # Smoothly push p_interleaved toward 0.8 based on trigger count
            p_interleaved = min(0.8, p_interleaved + 0.1 * min(trig, 3))

    n_inter = max(1, int(round(n_total * p_interleaved)))
    n_text = max(1, n_total - n_inter)
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(n_total)]

    # Launch batch
    t0 = time.time()
    candidates: List[ChainCandidate] = []

    # 1) Text-only chains
    for i in range(n_text):
        if time_budget_s and (time.time() - t0) > time_budget_s:
            break
        res = run_chain_fn(question, image, False, seeds[i])
        candidates.append(
            ChainCandidate(
                mode="text",
                result=res,
                seed=seeds[i],
                meta={"slot": i, "p_interleaved": p_interleaved, "auto_bias": auto_bias_interleaved},
            )
        )

    # 2) Interleaved chains (force at least one early visual op via your runner)
    for j in range(n_inter):
        if time_budget_s and (time.time() - t0) > time_budget_s:
            break
        res = run_chain_fn(question, image, True, seeds[n_text + j])
        candidates.append(
            ChainCandidate(
                mode="interleaved",
                result=res,
                seed=seeds[n_text + j],
                meta={"slot": n_text + j, "p_interleaved": p_interleaved, "auto_bias": auto_bias_interleaved},
            )
        )

    if not candidates:
        raise RuntimeError("No candidates produced; reduce constraints or check run_chain_fn.")

    # Selector picks winner (e.g., SICQL/HRM/MARS fused score)
    winner = score_selector_fn(candidates)
    return winner, candidates

# ---------- Convenience helpers ----------
def basic_selector_sicql_hrm_mars(cands: List[ChainCandidate]) -> ChainCandidate:
    """
    Example selector: average of available scores (SICQL/HRM/MARS).
    Replace with your PolicyAnalyzer or existing fused selector.
    """
    def fused_score(res: ChainResult) -> float:
        keys = ("sicql", "hrm", "mars")
        vals = [res.scores[k] for k in keys if k in res.scores]
        return sum(vals) / max(1, len(vals))

    return max(cands, key=lambda c: fused_score(c.result))
