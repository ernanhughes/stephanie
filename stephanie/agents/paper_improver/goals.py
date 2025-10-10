# stephanie/agents/paper_improver/goals.py
"""
goal templates, normalization, and portfolio scoring for VPM rows
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml  # optional: only needed for load_yaml

_logger = logging.getLogger(__name__)

# ----------------------------- data models -----------------------------

@dataclass(frozen=True)
class Normalization:
    """
    Map a raw metric to [0,1] for scoring.

    kind:
      - "pass_through": value already in [0,1]
      - "invert": 1 - v (when lower raw is better)
      - "band": clamp into [min,max] then rescale to [0,1] (e.g., readability FKGL 9–11)
      - "logistic": 1/(1+exp(-k*(v-x0)))   (tunable S-curve via 'k' and 'x0' in min/max)
      - "zscore":   convert to z=(v-mean)/std then logistic via k, x0=0
      - "threshold": v>=min_val -> 1, else v/max(min_val,eps)
    extras:
      - gamma: optional power curve on the result (>=0). gamma>1 emphasizes high end.
      - clip_min/clip_max: clamp final value.
    """
    kind: str = "pass_through"
    min_val: float = 0.0   # band.min, logistic.x0, threshold.min
    max_val: float = 1.0   # band.max, logistic.k (>0), zscore.std
    gamma: float = 1.0
    clip_min: float = 0.0
    clip_max: float = 1.0

    def _clamp01(self, x: float) -> float:
        return max(self.clip_min, min(self.clip_max, x))

    def apply(self, value: float) -> float:
        try:
            v = float(value)
        except Exception:
            return 0.0

        if math.isnan(v) or math.isinf(v):
            return 0.0

        if self.kind == "pass_through":
            out = v

        elif self.kind == "invert":
            out = 1.0 - v

        elif self.kind == "band":
            if self.max_val <= self.min_val:
                return 0.0
            v2 = max(self.min_val, min(self.max_val, v))
            out = (v2 - self.min_val) / (self.max_val - self.min_val)

        elif self.kind == "logistic":
            k = self.max_val if self.max_val != 0 else 1.0  # reuse max_val as slope k
            x0 = self.min_val                                # reuse min_val as midpoint x0
            out = 1.0 / (1.0 + math.exp(-k * (v - x0)))

        elif self.kind == "zscore":
            mean = self.min_val
            std = max(1e-9, self.max_val)
            z = (v - mean) / std
            k = 1.0  # slope
            out = 1.0 / (1.0 + math.exp(-k * z))

        elif self.kind == "threshold":
            t = self.min_val
            if v >= t:
                out = 1.0
            else:
                out = v / max(t, 1e-9)

        else:
            # unknown → safe default
            out = v

        out = self._clamp01(out)
        if self.gamma and self.gamma != 1.0 and out > 0.0:
            out = self._clamp01(out ** self.gamma)
        return out


@dataclass
class GoalTemplate:
    """
    A weighted blend over normalized metrics.

    weights: metric -> weight (non-negative)
    norms:   metric -> Normalization
    min_bar: optional per-metric minimum normalized requirement (0..1)
    hard_bars: if True, any bar miss zeros the final score; else soft-penalty (default False)
    normalize_weights: if True, re-scales weights to sum to 1 automatically
    """
    name: str
    kind: str  # "text" | "code"
    weights: Dict[str, float]
    norms: Dict[str, Normalization] = field(default_factory=dict)
    min_bar: Dict[str, float] = field(default_factory=dict)
    hard_bars: bool = False
    normalize_weights: bool = True

    @classmethod
    def from_dims(cls, name: str, kind: str, dims: List[str], default_threshold: float = 0.5) -> "GoalTemplate":
        """
        Create a balanced fallback template from list of dimensions.
        Useful when goal template doesn't exist.
        """
        n = len(dims)
        weight_per_dim = 1.0 / max(1, n)
        weights = {d: weight_per_dim for d in dims}
        norms = {d: Normalization("pass_through") for d in dims}
        min_bar = {d: default_threshold for d in dims}
        return cls(
            name=name,
            kind=kind,
            weights=weights,
            norms=norms,
            min_bar=min_bar,
            hard_bars=False,
            normalize_weights=True
        )

    def _normed(self, vpm_dims: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, _w in self.weights.items():
            raw = vpm_dims.get(k, 0.0)
            norm = self.norms.get(k, Normalization("pass_through"))
            out[k] = norm.apply(raw)
        return out

    def score(self, vpm_dims: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Return (composite_score, per_metric_normalized)."""
        normed = self._normed(vpm_dims)

        # enforce min bars
        misses = self.unmet(vpm_dims)
        if self.hard_bars and misses:
            return 0.0, normed
        elif misses:
            # soft-penalty for misses (halve the missed dims)
            for k in misses:
                normed[k] = 0.5 * normed.get(k, 0.0)

        # weighted average
        weights = {k: max(0.0, w) for k, w in self.weights.items()}
        denom = sum(weights.values())
        if denom <= 0:
            return 0.0, normed
        if self.normalize_weights:
            score = sum(normed.get(k, 0.0) * (w/denom) for k, w in weights.items())
        else:
            score = sum(normed.get(k, 0.0) * w for k, w in weights.items()) / denom

        return float(score), normed

    def unmet(self, vpm_dims: Dict[str, float], hysteresis: float = 0.0) -> List[str]:
        """Return list of metric keys that are below their min_bar (after hysteresis)."""
        misses: List[str] = []
        if not self.min_bar:
            return misses
        for k, bar in self.min_bar.items():
            v = self.norms.get(k, Normalization()).apply(vpm_dims.get(k, 0.0))
            if v + hysteresis < bar:
                misses.append(k)
        return misses

    def explain(self, vpm_dims: Dict[str, float]) -> Dict[str, Any]:
        """Return per-metric normalized values and contributions (weight * norm)."""
        normed = self._normed(vpm_dims)
        weights = {k: max(0.0, w) for k, w in self.weights.items()}
        denom = sum(weights.values()) or 1.0
        contrib = {k: (normed.get(k, 0.0) * (weights[k]/denom)) for k in weights}
        return {"normalized": normed, "contrib": contrib}


# ----------------------------- defaults -----------------------------

# Text norms (FKGL target band ~9–11; logistic for novelty to avoid overfitting high values)
TEXT_NORMS_DEFAULT = {
    "coverage":           Normalization("pass_through"),
    "correctness":        Normalization("pass_through"),
    "coherence":          Normalization("pass_through"),
    "citation_support":   Normalization("pass_through"),
    "entity_consistency": Normalization("pass_through"),
    "readability":        Normalization("band", min_val=9.0, max_val=11.0, gamma=1.0),  # FKGL band
    "novelty":            Normalization("logistic", min_val=0.5, max_val=6.0),  # x0=0.5, k=6
    # Optional extended dims
    "stickiness":         Normalization("pass_through"),
}

# Code norms
CODE_NORMS_DEFAULT = {
    "tests_pass_rate": Normalization("pass_through"),
    "coverage":        Normalization("pass_through"),
    "type_safe":       Normalization("pass_through"),
    "lint_clean":      Normalization("pass_through"),
    "complexity_ok":   Normalization("pass_through"),
    "mutation_score":  Normalization("pass_through"),
}

# Default text goals
ACADEMIC_SUMMARY = GoalTemplate(
    name="academic_summary",
    kind="text",
    weights={
        "coverage": 0.30,
        "correctness": 0.25,
        "citation_support": 0.20,
        "coherence": 0.15,
        "readability": 0.10,
    },
    norms=TEXT_NORMS_DEFAULT,
    min_bar={
        "coverage": 0.80,
        "correctness": 0.75,
        "coherence": 0.75,
        "citation_support": 0.65,
        "entity_consistency": 0.80,
    },
    hard_bars=False,
)

PRACTITIONER_TUTORIAL = GoalTemplate(
    name="practitioner_tutorial",
    kind="text",
    weights={
        "correctness": 0.30,
        "coverage": 0.25,
        "coherence": 0.20,
        "entity_consistency": 0.15,
        "readability": 0.10,
    },
    norms=TEXT_NORMS_DEFAULT,
    min_bar={
        "coverage": 0.75,
        "correctness": 0.80,
        "readability": 0.60,  # normalized band score (maps FKGL into [0,1])
    },
    hard_bars=False,
)

BLOG_GENERAL = GoalTemplate(
    name="blog_general",
    kind="text",
    weights={
        "coherence": 0.25,
        "coverage": 0.20,
        "correctness": 0.20,
        "readability": 0.20,
        "novelty": 0.15,
    },
    norms=TEXT_NORMS_DEFAULT,
    min_bar={
        "coverage": 0.70,
        "coherence": 0.70,
    },
    hard_bars=False,
)

# Default code goals
STRICT_CI = GoalTemplate(
    name="strict_ci",
    kind="code",
    weights={
        "tests_pass_rate": 0.40,
        "coverage": 0.25,
        "type_safe": 0.15,
        "lint_clean": 0.10,
        "mutation_score": 0.10,
    },
    norms=CODE_NORMS_DEFAULT,
    min_bar={
        "tests_pass_rate": 1.00,
        "coverage": 0.70,
        "type_safe": 1.00,
        "lint_clean": 1.00,
    },
    hard_bars=True,  # if CI bars not met, score is 0
)

FAST_ITER = GoalTemplate(
    name="fast_iter",
    kind="code",
    weights={
        "tests_pass_rate": 0.50,
        "coverage": 0.15,
        "type_safe": 0.10,
        "lint_clean": 0.10,
        "complexity_ok": 0.10,
        "mutation_score": 0.05,
    },
    norms=CODE_NORMS_DEFAULT,
    min_bar={
        "tests_pass_rate": 1.00,
        "coverage": 0.60,
    },
    hard_bars=False,
)

DEFAULT_TEMPLATES: Dict[str, GoalTemplate] = {
    "text/academic_summary": ACADEMIC_SUMMARY,
    "text/practitioner_tutorial": PRACTITIONER_TUTORIAL,
    "text/blog_general": BLOG_GENERAL,
    "code/strict_ci": STRICT_CI,
    "code/fast_iter": FAST_ITER,
}


# ----------------------------- scorer -----------------------------

class GoalScorer:
    """
    Computes portfolio scores for VPM rows against goal templates.
    Surfaces unmet bars, per-metric contributions, and suggests next targets.
    Also supports scoring a *portfolio* (list) of rows with mean or EWMA.
    """

    def __init__(
        self,
        templates: Optional[Dict[str, GoalTemplate]] = None,
        judge: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
        logger=None,
    ):
        """
        judge (optional): function(row) -> extra dimension dict (e.g., external LLM judge).
        Any returned dims will be merged (and normalized via template norms if targeted).
        """
        self.templates = templates or dict(DEFAULT_TEMPLATES)  # copy defaults
        self.judge = judge
        self.logger = logger

    def available(self, *, kind: str) -> List[str]:
        pre = f"{kind}/"
        return [k.split("/", 1)[1] for k in self.templates.keys() if k.startswith(pre)]

    def score(
        self,
        kind: str,
        goal: str,
        vpm_row: Dict[str, float],
        hysteresis: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Return:
        {
          "goal": "...",
          "score": float,
          "normalized": {dim: norm_val},
          "contrib": {dim: contribution},
          "unmet": [dim...],
          "raw": vpm_row
        }
        """
        key = f"{kind}/{goal}"
        if key not in self.templates:
            # --- Dynamic fallback template creation ---
            dims = list(vpm_row.keys())

            # Use classmethod to create valid template
            self.templates[key] = GoalTemplate.from_dims(
                name=goal,
                kind=kind,
                dims=dims,
                default_threshold=0.5
            )

            if self.logger:
                _logger.debug("GoalTemplateCreated"
                    f"kind: {kind}"
                    f"goal: {goal}"
                    f"dims: {dims}"
                    "message: Dynamic goal template created on-the-fly"
                )

        tpl = self.templates[key]
        # Start with only numeric values
        dims = {
            k: v for k, v in vpm_row.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

        # Pull in optional judge signals
        if self.judge:
            try:
                extra = self.judge(vpm_row) or {}
                # Only add numeric values from judge too
                for k, v in extra.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        dims[k] = v
                    else:
                        _logger.debug("JudgeReturnedNonNumeric"
                            f"metric: {k}"
                            f"value: {v}"
                            f"type: {type(v).__name__}"
                            f"warning: Judge returned non-numeric value; skipping"
                        )
            except Exception as e:
                _logger.error(f"Judge execution failed: {e}")

        _logger.debug(
            "ScoringInputDims | "
            f"numeric_dims={list(dims.keys())} | "
            f"ignored_keys={[k for k in vpm_row.keys() if k not in dims]} | "
            f"raw_types={{{', '.join(f'{k}: {type(v).__name__}' for k, v in vpm_row.items())}}}"
        )

        score, _normed = tpl.score(dims)

        expl = tpl.explain(dims)

        _logger.debug("ScoringOutput "
            f"score={round(float(score),4)} "
            f"normalized={{{', '.join(f'{k}: {round(float(v),4)}' for k,v in expl['normalized'].items())}}} "
            f"contrib={{{', '.join(f'{k}: {round(float(v),4)}' for k,v in expl['contrib'].items())}}}"
        )
        unmet = tpl.unmet(dims, hysteresis=hysteresis)
        expl = tpl.explain(dims)

        return {
            "goal": goal,
            "score": round(float(score), 4),
            "normalized": {k: round(float(v), 4) for k, v in expl["normalized"].items()},
            "contrib": {k: round(float(v), 4) for k, v in expl["contrib"].items()},
            "unmet": unmet,
            "raw": vpm_row,
        }

    def score_portfolio(
        self,
        kind: str,
        goal: str,
        rows: Iterable[Dict[str, float]],
        method: str = "mean",
        alpha: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Score a list of VPM rows as a portfolio.
        method: "mean" | "ewma"  (EWMA with weight 'alpha' on the latest)
        Returns composite portfolio score and per-row scores + delta vs previous.
        """
        rows = list(rows)
        if not rows:
            return {"goal": goal, "portfolio_score": 0.0, "per_row": []}

        per = [self.score(kind, goal, r) for r in rows]
        scores = [p["score"] for p in per]

        if method == "ewma":
            agg = 0.0
            for s in scores:
                agg = alpha * s + (1 - alpha) * agg
            portfolio = agg
        else:
            portfolio = sum(scores) / len(scores)

        # deltas vs previous (for trend)
        deltas = [None] + [round(scores[i] - scores[i-1], 4) for i in range(1, len(scores))]
        for i, p in enumerate(per):
            p["delta_vs_prev"] = deltas[i]

        return {
            "goal": goal,
            "portfolio_score": round(float(portfolio), 4),
            "per_row": per,
        }

    def suggest_targets(
        self,
        kind: str,
        goal: str,
        vpm_row: Dict[str, float],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Returns the K weakest normalized metrics under this goal (ascending).
        Useful to route edits (e.g., add citations, raise coverage).
        """
        tpl = self.templates[f"{kind}/{goal}"]
        _, normed = tpl.score(vpm_row)
        pairs = [(k, normed.get(k, 0.0)) for k in tpl.weights.keys()]
        pairs.sort(key=lambda x: x[1])
        return pairs[:top_k]

    def suggest_actions(
        self,
        kind: str, goal: str, vpm_row: Dict[str, float], hysteresis: float = 0.0
    ) -> List[str]:
        """
        Turn unmet bars and weakest dims into actionable strings.
        """
        tpl = self.templates[f"{kind}/{goal}"]
        misses = tpl.unmet(vpm_row, hysteresis=hysteresis)
        actions: List[str] = []
        for k in misses:
            if kind == "text" and k == "citation_support":
                actions.append("Add or strengthen citations ([#]) for factual claims.")
            elif kind == "text" and k == "coverage":
                actions.append("Add missing claims/units from the content plan.")
            elif kind == "code" and k == "coverage":
                actions.append("Add more targeted tests to raise coverage.")
            elif kind == "code" and k == "mutation_score":
                actions.append("Harden tests (mutation testing not killing enough mutants).")
            else:
                actions.append(f"Improve '{k}' to meet minimum bar.")
        # add weakest dims (top-2)
        for k, _v in self.suggest_targets(kind, goal, vpm_row, top_k=2):
            if k not in misses:
                actions.append(f"Boost '{k}' — currently one of the weakest dimensions.")
        return actions


# ----------------------------- I/O helpers -----------------------------

def load_yaml(path: str | None) -> Dict[str, Any]:
    """
    Load custom templates from a YAML file.
    Format:
      text:
        blog_general:
          weights: {coverage: 0.2, correctness: 0.2, ...}
          norms:
            readability: {kind: band, min_val: 9.0, max_val: 11.0, gamma: 1.2}
          min_bar: {coverage: 0.75, coherence: 0.7}
          hard_bars: false
      code:
        strict_ci:
          ...
    """
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not installed; cannot load YAML templates.")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping with 'text'/'code' sections.")
    return data


def build_templates_from_yaml(config: Dict[str, Any]) -> Dict[str, GoalTemplate]:
    """
    Construct GoalTemplate mapping from a parsed YAML dict.
    Validates weights and norms; merges with defaults.
    """
    out: Dict[str, GoalTemplate] = dict(DEFAULT_TEMPLATES)  # start from defaults
    for kind in ("text", "code"):
        block = config.get(kind)
        if not isinstance(block, dict):
            if block is None:
                continue
            raise ValueError(f"'{kind}' section must be a mapping of templates.")
        for name, body in block.items():
            if not isinstance(body, dict):
                raise ValueError(f"Template '{name}' must be a mapping.")
            weights = body.get("weights", {})
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"Template '{name}' must define non-empty 'weights'.")
            min_bar = body.get("min_bar", {}) or {}
            if not isinstance(min_bar, dict):
                raise ValueError(f"Template '{name}': 'min_bar' must be a mapping.")
            norms_in = body.get("norms", {}) or {}
            if not isinstance(norms_in, dict):
                raise ValueError(f"Template '{name}': 'norms' must be a mapping.")
            hard_bars = bool(body.get("hard_bars", False))
            normalize_weights = bool(body.get("normalize_weights", True))

            # assemble normalizers
            norms: Dict[str, Normalization] = {}
            for k, ncfg in norms_in.items():
                if not isinstance(ncfg, dict):
                    raise ValueError(f"Template '{name}': norm for '{k}' must be a mapping.")
                norms[k] = Normalization(
                    kind=str(ncfg.get("kind", "pass_through")),
                    min_val=float(ncfg.get("min_val", 0.0)),
                    max_val=float(ncfg.get("max_val", 1.0)),
                    gamma=float(ncfg.get("gamma", 1.0)),
                    clip_min=float(ncfg.get("clip_min", 0.0)),
                    clip_max=float(ncfg.get("clip_max", 1.0)),
                )

            default_norms = TEXT_NORMS_DEFAULT if kind == "text" else CODE_NORMS_DEFAULT
            tpl = GoalTemplate(
                name=name,
                kind=kind,
                weights=weights,
                norms=(norms or default_norms),
                min_bar=min_bar,
                hard_bars=hard_bars,
                normalize_weights=normalize_weights,
            )
            out[f"{kind}/{name}"] = tpl
    return out


# ----------------------------- quick demo -----------------------------
if __name__ == "__main__":  # pragma: no cover
    # Example VPM rows
    text_row = {
        "coverage": 0.82, "correctness": 0.78, "coherence": 0.76,
        "citation_support": 0.68, "entity_consistency": 0.86, "readability": 10.1, "novelty": 0.55
    }
    code_row = {
        "tests_pass_rate": 1.0, "coverage": 0.73, "type_safe": 1.0,
        "lint_clean": 1.0, "complexity_ok": 0.8, "mutation_score": 0.62
    }

    gs = GoalScorer()
    print("[text/academic_summary]", json.dumps(gs.score("text", "academic_summary", text_row), indent=2))
    print("[code/strict_ci]", json.dumps(gs.score("code", "strict_ci", code_row), indent=2))
    print("[portfolio mean]", json.dumps(gs.score_portfolio("text", "blog_general", [text_row]*3, method="mean"), indent=2))
    print("[portfolio ewma]", json.dumps(gs.score_portfolio("code", "fast_iter", [code_row]*3, method="ewma"), indent=2))
    print("[suggest targets]", gs.suggest_targets("text", "academic_summary", text_row, top_k=3))
    print("[suggest actions]", gs.suggest_actions("text", "academic_summary", text_row))