# goals.py — goal templates, normalization, and portfolio scoring for VPM rows
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple, List, Callable
import math
import json

try:
    import yaml  # optional: only needed for load_yaml
except Exception:  # pragma: no cover
    yaml = None  # graceful degradation


# ----------------------------- data models -----------------------------

@dataclass(frozen=True)
class Normalization:
    """
    How to map a raw metric to [0,1] for scoring.
    - "pass_through": value already in [0,1]
    - "band": clamp into [min,max] then rescale to [0,1] (e.g., readability FKGL 9–11)
    - "invert": 1 - value (when lower raw is better)
    """
    kind: str = "pass_through"
    min_val: float = 0.0
    max_val: float = 1.0

    def apply(self, value: float) -> float:
        if value is None or math.isnan(float(value)):  # type: ignore[arg-type]
            return 0.0
        v = float(value)
        if self.kind == "pass_through":
            return max(0.0, min(1.0, v))
        if self.kind == "invert":
            return max(0.0, min(1.0, 1.0 - v))
        if self.kind == "band":
            if self.max_val <= self.min_val:
                return 0.0
            v = max(self.min_val, min(self.max_val, v))
            return (v - self.min_val) / (self.max_val - self.min_val)
        # unknown → safe default
        return max(0.0, min(1.0, v))


@dataclass
class GoalTemplate:
    """
    A weighted blend over normalized metrics.
    weights: metric -> weight (non-negative)
    norms:   metric -> Normalization
    min_bar: optional per-metric minimum normalized requirement
    """
    name: str
    kind: str  # "text" | "code"
    weights: Dict[str, float]
    norms: Dict[str, Normalization] = field(default_factory=dict)
    min_bar: Dict[str, float] = field(default_factory=dict)

    def score(self, vpm_dims: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Return (composite_score, per_metric_normalized)."""
        # normalize
        normed: Dict[str, float] = {}
        for k, w in self.weights.items():
            raw = vpm_dims.get(k)
            norm = self.norms.get(k, Normalization("pass_through"))
            normed[k] = norm.apply(raw if raw is not None else 0.0)

        # enforce min bars (if any)
        for k, bar in self.min_bar.items():
            if k in normed and normed[k] < bar:
                # penalize proportionally below bar
                normed[k] = normed[k] * 0.5

        # weighted average (avoid division by zero)
        denom = sum(max(0.0, w) for w in self.weights.values())
        if denom <= 0:
            return 0.0, normed
        score = sum(normed.get(k, 0.0) * max(0.0, w) for k, w in self.weights.items()) / denom
        return score, normed

    def unmet(self, vpm_dims: Dict[str, float], hysteresis: float = 0.0) -> List[str]:
        """Return list of metric keys that are below their min_bar (after hysteresis)."""
        misses = []
        if not self.min_bar:
            return misses
        for k, bar in self.min_bar.items():
            v = self.norms.get(k, Normalization()).apply(vpm_dims.get(k, 0.0))
            if v + hysteresis < bar:
                misses.append(k)
        return misses


# ----------------------------- defaults -----------------------------

# Normalizers for common text metrics
TEXT_NORMS_DEFAULT = {
    "coverage":           Normalization("pass_through"),
    "correctness":        Normalization("pass_through"),
    "coherence":          Normalization("pass_through"),
    "citation_support":   Normalization("pass_through"),
    "entity_consistency": Normalization("pass_through"),
    "readability":        Normalization("band", min_val=9.0, max_val=11.0),  # FKGL target band
    "novelty":            Normalization("pass_through"),
    # Optional extended dims
    "stickiness":         Normalization("pass_through"),
}

# Normalizers for common code metrics
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
        "entity_consistency": 0.80,  # not weighted strongly but must clear bar
    },
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
    Also surfaces unmet bars and suggests next targets (the lowest-scoring normalized dims).
    """

    def __init__(
        self,
        templates: Optional[Dict[str, GoalTemplate]] = None,
        judge: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
    ):
        """
        judge (optional): function(row) -> extra dimension dict (e.g., external LLM judge).
        Any returned dims will be merged (and normalized via template norms if targeted).
        """
        self.templates = templates or DEFAULT_TEMPLATES
        self.judge = judge

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
          "unmet": [dim...],
          "raw": vpm_row
        }
        """
        key = f"{kind}/{goal}"
        if key not in self.templates:
            raise KeyError(f"Unknown goal '{goal}' for kind '{kind}'. Available: {self.available(kind=kind)}")

        tpl = self.templates[key]
        dims = dict(vpm_row)

        # Pull in optional judge signals
        if self.judge:
            try:
                extra = self.judge(vpm_row) or {}
                dims.update(extra)
            except Exception:
                pass

        score, normed = tpl.score(dims)
        unmet = tpl.unmet(dims, hysteresis=hysteresis)
        return {
            "goal": goal,
            "score": round(float(score), 4),
            "normalized": {k: round(float(v), 4) for k, v in normed.items()},
            "unmet": unmet,
            "raw": vpm_row,
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
        # Only consider dimensions present in weights
        pairs = [(k, normed.get(k, 0.0)) for k in tpl.weights.keys()]
        pairs.sort(key=lambda x: x[1])
        return pairs[:top_k]


# ----------------------------- I/O helpers -----------------------------

def load_yaml(path: str | None) -> Dict[str, Any]:
    """
    Load custom templates from a YAML file.
    Format:
      text:
        blog_general:
          weights: {coverage: 0.2, correctness: 0.2, ...}
          norms:
            readability: {kind: band, min_val: 9.0, max_val: 11.0}
          min_bar: {coverage: 0.75, coherence: 0.7}
      code:
        strict_ci:
          ...
    """
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not installed; cannot load YAML templates.")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_templates_from_yaml(config: Dict[str, Any]) -> Dict[str, GoalTemplate]:
    """
    Construct GoalTemplate mapping from a parsed YAML dict.
    """
    out: Dict[str, GoalTemplate] = dict(DEFAULT_TEMPLATES)  # start from defaults
    for kind in ("text", "code"):
        if kind not in config:
            continue
        for name, body in (config.get(kind) or {}).items():
            weights = body.get("weights", {})
            min_bar = body.get("min_bar", {})
            norms_in = body.get("norms", {})
            norms: Dict[str, Normalization] = {}
            for k, ncfg in (norms_in or {}).items():
                if isinstance(ncfg, dict):
                    norms[k] = Normalization(
                        kind=ncfg.get("kind", "pass_through"),
                        min_val=float(ncfg.get("min_val", 0.0)),
                        max_val=float(ncfg.get("max_val", 1.0)),
                    )
            tpl = GoalTemplate(name=name, kind=kind, weights=weights, norms=norms or (TEXT_NORMS_DEFAULT if kind=="text" else CODE_NORMS_DEFAULT), min_bar=min_bar)
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
    print("[suggest targets]", gs.suggest_targets("text", "academic_summary", text_row, top_k=3))
