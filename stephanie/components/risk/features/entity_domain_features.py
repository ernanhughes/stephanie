# stephanie/components/risk/features/entity_domain_features.py
from __future__ import annotations
from typing import Dict, Any, List
import math

def _safe_div(a: float, b: float) -> float:
    return 0.0 if b <= 0 else a / b

def build_entity_table(ner: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ents = []
    for e in ner or []:
        txt = (e.get("text") or "").strip()
        if not txt:
            continue
        ents.append({
            "text": txt,
            "type": (e.get("type") or e.get("label") or "UNK").lower(),
            "norm": txt.lower(),
        })
    return ents

async def score_entity_support(memory, domain: str, ent: Dict[str, Any]) -> Dict[str, Any]:
    # Replace with your real KB / CaseBooks lookups:
    kb = getattr(memory, "kb", None)
    kb_hits = []
    if kb and hasattr(kb, "lookup"):
        try:
            kb_hits = await kb.lookup(ent["norm"], domain=domain)
        except Exception:
            kb_hits = []
    supported = bool(kb_hits)
    contradicted = False  # wire your contradiction detector if available
    ambiguity = 0.0
    if len(kb_hits) > 1:
        ambiguity = min(1.0, math.log(1 + len(kb_hits)) / 3.0)

    return {
        "supported": supported,
        "contradicted": contradicted,
        "ambiguity": ambiguity,
        "kb_hits": len(kb_hits),
    }

async def compute_entity_metrics(memory, domain: str, ner: List[Dict[str, Any]]) -> Dict[str, float]:
    ents = build_entity_table(ner)
    n = len(ents)
    if n == 0:
        return {
            "entity_support_ratio": 1.0,
            "entity_contradiction_ratio": 0.0,
            "entity_unresolved_ratio": 0.0,
            "entity_ambiguity": 0.0,
        }
    sup = contra = unresolved = 0
    amb = 0.0
    for e in ents:
        res = await score_entity_support(memory, domain, e)
        sup += 1 if res["supported"] else 0
        contra += 1 if res["contradicted"] else 0
        unresolved += 1 if (not res["supported"] and not res["contradicted"]) else 0
        amb += float(res["ambiguity"])
    return {
        "entity_support_ratio": _safe_div(sup, n),
        "entity_contradiction_ratio": _safe_div(contra, n),
        "entity_unresolved_ratio": _safe_div(unresolved, n),
        "entity_ambiguity": amb / max(1, n),
    }

DEFAULT_WEIGHTS = {
    "programming": {"b": -0.6, "w": {
        "one_minus_support": 1.4, "contradiction": 1.8, "unresolved": 0.8,
        "ambiguity": 0.6, "one_minus_overlap": 0.8, "oov_rate": 1.2, "numeric_incons": 0.8}},
    "research": {"b": -0.4, "w": {
        "one_minus_support": 1.2, "contradiction": 1.6, "unresolved": 0.6,
        "ambiguity": 0.5, "one_minus_overlap": 0.9, "oov_rate": 0.5, "numeric_incons": 1.0}},
    "general": {"b": -0.3, "w": {
        "one_minus_support": 1.0, "contradiction": 1.4, "unresolved": 0.5,
        "ambiguity": 0.4, "one_minus_overlap": 0.7, "oov_rate": 0.4, "numeric_incons": 0.6}},
}

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def risk_from_features(domain: str, feats: Dict[str, float], weights=DEFAULT_WEIGHTS) -> float:
    cfg = weights.get(domain.lower(), weights["general"])
    b = cfg["b"]; w = cfg["w"]
    one_minus_support = 1.0 - float(feats.get("entity_support_ratio", 1.0))
    contradiction     = float(feats.get("entity_contradiction_ratio", 0.0))
    unresolved        = float(feats.get("entity_unresolved_ratio", 0.0))
    ambiguity         = float(feats.get("entity_ambiguity", 0.0))
    one_minus_overlap = 1.0 - float(feats.get("coverage_overlap", 0.0))
    oov_rate          = float(feats.get("oov_rate", 0.0))
    numeric_incons    = float(feats.get("numeric_inconsistency", 0.0))

    z = (b
         + w["one_minus_support"] * one_minus_support
         + w["contradiction"]     * contradiction
         + w["unresolved"]        * unresolved
         + w["ambiguity"]         * ambiguity
         + w["one_minus_overlap"] * one_minus_overlap
         + w["oov_rate"]          * oov_rate
         + w["numeric_incons"]    * numeric_incons)
    return max(0.0, min(1.0, _sigmoid(z)))
