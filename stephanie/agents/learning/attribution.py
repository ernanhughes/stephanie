# stephanie/agents/learning/attribution.py
from __future__ import annotations
from typing import Dict, Any, List
import time


class AttributionTracker:
    """Tracks knowledge contributions and their ablation impact."""

    def __init__(self):
        self.contributions: Dict[str, Dict[str, Any]] = {}
        self.ablation_results: List[Dict[str, Any]] = []

    def record_contribution(self, key: str, data: Dict[str, Any]):
        self.contributions.setdefault(
            key,
            {"key": key, "data": data, "used_in": [], "ablation_impact": None},
        )

    def mark_used(self, key: str, section_id: str, metrics: Dict[str, float]):
        c = self.contributions.get(key)
        if not c:
            return
        c["used_in"].append(
            {"section_id": section_id, "metrics": metrics, "ts": time.time()}
        )

    def record_ablation(
        self,
        key: str,
        with_metrics: Dict[str, float],
        without_metrics: Dict[str, float],
    ):
        c = self.contributions.get(key)
        if not c:
            return
        delta = {
            "overall": with_metrics["overall"] - without_metrics["overall"],
            "knowledge_score": with_metrics["knowledge_score"]
            - without_metrics["knowledge_score"],
            "grounding": with_metrics["grounding"]
            - without_metrics["grounding"],
        }
        impact = {
            "with": with_metrics,
            "without": without_metrics,
            "delta": delta,
            "ts": time.time(),
        }
        c["ablation_impact"] = impact
        self.ablation_results.append(
            {"key": key, "contribution": c, "delta": delta}
        )

    def get_significant_contributions(
        self, min_impact: float = 0.03
    ) -> List[Dict[str, Any]]:
        return [
            r
            for r in self.ablation_results
            if r["delta"]["overall"] >= min_impact
        ]

    def evidence_md(self) -> str:
        sig = self.get_significant_contributions()
        if not sig:
            return "**No significant ablation impacts yet.** Try enabling ablations on high-impact supports."
        avg = sum(r["delta"]["overall"] for r in sig) / max(1, len(sig))
        lines = [
            "## 🔬 Ablation Evidence (Applied Knowledge)",
            f"- Significant contributions: **{len(sig)}**",
            f"- Avg Δ verification score (with − without): **{avg:+.3f}**",
            "",
        ]
        for i, r in enumerate(
            sorted(sig, key=lambda x: x["delta"]["overall"], reverse=True)[:3],
            1,
        ):
            d = r["delta"]
            data = r["contribution"]["data"]
            lines += [
                f"### Example #{i}",
                f"- Source: `{data.get('source')}` (id={data.get('id')})",
                f"- Context: {data.get('retrieval_context', '')}",
                f"- Excerpt: “{(data.get('section_text') or '')[:180]}…”",
                f"- Impact: overall {d['overall']:+.3f}, knowledge {d['knowledge_score']:+.3f}, grounding {d['grounding']:+.3f}",
                "",
            ]
        return "\n".join(lines)
