# stephanie/components/gap/processors/report.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


class ReportBuilder:
    """Create a lightweight Markdown report that links all key artifacts."""

    def __init__(self, config, container, logger):
        self.config = config
        self.container = container
        self.logger = logger

    async def build(
        self,
        run_id: str,
        analysis_out: Dict[str, Any],
        scoring_out: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Build a markdown report for a GAP run.

        Parameters
        ----------
        run_id : str
            GAP run identifier.
        analysis_out : Dict[str, Any]
            Output map from the analysis stage (SCM visuals, frontier, PHOS, topology, etc.).
        scoring_out : Dict[str, Any]
            Output map from the scoring stage (matrices, GIFs, EpistemicGuard assets, risk numbers).

        Returns
        -------
        Dict[str, str]
            {"report_path": "<absolute-path-to-markdown>"}
        """
        storage = self.container.get("storage")
        root: Path = storage.base_dir / run_id
        reports_dir: Path = root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path: Path = reports_dir / "report.md"

        # Optional: router calibration summary, if present
        calib_summary = root / "metrics" / "routing_summary.json"
        usage_rate = avg_mae = None
        if calib_summary.exists():
            try:
                d = json.loads(calib_summary.read_text(encoding="utf-8"))
                usage_rate = d.get("usage_rate")
                avg_mae = d.get("avg_mae_vs_hrm")
            except Exception:
                pass

        # Pull common visuals from analysis
        visuals = analysis_out.get("scm_visuals", {}) if isinstance(analysis_out, dict) else {}
        frontier = analysis_out.get("frontier", {}) if isinstance(analysis_out, dict) else {}
        delta    = analysis_out.get("delta_analysis", {}) if isinstance(analysis_out, dict) else {}
        phos     = analysis_out.get("phos", {}) if isinstance(analysis_out, dict) else {}
        topo     = analysis_out.get("topology", {}) if isinstance(analysis_out, dict) else {}

        vpm_hrm_gif  = scoring_out.get("hrm_gif")
        vpm_tiny_gif = scoring_out.get("tiny_gif")

        # Epistemic Guard (scoring_out["eg"] is a dict with lists: hal_badges, vpm_stacks, truth_gifs)
        eg: Dict[str, Any] = scoring_out.get("eg", {}) if isinstance(scoring_out, dict) else {}

        # Canonical risk numbers (single source of truth from scoring pass)
        risk_val = scoring_out.get("risk")
        risk_thresholds = scoring_out.get("risk_thresholds")
        route = scoring_out.get("route")

        def _img(path: str | None) -> str:
            if not path:
                return ""
            # prefer relative path under run dir if applicable (keeps report portable)
            try:
                p = Path(path)
                rel = p if str(p).startswith(str(root)) else p
                return f"![viz]({rel})"
            except Exception:
                return f"![viz]({path})"

        def _bulleted_files(title: str, items: List[str], exts: tuple[str, ...]) -> List[str]:
            if not items:
                return []
            out = [f"## {title}"]
            for v in items:
                if isinstance(v, str) and v.lower().endswith(exts):
                    out.append(f"- {v}")
            out.append("")
            return out

        lines: List[str] = []
        lines.append(f"# GAP Run Report – `{run_id}`")
        lines.append("")
        lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}_")
        lines.append("")

        # Router summary (if any)
        if usage_rate is not None and avg_mae is not None:
            lines.append(f"**Router usage**: {usage_rate:.3f} | **Avg MAE vs HRM**: {avg_mae:.3f}")
            lines.append("")

        # VPM timelines from scoring
        if vpm_hrm_gif or vpm_tiny_gif:
            lines.append("## VPM Timelines")
            if vpm_hrm_gif:
                lines.append(f"- HRM: {vpm_hrm_gif}")
            if vpm_tiny_gif:
                lines.append(f"- Tiny: {vpm_tiny_gif}")
            lines.append("")

        # SCM visuals (analysis)
        lines.append("## SCM Visuals")
        if isinstance(visuals, dict):
            core5 = visuals.get("scm_core5_radar")
            delta_bar = visuals.get("scm_delta_bar")
            agg_scatter = visuals.get("scm_aggregate_scatter")
            if core5:       lines.append(_img(core5))
            if delta_bar:   lines.append(_img(delta_bar))
            if agg_scatter: lines.append(_img(agg_scatter))
        lines.append("")

        # Frontier & Δ
        lines.append("## Frontier & Δ")
        if isinstance(delta, dict):
            if "delta_abs_heat" in delta:
                lines.append(_img(delta["delta_abs_heat"]))
            if "frontier_scatter" in frontier:
                lines.append(_img(frontier["frontier_scatter"]))
        lines.append("")

        # PHOS block (links only – images already written by zeromodel helper)
        lines.append("## PHOS")
        if isinstance(phos, dict):
            for k, v in phos.items():
                if isinstance(v, str) and (v.lower().endswith(".png") or v.lower().endswith(".gif")):
                    lines.append(f"- {k}: {v}")
        lines.append("")

        # Topology (if present)
        lines.append("## Topology")
        if isinstance(topo, dict):
            for k, v in topo.items():
                if isinstance(v, str) and (v.lower().endswith(".png") or v.lower().endswith(".svg") or v.lower().endswith(".gif")):
                    lines.append(f"- {k}: {v}")
        lines.append("")

        # ---------- Epistemic Guard (EG) ----------
        # This section consolidates the canonical risk values (from the predictor-driven scoring)
        # and links all EG visual artifacts: risk badges, truth GIFs, and VPM stacks.
        lines.append("## Epistemic Guard")
        if risk_val is not None or risk_thresholds or route:
            low, high = (None, None)
            if isinstance(risk_thresholds, (list, tuple)) and len(risk_thresholds) == 2:
                low, high = risk_thresholds
            pill = _format_risk_pill(risk_val, low, high, route)
            lines.append(pill)
            lines.append("")
        else:
            lines.append("_No risk values recorded in scoring artifacts._")
            lines.append("")

        if isinstance(eg, dict) and eg:
            # Badge images (usually small square signal images)
            badges = [b for b in eg.get("hal_badges", []) if isinstance(b, str)]
            lines += _bulleted_files("EG Badges", badges, (".png", ".svg", ".jpg", ".jpeg", ".webp"))

            # Truth GIFs (temporal hallucination / topology evolution)
            gifs = [g for g in eg.get("truth_gifs", []) if isinstance(g, str)]
            lines += _bulleted_files("EG Truth GIFs", gifs, (".gif",))

            # VPM stacks (npz for downstream training)
            stacks = [n for n in eg.get("vpm_stacks", []) if isinstance(n, str)]
            if stacks:
                lines.append("## EG VPM Stacks")
                for p in stacks:
                    lines.append(f"- {p}")
                lines.append("")
        else:
            lines.append("_No Epistemic Guard assets were produced for this run._")
            lines.append("")

        # Write report
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return {"report_path": str(report_path)}


def _format_risk_pill(
    risk: Any,
    low: Any,
    high: Any,
    route: Any,
) -> str:
    """
    Render a compact, self-describing line for the EG section with risk and routing.
    Example:
      **Risk**: 0.42 (low=0.20, high=0.55) • **Route**: MEDIUM
    """
    try:
        r = float(risk) if risk is not None else None
    except Exception:
        r = None
    try:
        lo = float(low) if low is not None else None
    except Exception:
        lo = None
    try:
        hi = float(high) if high is not None else None
    except Exception:
        hi = None

    parts = []
    if r is not None:
        parts.append(f"**Risk**: {r:.2f}")
    if lo is not None and hi is not None:
        parts.append(f"(low={lo:.2f}, high={hi:.2f})")
    if route:
        parts.append(f"• **Route**: {str(route)}")
    return " ".join(parts) if parts else "_No risk numbers available._"
