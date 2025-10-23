# stephanie/components/gap/processors/report.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


class ReportBuilder:
    """Create a lightweight Markdown report that links all key artifacts."""

    def __init__(self, config, container, logger):
        self.config = config
        self.container = container
        self.logger = logger

    async def build(self, run_id: str, analysis_out: Dict[str, Any], scoring_out: Dict[str, Any]) -> Dict[str, str]:
        storage = self.container.get("gap_storage")
        root = storage.base_dir / run_id
        reports_dir = root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "report.md"

        # try to load calibration summary if exists
        calib_summary = root / "metrics" / "routing_summary.json"
        usage_rate = avg_mae = None
        if calib_summary.exists():
            try:
                d = json.loads(calib_summary.read_text(encoding="utf-8"))
                usage_rate = d.get("usage_rate")
                avg_mae = d.get("avg_mae_vs_hrm")
            except Exception:
                pass

        # images we emitted in analysis
        visuals = analysis_out.get("scm_visuals", {}) if isinstance(analysis_out, dict) else {}
        frontier = analysis_out.get("frontier", {})
        delta    = analysis_out.get("delta_analysis", {})
        phos     = analysis_out.get("phos", {})
        topo     = analysis_out.get("topology", {})

        vpm_hrm_gif  = scoring_out.get("hrm_gif")
        vpm_tiny_gif = scoring_out.get("tiny_gif")

        def _img(path: str | None) -> str:
            return f"![viz]({path})" if path else ""

        lines = []
        lines.append(f"# GAP Run Report – `{run_id}`")
        lines.append("")
        lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}_")
        lines.append("")
        if usage_rate is not None and avg_mae is not None:
            lines.append(f"**Router usage**: {usage_rate:.3f} | **Avg MAE vs HRM**: {avg_mae:.3f}")
            lines.append("")
        if vpm_hrm_gif or vpm_tiny_gif:
            lines.append("## VPM Timelines")
            if vpm_hrm_gif:  lines.append(f"- HRM: {vpm_hrm_gif}")
            if vpm_tiny_gif: lines.append(f"- Tiny: {vpm_tiny_gif}")
            lines.append("")

        lines.append("## SCM Visuals")
        if isinstance(visuals, dict):
            core5 = visuals.get("scm_core5_radar")
            delta_bar = visuals.get("scm_delta_bar")
            agg_scatter = visuals.get("scm_aggregate_scatter")
            if core5:      lines.append(_img(core5))
            if delta_bar:  lines.append(_img(delta_bar))
            if agg_scatter:lines.append(_img(agg_scatter))
        lines.append("")

        lines.append("## Frontier & Δ")
        if isinstance(delta, dict) and "delta_abs_heat" in delta:
            lines.append(_img(delta["delta_abs_heat"]))
        lines.append("")

        # PHOS block (links only – images already written by zeromodel helper)
        lines.append("## PHOS")
        if isinstance(phos, dict):
            for k, v in phos.items():
                if isinstance(v, str) and (v.endswith(".png") or v.endswith(".gif")):
                    lines.append(f"- {k}: {v}")
        lines.append("")

        # Topology (if present)
        lines.append("## Topology")
        if isinstance(topo, dict):
            for k, v in topo.items():
                if isinstance(v, str) and (v.endswith(".png") or v.endswith(".svg")):
                    lines.append(f"- {k}: {v}")
        lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return {"report_path": str(report_path)}
