# stephanie/analysis/skill_reporter.py
from __future__ import annotations


import os
from typing import Any, Dict, Optional

import numpy as np
from jinja2 import Template

from stephanie.zero.vpm_builder import CaseBookVPMBuilder


class SkillReporter:
    def __init__(self, tokenizer, logger=None):
        self.builder = CaseBookVPMBuilder(tokenizer, metrics=["sicql", "ebt", "llm"])
        self.logger = logger or (lambda m: print(m))

    def generate_enhancement_report(
        self,
        model,
        filter_obj,
        casebook,
        base_vpm: np.ndarray | None,
        enhanced_vpm: np.ndarray | None,
        base_metrics: Dict[str, Any] | None,
        enhanced_metrics: Dict[str, Any] | None,
        output_dir: str,
        alpha: float,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        cbname = casebook.name

        if base_vpm is None:
            base_vpm = self.builder.build(casebook, model)
        self.builder.save_image(base_vpm, f"{output_dir}/{cbname}_base_vpm.png", title=f"Base VPM - {cbname}")

        if enhanced_vpm is not None:
            self.builder.save_image(enhanced_vpm, f"{output_dir}/{cbname}_enhanced_vpm.png",
                                    title=f"Enhanced VPM (α={alpha}) - {filter_obj.id}")
            self.builder.save_image(np.clip(enhanced_vpm - base_vpm, 0, 1),
                                    f"{output_dir}/{cbname}_residual_vpm.png",
                                    title=f"Applied Residual (α={alpha}) - {filter_obj.id}")

        tmpl = Template("""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Filter Report {{ filter_id }}</title>
<style>body{font-family:Arial;margin:20px} .sec{margin-bottom:24px} table{border-collapse:collapse}
td,th{border:1px solid #ddd;padding:8px}</style></head>
<body>
<h1>Filter Enhancement Report</h1>
<p><b>Filter:</b> {{ filter_id }} ({{ domain }})<br/>
<b>CaseBook:</b> {{ casebook_name }}<br/>
<b>Alpha:</b> {{ alpha }}</p>

<div class="sec"><h2>VPM Comparison</h2>
<h3>Base</h3><img src="{{ casebook_name }}_base_vpm.png"/>
{% if has_enhanced %}
<h3>Enhanced</h3><img src="{{ casebook_name }}_enhanced_vpm.png"/>
<h3>Residual</h3><img src="{{ casebook_name }}_residual_vpm.png"/>
{% endif %}
</div>

{% if improvements %}
<div class="sec"><h2>Performance Impact</h2>
<table><tr><th>Metric</th><th>Base</th><th>Enhanced</th><th>Δ</th><th>%</th></tr>
{% for m in improvements %}
<tr class="{{ 'improved' if m.delta>=0 else 'declined' }}">
<td>{{ m.name }}</td>
<td>{{ "%.4f"|format(m.base) }}</td>
<td>{{ "%.4f"|format(m.enhanced) }}</td>
<td>{{ "%.4f"|format(m.delta) }}</td>
<td>{{ "%.2f%%"|format(m.pct) }}</td>
</tr>
{% endfor %}
</table>
</div>
{% endif %}
</body></html>
""")
        improvements = []
        if base_metrics and enhanced_metrics and "summary" in base_metrics and "summary" in enhanced_metrics:
            for k, v in base_metrics["summary"].items():
                if k in enhanced_metrics["summary"]:
                    b = float(v); e = float(enhanced_metrics["summary"][k])
                    d = e - b; pct = (d / (b + 1e-8)) * 100.0
                    improvements.append({"name": k, "base": b, "enhanced": e, "delta": d, "pct": pct})

        html = tmpl.render(
            filter_id=filter_obj.id,
            domain=filter_obj.domain,
            casebook_name=cbname,
            alpha=alpha,
            has_enhanced=enhanced_vpm is not None,
            improvements=improvements
        )
        out = os.path.join(output_dir, f"{cbname}_filter_report.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        self.logger(f"Report written: {out}")
        return out
