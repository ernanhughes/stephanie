# stephanie/tools/metric_filter_group_tool.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
from stephanie.scoring.metrics.metric_filter import MetricFilter

log = logging.getLogger(__name__)

class MetricFilterGroupTool:
    name = "metric_filter_group"

    def __init__(self, cfg: Dict, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        self.mf = MetricFilter(
            k=int(self.cfg.get("top_k", 100)),
            dup_threshold=float(self.cfg.get("dup_threshold", 0.995)),
            min_variance=float(self.cfg.get("min_variance", 1e-8)),
            normalize=bool(self.cfg.get("normalize", True)),
            include_patterns=self.cfg.get("include", []),
            exclude_patterns=self.cfg.get("exclude", []),
            alias_strip=True,
        )

    async def apply(self, rows: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not rows:
            return rows

        # Optional supervision for filter
        labels = None
        if self.cfg.get("labels_from") == "rollout.is_correct":
            tmp = []
            any_lab = False
            for r in rows:
                ic = (r.get("rollout") or {}).get("is_correct")
                if ic is None:
                    tmp.append(0)
                else:
                    any_lab = True
                    tmp.append(1 if ic else 0)
            labels = tmp if any_lab else None

        Xf, names_f, report = self.mf.run(rows, labels=labels)

        # Write filtered vectors back into rows
        for i, r in enumerate(rows):
            r["filtered_metric_names"] = names_f
            r["filtered_metric_values"] = Xf[i].tolist()

        # Persist chosen set for the run (optional)
        group_run_id = context.get("run_id") or context.get("pipeline_run_id")
        if group_run_id and getattr(self.memory, "metrics", None):
            try:
                meta = {
                    "filtered_metric_names": names_f,
                    "metric_filter_report": report.to_dict(),
                }
                # You may have helper methods—adapt if needed:
                self.memory.metrics.upsert_group_meta(group_run_id, meta)
            except Exception as e:
                log.warning("[MetricFilterGroupTool] persist skipped: %s", e)

        return rows
