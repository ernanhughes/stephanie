from __future__ import annotations

import fnmatch
import logging
from typing import Any, Dict, List, Optional

from stephanie.constants import PIPELINE_RUN_ID
from stephanie.scoring.metrics.core_metrics import CORE_METRIC_MAPPING
from stephanie.scoring.metrics.metric_filter import MetricFilter
from stephanie.utils.hash_utils import hash_list

log = logging.getLogger(__name__)

def _casefold(s: str) -> str:
    return s.casefold() if hasattr(s, "casefold") else s.lower()

def _match_any(name: str, patterns: List[str]) -> bool:
    ncf = _casefold(name)
    for p in (patterns or []):
        if fnmatch.fnmatch(ncf, _casefold(p)):
            return True
    return False

class MetricFilterGroupTool:
    name = "metric_filter_group"

    def __init__(self, cfg: Dict, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        self.short_circuit_if_locked = bool(self.cfg.get("short_circuit_if_locked", True))
        self.always_include: List[str] = list(self.cfg.get("always_include", []) or [])

        if self.cfg.get("include_visicalc_core", False):
            core = list(self.cfg.get("visicalc_core_names", []) or [])
            # Ensure we're using canonical names
            core = [CORE_METRIC_MAPPING.get(name.lower(), name) for name in core]
            self.always_include.extend([c for c in core if c not in self.always_include])

        self.mf = MetricFilter(
            k=int(self.cfg.get("top_k", 100)),
            dup_threshold=float(self.cfg.get("dup_threshold", 0.995)),
            min_variance=float(self.cfg.get("min_variance", 1e-8)),
            normalize=bool(self.cfg.get("normalize", True)),
            include_patterns=self.cfg.get("include", []),
            exclude_patterns=self.cfg.get("exclude", []),
            alias_strip=bool(self.cfg.get("alias_strip", True)),
        )

    async def apply(self, rows: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not rows:
            return rows
    
        run_id = context.get(PIPELINE_RUN_ID)

        # Short-circuit if locked kept columns exist
        if self.short_circuit_if_locked and run_id:
            kept_locked = self.memory.metrics.get_kept_columns(run_id)
            if kept_locked:
                dig = hash_list(kept_locked)
                for i, r in enumerate(rows):
                    mapping = dict(zip(r.get("metrics_columns") or [], r.get("metrics_values") or []))
                    r["filtered_metric_names"] = list(kept_locked)
                    r["filtered_metric_values"] = [float(mapping.get(k, 0.0)) for k in kept_locked]
                try:
                    self.memory.metrics.upsert_group_meta(
                        run_id,
                        {
                            "metric_filter_summary": {
                                "status": "short_circuit",
                                "kept_count": len(kept_locked),
                                "kept_digest": dig,
                                "source": "MetricStore.get_kept_columns",
                            }
                        },
                    )
                except Exception:
                    pass
                return rows

        # Build union names across rows for diagnostics
        all_cols: list[str] = []
        for r in rows:
            cols = r.get("metrics_columns") or []
            if cols:
                all_cols.extend(cols)
        uniq_cols = list(dict.fromkeys(all_cols))

        # Diagnostic preview for patterns
        include_pats = list(self.mf.include_patterns or [])
        exclude_pats = list(self.mf.exclude_patterns or [])
        dropped_by_exclude = [n for n in uniq_cols if _match_any(n, exclude_pats)]
        not_included = [n for n in uniq_cols if include_pats and not _match_any(n, include_pats)]

        Xf, names_f, report = self.mf.run(rows, labels=self._labels_from(rows))

        # Always-include enforcement
        have = set(names_f)
        forced_in = []
        for col in self.always_include:
            if col not in have:
                names_f.append(col)
                forced_in.append(col)
                have.add(col)

        # Write filtered vectors back into rows
        for i, r in enumerate(rows):
            mapping = dict(zip(r.get("metrics_columns") or [], r.get("metrics_values") or []))
            r["filtered_metric_names"] = list(names_f)
            r["filtered_metric_values"] = [float(mapping.get(k, 0.0)) for k in names_f]

        # Persist chosen set + rich summary
        if run_id:
            dig = hash_list(names_f)
            meta = {
                "filtered_metric_names": names_f,
                "metric_filter_report": (report.to_dict() if hasattr(report, "to_dict") else dict(report or {})),
                "metric_filter_summary": {
                    "status": "ok",
                    "kept_count": len(names_f),
                    "kept_digest": dig,
                    "forced_in": forced_in[:20],
                    "patterns": {
                        "include": include_pats,
                        "exclude": exclude_pats,
                        "preview": {
                            "exclude_hits": len(dropped_by_exclude),
                            "not_included": len(not_included),
                            "exclude_examples": dropped_by_exclude[:20],
                            "not_included_examples": not_included[:20],
                        },
                    },
                },
            }
            self.memory.metrics.upsert_group_meta(run_id, meta)
        return rows

    def _labels_from(self, rows: List[Dict[str, Any]]) -> Optional[List[int]]:
        if self.cfg.get("labels_from") == "rollout.is_correct":
            tmp, any_lab = [], False
            for r in rows:
                ic = (r.get("rollout") or {}).get("is_correct")
                if ic is None:
                    tmp.append(0)
                else:
                    any_lab = True
                    tmp.append(1 if ic else 0)
            return tmp if any_lab else None
        return None
