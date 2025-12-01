# stephanie/scoring/metrics/feature/metric_filter_group_feature.py
from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from stephanie.constants import PIPELINE_RUN_ID
from stephanie.scoring.metrics.core_metrics import CORE_METRIC_MAPPING
from stephanie.scoring.metrics.feature.base_group_feature import BaseGroupFeature
from stephanie.scoring.metrics.feature.feature_report import FeatureReport
from stephanie.scoring.metrics.metric_filter import MetricFilter
from stephanie.scoring.metrics.metric_filter_explain import \
    write_metric_filter_explain
from stephanie.utils.hash_utils import hash_list

log = logging.getLogger(__name__)

def _casefold(s: str) -> str:
    return s.casefold() if hasattr(s, "casefold") else s.lower()

def _match_any(name: str, patterns: List[str]) -> bool:
    """Case-insensitive glob check."""
    ncf = _casefold(name)
    for p in (patterns or []):
        if fnmatch.fnmatch(ncf, _casefold(p)):
            return True
    return False

def _project_rows_to_names(rows: List[Dict[str, Any]], kept: List[str]) -> None:
    """Rewrite each row to the kept column set (zeros if missing)."""
    for r in rows:
        cols = r.get("metrics_columns") or []
        vals = r.get("metrics_values") or []
        mapping = dict(zip(cols, vals)) if cols and vals else {}
        new_vals = [float(mapping.get(k, 0.0)) for k in kept]
        r["metrics_columns"] = list(kept)
        r["metrics_values"] = new_vals
        r["metrics_vector"] = {k: v for k, v in zip(kept, new_vals)}

class MetricFilterGroupFeature(BaseGroupFeature):
    name = "metric_filter"
    requires: list[str] = []

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.filter = MetricFilter(
            k=int(cfg.get("top_k", 100)),
            dup_threshold=float(cfg.get("dup_threshold", 0.995)),
            min_variance=float(cfg.get("min_variance", 1e-8)),
            normalize=bool(cfg.get("normalize", True)),
            include_patterns=list(cfg.get("include", []) or []),
            exclude_patterns=list(cfg.get("exclude", []) or []),
            alias_strip=bool(cfg.get("alias_strip", True)),
        )

        # Behavior knobs
        self.short_circuit_if_locked = bool(cfg.get("short_circuit_if_locked", True))
        self.always_include: List[str] = list(cfg.get("always_include", []) or [])

        # Optional “core” visicalc columns you always want present
        if cfg.get("include_visicalc_core", False):
            core = list(cfg.get("visicalc_core_names", []) or [])
            # common default if none provided
            if not core:
                core = [
                    "frontier_util","stability","middle_dip","std_dev",
                    "sparsity","entropy","trend","HRM.aggregate"
                ]
            # Ensure we're using canonical names
            core = [CORE_METRIC_MAPPING.get(name.lower(), name) for name in core]
            self.always_include.extend([c for c in core if c not in self.always_include])


        # Internal
        self._last_summary: Dict[str, Any] | None = None
        self._last_selected: List[str] | None = None

    # ---------- Main ----------
    async def apply(self, rows: list[dict], context: dict) -> list[dict]:
        if not self.enabled or not rows:
            return rows

        run_id = context.get(PIPELINE_RUN_ID)
        run_dir = Path(context.get("run_dir") or f"runs/critic/{run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)

        # (A) Short-circuit if kept already locked in DB
        if self.short_circuit_if_locked and getattr(self.memory, "metrics", None):
            try:
                kept_locked = self.memory.metrics.get_kept_columns(run_id)
            except Exception:
                kept_locked = None
            if kept_locked:
                log.info("[MetricFilterGroupFeature] short-circuit: using DB-locked %d kept columns", len(kept_locked))
                _project_rows_to_names(rows, kept_locked)
                dig = hash_list(kept_locked)
                summary = {
                    "status": "short_circuit",
                    "reason": "DB-locked kept columns found",
                    "kept_count": len(kept_locked),
                    "kept_digest": dig,
                    "source": "MetricStore.get_kept_columns",
                }
                self._last_summary = summary
                self._last_selected = list(kept_locked)
                self._persist_summary(context, summary, kept_locked)
                # still write lock file for reproducibility
                (run_dir / "kept_features.txt").write_text("\n".join(kept_locked), encoding="utf-8")
                return rows

        # (B) Build column universe
        all_cols: list[str] = []
        for r in rows:
            cols = r.get("metrics_columns") or []
            if cols:
                all_cols.extend(cols)
        if not all_cols:
            summary = {
                "status": "no_cols",
                "reason": "no metric columns present on any row",
                "total_rows": len(rows),
            }
            context["metric_filter_summary"] = summary
            self._last_summary = summary
            self._persist_summary(context, summary, kept=None)
            return rows

        uniq_cols = list(dict.fromkeys(all_cols))
        name_to_idx = {n: i for i, n in enumerate(uniq_cols)}
        X = np.asarray(
            [[float(dict(zip(r.get("metrics_columns") or [], r.get("metrics_values") or [])).get(n, 0.0)) for n in uniq_cols]
             for r in rows],
            dtype=np.float32
        )

        names_union = list(uniq_cols)
        # keep original X (pre-selection) for diagnostics
        X_union = X.copy()

        # (C) Pre-diagnostics for pattern filters (case-insensitive)
        include_pats = list(self.filter.include_patterns or [])
        exclude_pats = list(self.filter.exclude_patterns or [])
        dropped_by_exclude = [n for n in uniq_cols if _match_any(n, exclude_pats)]
        # If include_patterns specified, anything not matching include is at-risk
        if include_pats:
            not_included = [n for n in uniq_cols if not _match_any(n, include_pats)]
        else:
            not_included = []
        pattern_drops_preview = {
            "would_drop_by_exclude": dropped_by_exclude[:20],
            "would_drop_by_not_included": not_included[:20],
            "counts": {
                "exclude_hits": len(dropped_by_exclude),
                "not_included": len(not_included),
            },
            "patterns": {
                "include": include_pats,
                "exclude": exclude_pats,
            }
        }

        # (D) Resolve always-include/core metrics against the universe
        core_names = self._resolve_always_include(uniq_cols)
        core_set = set(core_names)

        # (E) Run the metric filter selection on NON-core columns only
        candidate_cols = [n for n in uniq_cols if n not in core_set]
        selected_names: List[str] = []
        if candidate_cols:
            # Build candidate matrix
            cand_idx = [name_to_idx[n] for n in candidate_cols]
            X_cand = X[:, cand_idx]

            keep_mask, selected_subset = self.filter.select(
                candidate_cols, X_cand, labels=context.get("labels")
            )
            # ensure list
            selected_names = list(selected_subset or [])
        else:
            keep_mask = None
            selected_names = []

        # (F) If everything got dropped, fall back gracefully
        if not core_names and not selected_names:
            log.warning(
                "[MetricFilterGroupFeature] selection empty; falling back to all columns"
            )
            selected_names = list(uniq_cols)

        # (G) Compose final kept set: core first, then filtered candidates
        final_selected = core_names + [n for n in selected_names if n not in core_set]

        # Track which came from always_include for the summary
        forced_in = [n for n in core_names if n not in candidate_cols]
        if not forced_in and core_names:
            # core_names were present in candidate_cols but bypassed the filter by design,
            # so treat them as forced as well.
            forced_in = list(core_names)

        # (H) Categorize drops for the report
        selected_set = set(final_selected)
        actually_dropped = [n for n in uniq_cols if n not in selected_set]

        dropped_by_pattern = [n for n in actually_dropped if (_match_any(n, exclude_pats) or (include_pats and not _match_any(n, include_pats)))]
        dropped_by_simvar = [n for n in actually_dropped if n not in dropped_by_pattern]

        # (I) Project rows and persist locks + summary
        _project_rows_to_names(rows, final_selected)

        # (J) Write lock file
        (run_dir / "kept_features.txt").write_text("\n".join(final_selected), encoding="utf-8")

        digest = hash_list(final_selected)
        summary = {
            "status": "ok",
            "kept_count": len(final_selected),
            "total_raw": len(uniq_cols),
            "kept_digest": digest,
            "forced_in": forced_in[:20],
            "drops": {
                "pattern": {
                    "count": len(dropped_by_pattern),
                    "examples": dropped_by_pattern[:20],
                    "patterns": {"include": include_pats, "exclude": exclude_pats},
                    "preview_counts": pattern_drops_preview["counts"],
                },
                "similarity_or_variance": {
                    "count": len(dropped_by_simvar),
                    "examples": dropped_by_simvar[:20],
                },
            },
            "samples": {
                "kept_head": final_selected[:20],
                "raw_head": uniq_cols[:20],
            },
        }
        self._last_summary = summary
        self._last_selected = list(final_selected)

        # (K) Persist to MetricStore
        self._persist_summary(context, summary, kept=final_selected)

        log.info("[MetricFilterGroupFeature] kept %d of %d metrics (forced +%d) digest=%s",
                 summary["kept_count"], summary["total_raw"], len(forced_in), digest)


        # (L) Write detailed explain report (MD + CSV + quick figs)
        try:
            # ---- Build diagnostics with safe fallbacks ----
            # union of columns seen pre-filter
            names_union = list(uniq_cols)

            # labels (optional supervision)
            labels = context.get("labels")

            # Attempt to pull diagnostics from MetricFilter, else fall back
            # dup_pairs: List[Tuple[kept_name, dropped_name, similarity]]
            dup_pairs = []
            if hasattr(self.filter, "last_dup_pairs") and self.filter.last_dup_pairs:
                dup_pairs = list(self.filter.last_dup_pairs)

            # indices of columns in names_union flagged as non-finite / low variance
            nonfinite_idx = []
            if hasattr(self.filter, "last_nonfinite_idx") and self.filter.last_nonfinite_idx is not None:
                nonfinite_idx = list(self.filter.last_nonfinite_idx)

            lowvar_idx = []
            if hasattr(self.filter, "last_lowvar_idx") and self.filter.last_lowvar_idx is not None:
                lowvar_idx = list(self.filter.last_lowvar_idx)

            # rank method (string label for the report)
            rank_method = getattr(self.filter, "rank_method", "variance+similarity")

            # whether normalization was used
            normalize_used = bool(getattr(self.filter, "normalize", True))

            # always-include list if you wired it on the feature (optional)
            always_include = list(getattr(self, "always_include", []) or [])

            # snapshot the filter config for reproducibility
            cfg_snapshot = {
                "k": getattr(self.filter, "k", None),
                "dup_threshold": getattr(self.filter, "dup_threshold", None),
                "min_variance": getattr(self.filter, "min_variance", None),
                "normalize": normalize_used,
                "include": list(getattr(self.filter, "include_patterns", []) or []),
                "exclude": list(getattr(self.filter, "exclude_patterns", []) or []),
                "alias_strip": getattr(self.filter, "alias_strip", True),
                "rank_method": rank_method,
            }

            write_metric_filter_explain(
                run_dir=run_dir,
                names_union=names_union,
                rows=rows,                              # current (post-application) rows are fine: we rebuild union X inside writer
                kept_names=list(final_selected),
                dup_pairs=list(dup_pairs),
                nonfinite_idx=list(nonfinite_idx),
                lowvar_idx=list(lowvar_idx),
                labels=labels,
                normalize_used=normalize_used,
                rank_method=str(rank_method),
                cfg_snapshot=cfg_snapshot,
                md_filename="metric_filter_explain.md",
                csv_filename="metric_filter_explain.csv",
                always_include=always_include,
            )
        except Exception as e:
            log.warning("[MetricFilterGroupFeature] explain writer failed: %s", e)

        return rows

    # ---------- Persistence ----------
    def _persist_summary(self, context: dict, summary: dict, kept: List[str] | None) -> None:
        try:
            run_id = context.get("pipeline_run_id", "unknown")
            patch = {"metric_filter_summary": summary}
            if kept is not None:
                patch["metric_filter"] = {
                    "kept_columns": list(kept),
                    "n_kept": len(kept),
                    "kept_digest": hash_list(kept),
                }
            self.memory.metrics.upsert_group_meta(run_id=run_id, patch=patch)
        except Exception as e:
            self._warn(f"[MetricFilterGroupFeature] persist skipped: {e}")

    # ---------- Feature report hook ----------
    def report(self) -> FeatureReport:
        if not self._last_summary:
            return FeatureReport(
                name=self.name, kind="group", ok=True, quality=None,
                summary="no-op", details={}, warnings=[]
            )
        kept = self._last_summary.get("kept_count", 0)
        total = self._last_summary.get("total_raw", 0)
        quality = float(kept / max(total, 1)) if total else None
        ok = kept > 0
        return FeatureReport(
            name=self.name,
            kind="group",
            ok=ok,
            quality=quality,
            summary=f"kept {kept} of {total} metrics; digest={self._last_summary.get('kept_digest')}",
            details=self._last_summary,
            warnings=[],
        )

    def _strip_alias(self, name: str) -> str:
        """
        Alias-stripping logic consistent with MetricFilter.alias_strip:
        'Visi.frontier_util' -> 'frontier_util'
        """
        return name.split(".", 1)[1] if "." in name else name

    def _resolve_always_include(self, names_union: List[str]) -> List[str]:
        """
        Map self.always_include (which may be bare names or alias-prefixed)
        to actual column names present in names_union.

        - If an entry already exists in names_union, use it.
        - Else try to match by alias-stripped stem.
        - Log a warning for anything that can't be resolved.
        """
        if not self.always_include:
            return []

        union_set = set(names_union)

        # Build stem -> [full names] index from the universe
        stem_to_full: Dict[str, List[str]] = {}
        for n in names_union:
            stem = self._strip_alias(n).casefold()
            stem_to_full.setdefault(stem, []).append(n)

        resolved: List[str] = []
        for raw in self.always_include:
            # 1) Exact match
            if raw in union_set:
                resolved.append(raw)
                continue

            # 2) Match by stem
            stem = self._strip_alias(raw).casefold()
            candidates = stem_to_full.get(stem)
            if candidates:
                # Pick a stable representative (first occurrence)
                resolved.append(candidates[0])
            else:
                log.warning(
                    "[MetricFilterGroupFeature] always_include '%s' not found in metric universe",
                    raw,
                )

        # De-duplicate while preserving order
        seen: set[str] = set()
        out: List[str] = []
        for n in resolved:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _order_core_first(self, selected_names: List[str], core_names: List[str]) -> List[str]:
        """
        Ensure core_names appear first (in the order given), and preserve
        the relative ordering of the remaining names.
        """
        core_set = set(core_names)
        core_block: List[str] = []
        other_block: List[str] = []

        for n in selected_names:
            if n in core_set:
                core_block.append(n)
            else:
                other_block.append(n)

        # Make sure any core_names not in selected_names (but resolved from universe)
        # are still included at the very front, in the order of core_names.
        final_core: List[str] = []
        seen = set()
        for n in core_names:
            if n not in seen:
                seen.add(n)
                final_core.append(n)
        # Now merge with any that came from selected_names but weren't in core_names order
        for n in core_block:
            if n not in seen:
                seen.add(n)
                final_core.append(n)

        return final_core + [n for n in other_block if n not in seen]
