# stephanie/scoring/metrics/metric_mapping.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Rule primitives
# ---------------------------------------------------------------------------

@dataclass
class MetricMappingRule:
    """
    One logical group of metrics in the final VPM / feature vector.

    Example rule:
      name: "HRM.aggregate"
      patterns: ["HRM.aggregate", "hrm.aggregate"]
      normalize: true
    """
    name: str
    patterns: List[str] = field(default_factory=list)
    normalize: bool = True
    allow_missing: bool = True


@dataclass
class MetricMappingConfig:
    """
    Full mapping configuration.

    fields:
      rules:
        - ordered list of groups to put at the *front* of the matrix
      include_unmatched:
        - whether to keep metrics that don't match any rule
      normalize_unmatched:
        - whether to min–max normalize unmatched columns as well
    """
    rules: List[MetricMappingRule] = field(default_factory=list)
    include_unmatched: bool = True
    normalize_unmatched: bool = True


# ---------------------------------------------------------------------------
# 2. Mapper
# ---------------------------------------------------------------------------

class MetricMapper:
    """
    Config-driven mapping from (metric_names, scores) → (new_names, new_scores).

    Responsibilities:
      - Select a subset of columns based on glob-style patterns
      - Order them according to preferred keys
      - Optionally apply rule-based grouping + normalization
    """

    def __init__(
        self,
        *,
        include_patterns: Optional[Sequence[str]] = None,
        exclude_patterns: Optional[Sequence[str]] = None,
        rename_map: Optional[Dict[str, str]] = None,
        preferred_keys: Optional[Sequence[str]] = None,
        rules: Optional[Sequence[Dict[str, Any]]] = None,
        include_unmatched: bool = True,
        normalize_unmatched: bool = True,
    ) -> None:
        # Simple include/exclude/rename path (used by VisiCalc)
        self.include_patterns: List[str] = list(include_patterns or [])
        self.exclude_patterns: List[str] = list(exclude_patterns or [])
        self.rename_map: Dict[str, str] = dict(rename_map or {})
        self.preferred_keys: List[str] = list(preferred_keys or [])

        # Rule-based path (used by .apply if you want it)
        raw_rules = list(rules or [])
        self.rules: List[MetricMappingRule] = [
            MetricMappingRule(
                name=str(r.get("name", f"rule_{i}")),
                patterns=list(r.get("patterns") or []),
                normalize=bool(r.get("normalize", True)),
                allow_missing=bool(r.get("allow_missing", True)),
            )
            for i, r in enumerate(raw_rules)
        ]

        self.include_unmatched: bool = bool(include_unmatched)
        self.normalize_unmatched: bool = bool(normalize_unmatched)
        self.case_insensitive: bool = True

        # ---- DEBUG: construction summary ----
        log.info(
            "MetricMapper.__init__: "
            "include_patterns=%r exclude_patterns=%r "
            "preferred_keys=%r include_unmatched=%r normalize_unmatched=%r "
            "rename_map_keys=%r rules=%d",
            self.include_patterns,
            self.exclude_patterns,
            self.preferred_keys,
            self.include_unmatched,
            self.normalize_unmatched,
            sorted(self.rename_map.keys()),
            len(self.rules),
        )
        for r in self.rules:
            log.info(
                "MetricMapper.__init__: rule name=%r patterns=%r "
                "normalize=%r allow_missing=%r",
                r.name,
                r.patterns,
                r.normalize,
                r.allow_missing,
            )

    # ------------------------------------------------------------------
    # Factory from Hydra-style config (your visicalc.yaml)
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, vis_cfg: Optional[Dict[str, Any]] = None) -> "MetricMapper":
        """
        Robust constructor from a Hydra-style visicalc config.

        Design goals:
          - If vis_cfg is None or {}, return a neutral mapper:
              * includes all columns
              * no renames
              * no rules
          - If metric_mapping is missing or badly typed, fall back to {}.
          - If fields are badly typed (string instead of list, etc.), coerce or drop.
        """
        vis_cfg = vis_cfg or {}
        if log.isEnabledFor(logging.DEBUG):
            # Don't dump the whole Hydra cfg, just the visicalc-relevant bits.
            log.info(
                "MetricMapper.from_config: vis_cfg keys=%r, raw metric_keys=%r, "
                "raw metric_mapping type=%r",
                list(vis_cfg.keys()),
                vis_cfg.get("metric_keys"),
                type(vis_cfg.get("metric_mapping")),
            )

        raw_mm = vis_cfg.get("metric_mapping") or {}
        if not isinstance(raw_mm, dict):
            log.warning(
                "MetricMapper.from_config: expected dict for metric_mapping, got %r; using empty config",
                type(raw_mm),
            )
            raw_mm = {}

        # metric_keys: preferred ordering (may be missing or wrong type)
        raw_metric_keys = vis_cfg.get("metric_keys") or []
        if not isinstance(raw_metric_keys, (list, tuple)):
            log.warning(
                "MetricMapper.from_config: metric_keys should be list/tuple, got %r; ignoring",
                type(raw_metric_keys),
            )
            raw_metric_keys = []

        def _as_str_list(val: Any) -> List[str]:
            """
            Coerce config values to a list of strings:
              - list/tuple → [str(...), ...]
              - string → [string]
              - None/empty → []
              - anything else → [].
            """
            if val is None:
                return []
            if isinstance(val, (list, tuple)):
                return [str(x) for x in val]
            if isinstance(val, str):
                return [val]
            return []

        include = _as_str_list(raw_mm.get("include"))
        exclude = _as_str_list(raw_mm.get("exclude"))

        rename = raw_mm.get("rename") or {}
        if not isinstance(rename, dict):
            log.warning(
                "MetricMapper.from_config: rename should be dict, got %r; ignoring",
                type(rename),
            )
            rename = {}

        raw_rules = raw_mm.get("rules") or []
        if not isinstance(raw_rules, (list, tuple)):
            log.warning(
                "MetricMapper.from_config: rules should be list/tuple, got %r; ignoring",
                type(raw_rules),
            )
            raw_rules = []

        # Only keep dict-like rules; everything else is ignored
        rules: List[Dict[str, Any]] = [r for r in raw_rules if isinstance(r, dict)]

        include_unmatched = bool(raw_mm.get("include_unmatched", True))
        normalize_unmatched = bool(raw_mm.get("normalize_unmatched", True))

        if log.isEnabledFor(logging.DEBUG):
            log.info(
                "MetricMapper.from_config: include=%r exclude=%r metric_keys=%r "
                "rename_keys=%r rules=%d include_unmatched=%r normalize_unmatched=%r",
                include,
                exclude,
                raw_metric_keys,
                sorted(rename.keys()),
                len(rules),
                include_unmatched,
                normalize_unmatched,
            )

        mapper = cls(
            include_patterns=include,
            exclude_patterns=exclude,
            rename_map=rename,
            preferred_keys=_as_str_list(raw_metric_keys),
            rules=rules,
            include_unmatched=include_unmatched,
            normalize_unmatched=normalize_unmatched,
        )

        return mapper

    # Optional sugar: explicit identity mapper if you ever want it
    @classmethod
    def identity(cls) -> "MetricMapper":
        """
        Identity-ish mapper: keep all columns, no renames, no rules,
        include_unmatched=True, normalize_unmatched=True.
        """
        return cls()

    # ------------------------------------------------------------------
    # Column selection / ordering (used by CriticCohortAgent)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Column selection / ordering (used by CriticCohortAgent)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Column selection / ordering (used by CriticCohortAgent)
    # ------------------------------------------------------------------

    def _match_any(self, name: str, patterns: Sequence[str]) -> bool:
        return any(fnmatch(name, pat) for pat in patterns)

    def select(self, all_columns: Sequence[str]) -> List[str]:
        """
        Filter metric columns based on include/exclude patterns.

        Design choice: do *not* attempt clever preferred-keys reordering here.
        That logic was brittle and caused crashes when configs didn't line up.
        We just:
          - start from the incoming order
          - apply include/exclude globs
          - return the result in stable order
        """
        cols = list(all_columns)
        log.info("MetricMapper.select: start with %d columns", len(cols))

        # 1) include filter
        if self.include_patterns:
            before = len(cols)
            cols = [c for c in cols if self._match_any(c, self.include_patterns)]
            log.info(
                "MetricMapper.select: include %r → %d → %d columns",
                self.include_patterns,
                before,
                len(cols),
            )

        # 2) exclude filter
        if self.exclude_patterns:
            before = len(cols)
            cols = [c for c in cols if not self._match_any(c, self.exclude_patterns)]
            log.info(
                "MetricMapper.select: exclude %r → %d → %d columns",
                self.exclude_patterns,
                before,
                len(cols),
            )

        # 3) Keep original order. We deliberately ignore preferred_keys here.
        log.info(
            "MetricMapper.select: final selected=%d columns (first few=%r)",
            len(cols),
            cols[:10],
        )
        return cols

    # Alias to keep your current agent code working
    def select_columns(self, all_columns: Sequence[str]) -> List[str]:
        selected = self.select(all_columns)
        log.info(
            "MetricMapper.select_columns: selected %d of %d: %s",
            len(selected),
            len(all_columns),
            selected[:20],
        )
        return selected

    def rename(self, name: str) -> str:
        new_name = self.rename_map.get(name, name)
        log.info("MetricMapper.rename: %r -> %r", name, new_name)
        return new_name

    # ------------------------------------------------------------------
    # Optional: rule-based mapping + normalization over a scores matrix
    # ------------------------------------------------------------------

    def apply(
        self,
        metric_names: Sequence[str],
        scores: np.ndarray,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Apply mapping to a (N, D) scores matrix.

        Args:
            metric_names: list of column names length D
            scores:       numpy array shape (N, D)

        Returns:
            (mapped_names, mapped_scores)
        """
        metric_names = list(metric_names)
        scores = np.asarray(scores, dtype=np.float32)

        if scores.ndim != 2:
            raise ValueError(f"MetricMapper.apply expected 2D scores, got {scores.shape}")

        if len(metric_names) != scores.shape[1]:
            raise ValueError(
                f"MetricMapper.apply: metric_names length {len(metric_names)} "
                f"!= scores.shape[1] {scores.shape[1]}"
            )

        if not metric_names or scores.size == 0:
            return metric_names, scores

        log.info(
            "MetricMapper.apply: n_items=%d n_metrics=%d rules=%d "
            "include_unmatched=%r normalize_unmatched=%r",
            scores.shape[0],
            scores.shape[1],
            len(self.rules),
            self.include_unmatched,
            self.normalize_unmatched,
        )
        if len(metric_names) <= 40:
            log.info("MetricMapper.apply: metric_names=%r", metric_names)
        else:
            log.info(
                "MetricMapper.apply: metric_names sample=%r (+%d more)",
                metric_names[:40],
                len(metric_names) - 40,
            )

        # Map metric_name -> index
        name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(metric_names)}

        used_indices: List[int] = []
        mapped_names: List[str] = []

        # 1) Apply explicit rules in order
        for rule in self.rules:
            matched_indices = self._match_rule(rule, metric_names, name_to_idx)
            log.info(
                "MetricMapper.apply: rule=%r patterns=%r matched_indices=%r",
                rule.name,
                rule.patterns,
                matched_indices,
            )
            if not matched_indices:
                if not rule.allow_missing:
                    log.warning(
                        "MetricMapper: rule %r had no matches and allow_missing=False",
                        rule.name,
                    )
                continue

            for idx in matched_indices:
                if idx not in used_indices:
                    used_indices.append(idx)
                    mapped_names.append(metric_names[idx])

        # 2) Append unmatched columns if requested
        if self.include_unmatched:
            for idx, name in enumerate(metric_names):
                if idx not in used_indices:
                    used_indices.append(idx)
                    mapped_names.append(name)

        log.info(
            "MetricMapper.apply: used_indices=%r mapped_names=%r",
            used_indices,
            mapped_names,
        )

        # 3) Build the remapped scores matrix
        mapped_scores = scores[:, used_indices]

        # 4) Normalize columns according to rules / defaults
        mapped_scores = self._normalize_columns(
            mapped_scores,
            mapped_names,
        )

        return mapped_names, mapped_scores

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _match_rule(
        self,
        rule: MetricMappingRule,
        metric_names: Sequence[str],
        name_to_idx: Dict[str, int],
    ) -> List[int]:
        """
        Return a list of column indices that match any of rule.patterns,
        in the *original* order of metric_names.
        """
        if not rule.patterns and self._canon(rule.name) in {self._canon(n): None for n in metric_names}:
            idx = next(i for i, n in enumerate(metric_names) if self._canon(n) == self._canon(rule.name))
            # Direct name match fallback
            log.info(
                "MetricMapper._match_rule: rule=%r direct name match %r -> idx=%d",
                rule.name,
                rule.name,
                name_to_idx[rule.name],
            )
            return [idx]

        matched: List[int] = []
        for i, name in enumerate(metric_names):
            if any(fnmatch(self._canon(name), self._canon(p)) for p in rule.patterns):
                matched.append(i)

        log.info(
            "MetricMapper._match_rule: rule=%r patterns=%r matched_names=%r",
            rule.name,
            rule.patterns,
            [metric_names[i] for i in matched],
        )
        return matched

    def _normalize_columns(
        self,
        scores: np.ndarray,
        names: Sequence[str],
    ) -> np.ndarray:
        """
        Min–max normalize each column into [0, 1] according to rule.normalize
        or normalize_unmatched for columns without a matching rule.
        """
        if scores.size == 0:
            return scores

        scores = scores.copy()
        col_count = scores.shape[1]

        log.info(
            "MetricMapper._normalize_columns: col_count=%d rules=%d normalize_unmatched=%r",
            col_count,
            len(self.rules),
            self.normalize_unmatched,
        )

        # Pre-build: name -> rule
        name_to_rule: Dict[str, MetricMappingRule] = {}
        for rule in self.rules:
            for name in names:
                if rule.patterns and any(fnmatch(name, p) for p in rule.patterns):
                    name_to_rule[name] = rule
                elif not rule.patterns and name == rule.name:
                    name_to_rule[name] = rule

        eps = 1e-8

        for j in range(col_count):
            col = scores[:, j]
            name = names[j]
            rule = name_to_rule.get(name)

            if rule is not None:
                if not rule.normalize:
                    log.info(
                        "MetricMapper._normalize_columns: skipping normalization for %r "
                        "(rule.normalize=False)",
                        name,
                    )
                    continue  # leave as-is
            else:
                if not self.normalize_unmatched:
                    log.info(
                        "MetricMapper._normalize_columns: skipping normalization for %r "
                        "(no rule, normalize_unmatched=False)",
                        name,
                    )
                    continue

            cmin = float(col.min())
            cmax = float(col.max())
            if cmax - cmin < eps:
                scores[:, j] = 0.5  # flat column → neutral mid value
                log.info(
                    "MetricMapper._normalize_columns: %r is flat (min=max=%.4f); set to 0.5",
                    name,
                    cmin,
                )
            else:
                scores[:, j] = (col - cmin) / (cmax - cmin)
                log.info(
                    "MetricMapper._normalize_columns: normalized %r from [%.4f, %.4f] to [0,1]",
                    name,
                    cmin,
                    cmax,
                )

        return scores


    def _canon(self, s: str) -> str:
        return s.lower() if self.case_insensitive and isinstance(s, str) else s

    def _match_any(self, name: str, patterns: Sequence[str]) -> bool:
        n = self._canon(name)
        return any(fnmatch(self._canon(p), n) or fnmatch(n, self._canon(p)) for p in patterns)
