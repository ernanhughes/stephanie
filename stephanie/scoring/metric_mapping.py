# stephanie/scoring/metric_mapping.py
from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


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


class MetricMapper:
    """
    Config-driven mapping from (metric_names, scores) → (new_names, new_scores).

    Responsibilities:
      - Select a subset of columns based on glob-style patterns
      - Order them according to config rules
      - Optionally append unmatched columns
      - Optionally min–max normalize each column into [0, 1]
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        raw_rules = cfg.get("rules") or []

        self.rules: List[MetricMappingRule] = [
            MetricMappingRule(
                name=str(r.get("name", f"rule_{i}")),
                patterns=list(r.get("patterns") or []),
                normalize=bool(r.get("normalize", True)),
                allow_missing=bool(r.get("allow_missing", True)),
            )
            for i, r in enumerate(raw_rules)
        ]
        self.include_unmatched: bool = bool(cfg.get("include_unmatched", True))
        self.normalize_unmatched: bool = bool(cfg.get("normalize_unmatched", True))

    # ------------------------------------------------------------------
    # Public API
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

        # Map metric_name -> index
        name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(metric_names)}

        used_indices: List[int] = []
        mapped_names: List[str] = []

        # 1) Apply explicit rules in order
        for rule in self.rules:
            matched_indices = self._match_rule(rule, metric_names, name_to_idx)
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
        if not rule.patterns:
            # Direct name match fallback
            if rule.name in name_to_idx:
                return [name_to_idx[rule.name]]
            return []

        matched: List[int] = []
        for i, name in enumerate(metric_names):
            if any(fnmatch.fnmatch(name, pat) for pat in rule.patterns):
                matched.append(i)
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

        # Pre-build: name -> rule
        name_to_rule: Dict[str, MetricMappingRule] = {}
        for rule in self.rules:
            # Associate any name that matched the rule patterns directly
            for name in names:
                if rule.patterns and any(fnmatch.fnmatch(name, p) for p in rule.patterns):
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
                    continue  # leave as-is
            else:
                if not self.normalize_unmatched:
                    continue

            cmin = float(col.min())
            cmax = float(col.max())
            if cmax - cmin < eps:
                scores[:, j] = 0.5  # flat column → neutral mid value
            else:
                scores[:, j] = (col - cmin) / (cmax - cmin)

        return scores

