# stephanie/scoring/metrics/scorable_processor.py
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Union

from stephanie.constants import PIPELINE_RUN_ID
from stephanie.scoring.metrics.domain_feature import DomainFeature
from stephanie.scoring.metrics.embedding_feature import EmbeddingFeature
from stephanie.scoring.metrics.feature_report import FeatureReport
from stephanie.scoring.metrics.metric_filter_group_feature import (
    MetricFilterGroupFeature,
)
from stephanie.scoring.metrics.metrics_feature import MetricsFeature
from stephanie.scoring.metrics.ner_feature import NerFeature
from stephanie.scoring.metrics.row_builder import RowBuilder
from stephanie.scoring.metrics.text_feature import TextFeature
from stephanie.scoring.metrics.visicalc_basic_feature import (
    VisiCalcBasicFeature,
)
from stephanie.scoring.metrics.visicalc_group_feature import (
    VisiCalcGroupFeature,
)
from stephanie.scoring.scorable import Scorable, ScorableFactory
from stephanie.utils.progress_mixin import ProgressMixin

log = logging.getLogger(__name__)


FEATURE_REGISTRY = {
    "metrics": MetricsFeature,
    "visicalc": VisiCalcBasicFeature,
    "embeddings": EmbeddingFeature,
    "ner": NerFeature,
    "domains": DomainFeature,
    "text": TextFeature,
}


GROUP_FEATURE_REGISTRY = {
    "metric_filter": MetricFilterGroupFeature,
    "visicalc_group": VisiCalcGroupFeature,
}


class ScorableProcessor(ProgressMixin):
    """
    PURE PROCESSOR:
      - Applies feature modules (DomainFeature, NerFeature, EmbeddingFeature, MetricsFeature, VpmFeature, etc)
      - Builds a standardized ScorableRow via RowBuilder
      - Persists deltas through Writers
      - NO bus
      - NO manifest
      - NO hydration providers (hydration is a feature)
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        # ---------- Row features ----------
        feature_cfgs = self.cfg.get("feature_configs", {}) or {}
        self.features = []

        for name, subcfg in feature_cfgs.items():  # preserves YAML order
            cls = FEATURE_REGISTRY.get(name)
            if cls is None:
                log.warning(f"[SP] unknown feature '{name}' ignored")
                continue
            feat = cls(
                cfg=subcfg or {},
                memory=self.memory,
                container=self.container,
                logger=self.logger,
            )
            if getattr(feat, "enabled", True):
                self.features.append(feat)
                log.info(f"[SP] feature loaded: {name} → {cls.__name__}")
            else:
                log.info(f"[SP] feature disabled: {name}")

        # ---------- Group features ----------
        self.group_features = []
        for name, subcfg in (
            self.cfg.get("group_feature_configs", {}) or {}
        ).items():
            cls = GROUP_FEATURE_REGISTRY.get(name)
            if not cls:
                log.warning("[SP] unknown group feature '%s'", name)
                continue
            gf = cls(  # MUST pass the sub-config
                cfg=subcfg,  # ← not `cfg`
                memory=self.memory,
                container=self.container,
                logger=self.logger,
            )
            if gf.enabled:
                self.group_features.append(gf)
                log.info(
                    "[SP] group feature loaded: %s → %s", name, cls.__name__
                )

        # ---------- Row builder ----------
        self.row_builder = RowBuilder()

        # ---------- Writers (optional legacy) ----------
        self.writers = self.cfg.get(
            "writers", []
        )  # consider removing entirely later

        # ---------- Progress + misc ----------
        self._init_progress(self.container, self.logger)
        self.skip_if_exists = bool(self.cfg.get("skip_if_exists", True))

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    async def process(
        self,
        input_data: Union[Scorable, Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        scorable = (
            input_data
            if isinstance(input_data, Scorable)
            else ScorableFactory.from_dict(input_data)
        )
        acc: Dict[str, Any] = {}

        for feature in self.features:
            try:
                log.info(
                    f"[ScorableProcessor] applying feature: {feature.name}"
                )
                acc = await feature.apply(scorable, acc, context)
                log.info(
                    f"[ScorableProcessor] applied feature: {feature.name}"
                )
            except Exception as e:
                log.warning(
                    f"[ScorableProcessor] Feature {feature.name} failed: {e}"
                )

        row_obj = self.row_builder.build(scorable, acc)
        row = row_obj.to_dict()

        for writer in self.writers:
            try:
                await writer.persist(scorable, acc)
            except Exception as e:
                log.warning(
                    f"[ScorableProcessor] Writer {getattr(writer, 'name', 'writer')} failed: {e}"
                )

        log.debug(
            "[ScorableProcessor] done id=%s in %.2f ms",
            scorable.id,
            (time.perf_counter() - t0) * 1000,
        )
        return row

    # -----------------------------------------------------
    # Batch processing
    # -----------------------------------------------------

    async def process_many(
        self,
        inputs: List[Union[Scorable, Dict[str, Any]]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        n = len(inputs)
        t_all = time.perf_counter()

        task_rows = (
            f"ScorableProcess:rows:{context.get(PIPELINE_RUN_ID, 'na')}"
        )
        self.pstart(task=task_rows, total=n)

        rows: List[Dict[str, Any]] = []
        try:
            for idx, sc in enumerate(inputs):
                row = await self.process(sc, context)
                rows.append(row)
                self.ptick(task=task_rows, done=idx + 1, total=n)
        finally:
            self.pdone(task=task_rows)

        # Optional: enforce simple dependencies if group features declare `requires`
        available = {
            getattr(f, "name", f"f{idx}")
            for idx, f in enumerate(self.group_features)
        }
        for gf in self.group_features:
            reqs = getattr(gf, "requires", []) or []
            missing = [r for r in reqs if r not in available]
            if missing:
                log.warning(
                    f"[SP] group feature '{gf.name}' missing requirements: {missing}"
                )

        task_group = (
            f"ScorableProcess:group:{context.get('pipeline_run_id', 'na')}"
        )
        self.pstart(task=task_group, total=len(self.group_features))
        try:
            for i, gf in enumerate(self.group_features, start=1):
                try:
                    log.info(
                        f"[ScorableProcessor] applying group feature: {gf.name}"
                    )
                    rows = await gf.apply(rows, context)
                    log.info(
                        f"[ScorableProcessor] applied group feature: {gf.name}"
                    )
                except Exception as e:
                    log.warning(
                        f"[ScorableProcessor] Group feature {gf.name} failed: {e}"
                    )
                self.ptick(
                    task=task_group, done=i, total=len(self.group_features)
                )
        finally:
            self.pdone(task=task_group)

        if context is not None:
            context.setdefault("feature_reports", self.feature_reports())

        log.debug(
            "[ScorableProcessor] batch_size=%d finished in %.2f ms",
            n,
            (time.perf_counter() - t_all) * 1000,
        )
        return rows

    def feature_reports(self) -> list[dict]:
        reps = []
        # per-row features
        for f in self.features:
            if hasattr(f, "report"):
                try:
                    reps.append(_report_to_dict(f.report(), f))
                except Exception as e:
                    reps.append(
                        {
                            "name": getattr(f, "name", f.__class__.__name__),
                            "kind": getattr(f, "kind", "row"),
                            "ok": False,
                            "summary": "report() raised",
                            "details": {"error": str(e)},
                        }
                    )
        # group features
        for gf in self.group_features:
            if hasattr(gf, "report"):
                try:
                    reps.append(_report_to_dict(gf.report(), gf))
                except Exception as e:
                    reps.append(
                        {
                            "name": getattr(gf, "name", gf.__class__.__name__),
                            "kind": getattr(gf, "kind", "group"),
                            "ok": False,
                            "summary": "report() raised",
                            "details": {"error": str(e)},
                        }
                    )
        return reps


def _report_to_dict(rep, feature):
    # 1) FeatureReport dataclass → dict
    try:
        from stephanie.scoring.metrics.feature_report import (
            FeatureReport,
        )  # adjust import if needed

        if isinstance(rep, FeatureReport):
            d = asdict(rep)
            # ensure minimal fields
            d.setdefault(
                "name", getattr(feature, "name", feature.__class__.__name__)
            )
            d.setdefault("kind", getattr(feature, "kind", "row"))
            return d
    except Exception:
        pass

    # 2) Already a mapping → ensure name/kind
    if isinstance(rep, dict):
        rep = dict(rep)
        rep.setdefault(
            "name", getattr(feature, "name", feature.__class__.__name__)
        )
        rep.setdefault("kind", getattr(feature, "kind", "row"))
        rep.setdefault("ok", rep.get("ok", True))
        return rep

    # 3) Fallback: opaque value
    return {
        "name": getattr(feature, "name", feature.__class__.__name__),
        "kind": getattr(feature, "kind", "row"),
        "ok": False,
        "summary": "Unknown report type",
        "details": {"raw": repr(rep)},
        "warnings": ["non-dict/non-FeatureReport report"],
    }
