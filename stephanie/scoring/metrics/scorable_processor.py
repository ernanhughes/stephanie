# stephanie/scoring/metrics/scorable_processor.py
from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Union

from stephanie.scoring.metrics.domain_feature import DomainFeature
from stephanie.scoring.metrics.embedding_feature import EmbeddingFeature
from stephanie.scoring.metrics.metric_filter_group_feature import (
    MetricFilterGroupFeature,
)
from stephanie.scoring.metrics.ner_feature import NerFeature
from stephanie.scoring.metrics.row_builder import RowBuilder
from stephanie.scoring.metrics.text_feature import TextFeature
from stephanie.scoring.metrics.visicalc_group_feature import (
    VisiCalcGroupFeature,
)
from stephanie.scoring.scorable import Scorable, ScorableFactory
from stephanie.utils.progress_mixin import ProgressMixin
from stephanie.scoring.metrics.metrics_feature import MetricsFeature
from stephanie.scoring.metrics.visicalc_basic_feature import (
    VisiCalcBasicFeature,
)
import importlib

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
        group_cfgs = self.cfg.get("group_feature_configs", {}) or {}
        self.group_features = []
        for name, subcfg in group_cfgs.items():
            cls_or_path = GROUP_FEATURE_REGISTRY.get(name, name)  # allow direct import path
            try:
                Cls = _import_path(cls_or_path)
            except Exception as e:
                log.warning(f"[SP] group feature '{name}' import failed: {e}")
                continue

            gf = Cls(
                cfg=subcfg or {},
                memory=self.memory,
                container=self.container,
                logger=self.logger,
            )
            if getattr(gf, "enabled", True):
                self.group_features.append(gf)
                log.info(f"[SP] group feature loaded: {name} → {Cls.__name__}")
            else:
                log.info(f"[SP] group feature disabled: {name}")

        # ---------- Row builder ----------
        self.row_builder = RowBuilder()

        # ---------- Writers (optional legacy) ----------
        self.writers = self.cfg.get("writers", [])  # consider removing entirely later

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
        scorable = input_data if isinstance(input_data, Scorable) else ScorableFactory.from_dict(input_data)
        acc: Dict[str, Any] = {}

        for feature in self.features:
            try:
                acc = await feature.apply(scorable, acc, context)
            except Exception as e:
                log.warning(f"[ScorableProcessor] Feature {feature.name} failed: {e}")

        row_obj = self.row_builder.build(scorable, acc)
        row = row_obj.to_dict()

        for writer in self.writers:
            try:
                await writer.persist(scorable, acc)
            except Exception as e:
                log.warning(f"[ScorableProcessor] Writer {getattr(writer,'name','writer')} failed: {e}")

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

        task_rows = f"ScorableProcess:rows:{context.get('pipeline_run_id','na')}"
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
        available = {getattr(f, "name", f"f{idx}") for idx, f in enumerate(self.features)}
        for gf in self.group_features:
            reqs = getattr(gf, "requires", []) or []
            missing = [r for r in reqs if r not in available]
            if missing:
                log.warning(f"[SP] group feature '{gf.name}' missing requirements: {missing}")

        task_group = f"ScorableProcess:group:{context.get('pipeline_run_id','na')}"
        self.pstart(task=task_group, total=len(self.group_features))
        try:
            for i, gf in enumerate(self.group_features, start=1):
                try:
                    rows = await gf.apply(rows, context)
                except Exception as e:
                    log.warning(f"[ScorableProcessor] Group feature {gf.name} failed: {e}")
                self.ptick(task=task_group, done=i, total=len(self.group_features))
        finally:
            self.pdone(task=task_group)

        log.debug(
            "[ScorableProcessor] batch_size=%d finished in %.2f ms",
            n,
            (time.perf_counter() - t_all) * 1000,
        )
        return rows

def _import_path(path_or_cls):
    """Allow registry entry to be a class OR a 'pkg.mod.Class' string."""
    if not isinstance(path_or_cls, str):
        return path_or_cls
    if "." not in path_or_cls:
        raise ValueError(f"Invalid import path: {path_or_cls}")
    mod, cls = path_or_cls.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)