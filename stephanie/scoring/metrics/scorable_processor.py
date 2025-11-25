# stephanie/scoring/metrics/scorable_processor.py
from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Union

from stephanie.scoring.metrics.domain_feature import DomainFeature
from stephanie.scoring.metrics.embedding_feature import EmbeddingFeature
from stephanie.scoring.metrics.ner_feature import NerFeature
from stephanie.scoring.metrics.row_builder import RowBuilder
from stephanie.scoring.metrics.text_feature import TextFeature
from stephanie.scoring.scorable import Scorable, ScorableFactory
from stephanie.utils.progress_mixin import ProgressMixin
from stephanie.scoring.metrics.metrics_feature import MetricsFeature
from stephanie.scoring.metrics.visicalc_basic_feature import VisiCalcBasicFeature

log = logging.getLogger(__name__)


FEATURE_REGISTRY = {
    "metrics": MetricsFeature,
    "visicalc": VisiCalcBasicFeature,
    "embeddings": EmbeddingFeature,
    "ner": NerFeature,
    "domains": DomainFeature,
    "text": TextFeature,

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

        # Features: List[BaseFeature]
        self.features = cfg.get("features", [])
        feature_cfgs = self.cfg.get("feature_configs", {})

        self.features = []

        for name in feature_cfgs.keys():
            cls = FEATURE_REGISTRY.get(name)
            if cls is None:
                log.warning(f"[SP] unknown feature '{name}' ignored")
                continue

            # Grab sub-config for this feature
            cfg = feature_cfgs.get(name, {})

            # Instantiate feature
            feature = cls(
                cfg=cfg,
                memory=self.memory,
                container=self.container,
                logger=self.logger,
            )

            self.features.append(feature)
            log.info(f"[SP] feature loaded: {name} → {cls.__name__}")

        # Writers: List[BaseWriter]
        self.writers = cfg.get("writers", [])

        # Row builder
        self.row_builder = RowBuilder()

        # Progress tracking
        self._init_progress(self.container, self.logger)

        # skip mode
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

        acc = {}

        # Apply each feature sequentially
        for feature in self.features:
            try:
                acc = await feature.apply(scorable, acc, context)
            except Exception as e:
                log.warning(
                    f"[ScorableProcessor] Feature {feature.name} failed: {e}"
                )

        # Build final row OK
        row_obj = self.row_builder.build(scorable, acc)
        row = row_obj.to_dict()

        # Persist deltas
        for writer in self.writers:
            try:
                await writer.persist(scorable, acc)
            except Exception as e:
                log.warning(f"[ScorableProcessor] Writer {writer.name} failed: {e}")

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

        task_name = f"ScorableProcess:{context.get('pipeline_run_id', 'na')}"
        self.pstart(task=task_name, total=n)

        out: List[Dict[str, Any]] = []

        try:
            for idx, sc in enumerate(inputs):
                row = await self.process(sc, context)
                out.append(row)
                self.ptick(task=task_name, done=idx + 1, total=n)
        finally:
            self.pdone(task=task_name)

        log.debug(
            "[ScorableProcessor] batch_size=%d finished in %.2f ms",
            n,
            (time.perf_counter() - t_all) * 1000,
        )

        return out
