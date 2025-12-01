# stephanie/components/critic/agents/critic_scorer.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.critic.agents.critic_cohort import CriticCohortAgent
from stephanie.components.critic.model.critic_model import CriticModel
from stephanie.components.critic.reports.critic_reporter import \
    build_critic_report
from stephanie.scoring.metrics.frontier_intelligence import \
    FrontierIntelligence
from stephanie.scoring.metrics.scorable_processor import ScorableProcessor

log = logging.getLogger(__name__)


class CriticScorerAgent(BaseAgent):
    """
    Unified inference agent for the Tiny Critic.

    Pipeline:
      - Convert scorables → canonical feature rows (ScorableProcessor)
      - Extract VPM matrix & metric names (reuse VisiCalc internals)
      - Load CriticModel (Pipeline + Meta)
      - Run batch scoring
      - Output critic_prob, critic_label, and rationale per scorable

    This mirrors SVM/MRQ scoring so it fits the Stephanie scoring layer.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.model_path = cfg.get("model_path", "models/critic.joblib")
        self.meta_path  = cfg.get("meta_path",  "models/critic.meta.json")

        self.scorable_processor = ScorableProcessor(
            cfg.get("processor", {"offload_mode": "inline"}),
            memory,
            container,
            logger,
        )

        # We will reuse CriticCohortAgent._build_vpm_and_metric_names
        self.visi_helper = CriticCohortAgent(
            cfg=cfg.get("visicalc", {"enabled": False}),
            memory=memory,
            container=container,
            logger=logger,
            run_id=self.run_id,
        )

        self.frontier_intelligence = FrontierIntelligence(self.cfg, self.memory, self.container, self.logger, self.run_id)
        self.report_dir = cfg.get("report_dir", f"runs/critic/{self.run_id}/reports")
        os.makedirs(self.report_dir, exist_ok=True)
        self.model: CriticModel | None = None

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    def _load_model(self):
        if self.model is None:
            self.model = CriticModel.load(
                model_path=self.model_path,
                meta_path=self.meta_path,
            )
            log.info(
                "CriticScorerAgent: loaded model=%s; meta features=%d",
                self.model_path,
                len(self.model.meta.feature_names),
            )
        return self.model

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        scorables = list(context.get(self.input_key) or [])

        if not scorables:
            log.warning("CriticScorerAgent: no scorables provided")
            return context

        # -----------------------------
        # 1. Canonical feature rows
        # -----------------------------
        rows = await self.scorable_processor.process_many(
            scorables, context=context
        )
        context["scorable_features"] = rows

        if not rows:
            log.warning("CriticScorerAgent: no rows from ScorableProcessor")
            return context

        # -----------------------------
        # 2. Build VPM + metric names
        # -----------------------------
        try:
            vpm, metric_names, item_ids = \
                self.visi_helper._build_vpm_and_metric_names(rows)
        except Exception:
            log.exception("CriticScorerAgent: failed to build VPM")
            return context

        # -----------------------------
        # 3. Score using CriticModel
        # -----------------------------
        model = self._load_model()
        probs = model.score_batch(vpm, incoming_names=metric_names)
        labels = (probs >= 0.5).astype(int)

        # -----------------------------
        # 4. Attach results
        # -----------------------------
        scored = []
        for sc, p, y in zip(scorables, probs, labels):
            out = {
                "scorable": sc,
                "critic_prob": float(p),
                "critic_label": int(y),
                "critic_rationale": f"prob={p:.4f}, label={y}",
            }
            scored.append(out)

        context[self.output_key] = scored
        context["critic_probs"] = probs.tolist()
        context["critic_labels"] = labels.tolist()
        context["critic_metric_names"] = metric_names

        # rows already built; context may have feature reports
        feature_reports = context.get("feature_reports") or []

        # load meta.feature_names from your saved critic meta
        model_feature_names = []
        try:
            import json
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                model_feature_names = meta.get("feature_names") or []
        except Exception:
            pass

        # MetricStore from memory
        metric_store = self.memory.metrics  # your MetricStore wrapper
        run_id = context.get("pipeline_run_id")

        out_md = f"{self.report_dir}/critic_run_{run_id}.md"
        path = build_critic_report(
            rows=rows,
            feature_reports=feature_reports,
            model_feature_names=model_feature_names,
            metric_store=metric_store,
            run_id=run_id,
            out_path=out_md,
        )
        log.info("Saved Critic report: %s", out_md)

        self.logger.info("Critic run report saved → %s", path)
        return context
