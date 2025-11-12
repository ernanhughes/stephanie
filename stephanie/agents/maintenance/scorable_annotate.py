# stephanie/agents/knowledge/scorable_annotate.py
"""
Scorable Annotation Agent (Processor-powered)

This agent enriches incoming scorables using the canonical ScorableProcessor.
It hydrates known features (DB), computes missing ones (domain/NER/embeddings,
vision), optionally attaches model scores, persists deltas, and reflects a thin
annotation back onto each scorable's 'meta' for downstream compatibility.

Key improvements:
- Pluggable, robust feature mediation via ScorableProcessor
- Batch processing with progress bars
- Idempotent persistence (writers only write deltas)
- Optional 'force' mode that recomputes without persisting to avoid dupes
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import asyncio
from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable, ScorableFactory
from stephanie.scoring.scorable_processor import ScorableProcessor

log = logging.getLogger(__name__)


class ScorableAnnotateAgent(BaseAgent):
    """
    Agent that places the ScorableProcessor at the front of the pipeline.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Behavior knobs (sane defaults)
        self.progress_enabled: bool = bool(cfg.get("progress", True))
        self.filter_role: bool = bool(cfg.get("filter_role", False))
        self.scorable_role: str = cfg.get("scorable_role", "candidate")

        # Legacy semantics (now implemented via processor overrides)
        self.only_missing: bool = bool(cfg.get("only_missing", True))
        self.force: bool = bool(cfg.get("force", False))

        # Batch + scoring options
        self.batch_size: int = int(cfg.get("batch_size", 64))
        self.attach_scores: bool = bool(cfg.get("attach_scores", True))
        self.scoring_dims: Optional[List[str]] = cfg.get("scoring_dims")

        # progress/concurrency knobs
        self.max_concurrency: int = int(cfg.get("max_concurrency", 8))
        self.progress_log_every: int = int(cfg.get("progress_log_every", 25))
        self.progress_leave: bool = bool(cfg.get("progress_leave", True))
        self.progress_position: int = int(cfg.get("progress_position", 0))

        # Manifest
        self.manifest_cfg: Optional[Dict[str, Any]] = cfg.get("manifest")  # expect dict: {run_id, dataset, models, base_root}
        self.manifest_path: Optional[str] = cfg.get("manifest_path")

        # Default processor config (can be overridden per-run by _build_processor)
        self.processor_cfg_defaults: Dict[str, Any] = {
            "enable_domain_hydrate": True,
            "enable_ner_hydrate": True,
            "enable_domain_persist": True,
            "enable_ner_persist": True,
            "enable_ner_model": True,
            "min_domains": int(cfg.get("min_domains", 1)),
            "attach_scores": self.attach_scores,
        }

    # ---------- Public entry point ----------

    async def run(self, context: dict) -> dict:
        """
        Expects context['scorables'] = List[dict|Scorable].
        Produces:
          - context['scorable_features'] = List[dict] (canonical features rows)
          - updates each scorable.meta with short-form 'domains' and 'ner'
          - context['scorable_annotation_summary'] with stats
        """
        scorables_in = list(context.get("scorables") or [])
        if not scorables_in:
            log.debug("ScorableAnnotateAgent: no scorables in context")
            return context

        if self.filter_role and self.scorable_role:
            before = len(scorables_in)
            scorables_in = [s for s in scorables_in if (s.get("role") if isinstance(s, dict) else (getattr(s, "meta", {}) or {}).get("role")) == self.scorable_role]
            log.debug("Filtered scorables by role '%s': %d → %d", self.scorable_role, before, len(scorables_in))
            if not scorables_in:
                self.logger.log("ScorableAnnotateSkip", {"reason": "role_filter_empty", "role": self.scorable_role})
                return context

        # Build a processor tuned for this run (handles only_missing/force)
        processor = self._build_processor_for_run()

        # Optional manifest
        if self.manifest_path:
            try:
                processor.start_manifest(self.manifest_path)
            except Exception as e:
                log.warning("ScorableAnnotateAgent: manifest start failed: %s", e)

        # Normalize to Scorable objects for consistent downstream fields
        norm_scorables: List[Scorable] = [self._coerce_to_scorable(x) for x in scorables_in]

        # Process in batches
        features_rows: List[Dict[str, Any]] = []
        pbar = tqdm(
            total=len(norm_scorables),
            desc=f"ScorableAnnotate ({self.scorable_role})",
            disable=not self.progress_enabled,
            leave=self.progress_leave,
            position=self.progress_position,
        )

        for i in range(0, len(norm_scorables), self.batch_size):
            batch = norm_scorables[i : i + self.batch_size]
            try:
                batch_rows = await self._process_batch_concurrent(processor, batch, context, pbar)
                features_rows.extend(batch_rows)
            except Exception as e:
                log.exception("ScorableAnnotateAgent: processor batch failed: %s", e)
                # Per-item salvage with live progress
                for sc in batch:
                    try:
                        row = await processor.process(sc, context=context)
                        features_rows.append(row)
                    except Exception:
                        pass
                    finally:
                        pbar.update(1)
                        pbar.refresh()
                        if self.progress_log_every and (pbar.n % self.progress_log_every == 0):
                            log.info("ScorableAnnotate progress: %d/%d", pbar.n, pbar.total)

        pbar.close()

        # Reflect lightweight domains/ner back into the in-memory scorables for downstream agents
        # NOTE: the DB already has the persisted full records; this is just for immediate pipeline consumers
        by_key = {(r.get("scorable_type"), str(r.get("scorable_id"))): r for r in features_rows}
        annotated_out: List[Dict[str, Any]] = []
        for original in scorables_in:
            sc = self._coerce_to_scorable(original)
            row = by_key.get((sc.target_type, str(sc.id)))
            annotated = original if isinstance(original, dict) else sc.to_dict()

            # ensure meta
            annotated.setdefault("meta", {})

            if row:
                # Domains: convert processor 'name'→ legacy 'domain'
                domains_short = [
                    {"domain": d.get("name") or d.get("domain"), "score": float(d.get("score", 1.0)), "source": d.get("source")}
                    for d in (row.get("domains") or [])
                    if d.get("name") or d.get("domain")
                ]
                if domains_short:
                    annotated["meta"]["domains"] = domains_short

                # NER: pass through
                if row.get("ner"):
                    annotated["meta"]["ner"] = row["ner"]

            annotated_out.append(annotated)

        # Stats
        stats = {
            "scorables_total": len(scorables_in),
            "features_rows": len(features_rows),
            "cache_hit_rate": processor.get_cache_stats().get("hit_rate", 0.0),
            "attach_scores": bool(self.attach_scores),
            "force": bool(self.force),
            "only_missing": bool(self.only_missing),
        }

        # Log & report
        self.logger.log("ScorableAnnotateDone", stats)
        self.report({"event": "scorables_annotated", **stats})


        try:
            if getattr(processor, "manifest_mgr", None) and getattr(processor, "manifest", None):
                processor.finish_manifest(result={"rows": len(features_rows)})
                run_id = processor.manifest.run_id
                man_path = processor.manifest_mgr.manifest_path(run_id)
                feat_path = processor.manifest_mgr.features_jsonl_path(run_id, name="features.jsonl")
                log.info("ScorableAnnotate: manifest saved → %s", man_path)
                log.info("ScorableAnnotate: features.jsonl saved → %s", feat_path)
        except Exception as e:
            log.warning("ScorableAnnotate: manifest finish failed: %s", e)
            
        # Write outputs into context
        context["scorable_features"] = features_rows
        context["scorables"] = annotated_out
        context["scorable_annotation_summary"] = stats
        return context

    # ---------- Helpers ----------

    def _coerce_to_scorable(self, x: Any) -> Scorable:
        if isinstance(x, Scorable):
            return x
        if isinstance(x, dict):
            # Normalize flexible incoming shapes via the factory
            return ScorableFactory.from_dict(x, target_type=x.get("scorable_type") or x.get("target_type"))
        # Last resort: treat as text-only custom
        return ScorableFactory.from_dict({"text": str(x or ""), "target_type": "custom"})

    def _build_processor_for_run(self) -> ScorableProcessor:
        """
        Compose a per-run processor config to honor only_missing/force semantics.

        - only_missing=True (default): allow hydration; model fills gaps; writers persist deltas.
        - force=True: recompute fresh (disable hydration) and DO NOT persist to avoid duplicates.
        """
        cfg = dict(self.processor_cfg_defaults)

        if self.attach_scores:
            cfg["attach_scores"] = True

        if self.only_missing and not self.force:
            # Keep defaults: hydrate → compute gaps → persist deltas
            pass
        elif self.force:
            # Recompute without reading from DB; also don't write to DB to avoid dupes
            cfg.update({
                "enable_domain_hydrate": False,
                "enable_ner_hydrate": False,
                "enable_domain_persist": False,
                "enable_ner_persist": False,
                # still OK to compute with models
                "enable_ner_model": True,
                "min_domains": int(self.cfg.get("min_domains", 1)),
            })

        # Build processor with memory/container; logger reused
        processor = ScorableProcessor(cfg, self.memory, self.container, self.logger)

        # If you want to hard-pin scoring dims for this agent, you can set them on the scoring_service via container
        scoring_service = getattr(processor, "scoring_service", None)
        if scoring_service and hasattr(scoring_service, "default_dims") and self.scoring_dims:
            try:
                scoring_service.default_dims = list(self.scoring_dims)
            except Exception:
                pass

        return processor

    async def _process_batch_concurrent(
        self,
        processor: ScorableProcessor,
        batch: List[Scorable],
        context: Dict[str, Any],
        pbar
    ) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(self.max_concurrency)
        rows: List[Dict[str, Any]] = []

        async def _run_one(sc: Scorable):
            async with sem:
                return await processor.process(sc, context=context)

        tasks = [asyncio.create_task(_run_one(sc)) for sc in batch]

        completed = 0
        for fut in asyncio.as_completed(tasks):
            row = None
            try:
                row = await fut
            except Exception as e:
                log.debug("ScorableAnnotate: item failed: %s", e)
            if row is not None:
                rows.append(row)
            completed += 1
            pbar.update(1)
            pbar.refresh()
            if self.progress_log_every and (pbar.n % self.progress_log_every == 0):
                log.info("ScorableAnnotate progress: %d/%d", pbar.n, pbar.total)

        return rows
