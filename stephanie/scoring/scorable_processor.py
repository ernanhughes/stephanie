# stephanie/scoring/scorable_processor.py
from __future__ import annotations

import asyncio
import logging
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.models.ner_retriever import EntityDetector
import torch
import numpy as np

from stephanie.utils.json_sanitize import dumps_safe
from stephanie.scoring.scorable import Scorable, ScorableFactory
from stephanie.scoring.feature_io import (
    FeatureProvider,
    FeatureWriter,
    ScoringService,
)
from stephanie.scoring.adapters.db_providers import (
    DomainDBProvider,
    EntityDBProvider,
)
from stephanie.scoring.adapters.db_writers import (
    DomainDBWriter,
    EntityDBWriter,
)
from stephanie.core.manifest import ManifestManager, Manifest
from stephanie.services.zeromodel_service import ZeroModelService

log = logging.getLogger(__name__)



class ScorableProcessor:
    """
    Canonical mediator:
      - Accepts Scorable or dict
      - Hydrates features from providers (DB/cache)
      - Computes missing features (domain/NER/embeddings/vision)
      - Optionally attaches model scores via ScoringService
      - Persists only deltas via writers
    Backward compatible with your existing fields.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        # Providers (hydrate)
        self.providers: List[FeatureProvider] = []
        if self.cfg.get("enable_domain_hydrate", True):
            self.providers.append(DomainDBProvider(memory))
        if self.cfg.get("enable_ner_hydrate", True):
            self.providers.append(EntityDBProvider(memory))

        # Writers (persist deltas)
        self.writers: List[FeatureWriter] = []
        if self.cfg.get("enable_domain_persist", True):
            self.writers.append(DomainDBWriter(memory))
        if self.cfg.get("enable_ner_persist", True):
            self.writers.append(EntityDBWriter(memory))

        self.scorers = cfg.get("scorers", ["sicql", "hrm", "tiny"])
        self.dimensions = cfg.get(
            "dimensions",
            ["coverage", "reasoning", "knowledge", "clarity", "faithfulness"],
        )
        self.persist: bool = bool(cfg.get("persist_scores", False))

        # classifier for domains
        self.domain_classifier: ScorableClassifier = ScorableClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=cfg.get(
                "domain_seed_config_path", "config/domain/seeds.yaml"
            ),
        )
        self.persist_domains: bool = bool(cfg.get("persist_domains", False))

        try:
            self.entity_extractor = EntityDetector(
                device=cfg.get(
                    "device",
                    "cuda"
                    if hasattr(torch, "cuda") and torch.cuda.is_available()
                    else "cpu",
                )
            )
            log.debug("NER detector loaded.")
        except Exception as e:
            self.entity_extractor = None   # <- keep same name
            log.error(f"Failed to initialize EntityDetector: {e}")

        self.zm: ZeroModelService = container.get("zeromodel")
        self.scoring: ScoringService = container.get("scoring")

        # Embeddings (memory embedding service must provide get_or_create / get_or_create_batch)
        self.embed = self.memory.embedding

        self.include_text_features = bool(cfg.get("include_text_features", True))
        self.persist_vpm_png = bool(cfg.get("persist_vpm_png", True))
        self.save_vpm_channels = bool(cfg.get("save_vpm_channels", False))
        self.vpm_out_root = Path(cfg.get("vpm_out_root", "runs/vpm"))
        
        # Cache & manifest
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self.manifest_mgr: Optional[ManifestManager] = None
        self.manifest: Optional[Manifest] = None
        self._manifest_features_path: Optional[Path] = None
        self._manifest_lock = asyncio.Lock()  # you already have this
        self._current_manifest_path: Optional[Path] = None
        self._manifest_lock = asyncio.Lock()


        log.info(
            "[ScorableProcessor:init] providers=%d writers=%d scorers=%s dims=%s persist=%s device=%s",
            len(self.providers),
            len(self.writers),
            self.scorers,
            self.dimensions,
            self.persist,
            (cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")),
        )


    # ---------- Manifest ----------

    def start_manifest(self, *, run_id: str, dataset: Optional[str] = None,
                    models: Optional[Dict[str, str]] = None,
                    base_root: str = "data/ssp_runs") -> None:
        """Initialize a run folder + manifest.json and a features.jsonl file."""
        self.manifest_mgr = ManifestManager(base_root=base_root)
        self.manifest = self.manifest_mgr.start_run(
            run_id=run_id,
            dataset=dataset,
            models=models or {},
        )
        # JSONL for per-row features
        features_path = self.manifest_mgr.features_jsonl_path(run_id, name="features.jsonl")
        features_path.parent.mkdir(parents=True, exist_ok=True)
        if not features_path.exists():
            features_path.write_text("", encoding="utf-8")
        self._manifest_features_path = features_path
        log.debug("[SP:manifest] started → %s", self.manifest_mgr.manifest_path(run_id))
        log.debug("[SP:manifest] features.jsonl → %s", self._manifest_features_path)

    async def write_to_manifest(self, features_row: Dict[str, Any]) -> None:
        """Append one JSON object per line to features.jsonl."""
        if not self._manifest_features_path:
            raise RuntimeError("No manifest has been started.")
        async with self._manifest_lock:
            with open(self._manifest_features_path, "a", encoding="utf-8") as f:
                f.write(dumps_safe(features_row) + "\n")

    def finish_manifest(self, *, result: Dict[str, Any] | None = None) -> None:
        if not (self.manifest and self.manifest_mgr):
            log.debug("[SP:manifest] finish requested but no active manifest")
            return
        run_id = self.manifest.run_id
        self.manifest_mgr.finish_run(run_id, result or {})
        man_p = self.manifest_mgr.manifest_path(run_id)
        log.info("[SP:manifest] complete → %s", man_p)
        if self._manifest_features_path:
            try:
                size = self._manifest_features_path.stat().st_size
            except Exception:
                size = "?"
            log.info("[SP:manifest] features.jsonl → %s (size=%s)", self._manifest_features_path, size)

    # ---------- Utilities ----------

    @staticmethod
    def _hash_text(text: str) -> str:
        return sha256((text or "").encode("utf-8")).hexdigest()[:16]

    def _generate_cache_key(
        self, scorable_or_dict: Union[Scorable, Dict[str, Any]]
    ) -> str:
        if isinstance(scorable_or_dict, Scorable):
            stype = scorable_or_dict.target_type or "unknown"
            sid = scorable_or_dict.id or ""
            text = scorable_or_dict.text or ""
        else:
            stype = str(
                scorable_or_dict.get("target_type")
                or scorable_or_dict.get("type")
                or "unknown"
            )
            sid = str(
                scorable_or_dict.get("id")
                or scorable_or_dict.get("scorable_id")
                or ""
            )
            text = (
                scorable_or_dict.get("text")
                or scorable_or_dict.get("body")
                or ""
            )
        content_hash = self._hash_text(text)
        return f"{stype}:{sid or 'noid'}:{content_hash}"

    def _text_feats(self, text: str) -> Dict[str, float]:
        text = text or ""
        n = len(text)
        words = text.split()
        nw = len(words)
        caps = sum(1 for c in text if c.isupper())
        punc = sum(1 for c in text if c in "!?;:,.()[]{}\"'`")
        lines = text.count("\n") + 1
        avgw = (sum(len(w) for w in words) / max(1, nw)) if nw else 0.0
        return {
            "text.len": float(n),
            "text.words": float(nw),
            "text.avgw": float(avgw),
            "text.caps_ratio": float(caps / max(1, n)),
            "text.punc_ratio": float(punc / max(1, n)),
            "text.lines": float(lines),
        }


    def _fold_score_bundle(self, bundle: Dict[str, Any]) -> Dict[str, float]:
        vec: Dict[str, float] = {}
        for dim, res in (bundle.get("results") or {}).items():
            attrs = res.get("attributes") or {}
            for k, v in attrs.items():
                key = f"{dim}.{k}"
                try:
                    vec[key] = float(v)
                except Exception:
                    # keep only numeric attributes in metrics_vector
                    continue
            if res.get("score") is not None:
                vec[f"{dim}.score"] = float(res["score"])
        return vec

    def _build_features_row(
        self, scorable: Scorable, acc: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = scorable.text or ""
        title = acc.get("title") or (scorable.meta or {}).get("title")
        title = title or text[:80] or f"{scorable.target_type}:{scorable.id}"

        # embeddings
        embeddings = dict(acc.get("embeddings") or {})
        embed_global_np = None
        gl = embeddings.get("global")
        if isinstance(gl, list) and gl:
            try:
                embed_global_np = np.asarray(gl, dtype=np.float32)
            except Exception:
                embed_global_np = None

        # metrics
        metrics_vector = dict(acc.get("metrics_vector") or {})
        metrics_columns = list(metrics_vector.keys())
        metrics_values = [float(metrics_vector[k]) for k in metrics_columns]

        row = {
            "scorable_id": str(scorable.id)
            or f"{scorable.target_type}:{self._hash_text(text)}",
            "scorable_type": scorable.target_type,
            "conversation_id": (scorable.meta or {}).get("conversation_id"),
            "external_id": (scorable.meta or {}).get("external_id"),
            "order_index": (scorable.meta or {}).get("order_index"),
            "text": text,
            "title": title,
            "near_identity": acc.get("near_identity")
            or (scorable.meta or {}).get("near_identity")
            or {},
            "domains": acc.get("domains") or [],
            "ner": acc.get("ner") or [],
            "ai_score": (scorable.meta or {}).get("ai_score"),
            "star": (scorable.meta or {}).get("star"),
            "goal_ref": (scorable.meta or {}).get("goal_ref"),
            "embeddings": embeddings,
            "embed_global": embed_global_np.tolist()
            if isinstance(embed_global_np, np.ndarray)
            else (gl if isinstance(gl, list) else None),
            "metrics_columns": metrics_columns,
            "metrics_values": metrics_values,
            "metrics_vector": metrics_vector,
            "agreement": (scorable.meta or {}).get("agreement"),
            "stability": (scorable.meta or {}).get("stability"),
            "chat_id": (scorable.meta or {}).get("chat_id"),
            "turn_index": (scorable.meta or {}).get("turn_index"),
            "parent_scorable_id": (scorable.meta or {}).get(
                "parent_scorable_id"
            ),
            "parent_scorable_type": (scorable.meta or {}).get(
                "parent_scorable_type"
            ),
            "order_in_parent": (scorable.meta or {}).get("order_in_parent"),
            "vpm_png": (scorable.meta or {}).get("vpm_png"),
            # "vision_signals": acc.get("vision_signals") or {},
            "rollout": (scorable.meta or {}).get("rollout") or {},
            "processor_version": "2.0",
            "content_hash16": self._hash_text(text),
            "created_utc": time.time(),
        }
        return row

    # ---------- Core ----------

    async def process(
        self, input_data: Union[Scorable, Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        t_all = time.perf_counter()
        cache_key = self._generate_cache_key(input_data)

        if cache_key in self._cache:
            self._cache_hits += 1
            log.debug("[SP:process] cache HIT key=%s hit_rate=%.2f",
                     cache_key, self.get_cache_stats()["hit_rate"])
            return self._cache[cache_key]
        self._cache_misses += 1
        log.debug("[SP:process] cache MISS key=%s", cache_key)

        scorable = (
            input_data
            if isinstance(input_data, Scorable)
            else ScorableFactory.from_dict(input_data)
        )
        text = scorable.text or ""
        log.debug("[SP:process] start id=%s type=%s text_len=%d ctx.run=%s",
                 scorable.id, scorable.target_type, len(text), context.get("pipeline_run_id"))

        acc: Dict[str, Any] = {}

        # 1) Hydrate
        for provider in self.providers:
            t0 = time.perf_counter()
            name = provider.__class__.__name__
            log.debug("[SP:hydrate] start %s id=%s", name, scorable.id)
            try:
                acc.update(await provider.hydrate(scorable))
                log.debug("[SP:hydrate] done %s in %s keys_now=%d",
                         name, self._t(t0), len(acc))
            except Exception as e:
                log.warning("[SP:hydrate] %s failed: %s", name, e)

        # 2) Embeddings
        gl = (acc.get("embeddings") or {}).get("global")
        if not (isinstance(gl, list) and gl) and text:
            t0 = time.perf_counter()
            log.debug("[SP:embed] computing global id=%s", scorable.id)
            emb = self.memory.embedding.get_or_create(text)
            floats = self._ensure_float_list(emb)
            if floats is not None:
                acc.setdefault("embeddings", {})
                acc["embeddings"]["global"] = floats
                log.debug("[SP:embed] ok dim=%d in %s",
                         len(acc['embeddings']['global']), self._t(t0))
            else:
                log.debug("[SP:embed] skipped/none")

        # 3) Domains
        need_domains = not acc.get("domains") or len(acc["domains"]) < int(
            self.cfg.get("min_domains", 1)
        )
        if need_domains:
            t0 = time.perf_counter()
            log.debug("[SP:domain] inferring id=%s", scorable.id)
            if self.persist_domains:
                # Clear existing to avoid duplicates
                acc["domains"] = []
            
            inferred = self.domain_classifier.classify(text)
            for name, score in inferred:
                acc.setdefault("domains", []).append({"name": name, "score": score})
            log.debug("[SP:domain] inferred %d in %s", len(acc.get("domains") or []), self._t(t0))
        else:
            log.debug("[SP:domain] hydrated %d", len(acc.get("domains") or []))

        # 4) NER
        need_ner = not acc.get("ner") and bool(self.cfg.get("enable_ner_model", True))
        if need_ner and self.entity_extractor:
            t0 = time.perf_counter()
            log.debug("[SP:ner] detecting entities id=%s", scorable.id)
            try:
                ner = self.entity_extractor.detect_entities(text)
                acc["ner"] = ner or []
                log.debug("[SP:ner] found %d in %s", len(acc["ner"]), self._t(t0))
            except Exception as e:
                log.warning("[SP:ner] failed: %s", e)
        else:
            log.debug("[SP:ner] skipped (hydrated=%s, enabled=%s, has_detector=%s)",
                     bool(acc.get("ner")), bool(self.cfg.get("enable_ner_model", True)),
                     bool(self.entity_extractor))
        
        # 5) Scores  → build canonical metrics vector
        metrics_columns: List[str] = []
        metrics_values:  List[float] = []
        if self.scoring and self.cfg.get("attach_scores", True):
            t0_scores = time.perf_counter()
            goal_text = Scorable.get_goal_text(scorable, context=context)
            run_id = context.get("pipeline_run_id")
            ctx = {"goal": {"goal_text": goal_text}, "pipeline_run_id": run_id}
            vector: Dict[str, float] = {}
            log.debug("[SP:score] start scorers=%s dims=%s", self.scorers, self.dimensions)

            for name in self.scorers:
                t0 = time.perf_counter()
                log.debug("[SP:score] → %s", name)
                bundle = (self.scoring.score_and_persist if self.persist else self.scoring.score)(
                    scorer_name=name, scorable=scorable, context=ctx, dimensions=self.dimensions
                )
                model_alias = self.scoring.get_model_name(name)
                agg = float(bundle.aggregate())
                flat = bundle.flatten(include_scores=True, include_attributes=True, numeric_only=True)
                for k, v in flat.items():
                    # vector keys look like: "{alias}.{dimension_or_attr}"
                    vector[f"{model_alias}.{k}"] = float(v)
                vector[f"{model_alias}.aggregate"] = agg
                log.debug("[SP:score] ← %s alias=%s agg=%.4f added=%d in %s",
                          name, model_alias, agg, len(flat) + 1, self._t(t0))
                await asyncio.sleep(0)

            # Deterministic ordering
            metrics_columns = sorted(vector.keys())
            metrics_values  = [float(vector[c]) for c in metrics_columns]

            # Stash both the dict and ordered vector for downstream consumers
            acc["metrics_vector"]   = vector
            acc["metrics_columns"]  = metrics_columns
            acc["metrics_values"]   = metrics_values

            log.debug("[SP:score] done total_keys=%d in %s",
                      len(vector), self._t(t0_scores))
        else:
            log.debug("[SP:score] skipped (scoring=%s attach=%s)",
                      bool(self.scoring), bool(self.cfg.get("attach_scores", True)))

        # 6) Vision/VPM  → ALWAYS from metrics 
        # Require metrics unless cfg explicitly allows empty-VPM skip
        require_metrics = bool(self.cfg.get("require_metrics_for_vpm", True))
        have_metrics    = bool(acc.get("metrics_columns") and acc.get("metrics_values"))

        if self.zm and not acc.get("vision_signals"):
            if not have_metrics:
                msg = "[SP:vpm] missing metrics; cannot render VPM deterministically"
                if require_metrics:
                    log.error(msg + " (set require_metrics_for_vpm=false to skip VPM)")
                    raise RuntimeError("ScorableProcessor: VPM requested but metrics are missing")
                else:
                    log.debug(msg + " (skipping VPM per config)")
            else:
                t0 = time.perf_counter()
                log.debug("[SP:vpm] rendering VPM from metrics id=%s cols=%d",
                          scorable.id, len(acc["metrics_columns"]))
                # Preferred path: pass metrics explicitly to ZeroModel
                    # If you've patched ZeroModelService.vpm_from_scorable(**, metrics=...), use that:
                vpm_u8_chw, meta = await self.zm.vpm_from_scorable(
                    scorable,
                    metrics_values=metrics_values,
                    metrics_columns=metrics_columns,
                )

                acc["vision_signals"]      = vpm_u8_chw
                acc["vision_signals_meta"] = meta
                shape_desc = self._shape_dtype(vpm_u8_chw)
                log.debug("[SP:vpm] ok %s in %s (meta keys=%d)",
                          shape_desc, self._t(t0), len(meta or {}))
        else:
            log.debug("[SP:vpm] skipped (has_zm=%s, already=%s)",
                      bool(self.zm), bool(acc.get("vision_signals")))

        # 7) Row build
        t0 = time.perf_counter()
        row = self._build_features_row(scorable, acc)
        log.debug("[SP:row] built metrics=%d domains=%d ner=%d in %s",
                 len(row.get("metrics_columns") or []),
                 len(row.get("domains") or []),
                 len(row.get("ner") or []),
                 self._t(t0))

        # 8) Persist deltas
        for writer in self.writers:
            t0w = time.perf_counter()
            name = writer.__class__.__name__
            log.debug("[SP:persist] start %s id=%s", name, scorable.id)
            try:
                await writer.persist(scorable, acc)
                log.debug("[SP:persist] done %s in %s", name, self._t(t0w))
            except Exception as e:
                log.warning("[SP:persist] %s failed: %s", name, e)

        # 9) Manifest
        if self._manifest_features_path:
            await self.write_to_manifest(row)

        # 10) Cache + return
        self._cache[cache_key] = row
        log.debug("[SP:process] done id=%s in %s (cache.size=%d hit_rate=%.2f)",
                 scorable.id, self._t(t_all), len(self._cache),
                 self.get_cache_stats()["hit_rate"])

        return row

    async def process_many(
        self, inputs: List[Union[Scorable, Dict[str, Any]]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        t_all = time.perf_counter()
        n = len(inputs)
        log.debug("[SP:many] start batch_size=%d ctx.run=%s", n, context.get("pipeline_run_id"))

        # Normalize to Scorable & spot texts for prebatch
        texts_to_embed: List[str] = []
        idxs: List[int] = []
        norm: List[Scorable] = []

        for i, item in enumerate(inputs):
            sc = item if isinstance(item, Scorable) else ScorableFactory.from_dict(item)
            norm.append(sc)
            if sc.text and self.embed:
                texts_to_embed.append(sc.text)
                idxs.append(i)

        log.debug("[SP:many] prebatch candidates=%d has_batch_fn=%s",
                 len(idxs), bool(getattr(self.embed, 'get_or_create_batch', None)))

        # Batch embeddings only
        batched: Dict[int, List[float]] = {}
        batch_fn = getattr(self.embed, "get_or_create_batch", None)
        if texts_to_embed and callable(batch_fn):
            t0 = time.perf_counter()
            try:
                arrs = await batch_fn(texts_to_embed)  # -> List[np.ndarray]
                for i, arr in zip(idxs, arrs):
                    if arr is not None:
                        batched[i] = np.asarray(arr, dtype=np.float32).tolist()
                log.debug("[SP:many] prebatch ok count=%d in %s",
                         len(batched), self._t(t0))
            except Exception as e:
                log.debug("[SP:many] prebatch failed: %s", e)
        else:
            log.debug("[SP:many] prebatch skipped")

        # Process items
        out: List[Dict[str, Any]] = []
        for i, sc in enumerate(norm):
            t0i = time.perf_counter()
            if i in batched:
                log.debug("[SP:many:item] %d/%d id=%s (with prebatched embed)", i+1, n, sc.id)
                row = await self.process(
                    {
                        "id": sc.id,
                        "target_type": sc.target_type,
                        "text": sc.text,
                        "embeddings": {"global": batched[i]},
                        "metadata": sc.meta,
                        "domains": sc.domains,
                        "ner": sc.ner,
                    },
                    context,
                )
            else:
                log.debug("[SP:many:item] %d/%d id=%s", i+1, n, sc.id)
                row = await self.process(sc, context)

            out.append(row)
            log.debug("[SP:many:item] %d/%d done in %s", i+1, n, self._t(t0i))

        log.debug("[SP:many] done batch_size=%d in %s", n, self._t(t_all))
        return out

    def get_cache_stats(self) -> Dict[str, float]:
        total = self._cache_hits + self._cache_misses
        return {
            "hits": float(self._cache_hits),
            "misses": float(self._cache_misses),
            "total": float(total),
            "hit_rate": (self._cache_hits / total) if total > 0 else 0.0,
        }

    def _t(self, t0: float) -> str:
        return f"{(time.perf_counter() - t0)*1000:.1f}ms"

    def _shape_dtype(self, x: Any) -> str:
        try:
            import numpy as np
            if isinstance(x, np.ndarray):
                return f"ndarray shape={x.shape} dtype={x.dtype}"
            if isinstance(x, (list, tuple)):
                return f"{type(x).__name__}[{len(x)}]"
        except Exception:
            pass
        return type(x).__name__

    def _ensure_float_list(self, emb: Any) -> Optional[List[float]]:
        """Return a flat list[float] for any embedding input (np.ndarray, list, scalar)."""
        if emb is None:
            return None
        try:
            arr = np.asarray(emb, dtype=np.float32)
            if arr.ndim == 0:
                return [float(arr)]
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            return arr.astype(np.float32, copy=False).tolist()
        except Exception as e:
            log.warning("[SP:embed] normalize failed: %s (type=%s)", e, type(emb).__name__)
            return None
