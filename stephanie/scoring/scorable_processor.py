# stephanie/scoring/scorable_processor.py
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from stephanie.utils.json_sanitize import dumps_safe

log = logging.getLogger(__name__)

# -------------------- Data types --------------------

@dataclass
class GoalRef:
    text: str
    kind: str           # e.g. "turn_user", "system_goal", "case_goal"
    id: Optional[str] = None

# Canonical shapes for annotations
Domain = Dict[str, Any]  # normalized to: {"name": str, "score": float, "evidence": Optional[str], "centroid": Optional[str]}
Entity = Dict[str, Any]  # normalized to: {"text": str, "type": Optional[str], "span": Optional[Tuple[int,int]], "id": Optional[str]}


@dataclass
class ScorableFeatures:
    """
    Canonical, serializable feature record for any scorable.
    Thin features that make raw scorables comparable and learnable.
    """
    # identity
    scorable_id: str
    scorable_type: str                    # "conversation_turn", "goal", "plan_step", ...
    conversation_id: Optional[int] = None
    external_id: Optional[str] = None
    order_index: Optional[int] = None

    # core text
    text: str = ""
    title: Optional[str] = None
    near_identity: Dict[str, Any] = field(default_factory=dict)

    # annotations (normalized shapes)
    domains: List[Domain] = field(default_factory=list)   # [{"name": "science", "score": 0.88, ...}, ...]
    ner: List[Entity] = field(default_factory=list)       # [{"text": "Ernan", "type": "PERSON", "span": [0,5]}, ...]

    # free global signals
    ai_score: Optional[float] = None     # typically 0..100
    star: Optional[float] = None         # -5..+5 or 0..5
    goal_ref: Optional[GoalRef] = None

    # embeddings & metrics
    embeddings: Dict[str, List[float]] = field(default_factory=dict)  # {"global":[...], "goal":[...], ...}
    embed_global: Optional[np.ndarray] = None                         # transient cache; mirrored to embeddings["global"]
    metrics_columns: List[str] = field(default_factory=list)
    metrics_values: List[float] = field(default_factory=list)
    metrics_vector: Dict[str, float] = field(default_factory=dict)

    # optional agreement/stability
    agreement: Optional[float] = None  # 0..1
    stability: Optional[float] = None  # 0..1

    # lineage/context
    chat_id: Optional[int] = None
    turn_index: Optional[int] = None
    parent_scorable_id: Optional[str] = None
    parent_scorable_type: Optional[str] = None
    order_in_parent: Optional[int] = None

    # vision / VPM signals (NEW)
    vpm_png: Optional[str] = None
    vision_signals: Dict[str, Any] = field(default_factory=dict)  # e.g., {"vpm_score": 0.73, "channels": {...}}

    # rollout / misc
    rollout: Dict[str, Any] = field(default_factory=dict)

    # provenance
    processor_version: str = "1.1"
    content_hash16: Optional[str] = None  # sha256(text)[:16] for stable identity
    created_utc: float = field(default_factory=lambda: time.time())

    # --------- Serialization helpers ---------

    def to_manifest_row(self) -> Dict[str, Any]:
        """
        Convert to a JSON-serializable dict. Includes an 'embed_global' list
        and a copy in embeddings["global"] for redundancy/consistency.
        """
        row = asdict(self)

        # GoalRef -> dict
        if isinstance(self.goal_ref, GoalRef):
            row["goal_ref"] = asdict(self.goal_ref)

        # Ensure embeddings["global"] is present if embed_global exists
        if self.embed_global is not None:
            try:
                gl = self.embed_global.astype(np.float32).tolist()
            except Exception:
                gl = self.embed_global.tolist()
            row["embed_global"] = gl
            row.setdefault("embeddings", {})
            if not row["embeddings"].get("global"):
                row["embeddings"]["global"] = gl

        # NumPy -> native
        for k, v in list(row.items()):
            if isinstance(v, np.ndarray):
                row[k] = v.tolist()
            elif isinstance(v, (np.floating,)):
                row[k] = float(v)
            elif isinstance(v, (np.integer,)):
                row[k] = int(v)

        return row
    
# -------------------- Processor --------------------

class ScorableProcessor:
    """
    One stop: turn an arbitrary scorable dict into ScorableFeatures.
    Wires embeddings, optional NER/domain/agree/stability, vision signals, and supports JSONL manifest writes.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # Optional dependencies (safe defaults)
        self.scoring               = container.get("scoring", None)
        self.zeromodel             = container.get("zeromodel", None)
        self.entity_extractor      = container.get("entity_extractor", None)
        self.domain_classifier     = container.get("domain_classifier", None)
        self.agreement_estimator   = container.get("agreement_estimator", None)
        self.stability_estimator   = container.get("stability_estimator", None)
        self.vision_signal_builder = container.get("vision_signal_builder", None)  # e.g., VPM scorer

        # Cache & manifest
        self._cache: Dict[str, ScorableFeatures] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self._current_manifest_path: Optional[Path] = None
        self._manifest_lock = asyncio.Lock()  # async-safe appends

    # -------- Manifest control --------

    def _generate_cache_key(self, scorable: Dict[str, Any]) -> str:
        stype = str(scorable.get("target_type") or scorable.get("type") or "unknown")
        sid = scorable.get("id") or scorable.get("scorable_id") or ""
        text = scorable.get("text") or scorable.get("body") or ""
        content_hash = sha256(text.encode("utf-8")).hexdigest()[:16]
        # include type for fewer collisions across namespaces
        return f"{stype}:{sid or 'noid'}:{content_hash}"

    def start_manifest(self, manifest_path: Union[str, Path]):
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        header = {
            "event": "manifest_start",
            "created_utc": time.time(),
            "processor_version": "1.1",
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(dumps_safe(header, indent=2) + "\n")

        self._current_manifest_path = path
        log.info(f"[ScorableProcessor] Started manifest at {path}")

    async def write_to_manifest(self, features: ScorableFeatures):
        if not self._current_manifest_path:
            raise RuntimeError("No manifest has been started. Call start_manifest() first.")
        async with self._manifest_lock:
            row = features.to_manifest_row()
            with open(self._current_manifest_path, "a", encoding="utf-8") as f:
                f.write(dumps_safe(row, indent=2) + "\n")

    # -------- Utilities --------

    @staticmethod
    def _normalize_domains(raw: Any) -> List[Domain]:
        out: List[Domain] = []
        if not raw:
            return out
        if isinstance(raw, dict):  # {"science": 0.9, ...}
            for k, v in raw.items():
                try:
                    out.append({"name": str(k), "score": float(v)})
                except Exception:
                    out.append({"name": str(k), "score": 0.0})
            return out
        for item in raw:
            if isinstance(item, str):
                out.append({"name": item, "score": 1.0})
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("domain") or "unknown")
                score = float(item.get("score", 1.0))
                d = {"name": name, "score": score}
                if "evidence" in item: d["evidence"] = item["evidence"]
                if "centroid" in item: d["centroid"] = item["centroid"]
                out.append(d)
        return out

    @staticmethod
    def _normalize_entities(raw: Any) -> List[Entity]:
        out: List[Entity] = []
        if not raw:
            return out
        if isinstance(raw, dict):  # possibly {"Ernan":"PERSON", ...}
            for k, v in raw.items():
                out.append({"text": str(k), "type": str(v) if v else None})
            return out
        for item in raw:
            if isinstance(item, str):
                out.append({"text": item})
            elif isinstance(item, dict):
                ent = {"text": str(item.get("text") or item.get("entity") or "")}
                if "type" in item: ent["type"] = item["type"]
                if "span" in item: ent["span"] = item["span"]
                if "id" in item:   ent["id"] = item["id"]
                out.append(ent)
        return out

    async def _compute_global_embedding(self, text: str) -> Optional[np.ndarray]:
        if not text or not text.strip():
            return None
        # Defer to your embedder; returns np.ndarray(float32)
        try:
            return await self.memory.embedding.get_or_create(text)
        except Exception as e:
            log.exception("Embedding failed: %s", e)
            return None

    # -------- Core processing (single) --------

    async def process(self, scorable: Dict[str, Any]) -> ScorableFeatures:
        """
        Process one scorable dict into ScorableFeatures.
        Populates: embeddings (global), domains/entities (normalized),
        agreement/stability (if available), vision signals, free signals, and lineage.
        """
        cache_key = self._generate_cache_key(scorable)
        if cache_key in self._cache:
            self._cache_hits += 1
            feat = self._cache[cache_key]
            log.debug(f"[ScorableProcessor] Cache HIT for {feat.scorable_type}:{feat.scorable_id}")
            return feat
        self._cache_misses += 1
        log.debug(f"[ScorableProcessor] Cache MISS for {scorable.get('id')}")

        # Identity + text
        stype = str(scorable.get("target_type") or scorable.get("type") or "unknown")
        sid   = str(scorable.get("id") or scorable.get("scorable_id") or "")
        text  = scorable.get("text") or scorable.get("body") or ""
        title = (
            scorable.get("title")
            or (scorable.get("near_identity") or {}).get("title")
            or (text[:80] if text else sid or stype)
        )
        content_hash16 = sha256((text or "").encode("utf-8")).hexdigest()[:16]

        # Embeddings: prefer provided, else compute
        embeddings: Dict[str, List[float]] = dict(scorable.get("embeddings") or {})
        embed_global: Optional[np.ndarray] = None
        gl = embeddings.get("global")
        if isinstance(gl, list) and gl:
            try:
                embed_global = np.asarray(gl, dtype=np.float32)
            except Exception:
                embed_global = None
        if embed_global is None:
            embed_global = await self._compute_global_embedding(text)
            if embed_global is not None:
                embeddings["global"] = embed_global.astype(np.float32).tolist()

        # Metrics: reconcile columns/values/vector
        mx_cols = list(scorable.get("metrics_columns") or [])
        mx_vals = [float(v) for v in (scorable.get("metrics_values") or [])]
        mx_vec  = {k: float(v) for k, v in (scorable.get("metrics_vector") or {}).items()}
        if not mx_vec and mx_cols and mx_vals and len(mx_cols) == len(mx_vals):
            mx_vec = {k: float(v) for k, v in zip(mx_cols, mx_vals)}

        # Domains (normalize or infer)
        domains = self._normalize_domains(scorable.get("domains"))
        if not domains and self.domain_classifier and text:
            try:
                # expect [["science", 0.88], ...] OR dict OR list[str]
                inferred = await self.domain_classifier.predict(text)
                domains = self._normalize_domains(inferred)
            except Exception:
                log.debug("Domain classifier failed; proceeding without domains")

        # Entities (normalize or extract)
        entities = self._normalize_entities(scorable.get("ner"))
        if not entities and self.entity_extractor and text:
            try:
                extracted = await self.entity_extractor.extract(text)
                entities = self._normalize_entities(extracted)
            except Exception:
                log.debug("Entity extractor failed; proceeding without NER")

        # Agreement/Stability
        agreement = scorable.get("agreement")
        if agreement is None and self.agreement_estimator:
            try:
                agreement = await self.agreement_estimator.estimate(scorable)
            except Exception:
                agreement = None

        stability = scorable.get("stability")
        if stability is None and self.stability_estimator:
            try:
                stability = await self.stability_estimator.estimate(scorable)
            except Exception:
                stability = None

        # Free signals
        ai_score = scorable.get("ai_score")
        star     = scorable.get("star")

        # Vision signals (VPM/image-based)
        vision_signals = dict(scorable.get("vision_signals") or {})
        if not vision_signals and self.vision_signal_builder:
            try:
                # contract: builder may look at scorable (e.g., vpm tensors, image path) to produce signals
                vision_signals = await self.vision_signal_builder.build(scorable)
            except Exception:
                log.debug("Vision signal builder failed; proceeding without vision signals")

        # GoalRef
        goal_ref = None
        goal_ref_raw = scorable.get("goal_ref")
        if isinstance(goal_ref_raw, dict) and "text" in goal_ref_raw:
            goal_ref = GoalRef(
                text=str(goal_ref_raw.get("text", "")),
                kind=str(goal_ref_raw.get("kind", "turn_user")),
                id=goal_ref_raw.get("id"),
            )
        else:
            umsg = scorable.get("user_text") or scorable.get("goal_text")
            if isinstance(umsg, str) and umsg.strip():
                goal_ref = GoalRef(text=umsg.strip(), kind="turn_user")

        # Lineage
        conversation_id = scorable.get("conversation_id")
        order_index     = scorable.get("order_index")
        chat_id         = scorable.get("chat_id", conversation_id)
        turn_index      = scorable.get("turn_index", order_index)
        parent_sid      = scorable.get("parent_scorable_id")
        parent_stype    = scorable.get("parent_scorable_type")
        order_in_parent = scorable.get("order_in_parent")

        # Build features
        features = ScorableFeatures(
            scorable_id=sid or f"{stype}:{content_hash16}",
            scorable_type=stype,
            conversation_id=conversation_id,
            external_id=scorable.get("external_id"),
            order_index=order_index,

            text=text or "",
            title=title,
            near_identity=scorable.get("near_identity") or {},

            domains=domains,
            ner=entities,

            ai_score=(float(ai_score) if ai_score is not None else None),
            star=(float(star) if star is not None else None),
            goal_ref=goal_ref,

            embeddings=embeddings,
            embed_global=embed_global,

            metrics_columns=mx_cols,
            metrics_values=mx_vals,
            metrics_vector=mx_vec,

            agreement=(float(agreement) if agreement is not None else None),
            stability=(float(stability) if stability is not None else None),

            chat_id=chat_id,
            turn_index=turn_index,
            parent_scorable_id=parent_sid,
            parent_scorable_type=parent_stype,
            order_in_parent=order_in_parent,

            vpm_png=scorable.get("vpm_png"),
            vision_signals=vision_signals,
            rollout=scorable.get("rollout") or {},

            processor_version="1.1",
            content_hash16=content_hash16,
        )

        # Cache
        self._cache[cache_key] = features

        # Optional: write JSONL row
        if self._current_manifest_path:
            await self.write_to_manifest(features)

        return features

    # -------- Core processing (batch) --------

    async def process_many(self, scorables: List[Dict[str, Any]]) -> List[ScorableFeatures]:
        """
        Batch-friendly processing with best-effort batched embeddings.
        Falls back to per-item if your embedder doesn't support batch.
        """
        # Identify which need embeddings
        need_embed = []
        texts = []
        for s in scorables:
            gl = (s.get("embeddings") or {}).get("global")
            text = s.get("text") or s.get("body") or ""
            if not (isinstance(gl, list) and gl) and text:
                need_embed.append(s); texts.append(text)

        # Try batch embedding if available
        batched = {}
        embedder = getattr(self.memory.embedding, "get_or_create_batch", None)
        if texts and callable(embedder):
            try:
                arrs = await embedder(texts)  # -> List[np.ndarray]
                for s, a in zip(need_embed, arrs):
                    if a is not None:
                        batched[id(s)] = a
            except Exception:
                log.debug("Batch embedding failed; will compute item-wise")

        # Process each
        out: List[ScorableFeatures] = []
        for s in scorables:
            # stash batch result so process() can pick it up
            if id(s) in batched:
                s.setdefault("embeddings", {})
                s["embeddings"]["global"] = np.asarray(batched[id(s)], dtype=np.float32).tolist()
            out.append(await self.process(s))
        return out

    # -------- Stats --------

    def get_cache_stats(self) -> Dict[str, float]:
        total = self._cache_hits + self._cache_misses
        return {
            "hits": float(self._cache_hits),
            "misses": float(self._cache_misses),
            "total": float(total),
            "hit_rate": (self._cache_hits / total) if total > 0 else 0.0,
        }
