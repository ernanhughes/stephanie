# stephanie/scoring/scorable_processor.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from hashlib import sha256
import asyncio
import time

import numpy as np

from stephanie.utils.json_sanitize import dumps_safe

import logging
log = logging.getLogger(__name__)

# -------------------- Data types --------------------

@dataclass
class GoalRef:
    text: str
    kind: str           # e.g. "turn_user", "system_goal", "case_goal"
    id: Optional[str] = None


@dataclass
class ScorableFeatures:
    """
    Canonical, serializable feature record for any scorable.
    """
    # identity
    scorable_id: str
    scorable_type: str                   # "conversation_turn", "goal", "plan_step", ...
    conversation_id: Optional[int] = None
    external_id: Optional[str] = None
    order_index: Optional[int] = None

    # core text
    text: str = ""
    title: Optional[str] = None
    near_identity: Dict[str, Any] = field(default_factory=dict)

    # annotations
    domains: List[Dict[str, Any]] = field(default_factory=list)    # or strings; keep flexible
    ner: List[Dict[str, Any]] = field(default_factory=list)   # or strings

    # free global signals
    ai_score: Optional[float] = None   # typically 0..100
    star: Optional[float] = None       # -5..+5 or 0..5; keep raw here
    goal_ref: Optional[GoalRef] = None

    # embeddings & metrics
    embeddings: Dict[str, List[float]] = field(default_factory=dict)   # {"global":[...], "goal":[...], ...}
    embed_global: Optional[np.ndarray] = None                           # transient: np.float32 vector (redundant with embeddings["global"])
    metrics_columns: List[str] = field(default_factory=list)
    metrics_values: List[float] = field(default_factory=list)
    metrics_vector: Dict[str, float] = field(default_factory=dict)

    # optional agreement/stability
    agreement: Optional[float] = None  # 0..1
    stability: Optional[float] = None  # 0..1

    # lineage/context
    chat_id: Optional[int] = None
    turn_index: Optional[int] = None

    # artifacts
    vpm_png: Optional[str] = None
    rollout: Dict[str, Any] = field(default_factory=dict)

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
                gl = self.embed_global.tolist()  # best effort
            row["embed_global"] = gl
            row.setdefault("embeddings", {})
            if "global" not in row["embeddings"] or not row["embeddings"]["global"]:
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
    Wires embeddings, optional NER/domain/agree/stability, and supports JSONL manifest writes.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        self.scoring     = container.get("scoring")      
        self.zeromodel   = container.get("zeromodel")    
        # self.entity_extractor    = container.get("entity_extractor", None)
        # self.domain_classifier   = container.get("domain_classifier", None)
        # self.agreement_estimator = container.get("agreement_estimator", None)
        # self.stability_estimator = container.get("stability_estimator", None)

        self._cache: Dict[str, ScorableFeatures] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self._current_manifest_path: Optional[Path] = None
        self._manifest_lock = asyncio.Lock()  # async-safe appends

    # -------- Manifest control --------

    def _generate_cache_key(self, scorable: Dict[str, Any]) -> str:
        sid = scorable.get("id") or scorable.get("scorable_id") or "unknown"
        text = scorable.get("text") or scorable.get("body") or ""
        content_hash = sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{sid}:{content_hash}"

    def start_manifest(self, manifest_path: Union[str, Path]):
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        header = {
            "event": "manifest_start",
            "created_utc": time.time(),
            "processor_version": "1.0",
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

    # -------- Core processing --------

    async def process(self, scorable: Dict[str, Any]) -> ScorableFeatures:
        """
        Process a scorable dict into ScorableFeatures.
        Populates: embeddings (global), embed_global, domains/entities (if needed),
        agreement/stability (if available), free signals (ai_score, star, goal_ref), and lineage.
        """
        cache_key = self._generate_cache_key(scorable)
        if cache_key in self._cache:
            self._cache_hits += 1
            feat = self._cache[cache_key]
            log.debug(f"[ScorableProcessor] Cache HIT for scorable {feat.scorable_id}")
            return feat
        self._cache_misses += 1
        log.debug(f"[ScorableProcessor] Cache MISS for scorable {scorable.get('id')}")

        # Identity + text
        sid   = scorable.get("id") or scorable.get("scorable_id") or ""
        stype = str(scorable.get("target_type") or scorable.get("type") or "unknown")
        text  = scorable.get("text") or scorable.get("body") or ""
        title = (
            scorable.get("title")
            or (scorable.get("near_identity") or {}).get("title")
            or (text[:80] if text else str(sid))
        )

        # Embedding(s)
        # Prefer existing embeddings if provided; else compute.
        embeddings: Dict[str, List[float]] = dict(scorable.get("embeddings") or {})
        embed_global: Optional[np.ndarray] = None
        if "global" in embeddings and isinstance(embeddings["global"], list):
            try:
                embed_global = np.asarray(embeddings["global"], dtype=np.float32)
            except Exception:
                embed_global = None

        if embed_global is None:
            # compute via memory.embedding (your existing embedder)
            embed_global = await self.memory.embedding.get_or_create(text)  # np.ndarray(float32)
            embeddings["global"] = embed_global.astype(np.float32).tolist()

        # Metrics (pass-through)
        mx_cols = list(scorable.get("metrics_columns") or [])
        mx_vals = [float(v) for v in (scorable.get("metrics_values") or [])]
        mx_vec  = {k: float(v) for k, v in (scorable.get("metrics_vector") or {}).items()}

        # Domains
        domains = list(scorable.get("domains") or [])
        if not domains and self.domain_classifier:
            try:
                domains = await self.domain_classifier.predict(text)
            except Exception:
                pass

        # Entities
        ner = scorable.get("ner")
        if isinstance(ner, dict):
            ner = list(ner.keys())
        entities = list(ner or [])
        if not entities and self.entity_extractor:
            try:
                entities = await self.entity_extractor.extract(text)
            except Exception:
                pass

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

        # GoalRef
        goal_ref_raw = scorable.get("goal_ref")
        goal_ref: Optional[GoalRef] = None
        if isinstance(goal_ref_raw, dict) and "text" in goal_ref_raw:
            goal_ref = GoalRef(
                text=str(goal_ref_raw.get("text", "")),
                kind=str(goal_ref_raw.get("kind", "turn_user")),
                id=goal_ref_raw.get("id"),
            )
        elif not goal_ref_raw:
            # Heuristic: if we have a user message in context, treat as goal
            umsg = scorable.get("user_text") or scorable.get("goal_text")
            if isinstance(umsg, str) and umsg.strip():
                goal_ref = GoalRef(text=umsg.strip(), kind="turn_user")

        # Lineage
        conversation_id = scorable.get("conversation_id")
        order_index     = scorable.get("order_index")
        chat_id         = scorable.get("chat_id", conversation_id)
        turn_index      = scorable.get("turn_index", order_index)

        # Build features
        features = ScorableFeatures(
            scorable_id=str(sid),
            scorable_type=stype,
            conversation_id=conversation_id,
            external_id=scorable.get("external_id"),
            order_index=order_index,

            text=text,
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

            vpm_png=scorable.get("vpm_png"),
            rollout=scorable.get("rollout") or {},
        )

        # Cache
        self._cache[cache_key] = features

        # Optional: write JSONL row
        if self._current_manifest_path:
            await self.write_to_manifest(features)

        return features

    # -------- Stats --------

    def get_cache_stats(self) -> Dict[str, float]:
        total = self._cache_hits + self._cache_misses
        return {
            "hits": float(self._cache_hits),
            "misses": float(self._cache_misses),
            "total": float(total),
            "hit_rate": (self._cache_hits / total) if total > 0 else 0.0,
        }
