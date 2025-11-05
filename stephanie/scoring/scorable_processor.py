# stephanie/scoring/scorable_processor.py
from __future__ import annotations

from stephanie.utils.json_sanitize import dumps_safe
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Union
import numpy as np
from pathlib import Path
from hashlib import sha256
import asyncio


@dataclass
class GoalRef:
    text: str
    kind: str           # e.g. "turn_user", "system_goal", "case_goal"
    id: Optional[str] = None

@dataclass
class ScorableFeatures:
    # identity
    scorable_id: str                     # stable id (row id, uuid, etc.)
    scorable_type: str                   # "conversation_turn", "goal", "plan_step", ...
    conversation_id: Optional[int]       # when applicable
    external_id: Optional[str] = None    # optional cross-ref (github sha, paper id, etc.)
    order_index: Optional[int] = None    # monotonic index within conversation

    # core text
    text: str
    title: Optional[str] = None
    near_identity: Dict[str, Any] = field(default_factory=dict)

    # annotations
    domains: List[Dict[str, Any]] = field(default_factory=list)   # [{domain: "...", score: ...}, ...] ok to be strings too
    entities: List[Dict[str, Any]] = field(default_factory=list)  # or strings; keep flexible

    # “free” global signals you mentioned
    ai_score: Optional[float] = None        # 0..100 (normalize internally to 0..1 if needed)
    star: Optional[int] = None              # user star rating (e.g., -5..+5 or 0..5); we’ll normalize too
    goal_ref: Optional[GoalRef] = None      # the goal this scorable serves (often the user message for assistant turn)

    # embeddings & metrics
    embeddings: Dict[str, List[float]] = field(default_factory=dict)   # {"global": [...], "goal": [...], ...}
    metrics_columns: List[str] = field(default_factory=list)
    metrics_values: List[float] = field(default_factory=list)
    metrics_vector: Dict[str, float] = field(default_factory=dict)     # flattened for fast joins

    # optional agreement/stability
    agreement: Optional[float] = None      # 0..1 (e.g., HRM vs Tiny agreement)
    stability: Optional[float] = None      # 0..1 (inverse halluc/uncertainty)

    # lineage/context
    chat_id: Optional[int] = None
    turn_index: Optional[int] = None

    # artifacts
    vpm_png: Optional[str] = None
    rollout: Dict[str, Any] = field(default_factory=dict)

    def to_manifest_row(self) -> Dict[str, Any]:
        """
        Convert this feature set into a single row for a manifest report.
        Handles non-serializable types like numpy arrays.
        """
        row = asdict(self)
        
        # Convert numpy array to list for JSON serialization
        if self.embed_global is not None:
            row['embed_global'] = self.embed_global.tolist()
            
        # Ensure all values are JSON serializable
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                row[key] = value.tolist()
            elif isinstance(value, np.floating):
                row[key] = float(value)
            elif isinstance(value, np.integer):
                row[key] = int(value)
                
        return row

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScorableFeatures':
        """
        Create a ScorableFeatures instance from a dictionary.
        Handles reconstruction of numpy arrays and ensures type safety.
        """
        # Reconstruct numpy array from list if present
        embed_data = data.get('embed_global')
        if embed_data is not None and isinstance(embed_data, list):
            data['embed_global'] = np.array(embed_data, dtype=np.float32)
        elif embed_data is not None and isinstance(embed_data, np.ndarray):
            # Already an array, ensure correct type
            data['embed_global'] = data['embed_global'].astype(np.float32)
            
        # Ensure optional fields have correct types or None
        for field_name in ['agreement', 'stability']:
            val = data.get(field_name)
            if val is not None:
                data[field_name] = float(val)
                
        # Provide defaults for lists that might be missing
        for list_field in ['domains', 'entities', 'metrics_columns', 'metrics_values']:
            if list_field not in data or data[list_field] is None:
                data[list_field] = []
                
        # Provide defaults for dicts
        for dict_field in ['near_identity', 'metrics_vector', 'rollout']:
            if dict_field not in data or data[dict_field] is None:
                data[dict_field] = {}
                
        # Ensure required string fields have a default
        for str_field in ['scorable_id', 'scorable_type', 'text', 'title']:
            if str_field not in data or data[str_field] is None:
                data[str_field] = ""
                
        return cls(**data)


class ScorableProcessor:
    """
    One stop: turn an arbitrary scorable dict into ScorableFeatures.
    Wire your existing services here: embeddings, scoring, near-identity, NER/entities, etc.
    
    Features:
    - Caching: Prevents redundant processing of the same scorable.
    - Manifest Writing: Appends processed features to a running manifest file.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.log = logger
        self.scoring  = container.get("scoring")           # ScoringService
        self.zeromodel= container.get("zeromodel")         # ZeroModelService
        
        # optional: entity extractor, domain classifier, agreement estimator…
        self.entity_extractor = container.get("entity_extractor", None)
        self.domain_classifier = container.get("domain_classifier", None)
        self.agreement_estimator = container.get("agreement_estimator", None)
        self.stability_estimator = container.get("stability_estimator", None)

        self._cache: Dict[str, ScorableFeatures] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self._current_manifest_path: Optional[Path] = None
        self._manifest_lock = asyncio.Lock() # For async safety if used concurrently

    def _generate_cache_key(self, scorable: Dict[str, Any]) -> str:
        """
        Generate a deterministic cache key based on the scorable's content and id.
        This ensures identical scorables get the same key.
        """
        # Combine scorable ID and a hash of its text/content
        sid = scorable.get("id") or scorable.get("scorable_id") or "unknown"
        text = scorable.get("text") or scorable.get("body") or ""
        content_hash = sha256(text.encode('utf-8')).hexdigest()[:16]
        return f"{sid}:{content_hash}"

    def start_manifest(self, manifest_path: Union[str, Path]):
        """
        Initialize a new manifest file at the given path.
        Creates the directory if it doesn't exist and writes a header.
        """
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write a header or initial metadata
        header = {
            "event": "manifest_start",
            "created_utc": asyncio.get_event_loop().time(),
            "processor_version": "1.0" # Could be tied to your code version
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(dumps_safe(header, indent=2) + '\n')
            
        self._current_manifest_path = path
        self.log.info(f"Started manifest at {path}")

    async def write_to_manifest(self, features: ScorableFeatures):
        """
        Append a single ScorableFeatures record to the current manifest file.
        The manifest is a JSONL (JSON Lines) file, where each line is a separate JSON object.
        """
        if not self._current_manifest_path:
            raise RuntimeError("No manifest has been started. Call start_manifest() first.")
            
        async with self._manifest_lock:
            try:
                row = features.to_manifest_row()
                with open(self._current_manifest_path, 'a', encoding='utf-8') as f:
                    f.write(dumps_safe(row, indent=2) + '\n')
            except Exception as e:
                self.log.error(f"Failed to write to manifest {self._current_manifest_path}: {e}")
                raise

    async def process(self, scorable: Dict[str, Any]) -> ScorableFeatures:
        """
        Process a scorable into features, using cache and optionally writing to a manifest.
        """
        cache_key = self._generate_cache_key(scorable)
        
        # --- Check Cache First ---
        if cache_key in self._cache:
            self._cache_hits += 1
            features = self._cache[cache_key]
            self.log.debug(f"Cache HIT for scorable {features.scorable_id}")
            return features
            
        self._cache_misses += 1
        self.log.debug(f"Cache MISS for scorable {scorable.get('id')}")

        # --- Process the scorable (your existing logic) ---
        sid  = scorable.get("id") or scorable.get("scorable_id")
        text = scorable.get("text") or scorable.get("body") or ""
        title= (scorable.get("title") or scorable.get("near_identity",{}).get("title")
                or text[:80] or str(sid))

        # Embeddings
        embed = await self.memory.embedding.get_or_create(text)  # np.ndarray (float32)
        
        # Metrics
        mx_cols = list(scorable.get("metrics_columns") or [])
        mx_vals = [float(v) for v in (scorable.get("metrics_values") or [])]
        mx_vec  = {k: float(v) for k, v in (scorable.get("metrics_vector") or {}).items()}

        # Domains/entities
        domains = list(scorable.get("domains") or [])
        if not domains and self.domain_classifier:
            domains = await self.domain_classifier.predict(text)

        ents = scorable.get("entities")
        if isinstance(ents, dict): 
            ents = list(ents.keys())
        entities = list(ents or [])
        if not entities and self.entity_extractor:
            entities = await self.entity_extractor.extract(text)

        # Agreement/stability
        agreement = scorable.get("agreement") 
        if agreement is None and self.agreement_estimator:
            agreement = await self.agreement_estimator.estimate(scorable)
        stability = scorable.get("stability")
        if stability is None and self.stability_estimator:
            stability = await self.stability_estimator.estimate(scorable)

        # --- Create the features object ---
        features = ScorableFeatures(
            scorable_id=sid,
            scorable_type=str(scorable.get("target_type") or scorable.get("type") or "unknown"),
            text=text,
            title=title,
            chat_id=scorable.get("chat_id"),
            turn_index=scorable.get("turn_index"),
            domains=domains,
            entities=entities,
            near_identity=scorable.get("near_identity") or {},
            embed_global=embed,
            metrics_columns=mx_cols,
            metrics_values=mx_vals,
            metrics_vector=mx_vec,
            agreement=(float(agreement) if agreement is not None else None),
            stability=(float(stability) if stability is not None else None),
            vpm_png=scorable.get("vpm_png"),
            rollout=scorable.get("rollout") or {},
        )

        # --- Cache the result ---
        self._cache[cache_key] = features

        # --- Optionally write to the manifest ---
        if self._current_manifest_path:
            await self.write_to_manifest(features)

        return features

    def get_cache_stats(self) -> Dict[str, int]:
        """Return hit/miss statistics for monitoring."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": self._cache_hits + self._cache_misses,
            "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0.0
        }
 