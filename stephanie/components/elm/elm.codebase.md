# Project Context: elm
# Path: C:\Users\ernan\Project\stephanie\stephanie\components\elm
# Generated for AI Review


==================================================
FILE: axes.py
==================================================

# components/elm/axes.py

from enum import Enum


class AxisDirection(str, Enum):
    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"


AXIS_SEMANTICS = {
    "hrm_alignment": AxisDirection.HIGHER_IS_BETTER,
    "hallucination_energy": AxisDirection.LOWER_IS_BETTER,
    "embedding_margin": AxisDirection.HIGHER_IS_BETTER,
    "policy_advantage": AxisDirection.HIGHER_IS_BETTER,
    "metric_alignment": AxisDirection.HIGHER_IS_BETTER,
    "coherence": AxisDirection.HIGHER_IS_BETTER,
    "context_fidelity": AxisDirection.HIGHER_IS_BETTER,
}


==================================================
FILE: config.py
==================================================

# ELM configuration definitions


==================================================
FILE: evaluator.py
==================================================



==================================================
FILE: __init__.py
==================================================

# stephanie/components/elm/__init__.py
"""
ELM: Experimental Learning Module for governed self-improvement.
"""

# Core primitives
from stephanie.components.elm.orchestration.system_interface import SystemInterface
from .core.context_pack import (
    ContextPack,
    ContextPackCollection,
    ContextType,
    Modality,
)
from .core.thresholds import (
    CalibratedThresholds,
)

# Tracking & diagnostics
from .tracking.retention_tracker import RetentionTracker, RetentionMetrics
from .tracking.collapse_detector import CollapseDetector, FailureEvent

# Governance layer
from .governance.signal_extractor import GovernanceSignalExtractor
from .governance.dominance_checker import DominanceChecker
from .governance.regime_controller import RegimeController

# Evaluation infrastructure
from .evaluation.governance_scorer import GovernanceScorer

# Experiment harness
from .experiment.baseline_calibrator import BaselineCalibrator
from .experiment.experiment_harness import ScoreBundleExperiment
from .experiment.experiment_persistence import ExperimentPersistence
from .experiment.perturbation_injector import (
    PerturbationInjector,
    PerturbationConfig,
    create_perturbation_config,
    register_custom_severity
)
from .orchestration.orchestrator import ELMOrchestrator
from .orchestration.system_interface import  SystemInterface

__all__ = [
    # Core
    "ContextPack",
    "ContextPackCollection",
    "ContextType",
    "Modality",
    "RewardVector",
    "RewardAxis",
    "CalibratedThresholds",
    
    # Tracking
    "RetentionTracker",
    "RetentionMetrics",
    "CollapseDetector",
    "FailureEvent",
    
    # Governance
    "GovernanceSignalExtractor",
    "DominanceChecker",
    "RegimeController",
    
    # Evaluation
    "GovernanceScorer",

    # Experiment
    "BaselineCalibrator",
    "ScoreBundleExperiment",
    "ExperimentPersistence",
    "PerturbationInjector",
    "PerturbationInjector",
    "PerturbationConfig",
    "create_perturbation_config",
    "register_custom_severity",
    
    # Orchestration
    "ELMOrchestrator",
    "SystemInterface",
]

==================================================
FILE: core\context_pack.py
==================================================

"""
ContextPack: Unified context container for Stephanie's cognitive architecture.

Provides:
- Multi-modal context aggregation
- Provenance tracking
- Schema validation
- Serialization/deserialization
- Extension hooks
- Type-safe accessors
- Immutable core with mutable extensions
"""

from __future__ import annotations

import json
import uuid
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Union, Callable, Protocol, runtime_checkable
)
from typing_extensions import TypedDict
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class ContextType(str, Enum):
    """Context classification for routing and processing"""
    USER_QUERY = "user_query"
    DOCUMENT = "document"
    GOAL = "goal"
    TASK = "task"
    REFLECTION = "reflection"
    MEMORY = "memory"
    TOOL_OUTPUT = "tool_output"
    EXTERNAL_KNOWLEDGE = "external_knowledge"
    SYSTEM_STATE = "system_state"
    METADATA = "metadata"


class Modality(str, Enum):
    """Modalities supported by ContextPack"""
    TEXT = "text"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"
    STRUCTURED = "structured"
    CODE = "code"
    MIXED = "mixed"


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects"""
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable": ...


class ContextMetadata(TypedDict, total=False):
    """Type-safe metadata structure"""
    source: str
    confidence: float
    timestamp: float
    version: str
    schema: str
    tags: List[str]
    provenance: List[str]
    priority: int
    ttl: Optional[float]  # Time to live in seconds


# ============================================================================
# CORE CONTEXT PACK
# ============================================================================

@dataclass
class ContextPack:
    """
    Unified context container for Stephanie's cognitive operations.
    
    Design principles:
    - Immutable core fields (hash-based identity)
    - Extensible metadata (mutable)
    - Type-safe accessors
    - Provenance tracking
    - Serialization support
    - Multi-modal support
    - Schema validation
    
    Usage:
        pack = ContextPack(
            content="User query text",
            context_type=ContextType.USER_QUERY,
            modality=Modality.TEXT
        )
        
        # Add metadata
        pack.add_metadata("source", "user_input")
        pack.add_tag("priority:high")
        
        # Access with type safety
        text = pack.get_text()
        emb = pack.get_embedding()  # Returns None if not present
    """
    
    # Core immutable fields (determine identity)
    content: Any  # Primary content (text, embedding, structured data)
    context_type: ContextType
    modality: Modality
    
    # Optional core fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    
    # Mutable metadata (extensible)
    _metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Provenance tracking
    _provenance: List[str] = field(default_factory=list)
    
    # Schema validation
    _schema_version: str = "1.0"
    _validated: bool = False
    
    # Extension registry
    _extensions: Dict[str, Any] = field(default_factory=dict)
    
    # Cache for computed properties
    _hash: Optional[str] = None
    _signature: Optional[str] = None
    
    def __post_init__(self):
        """Validate and initialize"""
        self._validate_content()
        self._validated = True
        self._compute_hash()
    
    # ============================================================================
    # CORE ACCESSORS
    # ============================================================================
    
    def get_text(self) -> Optional[str]:
        """Get text content if available"""
        if self.modality == Modality.TEXT:
            return str(self.content)
        elif isinstance(self.content, dict) and "text" in self.content:
            return str(self.content["text"])
        return None
    
    def get_embedding(self) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """Get embedding if available"""
        if self.modality == Modality.EMBEDDING:
            if isinstance(self.content, (np.ndarray, torch.Tensor)):
                return self.content
            elif isinstance(self.content, dict) and "embedding" in self.content:
                return self.content["embedding"]
        return None
    
    def get_structured(self) -> Optional[Dict[str, Any]]:
        """Get structured data if available"""
        if self.modality == Modality.STRUCTURED:
            if isinstance(self.content, dict):
                return self.content
            elif hasattr(self.content, "__dict__"):
                return self.content.__dict__
        return None
    
    def get_image(self) -> Optional[Any]:
        """Get image data if available"""
        if self.modality == Modality.IMAGE:
            return self.content
        return None
    
    def get_code(self) -> Optional[str]:
        """Get code content if available"""
        if self.modality == Modality.CODE:
            return str(self.content)
        return None
    
    # ============================================================================
    # METADATA MANAGEMENT
    # ============================================================================
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        return self._metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> "ContextPack":
        """Set metadata value (chainable)"""
        self._metadata[key] = value
        return self
    
    def add_metadata(self, **kwargs) -> "ContextPack":
        """Add multiple metadata fields (chainable)"""
        self._metadata.update(kwargs)
        return self
    
    def remove_metadata(self, key: str) -> "ContextPack":
        """Remove metadata field (chainable)"""
        self._metadata.pop(key, None)
        return self
    
    def get_all_metadata(self) -> Dict[str, Any]:
        """Get all metadata"""
        return dict(self._metadata)
    
    # ============================================================================
    # TAG MANAGEMENT
    # ============================================================================
    
    def get_tags(self) -> Set[str]:
        """Get all tags"""
        return set(self._metadata.get("tags", []))
    
    def add_tag(self, tag: str) -> "ContextPack":
        """Add tag (chainable)"""
        tags = self.get_tags()
        tags.add(tag)
        self._metadata["tags"] = sorted(list(tags))
        return self
    
    def remove_tag(self, tag: str) -> "ContextPack":
        """Remove tag (chainable)"""
        tags = self.get_tags()
        tags.discard(tag)
        self._metadata["tags"] = sorted(list(tags))
        return self
    
    def has_tag(self, tag: str) -> bool:
        """Check if tag exists"""
        return tag in self.get_tags()
    
    def add_tags(self, tags: List[str]) -> "ContextPack":
        """Add multiple tags (chainable)"""
        for tag in tags:
            self.add_tag(tag)
        return self
    
    # ============================================================================
    # PROVENANCE TRACKING
    # ============================================================================
    
    def get_provenance(self) -> List[str]:
        """Get provenance chain"""
        return list(self._provenance)
    
    def add_provenance(self, step: str) -> "ContextPack":
        """Add provenance step (chainable)"""
        self._provenance.append(f"{step}:{time.time()}")
        self._hash = None  # Invalidate cache
        return self
    
    def merge_provenance(self, other: "ContextPack") -> "ContextPack":
        """Merge provenance from another pack (chainable)"""
        self._provenance.extend(other._provenance)
        return self
    
    # ============================================================================
    # EXTENSION SYSTEM
    # ============================================================================
    
    def register_extension(self, name: str, extension: Any) -> "ContextPack":
        """Register extension (chainable)"""
        self._extensions[name] = extension
        return self
    
    def get_extension(self, name: str, default: Any = None) -> Any:
        """Get extension by name"""
        return self._extensions.get(name, default)
    
    def has_extension(self, name: str) -> bool:
        """Check if extension exists"""
        return name in self._extensions
    
    def remove_extension(self, name: str) -> "ContextPack":
        """Remove extension (chainable)"""
        self._extensions.pop(name, None)
        return self
    
    # ============================================================================
    # VALIDATION
    # ============================================================================
    
    def _validate_content(self):
        """Validate content based on modality"""
        if self.modality == Modality.EMBEDDING:
            if not isinstance(self.content, (np.ndarray, torch.Tensor, dict)):
                raise ValueError(f"Embedding modality requires array/tensor, got {type(self.content)}")
        
        elif self.modality == Modality.TEXT:
            if not isinstance(self.content, (str, dict)):
                raise ValueError(f"Text modality requires string, got {type(self.content)}")
        
        elif self.modality == Modality.STRUCTURED:
            if not isinstance(self.content, (dict, object)):
                raise ValueError(f"Structured modality requires dict/object, got {type(self.content)}")
    
    def is_valid(self) -> bool:
        """Check if pack is valid"""
        return self._validated
    
    # ============================================================================
    # HASHING & SIGNATURES
    # ============================================================================
    
    def _compute_hash(self) -> str:
        """Compute content hash for identity"""
        content_str = json.dumps(self._get_hashable_content(), sort_keys=True)
        self._hash = hashlib.sha256(content_str.encode()).hexdigest()
        return self._hash
    
    def _get_hashable_content(self) -> Dict[str, Any]:
        """Get hashable representation of core content"""
        content = self.content
        if isinstance(content, (np.ndarray, torch.Tensor)):
            content = content.tolist()
        elif hasattr(content, "__dict__"):
            content = content.__dict__
        
        return {
            "content": content,
            "context_type": self.context_type.value,
            "modality": self.modality.value,
            "schema_version": self._schema_version
        }
    
    def get_hash(self) -> str:
        """Get content hash"""
        if self._hash is None:
            self._compute_hash()
        return self._hash
    
    def compute_signature(self, secret: Optional[str] = None) -> str:
        """Compute cryptographic signature"""
        data = self.get_hash() + (secret or "")
        self._signature = hashlib.sha512(data.encode()).hexdigest()
        return self._signature
    
    def verify_signature(self, signature: str, secret: Optional[str] = None) -> bool:
        """Verify signature"""
        expected = self.compute_signature(secret)
        return signature == expected
    
    # ============================================================================
    # SERIALIZATION
    # ============================================================================
    
    def to_dict(self, include_extensions: bool = False) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "id": self.id,
            "content": self._serialize_content(),
            "context_type": self.context_type.value,
            "modality": self.modality.value,
            "created_at": self.created_at,
            "metadata": self._metadata,
            "provenance": self._provenance,
            "schema_version": self._schema_version,
            "hash": self.get_hash()
        }
        
        if include_extensions and self._extensions:
            result["extensions"] = {
                name: ext.to_dict() if hasattr(ext, "to_dict") else str(ext)
                for name, ext in self._extensions.items()
            }
        
        return result
    
    def _serialize_content(self) -> Any:
        """Serialize content for storage"""
        content = self.content
        
        if isinstance(content, (np.ndarray, torch.Tensor)):
            return {
                "_type": "tensor",
                "data": content.tolist(),
                "shape": list(content.shape),
                "dtype": str(content.dtype)
            }
        
        elif isinstance(content, Serializable):
            return {
                "_type": "serializable",
                "class": content.__class__.__name__,
                "data": content.to_dict()
            }
        
        elif hasattr(content, "__dict__"):
            return {
                "_type": "object",
                "class": content.__class__.__name__,
                "data": content.__dict__
            }
        
        return content
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextPack":
        """Create from dictionary"""
        # Deserialize content
        content = cls._deserialize_content(data.get("content"))
        
        # Create pack
        pack = cls(
            content=content,
            context_type=ContextType(data["context_type"]),
            modality=Modality(data["modality"]),
            id=data.get("id", str(uuid.uuid4())),
            created_at=data.get("created_at", time.time())
        )
        
        # Restore metadata
        pack._metadata = data.get("metadata", {})
        
        # Restore provenance
        pack._provenance = data.get("provenance", [])
        
        # Restore schema version
        pack._schema_version = data.get("schema_version", "1.0")
        
        # Restore hash
        pack._hash = data.get("hash")
        
        return pack
    
    @classmethod
    def _deserialize_content(cls, content: Any) -> Any:
        """Deserialize content from storage format"""
        if isinstance(content, dict):
            content_type = content.get("_type")
            
            if content_type == "tensor":
                data = content["data"]
                if isinstance(data, list):
                    return np.array(data)
            
            elif content_type == "serializable":
                # TODO: Implement class registry for deserialization
                return content["data"]
            
            elif content_type == "object":
                return content["data"]
        
        return content
    
    def to_json(self, **kwargs) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), **kwargs)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ContextPack":
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    # ============================================================================
    # COMPOSITION & MERGING
    # ============================================================================
    
    def merge(self, other: "ContextPack", strategy: str = "append") -> "ContextPack":
        """
        Merge with another ContextPack.
        
        Strategies:
        - "append": Append content (for text, lists)
        - "replace": Replace content
        - "combine": Combine structured data
        - "concat": Concatenate embeddings
        """
        if strategy == "append":
            new_content = self._append_content(other.content)
        elif strategy == "replace":
            new_content = other.content
        elif strategy == "combine":
            new_content = self._combine_structured(other.content)
        elif strategy == "concat":
            new_content = self._concat_embeddings(other.content)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        
        # Create merged pack
        merged = ContextPack(
            content=new_content,
            context_type=self.context_type,
            modality=self.modality
        )
        
        # Merge metadata
        merged._metadata = {**self._metadata, **other._metadata}
        
        # Merge provenance
        merged._provenance = self._provenance + other._provenance
        
        # Merge tags
        merged.add_tags(list(self.get_tags() | other.get_tags()))
        
        return merged
    
    def _append_content(self, other_content: Any) -> Any:
        """Append content"""
        if isinstance(self.content, str) and isinstance(other_content, str):
            return self.content + "\n" + other_content
        elif isinstance(self.content, list) and isinstance(other_content, list):
            return self.content + other_content
        elif isinstance(self.content, dict) and isinstance(other_content, dict):
            return {**self.content, **other_content}
        return other_content
    
    def _combine_structured(self, other_content: Any) -> Any:
        """Combine structured content"""
        if isinstance(self.content, dict) and isinstance(other_content, dict):
            result = dict(self.content)
            for key, value in other_content.items():
                if key in result and isinstance(result[key], list):
                    result[key].extend(value if isinstance(value, list) else [value])
                else:
                    result[key] = value
            return result
        return other_content
    
    def _concat_embeddings(self, other_content: Any) -> Any:
        """Concatenate embeddings"""
        if isinstance(self.content, np.ndarray) and isinstance(other_content, np.ndarray):
            return np.concatenate([self.content, other_content], axis=0)
        elif isinstance(self.content, torch.Tensor) and isinstance(other_content, torch.Tensor):
            return torch.cat([self.content, other_content], dim=0)
        return other_content
    
    # ============================================================================
    # FILTERING & TRANSFORMATION
    # ============================================================================
    
    def filter_by_tags(self, tags: List[str], match_all: bool = False) -> bool:
        """Check if pack matches tags"""
        pack_tags = self.get_tags()
        if match_all:
            return all(tag in pack_tags for tag in tags)
        else:
            return any(tag in pack_tags for tag in tags)
    
    def transform(
        self,
        func: Callable[[Any], Any],
        new_modality: Optional[Modality] = None
    ) -> "ContextPack":
        """Transform content with function"""
        new_content = func(self.content)
        return ContextPack(
            content=new_content,
            context_type=self.context_type,
            modality=new_modality or self.modality
        ).add_provenance(f"transform:{func.__name__}")
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_size(self) -> int:
        """Get approximate size in bytes"""
        try:
            return len(json.dumps(self.to_dict()).encode())
        except:
            return 0
    
    def get_age(self) -> float:
        """Get age in seconds"""
        return time.time() - self.created_at
    
    def is_expired(self) -> bool:
        """Check if expired based on TTL"""
        ttl = self.get_metadata("ttl")
        if ttl is None:
            return False
        return self.get_age() > ttl
    
    def clone(self) -> "ContextPack":
        """Create deep copy"""
        return ContextPack.from_dict(self.to_dict())
    
    def summary(self) -> str:
        """Get human-readable summary"""
        content_preview = str(self.content)[:100]
        if len(str(self.content)) > 100:
            content_preview += "..."
        
        return (
            f"ContextPack(id={self.id[:8]}, "
            f"type={self.context_type.value}, "
            f"modality={self.modality.value}, "
            f"content={content_preview}, "
            f"tags={sorted(list(self.get_tags()))})"
        )
    
    def __str__(self) -> str:
        return self.summary()
    
    def __repr__(self) -> str:
        return self.summary()
    
    def __hash__(self) -> int:
        return hash(self.get_hash())
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ContextPack):
            return False
        return self.get_hash() == other.get_hash()


# ============================================================================
# CONTEXT PACK COLLECTION
# ============================================================================

@dataclass
class ContextPackCollection:
    """
    Collection of ContextPacks with query and aggregation capabilities.
    
    Usage:
        collection = ContextPackCollection()
        collection.add(pack1)
        collection.add(pack2)
        
        # Query
        results = collection.query(
            context_type=ContextType.DOCUMENT,
            tags=["priority:high"]
        )
    """
    
    packs: List[ContextPack] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    
    def add(self, pack: ContextPack) -> "ContextPackCollection":
        """Add pack to collection"""
        self.packs.append(pack)
        return self
    
    def extend(self, packs: List[ContextPack]) -> "ContextPackCollection":
        """Add multiple packs"""
        self.packs.extend(packs)
        return self
    
    def query(
        self,
        context_type: Optional[ContextType] = None,
        modality: Optional[Modality] = None,
        tags: Optional[List[str]] = None,
        match_all_tags: bool = False,
        max_results: Optional[int] = None
    ) -> List[ContextPack]:
        """Query packs with filters"""
        results = self.packs
        
        if context_type is not None:
            results = [p for p in results if p.context_type == context_type]
        
        if modality is not None:
            results = [p for p in results if p.modality == modality]
        
        if tags is not None:
            results = [p for p in results if p.filter_by_tags(tags, match_all_tags)]
        
        if max_results is not None:
            results = results[:max_results]
        
        return results
    
    def group_by(self, key_func: Callable[[ContextPack], Any]) -> Dict[Any, List[ContextPack]]:
        """Group packs by key function"""
        groups = {}
        for pack in self.packs:
            key = key_func(pack)
            if key not in groups:
                groups[key] = []
            groups[key].append(pack)
        return groups
    
    def aggregate(self, func: Callable[[List[ContextPack]], Any]) -> Any:
        """Aggregate packs with function"""
        return func(self.packs)
    
    def merge_all(self, strategy: str = "append") -> ContextPack:
        """Merge all packs into one"""
        if not self.packs:
            raise ValueError("Cannot merge empty collection")
        
        result = self.packs[0]
        for pack in self.packs[1:]:
            result = result.merge(pack, strategy=strategy)
        
        return result
    
    def filter_expired(self) -> "ContextPackCollection":
        """Remove expired packs"""
        self.packs = [p for p in self.packs if not p.is_expired()]
        return self
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "total": len(self.packs),
            "by_type": self._count_by(lambda p: p.context_type.value),
            "by_modality": self._count_by(lambda p: p.modality.value),
            "by_tags": self._count_tags(),
            "total_size_bytes": sum(p.get_size() for p in self.packs),
            "avg_age_seconds": np.mean([p.get_age() for p in self.packs]) if self.packs else 0
        }
    
    def _count_by(self, key_func: Callable[[ContextPack], Any]) -> Dict[Any, int]:
        """Count by key function"""
        counts = {}
        for pack in self.packs:
            key = key_func(pack)
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    def _count_tags(self) -> Dict[str, int]:
        """Count tags"""
        counts = {}
        for pack in self.packs:
            for tag in pack.get_tags():
                counts[tag] = counts.get(tag, 0) + 1
        return counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "packs": [p.to_dict() for p in self.packs],
            "created_at": self.created_at,
            "statistics": self.get_statistics()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextPackCollection":
        """Create from dictionary"""
        collection = cls(
            id=data.get("id", str(uuid.uuid4())),
            created_at=data.get("created_at", time.time())
        )
        collection.packs = [ContextPack.from_dict(p) for p in data.get("packs", [])]
        return collection
    
    def __len__(self) -> int:
        return len(self.packs)
    
    def __getitem__(self, index: int) -> ContextPack:
        return self.packs[index]
    
    def __iter__(self):
        return iter(self.packs)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_user_query_context(
    text: str,
    metadata: Optional[Dict[str, Any]] = None
) -> ContextPack:
    """Create user query context"""
    return ContextPack(
        content=text,
        context_type=ContextType.USER_QUERY,
        modality=Modality.TEXT
    ).add_metadata(**(metadata or {})).add_tag("source:user")


def create_document_context(
    content: Union[str, Dict[str, Any]],
    source: str,
    metadata: Optional[Dict[str, Any]] = None
) -> ContextPack:
    """Create document context"""
    modality = Modality.STRUCTURED if isinstance(content, dict) else Modality.TEXT
    
    return ContextPack(
        content=content,
        context_type=ContextType.DOCUMENT,
        modality=modality
    ).add_metadata(
        source=source,
        **(metadata or {})
    ).add_tag(f"source:{source}")


def create_embedding_context(
    embedding: Union[np.ndarray, torch.Tensor],
    source: str,
    metadata: Optional[Dict[str, Any]] = None
) -> ContextPack:
    """Create embedding context"""
    return ContextPack(
        content=embedding,
        context_type=ContextType.MEMORY,
        modality=Modality.EMBEDDING
    ).add_metadata(
        source=source,
        dimension=embedding.shape[-1],
        **(metadata or {})
    ).add_tag(f"embedding:{source}")


def create_goal_context(
    goal: Union[str, Dict[str, Any]],
    priority: int = 1,
    metadata: Optional[Dict[str, Any]] = None
) -> ContextPack:
    """Create goal context"""
    modality = Modality.STRUCTURED if isinstance(goal, dict) else Modality.TEXT
    
    return ContextPack(
        content=goal,
        context_type=ContextType.GOAL,
        modality=modality
    ).add_metadata(
        priority=priority,
        **(metadata or {})
    ).add_tag(f"priority:{priority}")


def create_reflection_context(
    reflection: str,
    original_context: ContextPack,
    metadata: Optional[Dict[str, Any]] = None
) -> ContextPack:
    """Create reflection context"""
    return ContextPack(
        content=reflection,
        context_type=ContextType.REFLECTION,
        modality=Modality.TEXT
    ).add_metadata(
        original_context_id=original_context.id,
        **(metadata or {})
    ).add_provenance(f"reflection_from:{original_context.id}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def merge_context_packs(
    packs: List[ContextPack],
    strategy: str = "append"
) -> ContextPack:
    """Merge multiple context packs"""
    if not packs:
        raise ValueError("Cannot merge empty list")
    
    collection = ContextPackCollection(packs)
    return collection.merge_all(strategy=strategy)


def filter_context_packs(
    packs: List[ContextPack],
    context_type: Optional[ContextType] = None,
    tags: Optional[List[str]] = None
) -> List[ContextPack]:
    """Filter context packs"""
    collection = ContextPackCollection(packs)
    return collection.query(context_type=context_type, tags=tags)


def group_context_packs_by_type(
    packs: List[ContextPack]
) -> Dict[str, List[ContextPack]]:
    """Group context packs by type"""
    collection = ContextPackCollection(packs)
    return collection.group_by(lambda p: p.context_type.value)


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "ContextPack",
    "ContextPackCollection",
    "ContextType",
    "Modality",
    "ContextMetadata",
    "create_user_query_context",
    "create_document_context",
    "create_embedding_context",
    "create_goal_context",
    "create_reflection_context",
    "merge_context_packs",
    "filter_context_packs",
    "group_context_packs_by_type"
]

==================================================
FILE: core\thresholds.py
==================================================

"""
CalibratedThresholds: Statistically-derived safety boundaries for governed self-improvement.

All thresholds are computed from baseline system behavior (μ ± kσ) to ensure:
- Data-driven (not arbitrary)
- System-specific (adapts to your scoring distribution)
- Defensible (statistical justification)
- Serializable (for experiment reproducibility)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibratedThresholds:
    """
    Statistically calibrated safety thresholds for governance layer.
    
    All thresholds derived from baseline system behavior:
    - Critical thresholds: μ ± 2σ (95% confidence interval)
    - Warning thresholds: μ ± 1σ (68% confidence interval)
    - Fixed thresholds: validated through pilot studies
    
    Immutable by design - thresholds must not change during experiment execution.
    """
    
    # ===== ENERGY THRESHOLDS (Hallucination Safety) =====
    energy_max: float  # μ + 2σ: Absolute failure boundary
    energy_warning: float  # μ + 1σ: Trigger conservative updates
    
    # ===== HRM THRESHOLDS (Reasoning Quality) =====
    hrm_min: float  # μ - 2σ: Minimum acceptable alignment
    
    # ===== EMBEDDING THRESHOLDS (Geometry Stability) =====
    margin_min: float  # μ - 2σ: Minimum embedding margin
    variance_min: float  # Fixed: Absolute floor for embedding diversity
    collapse_index_max: float  # Fixed: Max eigenvalue ratio (λ_max/λ_min)
    drift_max: float  # Fixed: Max angular drift per update (radians)
    
    # ===== PROVENANCE METADATA =====
    calibration_timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    baseline_episodes: int = 200
    baseline_system: str = "scalar_rl_baseline"
    statistical_method: str = "mean_plus_2std"
    schema_version: str = "1.0"
    
    # ===== VALIDATION =====
    def __post_init__(self):
        """Validate threshold relationships"""
        # Energy thresholds must be ordered
        if not (0.0 <= self.energy_warning < self.energy_max <= 1.0):
            raise ValueError(
                f"Invalid energy thresholds: warning={self.energy_warning}, "
                f"max={self.energy_max} (must satisfy 0 ≤ warning < max ≤ 1)"
            )
        
        # HRM threshold must be valid
        if not (0.0 <= self.hrm_min <= 1.0):
            raise ValueError(f"Invalid HRM min: {self.hrm_min} (must be in [0,1])")
        
        # Embedding thresholds must be positive
        if self.variance_min <= 0:
            raise ValueError(f"Variance min must be > 0, got {self.variance_min}")
        if self.collapse_index_max < 1.0:
            raise ValueError(
                f"Collapse index max must be ≥ 1.0, got {self.collapse_index_max}"
            )
        if not (0.0 < self.drift_max < np.pi):
            raise ValueError(
                f"Drift max must be in (0, π), got {self.drift_max}"
            )
        
        # Logical relationships
        if self.margin_min < 0 or self.margin_min > 1.0:
            raise ValueError(
                f"Margin min must be in [0,1], got {self.margin_min}"
            )
        
        logger.info(
            f"✓ CalibratedThresholds validated | "
            f"energy: [{self.energy_warning:.3f}, {self.energy_max:.3f}] | "
            f"hrm_min: {self.hrm_min:.3f} | "
            f"margin_min: {self.margin_min:.3f}"
        )
    
    # ===== SERIALIZATION =====
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibratedThresholds":
        """Reconstruct from dictionary"""
        # Handle schema evolution
        kwargs = {
            "energy_max": data["energy_max"],
            "energy_warning": data["energy_warning"],
            "hrm_min": data["hrm_min"],
            "margin_min": data["margin_min"],
            "variance_min": data.get("variance_min", 0.3),
            "collapse_index_max": data.get("collapse_index_max", 10.0),
            "drift_max": data.get("drift_max", 0.15),
            "calibration_timestamp": data.get(
                "calibration_timestamp", 
                datetime.utcnow().isoformat()
            ),
            "baseline_episodes": data.get("baseline_episodes", 200),
            "baseline_system": data.get("baseline_system", "unknown"),
            "statistical_method": data.get("statistical_method", "mean_plus_2std"),
            "schema_version": data.get("schema_version", "1.0")
        }
        return cls(**kwargs)
    
    @classmethod
    def from_json(cls, json_str: str) -> "CalibratedThresholds":
        """Reconstruct from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    # ===== THRESHOLD CHECKING =====
    def is_energy_critical(self, energy: float) -> bool:
        """Check if energy exceeds critical threshold"""
        return energy > self.energy_max
    
    def is_energy_warning(self, energy: float) -> bool:
        """Check if energy exceeds warning threshold"""
        return energy > self.energy_warning
    
    def is_hrm_critical(self, hrm: float) -> bool:
        """Check if HRM alignment below critical threshold"""
        return hrm < self.hrm_min
    
    def is_margin_critical(self, margin: float) -> bool:
        """Check if embedding margin below critical threshold"""
        return margin < self.margin_min
    
    def is_variance_critical(self, variance: float) -> bool:
        """Check if embedding variance below critical threshold"""
        return variance < self.variance_min
    
    def is_collapse_critical(self, collapse_index: float) -> bool:
        """Check if collapse index exceeds critical threshold"""
        return collapse_index > self.collapse_index_max
    
    def is_drift_critical(self, drift: float) -> bool:
        """Check if angular drift exceeds critical threshold"""
        return drift > self.drift_max
    
    # ===== POLICY REGIME DETERMINATION =====
    def determine_regime(self, metrics: Dict[str, float]) -> str:
        """
        Determine policy regime based on current metrics.
        
        Returns: "safe", "warning", or "critical"
        """
        energy = metrics.get("energy_raw", 0.0)
        hrm = metrics.get("hrm_alignment", 1.0)
        margin = metrics.get("embedding_margin", 1.0)
        variance = metrics.get("embedding_variance", 1.0)
        collapse_index = metrics.get("collapse_index", 1.0)
        drift = metrics.get("angular_drift", 0.0)
        
        # Critical checks (any trigger critical regime)
        if (self.is_energy_critical(energy) or
            self.is_hrm_critical(hrm) or
            self.is_margin_critical(margin) or
            self.is_variance_critical(variance) or
            self.is_collapse_critical(collapse_index) or
            self.is_drift_critical(drift)):
            return "critical"
        
        # Warning checks
        if self.is_energy_warning(energy):
            return "warning"
        
        return "safe"
    
    # ===== HUMAN-READABLE REPORT =====
    def generate_report(self) -> str:
        """Generate human-readable threshold report"""
        lines = [
            "=" * 60,
            "CALIBRATED THRESHOLDS REPORT",
            "=" * 60,
            f"Calibration Time: {self.calibration_timestamp}",
            f"Baseline System: {self.baseline_system}",
            f"Episodes Used: {self.baseline_episodes}",
            f"Statistical Method: {self.statistical_method}",
            "",
            "┌─────────────────────────────────────────────────────────┐",
            "│ ENERGY THRESHOLDS (Hallucination Safety)                │",
            "├─────────────────────────────────────────────────────────┤",
            f"│ Warning:  {self.energy_warning:6.3f} (μ + 1σ)                     │",
            f"│ Critical: {self.energy_max:6.3f} (μ + 2σ)                     │",
            "├─────────────────────────────────────────────────────────┤",
            "│ HRM THRESHOLDS (Reasoning Quality)                      │",
            "├─────────────────────────────────────────────────────────┤",
            f"│ Minimum:  {self.hrm_min:6.3f} (μ - 2σ)                     │",
            "├─────────────────────────────────────────────────────────┤",
            "│ EMBEDDING THRESHOLDS (Geometry Stability)               │",
            "├─────────────────────────────────────────────────────────┤",
            f"│ Margin Min:      {self.margin_min:6.3f} (μ - 2σ)          │",
            f"│ Variance Min:    {self.variance_min:6.3f} (fixed)         │",
            f"│ Collapse Max:    {self.collapse_index_max:6.3f} (fixed)   │",
            f"│ Drift Max:       {self.drift_max:6.3f} rad (fixed)        │",
            "└─────────────────────────────────────────────────────────┘",
            "",
            "Thresholds derived from baseline system behavior.",
            "Critical violations trigger immediate governance intervention.",
            "=" * 60
        ]
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return (
            f"CalibratedThresholds("
            f"energy:[{self.energy_warning:.3f}, {self.energy_max:.3f}], "
            f"hrm_min:{self.hrm_min:.3f}, "
            f"margin_min:{self.margin_min:.3f}, "
            f"var_min:{self.variance_min:.3f})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()


# ============================================================================
# FACTORY FUNCTIONS FOR COMMON SCENARIOS
# ============================================================================

def create_from_baseline_stats(
    energy_stats: Dict[str, float],
    hrm_stats: Dict[str, float],
    margin_stats: Dict[str, float],
    baseline_episodes: int = 200,
    baseline_system: str = "scalar_rl_baseline"
) -> CalibratedThresholds:
    """
    Create thresholds from pre-computed baseline statistics.
    
    Args:
        energy_stats: {"mean": float, "std": float}
        hrm_stats: {"mean": float, "std": float}
        margin_stats: {"mean": float, "std": float}
        baseline_episodes: Number of episodes used for calibration
        baseline_system: Identifier for baseline system
    
    Returns:
        CalibratedThresholds instance
    """
    return CalibratedThresholds(
        energy_max=energy_stats["mean"] + 2 * energy_stats["std"],
        energy_warning=energy_stats["mean"] + 1 * energy_stats["std"],
        hrm_min=hrm_stats["mean"] - 2 * hrm_stats["std"],
        margin_min=margin_stats["mean"] - 2 * margin_stats["std"],
        variance_min=0.3,  # Fixed based on embedding geometry studies
        collapse_index_max=10.0,  # Fixed based on eigenvalue ratio analysis
        drift_max=0.15,  # Fixed based on angular drift studies (≈8.6 degrees)
        baseline_episodes=baseline_episodes,
        baseline_system=baseline_system,
        statistical_method="mean_plus_2std"
    )


def create_conservative_thresholds() -> CalibratedThresholds:
    """
    Create conservative thresholds for high-stakes applications.
    
    Tighter bounds than statistical calibration.
    """
    return CalibratedThresholds(
        energy_max=0.40,
        energy_warning=0.30,
        hrm_min=0.70,
        margin_min=0.50,
        variance_min=0.4,
        collapse_index_max=8.0,
        drift_max=0.10,
        baseline_system="conservative_preset",
        statistical_method="domain_expert"
    )


def create_permissive_thresholds() -> CalibratedThresholds:
    """
    Create permissive thresholds for exploratory research.
    
    Wider bounds to allow more learning velocity.
    """
    return CalibratedThresholds(
        energy_max=0.60,
        energy_warning=0.50,
        hrm_min=0.50,
        margin_min=0.30,
        variance_min=0.2,
        collapse_index_max=15.0,
        drift_max=0.25,
        baseline_system="permissive_preset",
        statistical_method="domain_expert"
    )


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_thresholds_against_distribution(
    thresholds: CalibratedThresholds,
    energy_samples: np.ndarray,
    hrm_samples: np.ndarray,
    margin_samples: np.ndarray
) -> Dict[str, Any]:
    """
    Validate thresholds against actual distribution.
    
    Returns diagnostic report showing:
    - % of samples above/below thresholds
    - Statistical coverage
    - Potential calibration issues
    """
    report = {
        "energy": {
            "warning_violation_pct": np.mean(energy_samples > thresholds.energy_warning) * 100,
            "critical_violation_pct": np.mean(energy_samples > thresholds.energy_max) * 100,
            "mean": np.mean(energy_samples),
            "std": np.std(energy_samples)
        },
        "hrm": {
            "critical_violation_pct": np.mean(hrm_samples < thresholds.hrm_min) * 100,
            "mean": np.mean(hrm_samples),
            "std": np.std(hrm_samples)
        },
        "margin": {
            "critical_violation_pct": np.mean(margin_samples < thresholds.margin_min) * 100,
            "mean": np.mean(margin_samples),
            "std": np.std(margin_samples)
        },
        "calibration_quality": "good"
    }
    
    # Flag potential issues
    issues = []
    if report["energy"]["critical_violation_pct"] > 5.0:
        issues.append("Energy critical threshold too strict (>5% baseline violations)")
    if report["hrm"]["critical_violation_pct"] > 5.0:
        issues.append("HRM critical threshold too strict (>5% baseline violations)")
    if report["margin"]["critical_violation_pct"] > 5.0:
        issues.append("Margin critical threshold too strict (>5% baseline violations)")
    
    if issues:
        report["calibration_quality"] = "needs_adjustment"
        report["issues"] = issues
    
    return report


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "CalibratedThresholds",
    "create_from_baseline_stats",
    "create_conservative_thresholds",
    "create_permissive_thresholds",
    "validate_thresholds_against_distribution"
]

==================================================
FILE: evaluation\bundle_comparator.py
==================================================

# components/elm/dominance/bundle_comparator.py

from typing import List, Dict
from stephanie.data.score_bundle import ScoreBundle
from components.elm.axes import AXIS_SEMANTICS, AxisDirection


class BundleComparator:

    @staticmethod
    def delta(
        before: ScoreBundle,
        after: ScoreBundle,
    ) -> Dict[str, float]:
        """
        Direction-normalized delta.
        Positive = improvement.
        """

        deltas: Dict[str, float] = {}

        dims = set(before.results.keys()) | set(after.results.keys())

        for dim in dims:
            b = before.get(dim)
            a = after.get(dim)

            if not b or not a:
                continue

            direction = AXIS_SEMANTICS.get(dim, AxisDirection.HIGHER_IS_BETTER)

            if direction == AxisDirection.HIGHER_IS_BETTER:
                delta = a.score - b.score
            else:
                delta = b.score - a.score

            deltas[dim] = delta

        return deltas

    @staticmethod
    def dominates(
        before: ScoreBundle,
        after: ScoreBundle,
        critical_axes: List[str],
    ) -> bool:
        """
        Strict Pareto dominance on critical axes.
        """

        for dim in critical_axes:
            b = before.get(dim)
            a = after.get(dim)

            if not b or not a:
                return False

            direction = AXIS_SEMANTICS.get(dim, AxisDirection.HIGHER_IS_BETTER)

            if direction == AxisDirection.HIGHER_IS_BETTER:
                if a.score <= b.score:
                    return False
            else:
                if a.score >= b.score:
                    return False

        return True


==================================================
FILE: evaluation\governance_scorer.py
==================================================

# stephanie/scoring/governance/governance_scorer.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Protocol

from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)


# ----------------------------------------
# Provider Protocol
# ----------------------------------------

class GovernanceProvider(Protocol):
    def compute(
        self,
        context: dict,
        scorable: Scorable,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Returns:
            {
                "dimension_name": {
                    "score": float,
                    "rationale": str,
                    "attributes": dict
                }
            }
        """
        ...


# ----------------------------------------
# GovernanceScorer
# ----------------------------------------

class GovernanceScorer(BaseScorer):
    """
    Governance layer implemented as a standard Stephanie scorer.

    - Each provider emits dimensions
    - Dimensions become ScoreResult objects
    - Attributes store diagnostics
    - No custom reward vector
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.model_type = "governance"
        self.providers: List[GovernanceProvider] = cfg.get("providers", [])

        if not self.providers:
            log.warning("GovernanceScorer initialized with no providers")

    # ----------------------------------------
    # Core Scoring
    # ----------------------------------------

    def _score_core(
        self,
        context: dict,
        scorable: Scorable,
        dimensions: List[str]
    ) -> ScoreBundle:

        results: Dict[str, ScoreResult] = {}

        for provider in self.providers:
            try:
                provider_output = provider.compute(
                    context=context,
                    scorable=scorable,
                )

                for dim, payload in provider_output.items():

                    if dimensions and dim not in dimensions:
                        continue

                    score = float(payload.get("score", 0.0))
                    rationale = payload.get("rationale", "")
                    attributes = payload.get("attributes", {})

                    results[dim] = ScoreResult(
                        dimension=dim,
                        score=score,
                        weight=1.0,
                        rationale=rationale,
                        source=self.model_type,
                        attributes=attributes,
                    )

            except Exception as e:
                log.error(f"Governance provider failure: {e}")
                self.logger.log(
                    "GovernanceProviderError",
                    {"error": str(e)}
                )

        return ScoreBundle(results=results)


==================================================
FILE: evaluation\multi_layer_evaluator.py
==================================================

# elm/evaluation/multi_layer_evaluator.py

from typing import List, Dict, Any
from datetime import datetime

from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult

from elm.providers.base import SignalProvider, SignalResult


class MultiLayerEvaluator:
    """
    Stephanie-compatible reducer.

    Produces:
        stephanie.scoring.score_bundle.ScoreBundle
    """

    def __init__(self, providers: List[SignalProvider]):
        self.providers = providers

    def evaluate(
        self,
        context_pack,
        plan_trace,
        output,
        **kwargs
    ) -> ScoreBundle:

        results: Dict[str, ScoreResult] = {}
        meta: Dict[str, Any] = {
            "trace_id": getattr(plan_trace, "trace_id", None),
            "evaluated_at": datetime.utcnow().isoformat(),
        }

        for provider in self.providers:
            signal: SignalResult = provider.compute(
                context_pack=context_pack,
                plan_trace=plan_trace,
                output=output,
                **kwargs
            )

            for axis, value in signal.axis_values.items():

                dim_name = axis.value  # dimension string

                results[dim_name] = ScoreResult(
                    dimension=dim_name,
                    score=float(value) * 100.0,   # Stephanie uses 0-100 scale
                    weight=1.0,
                    rationale="ELM signal",
                    source=provider.__class__.__name__,
                    target_type="plan_trace",
                    prompt_hash=None,
                    attributes={
                        "confidence": signal.confidence,
                        "failures": signal.failure_signatures,
                    }
                )

        return ScoreBundle(results=results, meta=meta)


==================================================
FILE: experiment\ablations.py
==================================================

# Ablation wrappers


==================================================
FILE: experiment\baseline_calibrator.py
==================================================

# experiment/baseline_calibrator.py
from typing import Any, List
import numpy as np
from stephanie.components.elm.core.thresholds import CalibratedThresholds
from stephanie.components.elm.governance.signal_extractor import GovernanceSignalExtractor
import logging

logger = logging.getLogger(__name__)

class BaselineCalibrator:
    """Calibrate thresholds using baseline system behavior"""
    
    def __init__(self, baseline_system: Any, extractor: GovernanceSignalExtractor):
        self.baseline = baseline_system
        self.extractor = extractor
    
    def calibrate(
        self,
        queries: List[Any],
        episodes: int = 200
    ) -> CalibratedThresholds:
        """Run baseline and compute statistical thresholds (μ ± 2σ)"""
        
        all_metrics = []
        
        for ep in range(episodes):
            query = np.random.choice(queries)
            bundle = self.baseline.evaluate(query)
            
            metrics = self.extractor.extract_from_bundle(bundle)
            all_metrics.append(metrics)
        
        # Compute statistics
        energies = [m.get("energy_raw", 0) for m in all_metrics]
        hrms = [m.get("hrm_alignment", 0) for m in all_metrics]
        margins = [m.get("embedding_margin", 0) for m in all_metrics]
        
        thresholds = CalibratedThresholds(
            energy_max=np.mean(energies) + 2 * np.std(energies),
            energy_warning=np.mean(energies) + 1 * np.std(energies),
            hrm_min=np.mean(hrms) - 2 * np.std(hrms),
            margin_min=np.mean(margins) - 2 * np.std(margins),
            variance_min=0.3,  # Fixed based on embedding geometry
            collapse_index_max=10.0,  # Fixed based on eigenvalue ratio
            drift_max=0.15,  # Fixed based on angular drift
            baseline_episodes=episodes,
            baseline_system=self.baseline.__class__.__name__,
            statistical_method="mean_plus_2std"
        )
        
        logger.info(f"Calibration complete: {thresholds}")
        return thresholds

==================================================
FILE: experiment\dynamic_stability_benchmark.py
==================================================

from dataclasses import dataclass
from typing import List, Any


@dataclass
class EpisodeLog:
    episode: int
    energy: float
    variance: float
    dominance: bool


class DynamicStabilityBenchmark:
    def __init__(self, system: Any):
        self.system = system
        self.logs: List[EpisodeLog] = []

    def run(self, queries: List[Any], episodes: int = 100):
        for ep in range(episodes):
            query = queries[ep % len(queries)]
            result = self.system.step(query)

            self.logs.append(
                EpisodeLog(
                    episode=ep,
                    energy=result.reward_vector.values.get("hallucination_energy", 0.0),
                    variance=0.0,
                    dominance=True,
                )
            )


==================================================
FILE: experiment\experiment.py
==================================================

# experiment/experiment.py
from typing import Any, Dict, List, Optional
import numpy as np
import logging
from stephanie.components.elm.core.thresholds import CalibratedThresholds
from stephanie.components.elm.governance.signal_extractor import GovernanceSignalExtractor
from stephanie.components.elm.tracking.retention_tracker import RetentionTracker
from stephanie.components.elm.tracking.collapse_detector import CollapseDetector

logger = logging.getLogger(__name__)

class ScoreBundleExperiment:
    def __init__(
        self,
        system: Any,
        queries: List[Any],
        thresholds: CalibratedThresholds,
        extractor: GovernanceSignalExtractor,
        episodes: int = 1000,
        perturbation_episode: Optional[int] = None,
        log_to_db: bool = True,
        seed: int = 42
    ):
        self.system = system
        self.queries = queries
        self.thresholds = thresholds
        self.extractor = extractor
        self.episodes = episodes
        self.perturbation_episode = perturbation_episode
        self.log_to_db = log_to_db
        self.seed = seed
        
        # State tracking
        self.episode_history: List[Dict] = []
        self.retention_tracker = RetentionTracker()
        self.collapse_detector = CollapseDetector(thresholds)
        
        np.random.seed(seed)
        logger.info(f"Experiment initialized: {episodes} episodes")

    def run(self) -> Dict[str, Any]:
        for episode in range(self.episodes):
            query = np.random.choice(self.queries)
            bundle_before = self.system.evaluate(query)
            
            improved = self.system.attempt_improvement(query, bundle_before)
            
            if improved:
                bundle_after = improved["bundle"]
                dominance_achieved = self.extractor.compute_dominance(
                    bundle_before, bundle_after
                )
                
                if dominance_achieved:
                    if self.log_to_db:
                        self.system.commit_improvement(query, bundle_after)
                    
                    metrics = self.extractor.extract_from_bundle(bundle_after)
                    self.episode_history.append({
                        "episode": episode,
                        "dominance_achieved": dominance_achieved,
                        "metrics": metrics
                    })
                    
                    self.retention_tracker.update(episode, metrics)
                    
                    failure = self._check_failure(metrics)
                    if failure:
                        return self._build_failure_result(episode, failure)
            
            if episode % 100 == 0:
                self._log_progress(episode)
        
        return self._build_success_result()
    
    def _check_failure(self, metrics: Dict[str, float]) -> Optional[str]:
        energy = metrics.get("energy_raw", 0)
        if energy > self.thresholds.energy_max:
            return f"Energy exceeded: {energy:.2f}"
        
        hrm = metrics.get("hrm_alignment", 1.0)
        if hrm < self.thresholds.hrm_min:
            return f"HRM alignment collapsed: {hrm:.2f}"
        
        margin = metrics.get("embedding_margin", 0.0)
        if margin < self.thresholds.margin_min:
            return f"Embedding margin collapsed: {margin:.2f}"
        
        return None
    
    def _log_progress(self, episode: int):
        if not self.episode_history:
            return
        
        recent = self.episode_history[-100:] if len(self.episode_history) >= 100 else self.episode_history
        energies = [ep["metrics"].get("energy_raw", 0) for ep in recent]
        dominances = [ep["dominance_achieved"] for ep in recent]
        
        logger.info(
            f"Episode {episode}/{self.episodes} | "
            f"Energy: {np.mean(energies):.3f} | "
            f"Dominance: {np.mean(dominances):.2%}"
        )
    
    def _build_success_result(self) -> Dict[str, Any]:
        if not self.episode_history:
            return {"status": "failed", "reason": "no episodes completed"}
        
        energies = [ep["metrics"].get("energy_raw", 0) for ep in self.episode_history]
        hrms = [ep["metrics"].get("hrm_alignment", 0) for ep in self.episode_history]
        dominances = [ep["dominance_achieved"] for ep in self.episode_history]
        
        return {
            "status": "success",
            "episodes_completed": len(self.episode_history),
            "metrics_summary": {
                "energy": {
                    "mean": float(np.mean(energies)),
                    "std": float(np.std(energies)),
                    "min": float(np.min(energies)),
                    "max": float(np.max(energies)),
                },
                "hrm_alignment": {
                    "mean": float(np.mean(hrms)),
                    "std": float(np.std(hrms)),
                },
                "dominance_rate": float(np.mean(dominances)),
            },
            "retention_scores": self.retention_tracker.get_scores(),
        }
    
    def _build_failure_result(self, episode: int, failure: str) -> Dict[str, Any]:
        return {
            "status": "failed",
            "episode": episode,
            "failure": failure,
            "metrics_summary": self._build_success_result()["metrics_summary"],
        }

==================================================
FILE: experiment\experiment_harness.py
==================================================

# experiment/experiment_harness.py
from typing import Any, Dict, List, Optional
import numpy as np
from stephanie.components.elm.core.thresholds import CalibratedThresholds
from stephanie.components.elm.governance.signal_extractor import GovernanceSignalExtractor
from stephanie.components.elm.tracking.retention_tracker import RetentionTracker
from stephanie.components.elm.tracking.collapse_detector import CollapseDetector
from stephanie.components.elm.experiment.perturbation_injector import PerturbationInjector
import logging

logger = logging.getLogger(__name__)

class ScoreBundleExperiment:
    """Experimental harness for governed self-improvement"""
    
    def __init__(
        self,
        system: Any,
        queries: List[Any],
        thresholds: CalibratedThresholds,
        extractor: GovernanceSignalExtractor,
        episodes: int = 1000,
        perturbation_episode: Optional[int] = None,
        perturbation_severity: str = "moderate",
        log_to_db: bool = True,
        seed: int = 42
    ):
        self.system = system
        self.queries = queries
        self.thresholds = thresholds
        self.extractor = extractor
        self.episodes = episodes
        self.perturbation_episode = perturbation_episode
        self.perturbation_severity = perturbation_severity
        self.log_to_db = log_to_db
        self.seed = seed
        
        # State tracking
        self.episode_history: List[Dict] = []
        self.retention_tracker = RetentionTracker()
        self.collapse_detector = CollapseDetector(thresholds)
        self.perturbation_injector = PerturbationInjector(system)
        
        np.random.seed(seed)
        logger.info(f"Experiment initialized: {episodes} episodes, seed={seed}")
    
    def run(self) -> Dict[str, Any]:
        """Execute full experiment"""
        
        for episode in range(self.episodes):
            # Check for perturbation injection
            if (self.perturbation_episode is not None and 
                episode == self.perturbation_episode):
                self.perturbation_injector.inject(self.perturbation_severity)
                logger.info(f"Perturbation injected at episode {episode}")
            
            # Sample query and evaluate
            query = np.random.choice(self.queries)
            bundle_before = self.system.evaluate(query)
            
            # Attempt improvement
            improved = self.system.attempt_improvement(query, bundle_before)
            
            if improved:
                bundle_after = improved["bundle"]
                reflection_trace = improved.get("reflection")
                
                # Check dominance
                dominance_achieved = self.extractor.compute_dominance(
                    bundle_before, bundle_after
                )
                
                if dominance_achieved:
                    # Commit improvement
                    if self.log_to_db:
                        self.system.commit_improvement(
                            query, bundle_after, reflection_trace
                        )
                    
                    # Extract metrics
                    metrics = self.extractor.extract_from_bundle(bundle_after)
                    delta_vector = self.extractor.compute_delta_vector(
                        bundle_before, bundle_after
                    )
                    
                    # Log episode
                    episode_data = {
                        "episode": episode,
                        "query_id": getattr(query, "id", None),
                        "dominance_achieved": dominance_achieved,
                        "metrics": metrics,
                        "delta_vector": delta_vector,
                    }
                    self.episode_history.append(episode_data)
                    
                    # Update retention tracking
                    self.retention_tracker.update(episode, metrics)
                    
                    # Check for collapse
                    failure = self.collapse_detector.check_failure(episode, metrics)
                    if failure and failure.severity == "critical":
                        logger.critical(f"COLLAPSE DETECTED: {failure}")
                        return self._build_failure_result(episode, failure)
            
            # Progress logging
            if episode % 100 == 0:
                self._log_progress(episode)
        
        return self._build_success_result()
    
    def _log_progress(self, episode: int):
        """Log progress summary"""
        if not self.episode_history:
            return
        
        recent = self.episode_history[-100:] if len(self.episode_history) >= 100 else self.episode_history
        energies = [ep["metrics"].get("energy_raw", 0) for ep in recent]
        dominances = [ep["dominance_achieved"] for ep in recent]
        
        logger.info(
            f"Episode {episode}/{self.episodes} | "
            f"Energy: {np.mean(energies):.3f} | "
            f"Dominance: {np.mean(dominances):.2%}"
        )
    
    def _check_failure(self, metrics: Dict[str, float]) -> Optional[str]:
        """Check governance metrics against thresholds"""
        energy = metrics.get("energy_raw", 0)
        if energy > self.thresholds.energy_max:
            return f"Energy exceeded: {energy:.2f} > {self.thresholds.energy_max:.2f}"
        
        hrm = metrics.get("hrm_alignment", 1.0)
        if hrm < self.thresholds.hrm_min:
            return f"HRM alignment collapsed: {hrm:.2f} < {self.thresholds.hrm_min:.2f}"
        
        margin = metrics.get("embedding_margin", 0.0)
        if margin < self.thresholds.margin_min:
            return f"Embedding margin collapsed: {margin:.2f} < {self.thresholds.margin_min:.2f}"
        
        return None
    
    def _build_success_result(self) -> Dict[str, Any]:
        """Build success result dictionary"""
        if not self.episode_history:
            return {"status": "failed", "reason": "no episodes completed"}
        
        energies = [ep["metrics"].get("energy_raw", 0) for ep in self.episode_history]
        hrms = [ep["metrics"].get("hrm_alignment", 0) for ep in self.episode_history]
        dominances = [ep["dominance_achieved"] for ep in self.episode_history]
        
        return {
            "status": "success",
            "episodes_completed": len(self.episode_history),
            "metrics_summary": {
                "energy": {
                    "mean": float(np.mean(energies)),
                    "std": float(np.std(energies)),
                    "min": float(np.min(energies)),
                    "max": float(np.max(energies)),
                },
                "hrm_alignment": {
                    "mean": float(np.mean(hrms)),
                    "std": float(np.std(hrms)),
                },
                "dominance_rate": float(np.mean(dominances)),
            },
            "retention_scores": self.retention_tracker.get_scores(),
            "failure_history": self.collapse_detector.get_failure_history(),
        }
    
    def _build_failure_result(self, episode: int, failure: Any) -> Dict[str, Any]:
        """Build failure result dictionary"""
        return {
            "status": "failed",
            "episode": episode,
            "failure": failure.to_dict() if hasattr(failure, "to_dict") else str(failure),
            "metrics_summary": self._build_success_result()["metrics_summary"],
        }

==================================================
FILE: experiment\experiment_persistence.py
==================================================

from typing import Any, Dict, List
from stephanie.data.score_bundle import ScoreBundle

import logging
logger = logging.getLogger(__name__)

class ExperimentPersistence:
    """
    Leverage your existing ScoreBundle persistence infrastructure.
    
    No new database schema needed.
    """
    
    def __init__(self, memory_container: Any):
        self.memory = memory_container
    
    def log_experiment_episode(
        self,
        episode: int,
        query: Any,
        bundle_before: "ScoreBundle",
        bundle_after: "ScoreBundle",
        dominance_achieved: bool,
        experiment_metadata: Dict[str, Any]
    ):
        """
        Log experiment episode using your existing EvaluationORM.
        
        Adds experiment-specific metadata to the bundle's meta field.
        """
        # Add experiment metadata to bundle
        bundle_after.meta.update({
            "experiment_episode": episode,
            "dominance_achieved": dominance_achieved,
            "experiment_metadata": experiment_metadata,
        })
        
        # Use your existing save_bundle method
        self.memory.evaluations.save_bundle(
            bundle_after,
            scorable=query,
            context={"experiment_episode": episode},
            cfg={},  # Empty or experiment-specific config
            source="experiment_governed",
            embedding_type="experiment",
            evaluator_name="GovernedSelfImprovement",
        )
    
    def query_experiment_results(
        self,
        experiment_episodes: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Query experiment results from your database.
        
        Uses your existing ORM layer.
        """
        from sqlalchemy import select
        from stephanie.data.orm import EvaluationORM
        
        with self.memory.session() as s:
            stmt = (
                select(EvaluationORM)
                .where(EvaluationORM.source == "experiment_governed")
                .where(EvaluationORM.meta["experiment_episode"].astext.in_(
                    [str(ep) for ep in experiment_episodes]
                ))
                .order_by(EvaluationORM.created_at)
            )
            
            results = s.execute(stmt).scalars().all()
            
            return [
                {
                    "episode": eval.meta.get("experiment_episode"),
                    "bundle": eval.scores,  # Your stored bundle dict
                    "dominance": eval.meta.get("dominance_achieved"),
                }
                for eval in results
            ]

==================================================
FILE: experiment\logging_schema.py
==================================================

# JSONL logging schema


==================================================
FILE: experiment\perturbation_injector.py
==================================================

"""
PerturbationInjector: Controlled stress testing for governed self-improvement systems.

Injects calibrated perturbations to validate:
- Governance regime switching responsiveness
- Recovery velocity from instability
- Retention of safety invariants under stress
- Collapse detector sensitivity

All perturbations are reversible via restore_original_state().
Designed for single-episode injection with explicit restoration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerturbationConfig:
    """Immutable configuration for perturbation severity levels"""
    misleading_evidence_prob: float  # Probability of injecting misleading evidence snippets
    complexity_factor: float         # Query complexity multiplier
    governance_disable_timeout: int  # Episodes to disable governance (0 = never disable)
    description: str
    
    def validate(self) -> None:
        """Validate configuration constraints"""
        if not (0.0 <= self.misleading_evidence_prob <= 1.0):
            raise ValueError(f"Evidence probability must be in [0,1], got {self.misleading_evidence_prob}")
        if self.complexity_factor < 1.0:
            raise ValueError(f"Complexity factor must be >= 1.0, got {self.complexity_factor}")
        if self.governance_disable_timeout < 0:
            raise ValueError(f"Timeout must be non-negative, got {self.governance_disable_timeout}")


class PerturbationInjector:
    """
    Controlled perturbation injection system for experimental stress testing.
    
    Design principles:
    - Explicit severity levels with documented effects
    - Full reversibility via system.restore_original_state()
    - No internal state persistence (stateless between injections)
    - Comprehensive logging for forensic analysis
    - Type-safe configuration with validation
    
    Usage:
        injector = PerturbationInjector(system)
        
        # Inject at specified episode
        injector.inject(severity="moderate")
        
        # System processes perturbed query
        
        # Restore after episode completes
        injector.restore()
    
    Critical: restore() MUST be called after perturbation episode to prevent
    contamination of subsequent episodes. Experiment harness should handle this.
    """
    
    # Predefined severity configurations
    SEVERITY_CONFIGS: Dict[str, PerturbationConfig] = {
        "light": PerturbationConfig(
            misleading_evidence_prob=0.2,
            complexity_factor=1.2,
            governance_disable_timeout=0,
            description="Mild stress test: minor evidence corruption, slight complexity increase"
        ),
        "moderate": PerturbationConfig(
            misleading_evidence_prob=0.4,
            complexity_factor=1.5,
            governance_disable_timeout=0,
            description="Standard stress test: significant evidence corruption, moderate complexity increase"
        ),
        "severe": PerturbationConfig(
            misleading_evidence_prob=0.6,
            complexity_factor=2.0,
            governance_disable_timeout=3,
            description="Extreme stress test: heavy evidence corruption, high complexity, temporary governance suspension"
        )
    }
    
    def __init__(self, system: Any):
        """
        Initialize perturbation injector.
        
        Args:
            system: System implementing SystemInterface perturbation methods
                Required methods:
                - inject_misleading_evidence(probability: float)
                - increase_query_complexity(factor: float)
                - temporarily_disable_governance(timeout: int) [optional]
                - restore_original_state()
        """
        self.system = system
        self._active_severity: Optional[str] = None
        self._injection_count: int = 0
        
        logger.info("PerturbationInjector initialized")
    
    def inject(self, severity: str = "moderate") -> Dict[str, Any]:
        """
        Inject controlled perturbation at specified severity level.
        
        Args:
            severity: One of "light", "moderate", "severe"
            
        Returns:
            Dict with injection details for experiment logging:
                {
                    "severity": str,
                    "evidence_prob": float,
                    "complexity_factor": float,
                    "governance_disabled": bool,
                    "timestamp": float
                }
        
        Raises:
            ValueError: If severity level is invalid
            RuntimeError: If system lacks required perturbation methods
            Exception: Any unexpected injection failure (logged and re-raised)
        """
        # Validate severity
        if severity not in self.SEVERITY_CONFIGS:
            valid = ", ".join(self.SEVERITY_CONFIGS.keys())
            raise ValueError(f"Invalid severity '{severity}'. Must be one of: {valid}")
        
        config = self.SEVERITY_CONFIGS[severity]
        config.validate()
        
        try:
            logger.warning(f".Injecting {severity.upper()} perturbation | {config.description}")
            
            # Inject misleading evidence
            if config.misleading_evidence_prob > 0:
                self.system.inject_misleading_evidence(
                    probability=config.misleading_evidence_prob
                )
                logger.info(
                    f"✓ Misleading evidence injected (p={config.misleading_evidence_prob:.2f})"
                )
            
            # Increase query complexity
            if config.complexity_factor > 1.0:
                self.system.increase_query_complexity(
                    factor=config.complexity_factor
                )
                logger.info(
                    f"✓ Query complexity increased (factor={config.complexity_factor:.1f})"
                )
            
            # Temporarily disable governance (if configured)
            governance_disabled = False
            if config.governance_disable_timeout > 0:
                try:
                    self.system.temporarily_disable_governance(
                        timeout=config.governance_disable_timeout
                    )
                    governance_disabled = True
                    logger.warning(
                        f"⚠️  Governance temporarily disabled for {config.governance_disable_timeout} episodes"
                    )
                except AttributeError:
                    logger.warning(
                        "System does not support governance disabling - skipping this perturbation component"
                    )
            
            # Track injection
            self._active_severity = severity
            self._injection_count += 1
            
            # Return injection metadata for experiment logging
            return {
                "severity": severity,
                "evidence_prob": config.misleading_evidence_prob,
                "complexity_factor": config.complexity_factor,
                "governance_disabled": governance_disabled,
                "governance_timeout": config.governance_disable_timeout,
                "injection_count": self._injection_count,
                "description": config.description
            }
            
        except AttributeError as e:
            missing_method = str(e).split("'")[-2] if "'" in str(e) else "unknown"
            error_msg = (
                f"System missing required perturbation method: {missing_method}. "
                f"Ensure system implements SystemInterface perturbation methods."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            logger.exception(f"Perturbation injection failed: {e}")
            raise
    
    def restore(self) -> bool:
        """
        Restore system to pre-perturbation state.
        
        Critical: Must be called after perturbation episode completes to prevent
        contamination of subsequent episodes.
        
        Returns:
            True if restoration successful, False if no active perturbation
        
        Raises:
            RuntimeError: If restoration fails catastrophically
        """
        if self._active_severity is None:
            logger.debug("No active perturbation to restore")
            return False
        
        try:
            logger.info(
                f"Restoring system state after {self._active_severity.upper()} perturbation "
                f"(injection #{self._injection_count})"
            )
            
            self.system.restore_original_state()
            
            # Clear state
            prev_severity = self._active_severity
            self._active_severity = None
            
            logger.info(
                f"✓ System restored to pre-perturbation state after {prev_severity} injection"
            )
            return True
            
        except Exception as e:
            logger.exception(f"Restoration failed: {e}")
            raise RuntimeError(f"Failed to restore system state: {e}") from e
    
    def is_active(self) -> bool:
        """Check if perturbation is currently active"""
        return self._active_severity is not None
    
    def get_active_severity(self) -> Optional[str]:
        """Get current active perturbation severity, or None if inactive"""
        return self._active_severity
    
    def get_injection_count(self) -> int:
        """Get total number of perturbations injected"""
        return self._injection_count
    
    def __enter__(self):
        """
        Context manager support for automatic restoration.
        
        Usage:
            with PerturbationInjector(system) as injector:
                injector.inject("moderate")
                # Perturbation active within context
            # Automatic restoration on exit
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure restoration on context exit"""
        if self.is_active():
            try:
                self.restore()
            except Exception as e:
                logger.error(f"Context manager restoration failed: {e}")
                # Don't suppress original exception if one occurred
                if exc_type is None:
                    raise
        return False  # Propagate exceptions
    
    def __str__(self) -> str:
        status = f"active:{self._active_severity}" if self.is_active() else "inactive"
        return f"PerturbationInjector({status}, injections={self._injection_count})"
    
    def __repr__(self) -> str:
        return self.__str__()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_perturbation_config(
    misleading_evidence_prob: float,
    complexity_factor: float,
    governance_disable_timeout: int = 0,
    description: str = "custom"
) -> PerturbationConfig:
    """
    Create custom perturbation configuration.
    
    Args:
        misleading_evidence_prob: Probability of injecting misleading evidence [0.0, 1.0]
        complexity_factor: Query complexity multiplier (>=1.0)
        governance_disable_timeout: Episodes to disable governance (0 = never)
        description: Human-readable description
    
    Returns:
        Validated PerturbationConfig instance
    """
    config = PerturbationConfig(
        misleading_evidence_prob=misleading_evidence_prob,
        complexity_factor=complexity_factor,
        governance_disable_timeout=governance_disable_timeout,
        description=description
    )
    config.validate()
    return config


def register_custom_severity(
    injector: PerturbationInjector,
    name: str,
    config: PerturbationConfig
) -> None:
    """
    Register custom severity level to injector's SEVERITY_CONFIGS.
    
    Args:
        injector: PerturbationInjector instance
        name: Severity name (e.g., "extreme", "debug")
        config: Validated PerturbationConfig
    
    Raises:
        ValueError: If name conflicts with existing severity
    """
    if name in injector.SEVERITY_CONFIGS:
        raise ValueError(f"Severity '{name}' already exists. Use unique name.")
    
    config.validate()
    injector.SEVERITY_CONFIGS[name] = config
    logger.info(f"Registered custom severity: {name} | {config.description}")


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "PerturbationInjector",
    "PerturbationConfig",
    "create_perturbation_config",
    "register_custom_severity"
]

==================================================
FILE: experiment\__init__.py
==================================================



==================================================
FILE: governance\dominance_checker.py
==================================================

"""
DominanceChecker: Validates strict multi-objective improvement across critical axes.

Ensures:
- No reward hacking (trading one axis for another)
- Direction-aware improvement (higher/lower is better)
- Configurable critical axis sets
- Detailed failure diagnostics
- Trace-native comparison via ScoreBundle.diff()

Critical for preventing unsafe updates in governed self-improvement.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from stephanie.data.score_bundle import ScoreBundle
from stephanie.components.elm.governance.signal_extractor import (
    AxisDirection,
    DIMENSION_TO_AXIS,
    AXIS_SEMANTICS
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DominanceResult:
    """Structured dominance validation result"""
    is_dominant: bool
    failed_axes: List[str] = field(default_factory=list)
    passed_axes: List[str] = field(default_factory=list)
    delta_summary: Dict[str, float] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "is_dominant": self.is_dominant,
            "failed_axes": self.failed_axes,
            "passed_axes": self.passed_axes,
            "delta_summary": self.delta_summary,
            "failure_reasons": self.failure_reasons
        }
    
    def __bool__(self) -> bool:
        return self.is_dominant


class DominanceChecker:
    """
    Validates strict Pareto dominance between ScoreBundle instances.
    
    Design principles:
    - Direction-aware improvement semantics
    - Configurable critical axis sets
    - Detailed failure diagnostics
    - Zero tolerance for safety axis regression
    - Trace-native comparison via ScoreBundle.diff()
    
    Usage:
        checker = DominanceChecker(
            critical_dimensions=["alignment", "energy", "margin"],
            safety_dimensions=["energy"]  # Zero-tolerance axes
        )
        
        result = checker.check(bundle_before, bundle_after)
        if result.is_dominant:
            system.commit_improvement(bundle_after)
        else:
            logger.warning(f"Dominance failed: {result.failure_reasons}")
    """
    
    def __init__(
        self,
        critical_dimensions: Optional[List[str]] = None,
        safety_dimensions: Optional[List[str]] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize dominance checker.
        
        Args:
            critical_dimensions: Dimensions requiring strict improvement
            safety_dimensions: Subset requiring zero-tolerance (no regression allowed)
            tolerance: Numerical tolerance for delta comparison
        """
        self.critical_dimensions = critical_dimensions or ["alignment", "energy", "margin"]
        self.safety_dimensions = safety_dimensions or ["energy"]  # Hallucination energy is safety-critical
        self.tolerance = tolerance
        
        # Validate dimension mappings
        self._validate_dimensions()
        
        logger.info(
            f"DominanceChecker initialized | "
            f"critical={self.critical_dimensions} | "
            f"safety={self.safety_dimensions}"
        )
    
    def _validate_dimensions(self):
        """Validate all dimensions map to known governance axes"""
        unknown = []
        for dim in self.critical_dimensions + self.safety_dimensions:
            if dim not in DIMENSION_TO_AXIS:
                unknown.append(dim)
        
        if unknown:
            raise ValueError(
                f"Unknown dimensions in dominance config: {unknown}. "
                f"Available dimensions: {list(DIMENSION_TO_AXIS.keys())}"
            )
    
    def check(
        self,
        bundle_before: ScoreBundle,
        bundle_after: ScoreBundle,
        strict_safety: bool = True
    ) -> DominanceResult:
        """
        Check if bundle_after dominates bundle_before on all critical dimensions.
        
        Args:
            bundle_before: Baseline ScoreBundle
            bundle_after: Candidate improved ScoreBundle
            strict_safety: If True, safety dimensions must show strict improvement (no tolerance)
            
        Returns:
            DominanceResult with detailed diagnostics
        """
        diff = bundle_after.diff(bundle_before)
        failed_axes = []
        passed_axes = []
        delta_summary = {}
        failure_reasons = []
        
        # Check each critical dimension
        for dim in self.critical_dimensions:
            if dim not in diff.get("dimensions", {}):
                failed_axes.append(dim)
                failure_reasons.append(f"Dimension '{dim}' missing in diff comparison")
                continue
            
            dim_diff = diff["dimensions"][dim]
            delta = dim_diff.get("score_delta", 0.0)
            delta_summary[dim] = delta
            
            # Map to governance axis
            axis = DIMENSION_TO_AXIS[dim]
            direction = AXIS_SEMANTICS[axis]
            
            # Determine if improvement occurred
            is_improved = self._is_improvement(delta, direction, dim in self.safety_dimensions and strict_safety)
            
            if is_improved:
                passed_axes.append(dim)
            else:
                failed_axes.append(dim)
                direction_str = "increase" if direction == AxisDirection.HIGHER_IS_BETTER else "decrease"
                failure_reasons.append(
                    f"Dimension '{dim}' failed: expected {direction_str} (delta={delta:+.4f})"
                )
        
        is_dominant = len(failed_axes) == 0
        
        if not is_dominant:
            logger.debug(
                f"Dominance check failed | "
                f"passed={passed_axes} | "
                f"failed={failed_axes} | "
                f"reasons={failure_reasons}"
            )
        
        return DominanceResult(
            is_dominant=is_dominant,
            failed_axes=failed_axes,
            passed_axes=passed_axes,
            delta_summary=delta_summary,
            failure_reasons=failure_reasons
        )
    
    def _is_improvement(
        self,
        delta: float,
        direction: AxisDirection,
        is_safety_axis: bool
    ) -> bool:
        """
        Determine if delta represents improvement given direction semantics.
        
        Args:
            delta: Score delta (after - before)
            direction: Axis direction semantics
            is_safety_axis: If True, apply zero-tolerance check
            
        Returns:
            True if delta represents improvement
        """
        if direction == AxisDirection.HIGHER_IS_BETTER:
            # For higher-is-better: delta must be positive
            if is_safety_axis:
                return delta > 0  # Strict: must improve
            return delta > -self.tolerance  # Allow tiny numerical noise
        else:  # LOWER_IS_BETTER
            # For lower-is-better: delta must be negative (value decreased)
            if is_safety_axis:
                return delta < 0  # Strict: must improve
            return delta < self.tolerance  # Allow tiny numerical noise
    
    def get_critical_axes(self) -> List[str]:
        """Get current critical dimension configuration"""
        return list(self.critical_dimensions)
    
    def add_critical_dimension(self, dimension: str) -> None:
        """Add dimension to critical set (runtime configuration)"""
        if dimension not in DIMENSION_TO_AXIS:
            raise ValueError(f"Unknown dimension: {dimension}")
        
        if dimension not in self.critical_dimensions:
            self.critical_dimensions.append(dimension)
            logger.info(f"Added critical dimension: {dimension}")
    
    def remove_critical_dimension(self, dimension: str) -> None:
        """Remove dimension from critical set"""
        if dimension in self.critical_dimensions:
            self.critical_dimensions.remove(dimension)
            logger.info(f"Removed critical dimension: {dimension}")
    
    def __str__(self) -> str:
        return (
            f"DominanceChecker("
            f"critical={len(self.critical_dimensions)}, "
            f"safety={len(self.safety_dimensions)})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_strict_dominance_checker() -> DominanceChecker:
    """
    Create dominance checker with strict safety constraints.
    
    Suitable for high-stakes applications where hallucination safety is non-negotiable.
    """
    return DominanceChecker(
        critical_dimensions=["alignment", "energy", "margin", "context_fidelity"],
        safety_dimensions=["energy", "alignment"],  # Zero tolerance on safety axes
        tolerance=1e-8  # Extremely strict numerical tolerance
    )


def create_research_dominance_checker() -> DominanceChecker:
    """
    Create dominance checker optimized for research settings.
    
    More permissive on non-safety axes to allow exploration.
    """
    return DominanceChecker(
        critical_dimensions=["alignment", "energy", "margin"],
        safety_dimensions=["energy"],  # Only energy is safety-critical
        tolerance=1e-4  # Allow minor numerical fluctuations
    )


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "DominanceChecker",
    "DominanceResult",
    "create_strict_dominance_checker",
    "create_research_dominance_checker"
]

==================================================
FILE: governance\regime_controller.py
==================================================

"""
RegimeController: Dynamically adapts system behavior based on stability metrics.

Implements energy-based regime control:
- SAFE: Normal operation velocity
- WARNING: Conservative updates, increased scrutiny
- CRITICAL: Safety interventions, potential rollback

Translates governance metrics into concrete policy actions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from stephanie.components.elm.core.thresholds import CalibratedThresholds

logger = logging.getLogger(__name__)


class PolicyRegime(str, Enum):
    """Policy regime states with semantic meaning"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class RegimeAction:
    """Structured policy action recommendation"""
    action_type: str  # "freeze", "reduce", "increase", "rollback"
    target_component: str  # "embedding_updates", "reflection_depth", "governance"
    magnitude: float  # 0.0 to 1.0 scaling factor
    description: str
    urgency: str  # "low", "medium", "high", "immediate"


@dataclass
class RegimeState:
    """Current regime state with transition history"""
    current_regime: PolicyRegime
    previous_regime: Optional[PolicyRegime] = None
    episode: int = 0
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)
    actions_taken: List[RegimeAction] = field(default_factory=list)
    regime_duration: int = 1  # Episodes in current regime
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "current_regime": self.current_regime.value,
            "previous_regime": self.previous_regime.value if self.previous_regime else None,
            "episode": self.episode,
            "metrics_snapshot": self.metrics_snapshot,
            "actions_taken": [a.__dict__ for a in self.actions_taken],
            "regime_duration": self.regime_duration
        }


class RegimeController:
    """
    Adaptive policy controller that responds to system stability metrics.
    
    Design principles:
    - Energy-driven regime transitions (fastest instability signal)
    - Hysteresis to prevent oscillation
    - Action recommendations with concrete parameters
    - Transition logging for forensic analysis
    - Configurable regime boundaries
    
    Usage:
        controller = RegimeController(thresholds)
        
        # Determine regime from current metrics
        regime_state = controller.update(metrics, episode=42)
        
        # Get recommended actions
        actions = regime_state.actions_taken
        
        # Apply actions to system
        for action in actions:
            if action.action_type == "freeze" and action.target_component == "embedding_updates":
                system.freeze_embedding_updates()
    """
    
    # Hysteresis parameters (prevent rapid regime oscillation)
    HYSTERESIS = {
        PolicyRegime.SAFE: 0.02,      # Must exceed warning threshold by margin to leave SAFE
        PolicyRegime.WARNING: 0.03,   # Must exceed critical threshold by margin to leave WARNING
        PolicyRegime.CRITICAL: 0.05   # Must drop well below critical to leave CRITICAL
    }
    
    def __init__(
        self,
        thresholds: CalibratedThresholds,
        enable_hysteresis: bool = True,
        action_callbacks: Optional[List[Callable[[RegimeAction], None]]] = None
    ):
        """
        Initialize regime controller.
        
        Args:
            thresholds: Calibrated safety thresholds
            enable_hysteresis: Prevent rapid regime oscillation
            action_callbacks: Functions to call when actions are recommended
        """
        self.thresholds = thresholds
        self.enable_hysteresis = enable_hysteresis
        self.action_callbacks = action_callbacks or []
        
        # State tracking
        self._current_regime: PolicyRegime = PolicyRegime.SAFE
        self._regime_start_episode: int = 0
        self._previous_metrics: Dict[str, float] = {}
        
        logger.info(
            f"RegimeController initialized | "
            f"hysteresis={'enabled' if enable_hysteresis else 'disabled'}"
        )
    
    def update(
        self,
        metrics: Dict[str, float],
        episode: int
    ) -> RegimeState:
        """
        Update regime state based on current metrics.
        
        Args:
            metrics: Current governance metrics
            episode: Current episode number
            
        Returns:
            RegimeState with current regime and recommended actions
        """
        # Determine target regime
        target_regime = self._determine_target_regime(metrics)
        
        # Apply hysteresis if enabled
        if self.enable_hysteresis:
            target_regime = self._apply_hysteresis(target_regime, metrics)
        
        # Detect regime transition
        is_transition = target_regime != self._current_regime
        
        # Update state
        if is_transition:
            logger.warning(
                f"REGIME TRANSITION: {self._current_regime.value.upper()} → "
                f"{target_regime.value.upper()} at episode {episode}"
            )
            self._previous_regime = self._current_regime
            self._current_regime = target_regime
            self._regime_start_episode = episode
            self._previous_metrics = metrics.copy()
        else:
            self._previous_metrics = metrics.copy()
        
        # Generate actions
        actions = self._generate_actions(target_regime, metrics, episode, is_transition)
        
        # Execute callbacks
        for action in actions:
            for callback in self.action_callbacks:
                try:
                    callback(action)
                except Exception as e:
                    logger.error(f"Action callback failed: {e}")
        
        return RegimeState(
            current_regime=target_regime,
            previous_regime=self._previous_regime if is_transition else self._current_regime,
            episode=episode,
            metrics_snapshot=metrics.copy(),
            actions_taken=actions,
            regime_duration=episode - self._regime_start_episode + 1
        )
    
    def _determine_target_regime(self, metrics: Dict[str, float]) -> PolicyRegime:
        """Determine target regime based on metrics and thresholds"""
        energy = metrics.get("energy_raw", 0.0)
        hrm = metrics.get("hrm_alignment", 1.0)
        margin = metrics.get("embedding_margin", 1.0)
        variance = metrics.get("embedding_variance", 1.0)
        collapse_index = metrics.get("collapse_index", 1.0)
        drift = metrics.get("angular_drift", 0.0)
        
        # CRITICAL checks (any trigger critical regime)
        if (energy > self.thresholds.energy_max or
            hrm < self.thresholds.hrm_min or
            margin < self.thresholds.margin_min or
            variance < self.thresholds.variance_min or
            collapse_index > self.thresholds.collapse_index_max or
            drift > self.thresholds.drift_max):
            return PolicyRegime.CRITICAL
        
        # WARNING checks
        if energy > self.thresholds.energy_warning:
            return PolicyRegime.WARNING
        
        return PolicyRegime.SAFE
    
    def _apply_hysteresis(
        self,
        target_regime: PolicyRegime,
        metrics: Dict[str, float]
    ) -> PolicyRegime:
        """Apply hysteresis to prevent rapid regime oscillation"""
        if self._current_regime == target_regime:
            return target_regime
        
        energy = metrics.get("energy_raw", 0.0)
        prev_energy = self._previous_metrics.get("energy_raw", energy)
        
        # Hysteresis when leaving SAFE regime
        if self._current_regime == PolicyRegime.SAFE and target_regime == PolicyRegime.WARNING:
            if energy < (self.thresholds.energy_warning + self.HYSTERESIS[PolicyRegime.SAFE]):
                return PolicyRegime.SAFE
        
        # Hysteresis when leaving WARNING regime
        if self._current_regime == PolicyRegime.WARNING and target_regime == PolicyRegime.CRITICAL:
            if energy < (self.thresholds.energy_max + self.HYSTERESIS[PolicyRegime.WARNING]):
                return PolicyRegime.WARNING
        
        # Hysteresis when leaving CRITICAL regime (requires significant improvement)
        if self._current_regime == PolicyRegime.CRITICAL and target_regime != PolicyRegime.CRITICAL:
            if energy > (self.thresholds.energy_max - self.HYSTERESIS[PolicyRegime.CRITICAL]):
                return PolicyRegime.CRITICAL
        
        return target_regime
    
    def _generate_actions(
        self,
        regime: PolicyRegime,
        metrics: Dict[str, float],
        episode: int,
        is_transition: bool
    ) -> List[RegimeAction]:
        """Generate concrete policy actions for current regime"""
        actions = []
        energy = metrics.get("energy_raw", 0.0)
        
        if regime == PolicyRegime.SAFE:
            if is_transition:
                actions.append(RegimeAction(
                    action_type="restore",
                    target_component="embedding_updates",
                    magnitude=1.0,
                    description="Restoring normal embedding update velocity",
                    urgency="low"
                ))
                actions.append(RegimeAction(
                    action_type="restore",
                    target_component="reflection_depth",
                    magnitude=1.0,
                    description="Restoring standard reflection depth",
                    urgency="low"
                ))
        
        elif regime == PolicyRegime.WARNING:
            actions.append(RegimeAction(
                action_type="reduce",
                target_component="embedding_update_magnitude",
                magnitude=0.5,
                description=f"Reducing embedding updates by 50% (energy={energy:.3f})",
                urgency="medium"
            ))
            actions.append(RegimeAction(
                action_type="increase",
                target_component="reflection_depth",
                magnitude=1.5,
                description="Increasing reflection depth for thorough correction",
                urgency="medium"
            ))
            actions.append(RegimeAction(
                action_type="increase",
                target_component="hard_negative_sampling",
                magnitude=2.0,
                description="Doubling hard negative sampling for stability",
                urgency="medium"
            ))
        
        elif regime == PolicyRegime.CRITICAL:
            actions.append(RegimeAction(
                action_type="freeze",
                target_component="embedding_updates",
                magnitude=0.0,
                description=f"FREEZING embedding updates (energy={energy:.3f} > {self.thresholds.energy_max:.3f})",
                urgency="immediate"
            ))
            actions.append(RegimeAction(
                action_type="increase",
                target_component="hrm_weighting",
                magnitude=2.0,
                description="Doubling HRM alignment weighting for grounding",
                urgency="high"
            ))
            actions.append(RegimeAction(
                action_type="enforce",
                target_component="grounding_constraints",
                magnitude=1.0,
                description="Enforcing strict evidence grounding constraints",
                urgency="high"
            ))
            
            # Consider rollback if this is a transition into CRITICAL
            if is_transition and episode > 0:
                actions.append(RegimeAction(
                    action_type="rollback",
                    target_component="recent_updates",
                    magnitude=1.0,
                    description="Recommending rollback of recent updates",
                    urgency="high"
                ))
        
        return actions
    
    def get_current_regime(self) -> PolicyRegime:
        """Get current regime state"""
        return self._current_regime
    
    def force_regime(self, regime: PolicyRegime, episode: int) -> RegimeState:
        """
        Force regime transition (for testing or emergency intervention).
        
        Use with extreme caution - bypasses threshold checks.
        """
        logger.critical(f"FORCING REGIME TRANSITION TO {regime.value.upper()}")
        self._previous_regime = self._current_regime
        self._current_regime = regime
        self._regime_start_episode = episode
        
        # Generate actions for forced regime
        dummy_metrics = {"energy_raw": 0.0}
        actions = self._generate_actions(regime, dummy_metrics, episode, is_transition=True)
        
        return RegimeState(
            current_regime=regime,
            previous_regime=self._previous_regime,
            episode=episode,
            metrics_snapshot=dummy_metrics,
            actions_taken=actions,
            regime_duration=1
        )
    
    def __str__(self) -> str:
        return f"RegimeController(current={self._current_regime.value})"
    
    def __repr__(self) -> str:
        return self.__str__()


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "RegimeController",
    "RegimeState",
    "RegimeAction",
    "PolicyRegime"
]

==================================================
FILE: governance\signal_extractor.py
==================================================

from typing import Dict, List
from enum import Enum
from stephanie.data.score_bundle import ScoreBundle
from stephanie.components.elm.reward_vector import RewardAxis
from stephanie.data.score_result import ScoreResult

class AxisDirection(str, Enum):
    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"

# Map ScoreBundle dimensions to governance axes
DIMENSION_TO_AXIS: Dict[str, RewardAxis] = {
    # HRM dimensions
    "alignment": RewardAxis.HRM_ALIGNMENT,
    "reasoning_quality": RewardAxis.HRM_ALIGNMENT,
    
    # EBT energy dimensions (lower is better)
    "energy": RewardAxis.HALLUCINATION_ENERGY,
    "speculation": RewardAxis.HALLUCINATION_ENERGY,
    
    # Embedding dimensions
    "margin": RewardAxis.EMBEDDING_MARGIN,
    "similarity": RewardAxis.EMBEDDING_MARGIN,
    
    # Policy dimensions
    "advantage": RewardAxis.POLICY_ADVANTAGE,
    "context_grounding": RewardAxis.CONTEXT_FIDELITY,
}

AXIS_SEMANTICS: Dict[RewardAxis, AxisDirection] = {
    RewardAxis.HRM_ALIGNMENT: AxisDirection.HIGHER_IS_BETTER,
    RewardAxis.HALLUCINATION_ENERGY: AxisDirection.LOWER_IS_BETTER,
    RewardAxis.EMBEDDING_MARGIN: AxisDirection.HIGHER_IS_BETTER,
    RewardAxis.POLICY_ADVANTAGE: AxisDirection.HIGHER_IS_BETTER,
    RewardAxis.METRIC_ALIGNMENT: AxisDirection.HIGHER_IS_BETTER,
    RewardAxis.COHERENCE: AxisDirection.HIGHER_IS_BETTER,
    RewardAxis.CONTEXT_FIDELITY: AxisDirection.HIGHER_IS_BETTER,
}

class GovernanceSignalExtractor:
    """
    Extract governance signals from ScoreBundle.
    
    Converts your dynamic ScoreBundle into structured governance metrics.
    """
    
    def __init__(self, critical_dimensions: List[str] = None):
        self.critical_dimensions = critical_dimensions or [
            "alignment", "energy", "margin"
        ]
    
    def extract_from_bundle(self, bundle: "ScoreBundle") -> Dict[str, float]:
        """
        Extract governance metrics from ScoreBundle.
        
        Returns:
            Dict with keys: energy, hrm_alignment, embedding_margin, etc.
        """
        metrics = {}
        
        for dim_name, result in bundle.results.items():
            # Map dimension to governance axis
            axis = DIMENSION_TO_AXIS.get(dim_name)
            if not axis:
                continue  # Skip non-governance dimensions
            
            # Extract score (normalized to 0-1)
            score = self._normalize_score(result.score, dim_name)
            
            # Store by axis name
            axis_key = axis.value
            metrics[axis_key] = score
            
            # Extract energy specifically (from attributes)
            if axis == RewardAxis.HALLUCINATION_ENERGY:
                energy = self._extract_energy(result)
                metrics["energy_raw"] = energy
        
        return metrics
    
    def _normalize_score(self, score: float, dimension: str) -> float:
        """
        Normalize score to [0, 1] based on dimension semantics.
        
        Your scores are 0-100, so divide by 100.
        For energy (lower=better), invert.
        """
        # Your scores are 0-100
        normalized = score / 100.0
        
        # Clamp to [0, 1]
        normalized = max(0.0, min(1.0, normalized))
        
        # Invert if lower-is-better dimension
        if dimension in ["energy", "speculation"]:
            normalized = 1.0 - normalized
        
        return normalized
    
    def _extract_energy(self, result: "ScoreResult") -> float:
        """
        Extract raw energy from attributes.
        
        Your EBT stores raw_energy in attributes.
        """
        if hasattr(result, "attributes") and result.attributes:
            raw_energy = result.attributes.get("raw_energy")
            if raw_energy is not None:
                return float(raw_energy)
        
        # Fallback: use score as proxy
        return 100.0 - result.score  # Higher score = lower energy
    
    def compute_dominance(
        self,
        bundle_before: "ScoreBundle",
        bundle_after: "ScoreBundle"
    ) -> bool:
        """
        Check if bundle_after dominates bundle_before on critical dimensions.
        
        Uses ScoreBundle.diff() for precise comparison.
        """
        diff = bundle_after.diff(bundle_before)
        
        for dim in self.critical_dimensions:
            if dim not in diff.get("dimensions", {}):
                continue
            
            dim_diff = diff["dimensions"][dim]
            delta = dim_diff.get("score_delta", 0)
            
            # Check direction semantics
            axis = DIMENSION_TO_AXIS.get(dim)
            if not axis:
                continue
            
            direction = AXIS_SEMANTICS[axis]
            
            if direction == AxisDirection.HIGHER_IS_BETTER:
                if delta <= 0:
                    return False  # Must improve
            else:  # LOWER_IS_BETTER
                if delta >= 0:
                    return False  # Must decrease
        
        return True  # All critical dimensions improved
    
    def compute_delta_vector(
        self,
        bundle_before: "ScoreBundle",
        bundle_after: "ScoreBundle"
    ) -> Dict[str, float]:
        """
        Compute direction-normalized delta vector.
        
        Positive = improvement, regardless of axis direction.
        """
        diff = bundle_after.diff(bundle_before)
        delta_vector = {}
        
        for dim, dim_diff in diff.get("dimensions", {}).items():
            axis = DIMENSION_TO_AXIS.get(dim)
            if not axis:
                continue
            
            delta = dim_diff.get("score_delta", 0)
            direction = AXIS_SEMANTICS[axis]
            
            # Normalize delta to [-1, 1]
            normalized_delta = delta / 100.0
            
            # Direction-aware: positive always = improvement
            if direction == AxisDirection.LOWER_IS_BETTER:
                normalized_delta = -normalized_delta
            
            delta_vector[axis.value] = normalized_delta
        
        return delta_vector

==================================================
FILE: orchestration\orchestrator.py
==================================================


class ELMOrchestrator:

    def __init__(
        self,
        core_evaluator,
        governance_reducer,
        reflection_engine
    ):
        self.core = core_evaluator
        self.governance = governance_reducer
        self.reflector = reflection_engine

    def step(self, context_pack, plan_trace, model_output):

        # 1. Core evaluation
        base_bundle = self.core.evaluate(
            context_pack=context_pack,
            plan_trace=plan_trace,
            output=model_output
        )

        # 2. Governance reduction
        governed_bundle = self.governance.evaluate(
            context_pack=context_pack,
            plan_trace=plan_trace,
            output=model_output,
            base_bundle=base_bundle
        )

        # 3. Reflection decision
        if governed_bundle.reward_vector.failure_signatures:
            reflection = self.reflector.generate(governed_bundle)
            return governed_bundle, reflection

        return governed_bundle, None


==================================================
FILE: orchestration\system_interface.py
==================================================

"""
SystemInterface: Contract definition for Stephanie engine integration with ELM experimental harness.

This protocol defines the minimal interface required for any system to participate in
governed self-improvement experiments. Implementations must satisfy all methods.

Design principles:
- Minimal surface area (only experiment-required methods)
- Type-safe with clear contracts
- Compatible with ScoreBundle persistence layer
- No internal system details exposed
- Testable via mock implementations
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable
from stephanie.data.score_bundle import ScoreBundle

@runtime_checkable
class SystemInterface(Protocol):
    """
    Protocol defining required methods for ELM experimental integration.
    
    Any Stephanie engine variant must implement this interface to participate
    in calibrated self-improvement experiments.
    
    Usage:
        class MyStephanieEngine(SystemInterface):
            def evaluate(self, query: Any) -> ScoreBundle:
                # Implementation
                pass
            
            def attempt_improvement(
                self, 
                query: Any, 
                bundle_before: ScoreBundle
            ) -> Optional[Dict[str, Any]]:
                # Implementation
                pass
            
            def commit_improvement(
                self,
                query: Any,
                bundle_after: ScoreBundle,
                reflection_trace: Optional[Any] = None
            ) -> None:
                # Implementation
                pass
        
        # Validate implementation
        assert isinstance(my_engine, SystemInterface)
    """
    
    def evaluate(self, query: Any) -> ScoreBundle:
        """
        Evaluate query and return ScoreBundle.
        
        Must:
        - Return fully populated ScoreBundle with all critical dimensions
        - Include raw_energy in attributes for energy extraction
        - Be deterministic for identical queries (for reproducibility)
        
        Args:
            query: Input query (type system-specific)
            
        Returns:
            ScoreBundle containing evaluation results
            
        Raises:
            EvaluationError: If evaluation fails catastrophically
        """
        ...
    
    def attempt_improvement(
        self,
        query: Any,
        bundle_before: ScoreBundle
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt self-improvement via reflection/retry cycle.
        
        Must:
        - Return None if no improvement attempt made
        - Return dict with "bundle" key containing improved ScoreBundle
        - Optionally include "reflection" key with trace metadata
        - Only return bundle if dominance check would pass (pre-filter)
        
        Args:
            query: Original query
            bundle_before: ScoreBundle from initial evaluation
            
        Returns:
            Dict with keys:
                - "bundle": Improved ScoreBundle (required)
                - "reflection": Reflection trace metadata (optional)
            None if no improvement attempted
            
        Raises:
            ImprovementError: If improvement process fails
        """
        ...
    
    def commit_improvement(
        self,
        query: Any,
        bundle_after: ScoreBundle,
        reflection_trace: Optional[Any] = None
    ) -> None:
        """
        Persist validated improvement to system state.
        
        Must:
        - Update internal model state with improvement
        - Persist to database via existing ScoreBundle.save_bundle()
        - Include reflection_trace in metadata if provided
        - Be idempotent (safe to call multiple times)
        
        Args:
            query: Original query context
            bundle_after: Validated improved ScoreBundle
            reflection_trace: Optional reflection metadata for provenance
            
        Raises:
            CommitError: If persistence fails
        """
        ...
    
    def inject_misleading_evidence(self, probability: float = 0.4) -> None:
        """
        Inject controlled perturbation for stress testing.
        
        Used by PerturbationInjector during experiment.
        Must be reversible via restore_original_state().
        
        Args:
            probability: Likelihood of injecting misleading evidence [0.0, 1.0]
        """
        ...
    
    def increase_query_complexity(self, factor: float = 1.5) -> None:
        """
        Increase query complexity for stress testing.
        
        Used by PerturbationInjector during experiment.
        Must be reversible via restore_original_state().
        
        Args:
            factor: Complexity multiplier (>1.0 increases complexity)
        """
        ...
    
    def restore_original_state(self) -> None:
        """
        Restore system to pre-perturbation state.
        
        Must reverse all effects of:
        - inject_misleading_evidence()
        - increase_query_complexity()
        - temporarily_disable_governance()
        
        Called after perturbation episode completes.
        """
        ...
    
    def get_query_id(self, query: Any) -> Optional[str]:
        """
        Extract stable identifier from query.
        
        Used for experiment logging and reproducibility.
        Must return consistent ID for identical queries.
        
        Args:
            query: Input query
            
        Returns:
            String identifier or None if not available
        """
        ...
    
    @property
    def name(self) -> str:
        """System identifier for experiment logging (e.g., 'stephanie_v3_governed')"""
        ...
    
    @property
    def version(self) -> str:
        """Semantic version string (e.g., '2.1.0')"""
        ...


# ============================================================================
# ERROR TYPES FOR INTERFACE CONTRACT
# ============================================================================

class EvaluationError(Exception):
    """Raised when query evaluation fails catastrophically"""
    pass

class ImprovementError(Exception):
    """Raised when improvement attempt fails"""
    pass

class CommitError(Exception):
    """Raised when improvement persistence fails"""
    pass


# ============================================================================
# MOCK IMPLEMENTATION FOR TESTING
# ============================================================================

class MockSystem(SystemInterface):
    """
    Minimal mock implementation for testing experiment harness.
    
    Usage:
        mock = MockSystem()
        experiment = ScoreBundleExperiment(system=mock, ...)
        result = experiment.run()  # Validates harness logic
    """
    
    def __init__(self, seed: int = 42):
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self._original_state = {}
        self._perturbed = False
    
    def evaluate(self, query: Any) -> ScoreBundle:
        from stephanie.data.score_result import ScoreResult
        
        # Simulate realistic scores with controlled variance
        base_energy = 0.25 + self.rng.normal(0, 0.05)
        base_hrm = 0.82 + self.rng.normal(0, 0.04)
        base_margin = 0.65 + self.rng.normal(0, 0.06)
        
        results = {
            "alignment": ScoreResult(
                dimension="alignment",
                score=base_hrm * 100,
                source="mock_hrm",
                rationale="Simulated HRM evaluation",
                weight=1.0,
                attributes={"raw_energy": base_energy * 100}
            ),
            "energy": ScoreResult(
                dimension="energy",
                score=(1.0 - base_energy) * 100,  # Inverted for lower=better
                source="mock_ebt",
                rationale="Simulated energy evaluation",
                weight=1.0,
                attributes={"raw_energy": base_energy * 100}
            ),
            "margin": ScoreResult(
                dimension="margin",
                score=base_margin * 100,
                source="mock_embedding",
                rationale="Simulated margin evaluation",
                weight=1.0,
                attributes={"embedding_margin": base_margin}
            )
        }
        
        return ScoreBundle(results=results)
    
    def attempt_improvement(
        self,
        query: Any,
        bundle_before: ScoreBundle
    ) -> Optional[Dict[str, Any]]:
        # Simulate 70% improvement success rate
        if self.rng.random() < 0.7:
            improved_bundle = self.evaluate(query)
            return {"bundle": improved_bundle, "reflection": {"applied": True}}
        return None
    
    def commit_improvement(
        self,
        query: Any,
        bundle_after: ScoreBundle,
        reflection_trace: Optional[Any] = None
    ) -> None:
        # No-op for mock (persistence simulated)
        pass
    
    def inject_misleading_evidence(self, probability: float = 0.4) -> None:
        self._perturbed = True
    
    def increase_query_complexity(self, factor: float = 1.5) -> None:
        self._perturbed = True
    
    def restore_original_state(self) -> None:
        self._perturbed = False
    
    def get_query_id(self, query: Any) -> Optional[str]:
        return str(hash(str(query)))[:16]
    
    @property
    def name(self) -> str:
        return "mock_stephanie_test_system"
    
    @property
    def version(self) -> str:
        return "test-1.0"


# ============================================================================
# VALIDATION UTILITY
# ============================================================================

def validate_system_implementation(system: Any) -> bool:
    """
    Validate that system satisfies SystemInterface contract.
    
    Usage:
        assert validate_system_implementation(my_engine)
    
    Checks:
    - All required methods exist
    - Properties are accessible
    - Minimal type compatibility
    
    Returns:
        True if valid, raises AssertionError otherwise
    """
    import inspect
    
    # Check protocol compliance
    assert isinstance(system, SystemInterface), \
        "System must implement SystemInterface protocol"
    
    # Check required methods exist
    required_methods = [
        'evaluate', 'attempt_improvement', 'commit_improvement',
        'inject_misleading_evidence', 'increase_query_complexity',
        'restore_original_state', 'get_query_id'
    ]
    for method in required_methods:
        assert hasattr(system, method), \
            f"System missing required method: {method}"
        assert callable(getattr(system, method)), \
            f"System.{method} must be callable"
    
    # Check required properties exist
    required_props = ['name', 'version']
    for prop in required_props:
        assert hasattr(system, prop), \
            f"System missing required property: {prop}"
    
    # Check method signatures (basic)
    sig = inspect.signature(system.evaluate)
    assert 'query' in sig.parameters, \
        "evaluate() must accept 'query' parameter"
    
    sig = inspect.signature(system.attempt_improvement)
    assert 'query' in sig.parameters and 'bundle_before' in sig.parameters, \
        "attempt_improvement() must accept 'query' and 'bundle_before'"
    
    sig = inspect.signature(system.commit_improvement)
    assert 'query' in sig.parameters and 'bundle_after' in sig.parameters, \
        "commit_improvement() must accept 'query' and 'bundle_after'"
    
    return True


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "SystemInterface",
    "EvaluationError",
    "ImprovementError",
    "CommitError",
    "MockSystem",
    "validate_system_implementation"
]

==================================================
FILE: plugin\adaptive_improvement_plugin.py
==================================================

class AdaptiveImprovementPlugin:

    def __init__(self, evaluator, comparator, governor, reflection_engine, applier):
        self.evaluator = evaluator
        self.comparator = comparator
        self.governor = governor
        self.reflection_engine = reflection_engine
        self.applier = applier

    def improve(self, context, trace, output, model):

        before_bundle = self.evaluator.evaluate(context, trace, output)

        reflection_trace = self.reflection_engine.generate_reflection(before_bundle)

        improved_output = self.applier.apply_reflection(
            original_output=output,
            reflection=reflection_trace,
            model=model,
            context_pack=context,
        )

        after_bundle = self.evaluator.evaluate(context, trace, improved_output)

        if not self.governor.should_accept_update(before_bundle, after_bundle):
            return output, before_bundle

        if self.comparator.dominates(
            before_bundle,
            after_bundle,
            critical_axes=["hallucination_energy", "hrm_alignment"],
        ):
            return improved_output, after_bundle

        return output, before_bundle


==================================================
FILE: plugin\__init__.py
==================================================



==================================================
FILE: policy\adaptive_policy.py
==================================================

# AdaptivePolicy implementation


==================================================
FILE: policy\policy_container.py
==================================================

# PolicyContainer implementation


==================================================
FILE: policy\regime_controller.py
==================================================

# RegimeController implementation


==================================================
FILE: policy\__init__.py
==================================================



==================================================
FILE: providers\base.py
==================================================

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Any

from ..axes import RewardAxis


@dataclass
class SignalResult:
    axis_values: Dict[RewardAxis, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    failure_signatures: List[str] = field(default_factory=list)
    confidence: float = 1.0


class SignalProvider(Protocol):
    def compute(
        self,
        context_pack: Any,
        plan_trace: Any,
        output: Any,
        **kwargs
    ) -> SignalResult:
        ...


==================================================
FILE: providers\certum_provider.py
==================================================

from typing import Any
from .base import SignalProvider, SignalResult
from ..axes import RewardAxis


class CertumProvider(SignalProvider):
    def __init__(self, energy_model: Any):
        self.model = energy_model

    def compute(self, context_pack: Any, plan_trace: Any, output: Any, **kwargs) -> SignalResult:
        energy = float(self.model.compute_energy(output))
        failures = ["energy_spike"] if energy > 0.5 else []

        return SignalResult(
            axis_values={RewardAxis.HALLUCINATION_ENERGY: energy},
            diagnostics={"energy_raw": energy},
            failure_signatures=failures,
            confidence=1.0,
        )


==================================================
FILE: providers\embedding_provider.py
==================================================

import torch
from typing import Any
from .base import SignalProvider, SignalResult
from ..axes import RewardAxis


class EmbeddingProvider(SignalProvider):
    def __init__(self, embedder: Any):
        self.embedder = embedder

    def compute(self, context_pack: Any, plan_trace: Any, output: Any, **kwargs) -> SignalResult:
        goal_emb = kwargs.get("goal_embedding")
        output_emb = self.embedder.encode(output)

        margin = torch.nn.functional.cosine_similarity(output_emb, goal_emb, dim=0).item()

        return SignalResult(
            axis_values={RewardAxis.EMBEDDING_MARGIN: margin},
            diagnostics={"margin": margin},
            confidence=0.95,
        )


==================================================
FILE: providers\embedding_ptovider.py
==================================================



==================================================
FILE: providers\hrm_provider.py
==================================================

from typing import Any
from .base import SignalProvider, SignalResult
from ..axes import RewardAxis


class HRMProvider(SignalProvider):
    def __init__(self, hrm_model: Any):
        self.model = hrm_model

    def compute(self, context_pack: Any, plan_trace: Any, output: Any, **kwargs) -> SignalResult:
        score = float(self.model.score(output))

        return SignalResult(
            axis_values={RewardAxis.HRM_ALIGNMENT: score},
            diagnostics={"hrm_raw": score},
            confidence=0.9,
        )


==================================================
FILE: providers\policy_provider.py
==================================================

# PolicyProvider implementation


==================================================
FILE: providers\__init__.py
==================================================



==================================================
FILE: reflection\reflection_application.py
==================================================

from stephanie.components.elm.reflection.reflection_trace import ReflectionTrace


class ReflectionApplier:

    def apply_reflection(
        self,
        original_output: str,
        reflection: ReflectionTrace,
        model,
        context_pack,
    ) -> str:
        """
        Applies structured reflection by re-running model
        with corrective constraints.
        """

        if not reflection.failed_axes:
            return original_output

        correction_prompt = (
            "You previously produced the following output:\n\n"
            f"{original_output}\n\n"
            "It had the following issues:\n"
        )

        for axis in reflection.failed_axes:
            correction_prompt += f"- {axis}\n"

        correction_prompt += "\nPlease revise the output to correct these issues.\n"

        return model.generate(
            context=context_pack,
            additional_constraints=correction_prompt,
        )


==================================================
FILE: reflection\reflection_engine.py
==================================================

# components/elm/reflection/reflection_engine.py

from typing import List
from stephanie.data.score_bundle import ScoreBundle
from components.elm.reflection.reflection_trace import ReflectionTrace


class ReflectionEngine:

    def __init__(self, energy_threshold: float = 55.0, hrm_threshold: float = 60.0):
        self.energy_threshold = energy_threshold
        self.hrm_threshold = hrm_threshold

    def generate_reflection(self, bundle: ScoreBundle) -> ReflectionTrace:

        failed_axes: List[str] = []
        instructions = {}
        focus = []

        # --- Hallucination Energy ---
        energy = bundle.get("hallucination_energy")
        if energy and energy.score > self.energy_threshold:
            failed_axes.append("hallucination_energy")
            focus.append("hallucination_energy")
            instructions["grounding"] = (
                "Re-evaluate claims against retrieved context. "
                "Remove speculative statements not supported by evidence."
            )

        # --- HRM Alignment ---
        hrm = bundle.get("hrm_alignment")
        if hrm and hrm.score < self.hrm_threshold:
            failed_axes.append("hrm_alignment")
            focus.append("hrm_alignment")
            instructions["reasoning"] = (
                "Clarify logical steps. Make reasoning explicit. "
                "Avoid implicit assumptions."
            )

        # --- Embedding Margin ---
        margin = bundle.get("embedding_margin")
        if margin and margin.score < 50.0:
            failed_axes.append("embedding_margin")
            focus.append("embedding_margin")
            instructions["alignment"] = (
                "Align output terminology with retrieved context anchors."
            )

        confidence = 1.0 if failed_axes else 0.0

        return ReflectionTrace(
            original_trace_id=bundle.meta.get("trace_id", "unknown"),
            failed_axes=failed_axes,
            correction_instructions=instructions,
            focus_axes=focus,
            confidence=confidence,
        )


==================================================
FILE: reflection\reflection_trace.py
==================================================

# components/elm/reflection/reflection_trace.py

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ReflectionTrace:
    original_trace_id: str
    failed_axes: List[str]
    correction_instructions: Dict[str, str]
    focus_axes: List[str]
    confidence: float


==================================================
FILE: reflection\__init__.py
==================================================



==================================================
FILE: stability\geometry_stability.py
==================================================

# geometry_governor.py

from stephanie.data.score_bundle import ScoreBundle


class GeometryStabilityGovernor:

    def should_accept_update(
        self,
        before: ScoreBundle,
        after: ScoreBundle,
    ) -> bool:

        energy_before = before.get("hallucination_energy")
        energy_after = after.get("hallucination_energy")

        if energy_before and energy_after:
            if energy_after.score > 55.0:  # critical threshold
                return False

        # embedding variance check
        var = after.get("embedding_variance")
        if var and var.score < 30.0:
            return False

        return True


==================================================
FILE: stability\retention_metrics.py
==================================================

# Retention metric formalization


==================================================
FILE: stability\__init__.py
==================================================



==================================================
FILE: tracking\collapse_detector.py
==================================================

"""
CollapseDetector: Real-time detection of representation collapse and instability.

Monitors 6 critical failure modes:
1. Energy spiral (hallucination instability)
2. HRM alignment collapse (reasoning quality degradation)
3. Embedding margin collapse (geometric failure)
4. Variance collapse (manifold degeneracy)
5. Collapse index explosion (eigenvalue distortion)
6. Angular drift violation (update instability)

All thresholds derived from CalibratedThresholds.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from ..core.thresholds import CalibratedThresholds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FailureEvent:
    """Structured failure event with diagnostic details"""
    episode: int
    failure_type: str
    metric_name: str
    metric_value: float
    threshold_value: float
    severity: str  # "warning", "critical"
    description: str
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "episode": self.episode,
            "failure_type": self.failure_type,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "severity": self.severity,
            "description": self.description,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.failure_type} | "
            f"{self.metric_name}={self.metric_value:.4f} "
            f"(threshold={self.threshold_value:.4f}) | "
            f"{self.description}"
        )


class CollapseDetector:
    """
    Real-time collapse detection system.
    
    Monitors governance metrics against calibrated thresholds.
    Detects 6 critical failure modes with severity levels.
    Maintains failure history for diagnostics.
    
    Usage:
        detector = CollapseDetector(thresholds)
        
        # Check metrics every episode
        failure = detector.check_failure(
            episode=42,
            metrics={
                "energy_raw": 0.52,
                "hrm_alignment": 0.68,
                "embedding_margin": 0.41,
                "embedding_variance": 0.28,
                "collapse_index": 12.3,
                "angular_drift": 0.18
            }
        )
        
        if failure:
            logger.warning(f"COLLAPSE DETECTED: {failure}")
            if failure.severity == "critical":
                system.freeze_embedding_updates()
    """
    
    # Failure type constants
    ENERGY_SPIRAL = "energy_spiral"
    HRM_COLLAPSE = "hrm_collapse"
    MARGIN_COLLAPSE = "margin_collapse"
    VARIANCE_COLLAPSE = "variance_collapse"
    COLLAPSE_INDEX_EXPLOSION = "collapse_index_explosion"
    ANGULAR_DRIFT_VIOLATION = "angular_drift_violation"
    
    def __init__(
        self,
        thresholds: "CalibratedThresholds",  # Forward reference
        consecutive_failures_required: int = 1,
        warning_buffer: float = 0.05  # 5% buffer below critical threshold for warnings
    ):
        """
        Initialize collapse detector.
        
        Args:
            thresholds: Calibrated safety thresholds
            consecutive_failures_required: Failures must occur this many times consecutively to trigger
            warning_buffer: Fraction below critical threshold where warnings activate
        """
        self.thresholds = thresholds
        self.consecutive_failures_required = consecutive_failures_required
        self.warning_buffer = warning_buffer
        
        # Failure tracking state
        self._failure_streaks: Dict[str, int] = {
            self.ENERGY_SPIRAL: 0,
            self.HRM_COLLAPSE: 0,
            self.MARGIN_COLLAPSE: 0,
            self.VARIANCE_COLLAPSE: 0,
            self.COLLAPSE_INDEX_EXPLOSION: 0,
            self.ANGULAR_DRIFT_VIOLATION: 0
        }
        
        # Failure history (last 100 events)
        self._failure_history: List[FailureEvent] = []
        self._max_history = 100
        
        logger.info(
            f"CollapseDetector initialized | "
            f"thresholds={thresholds} | "
            f"consecutive_required={consecutive_failures_required}"
        )
    
    def check_failure(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """
        Check current metrics against thresholds.
        
        Returns:
            FailureEvent if critical failure detected, None otherwise
        """
        # Check each failure mode
        checks = [
            self._check_energy_spiral(episode, metrics),
            self._check_hrm_collapse(episode, metrics),
            self._check_margin_collapse(episode, metrics),
            self._check_variance_collapse(episode, metrics),
            self._check_collapse_index_explosion(episode, metrics),
            self._check_angular_drift_violation(episode, metrics)
        ]
        
        # Return first critical failure (if any)
        for failure in checks:
            if failure and failure.severity == "critical":
                self._record_failure(failure)
                return failure
        
        # No critical failures
        self._reset_streaks()
        return None
    
    def _check_energy_spiral(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for hallucination energy spiral"""
        energy = metrics.get("energy_raw", 0.0)
        
        # Critical: exceeds absolute max
        if energy > self.thresholds.energy_max:
            return FailureEvent(
                episode=episode,
                failure_type=self.ENERGY_SPIRAL,
                metric_name="energy_raw",
                metric_value=energy,
                threshold_value=self.thresholds.energy_max,
                severity="critical",
                description=f"Energy exceeded critical threshold ({energy:.3f} > {self.thresholds.energy_max:.3f})"
            )
        
        # Warning: exceeds warning threshold
        if energy > self.thresholds.energy_warning:
            return FailureEvent(
                episode=episode,
                failure_type=self.ENERGY_SPIRAL,
                metric_name="energy_raw",
                metric_value=energy,
                threshold_value=self.thresholds.energy_warning,
                severity="warning",
                description=f"Energy in warning zone ({energy:.3f} > {self.thresholds.energy_warning:.3f})"
            )
        
        return None
    
    def _check_hrm_collapse(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for HRM alignment collapse"""
        hrm = metrics.get("hrm_alignment", 1.0)
        
        if hrm < self.thresholds.hrm_min:
            return FailureEvent(
                episode=episode,
                failure_type=self.HRM_COLLAPSE,
                metric_name="hrm_alignment",
                metric_value=hrm,
                threshold_value=self.thresholds.hrm_min,
                severity="critical",
                description=f"HRM alignment collapsed ({hrm:.3f} < {self.thresholds.hrm_min:.3f})"
            )
        
        return None
    
    def _check_margin_collapse(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for embedding margin collapse"""
        margin = metrics.get("embedding_margin", 1.0)
        
        if margin < self.thresholds.margin_min:
            return FailureEvent(
                episode=episode,
                failure_type=self.MARGIN_COLLAPSE,
                metric_name="embedding_margin",
                metric_value=margin,
                threshold_value=self.thresholds.margin_min,
                severity="critical",
                description=f"Embedding margin collapsed ({margin:.3f} < {self.thresholds.margin_min:.3f})"
            )
        
        return None
    
    def _check_variance_collapse(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for embedding variance collapse"""
        variance = metrics.get("embedding_variance", 1.0)
        
        if variance < self.thresholds.variance_min:
            return FailureEvent(
                episode=episode,
                failure_type=self.VARIANCE_COLLAPSE,
                metric_name="embedding_variance",
                metric_value=variance,
                threshold_value=self.thresholds.variance_min,
                severity="critical",
                description=f"Embedding variance collapsed ({variance:.3f} < {self.thresholds.variance_min:.3f})"
            )
        
        return None
    
    def _check_collapse_index_explosion(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for collapse index explosion (manifold distortion)"""
        collapse_index = metrics.get("collapse_index", 1.0)
        
        if collapse_index > self.thresholds.collapse_index_max:
            return FailureEvent(
                episode=episode,
                failure_type=self.COLLAPSE_INDEX_EXPLOSION,
                metric_name="collapse_index",
                metric_value=collapse_index,
                threshold_value=self.thresholds.collapse_index_max,
                severity="critical",
                description=f"Collapse index exploded ({collapse_index:.2f} > {self.thresholds.collapse_index_max:.2f})"
            )
        
        return None
    
    def _check_angular_drift_violation(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for excessive angular drift in embedding updates"""
        drift = metrics.get("angular_drift", 0.0)
        
        if drift > self.thresholds.drift_max:
            return FailureEvent(
                episode=episode,
                failure_type=self.ANGULAR_DRIFT_VIOLATION,
                metric_name="angular_drift",
                metric_value=drift,
                threshold_value=self.thresholds.drift_max,
                severity="critical",
                description=f"Angular drift exceeded limit ({drift:.3f} rad > {self.thresholds.drift_max:.3f} rad)"
            )
        
        return None
    
    def _record_failure(self, failure: FailureEvent) -> None:
        """Record failure event and update streaks"""
        # Update streak for this failure type
        self._failure_streaks[failure.failure_type] += 1
        
        # Add to history
        self._failure_history.append(failure)
        if len(self._failure_history) > self._max_history:
            self._failure_history.pop(0)
        
        # Log failure
        log_fn = logger.warning if failure.severity == "warning" else logger.error
        log_fn(f"COLLAPSE DETECTOR: {failure}")
    
    def _reset_streaks(self) -> None:
        """Reset all failure streaks (called when no failures detected)"""
        for key in self._failure_streaks:
            self._failure_streaks[key] = 0
    
    def get_failure_history(self, last_n: int = 10) -> List[FailureEvent]:
        """Get recent failure events"""
        return self._failure_history[-last_n:] if self._failure_history else []
    
    def get_streak(self, failure_type: str) -> int:
        """Get current streak count for failure type"""
        return self._failure_streaks.get(failure_type, 0)
    
    def generate_diagnostic_report(self) -> str:
        """
        Generate diagnostic report of recent failures.
        
        Returns:
            Formatted string with failure analysis
        """
        lines = [
            "=" * 70,
            "COLLAPSE DETECTOR DIAGNOSTIC REPORT",
            "=" * 70,
            "Current Thresholds:",
            f"  Energy Max: {self.thresholds.energy_max:.3f}",
            f"  HRM Min: {self.thresholds.hrm_min:.3f}",
            f"  Margin Min: {self.thresholds.margin_min:.3f}",
            f"  Variance Min: {self.thresholds.variance_min:.3f}",
            f"  Collapse Index Max: {self.thresholds.collapse_index_max:.2f}",
            f"  Drift Max: {self.thresholds.drift_max:.3f} rad",
            "",
            "Failure Streaks (consecutive episodes):"
        ]
        
        for failure_type, streak in self._failure_streaks.items():
            if streak > 0:
                lines.append(f"  {failure_type}: {streak} episodes")
        
        if not any(self._failure_streaks.values()):
            lines.append("  None (system stable)")
        
        lines.append("")
        lines.append("Recent Failures (last 5):")
        
        if self._failure_history:
            for failure in self._failure_history[-5:]:
                lines.append(f"  {failure}")
        else:
            lines.append("  None")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def __str__(self) -> str:
        active = sum(1 for s in self._failure_streaks.values() if s > 0)
        return f"CollapseDetector(active_failures={active})"
    
    def __repr__(self) -> str:
        return self.__str__()

==================================================
FILE: tracking\retention_tracker.py
==================================================

"""
RetentionTracker: Measures persistence of improvements across time horizons.

Critical for distinguishing:
- Short-term reward spikes (unstable)
- Long-term durable improvements (valuable)

Tracks retention per axis with direction-aware delta computation.
Uses exponential moving average for stable scoring.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetentionMetrics:
    """Structured retention metrics per axis and horizon"""
    axis: str
    horizon: int
    retention_score: float
    sample_count: int
    recent_deltas: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "axis": self.axis,
            "horizon": self.horizon,
            "retention_score": self.retention_score,
            "sample_count": self.sample_count,
            "recent_deltas_mean": np.mean(self.recent_deltas) if self.recent_deltas else 0.0
        }


class RetentionTracker:
    """
    Tracks long-term improvement retention across multiple horizons.
    
    Design principles:
    - Direction-aware delta computation (higher/lower is better)
    - Exponential moving average for stable scoring
    - Per-axis and per-horizon tracking
    - Statistical significance tracking (sample count)
    - Memory efficient (bounded history windows)
    
    Usage:
        tracker = RetentionTracker(
            critical_axes=["energy_raw", "hrm_alignment", "embedding_margin"],
            horizons=[5, 10, 20],
            alpha=0.1  # EMA smoothing factor
        )
        
        # Update every episode
        tracker.update(episode=42, metrics={
            "energy_raw": 0.28,
            "hrm_alignment": 0.85,
            "embedding_margin": 0.62
        })
        
        # Get current retention scores
        scores = tracker.get_scores()  # {"energy_raw": -0.02, "hrm_alignment": 0.03, ...}
        report = tracker.generate_report()  # Human-readable analysis
    """
    
    def __init__(
        self,
        critical_axes: List[str] = None,
        horizons: List[int] = None,
        alpha: float = 0.1,
        min_samples: int = 10
    ):
        """
        Initialize retention tracker.
        
        Args:
            critical_axes: List of axis names to track (must match metrics keys)
            horizons: Time horizons to evaluate retention (episodes)
            alpha: EMA smoothing factor (0.0 = no smoothing, 1.0 = only latest)
            min_samples: Minimum samples before reporting retention
        """
        self.critical_axes = critical_axes or [
            "energy_raw",        # Lower is better
            "hrm_alignment",     # Higher is better
            "embedding_margin"   # Higher is better
        ]
        self.horizons = sorted(horizons or [5, 10, 20])
        self.alpha = alpha
        self.min_samples = min_samples
        
        # Axis direction semantics (critical for delta computation)
        self.axis_directions: Dict[str, str] = {
            "energy_raw": "lower",      # Lower energy = better
            "hrm_alignment": "higher",  # Higher alignment = better
            "embedding_margin": "higher" # Higher margin = better
        }
        
        # History buffers: {axis: deque of (episode, value)}
        self.history: Dict[str, deque] = {
            axis: deque(maxlen=max(self.horizons) + 1)
            for axis in self.critical_axes
        }
        
        # Retention scores: {axis: {horizon: score}}
        self.retention_scores: Dict[str, Dict[int, float]] = {
            axis: {h: 0.0 for h in self.horizons}
            for axis in self.critical_axes
        }
        
        # Sample counts for statistical significance
        self.sample_counts: Dict[str, Dict[int, int]] = {
            axis: {h: 0 for h in self.horizons}
            for axis in self.critical_axes
        }
        
        # Recent deltas for diagnostics
        self.recent_deltas: Dict[str, Dict[int, deque]] = {
            axis: {h: deque(maxlen=50) for h in self.horizons}
            for axis in self.critical_axes
        }
        
        logger.info(
            f"RetentionTracker initialized | "
            f"axes={self.critical_axes} | "
            f"horizons={self.horizons} | "
            f"alpha={alpha}"
        )
    
    def update(self, episode: int, metrics: Dict[str, float]) -> None:
        """
        Update tracker with new episode metrics.
        
        Computes retention deltas for all horizons where sufficient history exists.
        Updates EMA retention scores.
        
        Args:
            episode: Current episode number
            metrics: Dict of metric values (must include critical axes)
        """
        # Store current values in history
        for axis in self.critical_axes:
            if axis in metrics:
                self.history[axis].append((episode, metrics[axis]))
        
        # Compute retention for each axis and horizon
        for axis in self.critical_axes:
            if len(self.history[axis]) < max(self.horizons) + 1:
                continue  # Not enough history yet
            
            current_value = metrics.get(axis)
            if current_value is None:
                continue
            
            # Compute retention for each horizon
            for horizon in self.horizons:
                if len(self.history[axis]) <= horizon:
                    continue
                
                # Get value from horizon episodes ago
                past_episode, past_value = self.history[axis][-horizon - 1]
                
                # Compute direction-aware delta (positive = improvement)
                delta = self._compute_delta(axis, current_value, past_value)
                
                # Update EMA retention score
                old_score = self.retention_scores[axis][horizon]
                new_score = self.alpha * delta + (1 - self.alpha) * old_score
                self.retention_scores[axis][horizon] = new_score
                
                # Update sample count
                self.sample_counts[axis][horizon] += 1
                
                # Store recent delta for diagnostics
                self.recent_deltas[axis][horizon].append(delta)
    
    def _compute_delta(self, axis: str, current: float, past: float) -> float:
        """
        Compute direction-aware improvement delta.
        
        Positive delta always means improvement, regardless of axis direction.
        
        Args:
            axis: Axis name
            current: Current value
            past: Value from horizon episodes ago
            
        Returns:
            Delta where positive = improvement
        """
        direction = self.axis_directions.get(axis, "higher")
        
        if direction == "lower":  # Lower is better (e.g., energy)
            # Improvement = decrease in value
            return past - current
        else:  # Higher is better (e.g., HRM alignment)
            # Improvement = increase in value
            return current - past
    
    def get_scores(self, horizon: Optional[int] = None) -> Dict[str, float]:
        """
        Get current retention scores.
        
        Args:
            horizon: Specific horizon to report (default: longest horizon)
            
        Returns:
            Dict of {axis: retention_score} for specified horizon
        """
        target_horizon = horizon if horizon is not None else max(self.horizons)
        
        return {
            axis: self.retention_scores[axis][target_horizon]
            for axis in self.critical_axes
            if self.sample_counts[axis][target_horizon] >= self.min_samples
        }
    
    def get_all_scores(self) -> Dict[str, Dict[int, float]]:
        """Get retention scores for all axes and all horizons"""
        return {
            axis: {
                h: score for h, score in horizons.items()
                if self.sample_counts[axis][h] >= self.min_samples
            }
            for axis, horizons in self.retention_scores.items()
        }
    
    def is_positive_retention(self, axis: str, horizon: Optional[int] = None) -> bool:
        """
        Check if retention is positive for given axis.
        
        Args:
            axis: Axis name
            horizon: Horizon to check (default: longest)
            
        Returns:
            True if retention score > 0 with sufficient samples
        """
        target_horizon = horizon if horizon is not None else max(self.horizons)
        
        if self.sample_counts[axis][target_horizon] < self.min_samples:
            return False
        
        return self.retention_scores[axis][target_horizon] > 0
    
    def get_metrics(self, axis: str, horizon: int) -> Optional[RetentionMetrics]:
        """Get detailed metrics for specific axis and horizon"""
        if axis not in self.critical_axes or horizon not in self.horizons:
            return None
        
        if self.sample_counts[axis][horizon] < self.min_samples:
            return None
        
        return RetentionMetrics(
            axis=axis,
            horizon=horizon,
            retention_score=self.retention_scores[axis][horizon],
            sample_count=self.sample_counts[axis][horizon],
            recent_deltas=list(self.recent_deltas[axis][horizon])
        )
    
    def generate_report(self) -> str:
        """
        Generate human-readable retention report.
        
        Returns:
            Formatted string with retention analysis
        """
        lines = [
            "=" * 70,
            "RETENTION TRACKER REPORT",
            "=" * 70,
            f"Critical Axes: {', '.join(self.critical_axes)}",
            f"Horizons Tracked: {self.horizons}",
            f"EMA Smoothing (alpha): {self.alpha}",
            f"Min Samples for Reporting: {self.min_samples}",
            "",
            "┌──────────────────────────────────────────────────────────────────┐",
            "│ RETENTION SCORES (Positive = Durable Improvement)                │",
            "├──────────────────┬──────────┬──────────┬──────────┬──────────────┤",
            "│ Axis             │ Horizon  │ Score    │ Samples  │ Status       │",
            "├──────────────────┼──────────┼──────────┼──────────┼──────────────┤",
        ]
        
        for axis in self.critical_axes:
            for horizon in self.horizons:
                score = self.retention_scores[axis][horizon]
                samples = self.sample_counts[axis][horizon]
                
                if samples < self.min_samples:
                    status = "INSUFFICIENT_DATA"
                    score_str = "N/A"
                elif score > 0.01:
                    status = "✅ IMPROVING"
                    score_str = f"{score:+.4f}"
                elif score < -0.01:
                    status = "⚠️  DEGRADING"
                    score_str = f"{score:+.4f}"
                else:
                    status = "→ STABLE"
                    score_str = f"{score:+.4f}"
                
                samples_str = f"{samples}/{self.min_samples}" if samples < self.min_samples else str(samples)
                
                lines.append(
                    f"│ {axis:<16} │ {horizon:8} │ {score_str:>8} │ {samples_str:>8} │ {status:<12} │"
                )
            lines.append("├──────────────────┼──────────┼──────────┼──────────┼──────────────┤")
        
        # Summary statistics
        lines.append("└──────────────────┴──────────┴──────────┴──────────┴──────────────┘")
        lines.append("")
        lines.append("SUMMARY:")
        
        all_positive = True
        for axis in self.critical_axes:
            if not self.is_positive_retention(axis):
                all_positive = False
                lines.append(f"  ⚠️  {axis}: Negative retention (system degrading)")
        
        if all_positive:
            lines.append("  ✅ All critical axes show positive retention")
        else:
            lines.append("  ⚠️  Some axes show negative retention - investigate")
        
        lines.append("")
        lines.append("Retention scores represent exponential moving average of")
        lines.append("improvement deltas over specified horizons.")
        lines.append("Positive score = durable improvement persists over time.")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, any]:
        """Serialize tracker state for persistence"""
        return {
            "critical_axes": self.critical_axes,
            "horizons": self.horizons,
            "alpha": self.alpha,
            "min_samples": self.min_samples,
            "axis_directions": self.axis_directions,
            "retention_scores": self.retention_scores,
            "sample_counts": self.sample_counts,
            # Note: history and recent_deltas not serialized (reconstructed from episodes)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "RetentionTracker":
        """Reconstruct tracker from serialized state"""
        tracker = cls(
            critical_axes=data["critical_axes"],
            horizons=data["horizons"],
            alpha=data["alpha"],
            min_samples=data["min_samples"]
        )
        
        # Restore state
        tracker.axis_directions = data.get("axis_directions", tracker.axis_directions)
        tracker.retention_scores = data["retention_scores"]
        tracker.sample_counts = data["sample_counts"]
        
        return tracker
    
    def __str__(self) -> str:
        scores = self.get_scores()
        summary = ", ".join(f"{axis}:{score:+.3f}" for axis, score in scores.items())
        return f"RetentionTracker({summary})"
    
    def __repr__(self) -> str:
        return self.__str__()

==================================================
FILE: utils\metrics.py
==================================================

# Utility metrics helpers


==================================================
FILE: utils\__init__.py
==================================================


