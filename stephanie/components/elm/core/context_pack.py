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