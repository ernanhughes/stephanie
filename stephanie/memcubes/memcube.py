# stephanie/memcubes/memcube.py

from datetime import datetime
from enum import Enum
from stephanie.scoring.scorable import Scorable
from stephanie.utils.file_utils import hash_text


class MemCubeType(Enum):
    DOCUMENT = "document"
    PROMPT = "prompt"
    RESPONSE = "response"
    HYPOTHESIS = "hypothesis"
    SYMBOL = "symbol"
    THEOREM = "theorem"
    TRIPLE = "triple"
    CARTRIDGE = "cartridge"
    REFINEMENT = "refinement"  # New for SRFT-style usage


class MemCube:
    def __init__(
        self,
        scorable: Scorable,
        dimension: str = None,
        created_at: datetime = None,
        last_modified: datetime = None,
        source: str = "user_input",
        model: str = "llama3",
        access_policy: dict = None,
        priority: int = 5,
        orriginal_score: float = None,
        refined_score: float = None,
        refined_content: str = None,    
        sensitivity: str = "public",
        ttl: int = None,
        version: str = "v1",
        usage_count: int = 0,
        extra_data: dict = None
    ):
        """
        MemCube wraps Scorable with versioning, governance, and lifecycle metadata.
        """
        self.scorable = scorable
        self.source = source
        self.dimension = dimension
        self.model = model
        self.priority = priority
        self.sensitivity = sensitivity
        self.original_score = orriginal_score
        self.refined_score = refined_score  
        self.refined_content = refined_content  
        self.ttl = ttl
        self.version = version
        self.created_at = created_at or datetime.utcnow()
        self.last_modified = last_modified or self.created_at
        self.usage_count = usage_count
        self.extra_data = extra_data or {}

        self.id = self._generate_human_id()
        self._validate_sensitivity()

    def _validate_sensitivity(self):
        valid_tags = ["public", "internal", "confidential", "restricted"]
        if self.sensitivity not in valid_tags:
            raise ValueError(f"Sensitivity must be one of {valid_tags}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scorable": self.scorable.to_dict(),
            "dimension": self.dimension,
            "version": self.version,
            "refined_content": self.refined_content,
            "original_score": self.original_score,
            "refined_score": self.refined_score,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "source": self.source,
            "model": self.model,
            "priority": self.priority,
            "sensitivity": self.sensitivity,
            "ttl": self.ttl,
            "usage_count": self.usage_count,
            "extra_data": self.extra_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemCube":
        from stephanie.scoring.scorable_factory import ScorableFactory
        scorable = ScorableFactory.from_dict(data["scorable"])
        return cls(
            scorable=scorable,
            version=data.get("version", "v1"),
            created_at=datetime.fromisoformat(data["created_at"])
                if "created_at" in data else None,
            last_modified=datetime.fromisoformat(data["last_modified"])
                if "last_modified" in data else None,
            source=data.get("source", "user_input"),
            model=data.get("model", "llama3"),
            priority=data.get("priority", 5),
            sensitivity=data.get("sensitivity", "public"),
            ttl=data.get("ttl"),
            usage_count=data.get("usage_count", 0),
            extra_data=data.get("extra_data", {})
        )

    def increment_usage(self):
        self.usage_count += 1
        self.last_modified = datetime.utcnow()
        return self.usage_count

    def has_expired(self) -> bool:
        if not self.ttl:
            return False
        from datetime import timedelta
        return (datetime.utcnow() - self.created_at) > timedelta(days=self.ttl)

    def apply_governance(self, user: str, action: str) -> bool:
        policy = self.extra_data.get("access_policy", {})
        allowed_roles = policy.get(action, ["admin", "researcher"])
        user_role = self._get_user_role(user)
        return user_role in allowed_roles

    def _get_user_role(self, user: str) -> str:
        return "researcher"  # Placeholder for real user role logic


    def _generate_human_id(self):
        ts = self.created_at.strftime("%Y%m%d%H%M%S")
        target_type = self.scorable.target_type
        scorable_id = str(self.scorable.id)
        return f"{ts}_{target_type}_{scorable_id}_{self.version}"