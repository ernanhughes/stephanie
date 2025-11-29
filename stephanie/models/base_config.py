from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, Mapping, Type, TypeVar, cast

# you already have something like:
# from sqlalchemy.orm import declarative_base
# Base = declarative_base()

T = TypeVar("T", bound="BaseConfig")


class BaseConfig:
    """
    Lightweight base class for Stephanie config dataclasses.

    Intended usage:

        @dataclass
        class CodeCheckConfig(BaseConfig):
            repo_root: str = "."

    Features:
      - to_dict(drop_none=False): convert to a plain dict (good for logging, DB, JSON).
      - from_dict(cls, data): construct an instance from a dict.
      - update_from_dict(data): in-place update of fields present in `data`.
      - with_overrides(**kwargs): return a *copy* with some fields overridden.
    """

    # -------------------------------
    # Introspection helpers
    # -------------------------------

    @classmethod
    def _field_names(cls) -> set[str]:
        """
        Return the names of dataclass fields defined on this config.

        NOTE: This assumes subclasses are dataclasses.
        """
        if not is_dataclass(cls):
            # If someone forgot @dataclass on a subclass, this is a useful error.
            raise TypeError(f"{cls.__name__} must be a dataclass to use BaseConfig helpers.")
        return {f.name for f in fields(cls)}  # type: ignore[arg-type]

    # -------------------------------
    # Dict <-> Config
    # -------------------------------

    def to_dict(self, *, drop_none: bool = False) -> Dict[str, Any]:
        """
        Convert this config (and nested dataclasses) into a plain dict.

        Args:
            drop_none: If True, omit keys whose values are None.
        """
        if not is_dataclass(self):
            raise TypeError(
                f"{self.__class__.__name__} must be a dataclass to use BaseConfig.to_dict()."
            )

        data = asdict(self)

        if drop_none:
            data = {k: v for k, v in data.items() if v is not None}

        return cast(Dict[str, Any], data)

    @classmethod
    def from_dict(cls: type[T], data: Mapping[str, Any]) -> T:
        """
        Construct an instance of this config from a dict.

        Unknown keys are ignored; missing keys fall back to dataclass defaults.
        """
        # Only keep keys that belong to this config
        valid_keys = cls._field_names()
        filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)  # type: ignore[arg-type]

    # -------------------------------
    # Mutation helpers
    # -------------------------------

    def update_from_dict(self, data: Mapping[str, Any]) -> None:
        """
        In-place update of fields from a dict.

        Only fields that already exist on this config are applied.
        """
        valid_keys = self._field_names()
        for key, value in data.items():
            if key in valid_keys:
                setattr(self, key, value)

    def with_overrides(self: T, **kwargs: Any) -> T:
        """
        Return a *copy* of this config with the given fields overridden.

        Example:
            cfg2 = cfg.with_overrides(repo_root="/tmp/newrepo")
        """
        d = self.to_dict()
        d.update(kwargs)
        return self.__class__.from_dict(d)  # type: ignore[return-value]

    # -------------------------------
    # Nice repr for logging
    # -------------------------------

    def __repr__(self) -> str:  # pragma: no cover - repr is cosmetic
        cls_name = self.__class__.__name__
        try:
            d = self.to_dict()
        except TypeError:
            # Fallback if someone uses this base without @dataclass
            d = self.__dict__
        inner = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"{cls_name}({inner})"
