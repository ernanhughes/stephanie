from __future__ import annotations

from typing import Any, Callable, Dict

from omegaconf import DictConfig


class SSPDeps:
    """
    Lightweight DI to wire implementations for testing or runtime changes.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._f: Dict[str, Callable[[DictConfig], Any]] = {}
        self._i: Dict[str, Any] = {}

    def register(self, name: str, factory: Callable[[DictConfig], Any]) -> None:
        self._f[name] = factory

    def get(self, name: str) -> Any:
        if name not in self._i:
            if name not in self._f:
                raise KeyError(f"SSPDeps: no factory registered for '{name}'")
            self._i[name] = self._f[name](self.cfg)
        return self._i[name]
