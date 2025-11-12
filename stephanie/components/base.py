from __future__ import annotations

from typing import Any, Dict, Iterable, Protocol

from fastapi import APIRouter


class Component(Protocol):
    """Minimal interface for SIS-loadable components."""
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def status(self) -> Dict[str, Any]: ...
    def routers(self) -> Iterable[APIRouter]: ...
    def exports(self) -> Dict[str, Any]: ...  # name -> object (service/function)
