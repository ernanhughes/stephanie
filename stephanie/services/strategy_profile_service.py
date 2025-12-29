# stephanie/services/strategy_profile_service.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from stephanie.memory.strategy_store import IStrategyStore, JsonStrategyStore
from stephanie.orm.strategy import StrategyProfile
from stephanie.services.service_protocol import Service


class StrategyProfileService(Service):
    """
    Thin façade over IStrategyStore (DBStrategyStore or JsonStrategyStore).
    """

    def __init__(
        self,
        *,
        cfg: Optional[Dict[str, Any]],
        memory,
        logger,
        store: Optional[IStrategyStore] = None,
        namespace: str = "default",
    ):
        self.cfg = cfg
        self.logger = logger
        self.memory = memory
        self.namespace = namespace
        self._initialized = False
        self.store: IStrategyStore = store or JsonStrategyStore("./runs/strategy")

    @property
    def name(self) -> str:
        return "strategy-profile-service-v1"

    def initialize(self, **kwargs) -> None:
        self._initialized = True
        try:
            self.logger and self.logger.log(
                "StrategyProfileServiceInit",
                {"backend": self.store.__class__.__name__, "namespace": self.namespace},
            )
        except Exception:
            pass

    def shutdown(self) -> None:
        self._initialized = False

    def health_check(self) -> Dict[str, Any]:
        # optional: expose minimal info
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dependencies": {"backend": self.store.__class__.__name__},
        }

    # ------------ API -------------
    def load(self, *, agent_name: str, scope: Optional[str] = None,
             default: Optional[StrategyProfile] = None) -> StrategyProfile:
        sc = scope or self.namespace
        prof = self.store.peek(agent_name=agent_name, scope=sc)
        if prof:
            return prof
        prof = default or StrategyProfile()
        return self.store.save(agent_name=agent_name, profile=prof, scope=sc)

    def save(self, *, agent_name: str, profile: StrategyProfile,
             scope: Optional[str] = None) -> None:
        sc = scope or self.namespace
        self.store.save(agent_name=agent_name, profile=profile, scope=sc)

    def update(self, *, agent_name: str, scope: Optional[str] = None,
               pacs_weights: Optional[Dict[str, float]] = None,
               verification_threshold: Optional[float] = None) -> StrategyProfile:
        sc = scope or self.namespace
        prof = self.load(agent_name=agent_name, scope=sc)
        prof.update(pacs_weights=pacs_weights, verification_threshold=verification_threshold)
        self.save(agent_name=agent_name, profile=prof, scope=sc)
        return prof

    def delete(self, *, agent_name: str, scope: Optional[str] = None) -> None:
        sc = scope or self.namespace
        # Optional to add on IStrategyStore; if you keep reset(), call that:
        try:
            self.store.reset(agent_name=agent_name, scope=sc)
        except AttributeError:
            # If your interface doesn't define delete/reset, you can no-op or implement it
            pass
