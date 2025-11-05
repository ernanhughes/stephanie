# stephanie/components/gap/risk/monitor.py
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from .plugins.cove import CoVeScorer
from .plugins.selfcheck import SelfCheckScorer

# ------------------------------ Contracts -----------------------------------
log = logging.getLogger(__name__)

@runtime_checkable
class PairScorer(Protocol):
    async def score_text_pair(
        self,
        goal: str,
        reply: str,
        *,
        model_alias: str = "chat",
        monitor_alias: str = "tiny",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Returns a dict with keys (normalized in [0,1] if possible):
          - confidence01          (higher = more confident)
          - faithfulness_risk01   (higher = more hallucination risk)
          - ood_hat01             (higher = more out-of-domain)
          - delta_gap01           (higher = more disagreement vs baseline)
        May include additional fields (token_entropy, etc.), which the
        Aligner can ignore or use.
        """
        ...


# ------------------------------ Helpers -------------------------------------

def _clamp01(x: float) -> float:
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_get(d: Dict[str, float], k: str, default: float = 0.5) -> float:
    try:
        return float(d.get(k, default))
    except Exception:
        return float(default)


async def _with_timeout(coro, timeout_s: float, logger: logging.Logger) -> Any:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning("Monitor scorer timed out after %.2fs", timeout_s)
        raise
    except Exception:
        # Let caller decide how to fallback; we log here.
        logger.exception("Monitor scorer failed")
        raise


# ------------------------------ Adapters ------------------------------------

@dataclass
class MetricsServiceAdapter(PairScorer):
    """
    Thin adapter over container.metrics_service if it exposes `score_text_pair`.
    """
    svc: Any
    logger: logging.Logger

    async def score_text_pair(  # type: ignore[override]
        self,
        goal: str,
        reply: str,
        *,
        model_alias: str = "chat",
        monitor_alias: str = "tiny",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        # Pass through; expect already-normalized fields if your service provides them.
        result = await self.svc.score_text_pair(
            goal, reply, model_alias=model_alias, monitor_alias=monitor_alias, context=context
        )
        # Ensure required keys exist (fallback to 0.5)
        return dict(
            confidence01=_clamp01(_safe_get(result, "confidence01", 0.5)),
            faithfulness_risk01=_clamp01(_safe_get(result, "faithfulness_risk01", 0.5)),
            ood_hat01=_clamp01(_safe_get(result, "ood_hat01", 0.5)),
            delta_gap01=_clamp01(_safe_get(result, "delta_gap01", 0.5)),
            # Keep any extras for downstream debugging/telemetry
            **{k: v for k, v in result.items() if k not in {
                "confidence01", "faithfulness_risk01", "ood_hat01", "delta_gap01"
            }},
        )


@dataclass
class TinyAdapter(PairScorer):
    """
    Adapter for a Tiny-class scorer (e.g., TinyRecursionModel inference service).
    Expected methods on `tiny` (all optional, best-effort):
      - score_text_pair(goal, reply, **kwargs) -> dict
      - score(goal, reply) -> dict
    """
    tiny: Any
    logger: logging.Logger

    async def score_text_pair(  # type: ignore[override]
        self,
        goal: str,
        reply: str,
        *,
        model_alias: str = "chat",
        monitor_alias: str = "tiny",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        # Prefer a compatible score_text_pair; otherwise try a generic score()
        if hasattr(self.tiny, "score_text_pair"):
            out = await self.tiny.score_text_pair(goal, reply, model_alias=model_alias, monitor_alias=monitor_alias, context=context)  # type: ignore
        elif hasattr(self.tiny, "score"):
            out = await self.tiny.score(goal, reply)  # type: ignore
        else:
            log.warning("TinyAdapter: no compatible method found, returning neutral metrics")
            return _neutral_metrics()

        return dict(
            confidence01=_clamp01(_safe_get(out, "confidence01", _invert01(_safe_get(out, "uncertainty01", 0.5)))),
            faithfulness_risk01=_clamp01(_safe_get(out, "faithfulness_risk01", _safe_get(out, "halluc_risk01", 0.5))),
            ood_hat01=_clamp01(_safe_get(out, "ood_hat01", _safe_get(out, "ood_risk01", 0.5))),
            delta_gap01=_clamp01(_safe_get(out, "delta_gap01", _estimate_delta_gap(out))),
            **{k: v for k, v in out.items() if k not in {
                "confidence01", "faithfulness_risk01", "ood_hat01", "delta_gap01"
            }},
        )


@dataclass
class HRMAdapter(PairScorer):
    """
    Adapter for an HRM-based scorer. We assume it can produce confidence/uncertainty
    and a disagreement proxy when compared to Tiny (if available upstream).
    """
    hrm: Any
    logger: logging.Logger

    async def score_text_pair(  # type: ignore[override]
        self,
        goal: str,
        reply: str,
        *,
        model_alias: str = "chat",
        monitor_alias: str = "hrm",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        # Expect similar contract; adapt field names if needed
        if hasattr(self.hrm, "score_text_pair"):
            out = await self.hrm.score_text_pair(goal, reply, model_alias=model_alias, monitor_alias=monitor_alias, context=context)  # type: ignore
        elif hasattr(self.hrm, "score"):
            out = await self.hrm.score(goal, reply)  # type: ignore
        else:
            _logger.warning("HRMAdapter: no compatible method found, returning neutral metrics")
            return _neutral_metrics()

        return dict(
            confidence01=_clamp01(_safe_get(out, "confidence01", _invert01(_safe_get(out, "uncertainty01", 0.5)))),
            faithfulness_risk01=_clamp01(_safe_get(out, "faithfulness_risk01", _safe_get(out, "halluc_risk01", 0.5))),
            ood_hat01=_clamp01(_safe_get(out, "ood_hat01", _safe_get(out, "ood_risk01", 0.5))),
            delta_gap01=_clamp01(_safe_get(out, "delta_gap01", 0.5)),  # true Δ computed in aligner if both models are present
            **{k: v for k, v in out.items() if k not in {
                "confidence01", "faithfulness_risk01", "ood_hat01", "delta_gap01"
            }},
        )


# ------------------------------ Fallbacks -----------------------------------

def _neutral_metrics() -> Dict[str, float]:
    return dict(
        confidence01=0.5,
        faithfulness_risk01=0.5,
        ood_hat01=0.5,
        delta_gap01=0.5,
    )


def _invert01(x: float) -> float:
    return _clamp01(1.0 - x)


def _estimate_delta_gap(out: Dict[str, float]) -> float:
    """
    If we only have a single-model view, approximate Δ as a blend of
    uncertainty and ood (as disagreement proxy). This is a placeholder;
    true Δ should be produced in the Aligner when both Tiny & HRM (or baseline)
    are available.
    """
    unc = _clamp01(_safe_get(out, "uncertainty01", 0.5))
    ood = _clamp01(_safe_get(out, "ood_hat01", _safe_get(out, "ood_risk01", 0.5)))
    return _clamp01(0.5 * unc + 0.5 * ood)


# ------------------------------ Monitor Service -----------------------------

class MonitorService:
    """
    Container-aware monitor that chooses the best available scorer:
      1) container.metrics_service (preferred, unified contract)
      2) container.tiny_scorer / container.tiny / container.monitor_tiny
      3) container.hrm_scorer / container.hrm
      4) neutral fallback

    Usage:
        monitor = MonitorService(container, logger=logger)
        metrics = await monitor.score(goal, reply, model_alias="chat-hrm", monitor_alias="tiny")
    """

    def __init__(
        self,
        container: Any,
        *,
        logger: Optional[logging.Logger] = None,
        timeout_s: float = 2.5,
    ) -> None:
        self.container = container
        self.logger = logger or logging.getLogger(__name__)
        self.timeout_s = float(timeout_s)

        self.primary: Optional[PairScorer] = None
        self.secondary: Optional[PairScorer] = None

        # Discover services in the container
        ms = container.get("scoring")
        if ms is not None and hasattr(ms, "score_text_pair"):
            self.primary = MetricsServiceAdapter(ms, self.logger)

        self._selfcheck_adapter = SelfCheckScorer(container, self.logger)
        self._cove_adapter = CoVeScorer(container, logger=self.logger)
        
        tiny = (
            getattr(container, "tiny_scorer", None)
            or getattr(container, "monitor_tiny", None)
            or getattr(container, "tiny", None)
        )
        if tiny is not None:
            self.secondary = TinyAdapter(tiny, self.logger)

        # HRM is optional; aligner can use it when present for Δ
        self._hrm_adapter: Optional[PairScorer] = None
        hrm = getattr(container, "hrm_scorer", None) or getattr(container, "hrm", None)
        if hrm is not None:
            self._hrm_adapter = HRMAdapter(hrm, self.logger)

        self.logger.info(
            "MonitorService ready | primary=%s secondary=%s hrm=%s",
            type(self.primary).__name__ if self.primary else "None",
            type(self.secondary).__name__ if self.secondary else "None",
            type(self._hrm_adapter).__name__ if self._hrm_adapter else "None",
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def score(
        self,
        goal: str,
        reply: str,
        *,
        model_alias: str = "chat",
        monitor_alias: str = "tiny",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Returns normalized monitor metrics. This method is resilient:
        it tries primary -> secondary -> HRM -> neutral fallback.
        """
        # Guard against context leakage: we do NOT inject retrieval/evidence here.
        # Let the Aligner compute evidence coverage/faithfulness against context later.
        scorer_chain: list[PairScorer] = []
        if self.primary:
            scorer_chain.append(self.primary)

        # If the caller explicitly asks for HRM
        if monitor_alias.lower().startswith("hrm") and self._hrm_adapter:
            scorer_chain.append(self._hrm_adapter)

        # Prefer Tiny otherwise
        if self.secondary and not monitor_alias.lower().startswith("hrm"):
            scorer_chain.append(self.secondary)

        # If nothing else, try HRM as a final option
        if self._hrm_adapter and self._hrm_adapter not in scorer_chain:
            scorer_chain.append(self._hrm_adapter)

        # Execute the chain with timeouts
        for scorer in scorer_chain:
            try:
                result = await _with_timeout(
                    scorer.score_text_pair(
                        goal, reply,
                        model_alias=model_alias,
                        monitor_alias=monitor_alias,
                        context=None  # intentionally stripped to avoid bias
                    ),
                    self.timeout_s,
                    self.logger,
                )
                # Ensure required keys exist & clamped
                return dict(
                    confidence01=_clamp01(_safe_get(result, "confidence01", 0.5)),
                    faithfulness_risk01=_clamp01(_safe_get(result, "faithfulness_risk01", 0.5)),
                    ood_hat01=_clamp01(_safe_get(result, "ood_hat01", 0.5)),
                    delta_gap01=_clamp01(_safe_get(result, "delta_gap01", 0.5)),
                    **{k: v for k, v in result.items() if k not in {
                        "confidence01", "faithfulness_risk01", "ood_hat01", "delta_gap01"
                    }},
                )
            except Exception:
                continue

        if monitor_alias.lower().startswith("selfcheck") and self._selfcheck_adapter:
            scorer_chain.insert(0, self._selfcheck_adapter)        

        # Total fallback
        _logger.warning("MonitorService: all scorers failed; returning neutral metrics")
        return _neutral_metrics()


__all__ = ["MonitorService", "PairScorer"]
