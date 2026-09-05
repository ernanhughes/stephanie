# stephanie/evaluation/shadow.py
"""Legacy funnel shadow hook (Stage 2.5, gate 5) — Stephanie save_bundle families.

Best-effort canonical shadow of ``EvaluationStore.save_bundle`` writes:

* gated by ``STEPHANIE_CANONICAL_SHADOW=1`` (default OFF);
* family allowlist via ``STEPHANIE_SHADOW_FAMILIES`` (default ``"A"``);
* legacy write always runs first; shadow outcome tracked separately;
* shadow failure never raises into production.

Families (inventory §4): A basic scorers, B rankers, C agent evaluations,
D inference/knowledge scorers, E ELM/specialized persistence.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

SHADOW_ENV = "STEPHANIE_CANONICAL_SHADOW"
SHADOW_FAMILIES_ENV = "STEPHANIE_SHADOW_FAMILIES"
SHADOW_DSN_ENV = "STEPHANIE_CANONICAL_SHADOW_DSN"

# Family A: basic scorers (agent_name values seen in save_bundle calls).
FAMILY_A_AGENTS = frozenset({"llm", "universal_scorer", "ScoreEvaluator"})

_repo = None


def shadow_enabled() -> bool:
    return os.environ.get(SHADOW_ENV, "").strip() == "1"


def enabled_families() -> set[str]:
    raw = os.environ.get(SHADOW_FAMILIES_ENV, "A")
    return {part.strip().upper() for part in raw.split(",") if part.strip()}


def agent_family(agent_name: Optional[str]) -> str:
    if (agent_name or "") in FAMILY_A_AGENTS:
        return "A"
    return "?"


def maybe_shadow_bundle(
    *,
    bundle: Any,
    scorable: Any,
    agent_name: Optional[str] = None,
    evaluator: Optional[str] = None,
    model_name: Optional[str] = None,
    source: Optional[str] = None,
    strategy: Optional[str] = None,
    run_id: Optional[str] = None,
) -> tuple[bool, bool, Optional[str]]:
    """Shadow one save_bundle call. Returns (attempted, ok, error)."""
    if not shadow_enabled():
        return False, True, None
    if agent_family(agent_name) not in enabled_families():
        return False, True, None
    try:
        from stephanie.evaluation.criterion import Criterion
        from stephanie.evaluation.evaluation import EvaluationObservation, EvaluatorRef
        from stephanie.evaluation.score import Score
        from stephanie.evaluation.subject import SubjectRef

        import asyncio
        import threading

        observation = EvaluationObservation(
            subject=SubjectRef(
                subject_type=str(getattr(scorable, "target_type", "custom")),
                subject_id=str(getattr(scorable, "id", "unknown")),
                text=getattr(scorable, "text", None),
            ),
            criterion=Criterion(name=str(strategy or "legacy")),
            evaluator=EvaluatorRef(name=str(evaluator or agent_name or "ScoreEvaluator")),
            model_id=str(model_name) if model_name else None,
            run_id=str(run_id) if run_id is not None else None,
            scores=[
                Score(
                    score_id="",
                    evaluation_id="",
                    dimension=str(result.dimension),
                    value=float(result.score or 0.0),
                    weight=result.weight,
                    source=result.source or source,
                    rationale=result.rationale,
                )
                for result in (getattr(bundle, "results", {}) or {}).values()
            ],
            metadata={"shadow_family": "A", "legacy_agent": agent_name},
        )

        outcome: dict[str, Any] = {}

        def _target() -> None:
            try:
                asyncio.run(_append_async(observation))
                outcome["ok"] = True
            except Exception as exc:  # pragma: no cover - defensive
                outcome["ok"] = False
                outcome["error"] = str(exc)

        # Separate thread: safe whether or not the caller holds a running loop.
        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        worker.join(timeout=10)
        if worker.is_alive():
            return True, False, "shadow write timed out"
        if not outcome.get("ok"):
            return True, False, outcome.get("error", "unknown shadow error")
        return True, True, None
    except Exception as exc:
        logger.debug("canonical shadow bundle skipped", exc_info=True)
        return True, False, str(exc)


async def _append_async(observation) -> None:
    from stephanie.evaluation.writer import append_observation

    await append_observation(_repository(), observation)


def _repository():
    global _repo
    if _repo is None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from stephanie.evaluation.persistence.orm import CanonicalBase
        from stephanie.evaluation.persistence.repository import SqlAlchemyEvaluationRepository

        dsn = os.environ.get(
            SHADOW_DSN_ENV, os.environ.get("DATABASE_URL", "postgresql://co:co@localhost:5432/co")
        )
        engine = create_engine(dsn, connect_args={"connect_timeout": 3})
        CanonicalBase.metadata.create_all(engine, checkfirst=True)
        _repo = SqlAlchemyEvaluationRepository(sessionmaker(bind=engine, expire_on_commit=False))
    return _repo
