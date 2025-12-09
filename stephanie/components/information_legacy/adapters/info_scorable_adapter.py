# stephanie/components/information/adapters/info_scorable_adapter.py
from __future__ import annotations

from typing import Any, Dict

from stephanie.models.memcube import MemCubeORM

CORE_INFO_METRICS = {
    # adjust to match your actual metric names from ScorableProcessor
    "clarity",
    "faithfulness",
    "coverage",
    "critic_risk",
    "critic_quality",
}


def memcube_to_scorable_dict(cube: MemCubeORM) -> Dict[str, Any]:
    """
    Adapt an Information MemCube to a generic Scorable dict for ScorableProcessor.

    Assumes cube.extra_data["topic"], ["target"], ["casebook_id"], etc. exist.
    """
    extra = cube.extra_data or {}
    topic = extra.get("topic") or cube.dimension or "topic"
    meta = {
        "topic": topic,
        "target": extra.get("target"),
        "casebook_id": extra.get("casebook_id"),
        "source_profile": extra.get("source_profile"),
        "tags": extra.get("tags", []),
    }
    return {
        "id": f"memcube:{cube.id}",
        "scorable_type": "memcube_info",
        "text": cube.content or "",
        "meta": meta,
    }


async def score_memcube_and_attach_attributes(
    cube: MemCubeORM,
    scorable_processor,
    memory,
    container,
    logger,
) -> dict:
    """
    Run ScorableProcessor on the MemCube and attach selected metrics into extra_data["scores"].

    We don't introduce a new attribute table here; we project into extra_data["scores"]
    so you can later migrate this to a dynamic attribute system.
    """
    scorable_dict = memcube_to_scorable_dict(cube)
    context = {
        # You can pass pipeline run id or anything else your features expect
    }
    row = await scorable_processor.process(scorable_dict, context)

    metrics_cols = row.get("metrics_columns") or []
    metrics_vals = row.get("metrics_values") or []

    scores: dict[str, float] = {}
    for name, val in zip(metrics_cols, metrics_vals):
        if name in CORE_INFO_METRICS:
            try:
                scores[name] = float(val)
            except Exception:
                continue

    # Attach into extra_data
    extra = dict(cube.extra_data or {})
    existing_scores = extra.get("scores", {})
    existing_scores.update(scores)
    extra["scores"] = existing_scores
    cube.extra_data = extra

    logger.log(
        "InfoMemCubeScored",
        {
            "memcube_id": cube.id,
            "scores": scores,
        },
    )
    return scores
