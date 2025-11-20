# stephanie/zeromodel/visicalc_report.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VisiCalcRegion:
    index: int
    start_row: int
    end_row: int
    row_count: int
    frontier_frac: float
    low_frac: float
    high_frac: float


@dataclass
class VisiCalcReport:
    frontier_metric: str
    frontier_low: float
    frontier_high: float
    row_region_splits: int
    global_frontier_frac: float
    global_low_frac: float
    global_high_frac: float
    regions: List[VisiCalcRegion]


def extract_visicalc_stats(
    ctx: Dict[str, Any],
    stage_index: Optional[int] = None,
    stage_name_hint: str = "visicalc",
) -> Optional[VisiCalcReport]:
    """
    Find the VisiCalcStage metadata in a ZeroModel PipelineExecutor context and
    convert it to a structured VisiCalcReport.

    Args:
        ctx: The context dict returned by PipelineExecutor.run(...)
        stage_index: If provided, use ctx[f"stage_{stage_index}"] directly.
        stage_name_hint: Fallback heuristic if you want to search by name.

    Returns:
        VisiCalcReport or None if not found.
    """
    stage_meta: Optional[Dict[str, Any]] = None

    # 1) Use explicit index if provided
    if stage_index is not None:
        key = f"stage_{stage_index}"
        stage = ctx.get(key)
        if stage and isinstance(stage, dict):
            stage_meta = stage.get("metadata")

    # 2) Fallback: try to find by a name hint (depends on how PipelineExecutor populates ctx)
    if stage_meta is None:
        for k, v in ctx.items():
            if not k.startswith("stage_") or not isinstance(v, dict):
                continue
            name = v.get("name") or v.get("stage") or ""
            if stage_name_hint in str(name).lower():
                stage_meta = v.get("metadata")
                break

    if not stage_meta:
        return None

    global_stats = stage_meta.get("global", {})
    regions_raw = stage_meta.get("regions", [])

    regions: List[VisiCalcRegion] = []
    for r in regions_raw:
        regions.append(
            VisiCalcRegion(
                index=int(r.get("index", 0)),
                start_row=int(r.get("start_row", 0)),
                end_row=int(r.get("end_row", 0)),
                row_count=int(r.get("row_count", 0)),
                frontier_frac=float(r.get("frontier_frac", 0.0)),
                low_frac=float(r.get("low_frac", 0.0)),
                high_frac=float(r.get("high_frac", 0.0)),
            )
        )

    return VisiCalcReport(
        frontier_metric=str(stage_meta.get("frontier_metric", "")),
        frontier_low=float(stage_meta.get("frontier_low", 0.0)),
        frontier_high=float(stage_meta.get("frontier_high", 1.0)),
        row_region_splits=int(stage_meta.get("row_region_splits", len(regions))),
        global_frontier_frac=float(global_stats.get("frontier_frac", 0.0)),
        global_low_frac=float(global_stats.get("low_frac", 0.0)),
        global_high_frac=float(global_stats.get("high_frac", 0.0)),
        regions=regions,
    )

def format_visicalc_report(report: VisiCalcReport) -> str:
    """
    Produce a human-readable multi-line summary of the VisiCalcReport.
    Perfect for logs and tests.
    """
    lines = []
    lines.append(
        f"VisiCalcReport: frontier_metric={report.frontier_metric!r} "
        f"[{report.frontier_low:.2f}, {report.frontier_high:.2f}] "
        f"regions={report.row_region_splits}"
    )
    lines.append(
        f"  Global: frontier={report.global_frontier_frac:.3f}  "
        f"low={report.global_low_frac:.3f}  "
        f"high={report.global_high_frac:.3f}"
    )

    for r in report.regions:
        lines.append(
            f"  [R{r.index}] rows {r.start_row:3d}-{r.end_row:3d} "
            f"(n={r.row_count:3d})  "
            f"frontier={r.frontier_frac:.3f}  "
            f"low={r.low_frac:.3f}  "
            f"high={r.high_frac:.3f}"
        )

    return "\n".join(lines)
