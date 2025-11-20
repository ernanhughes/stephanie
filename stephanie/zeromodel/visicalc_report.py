# stephanie/zeromodel/visicalc_report.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import logging
import numpy as np
import json

import csv
from pathlib import Path

log = logging.getLogger(__name__)


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class VisiCalcRegion:
    index: int
    start_row: int
    end_row: int
    row_count: int
    frontier_frac: float
    low_frac: float
    high_frac: float
    mean_frontier_value: float


@dataclass
class VisiCalcReport:
    frontier_metric: str
    frontier_low: float
    frontier_high: float
    row_region_splits: int
    regions: List[VisiCalcRegion]
    global_low_frac: float
    global_high_frac: float
    global_frontier_frac: float = 0.0
    global_mean: float = 0.0
    global_std: float = 0.0
    global_min: float = 0.0
    global_max: float = 0.0
    entropy: float = 0.0
    sparsity_level_e3: float = 0.0
    sparsity_level_e2: float = 0.0


# -----------------------------
# Stand-alone calculator
# -----------------------------

def compute_visicalc_report(
    vpm: np.ndarray,
    metric_names: Optional[Sequence[str]] = None,
    frontier_metric: Optional[str] = None,
    frontier_low: float = 0.25,
    frontier_high: float = 0.75,
    row_region_splits: int = 3,
) -> VisiCalcReport:
    """
    Pure VisiCalc implementation that works directly on a (rows x metrics) matrix.

    This is the canonical implementation for now. ZeroModel stages can later call
    into this function once we're happy with the behaviour.

    Args:
        vpm: 2D array, shape (num_rows, num_metrics)
        metric_names: optional list of names for each metric column
        frontier_metric: name of the metric to use as the frontier band anchor
        frontier_low / frontier_high: numeric band for frontier coverage
        row_region_splits: number of row regions (top/mid/bottom etc.)

    Returns:
        VisiCalcReport with global + regional coverage stats.
    """
    vpm = np.asarray(vpm)  
    # Try to infer a valid 2D view
    if vpm.ndim == 1:
        vpm = vpm[None, :]                     # (1, D)
    elif vpm.ndim == 3 and 1 in vpm.shape:
        # e.g., (3, 1, 708) → collapse singleton dimension
        vpm = np.squeeze(vpm)
        if vpm.ndim == 1:
            vpm = vpm[None, :]
    elif vpm.ndim > 2:
        # As a fallback, flatten all but the last dimension
        vpm = vpm.reshape(-1, vpm.shape[-1])

    if vpm.ndim != 2:
        raise ValueError(
            f"compute_visicalc_report expects a 2D matrix (rows x metrics), "
            f"got shape={vpm.shape!r}"
        )
    num_rows, num_metrics = vpm.shape
    if row_region_splits <= 0:
        log.warning(
            "compute_visicalc_report received non-positive row_region_splits=%r; "
            "falling back to 1.",
            row_region_splits,
        )
        row_region_splits = 1

    # Handle empty matrices early
    if num_rows == 0 or num_metrics == 0:
        log.warning(
            "compute_visicalc_report called with empty matrix shape=%r; "
            "returning empty report.",
            vpm.shape,
        )
        fm_name = frontier_metric or (metric_names[0] if metric_names else "")
        return VisiCalcReport(
            frontier_metric=fm_name,
            frontier_low=frontier_low,
            frontier_high=frontier_high,
            row_region_splits=row_region_splits,
            regions=[],
            global_low_frac=0.0,
            global_high_frac=0.0,
            global_frontier_frac=0.0,
        )

    # Metric names reconciliation
    if metric_names is None:
        metric_names = [f"m{i}" for i in range(num_metrics)]
    else:
        metric_names = list(metric_names)
        if len(metric_names) != num_metrics:
            if len(metric_names) > num_metrics:
                metric_names = metric_names[:num_metrics]
            else:
                metric_names = metric_names + [
                    f"m{i}" for i in range(len(metric_names), num_metrics)
                ]

    # Global stats
    global_mean = float(np.mean(vpm))
    global_std = float(np.std(vpm))
    global_min = float(np.min(vpm))
    global_max = float(np.max(vpm))

    sparsity_e3 = float(np.mean(vpm <= 1e-3))
    sparsity_e2 = float(np.mean(vpm <= 1e-2))

    hist, _ = np.histogram(vpm, bins=32, range=(0.0, 1.0), density=True)
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    if hist.size > 0:
        entropy = float(-np.sum(hist * np.log(hist)) / np.log(hist.size))
    else:
        entropy = 0.0

    # Frontier metric selection
    if frontier_metric is None:
        frontier_idx = 0
        frontier_name = metric_names[0]
    else:
        try:
            frontier_idx = metric_names.index(frontier_metric)
            frontier_name = frontier_metric
        except ValueError:
            log.warning(
                "compute_visicalc_report: frontier_metric %r not found in metric_names; "
                "falling back to %r",
                frontier_metric,
                metric_names[0],
            )
            frontier_idx = 0
            frontier_name = metric_names[0]

    frontier_values = vpm[:, frontier_idx]
    low = float(frontier_low)
    high = float(frontier_high)

    if low >= high:
        log.warning(
            "compute_visicalc_report: frontier_low=%.4f >= frontier_high=%.4f; "
            "clamping band to [0.0, 1.0].",
            low,
            high,
        )
        low, high = 0.0, 1.0

    band_mask = (frontier_values >= low) & (frontier_values <= high)
    frontier_fraction = float(band_mask.mean())
    num_frontier_rows = int(band_mask.sum())

    low_mask = frontier_values < low
    high_mask = frontier_values > high
    global_low_frac = float(low_mask.mean())
    global_high_frac = float(high_mask.mean())

    # Row regions
    splits = row_region_splits
    region_bounds: List[tuple[int, int]] = []
    base = num_rows // splits
    remainder = num_rows % splits
    start = 0
    for i in range(splits):
        extra = 1 if i < remainder else 0
        end = start + base + extra
        region_bounds.append((start, end))
        start = end

    regions: List[VisiCalcRegion] = []
    for i, (rs, re) in enumerate(region_bounds):
        if rs >= re:
            # Empty region
            regions.append(
                VisiCalcRegion(
                    index=i,
                    start_row=rs,
                    end_row=re,
                    row_count=0,
                    frontier_frac=0.0,
                    low_frac=0.0,
                    high_frac=0.0,
                    mean_frontier_value=0.0,
                )
            )
            continue

        region_frontier_vals = frontier_values[rs:re]
        region_band_mask = band_mask[rs:re]

        region_frontier_fraction = float(region_band_mask.mean())

        region_low_mask = region_frontier_vals < low
        region_high_mask = region_frontier_vals > high
        region_low_frac = float(region_low_mask.mean())
        region_high_frac = float(region_high_mask.mean())

        mean_frontier_value = float(region_frontier_vals.mean())

        region = VisiCalcRegion(
            index=i,
            start_row=int(rs),
            end_row=int(re),
            row_count=int(re - rs),
            frontier_frac=region_frontier_fraction,
            low_frac=region_low_frac,
            high_frac=region_high_frac,
            mean_frontier_value=mean_frontier_value,
        )
        regions.append(region)

    report = VisiCalcReport(
        frontier_metric=frontier_name,
        frontier_low=low,
        frontier_high=high,
        row_region_splits=splits,
        regions=regions,
        global_low_frac=global_low_frac,
        global_high_frac=global_high_frac,
        global_frontier_frac=frontier_fraction,
        global_mean=global_mean,
        global_std=global_std,
        global_min=global_min,
        global_max=global_max,
        entropy=entropy,
        sparsity_level_e3=sparsity_e3,
        sparsity_level_e2=sparsity_e2,
    )

    log.debug("compute_visicalc_report produced report:\n%s", format_visicalc_report(report))
    return report


# -----------------------------
# Legacy / stage-based extractor
# -----------------------------

def extract_visicalc_stats(
    ctx: Dict[str, Any],
    stage_index: Optional[int] = None,
    stage_name_hint: str = "visicalc",
) -> Optional[VisiCalcReport]:
    """
    Find the VisiCalcStage metadata in a ZeroModel PipelineExecutor context and
    convert it to a structured VisiCalcReport.

    This is kept for compatibility with ZeroModel's PipelineExecutor.
    For new code you can call compute_visicalc_report(...) directly.
    """
    if not isinstance(ctx, dict):
        log.warning(
            "extract_visicalc_stats expected ctx to be dict, got %r",
            type(ctx),
        )
        return None

    ctx_stage_keys = sorted(k for k in ctx.keys() if str(k).startswith("stage_"))
    log.debug(
        "extract_visicalc_stats: stage_index=%r, stage_name_hint=%r, "
        "stage_keys=%s",
        stage_index,
        stage_name_hint,
        ctx_stage_keys,
    )

    stage_meta: Optional[Dict[str, Any]] = None

    # 1) Explicit index
    if stage_index is not None:
        key = f"stage_{stage_index}"
        stage = ctx.get(key)
        log.debug(
            "Trying explicit stage_index=%r → key=%r, found=%s",
            stage_index,
            key,
            isinstance(stage, dict),
        )
        if stage and isinstance(stage, dict):
            stage_meta = stage.get("metadata")
            log.debug(
                "Explicit stage %s metadata keys: %s",
                key,
                sorted(stage_meta.keys()) if isinstance(stage_meta, dict) else None,
            )

    # 2) Fallback: search by name
    if stage_meta is None:
        log.debug(
            "Explicit stage_index path failed or not provided; "
            "falling back to name search with hint=%r",
            stage_name_hint,
        )
        for k, v in ctx.items():
            if not str(k).startswith("stage_") or not isinstance(v, dict):
                continue
            name = v.get("name") or v.get("stage") or ""
            log.debug(
                "Examining context entry %r with name=%r for visicalc match",
                k,
                name,
            )
            if stage_name_hint.lower() in str(name).lower():
                stage_meta = v.get("metadata")
                log.debug(
                    "Matched VisiCalc stage at %r; metadata keys=%s",
                    k,
                    sorted(stage_meta.keys())
                    if isinstance(stage_meta, dict)
                    else None,
                )
                break

    if not stage_meta:
        log.warning(
            "VisiCalc stage metadata not found in context; "
            "available stage_* keys: %s",
            ctx_stage_keys,
        )
        return None

    log.debug("Raw VisiCalc stage_meta keys: %s", sorted(stage_meta.keys()))

    visicalc = stage_meta.get("visicalc", {}) or {}
    global_stats = visicalc.get("global", {}) or {}
    row_regions_raw = visicalc.get("row_regions", {})

    # row_regions is typically a dict: {"region_0": {...}, ...}
    if isinstance(row_regions_raw, dict):
        region_items = list(row_regions_raw.values())
    elif isinstance(row_regions_raw, list):
        region_items = row_regions_raw
    else:
        region_items = []

    log.debug(
        "Extracted visicalc sub-structure: has_visicalc=%s, "
        "has_global=%s, regions_count=%d",
        bool(visicalc),
        bool(global_stats),
        len(region_items),
    )

    # Frontier config
    frontier_metric = str(visicalc.get("frontier_metric", ""))
    frontier_low = float(visicalc.get("frontier_low", 0.0))
    frontier_high = float(visicalc.get("frontier_high", 1.0))
    global_frontier_frac = float(visicalc.get("frontier_frac", 0.0))

    # Regions
    regions: List[VisiCalcRegion] = []
    for i, r in enumerate(region_items):
        region = VisiCalcRegion(
            index=int(r.get("index", i)),
            start_row=int(r.get("row_start", 0)),
            end_row=int(r.get("row_end", 0)),
            row_count=int(r.get("num_rows", 0)),
            frontier_frac=float(r.get("frontier_fraction", 0.0)),
            low_frac=float(r.get("low_frac", 0.0)),
            high_frac=float(r.get("high_frac", 0.0)),
            mean_frontier_value=float(r.get("mean_frontier_value", 0.0) or 0.0),
        )
        regions.append(region)
        log.debug(
            "Parsed VisiCalcRegion: idx=%d rows=%d-%d (n=%d) frontier=%.3f low=%.3f high=%.3f",
            region.index,
            region.start_row,
            region.end_row,
            region.row_count,
            region.frontier_frac,
            region.low_frac,
            region.high_frac,
        )

    # Global stats (sync names with stage)
    global_low_frac = float(global_stats.get("low_frac", 0.0))
    global_high_frac = float(global_stats.get("high_frac", 0.0))

    sparsity_e3 = float(
        global_stats.get("sparsity_le_1e-3", global_stats.get("sparsity_level_1e-3", 0.0))
    )
    sparsity_e2 = float(
        global_stats.get("sparsity_le_1e-2", global_stats.get("sparsity_level_1e-2", 0.0))
    )

    report = VisiCalcReport(
        frontier_metric=frontier_metric,
        frontier_low=frontier_low,
        frontier_high=frontier_high,
        row_region_splits=int(visicalc.get("row_region_splits", len(regions))),
        regions=regions,
        global_low_frac=global_low_frac,
        global_high_frac=global_high_frac,
        global_frontier_frac=global_frontier_frac,
        global_mean=float(global_stats.get("mean", 0.0)),
        global_std=float(global_stats.get("std", 0.0)),
        global_min=float(global_stats.get("min", 0.0)),
        global_max=float(global_stats.get("max", 0.0)),
        entropy=float(global_stats.get("entropy", 0.0)),
        sparsity_level_e3=sparsity_e3,
        sparsity_level_e2=sparsity_e2,
    )

    log.info(
        "Built VisiCalcReport for frontier_metric=%r, regions=%d, "
        "global frontier=%.3f low=%.3f high=%.3f",
        report.frontier_metric,
        len(report.regions),
        report.global_frontier_frac,
        report.global_low_frac,
        report.global_high_frac,
    )
    log.debug("VisiCalcReport detail:\n%s", format_visicalc_report(report))

    return report


# -----------------------------
# Pretty printer
# -----------------------------

def format_visicalc_report(report: VisiCalcReport) -> str:
    """
    Produce a human-readable multi-line summary of the VisiCalcReport.
    Perfect for logs and tests.
    """
    lines: List[str] = []

    lines.append(
        f"VisiCalcReport: frontier_metric={report.frontier_metric!r} "
        f"[{report.frontier_low:.2f}, {report.frontier_high:.2f}] "
        f"regions={report.row_region_splits}"
    )
    lines.append(
        f"  Global: frontier={report.global_frontier_frac:.3f}  "
        f"low={report.global_low_frac:.3f}  "
        f"high={report.global_high_frac:.3f}  "
        f"mean={report.global_mean:.4f}  "
        f"std={report.global_std:.4f}  "
        f"min={report.global_min:.4f}  "
        f"max={report.global_max:.4f}  "
        f"entropy={report.entropy:.3f}  "
        f"sparsity<=1e-3={report.sparsity_level_e3:.3f}"
    )

    for r in report.regions:
        lines.append(
            f"  [R{r.index}] rows {r.start_row:3d}-{r.end_row:3d} "
            f"(n={r.row_count:3d})  "
            f"frontier={r.frontier_frac:.3f}  "
            f"low={r.low_frac:.3f}  "
            f"high={r.high_frac:.3f}  "
            f"mean_frontier={r.mean_frontier_value:.4f}"
        )

    text = "\n".join(lines)
    log.debug(
        "format_visicalc_report produced %d lines for frontier_metric=%r",
        len(lines),
        report.frontier_metric,
    )
    return text

def validate_visicalc_report(
    report: Optional[VisiCalcReport],
    expected_regions: Optional[int] = None,
) -> None:
    """
    Lightweight sanity checker for VisiCalcReport.
    Intended for use in tests and debugging.

    Raises AssertionError if anything is obviously inconsistent.
    """

    # --- basic object checks ---
    assert report is not None, "VisiCalcReport is None"
    assert isinstance(report, VisiCalcReport), (
        f"Expected VisiCalcReport, got {type(report)}"
    )

    # --- frontier band sanity ---
    assert 0.0 <= report.frontier_low < report.frontier_high <= 1.0, (
        f"Invalid frontier band: [{report.frontier_low}, {report.frontier_high}]"
    )

    # --- region count / splits ---
    if expected_regions is not None:
        assert report.row_region_splits == expected_regions, (
            f"row_region_splits={report.row_region_splits} "
            f"!= expected_regions={expected_regions}"
        )
    assert len(report.regions) == report.row_region_splits, (
        f"len(regions)={len(report.regions)} "
        f"!= row_region_splits={report.row_region_splits}"
    )

    # --- global fractions ---
    for name, val in [
        ("global_frontier_frac", report.global_frontier_frac),
        ("global_low_frac", report.global_low_frac),
        ("global_high_frac", report.global_high_frac),
    ]:
        assert 0.0 <= val <= 1.0 + 1e-6, (
            f"{name} out of range [0,1]: {val}"
        )

    # --- global stats consistency (weak checks) ---
    assert report.global_min <= report.global_max, (
        f"global_min={report.global_min} > global_max={report.global_max}"
    )
    assert report.global_std >= 0.0, (
        f"global_std must be non-negative, got {report.global_std}"
    )

    # --- per-region checks ---
    for r in report.regions:
        assert 0 <= r.start_row <= r.end_row, (
            f"Region {r.index}: invalid row range "
            f"[{r.start_row}, {r.end_row})"
        )

        expected_count = max(0, r.end_row - r.start_row)
        assert r.row_count == expected_count, (
            f"Region {r.index}: row_count={r.row_count} "
            f"!= end_row-start_row={expected_count}"
        )

        for attr in ("frontier_frac", "low_frac", "high_frac"):
            v = getattr(r, attr)
            assert 0.0 <= v <= 1.0 + 1e-6, (
                f"Region {r.index}: {attr} out of range [0,1]: {v}"
            )

    log.debug(
        "validate_visicalc_report OK: frontier_metric=%r, regions=%d",
        report.frontier_metric,
        len(report.regions),
    )

def visicalc_region_to_dict(region: VisiCalcRegion) -> Dict[str, Any]:
    """
    Convert a VisiCalcRegion to a plain dict (JSON/CSV friendly).
    """
    return {
        "index": region.index,
        "start_row": region.start_row,
        "end_row": region.end_row,
        "row_count": region.row_count,
        "frontier_frac": region.frontier_frac,
        "low_frac": region.low_frac,
        "high_frac": region.high_frac,
        "mean_frontier_value": region.mean_frontier_value,
    }


def visicalc_report_to_dict(report: VisiCalcReport) -> Dict[str, Any]:
    """
    Convert a VisiCalcReport to a nested dict:

    {
      "frontier": { ... },
      "global":   { ... },
      "regions":  [ ... ]
    }

    This is the canonical serialization used by JSON / CSV writers.
    """
    frontier = {
        "metric": report.frontier_metric,
        "low": report.frontier_low,
        "high": report.frontier_high,
        "row_region_splits": report.row_region_splits,
    }

    global_stats = {
        "frontier_frac": report.global_frontier_frac,
        "low_frac": report.global_low_frac,
        "high_frac": report.global_high_frac,
        "mean": report.global_mean,
        "std": report.global_std,
        "min": report.global_min,
        "max": report.global_max,
        "entropy": report.entropy,
        "sparsity_level_e3": report.sparsity_level_e3,
        "sparsity_level_e2": report.sparsity_level_e2,
    }

    regions = [visicalc_region_to_dict(r) for r in report.regions]

    return {
        "frontier": frontier,
        "global": global_stats,
        "regions": regions,
    }

def save_visicalc_report_csv(
    report: VisiCalcReport,
    path: Union[str, Path],
) -> Path:
    """
    Write the VisiCalcReport to a CSV file.

    Structure:
      - One 'global' row (overall stats for the frontier metric)
      - One 'region' row per VisiCalcRegion

    Columns are fixed so the file is easy to ingest in Pandas / DuckDB.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "level",                # 'global' | 'region'
        # global + frontier config
        "frontier_metric",
        "frontier_low",
        "frontier_high",
        "row_region_splits",
        "global_frontier_frac",
        "global_low_frac",
        "global_high_frac",
        "global_mean",
        "global_std",
        "global_min",
        "global_max",
        "entropy",
        "sparsity_level_e3",
        "sparsity_level_e2",
        # region-specific (empty for global row)
        "region_index",
        "region_start_row",
        "region_end_row",
        "region_row_count",
        "region_frontier_frac",
        "region_low_frac",
        "region_high_frac",
        "region_mean_frontier_value",
    ]

    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # --- global row ---
        writer.writerow({
            "level": "global",
            "frontier_metric": report.frontier_metric,
            "frontier_low": report.frontier_low,
            "frontier_high": report.frontier_high,
            "row_region_splits": report.row_region_splits,
            "global_frontier_frac": report.global_frontier_frac,
            "global_low_frac": report.global_low_frac,
            "global_high_frac": report.global_high_frac,
            "global_mean": report.global_mean,
            "global_std": report.global_std,
            "global_min": report.global_min,
            "global_max": report.global_max,
            "entropy": report.entropy,
            "sparsity_level_e3": report.sparsity_level_e3,
            "sparsity_level_e2": report.sparsity_level_e2,
            # region fields left blank / None
            "region_index": None,
            "region_start_row": None,
            "region_end_row": None,
            "region_row_count": None,
            "region_frontier_frac": None,
            "region_low_frac": None,
            "region_high_frac": None,
            "region_mean_frontier_value": None,
        })

        # --- per-region rows ---
        for r in report.regions:
            writer.writerow({
                "level": "region",
                "frontier_metric": report.frontier_metric,
                "frontier_low": report.frontier_low,
                "frontier_high": report.frontier_high,
                "row_region_splits": report.row_region_splits,
                "global_frontier_frac": report.global_frontier_frac,
                "global_low_frac": report.global_low_frac,
                "global_high_frac": report.global_high_frac,
                "global_mean": report.global_mean,
                "global_std": report.global_std,
                "global_min": report.global_min,
                "global_max": report.global_max,
                "entropy": report.entropy,
                "sparsity_level_e3": report.sparsity_level_e3,
                "sparsity_level_e2": report.sparsity_level_e2,
                "region_index": r.index,
                "region_start_row": r.start_row,
                "region_end_row": r.end_row,
                "region_row_count": r.row_count,
                "region_frontier_frac": r.frontier_frac,
                "region_low_frac": r.low_frac,
                "region_high_frac": r.high_frac,
                "region_mean_frontier_value": r.mean_frontier_value,
            })

    log.info("Wrote VisiCalcReport CSV to %s", p.as_posix())
    return p


def save_visicalc_report_json(
    report: VisiCalcReport,
    path: Union[str, Path],
    *,
    indent: int = 2,
) -> Path:
    """
    Write the VisiCalcReport to a JSON file (global + regions).

    Returns:
        The resolved Path of the written file.
    """
    p = Path(path)
    data = visicalc_report_to_dict(report)

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

    log.info("Wrote VisiCalcReport JSON to %s", p.as_posix())
    return p
