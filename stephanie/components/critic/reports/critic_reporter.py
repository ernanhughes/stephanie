from __future__ import annotations
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _pct(x, n): 
    return 0.0 if n == 0 else 100.0 * (x / n)

def _safe(a, default=0.0):
    try: return float(a)
    except Exception: return default

def _is_prob_vec(vals: List[float]) -> Tuple[int, int]:
    ok = sum(1 for v in vals if 0.0 <= v <= 1.0 and not math.isnan(v))
    bad = len(vals) - ok
    return ok, bad

def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n_rows": 0}

    # metric columns/value integrity (after your MetricFilter group feature)
    cols_any = set()
    n_prob_ok, n_prob_bad = 0, 0

    for r in rows:
        cols = r.get("metrics_columns") or []
        vals = r.get("metrics_values") or []
        for c in cols: cols_any.add(c)
        ok, bad = _is_prob_vec([_safe(v) for v in vals])
        n_prob_ok += ok
        n_prob_bad += bad

    # VisiCalc (group feature) annotations (if present)
    # each row can carry: visicalc_feature_names / visicalc_features / visicalc_quality
    vpm_quality = None
    found_vpmq = 0
    for r in rows:
        q = r.get("visicalc_quality")
        if isinstance(q, (float, int)):
            found_vpmq += 1
            vpm_quality = q if vpm_quality is None else (vpm_quality + q)
    if found_vpmq:
        vpm_quality = float(vpm_quality) / float(found_vpmq)

    return {
        "n_rows": n,
        "n_unique_metrics": len(cols_any),
        "prob_values_ok": int(n_prob_ok),
        "prob_values_bad": int(n_prob_bad),
        "prob_values_ok_pct": _pct(n_prob_ok, n_prob_ok + n_prob_bad),
        "visicalc_quality_mean": vpm_quality,
    }

def compare_model_features(model_feature_names: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Collect produced feature names from VisiCalc, if available; otherwise from metrics
    produced = None
    for r in rows:
        names = r.get("visicalc_feature_names")
        if names:
            produced = names
            break
    if produced is None:
        # fallback: use metric column names after filter
        for r in rows:
            names = r.get("metrics_columns")
            if names:
                produced = names
                break

    produced = produced or []
    model_set   = set(model_feature_names or [])
    produced_set = set(produced)

    missing_for_model = sorted(list(model_set - produced_set))
    extra_from_rows   = sorted(list(produced_set - model_set))
    overlap           = sorted(list(model_set & produced_set))

    return {
        "model_features_total": len(model_set),
        "produced_features_total": len(produced_set),
        "overlap": overlap[:50],  # preview
        "overlap_count": len(overlap),
        "missing_for_model_preview": missing_for_model[:50],
        "missing_for_model_count": len(missing_for_model),
        "extra_from_rows_preview": extra_from_rows[:50],
        "extra_from_rows_count": len(extra_from_rows),
    }

def metrics_store_summary(metric_store, run_id: str) -> Dict[str, Any]:
    try:
        g = metric_store.get_or_create_group(run_id)
        meta = dict(g.meta or {})
        mf = meta.get("metric_filter_summary") or {}
        return {
            "metric_filter_kept": len(mf.get("kept") or []),
            "metric_filter_total_raw": mf.get("total_raw"),
            "metric_filter_kept_first20": (mf.get("kept") or [])[:20],
            "total_raw": mf.get("total_raw")
        }
    except Exception as e:
        return {"warning": f"MetricStore summary unavailable: {e}"}

def write_markdown(report_path: str, data: Dict[str, Any]) -> str:
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def j(d): return "```json\n" + json.dumps(d, indent=2) + "\n```"

    md = []
    md.append(f"# Critic Run Report\n\n_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    md.append("## 1) Batch Summary\n")
    md.append(j(data.get("rows_summary", {})))

    md.append("\n## 2) Metric Filter Summary (MetricStore)\n")
    md.append(j(data.get("metric_store_summary", {})))

    md.append("\n## 3) VisiCalc Quality\n")
    md.append(j({"visicalc_quality_mean": data.get("rows_summary", {}).get("visicalc_quality_mean")}))

    md.append("\n## 4) Model Feature Alignment\n")
    md.append(j(data.get("model_alignment", {})))

    md.append("\n## 5) Feature Reports (per feature)\n")
    for rep in data.get("feature_reports", []):
        md.append(f"### {rep.get('name')}\n")
        md.append(j(rep))

    p.write_text("\n".join(md), encoding="utf-8")
    return str(p)

def build_critic_report(
    *,
    rows: List[Dict[str, Any]],
    feature_reports: List[Dict[str, Any]],
    model_feature_names: List[str],
    metric_store,  # MetricStore
    run_id: str,
    out_path: str,
) -> str:
    data = {}
    data["rows_summary"] = summarize_rows(rows)
    data["metric_store_summary"] = metrics_store_summary(metric_store, run_id)
    data["model_alignment"] = compare_model_features(model_feature_names, rows)
    data["feature_reports"] = feature_reports or []
    return write_markdown(out_path, data)
