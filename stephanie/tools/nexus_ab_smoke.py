#!/usr/bin/env python3
"""
A/B smoke checks for nexus_vpm runs + NexusImprover integration.

Usage:
  python tools/nexus_ab_smoke.py <run_id> \
    [--root runs/nexus_vpm] [--strict] \
    [--json-out diff.json] [--csv-out diff.csv] [--md-out summary.md] \
    [--warn-length 1.5] [--warn-uncert 0.3] [--tiny-drop 0.02] [--adv-drop 0.3] \
    [--knn-drop 0.05] [--cluster-drop 0.08] [--spatial-blowup 0.20] \
    [--fail-on-zero-ga] \
    [--nexus-a runs/run-123/nexus_improver_report.json] \
    [--nexus-b runs/run-456/nexus_improver_report.json]

Notes
- Backward compatible with your existing run_metrics.json layout:
    <root>/<run_id>-baseline/run_metrics.json
    <root>/<run_id>-targeted/run_metrics.json
- If the above is missing per-metric aggregates, this script will
  auto-aggregate from features.jsonl written by ScorableProcessor.
- NexusImprover is optional. If provided, we compare:
    win_rate, mean_lift, topk_lift, diversity pre/post, novelty retention.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from math import isfinite

EPS = 1e-9

# ---------- helpers -----------------------------------------------------------

def dot_get(d, path, stat="mean"):
    """
    Try root-level dotted path first; if missing, try metric_columns.<path>.<stat>.
    Returns None if not found.
    """
    # 1) root-level dotted
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            cur = None
            break
        cur = cur[part]
    if cur is not None:
        return cur

    # 2) metric_columns fallback (aggregated)
    mc = d.get("metric_columns")
    if isinstance(mc, dict):
        col = mc.get(path)
        if isinstance(col, dict) and stat in col:
            return col.get(stat)
    return None

def rel_improvement(vb, vt, direction="up"):
    if vb is None or vt is None:
        return None
    base = abs(vb) if abs(vb) > EPS else (abs(vt) if abs(vt) > EPS else 1.0)
    if direction == "up":
        return (vt - vb) / base
    else:
        return (vb - vt) / base

def fmt(v, nd=4):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"

def colorize(s, color):
    C = dict(g="\033[92m", y="\033[93m", r="\033[91m", n="\033[0m")
    return f"{C.get(color,'')}{s}{C['n']}"

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def latest_nexus_report_under(runs_root: Path):
    # pick the most-recent run-*/nexus_improver_report.json
    cand = sorted(runs_root.glob("run-*/nexus_improver_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0] if cand else None

def safe_mean(lst):
    lst = [x for x in lst if isinstance(x, (int, float)) and isfinite(x)]
    return (sum(lst) / max(1, len(lst))) if lst else None

def p90(lst):
    lst = sorted(x for x in lst if isinstance(x, (int, float)) and isfinite(x))
    if not lst:
        return None
    idx = int(0.9 * (len(lst)-1))
    return lst[idx]

# ---------- metric registry (unchanged, can extend) --------------------------

# name, path, direction, stat
METRICS = [
    ("goal_alignment.mean",      "goal_alignment.mean",        "up",  "mean"),  # root
    ("goal_alignment.p90",       "goal_alignment.p90",         "up",  "p90"),   # root
    ("mutual_knn_frac",          "mutual_knn_frac",            "up",  "mean"),  # root
    ("clustering_coeff",         "clustering_coeff",           "up",  "mean"),  # root
    ("spatial.mean_edge_len",    "spatial.mean_edge_len",      "down","mean"),  # root

    # these come from per-item features via ScorableProcessor (we aggregate here)
    ("text.len",                 "text.len",                   "down","mean"),
    ("text.words",               "text.words",                 "down","mean"),
    ("sicql.aggregate",          "sicql.aggregate",            "up",  "mean"),
    ("tiny.aggregate",           "tiny.aggregate",             "up",  "mean"),
    ("sicql.clarity.uncertainty","sicql.clarity.attr.uncertainty","down","mean"),
    ("sicql.coverage.uncertainty","sicql.coverage.attr.uncertainty","down","mean"),
    ("sicql.faithfulness.uncertainty","sicql.faithfulness.attr.uncertainty","down","mean"),
    ("sicql.clarity.advantage",  "sicql.clarity.attr.advantage","up", "mean"),   # less negative → “up”
    ("sicql.coverage.advantage", "sicql.coverage.attr.advantage","up","mean"),
    ("sicql.faithfulness.advantage","sicql.faithfulness.attr.advantage","up","mean"),
]

# ---------- features.jsonl aggregation (NEW) ---------------------------------

def _find_features_file(run_arm_dir: Path) -> Path | None:
    """
    Try common locations; otherwise first recursive match under the arm directory.
    """
    for p in [
        run_arm_dir / "features.jsonl",
        run_arm_dir / "manifest" / "features.jsonl",
    ]:
        if p.exists():
            return p
    matches = list(run_arm_dir.rglob("features.jsonl"))
    return matches[0] if matches else None

def _iter_features_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _collect_metric_samples(features_path: Path) -> dict[str, list[float]]:
    """
    Build {metric_name -> [values...]} from features.jsonl rows.
    - Supports either metrics_vector dict or (metrics_columns, metrics_values) arrays.
    - Also derives text.len and text.words from the row's text.
    """
    pool: dict[str, list[float]] = {}
    for row in _iter_features_rows(features_path):
        # 1) metrics vector
        vec = row.get("metrics_vector")
        if not isinstance(vec, dict):
            cols = row.get("metrics_columns") or []
            vals = row.get("metrics_values") or []
            if isinstance(cols, list) and isinstance(vals, list) and len(cols) == len(vals):
                vec = {c: v for c, v in zip(cols, vals)}
            else:
                vec = {}

        for k, v in vec.items():
            if isinstance(v, (int, float)) and isfinite(v):
                pool.setdefault(k, []).append(float(v))

        # 2) cheap text features
        text = row.get("text") or ""
        pool.setdefault("text.len", []).append(float(len(text)))
        pool.setdefault("text.words", []).append(float(len(text.split())))

    return pool

def augment_with_features(run_arm_dir: Path, metrics_dict: dict, wanted=METRICS):
    """
    If metrics_dict lacks metric_columns stats for wanted metrics, compute means/p90 from features.jsonl.
    Injects them into metrics_dict['metric_columns'][name] = {'mean': ..., 'p90': ...}
    """
    fpath = _find_features_file(run_arm_dir)
    if not fpath:
        # soft warning only
        print(colorize(f"• features.jsonl not found under {run_arm_dir}", "y"))
        return

    samples = _collect_metric_samples(fpath)
    mc = metrics_dict.setdefault("metric_columns", {})

    for (_name, path, _direction, stat) in wanted:
        # if already present at root OR metric_columns, skip
        if dot_get(metrics_dict, path, stat) is not None:
            continue
        vals = samples.get(path)
        if not vals:
            continue
        mc[path] = {
            "mean": safe_mean(vals),
            "p90": p90(vals),
        }

# ---------- NexusImprover diff helpers ---------------------------------------

def load_nexus_report(path_or_none: str):
    if not path_or_none:
        return None
    p = Path(path_or_none)
    if p.is_dir():
        p = p / "nexus_improver_report.json"
    return load_json(p) if p.exists() else None

def nexus_summary(nr: dict):
    if not isinstance(nr, dict):
        return None
    dec = nr.get("decisions") or []
    ratios = []
    for d in dec:
        k  = d.get("k_generated")
        kp = d.get("k_after_novelty")
        if isinstance(k, (int, float)) and k > 0 and isinstance(kp, (int, float)):
            ratios.append(kp / float(k))
    return dict(
        win_rate = nr.get("win_rate"),
        mean_lift = nr.get("mean_lift"),
        topk_lift = nr.get("topk_lift"),
        diversity_raw = nr.get("mean_blossom_diversity_raw"),
        diversity_post = nr.get("mean_blossom_diversity_post"),
        novelty_retention = safe_mean(ratios),
        decision_count = nr.get("decision_count"),
        goal_preview = nr.get("goal_preview"),
    )

def rel(vb, vt):
    if vb is None or vt is None:
        return None
    base = abs(vb) if abs(vb) > EPS else (abs(vt) if abs(vt) > EPS else 1.0)
    return (vt - vb) / base

# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--root", default="runs/nexus_vpm")
    ap.add_argument("--json-out")
    ap.add_argument("--csv-out")
    ap.add_argument("--md-out")
    ap.add_argument("--strict", action="store_true", help="non-zero exit if any guard fails")
    # guard-rail thresholds
    ap.add_argument("--warn-length", type=float, default=1.50)
    ap.add_argument("--warn-uncert", type=float, default=0.30)
    ap.add_argument("--tiny-drop",   type=float, default=0.02)
    ap.add_argument("--adv-drop",    type=float, default=0.30)
    ap.add_argument("--knn-drop",    type=float, default=0.05)
    ap.add_argument("--cluster-drop",type=float, default=0.08)
    ap.add_argument("--spatial-blowup", type=float, default=0.20)
    ap.add_argument("--fail-on-zero-ga", action="store_true")
    # NexusImprover inputs (optional)
    ap.add_argument("--nexus-a", help="Path to baseline NexusImprover report or its run dir (runs/run-*)")
    ap.add_argument("--nexus-b", help="Path to targeted NexusImprover report or its run dir (runs/run-*)")
    ap.add_argument("--nexus-autodiscover", action="store_true",
                    help="If set and --nexus-a/--nexus-b missing, pick two latest runs under runs/")
    args = ap.parse_args()

    root = Path(args.root)
    rid = args.run_id
    base_dir = root / f"{rid}-baseline"
    targ_dir = root / f"{rid}-targeted"

    base_metrics = load_json(base_dir / "run_metrics.json") or {}
    targ_metrics = load_json(targ_dir / "run_metrics.json") or {}

    if not base_metrics and not targ_metrics:
        print(colorize("Missing run_metrics.json for both arms. Check --root and run_id.", "r"))
        sys.exit(2)

    # NEW: augment missing stats from features.jsonl (ScorableProcessor output)
    augment_with_features(base_dir, base_metrics, METRICS)
    augment_with_features(targ_dir, targ_metrics, METRICS)

    # ----- METRIC DELTAS -----------------------------------------------------
    rows = []
    print()
    print(colorize("=== Metric deltas (direction-aware) ===", "g"))
    for name, path, direction, stat in METRICS:
        vb = dot_get(base_metrics, path, stat)
        vt = dot_get(targ_metrics, path, stat)
        imp = rel_improvement(vb, vt, direction)
        base_s, targ_s = fmt(vb), fmt(vt)
        delta_s = "n/a" if imp is None else f"{imp*100:+.1f}%"
        print(f"{name:28s}  base={base_s:>8s}  targ={targ_s:>8s}  Δ%={delta_s}")
        rows.append(dict(name=name, path=path, base=vb, targ=vt, rel_impr=imp))

    # ----- GUARD-RAILS (original) -------------------------------------------
    print()
    print(colorize("=== Guard-rails ===", "g"))
    failures = []

    def guard(cond, ok_msg, fail_msg):
        if cond:
            print(colorize(f"✔ {ok_msg}", "g"))
        else:
            print(colorize(f"✖ {fail_msg}", "r"))
            failures.append(fail_msg)

    # Length blowup
    b_len = dot_get(base_metrics, "text.len"); t_len = dot_get(targ_metrics, "text.len")
    if b_len is not None and t_len is not None and b_len > EPS:
        ratio = t_len / max(EPS, b_len)
        guard(ratio <= args.warn_length,
              f"Length ratio OK: {ratio:.2f} (≤ {args.warn_length})",
              f"Length ratio high: {ratio:.2f} (> {args.warn_length})")
    else:
        print(colorize("• Length ratio: n/a (missing text.len)", "y"))

    # Tiny drop
    b_tiny = dot_get(base_metrics, "tiny.aggregate")
    t_tiny = dot_get(targ_metrics, "tiny.aggregate")
    if b_tiny is not None and t_tiny is not None:
        rel_tiny = (t_tiny - b_tiny) / (abs(b_tiny) + EPS)
        guard(rel_tiny >= -args.tiny_drop,
              f"Tiny aggregate OK: Δ={rel_tiny*100:+.1f}%",
              f"Tiny aggregate dropped: Δ={rel_tiny*100:+.1f}% (< -{args.tiny_drop*100:.0f}%)")
    else:
        print(colorize("• Tiny aggregate: n/a", "y"))

    # Uncertainty rise
    for head in ("clarity", "coverage", "faithfulness"):
        b_u = dot_get(base_metrics, f"sicql.{head}.uncertainty")
        t_u = dot_get(targ_metrics, f"sicql.{head}.uncertainty")
        if b_u is None or t_u is None:
            print(colorize(f"• sicql.{head}.uncertainty: n/a", "y"))
            continue
        rise = (t_u - b_u)
        guard(rise <= args.warn_uncert,
              f"Uncertainty {head} OK: +{rise:.2f} (≤ {args.warn_uncert:.2f})",
              f"Uncertainty {head} rose: +{rise:.2f} (> {args.warn_uncert:.2f})")

    # Advantage drift
    for head in ("clarity", "coverage", "faithfulness"):
        b_a = dot_get(base_metrics, f"sicql.{head}.advantage")
        t_a = dot_get(targ_metrics, f"sicql.{head}.advantage")
        if b_a is None or t_a is None:
            print(colorize(f"• sicql.{head}.advantage: n/a", "y"))
            continue
        drop = b_a - t_a  # positive if got more negative
        guard(drop <= args.adv_drop,
              f"Advantage {head} OK: Δ={-drop:+.2f}",
              f"Advantage {head} more negative by {drop:.2f} (> {args.adv_drop:.2f})")

    # Saturation note
    s_base = dot_get(base_metrics, "sicql.aggregate")
    s_targ = dot_get(targ_metrics, "sicql.aggregate")
    if s_base is not None and s_targ is not None:
        if s_base >= 99.0 and s_targ >= 99.0:
            print(colorize("• Note: sicql.aggregate appears saturated (~100).", "y"))
        else:
            print(colorize("✔ SICQL aggregate not saturated.", "g"))
    else:
        print(colorize("• SICQL aggregate: n/a", "y"))

    # KNN/cluster/spatial checks
    b_knn, t_knn = dot_get(base_metrics, "mutual_knn_frac"), dot_get(targ_metrics, "mutual_knn_frac")
    if b_knn is not None and t_knn is not None and abs(b_knn) > EPS:
        knn_rel = (t_knn - b_knn) / abs(b_knn)
        guard(knn_rel >= -args.knn_drop,
              f"mutual_knn_frac OK: Δ={knn_rel*100:+.1f}%",
              f"mutual_knn_frac dropped: Δ={knn_rel*100:+.1f}% (< -{args.knn_drop*100:.0f}%)")
    else:
        print(colorize("• mutual_knn_frac: n/a", "y"))

    b_cc, t_cc = dot_get(base_metrics, "clustering_coeff"), dot_get(targ_metrics, "clustering_coeff")
    if b_cc is not None and t_cc is not None and abs(b_cc) > EPS:
        cc_rel = (t_cc - b_cc) / abs(b_cc)
        guard(cc_rel >= -args.cluster_drop,
              f"clustering_coeff OK: Δ={cc_rel*100:+.1f}%",
              f"clustering_coeff dropped: Δ={cc_rel*100:+.1f}% (< -{args.cluster_drop*100:.0f}%)")
    else:
        print(colorize("• clustering_coeff: n/a", "y"))

    b_sp, t_sp = dot_get(base_metrics, "spatial.mean_edge_len"), dot_get(targ_metrics, "spatial.mean_edge_len")
    if b_sp is not None and t_sp is not None and abs(b_sp) > EPS:
        # down is better; relative worsening is (t - b)/|b|
        sp_rel = (t_sp - b_sp) / abs(b_sp)
        guard(sp_rel <= args.spatial_blowup,
              f"spatial.mean_edge_len OK: Δ={sp_rel*100:+.1f}% (≤ {args.spatial_blowup*100:.0f}%)",
              f"spatial.mean_edge_len worsened: Δ={sp_rel*100:+.1f}% (> {args.spatial_blowup*100:.0f}%)")
    else:
        print(colorize("• spatial.mean_edge_len: n/a", "y"))

    if args.fail_on_zero_ga:
        ga0 = (dot_get(base_metrics,"goal_alignment.mean") in (0,0.0) and
               dot_get(base_metrics,"goal_alignment.p90") in (0,0.0) and
               dot_get(targ_metrics,"goal_alignment.mean") in (0,0.0) and
               dot_get(targ_metrics,"goal_alignment.p90") in (0,0.0))
        guard(not ga0, "Goal alignment not zeroed.", "Goal alignment both zero (base & targ).")

    # ----- NexusImprover (optional) -----------------------------------------
    nexus_a = load_nexus_report(args.nexus_a) if args.nexus_a else None
    nexus_b = load_nexus_report(args.nexus_b) if args.nexus_b else None

    if args.nexus_autodiscover and (nexus_a is None or nexus_b is None):
        nr = Path("runs")
        newest = sorted(nr.glob("run-*/nexus_improver_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(newest) >= 2 and nexus_b is None:
            nexus_b = load_json(newest[0])
            nexus_a = load_json(newest[1])

    if nexus_a or nexus_b:
        print()
        print(colorize("=== NexusImprover summary ===", "g"))
        sa = nexus_summary(nexus_a) if nexus_a else None
        sb = nexus_summary(nexus_b) if nexus_b else None

        def line(label, key, nd=4):
            va = None if not sa else sa.get(key)
            vb = None if not sb else sb.get(key)
            relv = rel(va, vb)
            rels = "n/a" if relv is None else f"{relv*100:+.1f}%"
            print(f"{label:22s}  A={fmt(va,nd):>8s}  B={fmt(vb,nd):>8s}  Δ%={rels}")

        line("win_rate", "win_rate", 2)
        line("mean_lift", "mean_lift", 4)
        line("topk_lift", "topk_lift", 4)
        line("diversity_raw", "diversity_raw", 4)
        line("diversity_post", "diversity_post", 4)
        line("novelty_retention", "novelty_retention", 4)

        # Nexus-specific guard-rails (soft, unless --strict)
        if sa and sb:
            wr_a, wr_b = sa.get("win_rate"), sb.get("win_rate")
            ga_base = dot_get(base_metrics, "goal_alignment.mean")
            ga_targ = dot_get(targ_metrics, "goal_alignment.mean")

            if wr_a is not None and wr_b is not None:
                guard(wr_b >= max(0.0, wr_a - 0.05),
                      f"B win_rate OK: {wr_b:.2%} (A={wr_a:.2%})",
                      f"B win_rate regressed: {wr_b:.2%} < A={wr_a:.2%} (tolerance 5pp)")

            # If GA improved materially, topk_lift should not be negative
            if ga_base is not None and ga_targ is not None and (ga_targ - ga_base) > 0.05:
                tkl = sb.get("topk_lift")
                if tkl is not None:
                    guard(tkl >= -0.001,
                          f"topk_lift sane under GA↑: {tkl:+.4f}",
                          f"topk_lift negative ({tkl:+.4f}) despite GA↑ (>5pp)")

    # ----- outputs -----------------------------------------------------------
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(colorize(f"\nWrote JSON diff → {args.json_out}", "g"))

    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["name","path","base","targ","rel_impr"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(colorize(f"Wrote CSV diff  → {args.csv_out}", "g"))

    if args.md_out:
        md = []
        md.append(f"# A/B Smoke — run_id `{rid}`\n")
        md.append("## Headline deltas\n")
        for r in rows:
            name = r["name"]; imp = r["rel_impr"]
            if imp is None: continue
            md.append(f"- **{name}**: Δ {imp*100:+.1f}%")
        if nexus_a or nexus_b:
            md.append("\n## NexusImprover\n")
            sa = nexus_summary(nexus_a) if nexus_a else {}
            sb = nexus_summary(nexus_b) if nexus_b else {}
            def mline(lbl, key):
                relv = rel(sa.get(key), sb.get(key))
                md.append(f"- **{lbl}**: A={fmt(sa.get(key))} → B={fmt(sb.get(key))} (Δ%={fmt(None if relv is None else relv*100, 1)})")
            for k, lbl in [
                ("win_rate","Win rate"), ("mean_lift","Mean lift"),
                ("topk_lift","Top-K lift"), ("novelty_retention","Novelty retention"),
                ("diversity_raw","Diversity (pre)"), ("diversity_post","Diversity (post)")
            ]:
                mline(lbl, k)
        Path(args.md_out).write_text("\n".join(md), encoding="utf-8")
        print(colorize(f"Wrote Markdown  → {args.md_out}", "g"))

    # ----- summary / exit ----------------------------------------------------
    if ('failures' in locals()) and failures and args.strict:
        print(colorize(f"\nGuard checks failed: {len(failures)}", "r"))
        for fmsg in failures:
            print(colorize(f" - {fmsg}", "r"))
        sys.exit(1)
    elif ('failures' in locals()) and failures:
        print(colorize(f"\nGuard checks failed (non-strict).", "y"))
    else:
        print(colorize("\nAll guard checks passed.", "g"))

# python stephanie/tools/nexus_ab_smoke.py 8466
if __name__ == "__main__":
    main()
