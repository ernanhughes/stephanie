# !/usr/bin/env python3
# scripts/nexus_ab_smoke.py
"""
A/B smoke checks for nexus_vpm runs.

Usage:
  python tools/smoke_check.py <run_id> \
    [--root runs/nexus_vpm] [--strict] \
    [--json-out diff.json] [--csv-out diff.csv] \
    [--warn-length 1.5] [--warn-uncert 0.3] [--tiny-drop 0.02] [--adv-drop 0.3]

Notes
- Works with your existing run_metrics.json layout.
- Missing metrics are skipped gracefully (reported as "n/a").
- "Improvement %" is direction-aware:
    * higher-is-better:  (t - b) / |b|
    * lower-is-better:   (b - t) / |b|
"""

import argparse, csv, json, sys, math
from pathlib import Path

EPS = 1e-9

# ---- helpers ----------------------------------------------------------------
def dot_get(d, path):
    """Safe dotted lookup: 'a.b.c' -> d['a']['b']['c']; returns None if missing."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def rel_improvement(vb, vt, direction="up"):
    # direction: "up" (higher is better) or "down" (lower is better)
    if vb is None or vt is None:
        return None
    if direction == "up":
        base = abs(vb) if abs(vb) > EPS else (abs(vt) if abs(vt) > EPS else 1.0)
        return (vt - vb) / base
    else:
        base = abs(vb) if abs(vb) > EPS else (abs(vt) if abs(vt) > EPS else 1.0)
        return (vb - vt) / base

def fmt(v, nd=4):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"

def colorize(s, color):
    # minimal ANSI (works on most terminals)
    C = dict(g="\033[92m", y="\033[93m", r="\033[91m", n="\033[0m")
    return f"{C.get(color,'')}{s}{C['n']}"

# ---- metric registry ---------------------------------------------------------
# name, path, direction ("up"=higher is better, "down"=lower is better)
METRICS = [
    ("goal_alignment.mean",      "goal_alignment.mean",        "up"),
    ("goal_alignment.p90",       "goal_alignment.p90",         "up"),
    ("mutual_knn_frac",          "mutual_knn_frac",            "up"),
    ("clustering_coeff",         "clustering_coeff",           "up"),
    ("spatial.mean_edge_len",    "spatial.mean_edge_len",      "down"),
    # Optional extras (skip if absent in your JSONs):
    ("text.len",                 "text.len",                   "down"),   # usually want shorter unless you expect longer
    ("text.words",               "text.words",                 "down"),
    ("sicql.aggregate",          "sicql.aggregate",            "up"),
    ("tiny.aggregate",           "tiny.aggregate",             "up"),
    ("sicql.clarity.uncertainty","sicql.clarity.uncertainty",  "down"),
    ("sicql.coverage.uncertainty","sicql.coverage.uncertainty","down"),
    ("sicql.faithfulness.uncertainty","sicql.faithfulness.uncertainty","down"),
    ("sicql.clarity.advantage",  "sicql.clarity.advantage",    "up"),     # less negative → higher is better
    ("sicql.coverage.advantage", "sicql.coverage.advantage",   "up"),
    ("sicql.faithfulness.advantage","sicql.faithfulness.advantage","up"),
]

# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--root", default="runs/nexus_vpm")
    ap.add_argument("--json-out")
    ap.add_argument("--csv-out")
    ap.add_argument("--strict", action="store_true", help="non-zero exit if any guard fails")
    # guard-rail thresholds
    ap.add_argument("--warn-length", type=float, default=1.50, help="t_len/b_len ratio warn threshold")
    ap.add_argument("--warn-uncert", type=float, default=0.30, help="allowed +Δ in uncertainty (absolute)")
    ap.add_argument("--tiny-drop",   type=float, default=0.02, help="allowed relative drop in tiny.aggregate")
    ap.add_argument("--adv-drop",    type=float, default=0.30, help="allowed drop (absolute) in advantage (more negative)")
    ap.add_argument("--knn-drop", type=float, default=0.05,
                    help="allowed relative drop in mutual_knn_frac (e.g., 0.05 = 5%)")
    ap.add_argument("--cluster-drop", type=float, default=0.08,
                    help="allowed relative drop in clustering_coeff")
    ap.add_argument("--spatial-blowup", type=float, default=0.20,
                    help="allowed relative worsening in spatial.mean_edge_len (down is better)")
    ap.add_argument("--fail-on-zero-ga", action="store_true",
                    help="fail if goal_alignment.mean & .p90 are zero for both base and target")
    args = ap.parse_args()

    root = Path(args.root)
    rid = args.run_id
    b = json.load(open(root/f"{rid}-baseline/run_metrics.json"))
    t = json.load(open(root/f"{rid}-targeted/run_metrics.json"))

    rows = []
    print()
    print(colorize("=== Metric deltas (direction-aware) ===", "g"))
    for name, path, direction in METRICS:
        vb = dot_get(b, path)
        vt = dot_get(t, path)
        imp = rel_improvement(vb, vt, direction)
        # Keep your original spatial-line style for familiarity
        if "spatial.mean_edge_len" in name:
            # you previously inverted this one; our rel_improvement already handles it.
            pass
        # Print line
        base_s = fmt(vb)
        targ_s = fmt(vt)
        if imp is None:
            delta_s = "n/a"
        else:
            delta_s = f"{imp*100:+.1f}%"
        print(f"{name:28s}  base={base_s:>8s}  targ={targ_s:>8s}  Δ%={delta_s}")
        rows.append(dict(name=name, path=path, base=vb, targ=vt, rel_impr=imp))

    # ---- guard-rails / sanity checks ----------------------------------------
    print()
    print(colorize("=== Guard-rails ===", "g"))
    failures = []

    def guard(cond, ok_msg, fail_msg):
        if cond:
            print(colorize(f"✔ {ok_msg}", "g"))
        else:
            print(colorize(f"✖ {fail_msg}", "r"))
            failures.append(fail_msg)

    # 1) Length blowup (if text.len present)
    b_len = dot_get(b, "text.len")
    t_len = dot_get(t, "text.len")
    if b_len is not None and t_len is not None and b_len > EPS:
        ratio = t_len / max(EPS, b_len)
        guard(ratio <= args.warn_length,
              f"Length ratio OK: {ratio:.2f} (≤ {args.warn_length})",
              f"Length ratio high: {ratio:.2f} (> {args.warn_length})")
    else:
        print(colorize("• Length ratio: n/a (missing text.len)", "y"))

    # 2) Tiny drop (if tiny.aggregate present)
    b_tiny = dot_get(b, "tiny.aggregate")
    t_tiny = dot_get(t, "tiny.aggregate")
    if b_tiny is not None and t_tiny is not None:
        # relative change (negative is bad)
        rel = (t_tiny - b_tiny) / (abs(b_tiny) + EPS)
        guard(rel >= -args.tiny_drop,
              f"Tiny aggregate OK: Δ={rel*100:+.1f}% (≥ -{args.tiny_drop*100:.0f}%)",
              f"Tiny aggregate dropped: Δ={rel*100:+.1f}% (< -{args.tiny_drop*100:.0f}%)")
    else:
        print(colorize("• Tiny aggregate: n/a", "y"))

    # 3) Uncertainty rise (sicql.*.uncertainty)
    for head in ("clarity", "coverage", "faithfulness"):
        b_u = dot_get(b, f"sicql.{head}.uncertainty")
        t_u = dot_get(t, f"sicql.{head}.uncertainty")
        if b_u is None or t_u is None:
            print(colorize(f"• sicql.{head}.uncertainty: n/a", "y"))
            continue
        rise = (t_u - b_u)
        guard(rise <= args.warn_uncert,
              f"Uncertainty {head} OK: +{rise:.2f} (≤ {args.warn_uncert:.2f})",
              f"Uncertainty {head} rose: +{rise:.2f} (> {args.warn_uncert:.2f})")

    # 4) Advantage drift more negative (sicql.*.advantage)
    for head in ("clarity", "coverage", "faithfulness"):
        b_a = dot_get(b, f"sicql.{head}.advantage")
        t_a = dot_get(t, f"sicql.{head}.advantage")
        if b_a is None or t_a is None:
            print(colorize(f"• sicql.{head}.advantage: n/a", "y"))
            continue
        drop = b_a - t_a  # positive if got more negative
        guard(drop <= args.adv_drop,
              f"Advantage {head} OK: Δ={-drop:+.2f} (not overly negative)",
              f"Advantage {head} more negative by {drop:.2f} (> {args.adv_drop:.2f})")

    # 5) Saturation check (sicql.aggregate ~ 100)
    s_base = dot_get(b, "sicql.aggregate")
    s_targ = dot_get(t, "sicql.aggregate")
    if s_base is not None and s_targ is not None:
        if s_base >= 99.0 and s_targ >= 99.0:
            print(colorize("• Note: sicql.aggregate appears saturated (~100). Gains may be diminishing.", "y"))
        else:
            print(colorize("✔ SICQL aggregate not saturated.", "g"))
    else:
        print(colorize("• SICQL aggregate: n/a", "y"))

    # ---- outputs -------------------------------------------------------------
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(colorize(f"\nWrote JSON diff → {args.json_out}", "g"))

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name","path","base","targ","rel_impr"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(colorize(f"Wrote CSV diff  → {args.csv_out}", "g"))

    # ---- summary / exit ------------------------------------------------------
    if failures:
        print(colorize(f"\nGuard checks failed: {len(failures)}", "r"))
        for fmsg in failures:
            print(colorize(f" - {fmsg}", "r"))
        if args.strict:
            sys.exit(1)
    else:
        print(colorize("\nAll guard checks passed.", "g"))

if __name__ == "__main__":
    main()
