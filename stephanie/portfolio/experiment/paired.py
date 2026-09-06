# stephanie/portfolio/experiment/paired.py
"""Paired-case analysis for Corpus v1 (§12–§15): recall, clean-twin
specificity, differential detection, role table, VALIDATED verdict."""
from __future__ import annotations

from typing import Sequence

from stephanie.portfolio.experiment.arm import ExperimentArm
from stephanie.portfolio.experiment.finding import FindingClass

TP = FindingClass.TRUE_POSITIVE
FP = FindingClass.FALSE_POSITIVE


def _defect_code(case) -> str | None:
    codes = [e.code for e in case.expected]
    return codes[0] if codes else None


def paired_analysis(records: list[dict], cases: list) -> dict:
    by_case = {c.case_id: c for c in cases}
    pairs: dict[str, dict] = {}
    for case in cases:
        pair_id = (case.metadata or {}).get("pair_id", case.case_id)
        entry = pairs.setdefault(pair_id, {"defect": None, "twin": None, "trap": None})
        if (case.metadata or {}).get("twin_of") and not (case.metadata or {}).get("defect_present"):
            entry["twin"] = case
        elif (case.metadata or {}).get("defect_present"):
            entry["defect"] = case
        else:
            entry["trap"] = case

    rec_by_case_arm = {(r["case_id"], r["arm"]): r for r in records}

    def tp_on(findings, code):
        return any(f["classification"] == TP.value and f["matched_code"] == code for f in findings)

    def fp_on(findings, code):
        return any(f["classification"] == FP.value and f["matched_code"] == code for f in findings)

    arms = [a.value for a in ExperimentArm]
    recall: dict[str, float | None] = {}
    clean_fp: dict[str, float | None] = {}
    dd: dict[str, float | None] = {}
    unique_vs_a: dict[str, int] = {}
    pair_rows: list[dict] = []

    real_pairs = {pid: p for pid, p in pairs.items() if p["defect"] is not None}
    for arm in arms:
        dets = fps = n = 0
        for pid, pair in real_pairs.items():
            defect_code = _defect_code(pair["defect"])
            d_rec = rec_by_case_arm.get((pair["defect"].case_id, arm))
            t_rec = rec_by_case_arm.get((pair["twin"].case_id, arm))
            if d_rec is None or t_rec is None:
                continue
            n += 1
            dets += 1 if tp_on(d_rec["findings"], defect_code) else 0
            fps += 1 if fp_on(t_rec["findings"], defect_code) else 0
        recall[arm] = dets / n if n else None
        clean_fp[arm] = fps / n if n else None
        dd[arm] = (recall[arm] - clean_fp[arm]) if (recall[arm] is not None and clean_fp[arm] is not None) else None

    # Unique defects vs A (per case, defect code TP in arm but not in A).
    a_tps = {}
    for pid, pair in real_pairs.items():
        a_rec = rec_by_case_arm.get((pair["defect"].case_id, ExperimentArm.A_PRIMARY_ONLY.value))
        a_tps[pid] = {f["matched_code"] for f in (a_rec["findings"] if a_rec else [])
                      if f["classification"] == TP.value and f["matched_code"]} if a_rec else set()
    for arm in arms:
        if arm == ExperimentArm.A_PRIMARY_ONLY.value:
            unique_vs_a[arm] = 0
            continue
        unique_codes: set[tuple[str, str]] = set()
        for pid, pair in real_pairs.items():
            rec = rec_by_case_arm.get((pair["defect"].case_id, arm))
            if rec is None:
                continue
            mine = {f["matched_code"] for f in rec["findings"]
                    if f["classification"] == TP.value and f["matched_code"]}
            for code in mine - a_tps.get(pid, set()):
                unique_codes.add((pid, code))
        unique_vs_a[arm] = len(unique_codes)

    for pid, pair in real_pairs.items():
        defect_code = _defect_code(pair["defect"])
        row = {"pair": pid, "code": defect_code}
        for arm in arms:
            d_rec = rec_by_case_arm.get((pair["defect"].case_id, arm))
            t_rec = rec_by_case_arm.get((pair["twin"].case_id, arm))
            row[arm + "_defect"] = (
                "TP" if (d_rec and tp_on(d_rec["findings"], defect_code))
                else ("miss" if d_rec else "?"))
            row[arm + "_twin"] = (
                "FP" if (t_rec and fp_on(t_rec["findings"], defect_code))
                else ("clean" if t_rec else "?"))
        pair_rows.append(row)

    # Role table (additional roles only).
    role_rows: list[dict] = []
    for role in ("critic", "independent_reviewer", "breadth"):
        unique: set[tuple[str, str]] = set()
        dups = fps_clean = recovered = 0
        missed_a = missed_both = 0
        for pid, pair in real_pairs.items():
            defect_code = _defect_code(pair["defect"])
            a_rec = rec_by_case_arm.get((pair["defect"].case_id, ExperimentArm.A_PRIMARY_ONLY.value))
            a_hit = bool(a_rec and tp_on(a_rec["findings"], defect_code))
            role_hit = role_fp = False
            for arm in arms:
                if arm == ExperimentArm.A_PRIMARY_ONLY.value:
                    continue
                d_rec = rec_by_case_arm.get((pair["defect"].case_id, arm))
                t_rec = rec_by_case_arm.get((pair["twin"].case_id, arm))
                if d_rec:
                    for f in d_rec["findings"]:
                        if d_rec.get("candidate_roles", {}).get(f.get("candidate_id")) != role:
                            continue
                        if f["classification"] == TP.value and f["matched_code"] == defect_code:
                            role_hit = True
                            if not a_hit:
                                unique.add((pid, defect_code))
                            else:
                                dups += 1
                if t_rec:
                    for f in t_rec["findings"]:
                        if t_rec.get("candidate_roles", {}).get(f.get("candidate_id")) != role:
                            continue
                        if f["classification"] == FP.value and f["matched_code"] == defect_code:
                            role_fp = True
            if role_hit:
                recovered += 0  # counted in unique/dups above
            fps_clean += 1 if role_fp else 0
            if not a_hit:
                missed_a += 1
                if not role_hit:
                    missed_both += 1
        role_rows.append({
            "role": role,
            "unique_valid_defects": len(unique),
            "duplicate_findings": dups,
            "clean_twin_fps": fps_clean,
            "primary_misses_recovered": len(unique),
            "failure_overlap": (missed_both / missed_a) if missed_a else None,
        })

    # A-identity: primary leg identical across arms per case.
    identity_violations = []
    for case in cases:
        ids = {r["primary_request_id"] for r in records if r["case_id"] == case.case_id}
        if len(ids) > 1:
            identity_violations.append(case.case_id)

    raw_missing = [r["run_id"] for r in records if not r.get("raw_outputs")]
    return {
        "recall": recall, "clean_fp": clean_fp, "dd": dd,
        "unique_vs_a": unique_vs_a, "pair_rows": pair_rows,
        "role_rows": role_rows, "identity_violations": identity_violations,
        "raw_missing": raw_missing, "n_pairs": len(real_pairs),
    }


def verdict(analysis: dict) -> tuple[str, list[str]]:
    """VALIDATED iff gates pass; reasons listed otherwise."""
    reasons: list[str] = []
    if analysis["identity_violations"]:
        reasons.append(f"A-identity violated: {analysis['identity_violations']}")
    if analysis["raw_missing"]:
        reasons.append(f"raw outputs missing: {len(analysis['raw_missing'])} runs")
    dd_e = (analysis["dd"] or {}).get(ExperimentArm.E_FULL.value)
    dd_a = (analysis["dd"] or {}).get(ExperimentArm.A_PRIMARY_ONLY.value)
    if dd_e is None:
        reasons.append("DD_E unmeasurable")
    elif dd_e <= 0:
        reasons.append(f"DD_E={dd_e:.2f} not positive: portfolio does not discriminate")
    if dd_a is not None and dd_e is not None and dd_e < dd_a:
        reasons.append(f"DD_E={dd_e:.2f} < DD_A={dd_a:.2f}")
    return ("VALIDATED" if not reasons else "INVALID"), reasons


def render_paired_section(analysis: dict) -> str:
    lines = ["PAIRED RESULTS (defect recall | clean-twin FP | differential detection)",
             "----------------------------------"]
    for arm in [a.value for a in ExperimentArm]:
        lines.append(
            f"{arm}  recall={_fmt(analysis['recall'].get(arm))} "
            f"clean_fp={_fmt(analysis['clean_fp'].get(arm))} "
            f"DD={_fmt(analysis['dd'].get(arm))} "
            f"unique_vs_A={analysis['unique_vs_a'].get(arm)}")
    lines.append("")
    lines.append("ROLE TABLE (additional roles)")
    lines.append("----------------------------------")
    for row in analysis["role_rows"]:
        lines.append(
            f"{row['role']:<22} unique={row['unique_valid_defects']} "
            f"dups={row['duplicate_findings']} cleanFPs={row['clean_twin_fps']} "
            f"recovered={row['primary_misses_recovered']} "
            f"overlap={_fmt(row['failure_overlap'])}")
    lines.append("")
    lines.append("PER-PAIR (defect_variant | clean_twin)")
    lines.append("----------------------------------")
    for row in analysis["pair_rows"]:
        cells = " ".join(f"{arm}:{row[arm + '_defect']}/{row[arm + '_twin']}"
                         for arm in [a.value for a in ExperimentArm])
        lines.append(f"{row['pair']} {row['code']}  {cells}")
    return "\n".join(lines)


def _fmt(value) -> str:
    return f"{value:.2f}" if value is not None else "n/a"
