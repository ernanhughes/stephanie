"""Deterministic post-run readiness evaluation; separate from generation."""
from __future__ import annotations

import json
from pathlib import Path

from .evolution02q import DATA, TASKS, ReadinessAnswer


def evaluate() -> dict:
    tasks = {t["task_id"]:t for t in json.loads(TASKS.read_text(encoding="utf-8"))}
    rows = []
    for filename, answer_key, result_key in (
        ("readiness-results.json", "answer", None),
        ("readiness-results.json", None, "worker_result"),
        ("worker-1024-results.json", None, "work_result"),
    ):
        path=DATA/filename
        if not path.exists(): continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            is_direct=record["arm"]=="A_DIRECT_AI" if "arm" in record else False
            if answer_key and not is_direct: continue
            if result_key and not isinstance(record.get(result_key),dict): continue
            task=tasks[record["task_id"]]
            if answer_key:
                answer=record.get(answer_key)
                source_status=record.get("status")
            else:
                work=record.get(result_key,{})
                source_status=record.get("status")
                try: answer=json.loads(work.get("output", ""))
                except Exception: answer=None
            shape_valid=False
            if answer is not None:
                try:
                    validated=ReadinessAnswer.model_validate(answer)
                    shape_valid=True
                    answer=validated.model_dump(mode="json")
                except Exception: pass
            allowed={x["evidence_id"] for x in task["observations"]}
            cited=set(answer.get("evidence_ids",[])) if isinstance(answer,dict) else set()
            citations_valid=bool(cited) and cited.issubset(allowed)
            parts=[]
            if isinstance(answer,dict):
                for key in ("findings","unknowns","next_step"):
                    value=answer.get(key,[])
                    parts.extend(map(str,value if isinstance(value,list) else [value]))
            text=" ".join(parts).lower()
            concepts={concept:concept.lower() in text for concept in task["required_concepts"]}
            rows.append({"artifact":filename,"arm":record.get("arm","B_WORKER_ADAPTIVE"),"task_id":record["task_id"],
                "source_status":source_status,"schema_valid":shape_valid,"citations_valid":citations_valid,
                "invalid_citations":sorted(cited-allowed),"required_concepts":concepts,
                "all_required_concepts_present":bool(concepts) and all(concepts.values()),
                "readiness_output_usable":shape_valid and citations_valid})
    report={"scope":"development readiness only","records":rows,
        "summary":{"records":len(rows),"schema_valid":sum(r["schema_valid"] for r in rows),
            "citation_valid":sum(r["citations_valid"] for r in rows),
            "usable_outputs":sum(r["readiness_output_usable"] for r in rows),
            "semantic_checklist_complete":sum(r["readiness_output_usable"] and r["all_required_concepts_present"] for r in rows)},
        "limits":["Required-concept string checks are a narrow deterministic smoke rubric, not semantic correctness evaluation.",
                  "Development prompts are not a protected holdout.","Passing format and citation checks do not establish factual or operational effectiveness."]}
    (DATA/"independent-readiness-evaluation.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(),indent=2,sort_keys=True))
