"""Command line for offline Stephanie improvement investigations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import configure_mappers, sessionmaker

from .engine import ImprovementEngine
from .models import Candidate, Opportunity
from . import evolution02
from . import evolution02q
from .integrations import read_worker_strategies, read_worker_executions, read_mission_episodes, read_mission_signals


def _print(value):
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _with_local_memory(root: Path) -> ImprovementEngine:
    """Wire Stephanie's existing stores to the isolated programme SQLite DB."""
    from stephanie.memory.memory_tool import MemoryTool as _register_stephanie_mappers  # noqa: F401
    from stephanie.memory.casebook_store import CaseBookStore
    from stephanie.memory.experiment_store import ExperimentStore
    from stephanie.orm.base import Base
    from stephanie.orm.casebook import CaseBookORM, CaseORM, CaseScorableORM
    from stephanie.orm.experiment import ExperimentORM, VariantORM, TrialORM, TrialMetricORM

    configure_mappers()
    from .storage import memory_db_path
    db_path = memory_db_path(root, local_name="stephanie-improvement.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = create_engine(f"sqlite:///{db_path.resolve()}")
    tables = [CaseBookORM.__table__, CaseORM.__table__, CaseScorableORM.__table__, ExperimentORM.__table__,
              VariantORM.__table__, TrialORM.__table__, TrialMetricORM.__table__]
    Base.metadata.create_all(db, tables=tables)
    sessions = sessionmaker(bind=db, expire_on_commit=False)
    return ImprovementEngine(root, experiment_store=ExperimentStore(sessions), casebook_store=CaseBookStore(sessions))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m stephanie.improvement", description="Offline, evidence-bounded improvement lifecycle")
    parser.add_argument("--root", default="data/stephanie_evolution01_final", help="isolated evidence and experiment artifacts")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("import-harness", help="import sanitized frozen Harness evidence")
    imp = sub.add_parser("import-evidence", help="import a JSON array of evidence records"); imp.add_argument("path")
    sub.add_parser("discover", help="discover or abstain on opportunities")
    c = sub.add_parser("construct", help="construct the bounded evidence-gate candidate"); c.add_argument("opportunity_id", nargs="?", default="harness-acceptance-evidence")
    v = sub.add_parser("validate", help="validate a persisted candidate"); v.add_argument("candidate_id")
    show = sub.add_parser("inspect", help="inspect the candidate and its baseline/evaluation record"); show.add_argument("candidate_id")
    e = sub.add_parser("evaluate", help="run the immutable offline fixture evaluator"); e.add_argument("candidate_id")
    r = sub.add_parser("retrieve", help="retrieve prior relevant experience"); r.add_argument("query")
    a = sub.add_parser("ai-investigate", help="make one bounded local shared-AI inference call"); a.add_argument("--model", default="ornith-1.5:9b"); a.add_argument("--provider", default="ollama")
    ai_ev = sub.add_parser("ai-evidence", help="inspect shared-AI execution metadata"); ai_ev.add_argument("--include-raw", action="store_true", help="explicitly print locally stored raw response bodies")
    avail = sub.add_parser("integration-check", help="check bounded read-only Worker and Mission interfaces"); avail.add_argument("--worker-url", default="http://127.0.0.1:7317"); avail.add_argument("--mission-url", default="http://127.0.0.1:8790")
    ws = sub.add_parser("worker-strategies", help="read Worker strategy catalogue"); ws.add_argument("--url", default="http://127.0.0.1:7317")
    we = sub.add_parser("worker-executions", help="read bounded Worker execution window"); we.add_argument("--url", default="http://127.0.0.1:7317"); we.add_argument("--capture", action="store_true")
    me = sub.add_parser("mission-evidence", help="read bounded Mission episode and signal references"); me.add_argument("--url", default="http://127.0.0.1:8790"); me.add_argument("--limit", type=int, default=20); me.add_argument("--capture", action="store_true")
    sub.add_parser("propose-worker-diagnostic", help="persist an unapproved local diagnostic experiment proposal")
    mi = sub.add_parser("mission-investigate", help="perform a deterministic source-bound investigation of one Mission finding")
    mi.add_argument("source_snapshot", help="bounded JSON snapshot inside the configured evidence root")
    sub.add_parser("investigate-worker-issue", help="create/retrieve the bounded historical Worker empty-output investigation")
    inv = sub.add_parser("inspect-investigation", help="retrieve a locally retained investigation"); inv.add_argument("investigation_id")
    sub.add_parser("sc3-hypothesis", help="make the once-only bounded shared-AI SC3 hypothesis call")
    sub.add_parser("sc3-freeze", help="freeze the bounded Worker/Praxis JSON normalization experiment")
    sub.add_parser("sc3-run", help="execute frozen experiment through isolated Worker and Praxis")
    sub.add_parser("sc3-live-probe", help="run one bounded local-model readiness check and, if evaluable, one baseline/candidate pair")
    sub.add_parser("sc3-report", help="inspect machine-readable SC3 experiment results")
    sub.add_parser("sc3-recompare", help="reapply Praxis to preserved Worker records without re-executing")
    sub.add_parser("sc3-reconcile", help="finish an interrupted SC3 run only from a complete saved Worker ledger/result set; never dispatch")
    sc3ret = sub.add_parser("sc3-retrieve", help="retrieve retained SC3 experiment experience"); sc3ret.add_argument("query")
    sub.add_parser("se2-freeze", help="freeze SE2 task, memory, model and rubric identities")
    sub.add_parser("se2-run", help="execute the once-only frozen 24-call local comparison")
    sub.add_parser("se2-evaluate", help="score completed SE2 generations against the frozen rubric")
    sub.add_parser("se2q-freeze", help="freeze SE2Q development tasks and execution paths")
    sub.add_parser("se2q-run", help="run the once-only SE2Q direct-AI and Worker readiness comparison")
    sub.add_parser("se2q-report", help="report existing SE2Q readiness evidence")
    rec = sub.add_parser("recommend", help="derive the bounded recommendation from saved independent evaluation"); rec.add_argument("candidate_id")
    ns = parser.parse_args(argv); engine = ImprovementEngine(ns.root)
    if ns.command == "se2q-freeze": _print(evolution02q.freeze())
    elif ns.command == "se2q-run": _print(evolution02q.run())
    elif ns.command == "se2q-report": _print(evolution02q.report())
    elif ns.command == "se2-freeze": _print(evolution02.freeze())
    elif ns.command == "se2-run": _print(evolution02.run())
    elif ns.command == "se2-evaluate": _print(evolution02.evaluate())
    elif ns.command == "import-harness":
        fixture = Path(__file__).with_name("harness_fixtures.json")
        _print([x.model_dump(mode="json") for x in engine.import_json(fixture)])
    elif ns.command == "import-evidence": _print([x.model_dump(mode="json") for x in engine.import_json(ns.path)])
    elif ns.command == "discover": _print([x.model_dump(mode="json") for x in engine.opportunities()])
    elif ns.command == "construct":
        opportunities = {x.opportunity_id: x for x in map(Opportunity.model_validate, engine._read(engine.opportunities_path, []))}
        _print(engine.construct_candidate(opportunities[ns.opportunity_id]).model_dump(mode="json"))
    elif ns.command == "validate":
        candidates = {x["candidate_id"]: Candidate.model_validate(x) for x in engine._read(engine.candidates_path, [])}
        item = candidates[ns.candidate_id]; engine.validate_candidate(item, authorized=True); _print({"valid": True, "candidate": item.model_dump(mode="json")})
    elif ns.command == "inspect":
        candidates = {x["candidate_id"]: x for x in engine._read(engine.candidates_path, [])}
        results = [x for x in engine._read(engine.results_path, []) if x["candidate_id"] == ns.candidate_id]
        _print({"candidate": candidates.get(ns.candidate_id), "experiments": results})
    elif ns.command == "evaluate":
        candidates = {x["candidate_id"]: Candidate.model_validate(x) for x in engine._read(engine.candidates_path, [])}
        engine = _with_local_memory(Path(ns.root).resolve())
        _print(engine.evaluate_fixtures(candidates[ns.candidate_id]).model_dump(mode="json"))
    elif ns.command == "retrieve": _print(_with_local_memory(Path(ns.root).resolve()).retrieve_experience(ns.query))
    elif ns.command == "ai-investigate": _print(engine.investigate_with_ai(ns.provider, ns.model))
    elif ns.command == "ai-evidence":
        records = engine._read(engine.ai_path, [])
        if not ns.include_raw:
            records = [{k: v for k, v in row.items() if k not in {"raw_response", "result"}} | {"result_metadata": {k: (row.get("result") or {}).get(k) for k in ("request_id", "provider", "model", "status", "error", "usage", "latency_ms")}, "raw_response_present": row.get("raw_response") is not None} for row in records]
        _print(records)
    elif ns.command == "integration-check":
        _print({"worker_strategies": read_worker_strategies(ns.worker_url).__dict__, "worker_executions": read_worker_executions(ns.worker_url).__dict__, "mission_episodes": read_mission_episodes(ns.mission_url).__dict__, "mission_signals": read_mission_signals(ns.mission_url).__dict__})
    elif ns.command == "worker-strategies": _print(read_worker_strategies(ns.url).__dict__)
    elif ns.command == "worker-executions":
        from .integrations import as_evidence
        result = read_worker_executions(ns.url)
        imported = engine.import_evidence([as_evidence("worker", result.source, row).model_dump(mode="json") for row in result.records]) if ns.capture and result.records else []
        _print({**result.__dict__, "captured_evidence_ids": [x.evidence_id for x in imported]})
    elif ns.command == "mission-evidence":
        from .integrations import as_evidence
        episodes = read_mission_episodes(ns.url, limit=ns.limit)
        signals = read_mission_signals(ns.url, limit=ns.limit)
        records = [("mission_episode", episodes, row) for row in episodes.records] + [("mission_signal", signals, row) for row in signals.records]
        imported = engine.import_evidence([as_evidence(kind, result.source, row).model_dump(mode="json") for kind, result, row in records]) if ns.capture else []
        _print({"episodes": episodes.__dict__, "signals": signals.__dict__, "captured_evidence_ids": [x.evidence_id for x in imported]})
    elif ns.command == "propose-worker-diagnostic": _print(engine.propose_worker_diagnostic())
    elif ns.command == "mission-investigate":
        root = Path(ns.root).resolve()
        source = Path(ns.source_snapshot).resolve()
        if not source.is_relative_to(root) or not source.is_file() or source.stat().st_size > 24_000:
            raise SystemExit("Mission source snapshot must be a file under the evidence root and no larger than 24 KB")
        engine = _with_local_memory(root)
        _print(engine.investigate_mission_observation(json.loads(source.read_text(encoding="utf-8"))))
    elif ns.command == "investigate-worker-issue": _print(_with_local_memory(Path(ns.root).resolve()).investigate_worker_empty_output())
    elif ns.command == "inspect-investigation": _print(engine.retrieve_investigation(ns.investigation_id))
    elif ns.command == "sc3-hypothesis":
        from .sc3 import generate_hypothesis
        _print(generate_hypothesis(ns.root))
    elif ns.command == "sc3-freeze":
        from .sc3 import freeze
        _print(freeze(ns.root))
    elif ns.command == "sc3-run":
        from .sc3 import run
        _print(run(ns.root))
    elif ns.command == "sc3-live-probe":
        from .sc3 import run_live_probe
        _print(run_live_probe(ns.root))
    elif ns.command == "sc3-report":
        from .sc3 import report
        _print(report(ns.root))
    elif ns.command == "sc3-recompare":
        from .sc3 import recompare_saved
        _print(recompare_saved(ns.root))
    elif ns.command == "sc3-reconcile":
        from .sc3 import reconcile_interrupted_run
        _print(reconcile_interrupted_run(ns.root))
    elif ns.command == "sc3-retrieve":
        from .sc3 import retrieve_experience
        _print(retrieve_experience(ns.root, ns.query))
    elif ns.command == "recommend":
        rows = [x for x in engine._read(engine.results_path, []) if x["candidate_id"] == ns.candidate_id]
        if not rows:
            raise SystemExit("No evaluation exists for this candidate")
        result = rows[-1]["evaluation"]
        recommendation = ("KEEP AS FIXTURE-QUALIFIED POLICY; NO OPERATIONAL PROMOTION" if result["status"] == "PASS_FIXTURE_ONLY"
                          else "DO NOT PROMOTE; retain the rejected or inconclusive experience")
        _print({"candidate_id": ns.candidate_id, "recommendation": recommendation,
                "evaluation_status": result["status"], "evidence_scope": "synthetic/sanitized Harness fixtures"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
