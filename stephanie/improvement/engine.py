"""Evidence-bound improvement workflow; model output never authorizes changes."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, ConfigDict, Field

from .models import Candidate, Evaluation, Evidence, Hypothesis, Opportunity, digest
from .safety import assert_model_context_safe, validate_reviewed_experience

EVALUATOR_VERSION = "harness-evidence-gate.v1"
EVALUATOR_DIGEST = digest({
    "version": EVALUATOR_VERSION,
    "rules": ["evaluation PASS is not acceptance", "missing evidence stays unknown",
              "verified requires PASS and explicit ACCEPTED", "cost unknown is not zero",
              "fixture oracle derives only from independent evaluation and explicit acceptance"],
})


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


class HypothesisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str = Field(min_length=10, max_length=500)
    rationale: str = Field(min_length=10, max_length=1000)
    uncertainty: str = Field(min_length=5, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)


class ImprovementEngine:
    """File-backed evidence/candidate ledger with injectable Stephanie stores."""

    def __init__(self, root: str | Path, *, experiment_store=None, casebook_store=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.experiment_store = experiment_store
        self.casebook_store = casebook_store
        self.evidence_path = self.root / "evidence.json"
        self.opportunities_path = self.root / "opportunities.json"
        self.candidates_path = self.root / "candidates.json"
        self.results_path = self.root / "results.json"
        self.ai_path = self.root / "ai-executions.json"

    @staticmethod
    def _read(path: Path, default):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def import_evidence(self, records: list[dict]) -> list[Evidence]:
        current = self._read(self.evidence_path, [])
        assert_model_context_safe(current)
        existing = {x["evidence_id"]: Evidence.model_validate(x) for x in current}
        added = []
        for raw in records:
            assert_model_context_safe(raw)
            item = Evidence.model_validate(raw)
            if item.source_digest != digest({k: v for k, v in raw.items() if k != "source_digest"}) and raw.get("digest_payload"):
                raise ValueError(f"source digest mismatch: {item.evidence_id}")
            prev = existing.get(item.evidence_id)
            if prev and prev != item:
                raise ValueError(f"evidence identity collision: {item.evidence_id}")
            existing[item.evidence_id] = item
            added.append(item)
        _atomic_json(self.evidence_path, [v.model_dump(mode="json") for v in sorted(existing.values(), key=lambda e: e.evidence_id)])
        return added

    def import_json(self, path: str | Path) -> list[Evidence]:
        return self.import_evidence(json.loads(Path(path).read_text(encoding="utf-8")))

    def investigate_mission_observation(self, source_snapshot: dict[str, Any]) -> dict[str, Any]:
        """Record a local, deterministic investigation of one Mission finding.

        The source snapshot must already be sanitized by Mission and contain
        stable episode/finding identities. This creates no candidate or
        execution authority; a single observation normally yields an
        additional-evidence request.
        """
        if not isinstance(source_snapshot, dict) or len(json.dumps(source_snapshot, ensure_ascii=False)) > 24_000:
            raise ValueError("Mission source snapshot must be a bounded object")
        allowed = {"episode_id", "project_key", "episode_title", "finding_id",
                   "finding_title", "finding_kind", "finding_severity",
                   "finding_status", "finding_rationale", "evidence_refs",
                   "next_action"}
        if set(source_snapshot) - allowed:
            raise ValueError("Mission source snapshot contains unsupported fields")
        assert_model_context_safe(source_snapshot)
        episode_id = str(source_snapshot.get("episode_id") or "")
        finding_id = str(source_snapshot.get("finding_id") or "")
        if not episode_id or not finding_id:
            raise ValueError("Mission episode and finding identities are required")
        if len(episode_id) > 160 or len(finding_id) > 64:
            raise ValueError("Mission source identities exceed their bounds")
        for key, limit in (("project_key", 160), ("episode_title", 500),
                           ("finding_title", 300), ("finding_kind", 100),
                           ("finding_severity", 40), ("finding_status", 40),
                           ("finding_rationale", 1000), ("next_action", 500)):
            if len(str(source_snapshot.get(key) or "")) > limit:
                raise ValueError(f"Mission source field {key} exceeds its bound")
        refs = source_snapshot.get("evidence_refs", [])
        if not isinstance(refs, list) or len(refs) > 20 or any(not isinstance(x, str) or len(x) > 500 for x in refs):
            raise ValueError("Mission evidence_refs must be a bounded list of strings")
        source_identity = f"mission:episode/{episode_id}/finding/{finding_id}"
        source_digest = digest(source_snapshot)
        investigation_id = "mission-" + digest({"source_identity": source_identity,
                                                "source_digest": source_digest})[:24]
        path = self.root / "investigations.json"
        rows = self._read(path, [])
        existing = next((row for row in rows if row.get("investigation_id") == investigation_id), None)
        if existing is not None:
            if existing.get("source_digest") != source_digest:
                raise ValueError("Mission finding changed; a new investigation identity is required")
            return existing

        title = str(source_snapshot.get("finding_title") or "Untitled Mission finding")[:300]
        rationale = str(source_snapshot.get("finding_rationale") or "")[:1000]
        facts = (f"Mission finding: {title}",)
        if rationale:
            facts += (f"Mission recorded rationale: {rationale}",)
        evidence = Evidence(
            evidence_id=digest(source_identity)[:24], source_system="mission",
            source_identity=source_identity, source_digest=source_digest,
            observation_id=finding_id, task_id=episode_id,
            evaluation_status="UNKNOWN", acceptance_status="UNKNOWN",
            observed_facts=facts,
            limitations=("Source snapshot is a bounded copy; Mission remains authoritative.",
                         "A single finding does not establish recurrence, cause, or candidate effectiveness."),
            cost_status="unknown")
        imported = [evidence.model_dump(mode="json")]
        worker_evidence_id = None
        mission_experiment = None
        hypothesis = {
            "statement": f"The recorded finding '{title}' may warrant a bounded investigation.",
            "rationale": "This hypothesis is a deterministic restatement of Mission's source finding; it does not infer a cause or recurrence.",
            "uncertainty": "One finding is insufficient to establish a recurring mechanism or justify an experiment.",
            "evidence_ids": [evidence.evidence_id],
            "generation": "deterministic_source_restating",
        }
        eligible_kind = source_snapshot.get("finding_kind") == "worker_json_fence_normalization"
        if eligible_kind:
            # This one bounded experiment profile is eligible only when the
            # Mission finding cites the exact current Worker source identity.
            # Similar titles or timestamps are never used for association.
            from .sc3 import _git_head, _sha, _worker_sources, freeze
            source_files = _worker_sources()
            worker_digest = _sha(json.dumps(source_files, sort_keys=True).encode())
            worker_evidence_id = "worker-strategy-source-" + worker_digest[:12]
            refs = source_snapshot.get("evidence_refs", [])
            if worker_evidence_id in refs:
                worker_record = Evidence(
                    evidence_id=worker_evidence_id, source_system="worker",
                    source_identity=f"git:{_git_head(Path('C:/Projects/worker'))}:worker-strategy-runtime",
                    source_digest=worker_digest,
                    observed_facts=("Worker exposes versioned structured.v1 and adaptive.v1 strategies.",),
                    limitations=("Source worktree contains concurrent modifications; per-file content digests are frozen in SC3 protocol.",),
                    evaluation_status="UNKNOWN", acceptance_status="UNKNOWN", cost_status="unknown")
                imported.append(worker_record.model_dump(mode="json"))
                mission_observation = {"source_identity": source_identity,
                    "source_digest": source_digest, "evidence_id": evidence.evidence_id,
                    "episode_id": episode_id, "finding_id": finding_id}
                self.import_evidence(imported)
                hypothesis = {
                    "statement": "For the cited Worker source, adaptive.v1 with json-fence.v1 may handle a complete JSON code fence that structured.v1 rejects.",
                    "rationale": "The Mission finding explicitly classifies this structured-output case and cites the exact Worker source digest showing the versioned baseline and candidate strategies.",
                    "uncertainty": "This is a deterministic mechanism hypothesis; fixture results do not establish general effectiveness or operational acceptance.",
                    "evidence_ids": sorted([evidence.evidence_id, worker_evidence_id]),
                    "generation": "deterministic_source_analysis",
                }
                ai_hypothesis = {"schema_version": "stephanie.sc3-ai-hypothesis.v1",
                    "evidence_id": worker_evidence_id, "worker_head": _git_head(Path("C:/Projects/worker")),
                    "worker_source_sha256": source_files, "model": "deterministic-source-analysis",
                    "request_id": investigation_id, "hypothesis": {k: v for k, v in hypothesis.items() if k != "generation"},
                    "hypothesis_digest": digest({k: v for k, v in hypothesis.items() if k != "generation"}),
                    "authority": "hypothesis_only", "protected_evaluator_in_context": False,
                    "raw_response_artifact": None}
                _atomic_json(self.root / "ai-hypothesis.json", ai_hypothesis)
                mission_observation["worker_evidence_id"] = worker_evidence_id
                mission_observation["worker_source_sha256"] = source_files
                prepared = freeze(self.root, mission_observation=mission_observation)
                mission_experiment = {"experiment_id": prepared["experiment_id"],
                    "candidate_id": prepared["candidate_id"],
                    "candidate_digest": prepared["candidate_digest"],
                    "protocol_digest": prepared["protocol_digest"],
                    "resource_limits": self._read(self.root / "frozen-protocol.json", {}).get("resource_limits"),
                    "request_count": prepared["requests"],
                    "protocol": "SC3 deterministic Worker/Praxis fixture; zero model invocations"}
                hypothesis = ai_hypothesis["hypothesis"]
                imported = []
            else:
                self.import_evidence(imported)
        else:
            self.import_evidence(imported)
        retrieved = self.retrieve_experience(f"{title} {rationale}", limit=5,
            casebook_name="stephanie-sc3-integrated-experiments",
            experience_key="experience")
        investigation = {
            "schema_version": "stephanie.mission-investigation.v1",
            "investigation_id": investigation_id,
            "source_identity": source_identity,
            "source_digest": source_digest,
            "source_snapshot": source_snapshot,
            "status": "completed",
            "result": "PROPOSE_DETERMINISTIC_FIXTURE_EXPERIMENT" if mission_experiment else "REQUEST_ADDITIONAL_EVIDENCE",
            "hypothesis": hypothesis,
            "candidate": ({"candidate_id": mission_experiment["candidate_id"],
                           "candidate_digest": mission_experiment["candidate_digest"],
                           "name": "adaptive JSON-fence normalization",
                           "intervention": "Run adaptive.v1 with json-fence.v1 against the frozen structured.v1 baseline.",
                           "evidence_ids": hypothesis.get("evidence_ids", [])}
                          if mission_experiment else None),
            "proposed_experiments": [mission_experiment] if mission_experiment else [],
            "retrieved_experience": retrieved,
            "authority": "investigation_only; no experiment or execution authority",
        }
        rows.append(investigation)
        _atomic_json(path, rows)
        return investigation

    def opportunities(self) -> list[Opportunity]:
        raw_evidence = self._read(self.evidence_path, [])
        assert_model_context_safe(raw_evidence)
        evidence = [Evidence.model_validate(x) for x in raw_evidence]
        # Establish the recurring gap from observed status splits. Abstain if it
        # is not present in at least two independent evidence records.
        unresolved = [e for e in evidence if e.acceptance_status == "AWAITING_ACCEPTANCE"
                      or e.evaluation_status == "NOT_EVALUATED"]
        if len(unresolved) < 2:
            result = [Opportunity(opportunity_id="harness-acceptance-evidence", title="Evidence gate qualification",
                                  evidence_ids=tuple(e.evidence_id for e in evidence),
                                  observed_gap="Insufficient repeated evidence to establish a recurring gap",
                                  confidence="low", abstained=True,
                                  reason="At least two unresolved observations are required")]
        else:
            result = [Opportunity(opportunity_id="harness-acceptance-evidence", title="Separate execution, evaluation and acceptance",
                                  evidence_ids=tuple(e.evidence_id for e in unresolved),
                                  observed_gap=f"{len(unresolved)} records remain unevaluated or await acceptance; execution completion alone does not establish effectiveness",
                                  confidence="medium")]
        _atomic_json(self.opportunities_path, [x.model_dump(mode="json") for x in result])
        return result

    def construct_candidate(self, opportunity: Opportunity, hypothesis: Hypothesis | None = None,
                            *, additional_compute: str | None = None) -> Candidate:
        if opportunity.abstained or len(opportunity.evidence_ids) < 2:
            raise ValueError("cannot construct a candidate from an abstained opportunity")
        if hypothesis is None:
            prior_ai = self._read(self.ai_path, [])
            available = [row for row in prior_ai
                         if row.get("qualification_eligible") is not False
                         and row.get("authority") == "hypothesis_only"
                         and str(row.get("experiment_state", "historical")).lower() not in {"current", "running", "evaluation", "evaluated"}
                         and isinstance(row.get("hypothesis"), dict)]
            if available:
                raw = available[-1]["hypothesis"]
                hypothesis = Hypothesis(statement=raw["statement"], rationale=raw["rationale"],
                    evidence_ids=tuple(raw["evidence_ids"]), uncertainty=raw["uncertainty"])
        hyp = hypothesis or Hypothesis(
            statement="Require independent evaluation and explicit acceptance before labeling an outcome verified.",
            rationale="Harness records distinguish execution completion, independent evaluation, and pending human acceptance.",
            evidence_ids=opportunity.evidence_ids,
            uncertainty="Fixture evidence can qualify label handling but cannot establish operational effectiveness.",
        )
        if not set(hyp.evidence_ids).issubset(opportunity.evidence_ids):
            raise ValueError("hypothesis cites evidence outside its opportunity")
        evidence_rows = {row["evidence_id"]: Evidence.model_validate(row) for row in self._read(self.evidence_path, [])}
        if not set(opportunity.evidence_ids).issubset(evidence_rows):
            raise ValueError("opportunity refers to evidence that is not imported")
        evidence_digests = tuple(sorted((evidence_id, digest(evidence_rows[evidence_id].model_dump(mode="json")))
                                        for evidence_id in opportunity.evidence_ids))
        payload = {"name": "explicit-verification-gate", "intervention": "Use PASS plus explicit ACCEPTED; otherwise preserve evaluation/acceptance statuses.",
                   "parameters": {"require_independent_pass": True, "require_explicit_acceptance": True},
                   "evidence_ids": list(opportunity.evidence_ids), "evidence_digests": evidence_digests,
                   "hypothesis": hyp.model_dump(mode="json"),
                   "protected_evaluator_digest": EVALUATOR_DIGEST,
                   "requires_additional_compute": bool(additional_compute), "additional_compute": additional_compute}
        candidate = Candidate(candidate_id=digest(payload)[:24], name=payload["name"], intervention=payload["intervention"],
                              parameters=payload["parameters"], evidence_ids=tuple(payload["evidence_ids"]), hypothesis=hyp,
                              evidence_digests=evidence_digests,
                              protected_evaluator_digest=EVALUATOR_DIGEST,
                              requires_additional_compute=bool(additional_compute), additional_compute=additional_compute)
        rows = self._read(self.candidates_path, [])
        if rows and any(x["candidate_id"] == candidate.candidate_id and x != candidate.model_dump(mode="json") for x in rows):
            raise ValueError("candidate identity collision")
        if not any(x["candidate_id"] == candidate.candidate_id for x in rows):
            rows.append(candidate.model_dump(mode="json")); _atomic_json(self.candidates_path, rows)
        return candidate

    def validate_candidate(self, candidate: Candidate, *, authorized: bool = False) -> None:
        if not authorized:
            raise PermissionError("candidate is not authorized for this isolated fixture experiment")
        if candidate.protected_evaluator_digest != EVALUATOR_DIGEST:
            raise ValueError("candidate attempts to replace or weaken the protected evaluator")
        if candidate.requires_additional_compute and not candidate.additional_compute:
            raise ValueError("additional compute requirement must be identified")
        stored = {x["candidate_id"]: x for x in self._read(self.candidates_path, [])}
        if stored.get(candidate.candidate_id) != candidate.model_dump(mode="json"):
            raise ValueError("candidate is not the immutable persisted candidate")
        current = {row["evidence_id"]: digest(Evidence.model_validate(row).model_dump(mode="json"))
                   for row in self._read(self.evidence_path, [])}
        if dict(candidate.evidence_digests) != {key: current.get(key) for key, _ in candidate.evidence_digests}:
            raise ValueError("candidate evidence source digests no longer match the imported evidence")

    @staticmethod
    def classify(record: dict, candidate: Candidate | None = None) -> str:
        if candidate is None:
            # Deliberately flawed baseline policy: equates execution completion
            # with verification, as a fixture representing the known defect.
            return "VERIFIED" if record.get("execution_completed") is True else "UNVERIFIED"
        return "VERIFIED" if record.get("evaluation_status") == "PASS" and record.get("acceptance_status") == "ACCEPTED" else "UNVERIFIED"

    def evaluate_fixtures(self, candidate: Candidate) -> Evaluation:
        self.validate_candidate(candidate, authorized=True)
        rows = self._read(self.evidence_path, [])
        if not rows or any(row.get("evaluation_status") is None or row.get("acceptance_status") is None for row in rows):
            status, baseline, proposed = "INCONCLUSIVE", None, None
            reason = "Fixture oracle is missing; Stephanie abstains rather than deriving correctness from execution status."
        else:
            expected = ["VERIFIED" if row.get("evaluation_status") == "PASS" and row.get("acceptance_status") == "ACCEPTED" else "UNVERIFIED" for row in rows]
            baseline = sum(self.classify(row) == target for row, target in zip(rows, expected)) / len(rows)
            proposed = sum(self.classify(row, candidate) == target for row, target in zip(rows, expected)) / len(rows)
            status = "PASS_FIXTURE_ONLY" if proposed > baseline else ("REJECTED" if proposed < baseline else "INCONCLUSIVE")
        reason = ("P1: candidate improves fixture-label agreement with the frozen independent rule; this is deterministic fixture evidence, not operational effectiveness."
                  if status == "PASS_FIXTURE_ONLY" else "No fixture-label improvement established; no operational effectiveness claim.")
        if candidate.requires_additional_compute:
            status = "INCONCLUSIVE"
            reason = "P6: additional compute is declared, but no compute-matched control was run; the apparent gain cannot be attributed to the strategy."
        result = Evaluation(status=status, baseline_score=float(baseline) if baseline is not None else None,
                            candidate_score=float(proposed) if proposed is not None else None, reason=reason,
                            evaluator_digest=EVALUATOR_DIGEST,
                            additional_compute_identified=(not candidate.requires_additional_compute or bool(candidate.additional_compute)),
                            additional_compute=candidate.additional_compute)
        results = self._read(self.results_path, [])
        results.append({"candidate_id": candidate.candidate_id, "evaluation": result.model_dump(mode="json"),
                        "baseline": {"policy": "execution completion means verified", "score": result.baseline_score},
                        "candidate": {"policy": candidate.intervention, "score": result.candidate_score}, "fixtures": rows})
        _atomic_json(self.results_path, results)
        if self.experiment_store is not None:
            self._persist_experiment(candidate, result, rows)
        self.retain_experience(candidate, result)
        return result

    def _persist_experiment(self, candidate: Candidate, evaluation: Evaluation, rows: list[dict]) -> None:
        store = self.experiment_store
        exp = store.get_or_create_experiment("stephanie-evolution-harness-evidence-gate", domain="harness-evidence",
              config={"protocol_digest": EVALUATOR_DIGEST, "candidate_digest": candidate.candidate_id, "fixture_count": len(rows), "offline": True})
        experiment_id = int(store.get_by_name("stephanie-evolution-harness-evidence-gate", domain="harness-evidence")["id"])
        store.upsert_variant(experiment_id, "baseline", is_control=True, payload={"policy": "execution-completion-is-verified"})
        store.upsert_variant(experiment_id, "candidate", payload=candidate.model_dump(mode="json") | {"candidate_id": candidate.candidate_id})
        variants = {v["name"]: v for v in store.list_variants(experiment_id)}
        for idx, variant_name in enumerate(("baseline", "candidate")):
            score = evaluation.baseline_score if idx == 0 else evaluation.candidate_score
            store.complete_trial(variant_id=int(variants[variant_name]["id"]), case_id=idx + 1, performance=score, metrics={"oracle_accuracy": score or 0.0},
                                 cost=None, meta={"evaluator_digest": EVALUATOR_DIGEST, "evaluation_status": "fixture_only", "additional_compute": "none"},
                                 experiment_group="baseline" if idx == 0 else "candidate")

    def retain_experience(self, candidate: Candidate, evaluation: Evaluation) -> dict:
        item = {"candidate_id": candidate.candidate_id, "intervention": candidate.intervention,
                "outcome": evaluation.status, "evaluator_digest": evaluation.evaluator_digest,
                "evidence_ids": list(candidate.evidence_ids), "scope": "synthetic Harness evidence gate fixtures"}
        path = self.root / "experiences.json"
        rows = self._read(path, [])
        if not any(x["candidate_id"] == candidate.candidate_id for x in rows):
            rows.append(item); _atomic_json(path, rows)
            if self.casebook_store is not None:
                book = self.casebook_store.get_by_name("stephanie-evolution-harness")
                if book is None:
                    book = self.casebook_store.create_casebook("stephanie-evolution-harness", description="Offline Harness improvement experiences", tags=["evolution", "harness"])
                book_id = self.casebook_store.get_id_by_name("stephanie-evolution-harness")
                if book_id is None:
                    raise RuntimeError("CaseBook creation did not persist")
                self.casebook_store.add_case(book_id, None, "stephanie-improvement", name=candidate.candidate_id,
                    description=evaluation.reason, prompt_text=candidate.intervention,
                    meta={"experience": item, "reuse_eligible": False, "promotion_status": "NOT_PROMOTED"})
        return item

    def retrieve_experience(self, query: str, *, limit: int = 10,
                            casebook_name: str = "stephanie-evolution-harness",
                            experience_key: str = "experience",
                            require_reviewed: bool = False) -> list[dict]:
        tokens = {x.lower() for x in query.split() if len(x) > 2}
        rows = self._read(self.root / "experiences.json", [])
        if self.casebook_store is not None:
            book_id = self.casebook_store.get_id_by_name(casebook_name)
            if book_id is None:
                return []
            summaries = self.casebook_store.list_case_summaries(book_id)
            rows = []
            for summary in summaries:
                meta = summary.get("meta", {})
                item = meta.get(experience_key)
                if not item:
                    continue
                item = {**item, "experience_id": item.get("experience_id") or summary.get("name")}
                if require_reviewed:
                    validate_reviewed_experience(item)
                rows.append(item)
        ranked = []
        for row in rows:
            text = " ".join([row.get("intervention", ""), row.get("scope", ""), row.get("outcome", ""),
                              row.get("description", ""), row.get("prompt_text", ""),
                              row.get("summary", ""), row.get("observed_outcome", ""),
                              row.get("limitations", "")]).lower()
            overlap = len(tokens & set(text.split()))
            if overlap:
                ranked.append((overlap, row))
        return [r for _, r in sorted(ranked, key=lambda x: (-x[0], x[1].get("candidate_id") or x[1].get("experience_id") or ""))[:limit]]

    def propose_worker_diagnostic(self) -> dict:
        """Persist a bounded, unapproved investigation proposal; never submit it."""
        proposal = {
            "proposal_id": "worker-empty-output-lifecycle-diagnostic-v1",
            "observed_problem": "Two historical dispatched requests had empty bodies; inference execution is unknown.",
            "source_evidence": [
                "STEPHANIE-EVOLUTION-02Q/DIAGNOSTIC-RESULTS.md",
                "STEPHANIE-CONVERGENCE-01/WORKER-EXECUTION-ISSUE.md",
                "worker-executions.jsonl (archived; request IDs and digests retained)",
            ],
            "baseline": "Existing archived 5-case Worker readiness evidence: 1 usable, 2 dispatched empty, 1 malformed at requested output ceiling, 1 pre-dispatch blocked.",
            "suspected_mechanism": "Unknown; candidate explanations include provider termination, response extraction, timeout/cancellation, and persistence/transport.",
            "alternatives": ["admission or dispatch failure", "provider returned empty", "response extraction loss", "deadline or cancellation", "archive/persistence gap"],
            "intervention": "Capture a single authorized synthetic lifecycle with stage identities, response presence, termination, usage, and durable artifact digest; no retry.",
            "predicted_measurement": "Each stage has explicit observed/unknown state and source-linked artifact identity.",
            "regressions": ["sensitive response retention", "added latency", "misattributed dispatch"],
            "required_backend": "isolated Worker test/runtime owned by Worker maintainers",
            "required_evaluation": "independent review of event completeness and source identity",
            "resource_limits": {"runs": 1, "model_invocations": 0, "timeout_seconds": 120, "cost_usd": 0},
            "required_authority": "separate explicit Worker and Mission authorization before execution or governance submission",
            "rejection_conditions": ["cannot preserve a stable request identity", "state is inferred from HTTP acceptance", "model retry occurs", "raw content is retained without an approved data policy"],
            "praxis_contract": "praxis.effectiveness.compare_effectiveness; advisory; no eligible cohort supplied",
            "approval_status": "NOT_SUBMITTED_NOT_AUTHORIZED",
        }
        path = self.root / "experiment-proposal.json"
        _atomic_json(path, proposal)
        return proposal

    def investigate_worker_empty_output(self) -> dict:
        """Create/retrieve a source-cited investigation from the frozen SE2Q summary."""
        related = self.retrieve_experience("worker empty output inference execution unknown", limit=5)
        investigation = {
            "investigation_id": "worker-empty-output-se2q-reference-v1",
            "source_evidence": ["STEPHANIE-EVOLUTION-02Q/DIAGNOSTIC-RESULTS.md", "STEPHANIE-CONVERGENCE-01/WORKER-EXECUTION-ISSUE.md"],
            "observations": ["2 of 5 Worker requests were dispatched and had empty outputs; inference execution and provider termination remain UNKNOWN.",
                             "1 response ended as malformed at the requested 512-token output ceiling; provider finish reason was not retained.",
                             "One 1024-token follow-up was usable (533 output tokens); this single outcome does not explain the empty responses.",
                             "Current dirty Worker source labels empty output and retains request_dispatched=true with inference_executed=UNKNOWN; this resolves a classification gap, not historical execution attribution."],
            "hypothesis": None,
            "result": "REQUEST_ADDITIONAL_EVIDENCE",
            "rationale": "Historical records do not distinguish provider empty response from extraction, cancellation, timeout, persistence, or transport loss.",
            "unsupported_causes": ["provider failure", "timeout", "cancellation", "output-budget cause for empty cases"],
            "retrieved_experience": [{"candidate_id": row.get("candidate_id"), "outcome": row.get("outcome"), "scope": row.get("scope")} for row in related],
            "authority": "investigation_only; no execution or acceptance authority",
        }
        path = self.root / "investigations.json"
        rows = self._read(path, [])
        if not any(x["investigation_id"] == investigation["investigation_id"] for x in rows):
            rows.append(investigation); _atomic_json(path, rows)
        if self.casebook_store is not None:
            book = self.casebook_store.get_by_name("stephanie-sc2-investigations")
            if book is None:
                self.casebook_store.create_casebook("stephanie-sc2-investigations", description="SC2 source-linked investigations", tags=["sc2", "investigation"])
            book_id = self.casebook_store.get_id_by_name("stephanie-sc2-investigations")
            if book_id is None:
                raise RuntimeError("SC2 CaseBook creation did not persist")
            summaries = self.casebook_store.list_case_summaries(book_id)
            if not any(summary.get("name") == investigation["investigation_id"] for summary in summaries):
                self.casebook_store.add_case(book_id, None, "stephanie-investigation", name=investigation["investigation_id"],
                    description=investigation["rationale"], prompt_text="Worker empty-output lifecycle investigation",
                    meta={"investigation": investigation, "reuse_eligible": False, "promotion_status": "NOT_PROMOTED"})
        return next(x for x in self._read(path, []) if x["investigation_id"] == investigation["investigation_id"])

    def retrieve_investigation(self, investigation_id: str) -> dict | None:
        return next((x for x in self._read(self.root / "investigations.json", [])
                     if x["investigation_id"] == investigation_id), None)

    def investigate_with_ai(self, provider_name: str, model_name: str) -> dict:
        """Call shared ai.inference once; output is schema checked and evidence bounded."""
        from ai.inference import InferenceBudget, InferenceProfile, ReasoningRequest
        from ai.inference.providers.ollama import OllamaReasoningProvider

        raw_evidence = self._read(self.evidence_path, [])
        assert_model_context_safe(raw_evidence)
        evidence = [Evidence.model_validate(x) for x in raw_evidence]
        if not evidence:
            raise ValueError("import evidence before asking AI to investigate")
        if provider_name != "ollama":
            raise ValueError("only the approved local Ollama route is enabled for this invocation")
        if any(e.source_system not in {"synthetic", "harness_public_summary"} for e in evidence):
            raise ValueError("AI investigation accepts only synthetic or sanitized Harness evidence")
        profile = InferenceProfile(name="stephanie-evolution", provider="ollama", model=model_name,
                                   route=f"ollama/{model_name}", base_url="http://127.0.0.1:11434",
                                   temperature=0.0, max_output_tokens=300, timeout_s=120)
        provider = OllamaReasoningProvider(profile=profile)
        from ai.inference import InferenceService
        traces: list[dict] = []
        inference = InferenceService(provider=provider, budget=InferenceBudget(max_calls=1, max_input_chars=12000,
                                        max_output_tokens=300, max_cost_usd=0, per_call_timeout_s=120), trace_sink=traces.append)
        allowed = {e.evidence_id for e in evidence}
        safe_records = [{"evidence_id": e.evidence_id, "source_system": e.source_system,
                         "source_identity": e.source_identity, "source_digest": e.source_digest,
                         "observation_id": e.observation_id, "task_id": e.task_id,
                         "execution_id": e.execution_id, "strategy_id": e.strategy_id,
                         "artifact_id": e.artifact_id, "observed_facts": e.observed_facts,
                         "generated_claims": e.generated_claims, "limitations": e.limitations,
                         "cost_usd": e.cost_usd, "cost_status": e.cost_status}
                        for e in evidence]
        assert_model_context_safe(safe_records)
        request = ReasoningRequest(request_id=str(uuid.uuid4()), purpose="evidence_bounded_opportunity_hypothesis",
            system="Analyze only the supplied evidence. Do not invent facts. Cite only supplied evidence IDs. State uncertainty. Your output is a hypothesis, never an authorization or verified outcome.",
            prompt=json.dumps(safe_records, sort_keys=True),
            max_output_tokens=300, timeout_s=120, temperature=0.0, metadata={"namespace": "stephanie-improvement", "data_classification": "synthetic_or_sanitized"})
        try:
            result, response = inference.generate_structured(request, HypothesisResponse, namespace="stephanie-improvement")
            if not set(response.evidence_ids).issubset(allowed):
                raise ValueError("AI response cited evidence not provided to it")
            failure = None
        except Exception as exc:
            # The shared public API attaches provider results to post-dispatch
            # exceptions. Preserve them locally; never infer missing fields.
            from ai.inference import InferenceUnavailable, StructuredOutputError
            result = getattr(exc, "result", locals().get("result"))
            failure = {"category": getattr(exc, "failure_kind", type(exc).__name__),
                       "exception_type": type(exc).__name__,
                       "validation_category": "schema_validation" if isinstance(exc, StructuredOutputError) and getattr(exc, "failure_kind", "") == "schema_validation" else None}
            record = {"provider": provider.describe(), "result": result.model_dump(mode="json") if result else None,
                      "raw_response": result.content if result else None,
                      "hypothesis": None, "failure": failure,
                      "trace": traces, "authority": "hypothesis_only",
                      "qualification_eligible": False,
                      "request_id": request.request_id,
                      "effective_configuration": profile.model_dump(mode="json"),
                      "dispatch_status": "provider_result_available" if result else "unknown",
                      "inference_status": result.status if result else "unknown",
                      "artifact_references": list(result.execution_refs) if result else [],
                      "cost_status": result.usage.accounting if result else "unknown",
                      "budget": inference.describe()["budget"]}
            rows = self._read(self.ai_path, []); rows.append(record); _atomic_json(self.ai_path, rows)
            raise
        record = {"provider": provider.describe(), "result": result.model_dump(mode="json"), "hypothesis": response.model_dump(mode="json"),
                  "authority": "hypothesis_only", "qualification_eligible": True, "cost_status": result.usage.accounting,
                  "budget": inference.describe()["budget"], "raw_response": result.content,
                  "trace": traces, "request_id": request.request_id,
                  "effective_configuration": profile.model_dump(mode="json"),
                  "failure": failure, "artifact_references": list(result.execution_refs)}
        rows = self._read(self.ai_path, []); rows.append(record); _atomic_json(self.ai_path, rows)
        return record

