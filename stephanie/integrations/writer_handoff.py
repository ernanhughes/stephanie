"""Stephanie -> Writer editorial handoff (SWF01 phase 4).

Sends **one** generated Markdown draft to a supported, non-destructive Writer
operation and records the outcome as a provenance-backed handoff record.

Design rules (from the SWF01 brief):

* use Writer's **stable public CLI** (``writer validation validate-file``);
  never reach into Writer's database, tables or private packages;
* an actual operation must be invoked -- successful *discovery* is never treated
  as a successful handoff;
* the record carries Writer's decision, but Stephanie never acts on it: it never
  auto-accepts editorial changes and never bypasses Writer's review/policy gates;
* a missing, disabled or failing Writer produces an explicit
  ``unavailable``/``error`` record -- never a silent fallback to another engine;
* the operation is read-only with respect to the draft, so no manuscript content
  is mutated by Stephanie.

The only Writer operation this module is allowed to invoke is
:data:`WRITER_OPERATION`; :data:`ALLOWED_OPERATIONS` is enforced at call time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.integrations.writer_runtime import (
    WriterRuntimeConfig,
    WriterRuntimeDiscovery,
)

logger = logging.getLogger(__name__)

#: The single supported Writer operation for a draft handoff.
WRITER_OPERATION = "validation.validate-file"

#: Only these (command, subcommand) pairs may be invoked operationally.
ALLOWED_OPERATIONS = {("validation", "validate-file")}

DEFAULT_PROFILE = "fiction"
_STDERR_LIMIT = 2000

#: Writer decisions that do NOT require a human to look at the artifact.
_ACCEPT_DECISIONS = {"ACCEPT"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class WriterHandoffRequest:
    """Everything Stephanie must pass to Writer for one handoff."""

    artifact_path: str
    paper_arxiv_id: Optional[str] = None
    research_run_id: Optional[str] = None
    profile: str = DEFAULT_PROFILE
    #: Idempotency / correlation reference (generated when omitted).
    correlation_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WriterHandoffRecord:
    """What Stephanie records after the handoff (and shows for review)."""

    correlation_id: str
    artifact_path: str
    writer_operation: str = WRITER_OPERATION
    #: PASS | FAIL | unavailable | error
    status: str = "error"
    created_at: str = field(default_factory=_now)

    # artifact identity
    artifact_sha256: Optional[str] = None
    artifact_bytes: Optional[int] = None

    # Writer identity
    writer_cli: Optional[str] = None
    writer_report_id: Optional[str] = None
    writer_validation_version: Optional[str] = None

    # Writer's verdict
    decision: Optional[str] = None
    gate: Optional[str] = None
    readiness: Optional[int] = None
    grade: Optional[str] = None
    coverage: Dict[str, Any] = field(default_factory=dict)
    finding_counts: Dict[str, int] = field(default_factory=dict)
    report_markdown: Optional[str] = None

    # review / provenance
    review_required: bool = True
    human_review_reasons: List[str] = field(default_factory=list)
    paper_arxiv_id: Optional[str] = None
    research_run_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    record_path: Optional[str] = None
    error: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WriterDraftHandoff:
    """Invoke Writer's supported validation operation for one draft."""

    def __init__(
        self,
        cfg: Optional[WriterRuntimeConfig] = None,
        *,
        discovery: Optional[WriterRuntimeDiscovery] = None,
        record_dir: Optional[str] = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg or WriterRuntimeConfig()
        self.discovery = discovery or WriterRuntimeDiscovery(self.cfg)
        self.record_dir = Path(record_dir) if record_dir else None
        self.log = logger_ or logger

    # ------------------------------------------------------------------ #
    def submit(self, request: WriterHandoffRequest) -> WriterHandoffRecord:
        correlation_id = request.correlation_id or f"swh-{uuid.uuid4().hex[:12]}"
        artifact = self._resolve_artifact(request.artifact_path)
        record = WriterHandoffRecord(
            correlation_id=correlation_id,
            artifact_path=str(artifact),
            paper_arxiv_id=request.paper_arxiv_id,
            research_run_id=request.research_run_id,
            provenance=dict(request.provenance or {}),
        )

        if not artifact.is_file():
            record.status = "error"
            record.error = "artifact not found"
            record.detail = str(artifact)
            record.human_review_reasons = ["draft artifact could not be read"]
            return self._finish(record)

        try:
            record.artifact_sha256 = _sha256_file(artifact)
            record.artifact_bytes = artifact.stat().st_size
        except Exception as exc:  # noqa: BLE001
            record.status = "error"
            record.error = "artifact unreadable"
            record.detail = str(exc)
            return self._finish(record)

        try:
            exe = self.discovery._resolve_executable()
        except Exception as exc:  # noqa: BLE001
            record.status = "unavailable"
            record.error = "writer executable not found"
            record.detail = str(exc)
            record.human_review_reasons = ["Writer is unavailable; draft not reviewed"]
            return self._finish(record)

        record.writer_cli = exe
        result = self._invoke(exe, artifact, profile=request.profile)
        if result.get("error"):
            record.status = result.get("status", "error")
            record.error = result["error"]
            record.detail = result.get("detail")
            record.human_review_reasons = [f"Writer handoff failed: {result['error']}"]
            return self._finish(record)

        payload: Dict[str, Any] = result["payload"]
        record.status = str(payload.get("status") or "error")
        record.writer_report_id = payload.get("id")
        record.writer_validation_version = payload.get("validation_version")
        record.decision = payload.get("decision")
        record.gate = payload.get("gate")
        record.readiness = payload.get("readiness")
        record.grade = payload.get("grade")
        record.coverage = payload.get("coverage") or {}
        record.report_markdown = payload.get("report_markdown")
        record.finding_counts = {
            "total": int(payload.get("total_findings") or 0),
            "critical": int(payload.get("critical_count") or 0),
            "high": int(payload.get("high_count") or 0),
            "medium": int(payload.get("medium_count") or 0),
            "low": int(payload.get("low_count") or 0),
            "info": int(payload.get("info_count") or 0),
        }
        self._assess_review(record)
        return self._finish(record)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_artifact(path: str) -> Path:
        """Resolve the draft to an ABSOLUTE path.

        Writer runs as a separate process with its own working directory, so a
        relative path produced by the pipeline is meaningless to it. Observed
        live: the in-pipeline stage passed ``runs\\paper_blogs\\10039\\...md`` and
        Writer replied ``File not found: runs\\paper_blogs\\10039\\...md``.
        """
        p = Path(path).expanduser()
        try:
            return p.resolve()
        except OSError:  # pragma: no cover - defensive
            return p if p.is_absolute() else (Path.cwd() / p)

    def _invoke(self, exe: str, artifact: Path, *, profile: str) -> Dict[str, Any]:
        command, subcommand = WRITER_OPERATION.split(".")
        if (command, subcommand) not in ALLOWED_OPERATIONS:
            return {"error": f"operation not allowed: {WRITER_OPERATION}"}

        args = [
            exe,
            command,
            subcommand,
            str(artifact),
            "--profile",
            str(profile or DEFAULT_PROFILE),
            "--json",
        ]
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.cfg.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {"status": "error", "error": "timeout", "detail": str(exc)[:_STDERR_LIMIT]}
        except OSError as exc:
            return {"status": "unavailable", "error": "writer invocation failed", "detail": str(exc)}

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "")[:_STDERR_LIMIT]
        if completed.returncode != 0:
            return {
                "status": "error",
                "error": f"writer exited {completed.returncode}",
                "detail": (stderr or stdout)[:_STDERR_LIMIT],
            }
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "error": "invalid JSON from writer",
                "detail": stdout[:_STDERR_LIMIT] or stderr,
            }
        if not isinstance(payload, dict):
            return {"status": "error", "error": "unexpected writer payload", "detail": str(payload)[:_STDERR_LIMIT]}
        return {"payload": payload}

    @staticmethod
    def _assess_review(record: WriterHandoffRecord) -> None:
        """Decide whether a human must review before this advances.

        Stephanie records this; it never acts on it.
        """
        reasons: List[str] = []
        if record.status != "PASS":
            reasons.append(f"Writer status is {record.status}")
        if record.decision and record.decision not in _ACCEPT_DECISIONS:
            reasons.append(f"Writer decision is {record.decision}")
        if record.gate and record.gate != "PASS":
            reasons.append(f"Writer gate is {record.gate}")
        if record.finding_counts.get("critical"):
            reasons.append(f"{record.finding_counts['critical']} critical finding(s)")
        if record.finding_counts.get("high"):
            reasons.append(f"{record.finding_counts['high']} high finding(s)")
        record.human_review_reasons = reasons
        record.review_required = bool(reasons)

    # ------------------------------------------------------------------ #
    def record_path_for(self, record: WriterHandoffRecord) -> Path:
        if self.record_dir:
            base_dir = self.record_dir
        else:
            base_dir = Path(record.artifact_path).parent
        stem = Path(record.artifact_path).stem
        return base_dir / f"{stem}.writer-handoff.json"

    def _finish(self, record: WriterHandoffRecord) -> WriterHandoffRecord:
        try:
            path = self.record_path_for(record)
            # Set record_path BEFORE serialising so the persisted record is
            # self-describing.
            record.record_path = str(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.log.warning("WriterDraftHandoff: could not persist record: %s", exc)
        if record.status in {"unavailable", "error"}:
            self.log.error(
                "WriterDraftHandoff %s: %s (%s)",
                record.status,
                record.error,
                record.detail,
            )
        else:
            self.log.info(
                "WriterDraftHandoff %s: report=%s decision=%s review_required=%s",
                record.status,
                record.writer_report_id,
                record.decision,
                record.review_required,
            )
        return record
