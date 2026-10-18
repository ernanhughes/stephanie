"""Submit one Markdown draft to Writer and print the handoff record.

SWF01 phase 4. Examples:

    python scripts/writer_handoff.py runs/paper_blogs/10038/2506.21734_blog_*.md
    python scripts/writer_handoff.py draft.md --paper 2506.21734 --run-id 10038 \
        --profile fiction --json

Exit codes: 0 = Writer accepted (PASS), 2 = Writer returned a non-PASS verdict,
3 = handoff could not be performed (Writer unavailable / error).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stephanie.integrations.writer_handoff import (
    WriterDraftHandoff,
    WriterHandoffRequest,
)
from stephanie.integrations.writer_runtime import WriterRuntimeConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="path to the Markdown draft")
    parser.add_argument("--paper", dest="paper_arxiv_id", default=None)
    parser.add_argument("--run-id", dest="research_run_id", default=None)
    parser.add_argument("--profile", default="fiction")
    parser.add_argument("--correlation-id", default=None)
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--json", action="store_true", help="print the raw record JSON")
    args = parser.parse_args(argv)

    artifact = Path(args.artifact)
    if not artifact.is_file():
        print(f"artifact not found: {artifact}", file=sys.stderr)
        return 3

    cfg = WriterRuntimeConfig(
        project_root=args.project_root,
        timeout_seconds=args.timeout,
    )
    handoff = WriterDraftHandoff(cfg=cfg, record_dir=args.record_dir)
    record = handoff.submit(
        WriterHandoffRequest(
            artifact_path=str(artifact),
            paper_arxiv_id=args.paper_arxiv_id,
            research_run_id=args.research_run_id,
            profile=args.profile,
            correlation_id=args.correlation_id,
            provenance={"stage": "scripts.writer_handoff"},
        )
    )

    if args.json:
        print(json.dumps(record.to_dict(), indent=2))
    else:
        print(f"artifact        : {record.artifact_path}")
        print(f"artifact sha256 : {record.artifact_sha256}")
        print(f"writer cli      : {record.writer_cli}")
        print(f"operation       : {record.writer_operation}")
        print(f"report id       : {record.writer_report_id}")
        print(f"status          : {record.status}  decision={record.decision} gate={record.gate}")
        print(f"readiness/grade : {record.readiness}/{record.grade}")
        print(f"findings        : {record.finding_counts}")
        print(f"review required : {record.review_required}")
        for reason in record.human_review_reasons:
            print(f"  - {reason}")
        print(f"record          : {record.record_path}")
        if record.error:
            print(f"error           : {record.error} ({record.detail})")

    if record.status == "PASS":
        return 0
    if record.status in {"unavailable", "error"}:
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
