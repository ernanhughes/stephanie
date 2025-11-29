"""
VeriCoTAgent — single‑file tool to verify (and optionally repair) PlanTrace
chains-of-thought using the production VeriCoT verifier, and emit VPM tiles.

Usage (CLI):
  python vericot_agent.py \
    --trace tests/fixtures/plantrace.json \
    --context-file docs/statute_42_1.txt \
    --output vpm_tiles/trace_vpm.json \
    --repair-on-fail 2

Key features
- Wraps the **production** VeriCoT verifier (imports your module) and provides:
  • PlanTrace loading (JSON)
  • Step → result mapping with preserved step IDs
  • Optional one-shot or multi-shot self-repair
  • VPM tile emission (JSON) with HRM-grade telemetry
- Pluggable LLM generator for repairs (pass a callable or set env for a stub)
- Zero breaking assumptions: if the verifier import fails, a clear error is shown.

Expected PlanTrace JSON shape (minimal):
{
  "id": "run_123",
  "steps": [{"id": "s1", "nl": "..."}, {"id":"s2","nl":"..."}, ...],
  "context_docs": ["optional long string", "..."]
}

This file is intentionally standalone (no project-specific imports beyond the
verifier). Drop it anywhere in your repo and run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import the production VeriCoT verifier from your codebase.
# If you use a different path, update VERICOT_IMPORT_PATH below.
# ---------------------------------------------------------------------------

VERICOT_IMPORT_PATH = "stephanie.components.vericot.vericot_verifier"

try:
    from stephanie.components.vericot.vericot_verifier import \
        VeriCoTVerifier  # type: ignore
    HAS_VERIFIER = True
except Exception as e:  # pragma: no cover
    HAS_VERIFIER = False
    _IMPORT_ERROR = e


# ---------------------------------------------------------------------------
# Light dataclasses for agent I/O (keeps agent decoupled from verifier types)
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step_id: str
    idx: int
    status: str
    ungrounded_detail: Optional[str] = None
    premises: List[str] = field(default_factory=list)
    premise_votes: List[Tuple[str, bool]] = field(default_factory=list)
    llmaj_ok: Optional[bool] = None
    elapsed_ms: float = 0.0
    unsat_core_size: int = 0
    hr_tags: List[str] = field(default_factory=list)
    # Optional repair fields
    status_before: Optional[str] = None
    ungrounded_detail_before: Optional[str] = None
    repaired: bool = False


@dataclass
class TraceResult:
    trace_id: str
    pass_rate: float
    context_size_final: int
    sat_calls: int
    unsat_calls: int
    avg_check_ms: float
    steps: List[StepResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Optional LLM generator stub for repair (replace with your own callable)
# ---------------------------------------------------------------------------

def make_default_generator() -> Callable[[str], str]:
    """A minimal no-op generator for repairs.

    Replace with your Anthropic/OpenAI client if desired. If the environment
    variable VERICOT_ECHO_REPAIR=1 is set, we echo back the prompt tail to make
    behavior explicit in tests; otherwise we return the original step unchanged.
    """
    echo = os.getenv("VERICOT_ECHO_REPAIR", "0") == "1"

    def _gen(prompt: str) -> str:
        if echo:
            # Return the last line after "Return ONLY the revised step text".
            lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
            return lines[-1] if lines else ""
        # No-op: caller should provide a real generator for repair.
        return ""

    return _gen


# ---------------------------------------------------------------------------
# Repair utility using your verifier (mirrors the paper's reflection pattern)
# ---------------------------------------------------------------------------

def repair_step(
    verifier: Any,
    failed_step_text: str,
    step_idx: int,
    context_docs: List[str],
    step_verif_obj: Any,
    generator: Callable[[str], str],
) -> Tuple[str, Any]:
    """Attempt one repair cycle.

    Returns (revised_step_text, new_step_verification_obj).
    The verifier object is used directly (no side effects on its context).
    Caller is responsible for accepting the revision and updating context.
    """
    # Build a compact reflection prompt
    prompt = f"""
You produced a reasoning step that failed verification.
NL Step: {failed_step_text}
Solver result: {getattr(step_verif_obj, 'status', 'unknown')}
Ungrounded detail: {getattr(step_verif_obj, 'ungrounded_detail', None)}
Supporting premises found: {getattr(step_verif_obj, 'premises', [])}
Premise votes: {getattr(step_verif_obj, 'premise_votes', [])}

Revise the step so it becomes entailed by provided premises (or propose minimal,
source-attributed premises). Return ONLY the revised step text (no explanations).
"""
    revised = generator(prompt) or failed_step_text

    # Verify in isolation by creating a temporary verifier clone
    import copy
    temp = copy.deepcopy(verifier)
    new_verif = temp._verify_single_step(revised, step_idx, context_docs)  # type: ignore
    return revised, new_verif


# ---------------------------------------------------------------------------
# The Agent wrapper
# ---------------------------------------------------------------------------

class VeriCoTAgent:
    """High-level wrapper that:
    - loads a PlanTrace JSON
    - runs VeriCoT verification (and optional repair)
    - emits a VPM-friendly JSON tile
    """

    def __init__(
        self,
        verifier: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not HAS_VERIFIER and verifier is None:  # pragma: no cover
            raise ImportError(
                f"Could not import VeriCoTVerifier from '{VERICOT_IMPORT_PATH}': {_IMPORT_ERROR}"
            )
        self.verifier = verifier or VeriCoTVerifier(
            enable_llmaj=True,
            deterministic_atoms=True,
            short_circuit={"on_contradiction": True, "max_ungrounded": 3},
        )
        self.log = logger or logging.getLogger("VeriCoTAgent")

    # ---- Core run ----
    def run(
        self,
        plantrace: Dict[str, Any],
        repair_on_fail: int = 0,
        generator: Optional[Callable[[str], str]] = None,
    ) -> TraceResult:
        trace_id = str(plantrace.get("id", "unknown"))
        steps_in = plantrace.get("steps", [])
        context_docs = plantrace.get("context_docs", [])

        cot_steps = [str(s.get("nl", "")) for s in steps_in]
        step_ids = [str(s.get("id", f"idx_{i}")) for i, s in enumerate(steps_in)]

        # Initial verify
        verif = self.verifier.verify(cot_steps, context_docs)

        # Build results + optional repair
        gen = generator or make_default_generator()
        out_steps: List[StepResult] = []

        for i, s in enumerate(getattr(verif, "steps", [])):
            sr = StepResult(
                step_id=step_ids[i],
                idx=i,
                status=str(getattr(s, "status", "unknown")),
                ungrounded_detail=getattr(s, "ungrounded_detail", None),
                premises=list(getattr(s, "premises", []) or []),
                premise_votes=list(getattr(s, "premise_votes", []) or []),
                llmaj_ok=getattr(s, "llmaj_ok", None),
                elapsed_ms=float(getattr(s, "elapsed_ms", 0.0) or 0.0),
                unsat_core_size=int(getattr(s, "unsat_core_size", 0) or 0),
                hr_tags=list(getattr(s, "hr_tags", []) or []),
            )

            # Attempt repairs if requested and not valid
            attempts = 0
            while repair_on_fail > 0 and sr.status != "valid" and attempts < repair_on_fail:
                sr.status_before = sr.status
                sr.ungrounded_detail_before = sr.ungrounded_detail

                revised_text, new_s = repair_step(
                    verifier=self.verifier,
                    failed_step_text=cot_steps[i],
                    step_idx=i,
                    context_docs=context_docs,
                    step_verif_obj=s,
                    generator=gen,
                )

                # If valid after repair, accept and update context + record
                if getattr(new_s, "status", None) == "valid":
                    # Update persistent context with revised step & used premises
                    try:
                        expr = self.verifier._autoformalize(revised_text, context_docs)  # type: ignore
                        self.verifier._add_to_context(expr)  # type: ignore
                        for p in getattr(new_s, "premises", []) or []:
                            pexpr = self.verifier._autoformalize(p, context_docs)  # type: ignore
                            self.verifier._add_to_context(pexpr)  # type: ignore
                    except Exception:
                        pass

                    # Update sr
                    sr.status = "valid"
                    sr.premises = list(getattr(new_s, "premises", []) or [])
                    sr.premise_votes = list(getattr(new_s, "premise_votes", []) or [])
                    sr.llmaj_ok = getattr(new_s, "llmaj_ok", sr.llmaj_ok)
                    sr.unsat_core_size = int(getattr(new_s, "unsat_core_size", 0) or 0)
                    sr.repaired = True
                    break
                else:
                    attempts += 1

            out_steps.append(sr)

        # Final roll-up (prefer verifier’s telemetry where present)
        pass_rate = float(getattr(verif, "pass_rate", 0.0) or 0.0)
        ctx_size = int(getattr(verif, "context_size_final", 0) or 0)
        sat_calls = int(getattr(verif, "sat_calls", 0) or 0)
        unsat_calls = int(getattr(verif, "unsat_calls", 0) or 0)
        avg_ms = float(getattr(verif, "avg_check_ms", 0.0) or 0.0)

        return TraceResult(
            trace_id=trace_id,
            pass_rate=pass_rate,
            context_size_final=ctx_size,
            sat_calls=sat_calls,
            unsat_calls=unsat_calls,
            avg_check_ms=avg_ms,
            steps=out_steps,
        )

    # ---- VPM tile writer ----
    def emit_vpm_tile(self, result: TraceResult, path: str | Path) -> None:
        tile = {
            "trace_id": result.trace_id,
            "steps": [
                {
                    "step_id": s.step_id,
                    "idx": s.idx,
                    "status": s.status,
                    **({"status_before": s.status_before} if s.status_before else {}),
                    **({"ungrounded_detail": s.ungrounded_detail} if s.ungrounded_detail else {}),
                    **({"ungrounded_detail_before": s.ungrounded_detail_before} if s.ungrounded_detail_before else {}),
                    "premises": s.premises,
                    "premise_votes": s.premise_votes,
                    **({"llmaj_ok": s.llmaj_ok} if s.llmaj_ok is not None else {}),
                    "unsat_core_size": s.unsat_core_size,
                    "elapsed_ms": s.elapsed_ms,
                    "repaired": s.repaired,
                    "hr_tags": s.hr_tags,
                }
                for s in result.steps
            ],
            "telemetry": {
                "pass_rate": result.pass_rate,
                "context_size_final": result.context_size_final,
                "sat_calls": result.sat_calls,
                "unsat_calls": result.unsat_calls,
                "avg_check_ms": result.avg_check_ms,
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tile, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_plantrace(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_context_from_files(files: List[str]) -> List[str]:
    out: List[str] = []
    for fp in files:
        try:
            out.append(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            # Treat as raw string if reading fails
            out.append(fp)
    return out

def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="VeriCoTAgent — PlanTrace verifier + VPM emitter")
    p.add_argument("--trace", required=True, help="Path to PlanTrace JSON")
    p.add_argument("--output", required=True, help="Path to VPM tile JSON")
    p.add_argument("--context", nargs="*", default=[], help="Inline context strings")
    p.add_argument("--context-file", nargs="*", default=[], help="Paths to context files")
    p.add_argument("--repair-on-fail", type=int, default=0, help="Max repair attempts per failing step")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    plantrace = _load_plantrace(args.trace)
    # Merge inline context and file-based context
    context_docs = (plantrace.get("context_docs") or []) + args.context + _load_context_from_files(args.context_file)
    plantrace["context_docs"] = context_docs

    agent = VeriCoTAgent()
    result = agent.run(plantrace, repair_on_fail=args.repair_on_fail)
    agent.emit_vpm_tile(result, args.output)

    print(f"✅ Verified {sum(s.status=='valid' for s in result.steps)}/{len(result.steps)} steps")
    print(f"📊 Context size: {result.context_size_final} | Avg check: {result.avg_check_ms:.2f} ms")


if __name__ == "__main__":  # pragma: no cover
    main()
