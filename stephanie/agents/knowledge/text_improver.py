# stephanie/agents/knowledge/text_improver.py
"""
TextImproverAgent — Advanced refinement using policy-driven edits and VPM feedback.
Leverages full ecosystem: KnowledgeBus, CasebookStore, CalibrationManager, GoalScorer.
"""

from __future__ import annotations

import json
import os
import re
import signal as _signal
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.paper_improver.goals import GoalScorer
from stephanie.memory.casebook_store import CaseBookStore
from stephanie.scoring.calibration_manager import CalibrationManager
from stephanie.scoring.scorable import ScorableType
from stephanie.utils.json_sanitize import safe_json


def _supports_alarm() -> bool:
    """Return True if signal.alarm is usable on this platform/thread."""
    return hasattr(_signal, "SIGALRM") and os.name != "nt"


def _timeout_handler(signum, frame):
    raise TimeoutError("TextImprover timed out")


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    tmp.replace(path)


class TextImproverAgent(BaseAgent):
    """
    Production-ready text improver that applies targeted edits based on VPM gaps.
    Fully integrated with casebook, calibration, and event bus.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: Any):
        super().__init__(cfg, memory, container, logger)

        # Core components
        self.workdir = Path(cfg.get("text_improve_workdir", "./data/text_runs"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout = cfg.get("improve_timeout", 60)
        self.seed = cfg.get("seed", 0)
        self.faithfulness_topk = cfg.get("faithfulness_topk", 5)

        # Optional services
        self.kb = container.get("knowledge_graph")  # type: ignore
        self.casebooks = self.memory.casebooks
        self.calibration = cfg.get("calibration") or CalibrationManager(
            cfg=cfg.get("calibration", {}),
            memory=memory,
            logger=logger
        )
        self.gs = GoalScorer(logger=logger)

        # State
        self.run_id = 0

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
          - knowledge_plan: dict
          - draft_text: str
          - goal_template: str
          - initial_goal_score: float
          - session_id: str
          - casebook_name: str
          - case_id: int

        Output:
          - improved_draft: str
          - improvement_trajectory / edit_log: list
          - final_vpm_row: dict
          - final_goal_score: dict
          - run_dir: Path
          - initial_scores, final_scores: raw scoring breakdowns
        """
        plan = context.get("knowledge_plan")
        draft_text = context.get("draft_text", "").strip()
        goal_template = context.get("goal_template", "academic_summary")
        initial_score = context.get("initial_goal_score", 0.0)
        session_id = context.get("session_id", "unknown")
        casebook_name = context.get("casebook_name", "default")
        case_id = context.get("case_id")

        if not plan or not draft_text:
            self.logger.log("TextImproverSkipped", {
                "reason": "missing_input",
                "has_plan": bool(plan),
                "has_draft": bool(draft_text)
            })
            return context

        # Create run directory
        self.run_id += 1
        run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = self.workdir / run_id
        run_dir.mkdir(exist_ok=True)

        try:
            # Optional timeout guard (POSIX only)
            alarm_installed = False
            if _supports_alarm():
                _signal.signal(_signal.SIGALRM, _timeout_handler)
                _signal.alarm(int(self.timeout))
                alarm_installed = True

            # Normalize plan
            plan_norm = self._normalize_plan(plan, run_dir)

            # Save initial draft
            draft_path = run_dir / "draft.md"
            atomic_write(draft_path, draft_text)
            atomic_write(run_dir / "initial_draft.md", draft_text)  # traceable

            # Score BEFORE edits
            initial_scores = self._score_draft(draft_path, plan_norm)

            # Apply edit policy (writes back to draft_path)
            final_text, edits = self._apply_edit_policy(
                draft_path=draft_path,
                plan=plan_norm,
                max_edits=int(context.get("max_edits", 6)),
                trace_path=run_dir / "trace.ndjson"
            )

            # Score AFTER edits
            final_scores = self._score_draft(draft_path, plan_norm)
            vpm_row = self._build_vpm_row(initial_scores, final_scores, plan_norm)

            # Robust goal scoring with fallback
            try:
                goal_eval = self.gs.score("text", goal_template, vpm_row)
            except KeyError:
                self.logger.log("GoalTemplateFallback", {
                    "requested": goal_template,
                    "fallback": "academic_summary"
                })
                goal_eval = self.gs.score("text", "academic_summary", vpm_row)
            label = bool(goal_eval.get("score", 0.0) >= 0.7)

            # Log to casebook
            self.casebooks.add_scorable(
                casebook_name=casebook_name,
                case_id=case_id,
                role="vpm",
                text=safe_json(vpm_row),
                meta={"goal": goal_eval},
                scorable_type=ScorableType.DYNAMIC
            )
            self.casebooks.add_scorable(
                casebook_name=casebook_name,
                case_id=case_id,
                role="text",
                text=final_text,
                meta={"stage": "final"},
                scorable_type=ScorableType.DYNAMIC
            )
            # Persist final draft explicitly too
            atomic_write(run_dir / "final_draft.md", final_text)

            # Publish trajectory event
            self.kb.publish("trajectory.step", {
                "casebook": casebook_name,
                "case_id": case_id,
                "vpm": vpm_row,
                "goal": goal_eval,
            })

            # Update context
            context.update({
                "improved_draft": final_text,
                "improvement_trajectory": edits,
                "edit_log": edits,  # alias for downstream consumers
                "final_vpm_row": vpm_row,
                "final_goal_score": goal_eval,
                "run_dir": str(run_dir),
                "draft_path": str(draft_path),
                "initial_scores": initial_scores,
                "final_scores": final_scores,
            })

            self.report({
                "event": "end",
                "step": "TextImprover",
                "details": f"Scored {goal_eval['score']:.3f} → {'PASS' if label else 'FAIL'}"
            })

            return context

        except TimeoutError:
            self.logger.log("TextImproverTimeout", {"timeout_sec": self.timeout})
            context["error"] = "timeout"
            return context

        except Exception as e:
            self.logger.log("TextImproverUnexpectedError", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            context["error"] = f"unexpected: {str(e)}"
            return context

        finally:
            if _supports_alarm():
                _signal.alarm(0)  # clear any pending alarms

    def _normalize_plan(self, plan: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """Ensure plan has consistent structure."""
        out = {}
        out["section_title"] = (plan.get("section_title") or "Section").strip() or "Section"

        units_in = plan.get("units") or []
        clean_units = []
        for u in units_in:
            if not isinstance(u, dict):
                continue
            claim = (u.get("claim") or "").strip()
            evidence = (u.get("evidence") or "See paper").strip()
            cid = u.get("claim_id")
            if not claim:
                continue
            clean_units.append({"claim": claim, "evidence": evidence, "claim_id": cid})

        if not clean_units:
            clean_units = [{"claim": f"Overview of {out['section_title']}.", "evidence": "", "claim_id": "C1"}]

        out["units"] = clean_units
        out["entities"] = plan.get("entities", {})
        out["paper_text"] = plan.get("paper_text", "")
        out["domains"] = plan.get("domains", [])
        out["tags"] = plan.get("tags", [])

        # Save for debugging
        atomic_write(run_dir / "plan.json", json.dumps(out, indent=2))
        return out

    def _score_draft(self, draft_path: Path, plan: Dict[str, Any]) -> Dict[str, float]:
        """Compute fine-grained scores for edit policy."""
        if not draft_path.exists():
            return {}

        text = draft_path.read_text()

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        units = plan.get("units", [])

        # Coverage
        ids = [u.get("claim_id") for u in units if u.get("claim_id")]
        covered_ids = sum(1 for cid in ids if f"[#{cid}]" in text) if ids else 0
        coverage = covered_ids / max(1, len(ids)) if ids else 0.0

        # Citation support / correctness proxy
        factual = [s for s in sentences if self._is_factual_sentence(s)]
        cited = sum(1 for s in factual if "[#]" in s)
        citation_support = (cited / max(1, len(factual))) if factual else 1.0
        correctness = citation_support

        # Entity consistency (ABBR handling)
        abbrs = plan.get("entities", {}).get("ABBR", {})
        entity_consistency = self._compute_abbr_consistency(text, abbrs)

        # Readability (FKGL)
        words = text.split()
        num_words = len(words)
        num_sentences = len(sentences)
        syllables = sum(self._count_syllables(w) for w in words) or 1
        fkgl_raw = (
            0.39 * (num_words / max(1, num_sentences)) +
            11.8 * (syllables / max(1, num_words)) -
            15.59
        )
        readability = float(max(6.0, min(15.0, fkgl_raw)))

        # Coherence
        coh_scores = []
        for i in range(len(sentences) - 1):
            s1 = set(re.findall(r"\w+", sentences[i].lower()))
            s2 = set(re.findall(r"\w+", sentences[i + 1].lower()))
            denom = len(s1 | s2)
            coh_scores.append((len(s1 & s2) / denom) if denom else 1.0)
        coherence = sum(coh_scores) / len(coh_scores) if coh_scores else 1.0

        # Novelty & Stickiness
        novelty = 0.5  # placeholder
        stickiness = self._compute_stickiness(text, plan)

        len_chars = len(text)
        compactness = len(re.sub(r"\s+", " ", text)) / max(1, len(text))

        return {
            "coverage": coverage,
            "correctness": correctness,
            "coherence": coherence,
            "citation_support": citation_support,
            "entity_consistency": entity_consistency,
            "readability": readability,
            "novelty": novelty,
            "stickiness": stickiness,
            "len_chars": float(len_chars),
            "compactness": float(compactness),
            "fkgl_raw": float(fkgl_raw),
        }

    def _apply_edit_policy(self, draft_path: Path, plan: Dict[str, Any], max_edits: int = 6, trace_path: Optional[Path] = None) -> tuple[str, List[str]]:
        text = draft_path.read_text()
        edits = []

        for i in range(max_edits):
            scores = self._score_draft(draft_path, plan)
            change = False

            if scores["coverage"] < 0.8:
                missing = [
                    u for u in plan.get("units", [])
                    if u.get("claim_id") and f"[#{u['claim_id']}]" not in text
                ]
                if missing:
                    u = missing[0]
                    line = (
                        f"- {u.get('claim','Claim')} [#{u['claim_id']}].\n"
                        f"  *Evidence: {u.get('evidence','See paper')}* [#]\n"
                    )
                    text = text.rstrip() + "\n\n" + line
                    edits.append(f"add_claim:{u['claim_id']}")
                    change = True

            if not change and scores["citation_support"] < 0.9:
                lines = text.splitlines()
                new_lines = []
                for line in lines:
                    if any(kw in line.lower() for kw in ["show","prove","result","increase","decrease","outperform","statistically"]) and "[#]" not in line:
                        line += " [#]"
                    new_lines.append(line)
                new_text = "\n".join(new_lines)
                if new_text != text:
                    text = new_text
                    edits.append("add_citation_placeholders")
                    change = True

            if not change and not (9.0 <= scores["readability"] <= 11.0):
                before = text
                if scores["readability"] > 11.0:
                    text = re.sub(r";\s+", ". ", text)
                    text = re.sub(r"\s+and\s+", ". ", text, count=1)
                    edits.append("split_long_sentences")
                else:
                    text = re.sub(r"\.\s+([a-z])", r", \1", text, count=1)
                    edits.append("join_short_sentences")
                change = True

            if not change and scores["coherence"] < 0.7:
                before = text
                # Merge short adjacent bullets: "- aaa.\n- bbb." -> "- aaa; bbb."
                text = re.sub(r"\n- ([^.\n]{0,60})\.\s*\n- ", r"\n- \1; ", text)
                if text != before:
                    edits.append("merge_adjacent_bullets")
                else:
                    text = self._regenerate_lead_in(text, plan)
                    edits.append("regen_lead_in")
                change = True

            if not change and self._has_duplicate_bullets(text):
                text = self._dedup_bullets(text)
                edits.append("dedup_bullets")
                change = True

            if not change:
                break

            atomic_write(draft_path, self._normalize_ws(text))
            if trace_path:
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"edit": i + 1, "scores": scores, "op": edits[-1]}) + "\n")

        return text, edits

    def _build_vpm_row(self, initial: Dict, final: Dict, plan: Dict) -> Dict[str, float]:
        return {
            "coverage": final["coverage"],
            "correctness": final["correctness"],
            "coherence": final["coherence"],
            "citation_support": final["citation_support"],
            "entity_consistency": final["entity_consistency"],
            "readability": final["readability"],
            "fkgl_raw": final["fkgl_raw"],
            "novelty": final["novelty"],
            "stickiness": final["stickiness"],
            "len_chars": final["len_chars"],
            "compactness": final["compactness"]
        }

    def _is_factual_sentence(self, s: str) -> bool:
        return any(kw in s.lower() for kw in ("show", "prove", "result", "achiev", "increase", "decrease", "outperform", "error", "accuracy", "loss", "significant", "statistically"))

    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        if word and word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        return count

    def _compute_abbr_consistency(self, text: str, abbrs: Dict[str, str]) -> float:
        if not abbrs:
            return 1.0
        matches = 0
        for full, abbr in abbrs.items():
            instances = len(re.findall(rf"\b{re.escape(full)}\b", text, re.IGNORECASE))
            abbreviations = len(re.findall(rf"\b{re.escape(abbr)}\b", text, re.IGNORECASE))
            if instances > 1:
                matches += 1 if abbreviations >= 1 else 0
            else:
                matches += 1 if abbreviations == 0 else 0
        return matches / max(1, len(abbrs))

    def _compute_stickiness(self, text: str, plan: Dict[str, Any]) -> float:
        plan_terms = set()
        for unit in plan.get("units", []):
            claim = unit.get("claim", "") or ""
            for w in re.findall(r"\b[a-zA-Z]{5,}\b", claim.lower()):
                plan_terms.add(w)
        if not plan_terms:
            return 1.0
        text_words = set(re.findall(r"\b[a-zA-Z]{5,}\b", text.lower()))
        return len(plan_terms & text_words) / max(1, len(plan_terms))

    def _normalize_ws(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n"

    def _regenerate_lead_in(self, text: str, plan: Dict[str, Any]) -> str:
        lead = f"This section covers {plan['section_title'].lower()} with key insights."
        bullets = text.splitlines()
        return lead + "\n\n" + "\n".join(b for b in bullets if b.strip())

    def _has_duplicate_bullets(self, text: str) -> bool:
        lines = [b.strip() for b in text.splitlines() if b.strip()]
        seen = set()
        for line in lines:
            if line in seen:
                return True
            seen.add(line)
        return False

    def _dedup_bullets(self, text: str) -> str:
        lines = text.splitlines()
        seen = set()
        unique = []
        for line in lines:
            if line.strip() not in seen:
                seen.add(line.strip())
                unique.append(line)
        return "\n".join(unique)