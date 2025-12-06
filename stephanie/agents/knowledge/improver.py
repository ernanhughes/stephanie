# stephanie/agents/knowledge/text_improver.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import signal as _signal
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.agents.paper_improver.goals import GoalScorer
from stephanie.memory.casebook_store import CaseBookStore
from stephanie.scoring.calibration_manager import CalibrationManager
from stephanie.scoring.scorable import ScorableType
from stephanie.services.bus.knowledge_bus import KnowledgeBus
from stephanie.utils.hash_utils import hash_text
from stephanie.utils.json_sanitize import safe_json

from ..paper_improver.faithfulness import FaithfulnessBot

log = logging.getLogger(__name__)

FACTUAL_KWS = (
    "show",
    "prove",
    "result",
    "achiev",
    "increase",
    "decrease",
    "outperform",
    "error",
    "accuracy",
    "loss",
    "significant",
    "statistically",
)


def _supports_alarm() -> bool:
    """Return True if signal.alarm is usable on this platform/thread."""
    return hasattr(_signal, "SIGALRM") and os.name != "nt"


def _timeout_handler(signum, frame):
    raise TimeoutError("TextImprover timed out")


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    tmp.replace(path)


class Improver:
    def __init__(
        self,
        cfg, 
        memory,
        context: Dict[str, Any],
        workdir: str = "./data/text_runs",
        timeout: int = 60,
        seed: int = 0,
        faithfulness_topk: int = 5,
        kb: KnowledgeBus | None = None,
        casebooks: CaseBookStore | None = None,
        calibration: CalibrationManager | None = None,  
        logger=None,                                    
    ):
        self.cfg = cfg
        self.memory = memory
        self.context = context
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.run_id = 0
        self.timeout = timeout
        self.seed = seed
        self.faithfulness_topk = faithfulness_topk
        self.kb = kb or KnowledgeBus()
        self.casebooks = casebooks or CaseBookStore()

        self.logger = logger
        self.calibration = calibration or CalibrationManager(
            cfg=self.cfg.get("calibration", {}),  # pass calibration sub-config
            memory=self.memory,
            logger=self.logger,
        )

        self.gs = GoalScorer()

    # --------------------------- public ---------------------------

    def improve(self, content_plan: Dict[str, Any]) -> Dict[str, Any]:
        # Optional timeout (Unix main thread only)
        alarm_installed = False
        if _supports_alarm():
            _signal.signal(_signal.SIGALRM, _timeout_handler)
            _signal.alarm(self.timeout)
            alarm_installed = True

        try:
            return self._improve_inner(content_plan)
        except TimeoutError:
            self._log("TextImproverTimeout", {"timeout_sec": self.timeout})
            result = {
                "error": "timeout",
                "passed": False,
                "scores": {},
                "vpm_row": {},
                "run_dir": "",
                "dpo_pair_path": "",
            }
        except Exception as e:
            self._log(
                "TextImproverUnexpectedError",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            result = {
                "error": f"unexpected: {str(e)}",
                "traceback": traceback.format_exc(),
                "passed": False,
                "scores": {},
                "vpm_row": {},
                "run_dir": "",
            }
        finally:
            if alarm_installed:
                _signal.alarm(0)

        # Best-effort error artifact
        if result.get("run_dir"):
            try:
                rd = Path(result["run_dir"])
                atomic_write(rd / "ERROR.json", json.dumps(result, indent=2))
            except Exception:
                pass

        return result

    def _improve_inner(self, content_plan: Dict[str, Any]) -> Dict[str, Any]:
        self._seed_everything(self.seed)
        self.run_id += 1

        # 0) Validate & sanitize plan
        plan_norm = self._sanitize_plan(content_plan)

        # 1) Prepare run dir + meta
        plan_hash = hash_text(json.dumps(plan_norm, sort_keys=True))[:8]
        run_dir = (
            self.workdir
            / f"run_{int(time.time())}_{uuid.uuid4().hex}_{plan_hash}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)

        atomic_write(
            run_dir / "meta.json",
            json.dumps(
                {
                    "plan_sha": plan_hash,
                    "seeds": {"python": self.seed},
                    "timeout": self.timeout,
                    "timestamp": time.time(),
                },
                indent=2,
            ),
        )
        plan_path = run_dir / "plan.json"
        atomic_write(plan_path, json.dumps(plan_norm, indent=2))

        # 2) Casebook + Case
        casebook_name = f"text_{plan_hash}_{(content_plan.get('section_title') or 'section')}"
        pipeline_run_id = self.context.get("pipeline_run_id")
        cb = self.casebooks.ensure_casebook(
            name=casebook_name,
            pipeline_run_id=pipeline_run_id,
            tags=["text_improver", "exemplar_text"],
            meta={"plan_sha": plan_hash},
        )
        case = self.casebooks.add_case(
            casebook_name=casebook_name,
            prompt_text=json.dumps(content_plan),
            agent_name="text_improver",
            meta={"run_dir": str(run_dir)},
        )

        # 3) Initial draft
        draft_path = self._generate_draft(plan_norm, run_dir)

        # 4) Score → Edit → Rescore
        initial_scores = self._score_draft(draft_path, plan_norm)
        final_text, edits = self._apply_edit_policy(
            draft_path=draft_path,
            plan=plan_norm,
            max_edits=6,
            trace_path=run_dir / "trace.ndjson",
        )
        final_scores = self._score_draft(draft_path, plan_norm)

        # 5) Build VPM + log
        vpm_row = self._build_vpm_row(initial_scores, final_scores, plan_norm)
        goal_eval = self.gs.score("text", "academic_summary", vpm_row)

        # Log to casebook
        self.casebooks.add_scorable(
            casebook_name=casebook_name,
            case_id=case.id,
            role="vpm",
            text=safe_json(vpm_row),
            meta={"goal": goal_eval},
            scorable_type=ScorableType.DYNAMIC,
        )
        self.casebooks.add_scorable(
            casebook_name=casebook_name,
            case_id=case.id,
            role="text",
            text=(run_dir / "draft.md").read_text(),
            meta={"stage": "final"},
            scorable_type=ScorableType.DYNAMIC,
        )

        # 6) Optional faithfulness
        faithfulness_score = None
        paper_text = plan_norm.get("paper_text")
        if not paper_text and plan_norm.get("paper_text_path"):
            try:
                p = Path(plan_norm["paper_text_path"])
                if p.exists():
                    paper_text = p.read_text()
            except Exception as e:
                self._log("PaperTextLoadFailed", {"error": str(e)})

        if paper_text and len(paper_text.strip()) > 100:
            try:
                bot = FaithfulnessBot(top_k=self.faithfulness_topk)
                bot.prepare_paper(paper_text)
                claims = [
                    {
                        "claim_id": u.get("claim_id"),
                        "claim": u.get("claim", ""),
                    }
                    for u in plan_norm.get("units", [])
                    if u.get("claim")
                ]
                faithfulness_score = bot.get_faithfulness_score(claims)
                final_scores["faithfulness"] = float(faithfulness_score)
                vpm_row["faithfulness"] = round(float(faithfulness_score), 3)
            except Exception as e:
                self._log("FaithfulnessCheckFailed", {"error": str(e)})

        # 7) DPO pair (persist locally + scorable)
        dpo_pair = {
            "content_plan_slice": self._extract_plan_slice(plan_norm),
            "prompt": "Generate faithful, clear, well-cited prose from this plan.",
            "rejected": (run_dir / "initial_draft.md").read_text(),
            "chosen": final_text,
            "metadata": {
                "run_id": self.run_id,
                "plan_hash": plan_hash,
                "initial_scores": initial_scores,
                "final_scores": final_scores,
                "score_deltas": {
                    k: round(
                        final_scores.get(k, 0.0) - initial_scores.get(k, 0.0),
                        4,
                    )
                    for k in set(initial_scores) | set(final_scores)
                },
                "applied_edits": edits,
            },
        }
        atomic_write(
            run_dir / "text_dpo_pair.json", json.dumps(dpo_pair, indent=2)
        )

        # Pass criteria
        core_ok = all(
            final_scores.get(d, 0.0) >= 0.7
            for d in ("coverage", "correctness", "coherence")
        )
        faithful_ok = (
            True if faithfulness_score is None else (faithfulness_score >= 0.7)
        )

        # Final logging
        self.casebooks.add_scorable(
            casebook_name=casebook_name,
            case_id=case.id,
            role="dpo_pair",
            text=safe_json(dpo_pair),
            meta=dpo_pair["metadata"],
            scorable_type=ScorableType.DYNAMIC,
        )


        # --- Log a calibration sample per section/domain ---

        # choose a primary domain (however you’re tagging; fallback to "general")
        primary_domain = (plan_norm.get("domains") or ["general"])[0] if plan_norm.get("domains") else "general"

        # Simple features from your VPM row (extend as needed)
        calib_features = {
            "coverage": vpm_row.get("coverage_final", 0.0),
            "correctness": vpm_row.get("correctness", 0.0),
            "coherence": vpm_row.get("coherence", 0.0),
            "citation_support": vpm_row.get("citation_support", 0.0),
        }

        # Label: pass/fail → 1/0
        label = 1 if (core_ok and faithful_ok) else 0

        # Optional: a single “raw similarity” proxy (you can keep it simple)
        raw_similarity = 0.25*calib_features["coverage"] + 0.25*calib_features["correctness"] + \
                        0.25*calib_features["coherence"] + 0.25*calib_features["citation_support"]

        try:
            # GOOD: use manager's sanitizer
            self.calibration.log_event(
                domain=primary_domain or "general",
                query=(plan_norm.get("section_title") or "")[:2000],
                raw_sim=float(raw_similarity or 0.0),
                is_relevant=bool(label),
                scorable_id=str(case.id),
                scorable_type="text_draft",
                entity_type=None,
            )
        except Exception as e:
            log.warning("CalibrationEventLogFailed", {"error": str(e)})



        self.kb.publish(
            "trajectory.step",
            {
                "casebook": cb.name if hasattr(cb, "name") else casebook_name,
                "case_id": case.id,
                "vpm": vpm_row,
                "goal": goal_eval,
            },
        )

        # 9) Pass criteria
        core_ok = all(
            final_scores.get(d, 0.0) >= 0.7
            for d in ("coverage", "correctness", "coherence")
        )
        faithful_ok = (
            True if faithfulness_score is None else (faithfulness_score >= 0.7)
        )

        return {
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "final_draft_path": str(draft_path),
            "vpm_row": vpm_row,
            "dpo_pair_path": str(run_dir / "text_dpo_pair.json"),
            "scores": final_scores,
            "passed": bool(core_ok and faithful_ok),
        }

    def _sanitize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure required shape; drop nulls; keep keys we use. Never hard-fail on empty units."""
        if not plan:
            raise ValueError("Plan is None")

        out: Dict[str, Any] = {}
        out["section_title"] = (
            plan.get("section_title") or "Section"
        ).strip() or "Section"

        units_in = plan.get("units") or []
        clean_units: List[Dict[str, Any]] = []
        for u in units_in:
            if not isinstance(u, dict):
                continue
            claim = (u.get("claim") or "").strip()
            evidence = (u.get("evidence") or "See paper").strip()
            cid = u.get("claim_id")
            if not claim:
                continue
            clean_units.append(
                {"claim": claim, "evidence": evidence, "claim_id": cid}
            )

        # Fallback: if no valid units, synthesize a placeholder so pipeline can proceed
        if not clean_units:
            placeholder = f"Overview of {out['section_title']}."
            clean_units = [
                {
                    "claim": placeholder,
                    "evidence": "See paper",
                    "claim_id": None,
                }
            ]

        out["units"] = clean_units

        ents = plan.get("entities") or {}
        out["entities"] = {
            "ABBR": dict(ents.get("ABBR") or {}),
            "REQUIRED": list(ents.get("REQUIRED") or []),
        }

        # Optional extras
        if plan.get("paper_text"):
            out["paper_text"] = plan["paper_text"]
        if plan.get("paper_text_path"):
            out["paper_text_path"] = plan["paper_text_path"]
        if plan.get("outline"):
            out["outline"] = plan["outline"]

        return out

    # --------------------------- generation ---------------------------

    def _generate_draft(self, plan: Dict[str, Any], run_dir: Path) -> Path:
        title = plan.get("section_title", "Section")
        units = plan.get("units", [])
        abbrs = plan.get("entities", {}).get("ABBR", {})

        outline = plan.get("outline") or [
            u.get("claim_id") for u in units if u.get("claim_id")
        ]
        outline = [cid for cid in outline if cid]

        lead = self._lead_paragraph(title, units, abbrs)
        kg_neighbors = (plan.get("kg") or {}).get("neighbors", {})
        seen_ids = set()
        bullets: List[str] = []
        for u in units:
            cid = u.get("claim_id") or ""
            if cid and cid in seen_ids:
                continue
            seen_ids.add(cid)
            claim = u.get("claim", "No claim")
            evidence = u.get("evidence", "See paper")
            tag = f" [#{cid}]" if cid else ""
            cite = " [#]" if evidence and evidence != "See paper" else ""
            bullets.append(f"- {claim}{tag}.\n  *Evidence: {evidence}*{cite}")
            top = (kg_neighbors.get(cid) or [])
            if top:
                src = (top[0].get("sources") or [{}])[0]
                src_hint = src.get("scorable_id") or src.get("source_text") or top[0]["text"]
                if src_hint:
                    evidence = f"KG: {src_hint}"


        draft = f"# {title}\n\n{lead}\n\n"
        if outline:
            draft += (
                "### Outline\n"
                + "\n".join(f"- [{cid}]" for cid in outline)
                + "\n\n"
            )
        draft += "### Details\n" + "\n\n".join(bullets) + "\n"
        cite = " [#]" if evidence and evidence != "See paper" else ""
        bullets.append(f"- {claim}{tag}.\n  *Evidence: {evidence}*{cite}")

        draft_path = run_dir / "draft.md"
        atomic_write(draft_path, self._normalize_ws(draft))
        atomic_write(run_dir / "initial_draft.md", draft_path.read_text())
        return draft_path

    def _lead_paragraph(
        self, title: str, units: List[Dict[str, Any]], abbrs: Dict[str, str]
    ) -> str:
        claims = [u.get("claim", "") for u in units if u.get("claim")]
        head = (
            " ".join([c.rstrip(".") + "." for c in claims[:3]])
            or "This section summarizes key findings and method decisions from the paper."
        )
        for full, abbr in abbrs.items():
            if full in title and abbr not in head:
                head = head.replace(full, f"{full} ({abbr})", 1)
                break
        return head

    # --------------------------- scoring ---------------------------

    def _score_draft(
        self, draft_path: Path, plan: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Heuristic scorer for a draft, returning VPM-style dimensions.
        Guarded against empty/degenerate inputs so the pipeline never crashes.
        """
        # --- Load & normalize ---
        try:
            text = (draft_path.read_text() or "").strip()
        except Exception:
            text = ""
        # Safe default when text is empty
        if not text:
            return {
                "coverage": 0.0,
                "correctness": 0.0,
                "coherence": 0.0,
                "citation_support": 0.0,
                "entity_consistency": 0.0,
                "readability": 10.0,  # neutral band default
                "fkgl_raw": 10.0,
                "novelty": 0.5,
                "stickiness": 0.0,
                "len_chars": 0.0,
                "compactness": 1.0,
            }

        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()
        ]
        units = plan.get("units", [])

        # --- Coverage ---
        ids = [u.get("claim_id") for u in units if u.get("claim_id")]
        covered_ids = (
            sum(1 for cid in ids if f"[#{cid}]" in text) if ids else 0
        )
        if ids:
            coverage = covered_ids / max(1, len(ids))
        else:
            # Fallback: fuzzy overlap between unit claims and text terms
            unit_terms = [
                set(
                    re.findall(
                        r"\b[a-zA-Z]{5,}\b", (u.get("claim") or "").lower()
                    )
                )
                for u in units
            ]
            text_terms = set(re.findall(r"\b[a-zA-Z]{5,}\b", text.lower()))
            if unit_terms:
                hits = sum(
                    1
                    for terms in unit_terms
                    if terms and len(terms & text_terms) / len(terms) >= 0.5
                )
                coverage = hits / max(1, len(unit_terms))
            else:
                coverage = 0.0

        # --- Citation support / correctness proxy ---
        factual = [s for s in sentences if self._is_factual_sentence(s)]
        cited = sum(1 for s in factual if "[#]" in s)
        citation_support = (cited / max(1, len(factual))) if factual else 1.0
        correctness = citation_support  # proxy until abstract-alignment or judge is plugged in

        # --- Entity consistency (ABBR handling) ---
        abbrs = plan.get("entities", {}).get("ABBR", {}) or {}
        entity_consistency = 1.0
        if abbrs:
            for full, abbr in abbrs.items():
                # If neither appears, penalize (missing entity mention)
                if full not in text and abbr not in text:
                    entity_consistency = min(entity_consistency, 0.0)
                # If full term repeats many times without abbreviation after first use, light penalty
                elif text.count(full) > 1:
                    entity_consistency = min(entity_consistency, 0.5)

        # --- Readability (FKGL) ---
        words = re.findall(r"[A-Za-z]+", text)
        num_words = max(1, len(words))
        num_sentences = max(1, len(sentences))
        syllables = sum(self._count_syllables(w) for w in words) or 1
        fkgl_raw = (
            0.39 * (num_words / num_sentences)
            + 11.8 * (syllables / num_words)
            - 15.59
        )
        # Clamp to a sane band for readability score used by edit policy
        readability = float(max(6.0, min(15.0, fkgl_raw)))

        # --- Coherence (adjacency Jaccard) + title drift penalty ---
        coh_scores = []
        for i in range(len(sentences) - 1):
            s1 = set(re.findall(r"\w+", sentences[i].lower()))
            s2 = set(re.findall(r"\w+", sentences[i + 1].lower()))
            denom = len(s1 | s2)
            coh_scores.append((len(s1 & s2) / denom) if denom else 1.0)
        coherence = (
            sum(coh_scores) / max(1, len(coh_scores)) if coh_scores else 1.0
        )

        title_terms = set(
            re.findall(
                r"\b[a-zA-Z]{5,}\b", (plan.get("section_title") or "").lower()
            )
        )
        text_terms = set(re.findall(r"\b[a-zA-Z]{5,}\b", text.lower()))
        drift_penalty = (
            0.0
            if not title_terms
            else max(
                0.0,
                0.2 - len(title_terms & text_terms) / max(1, len(title_terms)),
            )
        )
        coherence = max(0.0, min(1.0, coherence - drift_penalty))

        # --- Stickiness (plan-term retention) ---
        stickiness = self._compute_stickiness(text, plan)

        # --- Compactness / size ---
        len_chars = float(len(text))
        compactness = len(re.sub(r"\s+", " ", text)) / max(
            1, len(text)
        )  # ~1 means minimal extra whitespace

        # --- Novelty (placeholder): prefer some lexical variety over near-duplication ---
        # Ratio of unique sentences to total sentences, softened into [0.3, 1.0]
        uniq_sent_ratio = len(set(sentences)) / max(1, len(sentences))
        novelty = max(0.3, min(1.0, 0.6 + 0.4 * uniq_sent_ratio))


        kg_neighbors = (plan.get("kg") or {}).get("neighbors", {})
        cid_set = {u.get("claim_id") for u in plan.get("units", []) if u.get("claim_id")}
        if kg_neighbors and cid_set:
            # count claims that have a strong neighbor
            strong = 0
            max_sim = 0.0
            for cid in cid_set:
                hits = kg_neighbors.get(cid) or []
                if hits:
                    best = max(h["score"] for h in hits)
                    max_sim = max(max_sim, best)
                    if best >= 0.75:  # tune threshold
                        strong += 1
            kg_support = strong / max(1, len(cid_set))
            novelty = 1.0 - max_sim  # crude novelty proxy
        else:
            kg_support = 0.0
            novelty = 0.6  # prior

        # blend KG support into correctness (optional)
        correctness = 0.5 * citation_support + 0.5 * kg_support


        return {
            "kg_support": float(kg_support), 
            "coverage": float(coverage),
            "correctness": float(correctness),
            "coherence": float(coherence),
            "citation_support": float(citation_support),
            "entity_consistency": float(entity_consistency),
            "readability": float(readability),
            "fkgl_raw": float(fkgl_raw),
            "novelty": float(novelty),
            "stickiness": float(stickiness),
            "len_chars": float(len_chars),
            "compactness": float(compactness),
        }

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

    # --------------------------- edits ---------------------------

    def _apply_edit_policy(
        self,
        draft_path: Path,
        plan: Dict[str, Any],
        max_edits: int = 6,
        trace_path: Optional[Path] = None,
    ):
        text = draft_path.read_text()
        edits: List[str] = []

        for i in range(max_edits):
            scores = self._score_draft(draft_path, plan)
            change = False

            if scores["coverage"] < 0.8:
                missing = [
                    u
                    for u in plan.get("units", [])
                    if u.get("claim_id") and f"[#{u['claim_id']}]" not in text
                ]
                if missing:
                    u = missing[0]
                    line = f"- {u.get('claim', 'Claim')} [#{u['claim_id']}].\n  *Evidence: {u.get('evidence', 'See paper')}* [#]\n\n"
                    text = text.rstrip() + "\n" + line
                    edits.append(f"add_claim:{u['claim_id']}")
                    change = True

            if not change and scores["citation_support"] < 0.7:
                sentences = re.split(r"(?<=[.!?])\s+", text)
                for j, s in enumerate(sentences):
                    if self._is_factual_sentence(s) and "[#]" not in s:
                        sentences[j] = s.rstrip() + " [#]"
                        text = " ".join(sentences)
                        edits.append("add_citation_marker")
                        change = True
                        break

            if not change and scores["entity_consistency"] < 1.0:
                abbrs = plan.get("entities", {}).get("ABBR", {})
                for full, abbr in abbrs.items():
                    if full not in text and abbr in text:
                        text = re.sub(
                            rf"\b{re.escape(abbr)}\b",
                            f"{full} ({abbr})",
                            text,
                            count=1,
                        )
                        edits.append(f"expand_abbr:{full}->{abbr}")
                        change = True
                        break
                    if text.count(full) > 1:
                        first = True

                        def _swap(m):
                            nonlocal first
                            if first:
                                first = False
                                return m.group(0)
                            return abbr

                        text = re.sub(rf"\b{re.escape(full)}\b", _swap, text)
                        edits.append(f"abbreviate_repeats:{full}->{abbr}")
                        change = True
                        break

            if not change and not (9.0 <= scores["readability"] <= 11.0):
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
                text = re.sub(r"\n- ([^.\n]{0,60})\.\n- ", r"\n- \1; ", text)
                if text != before:
                    edits.append("merge_adjacent_bullets")
                    change = True
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
                    f.write(
                        json.dumps(
                            {"edit": i + 1, "scores": scores, "op": edits[-1]}
                        )
                        + "\n"
                    )

        return text, edits

    def _regenerate_lead_in(self, text: str, plan: Dict[str, Any]) -> str:
        m = re.search(r"^# .+\n\n(.+?)\n\n", text, flags=re.DOTALL)
        if not m:
            return text
        start, end = m.span(1)
        title = plan.get("section_title", "Section")
        abbrs = plan.get("entities", {}).get("ABBR", {})
        new_lead = self._lead_paragraph(title, plan.get("units", []), abbrs)
        return text[:start] + new_lead + text[end:]

    # --------------------------- helpers ---------------------------

    def _is_factual_sentence(self, s: str) -> bool:
        s_low = s.lower()
        return any(kw in s_low for kw in FACTUAL_KWS)

    def _build_vpm_row(
        self,
        initial: Dict[str, float],
        final: Dict[str, float],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "section": plan.get("section_title", "unknown"),
            "coverage_initial": round(initial.get("coverage", 0.0), 3),
            "coverage_final": round(final.get("coverage", 0.0), 3),
            "correctness": round(final.get("correctness", 0.0), 3),
            "coherence": round(final.get("coherence", 0.0), 3),
            "citation_support": round(final.get("citation_support", 0.0), 3),
            "entity_consistency": round(
                final.get("entity_consistency", 0.0), 3
            ),
            "readability": round(final.get("readability", 0.0), 2),
            "fkgl_raw": round(final.get("fkgl_raw", 0.0), 2),
            "novelty": round(final.get("novelty", 0.0), 3),
            "stickiness": round(final.get("stickiness", 0.0), 3),
            "len_chars": int(final.get("len_chars", 0)),
            "compactness": round(final.get("compactness", 0.0), 3),
        }

    def _extract_plan_slice(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "section_title": plan.get("section_title"),
            "claim_count": len(plan.get("units", [])),
            "required_entities": plan.get("entities", {}).get("REQUIRED", []),
            "abbr": plan.get("entities", {}).get("ABBR", {}),
        }

    def _count_syllables(self, word: str) -> int:
        word = (word or "").lower()
        vowels = "aeiouy"
        if not word:
            return 1
        count = 1 if word[0] in vowels else 0
        for idx in range(1, len(word)):
            if word[idx] in vowels and word[idx - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        return max(1, count)

    def _normalize_ws(self, s: str) -> str:
        s = re.sub(r"[ \t]+\n", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip() + "\n"

    def _has_duplicate_bullets(self, text: str) -> bool:
        bullets = re.findall(r"^- .+$", text, flags=re.MULTILINE)
        return len(bullets) != len(set(bullets))

    def _dedup_bullets(self, text: str) -> str:
        lines = text.splitlines()
        seen = set()
        out = []
        for ln in lines:
            if ln.startswith("- "):
                if ln in seen:
                    continue
                seen.add(ln)
            out.append(ln)
        return "\n".join(out) + ("\n" if not out or out[-1] != "" else "")

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        if self.logger and hasattr(self.logger, "log"):
            try:
                self.logger.log(event, payload)
                return
            except Exception:
                pass
        # Fallback
        print(f"[{event}] {payload}")

    def _log_calibration_event(
        self,
        *,
        domain: str,
        query: str | None,
        raw_sim: float,
        is_relevant: bool,
        scorable_id: str | None = None,
        scorable_type: str | None = None,
        entity_type: str | None = None,
    ) -> None:
        # Ensure non-null & length-limited strings to satisfy NOT NULL constraints
        q = (query or "").strip()[:2000] or "N/A"
        sid = (scorable_id or "").strip() or "unknown"
        st = (scorable_type or "").strip() or "unknown"
        et = (entity_type or "").strip() or None
        try:
            self.calibration.log_event(
                domain=domain or "general",
                query=q,
                raw_sim=float(raw_sim),
                is_relevant=bool(is_relevant),
                scorable_id=sid,
                scorable_type=st,
                entity_type=et,
            )
        except Exception as e:
            self.logger.error("CalibrationEventLogFailed", {"error": str(e)})
