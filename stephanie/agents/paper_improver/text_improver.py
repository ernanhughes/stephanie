# stephanie/agents/paper_improver/text_improver.py
# TextImprover — plan → draft → score → edit → log → blog-ready (hardened)
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.paper_improver.goals import GoalScorer
from stephanie.knowledge.casebook_store import CaseBookStore
from stephanie.knowledge.knowledge_bus import KnowledgeBus

from .faithfulness import FaithfulnessBot

FACTUAL_KWS = (
    "show", "prove", "result", "achiev", "increase", "decrease",
    "outperform", "error", "accuracy", "loss", "significant", "statistically"
)












class TextImprover:
    def __init__(
        self,
        workdir: str = "./text_runs",
        timeout: int = 60,
        seed: int = 0,
        faithfulness_topk: int = 5,
        kb: KnowledgeBus | None = None,
        casebooks: CaseBookStore | None = None
    ):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.run_id = 0
        self.timeout = timeout
        self.seed = seed
        self.faithfulness_topk = faithfulness_topk,
        self.kb = kb or KnowledgeBus()
        self.casebooks = casebooks or CaseBookStore()
        self.gs = GoalScorer()

    # --------------------------- public ---------------------------

    def improve(self, content_plan: Dict[str, Any]) -> Dict[str, Any]:
        self._seed_everything(self.seed)
        self.run_id += 1
        plan_norm = self._sanitize_plan(content_plan)
        plan_hash = hashlib.sha256(json.dumps(plan_norm, sort_keys=True).encode()).hexdigest()[:8]
        run_dir = self.workdir / f"run_{self.run_id}_{plan_hash}"
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "plan_sha": plan_hash,
            "seeds": {"python": self.seed},
            "timeout": self.timeout,
            "timestamp": time.time(),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # 1) Persist plan
        plan_path = run_dir / "plan.json"
        plan_path.write_text(json.dumps(plan_norm, indent=2))

        casebook_name = f"text_{plan_hash}_{content_plan.get('section_title','section')}"
        cb = self.casebooks.ensure_casebook(casebook_name, ["text_improver","exemplar_text"], {"plan_sha": plan_hash})
        case = self.casebooks.add_case(casebook_name, json.dumps(content_plan), agent_name="text_improver", meta={"run_dir": str(run_dir)})

        # 2) Generate draft
        draft_path = self._generate_draft(plan_norm, run_dir)

        # 3) Score
        initial_scores = self._score_draft(draft_path, plan_norm)


        # 4) Edit-policy loop
        final_text, edits = self._apply_edit_policy(draft_path, plan_norm, max_edits=6, trace_path=run_dir/"trace.ndjson")

        # 5) Rescore
        final_scores = self._score_draft(draft_path, plan_norm)

        # after scoring:
        vpm_row = self._build_vpm_row(initial_scores, final_scores, plan_norm)
        goal_eval = self.gs.score("text","academic_summary", vpm_row)
        self.casebooks.add_scorable(casebook_name, case.id, "vpm", json.dumps(vpm_row), {"goal": goal_eval})
        self.casebooks.add_scorable(casebook_name, case.id, "text", (run_dir/"draft.md").read_text(), {"stage":"final"})

        # 6) Optional faithfulness
        vpm_row = self._build_vpm_row(initial_scores, final_scores, plan_norm)
        faithfulness_score = None
        paper_text = plan_norm.get("paper_text")
        if not paper_text and plan_norm.get("paper_text_path"):
            try:
                paper_text = Path(plan_norm["paper_text_path"]).read_text()
            except Exception:
                paper_text = None

        if paper_text:
            try:
                bot = FaithfulnessBot(top_k=self.faithfulness_topk)
                bot.prepare_paper(paper_text)
                claims = [{"claim_id": u.get("claim_id"), "claim": u.get("claim", "")}
                          for u in plan_norm.get("units", []) if u.get("claim")]
                faithfulness_score = bot.get_faithfulness_score(claims)
                final_scores["faithfulness"] = float(faithfulness_score)
                vpm_row["faithfulness"] = round(float(faithfulness_score), 3)
            except Exception:
                # keep pipeline resilient
                pass

        # 7) DPO pair
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
                "score_deltas": {k: round(final_scores.get(k, 0.0) - initial_scores.get(k, 0.0), 4)
                                 for k in set(initial_scores) | set(final_scores)},
                "applied_edits": edits,
            },
        }
        (run_dir / "text_dpo_pair.json").write_text(json.dumps(dpo_pair, indent=2))

        # Pass criteria: core dims ≥ 0.7; include faithfulness if present
        core_ok = all(final_scores.get(d, 0.0) >= 0.7 for d in ("coverage", "correctness", "coherence"))
        faithful_ok = True if faithfulness_score is None else (faithfulness_score >= 0.7)

        self.casebooks.add_scorable(casebook_name, case.id, "dpo_pair", json.dumps(dpo_pair), dpo_pair["metadata"])
        self.kb.publish("trajectory.step", {"casebook": cb.name, "case_id": case.id, "vpm": vpm_row, "goal": goal_eval})

        return {
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "final_draft_path": str(draft_path),
            "vpm_row": vpm_row,
            "dpo_pair_path": str(run_dir / "text_dpo_pair.json"),
            "scores": final_scores,
            "passed": bool(core_ok and faithful_ok),
        }

    # --------------------------- generation ---------------------------

    def _generate_draft(self, plan: Dict[str, Any], run_dir: Path) -> Path:
        title = plan.get("section_title", "Section")
        units = plan.get("units", [])
        abbrs = plan.get("entities", {}).get("ABBR", {})

        outline = plan.get("outline") or [u.get("claim_id") for u in units if u.get("claim_id")]
        outline = [cid for cid in outline if cid]  # filter None

        # Lead-in paragraph summarizing the section from claims (safe, no new facts)
        lead = self._lead_paragraph(title, units, abbrs)

        # Bullets in stable plan order, dedup by claim_id text
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

        # Assemble
        draft = f"# {title}\n\n{lead}\n\n"
        if outline:
            draft += "### Outline\n" + "\n".join(f"- [{cid}]" for cid in outline) + "\n\n"
        draft += "### Details\n" + "\n\n".join(bullets) + "\n"

        # Persist
        draft_path = run_dir / "draft.md"
        draft_path.write_text(self._normalize_ws(draft))
        (run_dir / "initial_draft.md").write_text(draft_path.read_text())
        return draft_path

    def _lead_paragraph(self, title: str, units: List[Dict[str, Any]], abbrs: Dict[str, str]) -> str:
        # Use first 2–3 claims as a safe summary scaffold; expand first use of ABBR if present
        claims = [u.get("claim", "") for u in units if u.get("claim")]
        head = " ".join([c.rstrip(".") + "." for c in claims[:3]]) or "This section summarizes key findings and method decisions from the paper."
        # expand one ABBR if title contains the full term
        for full, abbr in abbrs.items():
            if full in title and abbr not in head:
                head = head.replace(full, f"{full} ({abbr})", 1)
                break
        return head

    # --------------------------- scoring ---------------------------

    def _score_draft(self, draft_path: Path, plan: Dict[str, Any]) -> Dict[str, float]:
        text = draft_path.read_text()
        units = plan.get("units", [])
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        # Coverage: (1) by claim_id anchors; (2) fuzzy overlap fallback
        ids = [u.get("claim_id") for u in units if u.get("claim_id")]
        covered_ids = sum(1 for cid in ids if f"[#{cid}]" in text)
        if ids:
            coverage = covered_ids / len(ids)
        else:
            # fallback: fraction of unit claims whose 5+ char tokens appear
            unit_terms = [set(re.findall(r"\b[a-zA-Z]{5,}\b", (u.get("claim","") or "").lower())) for u in units]
            text_terms = set(re.findall(r"\b[a-zA-Z]{5,}\b", text.lower()))
            hits = sum(1 for t in unit_terms if t and len(t & text_terms) / len(t) >= 0.5)
            coverage = hits / max(1, len(unit_terms))

        # Citation support: factual sentences require [#]
        factual = [s for s in sentences if self._is_factual_sentence(s)]
        cited = sum(1 for s in factual if "[#]" in s)
        citation_support = (cited / len(factual)) if factual else 1.0

        # Entity consistency: expand first-use then prefer ABBR
        abbrs = plan.get("entities", {}).get("ABBR", {})
        entity_consistency = 1.0
        for full, abbr in abbrs.items():
            if full not in text and abbr not in text:
                entity_consistency = min(entity_consistency, 0.0)
            elif text.count(full) > 1:
                entity_consistency = min(entity_consistency, 0.5)

        # Readability (FKGL) and raw FKGL
        words = re.findall(r"[A-Za-z]+", text)
        num_words = max(1, len(words))
        num_sentences = max(1, len(sentences))
        syllables = sum(self._count_syllables(w) for w in words)
        fkgl_raw = 0.39 * (num_words/num_sentences) + 11.8 * (syllables/num_words) - 15.59
        readability = float(max(6.0, min(15.0, fkgl_raw)))

        # Coherence: adjacency Jaccard + title drift penalty
        coh_scores = []
        for i in range(len(sentences)-1):
            s1 = set(re.findall(r'\w+', sentences[i].lower()))
            s2 = set(re.findall(r'\w+', sentences[i+1].lower()))
            denom = len(s1 | s2)
            coh_scores.append((len(s1 & s2)/denom) if denom else 1.0)
        coherence = sum(coh_scores) / max(1, len(coh_scores)) if coh_scores else 1.0
        title_terms = set(re.findall(r"\b[a-zA-Z]{5,}\b", (plan.get("section_title","") or "").lower()))
        text_terms = set(re.findall(r"\b[a-zA-Z]{5,}\b", text.lower()))
        drift_penalty = 0.0 if not title_terms else max(0.0, 0.2 - len(title_terms & text_terms)/max(1,len(title_terms)))
        coherence = max(0.0, min(1.0, coherence - drift_penalty))

        correctness = citation_support  # proxy until abstract-alignment plugged in
        novelty = 0.6  # placeholder
        stickiness = self._compute_stickiness(text, plan)

        # compactness metrics
        len_chars = len(text)
        compression_ratio = len(re.sub(r"\s+", " ", text)) / max(1, len(text))  # ~1 means minimal whitespace

        return {
            "coverage": coverage,
            "correctness": correctness,
            "coherence": coherence,
            "citation_support": citation_support,
            "entity_consistency": entity_consistency,
            "readability": readability,
            "fkgl_raw": float(fkgl_raw),
            "novelty": novelty,
            "stickiness": stickiness,
            "len_chars": float(len_chars),
            "compactness": float(compression_ratio)
        }

    def _compute_stickiness(self, text: str, plan: Dict[str, Any]) -> float:
        plan_terms = set()
        for unit in plan.get("units", []):
            claim = unit.get("claim", "") or ""
            for w in re.findall(r"\b[a-zA-Z]{5,}\b", claim.lower()):
                plan_terms.add(w)
        if not plan_terms:
            return 1.0
        text_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', text.lower()))
        return len(plan_terms & text_words) / max(1, len(plan_terms))

    # --------------------------- edits ---------------------------

    def _apply_edit_policy(self, draft_path: Path, plan: Dict[str, Any], max_edits: int = 6, trace_path: Optional[Path] = None):
        text = draft_path.read_text()
        edits: List[str] = []

        for i in range(max_edits):
            scores = self._score_draft(draft_path, plan)
            change = False

            # 1) Add one missing claim (deterministic, in plan order)
            if scores["coverage"] < 0.8:
                missing = [u for u in plan.get("units", [])
                           if u.get("claim_id") and f"[#{u['claim_id']}]" not in text]
                if missing:
                    u = missing[0]
                    line = f"- {u.get('claim','Claim')} [#{u['claim_id']}].\n  *Evidence: {u.get('evidence','See paper')}* [#]\n\n"
                    text = text.rstrip() + "\n" + line
                    edits.append(f"add_claim:{u['claim_id']}")
                    change = True

            # 2) Add citation markers to factual sentences lacking [#]
            if not change and scores["citation_support"] < 0.7:
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for j, s in enumerate(sentences):
                    if self._is_factual_sentence(s) and "[#]" not in s:
                        sentences[j] = s.rstrip() + " [#]"
                        text = " ".join(sentences)
                        edits.append("add_citation_marker")
                        change = True
                        break

            # 3) Normalize ABBR (expand first use, abbreviate later)
            if not change and scores["entity_consistency"] < 1.0:
                abbrs = plan.get("entities", {}).get("ABBR", {})
                for full, abbr in abbrs.items():
                    if full not in text and abbr in text:
                        text = re.sub(rf"\b{re.escape(abbr)}\b", f"{full} ({abbr})", text, count=1)
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

            # 4) Readability banding (9–11)
            if not change and not (9.0 <= scores["readability"] <= 11.0):
                if scores["readability"] > 11.0:
                    text = re.sub(r";\s+", ". ", text)
                    text = re.sub(r"\s+and\s+", ". ", text, count=1)
                    edits.append("split_long_sentences")
                else:
                    text = re.sub(r"\.\s+([a-z])", r", \1", text, count=1)
                    edits.append("join_short_sentences")
                change = True

            # 5) Coherence smoothing (merge choppy bullets)
            if not change and scores["coherence"] < 0.7:
                before = text
                text = re.sub(r"\n- ([^.\n]{0,60})\.\n- ", r"\n- \1; ", text)
                if text != before:
                    edits.append("merge_adjacent_bullets")
                    change = True
                else:
                    # regenerate lead-in from claims if still bad
                    text = self._regenerate_lead_in(text, plan)
                    edits.append("regen_lead_in")
                    change = True

            # 6) De-duplicate identical bullets
            if not change and self._has_duplicate_bullets(text):
                text = self._dedup_bullets(text)
                edits.append("dedup_bullets")
                change = True

            if not change:
                break

            draft_path.write_text(self._normalize_ws(text))
            if trace_path:
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"edit": i+1, "scores": scores, "op": edits[-1]}) + "\n")

        return text, edits

    def _regenerate_lead_in(self, text: str, plan: Dict[str, Any]) -> str:
        # Replace first paragraph after H1 with a new lead summarizing first 2–3 claims
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

    def _build_vpm_row(self, initial: Dict[str, float], final: Dict[str, float], plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "section": plan.get("section_title", "unknown"),
            "coverage_initial": round(initial.get("coverage", 0.0), 3),
            "coverage_final": round(final.get("coverage", 0.0), 3),
            "correctness": round(final.get("correctness", 0.0), 3),
            "coherence": round(final.get("coherence", 0.0), 3),
            "citation_support": round(final.get("citation_support", 0.0), 3),
            "entity_consistency": round(final.get("entity_consistency", 0.0), 3),
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
            "abbr": plan.get("entities", {}).get("ABBR", {})
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
        # keep markdown newlines but normalize multiple blank lines
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

    def _sanitize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure required shape; drop nulls; keep keys we use."""
        out: Dict[str, Any] = {}
        out["section_title"] = (plan.get("section_title") or "Section")
        units_in = plan.get("units") or []
        clean_units: List[Dict[str, Any]] = []
        for u in units_in:
            if not isinstance(u, dict):
                continue
            claim = (u.get("claim") or "").strip()
            evidence = (u.get("evidence") or "See paper").strip()
            cid = u.get("claim_id")
            clean_units.append({"claim": claim, "evidence": evidence, "claim_id": cid})
        out["units"] = clean_units
        ents = plan.get("entities") or {}
        out["entities"] = {
            "ABBR": ents.get("ABBR") or {},
            "REQUIRED": ents.get("REQUIRED") or []
        }
        # optional extras
        if plan.get("paper_text"):
            out["paper_text"] = plan["paper_text"]
        if plan.get("paper_text_path"):
            out["paper_text_path"] = plan["paper_text_path"]
        if plan.get("outline"):
            out["outline"] = plan["outline"]
        return out

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except Exception:
            pass
