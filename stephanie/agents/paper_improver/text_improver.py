# stephanie/agents/paper_improver/text_improver.py

# text_improver.py — plan → draft → score → edit → log → blog-ready
import json
import re
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List

class TextImprover:
    def __init__(self, workdir: str = "./text_runs", timeout: int = 60):
        self.workdir = Path(workdir)
        self.workdir.mkdir(exist_ok=True)
        self.run_id = 0
        self.timeout = timeout

    def improve(self, content_plan: Dict[str, Any]) -> Dict[str, Any]:
        self.run_id += 1
        plan_hash = hashlib.sha256(
            json.dumps(content_plan, sort_keys=True).encode()
        ).hexdigest()[:8]
        run_dir = self.workdir / f"run_{self.run_id}_{plan_hash}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # metadata for determinism
        meta = {
            "plan_sha": plan_hash,
            "seeds": {"python": 0},
            "timeout": self.timeout,
            "timestamp": time.time(),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # 1. save plan
        plan_path = run_dir / "plan.json"
        plan_path.write_text(json.dumps(content_plan, indent=2))

        # 2. initial draft
        draft_path = self._generate_draft(content_plan, run_dir)

        # 3. score draft
        initial_score = self._score_draft(draft_path, content_plan)

        # 4. edit loop
        final_draft, edits = self._apply_edit_policy(draft_path, content_plan, max_edits=5)

        # 5. rescore
        final_score = self._score_draft(draft_path, content_plan)

        # 6. vpm row
        vpm_row = self._build_vpm_row(initial_score, final_score, content_plan)

        # 7. dpo pair
        dpo_pair = {
            "content_plan_slice": self._extract_plan_slice(content_plan),
            "prompt": "Generate faithful, clear prose from this plan.",
            "rejected": (run_dir / "initial_draft.md").read_text(),
            "chosen": final_draft,
            "metadata": {
                "run_id": self.run_id,
                "plan_hash": plan_hash,
                "initial_scores": initial_score,
                "final_scores": final_score,
                "applied_edits": edits,
            },
        }
        (run_dir / "text_dpo_pair.json").write_text(json.dumps(dpo_pair, indent=2))

        return {
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "final_draft_path": str(draft_path),
            "vpm_row": vpm_row,
            "dpo_pair_path": str(run_dir / "text_dpo_pair.json"),
            "scores": final_score,
            "passed": all(
                final_score[d] >= 0.7
                for d in ["coverage", "correctness", "coherence"]
            ),
        }

    # ---------------- draft + scoring ----------------

    def _generate_draft(self, plan: Dict[str, Any], run_dir: Path) -> Path:
        draft = f"# {plan.get('section_title', 'Section')}\n\n"
        for unit in plan.get("units", []):
            claim = unit.get("claim", "No claim")
            evidence = unit.get("evidence", "See paper")
            claim_id = unit.get("claim_id", "")
            tag = f" [#{claim_id}]" if claim_id else ""
            cite = " [#]" if evidence and evidence != "See paper" else ""
            draft += f"- {claim}{tag}.\n"
            draft += f"  *Evidence: {evidence}*{cite}\n\n"
        draft_path = run_dir / "draft.md"
        draft_path.write_text(draft)
        (run_dir / "initial_draft.md").write_text(draft)
        return draft_path

    def _score_draft(self, draft_path: Path, plan: Dict[str, Any]) -> Dict[str, float]:
        text = draft_path.read_text()
        units = plan.get("units", [])
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        # coverage: % of claims referenced
        covered = sum(1 for u in units if u.get("claim_id") and f"[#{u['claim_id']}]" in text)
        coverage = covered / max(1, len(units))

        # citation support
        factual_kw = ("show","prove","result","achiev","increase","decrease","outperform","error","accuracy","loss")
        factual_sentences = [s for s in sentences if any(kw in s.lower() for kw in factual_kw)]
        cited = sum(1 for s in factual_sentences if "[#]" in s)
        citation_support = cited / max(1, len(factual_sentences)) if factual_sentences else 1.0

        # entity consistency
        abbrs = plan.get("entities", {}).get("ABBR", {})
        entity_consistency = 1.0
        for full, abbr in abbrs.items():
            if full not in text and abbr not in text:
                entity_consistency = min(entity_consistency, 0.0)
            elif full in text and text.count(full) > 1:
                entity_consistency = min(entity_consistency, 0.5)

        # readability: FKGL
        words = re.findall(r"[A-Za-z]+", text)
        num_words = max(1, len(words))
        num_sentences = max(1, len(sentences))
        syllables = sum(self._count_syllables(w) for w in words)
        fkgl = 0.39 * (num_words/num_sentences) + 11.8 * (syllables/num_words) - 15.59
        readability = float(max(6.0, min(15.0, fkgl)))

        # coherence: jaccard similarity between adjacent sentences
        coh_scores = []
        for i in range(len(sentences)-1):
            s1 = set(re.findall(r'\w+', sentences[i].lower()))
            s2 = set(re.findall(r'\w+', sentences[i+1].lower()))
            denom = len(s1 | s2)
            coh_scores.append((len(s1 & s2)/denom) if denom else 1.0)
        coherence = sum(coh_scores) / max(1, len(coh_scores)) if coh_scores else 1.0

        correctness = citation_support
        novelty = 0.6  # placeholder until abstract-sim is wired

        return {
            "coverage": coverage,
            "correctness": correctness,
            "coherence": coherence,
            "citation_support": citation_support,
            "entity_consistency": entity_consistency,
            "readability": readability,
            "novelty": novelty,
        }

    # ---------------- edit policy ----------------

    def _apply_edit_policy(self, draft_path: Path, plan: Dict[str, Any], max_edits: int = 5):
        text = draft_path.read_text()
        edits: List[str] = []

        for _ in range(max_edits):
            scores = self._score_draft(draft_path, plan)
            made_change = False

            # add missing claims
            if scores["coverage"] < 0.8:
                missing = [u for u in plan.get("units", [])
                           if u.get("claim_id") and f"[#{u['claim_id']}]" not in text]
                if missing:
                    u = missing[0]
                    line = f"- {u.get('claim','Claim')} [#{u['claim_id']}].\n  *Evidence: {u.get('evidence','See paper')}* [#]\n\n"
                    text += line
                    edits.append(f"Add claim {u['claim_id']}")
                    made_change = True

            # add citation markers
            if not made_change and scores["citation_support"] < 0.7:
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for j, s in enumerate(sentences):
                    if self._is_factual_sentence(s) and "[#]" not in s:
                        sentences[j] = s.rstrip() + " [#]"
                        text = " ".join(sentences)
                        edits.append("Add citation marker")
                        made_change = True
                        break

            # normalize ABBRs
            if not made_change and scores["entity_consistency"] < 1.0:
                abbrs = plan.get("entities", {}).get("ABBR", {})
                for full, abbr in abbrs.items():
                    if full not in text and abbr in text:
                        text = re.sub(rf"\b{re.escape(abbr)}\b", f"{full} ({abbr})", text, count=1)
                        edits.append(f"Expand ABBR: {full} ({abbr})")
                        made_change = True
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
                        edits.append(f"Normalize ABBR usage: {full}→{abbr}")
                        made_change = True
                        break

            # readability fix
            if not made_change and not (9.0 <= scores["readability"] <= 11.0):
                if scores["readability"] > 11.0:
                    text = re.sub(r";\s+", ". ", text)
                    text = re.sub(r"\s+and\s+", ". ", text, count=1)
                    edits.append("Split long sentences")
                    made_change = True
                else:
                    text = re.sub(r"\.\s+([a-z])", r", \1", text, count=1)
                    edits.append("Join short sentences")
                    made_change = True

            # coherence smoothing
            if not made_change and scores["coherence"] < 0.7:
                text_before = text
                text = re.sub(r"\n- ([^.\n]{0,40})\.\n- ", r"\n- \1; ", text)
                if text != text_before:
                    edits.append("Merge short adjacent bullets")
                    made_change = True

            if not made_change:
                break

            draft_path.write_text(text)

        return text, edits

    def _is_factual_sentence(self, s: str) -> bool:
        s_low = s.lower()
        return any(kw in s_low for kw in (
            "show","prove","result","achiev","increase","decrease","outperform","error","accuracy","loss"
        ))

    # ---------------- helpers ----------------

    def _build_vpm_row(self, initial: Dict[str,float], final: Dict[str,float], plan: Dict[str,Any]) -> Dict[str, Any]:
        return {
            "section": plan.get("section_title","unknown"),
            "coverage_initial": round(initial["coverage"],3),
            "coverage_final": round(final["coverage"],3),
            "correctness": round(final["correctness"],3),
            "coherence": round(final["coherence"],3),
            "citation_support": round(final["citation_support"],3),
            "entity_consistency": round(final["entity_consistency"],3),
            "readability": round(final["readability"],2),
            "novelty": round(final["novelty"],3),
        }

    def _extract_plan_slice(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "section_title": plan.get("section_title"),
            "claim_count": len(plan.get("units", [])),
            "required_entities": plan.get("entities", {}).get("REQUIRED", []),
            "abbr": plan.get("entities", {}).get("ABBR", {})
        }

    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        if word and word[0] in vowels:
            count += 1
        for idx in range(1, len(word)):
            if word[idx] in vowels and word[idx - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        return max(1, count)
