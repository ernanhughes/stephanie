# stephanie/agents/thought/anti_hallucination.py
"""
AntiHallucination
-----------------
Hard guardrails against unsupported claims in generated content.
Fails verification for sections with hallucinated content.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple


class AntiHallucination:
    """
    Hard guardrails against unsupported claims in generated content.
    
    Usage:
        ah = AntiHallucination(logger)
        is_valid, issues = ah.verify_section(section, knowledge_tree)
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.hallucination_patterns = [
            # Quantitative claims without citations
            r"\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|F1|AUC|RMSE|MAE|BLEU)\b(?!.*\b(figure|fig|table|tbl)\b)",
            # Claims about paper content not in paper
            r"\bthe paper shows|the authors demonstrate|the study proves\b(?!.*\b(claim|insight)\b)",
            # Fabricated references
            r"\bsee \[.*?\d+.*?\]",  # [1], [2], etc. that don't match paper
            r"\bcited in \d{4}\b",   # "cited in 2023" when paper is from 2022
        ]
    
    # verify_section should already wrap these calls and return (bool, issues).
    # If yours raises, keep a try/except and soft-pass with a diagnostic token.

    def _entity_text(self, e) -> str:
        """Return lowercase text for an entity-like object (dict or str)."""
        if isinstance(e, str):
            return e.strip()
        if isinstance(e, dict):
            return (e.get("text")
                    or e.get("name")
                    or e.get("label")
                    or e.get("value")
                    or "").strip()
        return ""

    def _normalize_tree(self, tree: dict) -> dict:
        """Coerce various KG/KT shapes into a stable schema we can consume."""
        t = tree or {}

        # entities may live under 'entities' or 'nodes'
        raw_entities = t.get("entities")
        if raw_entities is None:
            raw_entities = t.get("nodes") or []

        entities = []
        for e in raw_entities if isinstance(raw_entities, (list, tuple)) else []:
            txt = self._entity_text(e)
            if txt:
                entities.append({"text": txt})

        # claims are sometimes strings or dicts with various keys
        raw_claims = t.get("claims") or []
        claims = []
        for c in raw_claims if isinstance(raw_claims, (list, tuple)) else []:
            if isinstance(c, str):
                claims.append({"text": c.strip()})
            elif isinstance(c, dict):
                ctxt = (c.get("text") or c.get("claim") or c.get("value") or "").strip()
                if ctxt:
                    claims.append({"text": ctxt})

        # relationships (optional – pass through as-is if list[dict])
        relationships = t.get("relationships")
        if not isinstance(relationships, list):
            relationships = []

        return {
            "entities": entities,                # [{text}]
            "claims": claims,                    # [{text}]
            "relationships": relationships,      # [dict, ...]
            "claim_coverage": float(t.get("claim_coverage", 0.0) or 0.0),
            "evidence_strength": float(t.get("evidence_strength", 0.0) or 0.0),
        }

    def _as_list(self, x):
        return x if isinstance(x, (list, tuple)) else []

    def verify_section(self, 
                      section: str, 
                      knowledge_tree: Dict[str, Any],
                      paper_section: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Verify section for hallucinations.
        
        Args:
            section: Generated blog section
            knowledge_tree: Knowledge tree for verification
            paper_section: Original paper section
            
        Returns:
            (is_valid, issues) where is_valid is False if hallucinations found
        """
        issues = []
        try:
            knowledge_tree = self._normalize_tree(knowledge_tree or {})
        except Exception:
            knowledge_tree = {"entities": [], "claims": [], "relationships": [],
                          "claim_coverage": 0.0, "evidence_strength": 0.0}

        # 1. Check for quantitative claims without citations
        quant_issues = self._check_quantitative_claims(section, knowledge_tree)
        issues.extend(quant_issues)
        
        # 2. Check for claims not supported by knowledge tree
        unsupported_issues = self._check_unsupported_claims(section, knowledge_tree)
        issues.extend(unsupported_issues)
        
        # 3. Check for fabricated references
        fabricated_issues = self._check_fabricated_references(section, paper_section)
        issues.extend(fabricated_issues)
        
        # 4. Check for figure/table claims that don't match paper
        figure_issues = self._check_figure_alignment(section, paper_section)
        issues.extend(figure_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_quantitative_claims(self, section: str, tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for quantitative claims without proper citations."""
        issues = []
        sentences = re.split(r"(?<=[.!?])\s+", section)
        
        for sent in sentences:
            # Look for quantitative claims
            if re.search(r"\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|F1|AUC|RMSE|MAE|BLEU)\b", sent, re.I):
                # Check for figure/table citation
                if not any(marker in sent.lower() for marker in ["fig", "figure", "table"]):
                    issues.append({
                        "type": "quantitative_claim_uncited",
                        "text": sent,
                        "severity": "high",
                        "suggestion": "Add citation to relevant figure or table"
                    })
                    
        return issues
    
    def _check_unsupported_claims(self, section: str, tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for claims not supported by the knowledge tree."""
        issues = []
        sentences = re.split(r"(?<=[.!?])\s+", section)
        
        for sent in sentences:
            # Skip very short sentences (likely not claims)
            if len(sent) < 20:
                continue
                
            # Check if this sentence is supported by the knowledge tree
            if not self._is_claim_supported(sent, tree):
                issues.append({
                    "type": "unsupported_claim",
                    "text": sent,
                    "severity": "high",
                    "suggestion": "Provide evidence from paper or remove claim"
                })
                
        return issues
    
    def _is_claim_supported(self, claim_sentence: str, tree: dict) -> bool:
        """
        Decide if a claim sentence is supported by known entities/claims in the tree.
        We consider it supported if any normalized entity overlaps with a claim entity
        span or the claim itself (lexical containment), with optional fuzzy thresholds elsewhere.
        """
        # Claim entities: however you extract them now; ensure strings
        claim_entities = self._as_list(self._extract_claim_entities(claim_sentence))
        norm_claim_entities = []
        for ce in claim_entities:
            if isinstance(ce, str):
                s = ce.strip()
            elif isinstance(ce, dict):
                s = self._entity_text(ce)
            else:
                s = ""
            if s:
                norm_claim_entities.append(s)

        # If the extractor returns nothing, fall back to the full sentence
        if not norm_claim_entities:
            norm_claim_entities = [claim_sentence.strip()]

        # KG entities (already normalized to [{"text": ...}] by _normalize_tree)
        kg_entities = [e["text"] for e in self._as_list(tree.get("entities")) if isinstance(e, dict) and e.get("text")]

        # Simple lexical overlap / containment
        for ent in kg_entities:
            ent_l = ent.lower()
            for ce in norm_claim_entities:
                ce_l = ce.lower()
                if ent_l and ((ent_l in ce_l) or (ce_l in ent_l)):
                    return True

        return False
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text."""
        # Simple heuristic for now
        return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    
    def _check_fabricated_references(self, section: str, paper_section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for fabricated references not in the paper."""
        issues = []
        
        # Check for citation patterns like [1], [2-5], etc.
        citations = re.findall(r"\[(\d+(?:-\d+)?)\]", section)
        for cit in citations:
            # Check if this citation exists in the paper
            if not self._citation_exists(cit, paper_section):
                issues.append({
                    "type": "fabricated_citation",
                    "text": f"[{cit}]",
                    "severity": "high",
                    "suggestion": "Remove or replace with valid citation"
                })
                
        return issues
    
    def _citation_exists(self, citation: str, paper_section: Dict[str, Any]) -> bool:
        """Check if a citation exists in the paper."""
        # Simple check for now - in reality would use citation database
        text = paper_section.get("section_text")
        if not text:
            # fallback: stitch from available fields
            text = "\n\n".join(f"{k}: {v}" for k, v in paper_section.items() if isinstance(v, str))
            
        
        # Check for reference section
        if "references" in text.lower():
            # Check if citation appears in references
            return re.search(rf"\[{citation}\]", text) is not None
            
        return True  # Assume citations are valid if no references section
    
    def _check_figure_alignment(self, section: str, paper_section: Dict[str, Any]) -> List[str]:
        issues = []
        paper_figures = self._extract_figure_content(paper_section)

        if not paper_figures:
            return issues  # no figures found, nothing to align

        cited_figs = re.findall(r"(?:Figure|Fig\.)\s*(\d+)", section, re.I)
        for fig_num in cited_figs:
            if fig_num not in paper_figures:
                issues.append(f"Figure {fig_num} cited but not found in paper.")
        return issues
    
    def _extract_figure_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract figure references from text."""
        references = []
        patterns = [
            r"figure\s+(\d+)(?:\s*:\s*(.*?))?(?:\s*\((.*?)\))?",
            r"fig\.\s*(\d+)(?:\s*:\s*(.*?))?(?:\s*\((.*?)\))?",
            r"fig\s*(\d+)(?:\s*:\s*(.*?))?(?:\s*\((.*?)\))?"
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                figure_num = match.group(1)
                caption = match.group(2) or ""
                context = match.group(3) or ""
                
                references.append({
                    "figure_num": figure_num,
                    "caption": caption.strip(),
                    "context": context.strip()
                })
                
        return references
    
    def _extract_figure_description(self, text: str, figure_num: str) -> str:
        """Extract description related to a figure."""
        pattern = r"(Figure\s+{0}.*?)(?=\s*Figure\s+\d+|\Z)".format(figure_num)
        match = re.search(pattern, text, re.I | re.S)
        
        if match:
            sentences = re.split(r"(?<=[.!?])\s+", match.group(0))
            description = " ".join([
                s for s in sentences 
                if any(kw in s.lower() for kw in ["show", "demonstrate", "indicate", "reveal", "illustrate"])
            ])
            return description[:500]
            
        return ""

    def _extract_figure_content(self, paper_section: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract figure content from paper section.
        Returns dict like { "1": {"caption": "...", "content": "..."}, ... }
        """
        # Flatten dict into one big text block if needed
        if isinstance(paper_section, dict):
            text = "\n".join(f"{k}: {v}" for k, v in paper_section.items() if isinstance(v, str))
        elif isinstance(paper_section, str):
            text = paper_section
        else:
            text = str(paper_section)

        figures: Dict[str, Dict[str, Any]] = {}

        # Simple regex: look for "Figure 1: caption text..."
        import re
        matches = re.finditer(r"(?:Figure|Fig\.)\s*(\d+)[\.:]?\s*(.+?)(?=(?:Figure|Fig\.|\Z))",
                            text, re.IGNORECASE | re.DOTALL)
        for m in matches:
            fig_num = m.group(1)
            caption = m.group(2).strip()
            figures[fig_num] = {
                "caption": caption,
                "content": caption  # can later add OCR or embedding if needed
            }

        return figures


    def _figure_claim_matches(self, claim: str, paper_fig: Dict[str, Any]) -> bool:
        """Check if a figure claim matches paper content."""
        # Check if claim mentions key elements from figure caption
        caption_words = set(re.findall(r"\b\w+\b", paper_fig["caption"].lower()))
        claim_words = set(re.findall(r"\b\w+\b", claim.lower()))
        
        # Jaccard similarity
        intersection = len(caption_words & claim_words)
        union = len(caption_words | claim_words)
        jaccard = intersection / max(1, union)
        
        return jaccard > 0.4
    
    def _text_similarity(self, a: str, b: str) -> float:
        """Calculate text similarity between two strings."""
        a_words = set(re.findall(r"\b\w+\b", a.lower()))
        b_words = set(re.findall(r"\b\w+\b", b.lower()))
        
        if not a_words or not b_words:
            return 0.0
            
        return len(a_words & b_words) / max(len(a_words), len(b_words))
    
    def get_hallucination_scorecard(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a human-readable hallucination scorecard."""
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity = issue["severity"]
            if severity in severity_counts:
                severity_counts[severity] += 1
                
        return {
            "total_issues": len(issues),
            "severity_breakdown": severity_counts,
            "critical_issues": severity_counts["high"],
            "needs_revision": len(issues) > 0
        }
    
        # optional spaCy cache
    _nlp = None

    def _get_nlp(self):
        """Lazy-load spaCy; tolerate absence gracefully."""
        if getattr(self, "_nlp", None) is not None:
            return self._nlp
        # try:
        #     # import spacy
        #     # try:
        #     #     self._nlp = spacy.load("en_core_web_sm")
        #     # except Exception:
        #     #     # fallback: blank model (no NER), we'll still use regex
        #     #     self._nlp = spacy.blank("en")
        # except Exception:
        self._nlp = None
        return self._nlp

    def _extract_claim_entities(self, text: str):
        """
        Return a list of entity-like strings from a claim sentence.
        Order of preference:
        1) spaCy NER + noun chunks (if available)
        2) regex heuristics (TitleCase, model names, quoted phrases)
        3) simple NP-ish fallback
        Always returns list[str]; duplicates removed case-insensitively.
        """
        out = []

        # 1) spaCy (if available)
        nlp = self._get_nlp()
        if nlp and getattr(nlp, "pipe_names", None):
            try:
                doc = nlp(text)
                for ent in getattr(doc, "ents", []):
                    t = ent.text.strip()
                    if len(t) >= 2:
                        out.append(t)
                for chunk in getattr(doc, "noun_chunks", []):
                    t = chunk.text.strip()
                    if len(t) >= 2:
                        out.append(t)
            except Exception:
                pass

        # 2) Heuristics if nothing found
        if not out:
            patterns = [
                r"[A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,3}",      # TitleCase spans up to 4 words
                r"\b(?:[A-Za-z]+Net|ResNet|BERT|RoBERTa|GPT-\d+|ViT|T5)\b",  # common model names
                r"[\"“”']([^\"“”']{3,})[\"“”']"               # quoted phrases
            ]
            for pat in patterns:
                for m in re.finditer(pat, text):
                    out.append(m.group(0).strip())

            # lightweight NP-ish fallback (see helper below)
            out.extend(self._simple_np_chunks(text))

        # Dedup by lowercase key, keep longest first
        uniq = {}
        for span in sorted(out, key=lambda s: (-len(s), s.lower())):
            key = span.lower()
            if key not in uniq:
                uniq[key] = span

        return list(uniq.values())

    def _simple_np_chunks(self, text: str):
        """
        Very simple phrase builder: groups alnum-ish tokens into short phrases.
        Not linguistic—just a safety net when no NER is available.
        """
        tokens = re.findall(r"[A-Za-z0-9%\.\/\-\+\_]{2,}", text)
        phrases, buf = [], []
        for tok in tokens:
            buf.append(tok)
            if len(buf) >= 3:
                phrases.append(" ".join(buf))
                buf = []
        if buf:
            phrases.append(" ".join(buf))
        # Filter to phrases that have at least one letter
        return [p for p in phrases if any(c.isalpha() for c in p)]