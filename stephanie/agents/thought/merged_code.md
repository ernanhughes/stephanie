<!-- Merged Python Code Files -->


## File: anti_hallucination.py

`python
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
``n

## File: figure_grounding.py

`python
# stephanie/agents/thought/figure_grounding.py
"""
FigureGrounding
---------------
Specialized system for grounding figure/table references in blog sections.
Ensures quantitative claims properly cite and align with paper figures/tables.
"""

from __future__ import annotations

import logging
import re
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class FigureGrounding:
    """
    System for ensuring proper figure/table grounding in blog sections.
    
    Usage:
        fg = FigureGrounding(logger)
        results = fg.verify_section(section, paper_section, knowledge_tree)
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.figure_patterns = [
            r"figure\s+(\d+)(?:\s*[:.]?\s*(.*?))?(?:\s*\((.*?)\))?",
            r"fig\.\s*(\d+)(?:\s*[:.]?\s*(.*?))?(?:\s*\((.*?)\))?",
            r"fig\s*(\d+)(?:\s*[:.]?\s*(.*?))?(?:\s*\((.*?)\))?"
        ]
        self.table_patterns = [
            r"table\s+(\d+)(?:\s*[:.]?\s*(.*?))?(?:\s*\((.*?)\))?",
            r"tbl\.\s*(\d+)(?:\s*[:.]?\s*(.*?))?(?:\s*\((.*?)\))?",
            r"tbl\s*(\d+)(?:\s*[:.]?\s*(.*?))?(?:\s*\((.*?)\))?"
        ]
        self.quantitative_patterns = [
            r"\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b",
            r"(increased|decreased|improved|reduced|higher|lower|better|worse)\s+by\s+\d+\.?\d*%?\b"
        ]
        self.logger.info("FigureGrounding initialized", {
            "figure_patterns": len(self.figure_patterns),
            "table_patterns": len(self.table_patterns),
            "quantitative_patterns": len(self.quantitative_patterns)
        })
    
    def verify_section(self, 
                      section: str, 
                      paper_section: Dict[str, Any],
                      knowledge_tree: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify figure/table grounding in a section.
        
        Args:
            section: Generated blog section
            paper_section: Original paper section
            knowledge_tree: Knowledge tree for verification (optional)
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Extract figure/table references from section
            section_figures = self._extract_figure_references(section)
            section_tables = self._extract_table_references(section)
            
            # Extract figure/table content from paper
            paper_figures = self._extract_figure_content(paper_section)
            paper_tables = self._extract_table_content(paper_section)
            
            # Verify grounding
            figure_results = self._verify_grounding(
                section_figures, 
                paper_figures,
                "figure"
            )
            table_results = self._verify_grounding(
                section_tables,
                paper_tables,
                "table"
            )
            
            # Check quantitative claims
            quant_claim_results = self._verify_quantitative_claims(
                section,
                paper_section,
                section_figures,
                section_tables,
                knowledge_tree
            )
            
            # Calculate overall scores
            overall_figure_score = self._calculate_figure_score(figure_results, quant_claim_results)
            overall_table_score = self._calculate_table_score(table_results, quant_claim_results)
            
            # Create verification trace
            verification_trace = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "section_length": len(section),
                "figure_references": len(section_figures),
                "table_references": len(section_tables),
                "quantitative_claims": quant_claim_results["total_claims"]
            }
            
            results = {
                "figure_verification": figure_results,
                "table_verification": table_results,
                "quant_claim_verification": quant_claim_results,
                "overall_figure_score": overall_figure_score,
                "overall_table_score": overall_table_score,
                "verification_trace": verification_trace
            }
            
            self.logger.log("FigureGroundingVerificationComplete", {
                "overall_figure_score": overall_figure_score,
                "overall_table_score": overall_table_score,
                "quant_claims_verified": quant_claim_results["properly_cited"],
                "total_quant_claims": quant_claim_results["total_claims"]
            })
            
            return results
            
        except Exception as e:
            self.logger.log("FigureGroundingVerificationError", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            # Return minimal results with error flag
            return {
                "figure_verification": {"error": "verification_failed"},
                "table_verification": {"error": "verification_failed"},
                "quant_claim_verification": {"error": "verification_failed"},
                "overall_figure_score": 0.0,
                "overall_table_score": 0.0,
                "verification_trace": {
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
    
    def _extract_figure_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract figure references from text with context."""
        references = []
        
        for pattern in self.figure_patterns:
            for match in re.finditer(pattern, text, re.I):
                figure_num = match.group(1)
                caption = match.group(2) or ""
                context = match.group(3) or ""
                
                # Find the surrounding sentence
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(text), match.end() + 100)
                context_snippet = text[start_pos:end_pos]
                
                references.append({
                    "figure_num": figure_num,
                    "caption": caption.strip(),
                    "context": context.strip(),
                    "context_snippet": context_snippet,
                    "full_match": match.group(0),
                    "position": match.start()
                })
                
        # Sort by position in text
        references.sort(key=lambda x: x["position"])
        
        # Deduplicate (keep first occurrence)
        seen = set()
        unique_references = []
        for ref in references:
            key = (ref["figure_num"], ref["caption"][:20] if ref["caption"] else "")
            if key not in seen:
                seen.add(key)
                unique_references.append(ref)
                
        return unique_references
    
    def _extract_table_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract table references from text with context."""
        references = []
        
        for pattern in self.table_patterns:
            for match in re.finditer(pattern, text, re.I):
                table_num = match.group(1)
                caption = match.group(2) or ""
                context = match.group(3) or ""
                
                # Find the surrounding sentence
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(text), match.end() + 100)
                context_snippet = text[start_pos:end_pos]
                
                references.append({
                    "table_num": table_num,
                    "caption": caption.strip(),
                    "context": context.strip(),
                    "context_snippet": context_snippet,
                    "full_match": match.group(0),
                    "position": match.start()
                })
                
        # Sort by position in text
        references.sort(key=lambda x: x["position"])
        
        # Deduplicate (keep first occurrence)
        seen = set()
        unique_references = []
        for ref in references:
            key = (ref["table_num"], ref["caption"][:20] if ref["caption"] else "")
            if key not in seen:
                seen.add(key)
                unique_references.append(ref)
                
        return unique_references
    
    def _extract_figure_content(self, paper_section: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract figure content from paper section with metadata."""
        content = {}
        text = paper_section.get("section_text", "")
        
        # Look for figure captions in the paper text
        figure_captions = re.findall(
            r"(Figure\s+(\d+)[.:]\s*(.*?))(?=\s*(?:Figure\s+\d+|Table\s+\d+|\Z))", 
            text, 
            re.I | re.S
        )
        
        for full_match, num, caption in figure_captions:
            # Extract relevant surrounding context
            description = self._extract_figure_description(text, num, full_match)
            
            # Extract metrics mentioned in the figure
            metrics = self._extract_metrics_from_figure(full_match)
            
            content[num] = {
                "caption": caption.strip(),
                "description": description,
                "metrics": metrics,
                "full_text": full_match
            }
            
        return content
    
    def _extract_table_content(self, paper_section: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract table content from paper section with metadata."""
        content = {}
        text = paper_section.get("section_text", "")
        
        # Look for table captions in the paper text
        table_captions = re.findall(
            r"(Table\s+(\d+)[.:]\s*(.*?))(?=\s*(?:Table\s+\d+|Figure\s+\d+|\Z))", 
            text, 
            re.I | re.S
        )
        
        for full_match, num, caption in table_captions:
            # Extract relevant surrounding context
            description = self._extract_table_description(text, num, full_match)
            
            # Extract metrics mentioned in the table
            metrics = self._extract_metrics_from_table(full_match)
            
            content[num] = {
                "caption": caption.strip(),
                "description": description,
                "metrics": metrics,
                "full_text": full_match
            }
            
        return content
    
    def _extract_figure_description(self, text: str, figure_num: str, figure_match: str) -> str:
        """Extract description related to a figure with context."""
        # Look for nearby text that describes the figure
        pattern = r"((?:[^.!?]*?Figure\s+{0}[^.!?]*[.!?])+|(?:(?:[^.!?]*[.!?])+[^.!?]*Figure\s+{0}[^.!?]*[.!?]))".format(figure_num)
        match = re.search(pattern, text, re.I | re.S)
        
        if match:
            # Extract relevant sentences
            sentences = re.split(r"(?<=[.!?])\s+", match.group(0))
            # Keep sentences that likely describe the figure
            description = " ".join([
                s for s in sentences 
                if any(kw in s.lower() for kw in ["show", "demonstrate", "indicate", "reveal", "illustrate", "depict", "present"])
            ])
            return description[:1000]  # Limit length
            
        # Fallback: use the figure match itself
        return figure_match
    
    def _extract_table_description(self, text: str, table_num: str, table_match: str) -> str:
        """Extract description related to a table with context."""
        # Similar to figure description extraction
        pattern = r"((?:[^.!?]*?Table\s+{0}[^.!?]*[.!?])+|(?:(?:[^.!?]*[.!?])+[^.!?]*Table\s+{0}[^.!?]*[.!?]))".format(table_num)
        match = re.search(pattern, text, re.I | re.S)
        
        if match:
            sentences = re.split(r"(?<=[.!?])\s+", match.group(0))
            description = " ".join([
                s for s in sentences 
                if any(kw in s.lower() for kw in ["show", "demonstrate", "indicate", "reveal", "illustrate", "depict", "present"])
            ])
            return description[:1000]
            
        # Fallback: use the table match itself
        return table_match
    
    def _extract_metrics_from_figure(self, figure_text: str) -> List[Dict[str, Any]]:
        """Extract metrics mentioned in a figure."""
        metrics = []
        
        # Look for quantitative statements
        for pattern in self.quantitative_patterns:
            for match in re.finditer(pattern, figure_text, re.I):
                value = match.group(0)
                metric_type = match.group(1) if match.lastindex >= 1 else "value"
                
                metrics.append({
                    "value": value,
                    "type": metric_type.lower(),
                    "context": self._get_sentence_context(figure_text, match.start())
                })
                
        return metrics
    
    def _extract_metrics_from_table(self, table_text: str) -> List[Dict[str, Any]]:
        """Extract metrics mentioned in a table."""
        metrics = []
        
        # Look for quantitative statements
        for pattern in self.quantitative_patterns:
            for match in re.finditer(pattern, table_text, re.I):
                value = match.group(0)
                metric_type = match.group(1) if match.lastindex >= 1 else "value"
                
                metrics.append({
                    "value": value,
                    "type": metric_type.lower(),
                    "context": self._get_sentence_context(table_text, match.start())
                })
                
        return metrics
    
    def _get_sentence_context(self, text: str, position: int) -> str:
        """Get the sentence containing the given position."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current_pos = 0
        
        for sentence in sentences:
            if current_pos <= position < current_pos + len(sentence):
                return sentence.strip()
            current_pos += len(sentence) + 1  # +1 for the split character
            
        return text[max(0, position-50):min(len(text), position+50)].strip()
    
    def _verify_grounding(self, 
                         references: List[Dict[str, Any]], 
                         paper_content: Dict[str, Dict[str, Any]],
                         content_type: str) -> Dict[str, Any]:
        """Verify that references are properly grounded in paper content."""
        verified = []
        unverified = []
        
        for ref in references:
            num = ref.get("figure_num") if content_type == "figure" else ref.get("table_num")
            if not num:
                continue
                
            if num in paper_content:
                paper_item = paper_content[num]
                # Calculate alignment score
                alignment_score = self._calculate_alignment_score(ref, paper_item)
                # Check metric alignment if applicable
                metric_alignment = self._check_metric_alignment(ref, paper_item)
                
                verified.append({
                    **ref,
                    "paper_content": paper_item,
                    "alignment_score": alignment_score,
                    "metric_alignment": metric_alignment,
                    "is_valid": alignment_score >= 0.6 and metric_alignment["score"] >= 0.5
                })
            else:
                unverified.append({
                    **ref,
                    "reason": f"{content_type.capitalize()} {num} not found in paper"
                })
        
       
``n

## File: image_processor_agent.py

`python
# stephanie/agents/thought/image_processor_agent.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import time
import os

try:
    from PIL import Image, ImageEnhance, ImageFilter
except Exception:
    Image = None  # type: ignore

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.image_quality_metrics import ImageQualityMetrics
from stephanie.services.image_profile_service import ImageProfileService

class ImageProcessorAgent(BaseAgent):
    """
    Applies a small set of deterministic augmenters, scores each step,
    emits VPM-like rows, and saves results to CaseBook/DynamicScorables.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.vpm_rows: List[Dict[str, Any]] = []
        self.image_prof = ImageProfileService(memory, logger)
        self.reward = container.get("lfl_reward")  # your LFLRewardService instance
        self.augmenters = [
            ("contrast+sharpen", self._aug_contrast_sharpen),
            ("color_balance", self._aug_color_balance),
            ("denoise", self._aug_denoise),
            ("detail", self._aug_detail)
        ]

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        image_path = context.get("image_path")
        prompt = context.get("goal", {}).get("goal_text") or context.get("prompt") or ""
        casebook = context.get("casebook_name")

        if not image_path or Image is None:
            self.logger.log("ImageProcessorStart", {"ok": False, "reason": "no_image_or_pil"})
            return context

        # load base
        base_img = Image.open(image_path).convert("RGB")

        # step 0 metrics
        qim = ImageQualityMetrics()
        m0 = qim.get_metrics(base_img)
        vs = self.image_prof.score_image(image_path, prompt=prompt)
        vpm0 = self._emit_vpm("initial", m0, vs)
        lfl0 = self._to_lfl(vs, m0)

        best = {"step": "initial", "image": base_img, "path": image_path, "metrics": m0, "vs": vs, "lfl": lfl0}
        curr = base_img

        # iterate simple pipeline
        for name, fn in self.augmenters:
            try:
                nxt = fn(curr)
                m = qim.get_metrics(nxt)
                # fill relevance/style via profile
                # save tmp file to score with profile (CLIP uses file only for logging; we can pass in-memory if modified)
                tmp_path = self._save_tmp(nxt, suffix=name)
                vs_m = self.image_prof.score_image(tmp_path, prompt=prompt)
                vpm = self._emit_vpm(name, m, vs_m)
                lfl = self._to_lfl(vs_m, m)
                if lfl > best["lfl"]:
                    best = {"step": name, "image": nxt, "path": tmp_path, "metrics": m, "vs": vs_m, "lfl": lfl}
                curr = nxt
            except Exception as e:
                self.logger.log("ImageAugError", {"augmentation": name, "error": str(e)})

        # persist best result to casebook
        self._save_case(casebook, prompt, best, context)
        context.setdefault(self.output_key, {})
        context[self.output_key]["image_processing"] = {
            "best_step": best["step"],
            "best_lfl": best["lfl"],
            "vpm_rows": self.vpm_rows,
            "best_meta": {
                "metrics": best["metrics"],
                "vs": best["vs"],
                "path": best["path"]
            }
        }
        return context

    # ---------- augmenters ----------
    def _aug_contrast_sharpen(self, img: "Image.Image") -> "Image.Image":
        c = ImageEnhance.Contrast(img).enhance(1.15)
        s = c.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
        return s

    def _aug_color_balance(self, img: "Image.Image") -> "Image.Image":
        # slight saturation + brightness
        s = ImageEnhance.Color(img).enhance(1.10)
        b = ImageEnhance.Brightness(s).enhance(1.03)
        return b

    def _aug_denoise(self, img: "Image.Image") -> "Image.Image":
        return img.filter(ImageFilter.MedianFilter(size=3))

    def _aug_detail(self, img: "Image.Image") -> "Image.Image":
        return img.filter(ImageFilter.DETAIL)

    # ---------- helpers ----------
    def _emit_vpm(self, step: str, m: Dict[str, float], vs: Dict[str, float]) -> Dict[str, Any]:
        row = {
            "unit": f"image:{step}",
            "kind": "image",
            "timestamp": time.time(),
            "dims": {
                # RS_img-ish
                "sharpness": m["sharpness"],
                "color_diversity": m["color_diversity"],
                "composition": m["composition"],
                "aesthetic_score": m["aesthetic_score"],
                "relevance": vs.get("relevance", 0.5),
                "noise_level": m["noise_level"],
                "contrast": m["contrast"],
                "color_balance": m["color_balance"],
                # VS_img-ish
                "style_clip": vs.get("style_clip", 0.5),
                "palette_match": vs.get("palette_match", 0.5),
                "composition_match": vs.get("composition_match", 0.5),
            },
            "meta": {"step": step}
        }
        self.vpm_rows.append(row)
        self.logger.log("VPMRow", {"step": step, "dims": row["dims"]})
        return row

    def _to_lfl(self, vs: Dict[str, float], m: Dict[str, float]) -> float:
        # adapt to your LFLRewardService: VS components + RS components passed through your existing API
        vs_metrics = {
            "VS1_embed": vs.get("style_clip", 0.5),   # style similarity
            "VS2_style": vs.get("palette_match", 0.5),
            "VS3_moves": vs.get("composition_match", 0.5)
        }
        rs_metrics = {
            "claim_coverage": m.get("color_diversity", 0.5),   # proxy (for images: diversity as coverage)
            "hallucination_rate": 1.0 - m.get("relevance", 0.5),  # low relevance ~ hallucination
            "structure": m.get("composition", 0.5),
            "faithfulness": m.get("relevance", 0.5),
            "HRM_norm": m.get("aesthetic_score", 0.5)  # stand-in if you want
        }
        try:
            return float(self.reward.calculate_lfl(vs_metrics, rs_metrics))
        except Exception:
            return 0.5

    def _save_tmp(self, img: "Image.Image", suffix: str) -> str:
        # deterministic-ish temp path under working dir
        out = f".image_tmp_{int(time.time()*1000)}_{suffix}.png"
        img.save(out, format="PNG")
        return out

    def _save_case(self, casebook_name: Optional[str], prompt: str, best: Dict[str, Any], context: Dict[str, Any]):
        try:
            # Ensure casebook (use your CaseBookStore)
            cb = self.memory.casebooks.ensure_casebook(casebook_name or "images::default", tag="images")
            # Create case and store best image path + metrics
            meta = {
                "type": "image_result",
                "step": best["step"],
                "metrics": best["metrics"],
                "vs": best["vs"],
                "lfl": best["lfl"],
                "source_path": best["path"],
                "prompt": prompt
            }
            case = self.memory.casebooks.add_case(
                casebook_id=cb.id,
                goal_id=context.get("goal", {}).get("id"),
                agent_name=self.name,
                prompt_text=prompt,
                meta=meta,
                scorables=[{
                    "role": "output",
                    "target_type": "image",
                    "meta": {"text": "", "image_path": best["path"], "metrics": best["metrics"], "vs": best["vs"], "lfl": best["lfl"]}
                }]
            )
            self.logger.log("ImageCaseSaved", {"case_id": case.id, "casebook_id": cb.id})
        except Exception as e:
            self.logger.log("ImageCaseSaveError", {"error": str(e)})
``n

## File: knowledge_infused_summarizer.py

`python
# stephanie/agents/thought/knowledge_infused_summarizer.py
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import logging

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.thought.paper_blog import \
    SimplePaperBlogAgent
from stephanie.agents.thought.anti_hallucination import AntiHallucination
from stephanie.agents.thought.figure_grounding import FigureGrounding
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.utils.json_sanitize import sanitize_for_json
from stephanie.models.strategy import StrategyProfile

import matplotlib
if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Defaults --------------------
MAX_ITERS_DEFAULT = 5
MIN_GAIN_DEFAULT = 0.015
MIN_OVERALL_DEFAULT = 0.80
TARGET_CONFIDENCE_DEFAULT = 0.95
MIN_FIGURE_SCORE_DEFAULT = 0.80
VERIFICATION_THRESHOLD_DEFAULT = 0.90
CONVERGENCE_WINDOW_DEFAULT = 2
KNOWLEDGE_GRAPH_CONF_DEFAULT = 0.70
SENTS_MIN_DEFAULT = 4
SENTS_MAX_DEFAULT = 20
CBR_CASES_DEFAULT = 3
PACS_WEIGHTS_DEFAULT = {"skeptic": 0.34, "editor": 0.33, "risk": 0.33}


_logger = logging.getLogger(__name__)

class KnowledgeInfusedVerifierAgent(BaseAgent):
    """
    Track C: Knowledge-Infused Verifier with true Learning-From-Learning

    Adds:
      • CBR reuse of prior wins (patches/lessons)
      • PACS multi-critic refinement with role-aware re-ranking
      • HRM epistemic judge blended into overall score
      • ZeroModel visibility (ABC quality tile + iteration strips)
      • Strategy evolution (thresholds + PACS weights persisted)
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # --- config knobs
        self.max_iters = int(cfg.get("max_iters", MAX_ITERS_DEFAULT))
        self.min_gain = float(cfg.get("min_gain", MIN_GAIN_DEFAULT))
        self.min_overall = float(cfg.get("min_overall", MIN_OVERALL_DEFAULT))
        self.target_confidence = float(
            cfg.get("target_confidence", TARGET_CONFIDENCE_DEFAULT)
        )
        self.min_figure_score = float(
            cfg.get("min_figure_score", MIN_FIGURE_SCORE_DEFAULT)
        )
        self.verification_threshold = float(
            cfg.get("verification_threshold", VERIFICATION_THRESHOLD_DEFAULT)
        )
        self.convergence_window = int(
            cfg.get("convergence_window", CONVERGENCE_WINDOW_DEFAULT)
        )
        self.knowledge_graph_conf = float(
            cfg.get("knowledge_graph_conf", KNOWLEDGE_GRAPH_CONF_DEFAULT)
        )
        self.cbr_cases = int(cfg.get("cbr_cases", CBR_CASES_DEFAULT))

        # feature flags
        self.use_cbr = bool(cfg.get("use_cbr", True))
        self.use_hrm = bool(cfg.get("use_hrm", True))
        self.use_zeromodel = bool(cfg.get("use_zeromodel", True))
        self.use_descendants_metric = bool(
            cfg.get("use_descendants_metric", False)
        )
        self.hrm_weight = float(cfg.get("hrm_weight", 0.10))

        # services
        self.cbr = container.get("cbr") if self.use_cbr else None
        self.scoring = container.get(
            "scoring"
        )  # exposes HRM scorer if configured
        self.zero_model_service = (
            container.get("zeromodel") if self.use_zeromodel else None
        )
        self.kbase = container.get("kbase")  # KnowledgeBaseService

        # strategy state (persist across runs)
        self.strategy_scope = cfg.get("strategy_scope", "track_c")
        self.strategy_store = container.get(
            "strategy"
        )  # StrategyProfileService
        self.strategy = self._load_strategy_profile()

        # dependencies
        self.metrics_calculator = SimplePaperBlogAgent(
            cfg, memory, container, logger
        )
        self.anti_hallucination = AntiHallucination(logger)
        self.figure_grounding = FigureGrounding(logger)

        # sentence window (align with A/B)
        self.min_sents = int(cfg.get("min_sents", SENTS_MIN_DEFAULT))
        self.max_sents = int(cfg.get("max_sents", SENTS_MAX_DEFAULT))

        # model keys
        self.model_key_ranker = cfg.get("model_key_ranker", "ranker.sicql.v1")
        self.model_key_retriever = cfg.get(
            "model_key_retriever", "retriever.mrq.v1"
        )

        self.audit_enabled = bool(cfg.get("enable_audit_report", True))
        self.report_dir = str(cfg.get("audit_report_dir", "reports/track_c"))
        os.makedirs(self.report_dir, exist_ok=True)

        self.logger.log(
            "KnowledgeInfusedVerifierInit",
            {
                "max_iters": self.max_iters,
                "verification_threshold": self.verification_threshold,
                "convergence_window": self.convergence_window,
                "cbr_cases": self.cbr_cases,
                "use_cbr": self.use_cbr,
                "use_hrm": self.use_hrm,
                "use_zeromodel": self.use_zeromodel,
                "strategy_version": self.strategy.strategy_version,
            },
        )

    # -------------------- strategy persistence --------------------
    def _load_strategy_profile(self) -> StrategyProfile:
        # Prefer service; never assume memory.meta exists
        if getattr(self, "strategy_store", None):
            return self.strategy_store.load(
                agent_name=self.name, scope=self.strategy_scope
            )
        # ephemeral fallback (won't persist across runs)
        return StrategyProfile()

    def _save_strategy_profile(self, strategy: StrategyProfile):
        if getattr(self, "strategy_store", None):
            self.strategy_store.save(
                agent_name=self.name, profile=strategy, scope=self.strategy_scope
            )
            self.strategy = strategy
            self.verification_threshold = strategy.verification_threshold

    def _derive_domain(self, paper_data, context):
        doms = context.get("domains") or []
        if doms and isinstance(doms, list):
            d = doms[0]
            return str(
                (d.get("domain") if isinstance(d, dict) else d) or "unknown"
            )
        return "unknown"

    # -------------------- entrypoint --------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.report(
            {
                "event": "start",
                "step": "KnowledgeInfusedVerifier",
                "details": "Track C verification loop with learning",
            }
        )

        documents = context.get("documents", []) or context.get(
            self.input_key, []
        )
        chat_corpus = context.get("chat_corpus", [])
        verified_outputs: Dict[Any, Dict[str, Any]] = {}

        def _extract_summary_from_text(text: str) -> str:
            m = re.search(
                r"(?mi)^##\s*Summary\s*\n(.+?)(?=^##|\Z)", text or "", re.S
            )
            return m.group(1).strip() if m else (text or "").strip()

        for doc in documents:
            doc_id = doc.get("id") or doc.get("paper_id")
            if doc_id is None:
                self.logger.log("TrackCSkipNoDocId", {"doc": str(doc)[:200]})
                continue

            # --- Track A (baseline)
            try:
                track_a_obj = (
                    self.memory.dynamic_scorables.get_latest_by_source_pointer(
                        source="paper_summarizer",
                        source_scorable_type="document",
                        source_scorable_id=int(doc_id),
                    )
                )
            except Exception as e:
                track_a_obj = None
                self.logger.log(
                    "TrackALoadError", {"doc_id": doc_id, "error": str(e)}
                )
            if not track_a_obj:
                self.logger.log(
                    "TrackAMissing",
                    {
                        "doc_id": doc_id,
                        "hint": "Ensure Track A persisted with source_scorable_id=document_id and type=document",
                    },
                )
                continue
            a_meta = self._safe_meta(track_a_obj)
            a_metrics = a_meta.get("metrics") or {}

            # --- Track B (sharpened)
            try:
                track_b_obj = (
                    self.memory.dynamic_scorables.get_latest_by_source_pointer(
                        source="sharpened_paper_summarizer",
                        source_scorable_type="dynamic",
                        source_scorable_id=int(track_a_obj.id),
                    )
                )
            except Exception as e:
                track_b_obj = None
                self.logger.log(
                    "TrackBLoadError", {"doc_id": doc_id, "error": str(e)}
                )
            if not track_b_obj:
                self.logger.log(
                    "TrackBMissing",
                    {
                        "doc_id": doc_id,
                        "hint": "Ensure Track B persisted with source_scorable_id=<Track A dynamic id> and type=dynamic",
                    },
                )
                continue
            b_meta = self._safe_meta(track_b_obj)

            b_text = (getattr(track_b_obj, "text", "") or "").strip()
            baseline_b_summary = _extract_summary_from_text(b_text) or (
                b_meta.get("summary") or b_text
            )

            title = doc.get("title", "") or (a_meta.get("title") or "")
            abstract = (
                a_meta.get("abstract")
                or b_meta.get("abstract")
                or self._fetch_abstract(doc_id)
            )
            arxiv_summary = (
                a_meta.get("arxiv_summary")
                or b_meta.get("arxiv_summary")
                or (doc.get("summary", "") or "")
            )

            baseline_b_metrics = b_meta.get("metrics")
            if not baseline_b_metrics:
                baseline_b_metrics = self._score_summary(
                    baseline_b_summary,
                    abstract,
                    arxiv_summary,
                    {},
                    title,
                    context,
                )

            # --- Track C (verify + learn)
            try:
                verified = await self._verify_summary(
                    doc_id=str(doc_id),
                    enhanced_summary=baseline_b_summary,
                    paper_data=doc,
                    chat_corpus=chat_corpus,
                    context=context,
                    track_a=track_a_obj,
                    track_b=track_b_obj,
                )
            except Exception as e:
                self.logger.log(
                    "TrackCVerifyError",
                    {
                        "doc_id": doc_id,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                continue

            verified_outputs[doc_id] = verified

            # --- training events + VPM tiles
            try:
                v_metrics = verified.get("metrics") or {}
                if v_metrics.get(
                    "overall", 0.0
                ) >= self.min_overall and verified.get(
                    "passes_guardrails", False
                ):
                    self._emit_training_events(
                        paper={
                            "paper_id": doc.get("paper_id", doc_id),
                            "title": title,
                            "abstract": abstract,
                            "author_summary": arxiv_summary,
                        },
                        baseline_summary=baseline_b_summary,
                        verified_summary=verified.get("summary", ""),
                        baseline_metrics=baseline_b_metrics,
                        verified_metrics=v_metrics,
                        context=context,
                    )

                self._emit_vpm_tiles(
                    doc_id=doc_id,
                    title=title,
                    metrics_a=a_metrics,
                    metrics_b=baseline_b_metrics or {},
                    metrics_c=v_metrics,
                    iterations_c=verified.get("iterations", []),
                    out_dir="reports/vpm",
                    lineage_ids=[
                        getattr(track_a_obj, "id", None),
                        getattr(track_b_obj, "id", None),
                    ],
                )
            except Exception as e:
                try:
                    self.memory.session.rollback()
                except Exception:
                    pass
                self.logger.log(
                    "TrackCPostProcessError",
                    {"doc_id": doc_id, "error": str(e)},
                )

        context.setdefault("summary_v2", {})
        context["summary_v2"] = verified_outputs

        if self.audit_enabled:
            context.setdefault("reports", [])
            # push all generated .md files for this batch
            # (we already called self.report() for each, but some UIs read context["reports"])
            try:
                md_files = [
                    f for f in os.listdir(self.report_dir) if f.endswith(".md")
                ]
                for f in md_files:
                    context["reports"].append(
                        {
                            "agent": self.name,
                            "type": "markdown",
                            "path": os.path.join(self.report_dir, f),
                        }
                    )
            except Exception:
                pass

        return context

    # -------------------- core verification loop --------------------
    async def _verify_summary(
        self,
        doc_id: str,
        enhanced_summary: str,
        paper_data: Dict[str, Any],
        chat_corpus: List[Dict[str, Any]],
        context: Dict[str, Any],
        track_a: Any,
        track_b: Any,
    ) -> Dict[str, Any]:
        start_time = time.time()

        abstract = self._fetch_abstract(doc_id)
        arxiv_summary = paper_data.get("summary", "")
        goal_title = paper_data.get("title", "")

        knowledge_graph = context.get("knowledge_graph")
        if not knowledge_graph:
            knowledge_graph = await self._build_knowledge_graph(
                doc_id, paper_data, chat_corpus, context
            )

        current_summary = enhanced_summary
        current_metrics = self._score_summary(
            current_summary,
            abstract,
            arxiv_summary,
            knowledge_graph,
            goal_title,
            context,
        )
        start_overall = current_metrics.get("overall", 0.0)
        best_summary, best_metrics = current_summary, current_metrics

        iterations: List[Dict[str, Any]] = []
        no_improve_count = 0
        convergence_track: List[float] = []
        lineage_ids = [
            getattr(track_a, "id", None),
            getattr(track_b, "id", None),
        ]

        audit = {
            "doc_id": str(doc_id),
            "title": goal_title,
            "start_overall": float(current_metrics.get("overall", 0.0)),
            "baseline_metrics": current_metrics,
            "iterations": [],  # we’ll append per-iter snapshots here
            "strategy_before": self.strategy.to_dict(),
            "track_a_id": getattr(track_a, "id", None),
            "track_b_id": getattr(track_b, "id", None),
            "kbase_hints": [],
        }

        for iter_idx in range(self.max_iters):
            iter_start = time.time()

            # CBR pack
            case_pack = self._retrieve_case_pack(goal_title, k=self.cbr_cases)
            self.report({"event": "cbr_pack", "k": len(case_pack)})

            # prompt with CBR
            prompt = self._build_verification_prompt(
                current_summary=current_summary,
                claims=(knowledge_graph or {}).get("claims", []),
                paper_data=paper_data,
                case_pack=case_pack,
                context=context,
                abstract=abstract,
            )
            # hash + excerpt for report (avoid dumping huge prompts verbatim)
            import hashlib

            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[
                :12
            ]
            prompt_excerpt = prompt[:600]

            # PACS refinement (get details for the report)
            raw_llm = self.call_llm(prompt, context=context) or current_summary
            candidate, panel_detail = self._pacs_refine(
                raw_llm,
                abstract,
                context,
                paper_data,
                knowledge_graph,
                return_panel=True,
            )

            # score candidate
            cand_metrics = self._score_summary(
                candidate,
                abstract,
                arxiv_summary,
                knowledge_graph,
                goal_title,
                context,
            )
            gain = cand_metrics["overall"] - current_metrics["overall"]

            # emit iteration tile
            try:
                self.zero_model_service.emit_iteration_tile(
                    doc_id=str(doc_id),
                    iteration=iter_idx + 1,
                    metrics={
                        "overall": cand_metrics.get("overall", 0.0),
                        "knowledge_verification": cand_metrics.get(
                            "knowledge_verification", 0.0
                        ),
                        "hrm_score": cand_metrics.get("hrm_score", 0.0)
                        if cand_metrics.get("hrm_score") is not None
                        else 0.0,
                    },
                    output_dir="reports/vpm/iters",
                )
            except Exception as e:
                self.logger.log(
                    "VPMIterTileWarn", {"doc_id": doc_id, "error": str(e)}
                )

            # record iteration
            iter_payload = {
                "iteration": iter_idx + 1,
                "current_score": current_metrics["overall"],
                "best_candidate_score": cand_metrics["overall"],
                "gain": gain,
                "processing_time": time.time() - iter_start,
                "knowledge_graph_conf": self.knowledge_graph_conf,
                "prompt_hash": prompt_hash,
                "prompt_excerpt": prompt_excerpt,
                "panel_detail": panel_detail or {},
            }
            if knowledge_graph:
                iter_payload["claim_coverage"] = knowledge_graph.get(
                    "claim_coverage", 0.0
                )
                iter_payload["evidence_strength"] = knowledge_graph.get(
                    "evidence_strength", 0.0
                )
            iterations.append(iter_payload)
            if self.audit_enabled:
                audit["iterations"].append(iter_payload)
                if panel_detail and not audit.get("kbase_hints"):
                    audit["kbase_hints"] = panel_detail.get("kb_hints", [])

            # accept if improves enough
            if (
                cand_metrics["overall"] >= self.min_overall
                and gain >= self.min_gain
            ):
                current_summary = candidate
                current_metrics = cand_metrics
                if cand_metrics["overall"] > best_metrics["overall"]:
                    best_summary, best_metrics = (
                        current_summary,
                        current_metrics,
                    )
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            convergence_track.append(best_metrics["overall"])

            # stops
            if best_metrics["overall"] >= self.target_confidence:
                self.report(
                    {
                        "event": "verification_converged",
                        "reason": "target_confidence",
                    }
                )
                break
            if no_improve_count >= 2:
                self.report(
                    {"event": "verification_converged", "reason": "no_improve"}
                )
                break
            if len(convergence_track) >= self.convergence_window:
                recent = convergence_track[-self.convergence_window :]
                if (
                    np.std(recent) < 5e-3
                    and (max(recent) - min(recent)) < 2e-2
                ):
                    self.report(
                        {
                            "event": "verification_converged",
                            "reason": "convergence_window",
                        }
                    )
                    break

        # guardrails
        is_valid, hallucination_issues = self._verify_hallucinations(
            best_summary, abstract, arxiv_summary, knowledge_graph
        )
        figure_results = self._verify_figure_grounding(
            best_summary, paper_data, knowledge_graph
        )

        # strategy evolution
        if best_metrics["overall"] > start_overall + self.min_gain:
            new_weights = self._adjust_pacs_weights(
                {**best_metrics, "figure_results": figure_results}
            )
            new_threshold = min(
                0.99, self.strategy.verification_threshold + 0.01
            )
            self.strategy.update(
                pacs_weights=new_weights, verification_threshold=new_threshold
            )
            self._save_strategy_profile(self.strategy)
            self.report(
                {
                    "event": "strategy_updated",
                    "new_weights": new_weights,
                    "new_threshold": new_threshold,
                }
            )

        result = {
            "summary": best_summary,
            "metrics": best_metrics,
            "iterations": iterations,
            "processing_time": time.time() - start_time,
            "hallucination_issues": hallucination_issues,
            "figure_results": figure_results,
            "passes_guardrails": bool(is_valid)
            and (
                figure_results.get("overall_figure_score", 0.0)
                >= self.min_figure_score
            ),
            "converged": best_metrics["overall"] >= self.target_confidence,
            "knowledge_graph": knowledge_graph,
            "verification_trace": {
                "iterations": len(iterations),
                "final_score": best_metrics["overall"],
                "converged": len(convergence_track) >= self.convergence_window
                and np.std(convergence_track[-self.convergence_window :])
                < 1e-2,
            },
        }

        try:
            # 1) ensure blog casebook
            paper_id = str(paper_data.get("paper_id", doc_id))
            post_slug = (paper_data.get("post_slug") or "main")

            title = paper_data.get("title")
            name = f"blog::{title}"
            meta = {"paper_id": paper_id, "arxiv_id": paper_data.get("arxiv_id"), "title": paper_data.get("title", ""), "post_slug": post_slug}
            casebook = self.memory.casebooks.ensure_casebook(name=name, tag="blog", meta=meta)

            meta ={}
            response_texts = [raw_llm, candidate]
            case = self.memory.casebooks.add_case(
                casebook_id=casebook.id, 
                goal_id=casebook.goal_id,
                prompt_text=prompt,
                agent_name=self.name,
                response_texts=response_texts,
                meta=meta,
            )

            # 3) auto-promote champion if better than any existing champion by overall score
            # try:
            #     existing = blog.get_champion_text(casebook_name=cb.name, section="summary")
            #     should_promote = False
            #     if not existing:
            #         should_promote = True
            #     else:
            #         # crude comparison on overall; if you want, fetch champion metrics from cases for precision
            #         should_promote = float(best_metrics.get("overall", 0.0)) >= self.min_overall

            #     if should_promote:
            #         blog.mark_champion(
            #             casebook_name=casebook_name,
            #             case_id=str(add_res["case_id"]),
            #             section="summary",
            #         )
            # except Exception as e:
            #     self.logger.log("BlogChampionEvalWarn", {"error": str(e)})
        except Exception as e:
            self.logger.log("BlogCasebookWriteWarn", {"doc_id": doc_id, "error": str(e)})


        # persist as dynamic scorable
        try:
            safe_meta = sanitize_for_json(
                {
                    "paper_id": paper_data.get("paper_id", doc_id),
                    "title": paper_data.get("title", ""),
                    "metrics": best_metrics,
                    "origin": "track_c_verified",
                    "verification_trace": result["verification_trace"],
                    "hallucination_issues": result.get(
                        "hallucination_issues", []
                    ),
                    "origin_ids": lineage_ids,
                    "figure_results": result.get("figure_results", {}),
                }
            )

            scorable_id = self.memory.dynamic_scorables.add(
                pipeline_run_id=context.get("pipeline_run_id"),
                scorable_type=TargetType.DYNAMIC,
                source=self.name,
                text=best_summary,
                meta=safe_meta,  # ← sanitized!
                source_scorable_id=getattr(track_b, "id", None),
                source_scorable_type="dynamic",
            )
            result["scorable_id"] = scorable_id
            scorable = self.memory.casebooks.add_scorable(
                case_id=case.id,
                pipeline_run_id=context.get("pipeline_run_id"),
                role="text",
                scorable_id=scorable_id,
                text=enhanced_summary,
                scorable_type=TargetType.DYNAMIC,
                meta=meta,
            )
            result["case_scorable_id"] = scorable_id
        except Exception as e:
            self.logger.log(
                "DynamicScorableSaveError",
                {
                    "doc_id": doc_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )




        domain = self._derive_domain(paper_data, context)
        strategy_delta = {
            "skeptic": self.strategy.pacs_weights.get("skeptic", 0.0)
            - PACS_WEIGHTS_DEFAULT["skeptic"],
            "editor": self.strategy.pacs_weights.get("editor", 0.0)
            - PACS_WEIGHTS_DEFAULT["editor"],
            "risk": self.strategy.pacs_weights.get("risk", 0.0)
            - PACS_WEIGHTS_DEFAULT["risk"],
            "verification_threshold": self.strategy.verification_threshold
            - VERIFICATION_THRESHOLD_DEFAULT,
        }
        try:
            if self.kbase:
                self.kbase.update_from_paper(
                    domain=domain,
                    summary_text=result["summary"],
                    metrics=result["metrics"],
                    iterations=result["iterations"],
                    strategy_delta=strategy_delta,
                )

            self._capture_cross_paper_signals(
                paper_id=str(paper_data.get("paper_id", doc_id)),
                domain=domain,
                metrics=result["metrics"],
                iterations=result["iterations"],
                strategy=self.strategy,
                strategy_delta=strategy_delta,
            )
        except Exception as e:
            self.logger.log(
                "KBUpdateWarn", {"doc_id": doc_id, "error": str(e)}
            )

        if self.audit_enabled:
            try:
                audit["strategy_after"] = self.strategy.to_dict()
                audit["final_metrics"] = result["metrics"]
                audit["passes_guardrails"] = result["passes_guardrails"]
                audit["hallucination_issues"] = result.get(
                    "hallucination_issues", []
                )
                audit["figure_results"] = result.get("figure_results", {})
                timeline_png = self._plot_iteration_timeline(
                    audit["iterations"],
                    out_path=os.path.join(
                        self.report_dir, f"{doc_id}_timeline.png"
                    ),
                )
                transfer_png = self.generate_transfer_curve(
                    output_path=os.path.join(
                        self.report_dir, "transfer_curve.png"
                    )
                )
                report_md = self._write_audit_report(
                    doc_id=str(doc_id),
                    title=goal_title,
                    audit=audit,
                    timeline_path=timeline_png,
                    transfer_curve_path=transfer_png,
                    abc_gif_path=result.get(
                        "quality_tile_path"
                    ),  # if ZeroModel returned one
                )
                # expose to pipeline context + report stream
                self.report(
                    {
                        "event": "verification_report",
                        "doc_id": str(doc_id),
                        "path": report_md,
                    }
                )
            except Exception as e:
                self.logger.log(
                    "AuditReportError", {"doc_id": doc_id, "error": str(e)}
                )

        return result

    # --- Cross-paper signals & evaluation ---------------------------------
    def _capture_cross_paper_signals(
        self,
        *,
        paper_id: str,
        domain: str,
        metrics: Dict[str, Any],
        iterations: List[Dict[str, Any]],
        strategy: StrategyProfile,
        strategy_delta: Dict[str, float],
    ) -> None:
        """
        Persist tiny signals that let us measure transfer across papers.
        Plays nice if tables aren't present (no hard deps).
        """
        payload = {
            "paper_id": paper_id,
            "domain": domain,
            "strategy_version": int(getattr(strategy, "strategy_version", 0)),
            "verification_threshold": float(
                getattr(strategy, "verification_threshold", 0.0)
            ),
            "pacs_weights": dict(getattr(strategy, "pacs_weights", {})),
            "strategy_delta": dict(strategy_delta or {}),
            "final_quality": float(metrics.get("overall", 0.0)),
            "knowledge_verification": float(
                metrics.get("knowledge_verification", 0.0)
            ),
            "hrm_score": float(metrics.get("hrm_score", 0.0))
            if metrics.get("hrm_score") is not None
            else None,
            "iterations": int(len(iterations or [])),
            "first_iter_score": float(
                (iterations or [{}])[0].get("current_score", 0.0)
            )
            if iterations
            else None,
            "last_iter_score": float(
                (iterations or [{}])[-1].get("best_candidate_score", 0.0)
            )
            if iterations
            else None,
            "ts": time.time(),
        }
        # Optional: calibration events (soft dependency)
        try:
            self.memory.calibration_events.add({
                    "domain": domain or "general",
                    "query": f"{paper_id}:{domain}",                # any non-null string
                    "raw_similarity": float(metrics.get("overall", 0.0)),
                    "is_relevant": bool(float(metrics.get("overall", 0.0)) >= self.min_overall),
                    "scorable_id": str(paper_id),
                    "scorable_type": "paper",
                    "entity_type": "summary_verification",
                    "features": {
                        "quality": float(metrics.get("overall", 0.0)),
                        "knowledge_verification": float(metrics.get("knowledge_verification", 0.0)),
                        "hrm_score": None if metrics.get("hrm_score") is None else float(metrics["hrm_score"]),
                        "verification_threshold": float(strategy.verification_threshold),
                        "pacs_weights": dict(strategy.pacs_weights or {}),
                        "iterations": int(len(iterations or [])),
                        "first_iter_score": payload.get("first_iter_score"),
                        "last_iter_score": payload.get("last_iter_score"),
                    },
                })
        except Exception as e:
            _logger.error("CalibrationAddWarn", {"error": str(e)})

        # Optional: casebook of signals (simple append-only log)
        try:
            casebooks = getattr(self.memory, "casebooks", None)
            if casebooks and hasattr(casebooks, "add"):
                casebooks.add(
                    casebook_name="verification_signals",
                    case_id=f"{paper_id}",
                    role="signal",
                    text=json.dumps(payload),
                    meta={"domain": domain, "timestamp": payload["ts"]},
                )
        except Exception as e:
            self.logger.log("CasebookAddWarn", {"error": str(e)})

        self.logger.log(
            "CrossPaperSignalCaptured",
            {
                "paper_id": paper_id,
                "domain": domain,
                "quality": payload["final_quality"],
                "strategy_version": payload["strategy_version"],
            },
        )

    def analyze_transfer_effect(
        self, learning_split: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        Reads 'verification_signals' casebook and checks if baseline performance
        (papers after the split, treated as 'no-learning' runs) rises over time.
        """
        try:
            casebooks = getattr(self.memory, "casebooks", None)
            if not (casebooks and hasattr(casebooks, "get_by_casebook")):
                self.logger.log(
                    "TransferAnalyzeSkip", {"reason": "casebooks missing"}
                ) 
                return None

            rows = (
                casebooks.get_by_casebook(
                    casebook_name="verification_signals", role="signal"
                )
                or []
            )
            if len(rows) < 20:
                return None

            # Expect sequential ids or anything we can sort on 'timestamp'
            data = []
            for r in rows:
                try:
                    d = json.loads(getattr(r, "text", "{}") or "{}")
                    data.append(d)
                except Exception:
                    continue
            if not data:
                return None

            data.sort(key=lambda d: d.get("ts", 0.0))
            post = [x for i, x in enumerate(data) if i >= learning_split]
            if len(post) < 10:
                return None

            # simple start/end window means
            head = post[: max(5, min(10, len(post) // 4))]
            tail = post[-max(5, min(10, len(post) // 4)) :]

            initial = (
                float(np.mean([h.get("final_quality", 0.0) for h in head]))
                if head
                else 0.0
            )
            final = (
                float(np.mean([t.get("final_quality", 0.0) for t in tail]))
                if tail
                else 0.0
            )
            improvement = final - initial

            return {
                "initial_baseline": initial,
                "final_baseline": final,
                "improvement": improvement,
                "sample_size": len(post),
                "significant": improvement > 0.05,  # coarse heuristic
            }
        except Exception as e:
            self.logger.log("TransferAnalyzeError", {"error": str(e)})
            return None

    def generate_transfer_curve(
        self, output_path: str = "reports/vpm/transfer_curve.png"
    ) -> Optional[str]:
        """
        Produce a simple PNG of baseline performance drift after the learning split.
        """
        try:
            import matplotlib
        except Exception:
            self.logger.log(
                "TransferCurveSkip", {"reason": "matplotlib not available"}
            )
            return None

        res = self.analyze_transfer_effect()
        if not res:
            return None

        # Rebuild the time series from signals
        rows = (
            self.memory.casebooks.get_by_casebook(
                casebook_name="verification_signals", role="signal"
            )
            or []
        )
        rows = sorted(rows, key=lambda r: json.loads(r.text).get("ts", 0.0))
        perf = [json.loads(r.text).get("final_quality", 0.0) for r in rows]

        plt.figure(figsize=(9, 5.2))
        plt.plot(range(1, len(perf) + 1), perf, linewidth=2)
        plt.axhline(
            y=res["initial_baseline"], linestyle="--", label="Initial baseline"
        )
        plt.axhline(
            y=res["final_baseline"], linestyle="--", label="Final baseline"
        )
        plt.title("Transfer Learning Effect: Baseline Performance Over Time")
        plt.xlabel("Paper Index")
        plt.ylabel("Quality (overall)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.log("TransferCurveSaved", {"path": output_path, **res})
        return output_path

    # -------------------- CBR / PACS / HRM helpers --------------------
    def _retrieve_case_pack(
        self, title: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        if not self.use_cbr or not self.cbr:
            return []
        try:
            cases = self.cbr.retrieve(goal_text=title, top_k=k)
            pack = []
            for c in cases or []:
                pack.append(
                    {
                        "title": (c.get("goal_text") or "")[:160],
                        "why_it_won": (
                            c.get("scores", {}).get("winner_rationale") or ""
                        )[:240],
                        "patch": (c.get("lessons") or "")[:240],
                        "summary": (
                            c.get("best_text") or c.get("summary") or ""
                        )[:400],
                    }
                )
            return pack
        except Exception as e:
            self.logger.log("CBRRetrieveError", {"error": str(e)})
            return []

    # change signature to accept paper_data and knowledge_tree
    def _pacs_refine(
        self,
        candidate: str,
        abstract: str,
        context: Dict[str, Any],
        paper_data: Dict[str, Any] | None = None,
        knowledge_tree: Dict[str, Any] | None = None,
        *,
        return_panel: bool = False,
    ) -> str:
        title = (
            (paper_data or {}).get("title")
            or context.get("goal", {}).get("goal_text", "")
            or ""
        )
        domain = self._derive_domain(paper_data or {}, context or {})
        kb_ctx = (
            self.kbase.context_for_paper(
                title=title, abstract=abstract, domain=domain
            )
            if self.kbase
            else {}
        )
        nudges = kb_ctx.get("weight_nudges", {}) or {}

        # Ephemeral weights: do NOT mutate self.strategy here
        base_w = dict(self.strategy.pacs_weights)
        work_w = dict(base_w)
        for k, dv in nudges.items():
            work_w[k] = max(0.2, min(0.4, work_w.get(k, 0.33) + float(dv)))
        s = sum(work_w.values()) or 1.0
        work_w = {k: v / s for k, v in work_w.items()}

        roles = [
            ("skeptic", "remove speculation; flag ungrounded claims"),
            (
                "editor",
                f"tighten structure; keep {self.min_sents}-{self.max_sents} sentences",
            ),
            ("risk", "require figure/table citation for any numeric claim"),
        ]

        panel: List[Tuple[str, str]] = []
        for role, brief in roles:
            prompt = f"""Role: {role.title()}. Brief: {brief}
    Abstract:
    \"\"\"{abstract[:1000]}\"\"\"

    Text to review:
    \"\"\"{candidate}\"\"\"\n
    Return ONLY the revised paragraph."""
            try:
                out = self.call_llm(prompt, context=context)
                if out:
                    panel.append((role, out.strip()))
            except Exception:
                continue
        if not panel:
            return (candidate, None) if return_panel else candidate

        # score by role...
        best_text, best_score = candidate, -1.0
        role_scores = []
        for role, text in panel:
            m = self.metrics_calculator._compute_metrics(text, abstract, "")
            if role == "risk":
                m["figure_results"] = self._verify_figure_grounding(
                    text, paper_data or {}, knowledge_tree or {}
                )
            role_score = self._role_weighted_score(role, m, weights=work_w)
            role_scores.append(
                {"role": role, "score": role_score, "metrics": m, "text": text}
            )
            if role_score > best_score:
                best_text, best_score = text, role_score

        details = {
            "kb_hints": kb_ctx.get("hints", []),
            "kb_templates_count": len(kb_ctx.get("templates", [])),
            "nudges": nudges,
            "weights_used": work_w,
            "panel": role_scores,
        }
        self.logger.log("PACSRefine", details)
        return (best_text, details) if return_panel else best_text

    def _role_weighted_score(
        self,
        role: str,
        m: Dict[str, float],
        weights: Dict[str, float] | None = None,
    ) -> float:
        skeptic_focus = 0.6 * (
            1.0 - float(m.get("hallucination_rate", 0.0))
        ) + 0.4 * float(m.get("faithfulness", 0.0))
        editor_focus = 0.5 * float(m.get("coherence", 0.0)) + 0.5 * float(
            m.get("structure", 0.0)
        )
        risk_focus = (
            float(m.get("figure_results", {}).get("overall_figure_score", 0.0))
            if isinstance(m.get("figure_results"), dict)
            else 0.0
        )
        base = float(m.get("overall", 0.0))
        wmap = weights or self.strategy.pacs_weights
        w = wmap.get(role, 0.33)

        if role == "skeptic":
            role_focus = skeptic_focus
        elif role == "editor":
            role_focus = editor_focus
        else:
            role_focus = risk_focus

        score = 0.5 * base + 0.5 * role_focus
        return w * score

    def _hrm_epistemic(
        self, text: str, goal: str, context: Dict[str, Any]
    ) -> Tuple[Optional[float], str]:
        if not self.use_hrm or not self.scoring:
            return None, ""
        try:
            scorable = ScorableFactory.from_dict(
                {"text": text, "goal": goal, "type": "document"}
            )
            bundle = self.scoring.score(
                "hrm",
                context=context,
                scorable=scorable,
                dimensions=["alignment"],
            )
            res = getattr(bundle, "results", {}).get("alignment")
            if res is None:
                return None, ""
            score = (
                float(getattr(res, "score", None))
                if getattr(res, "score", None) is not None
                else None
            )
            rationale = getattr(res, "rationale", "")
            return score, rationale
        except Exception as e:
            self.logger.log("HRMScoreError", {"error": str(e)})
            return None, ""

    def _build_verification_prompt(
        self,
        current_summary: str,
        claims: List[Dict[str, Any]],
        paper_data: Dict[str, Any],
        case_pack: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        abstract: Optional[str] = None,  # <-- add param
    ) -> str:
        title = paper_data.get("title", "")
        domain = self._derive_domain(paper_data, context or {})
        abs_text = (
            abstract
            if abstract is not None
            else self._fetch_abstract(
                paper_data.get("id") or paper_data.get("paper_id")
            )
        )
        kb_ctx = (
            self.kbase.context_for_paper(
                title=title, abstract=abs_text, domain=domain
            )
            if self.kbase
            else {}
        )
        tmpl_text = ""
        if kb_ctx.get("templates"):
            bullets = []
            for t in kb_ctx["templates"]:
                bullets.append("- " + " ".join(t.get("outline", [])[:3]))
            tmpl_text = "\n\nTemplates that worked before:\n" + "\n".join(
                bullets
            )

        hints_text = ""
        if kb_ctx.get("hints"):
            hints_text = "\n\nStrategy hints:\n" + "\n".join(
                f"- {h}" for h in kb_ctx["hints"]
            )

        claims_text = "\n".join(
            f"- {c.get('text', '').strip()}"
            for c in (claims or [])[:5]
            if c.get("text")
        )
        examples = ""
        if case_pack:
            ex_lines = []
            for ex in case_pack[:3]:
                ex_lines.append(
                    f"- Lesson: {ex.get('patch', '')}\n  Why it won: {ex.get('why_it_won', '')}"
                )
            if ex_lines:
                examples = "\n\nPrior improvements to emulate:\n" + "\n".join(
                    ex_lines
                )
        return f"""
You are a verification expert checking this academic paper summary against the paper's key claims.

Title: {title}

Key Claims:
{claims_text}{examples}{tmpl_text}{hints_text}

Current summary:
\"\"\"{current_summary}\"\"\"

Improve the summary by:
1) Ensuring all key claims are accurately represented
2) Citing figures/tables for quantitative claims when warranted
3) Removing unsupported statements 
4) Preserving clarity and neutrality

Constraints:
- Keep to {self.min_sents}-{self.max_sents} sentences
- Use ONLY facts present in the paper and allowed context
- Do not invent numbers or facts

Verified summary:
""".strip()

    def _verify_against_knowledge_tree(self, summary: str, knowledge_tree: Dict[str, Any]) -> float:
        if not knowledge_tree:
            return 0.5
        claims = knowledge_tree.get("claims", []) or []
        covered = 0
        for claim in claims:
            text = claim.get("text", "")
            if text and self.metrics_calculator._contains_concept(summary, text):
                covered += 1
        claim_coverage = covered / max(1, len(claims))
        rels = knowledge_tree.get("relationships", []) or []
        if self.strategy:
            threshold = self.strategy.verification_threshold
            _logger.info(f"Using strategy threshold: {threshold}")
        else:
            threshold = self.verification_threshold
        strong = [r for r in rels if float(r.get("confidence", 0.0)) >= threshold]
        evidence_strength = len(strong) / max(1, len(rels))
        return (0.7 * claim_coverage) + (0.3 * evidence_strength)

    # -------------------- guardrails --------------------
    def _verify_hallucinations(
        self,
        summary: str,
        abstract: str,
        author_summary: str,
        knowledge_tree: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        # Make AntiHallucination resilient to key-type mismatches in figure maps etc.
        try:
            is_valid, issues = self.anti_hallucination.verify_section(
                summary,
                knowledge_tree,
                {"abstract": abstract, "summary": author_summary},
            )
            return (bool(is_valid), issues or [])
        except Exception as e:
            self.logger.log("AntiHallucinationError", {"error": str(e)})
            return True, ["anti_hallucination_failed_soft"]

    def _verify_figure_grounding(
        self,
        summary: str,
        paper_data: Dict[str, Any],
        knowledge_tree: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Simple heuristic extractor for quant claims → expected to be replaced by FigureGrounding
        quant_claims = []
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", summary or "")
            if s.strip()
        ]
        for sent in sentences:
            matches = re.findall(
                r"(\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b)",
                sent,
                flags=re.I,
            )
            if matches:
                quant_claims.append(
                    {
                        "claim": sent,
                        "value": matches[0][0],
                        "metric": matches[0][1],
                        "has_citation": any(
                            marker in sent.lower()
                            for marker in [
                                "figure",
                                "fig.",
                                "table",
                                "as shown",
                                "see",
                            ]
                        ),
                    }
                )
        properly_cited = sum(1 for c in quant_claims if c.get("has_citation"))
        citation_rate = properly_cited / max(1, len(quant_claims))
        return {
            "total_claims": len(quant_claims),
            "properly_cited": properly_cited,
            "citation_rate": citation_rate,
            "overall_figure_score": citation_rate,
            "claims": quant_claims,
        }

    # -------------------- strategy evolution --------------------
    def _adjust_pacs_weights(
        self, metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        weights = dict(self.strategy.pacs_weights)
        if float(metrics.get("knowledge_verification", 0.0)) > 0.8:
            weights["editor"] = min(0.4, weights.get("editor", 0.33) + 0.05)
            weights["skeptic"] = max(0.2, weights.get("skeptic", 0.33) - 0.05)
        if float(metrics.get("hallucination_rate", 1.0)) > 0.2:
            weights["skeptic"] = min(0.4, weights.get("skeptic", 0.33) + 0.05)
            weights["editor"] = max(0.2, weights.get("editor", 0.33) - 0.05)
        if float(metrics.get("coverage", 0.0)) < 0.7:
            weights["skeptic"] = min(0.4, weights.get("skeptic", 0.33) + 0.03)
        if float(metrics.get("coherence", 0.0)) < 0.7:
            weights["editor"] = min(0.4, weights.get("editor", 0.33) + 0.03)
        fig_score = 0.0
        fr = metrics.get("figure_results")
        if isinstance(fr, dict):
            fig_score = float(fr.get("overall_figure_score", 0.0))
        if fig_score < 0.7:
            weights["risk"] = min(0.4, weights.get("risk", 0.33) + 0.03)
        # normalize
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    # -------------------- utilities --------------------
    def _safe_meta(self, obj) -> dict:
        meta = getattr(obj, "meta", {}) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return meta

    def _emit_vpm_tiles(
        self,
        *,
        doc_id,
        title: str,
        metrics_a: dict | None,
        metrics_b: dict | None,
        metrics_c: dict | None,
        iterations_c: list[dict] | None,
        out_dir: str = "reports/vpm",
        lineage_ids: List[Any] | None = None,
    ):
        try:
            svc = self.zero_model_service
            if not svc:
                self.logger.log(
                    "VPMSkipServiceMissing",
                    {"doc_id": doc_id, "reason": "zero_model service missing"},
                )
                return
            vpm_data = self._prepare_vpm_data(
                doc_id,
                title,
                metrics_a or {},
                metrics_b or {},
                metrics_c or {},
                iterations_c or [],
            )
            if hasattr(svc, "generate_summary_vpm_tiles"):
                result = svc.generate_summary_vpm_tiles(
                    vpm_data=vpm_data, output_dir=out_dir
                )
            else:
                # minimal fallback: ABC triptych only
                names = [
                    "overall",
                    "coverage",
                    "faithfulness",
                    "structure",
                    "no_halluc",
                ]
                import numpy as _np

                rows = []
                for label, mm in (
                    ("A", metrics_a),
                    ("B", metrics_b),
                    ("C", metrics_c),
                ):
                    mm = mm or {}
                    rows.append(
                        [
                            float(mm.get("overall", 0.0)),
                            float(mm.get("claim_coverage", 0.0)),
                            float(mm.get("faithfulness", 0.0)),
                            float(mm.get("structure", 0.0)),
                            float(1.0 - mm.get("hallucination_rate", 1.0)),
                        ]
                    )
                mat = _np.asarray(rows, dtype=_np.float32)
                out = f"{out_dir}/{doc_id}_abc.gif"
                if hasattr(svc, "_emit_timeline"):
                    svc._emit_timeline(mat, out)
                result = {"quality_tile_path": out}
            self.logger.log(
                "VPMTilesGenerated", {"doc_id": doc_id, **(result or {})}
            )
        except Exception as e:
            self.logger.log(
                "VPMTileGenerationError",
                {
                    "doc_id": doc_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

    def _prepare_vpm_data(
        self, doc_id, title, metrics_a, metrics_b, metrics_c, iterations_c
    ):
        def pack(m):
            # Map into a compact, consistent bundle
            return {
                "overall": float(m.get("overall", 0.0)),
                "coverage": float(
                    m.get("claim_coverage", m.get("coverage", 0.0))
                ),
                "faithfulness": float(m.get("faithfulness", 0.0)),
                "structure": float(m.get("structure", 0.0)),
                "no_halluc": float(1.0 - m.get("hallucination_rate", 1.0)),
                "figure_ground": float(
                    (m.get("figure_results", {}) or {}).get(
                        "overall_figure_score", 0.0
                    )
                )
                if isinstance(m.get("figure_results"), dict)
                else 0.0,
            }

        return {
            "doc_id": doc_id,
            "title": title[:80],
            "metrics": {
                "A": pack(metrics_a),
                "B": pack(metrics_b),
                "C": pack(metrics_c),
            },
            "iterations": iterations_c or [],
            "timestamp": time.time(),
        }

    def _fetch_abstract(self, doc_id) -> str:
        try:
            sections = self.memory.document_sections.get_by_document(doc_id)
            for s in sections:
                sd = s.to_dict()
                if (
                    sd.get("section_name") or ""
                ).lower().strip() == "abstract":
                    return sd.get("section_text", "") or ""
        except Exception as e:
            self.logger.log(
                "AbstractFetchFailed", {"doc_id": doc_id, "error": str(e)}
            )
        return ""

    def _emit_training_events(
        self,
        paper: Dict[str, Any],
        baseline_summary: str,
        verified_summary: str,
        baseline_metrics: Dict[str, float],
        verified_metrics: Dict[str, float],
        context: Dict[str, Any],
    ):
        title = paper.get("title", "paper")
        gain = float(
            verified_metrics.get("overall", 0.0)
            - (baseline_metrics or {}).get("overall", 0.0)
        )
        w = max(0.1, min(1.0, gain + 0.3))

        # pointwise
        self.memory.training_events.add_pointwise(
            model_key=self.model_key_retriever,
            dimension="alignment",
            query_text=title,
            cand_text=verified_summary,
            label=1,
            weight=float(verified_metrics.get("overall", 0.7)),
            trust=float(verified_metrics.get("overall", 0.7)),
            goal_id=context.get("goal", {}).get("id"),
            pipeline_run_id=context.get("pipeline_run_id"),
            agent_name=self.name,
            source="track_c",
            meta={
                "stage": "track_c",
                "gain": gain,
                "knowledge_verification": verified_metrics.get(
                    "knowledge_verification", 0.0
                ),
            },
        )

        # pairwise vs. Track B
        self.memory.training_events.add_pairwise(
            model_key=self.model_key_ranker,
            dimension="alignment",
            query_text=title,
            pos_text=verified_summary,
            neg_text=baseline_summary,
            weight=w,
            trust=w * 0.6,
            goal_id=context.get("goal", {}).get("id"),
            pipeline_run_id=context.get("pipeline_run_id"),
            agent_name=self.name,
            source="track_c",
            meta={
                "stage": "track_c",
                "verified_score": verified_metrics.get("overall"),
                "baseline_score": (baseline_metrics or {}).get("overall"),
                "gain": gain,
            },
        )

        # optional pairwise vs author/arXiv summary
        author_summary = paper.get("author_summary", "") or ""
        if author_summary.strip():
            author_metrics = self._score_summary(
                author_summary,
                paper.get("abstract", ""),
                author_summary,
                {},
                title,
                context,
            )
            prefer_verified = verified_metrics.get(
                "overall", 0.0
            ) > author_metrics.get("overall", 0.0)
            pos = verified_summary if prefer_verified else author_summary
            neg = author_summary if prefer_verified else verified_summary
            self.memory.training_events.add_pairwise(
                model_key=self.model_key_ranker,
                dimension="alignment",
                query_text=title,
                pos_text=pos,
                neg_text=neg,
                weight=0.5,
                trust=0.3,
                goal_id=context.get("goal", {}).get("id"),
                pipeline_run_id=context.get("pipeline_run_id"),
                agent_name=self.name,
                source="track_c",
                meta={
                    "stage": "track_c",
                    "verified_score": verified_metrics.get("overall", 0.0),
                    "author_score": author_metrics.get("overall", 0.0),
                    "prefer_verified": prefer_verified,
                },
            )

    async def _build_knowledge_graph(
        self,
        doc_id: str,
        paper_data: Dict[str, Any],
        chat_corpus: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Minimal KG builder that ONLY uses the service's build_tree(...).
        Returns a normalized dict with expected keys.
        """

        def _empty_kg() -> Dict[str, Any]:
            return {
                "nodes": [],
                "relationships": [],
                "claims": [],
                "claim_coverage": 0.0,
                "evidence_strength": 0.0,
                "temporal_coherence": 0.0,
                "domain_alignment": 0.0,
                "knowledge_gaps": [],
                "meta": {"paper_id": str(doc_id)},
            }

        def _normalize(kg: Any) -> Dict[str, Any]:
            if not isinstance(kg, dict):
                kg = {}
            kg = kg.get("knowledge_graph") or kg
            if not isinstance(kg, dict):
                kg = {}
            # ensure expected fields
            kg.setdefault("nodes", [])
            kg.setdefault("relationships", [])
            kg.setdefault("claims", [])
            kg.setdefault("claim_coverage", 0.0)
            kg.setdefault("evidence_strength", 0.0)
            kg.setdefault("temporal_coherence", 0.0)
            kg.setdefault("domain_alignment", 0.0)
            kg.setdefault("knowledge_gaps", [])
            kg.setdefault("meta", {})
            kg["meta"].setdefault("paper_id", str(doc_id))
            return kg

        svc = self.container.get("knowledge_graph")
        if not (svc and hasattr(svc, "build_tree")):
            self.logger.log("KGMissingBuildTree", {"doc_id": doc_id})
            return _empty_kg()

        paper_text = (paper_data.get("text") or "").strip()
        try:
            # build_tree is sync; run it in a worker so we don't block the event loop
            kg = await asyncio.to_thread(
                svc.build_tree,
                paper_text=paper_text,
                paper_id=str(doc_id),
                chat_corpus=chat_corpus or [],
                trajectories=context.get("conversation_trajectories", [])
                or [],
                domains=context.get("domains", []) or [],
            )
            self.logger.log(
                "KGBuildPath",
                {"service": svc.__class__.__name__, "method": "build_tree"},
            )
            return _normalize(kg)
        except Exception as e:
            self.logger.log(
                "KnowledgeGraphBuildFailed",
                {
                    "doc_id": doc_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            return _empty_kg()

    # --- Emit one VPM tile per iteration (safe wrapper) ---
    def _emit_vpm_tile(
        self,
        doc_id: str,
        stage: str,
        metrics: Dict[str, Any],
        lineage_ids: List[Any] | None,
        context: Dict[str, Any],
    ) -> None:
        """
        Best-effort tile emission. If a ZeroModel/visualization service is
        available, we forward the request; otherwise we no-op.
        """
        try:
            zm = getattr(
                self, "zero_model_service", None
            ) or self.container.get("zeromodel")
        except Exception:
            zm = None

        if not zm:
            # Nothing to do; keep pipeline robust
            self.logger.log(
                "VPMTileSkipNoService", {"doc_id": doc_id, "stage": stage}
            )
            return

        try:
            payload = {
                "doc_id": str(doc_id),
                "stage": stage,
                "metrics": dict(metrics or {}),
                "lineage": [x for x in (lineage_ids or []) if x is not None],
                "vpf": {
                    "pipeline_run_id": context.get("pipeline_run_id"),
                    "agent": self.name,
                    "stage": stage,
                },
            }
            # Delegate to the service; tolerate either method name
            fn = (
                getattr(zm, "create_tile", None)
                or getattr(zm, "generate_summary_vpm_tiles", None)
                or getattr(zm, "emit_tile", None)
            )
            if callable(fn):
                fn(**payload) if fn.__code__.co_kwonlyargcount else fn(payload)
            else:
                self.logger.log(
                    "VPMTileSkipNoAPI", {"doc_id": doc_id, "stage": stage}
                )
        except Exception as e:
            self.logger.log(
                "VPMTileEmitError",
                {
                    "doc_id": doc_id,
                    "stage": stage,
                    "error": str(e),
                },
            )

    # --- Scoring (with knowledge + optional HRM) ---
    def _score_summary(
        self,
        summary: str,
        abstract: str,
        author_summary: str,
        knowledge_tree: Dict[str, Any],
        goal_title: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Base Track-A metrics + knowledge verification (+ optional HRM).
        Compatible with older callers that don’t pass goal_title/context.
        """
        # 1) deterministic base metrics
        base = self.metrics_calculator._compute_metrics(
            summary, abstract, author_summary
        )

        # 2) knowledge verification (claim coverage + evidence strength)
        ver = self._verify_against_knowledge_tree(summary, knowledge_tree)

        # 3) optional HRM epistemic judge
        hrm_score = None
        try:
            if self.use_hrm:
                _goal = goal_title or (
                    context.get("goal", {}).get("goal_text", "")
                )
                hrm_score, _ = self._hrm_epistemic(summary, _goal, context)
        except Exception:
            hrm_score = None

        # normalize HRM to [0,1]
        if hrm_score is not None:
            hs = float(hrm_score)
            if hs < 0.0 or hs > 1.0:
                # treat as a logit-like raw signal
                hs = 1.0 / (1.0 + math.exp(-hs))
                self.logger.log(
                    "HRMScoreNormalized", {"raw": hrm_score, "norm": hs}
                )
            # hard clamp
            hrm_score = max(0.0, min(1.0, hs))

        # 4) blend
        #   keep your prior weighting; add a small HRM term if present
        overall = 0.8 * base.get("overall", 0.0) + 0.2 * ver
        if hrm_score is not None:
            overall = (
                1.0 - self.hrm_weight
            ) * overall + self.hrm_weight * float(hrm_score)

        out = dict(base)
        out["knowledge_verification"] = float(ver)
        if hrm_score is not None:
            out["hrm_score"] = float(hrm_score)
        out["overall"] = float(overall)
        return out

    def _plot_iteration_timeline(
        self, iters: List[Dict[str, Any]], out_path: str
    ) -> Optional[str]:
        if not iters:
            return None

        xs = [it["iteration"] for it in iters]
        ys = [float(it.get("best_candidate_score", 0.0)) for it in iters]
        cs = [float(it.get("current_score", 0.0)) for it in iters]

        plt.figure(figsize=(8.6, 4.2))
        plt.plot(xs, cs, linewidth=2, label="current score")
        plt.plot(xs, ys, linewidth=2, label="candidate score")
        plt.title("Track C: Per-Iteration Verification Scores")
        plt.xlabel("Iteration")
        plt.ylabel("Overall score")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path

    def _write_audit_report(
        self,
        *,
        doc_id: str,
        title: str,
        audit: Dict[str, Any],
        timeline_path: Optional[str],
        transfer_curve_path: Optional[str],
        abc_gif_path: Optional[str] = None,
    ) -> str:
        """
        Render a compact Markdown report that shows:
        - overview & baseline vs. final metrics,
        - iteration timeline figure,
        - PACS panel snapshots (scores per role),
        - knowledge verification, hallucination & figure grounding,
        - strategy shifts,
        - transfer learning curve (global).
        """

        def f(x):  # short float
            try:
                return f"{float(x):.3f}"
            except Exception:
                return str(x)

        base = audit.get("baseline_metrics", {})
        final = audit.get("final_metrics", {})
        issues = audit.get("hallucination_issues", [])
        figure = audit.get("figure_results", {})
        hints = audit.get("kbase_hints", [])
        strat_b = audit.get("strategy_before", {})
        strat_a = audit.get("strategy_after", {})

        lines = []
        lines.append(f"# Verification Report — {title or doc_id}")
        lines.append("")
        lines.append(
            f"**Doc ID:** `{doc_id}`  |  **Start overall:** {f(audit.get('start_overall'))}  |  **Final overall:** {f(final.get('overall', 0.0))}"
        )
        lines.append("")
        if abc_gif_path:
            lines.append(
                f"![ABC tile]({os.path.relpath(abc_gif_path, self.report_dir)})"
            )
            lines.append("")

        # Overview table
        lines.append("## Overview (Baseline → Final)")
        lines.append("")
        rows = [
            ("overall", base.get("overall"), final.get("overall")),
            (
                "knowledge_verification",
                base.get("knowledge_verification"),
                final.get("knowledge_verification"),
            ),
            (
                "coverage",
                base.get("claim_coverage", base.get("coverage")),
                final.get("claim_coverage", final.get("coverage")),
            ),
            (
                "faithfulness",
                base.get("faithfulness"),
                final.get("faithfulness"),
            ),
            ("structure", base.get("structure"), final.get("structure")),
            (
                "hallucination_rate (↓)",
                base.get("hallucination_rate"),
                final.get("hallucination_rate"),
            ),
            (
                "figure_grounding",
                (base.get("figure_results") or {}).get("overall_figure_score")
                if isinstance(base.get("figure_results"), dict)
                else None,
                (final.get("figure_results") or {}).get("overall_figure_score")
                if isinstance(final.get("figure_results"), dict)
                else None,
            ),
        ]
        lines.append("| metric | baseline | final |")
        lines.append("|---|---:|---:|")
        for k, b, c in rows:
            lines.append(f"| {k} | {f(b)} | {f(c)} |")
        lines.append("")

        # Iteration timeline
        if timeline_path:
            rel = os.path.relpath(timeline_path, self.report_dir)
            lines.append("## Iteration Timeline")
            lines.append("")
            lines.append(f"![Iteration scores]({rel})")
            lines.append("")

        # Iteration snapshots (compact)
        lines.append("## Iteration Snapshots")
        lines.append("")
        for it in audit.get("iterations", []):
            lines.append(
                f"### Iter {it['iteration']} — gain: {f(it.get('gain', 0.0))}, cand: {f(it.get('best_candidate_score'))}"
            )
            lines.append(f"- prompt: `{it.get('prompt_hash')}` — excerpt:")
            excerpt = (it.get("prompt_excerpt") or "").replace("\n", " ")
            lines.append(
                f"  > {excerpt[:240]}{'…' if len(excerpt) > 240 else ''}"
            )
            pd = it.get("panel_detail") or {}
            if pd.get("weights_used"):
                w = pd["weights_used"]
                lines.append(
                    f"- PACS weights used: skeptic={f(w.get('skeptic'))}, editor={f(w.get('editor'))}, risk={f(w.get('risk'))}"
                )
            # show top-1 panel improvement
            best = None
            for entry in pd.get("panel") or []:
                if not best or float(entry.get("score", -1)) > float(
                    best.get("score", -1)
                ):
                    best = entry
            if best:
                lines.append(
                    f"- Best panel: **{best['role']}** (score {f(best['score'])})"
                )
            lines.append("")

        # Knowledge verification & guardrails
        lines.append("## Knowledge Verification & Guardrails")
        lines.append("")
        lines.append(
            f"- Claim coverage (final): {f(final.get('claim_coverage', final.get('coverage')))}"
        )
        lines.append(
            f"- Evidence strength (final): {f(audit['iterations'][-1].get('evidence_strength') if audit.get('iterations') else None)}"
        )
        if issues:
            lines.append(
                f"- Hallucination issues: {len(issues)} (listed below)"
            )
        if isinstance(figure, dict):
            lines.append(
                f"- Figure grounding: {figure.get('properly_cited', 0)}/{figure.get('total_claims', 0)} cited (rate={f(figure.get('citation_rate'))})"
            )
        lines.append("")
        if issues:
            lines.append("<details><summary>Hallucination issues</summary>\n")
            for x in issues[:20]:
                lines.append(f"- {str(x)[:240]}")
            lines.append("\n</details>\n")

        # Strategy evolution
        lines.append("## Strategy Evolution")
        lines.append("")
        try:
            wb = (
                (strat_b.get("pacs_weights") or {})
                if isinstance(strat_b, dict)
                else {}
            )
            wa = (
                (strat_a.get("pacs_weights") or {})
                if isinstance(strat_a, dict)
                else {}
            )
            lines.append(
                f"- Threshold: {f((strat_b or {}).get('verification_threshold'))} → {f((strat_a or {}).get('verification_threshold'))}"
            )
            lines.append(
                f"- PACS weights: skeptic {f(wb.get('skeptic'))}→{f(wa.get('skeptic'))}, editor {f(wb.get('editor'))}→{f(wa.get('editor'))}, risk {f(wa.get('risk'))}→{f(wa.get('risk'))}"
            )
        except Exception:
            pass
        if hints:
            lines.append("")
            lines.append("### KBase Hints Applied")
            for h in hints:
                lines.append(f"- {h}")
            lines.append("")

        # Transfer (global)
        if transfer_curve_path:
            rel = os.path.relpath(transfer_curve_path, self.report_dir)
            lines.append("## Transfer Learning Trend (Global)")
            lines.append("")
            lines.append(f"![Transfer curve]({rel}) But how you doing")
            lines.append("")

        # finalize
        out_md = os.path.join(self.report_dir, f"{doc_id}.md")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return out_md
``n

## File: paper_blog.py

`python
# stephanie/agents/thought/paper_blog.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.utils.casebook_utils import generate_casebook_name


SENTS_MIN_DEFAULT = 4
SENTS_MAX_DEFAULT = 20


class SimplePaperBlogAgent(BaseAgent):
    """
    Inputs (context[self.input_key]): list of docs with at least:
      - id (int/str)
      - title (str)
      - summary (str)  # arXiv summary (author provided / arXiv auto)
    Will look up abstract from memory.document_sections for the doc id.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.min_sents = int(cfg.get("min_sents", SENTS_MIN_DEFAULT))
        self.max_sents = int(cfg.get("max_sents", SENTS_MAX_DEFAULT))
        self.training_min_overall = float(
            cfg.get("training_min_overall", 0.75)
        )
        self.training_max_halluc = float(cfg.get("training_max_halluc", 0.10))
        self.scoring = container.get("scoring")  # optional
        # sensible defaults for model keys
        self.model_key_ranker = cfg.get("model_key_ranker", "ranker.sicql.v1")
        self.model_key_retriever = cfg.get(
            "model_key_retriever", "retriever.mrq.v1"
        )
        self.casebook_action = cfg.get("casebook_action", "blog")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        documents = context.get(self.input_key, [])
        self.report(
            {
                "event": "start",
                "step": "SimplePaperSummarizer",
                "details": f"Profiling {len(documents)} documents",
            }
        )

        out_map = {}
        for doc in documents:
            doc_id = doc.get("id")
            title = doc.get("title", "") or ""

            casebook_name = generate_casebook_name("summarization", title)
            casebook = self.memory.casebooks.ensure_cb(action, casebook_name, tag="summarization")

            arxiv_summary = (
                doc.get("summary", "") or ""
            )  # treat as "author/arXiv summary"

            # --- fetch abstract from sections ---
            abstract = self._fetch_abstract(doc_id)

            merged_context = {
                "title": title,
                "summary": arxiv_summary,
                "min_sents": self.min_sents,
                "max_sents": self.max_sents,
                "abstract": abstract,
                **context,
            }

            prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)
            model_out = self.call_llm(prompt, context=context)

            self.report(
                {
                    "event": "llm_output",
                    "step": "paper_summarizer",
                    "prompt": prompt[:12000],  # avoid huge logs
                    "response": (model_out or "")[:12000],
                }
            )

            parsed = self._parse_model_output(model_out or "")
            valid, msg, val_meta = self._validate_output(
                parsed.get("summary", ""), self.min_sents, self.max_sents
            )
            if not valid:
                self.logger.log(
                    "SummaryValidationFailed",
                    {"doc_id": doc_id, "reason": msg, **val_meta},
                )
                # still record raw, but mark invalid
                out_map[doc_id] = {
                    **parsed,
                    "valid": False,
                    "validation_reason": msg,
                }
                continue

            # --- compute metrics vs sources ---
            metrics = self._compute_metrics(
                parsed["summary"], abstract, arxiv_summary
            )

            # --- persist scorable w/ safe rollback on DB glitch ---
            paper = {
                "paper_id": doc.get("paper_id", doc_id),
                "title": title,
                "abstract": abstract,
                "arxiv_summary": arxiv_summary,
            }

            scorable_id = self._persist_scorable_document(
                paper=paper,
                summary_text=parsed["summary"],
                intro_text=parsed.get("intro", ""),
                metrics=metrics,
                context=context,
            )

            # --- training events (pointwise + optional pairwise vs arXiv summary) ---
            if scorable_id:
                try:
                    self._emit_training_events(
                        paper={
                            "paper_id": doc.get("paper_id", doc_id),
                            "title": title,
                            "abstract": abstract,
                            "arxiv_summary": arxiv_summary,
                        },
                        summary=parsed["summary"],
                        metrics=metrics,
                        context=context,
                    )
                except Exception as e:
                    self.memory.session.rollback()
                    self.logger.log(
                        "TrainingEventEmitError",
                        {"doc_id": doc_id, "error": str(e)},
                    )

            out_map[doc_id] = {
                **parsed,
                "valid": True,
                "metrics": metrics,
                "scorable_id": scorable_id,
            }

        context.setdefault(self.output_key, {})
        context[self.output_key]["summary_v0"] = out_map
        return context

    # ---------- persistence & events ----------

    def _persist_scorable_document(
        self,
        paper: Dict[str, Any],
        summary_text: str,
        intro_text: str,
        metrics: Dict[str, float],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Save a scorable/document and ensure an embedding exists. Rolls back on DB failure.
        """
        full_text = f"## Summary\n{summary_text}\n\n## Blog post introduction\n{intro_text}".strip()

        summary_scorable = self.memory.dynamic_scorables.add(
            pipeline_run_id=context.get("pipeline_run_id"),
            scorable_type=TargetType.DYNAMIC,
            source=self.name,
            text=full_text,
            source_scorable_id=paper.get("paper_id"),
            source_scorable_type="document",
            meta={
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "arxiv_summary": paper.get("arxiv_summary"), 
                "text": full_text,
                "summary": summary_text,
                "metrics": metrics,
            },
        )
        self.memory.embedding.get_or_create(full_text)
        return summary_scorable.id

    def _emit_training_events(
        self, paper: Dict[str, Any], summary: str, metrics: Dict[str, float], context: Dict[str, Any]
    ):
        """
        Emits:
          - Pointwise: positive example (summary) with weight = overall
          - Pairwise: baseline vs author/arXiv summary, if author summary present
        Uses thresholds to avoid training on low-quality samples.
        """
        if (
            metrics.get("overall", 0.0) < self.training_min_overall
            or metrics.get("hallucination_rate", 1.0)
            > self.training_max_halluc
        ):
            self.logger.log(
                "TrainingEventSkipped",
                {
                    "reason": "low_quality",
                    "overall": metrics.get("overall", 0.0),
                    "hallucination_rate": metrics.get(
                        "hallucination_rate", 1.0
                    ),
                },
            )
            return

        title = paper.get("title", "paper")
        # --- Pointwise (retriever) ---
        self.memory.training_events.add_pointwise(
            model_key=self.model_key_retriever,
            dimension="alignment",
            query_text=title,
            cand_text=summary,
            label=1,
            weight=float(metrics.get("overall", 0.7)),
            trust=float(metrics.get("overall", 0.7)),
            goal_id=None,
            pipeline_run_id=self._maybe_pipeline_run_id(),
            agent_name=self.name,
            source="track1_baseline",
            meta={
                "stage": "track1",
                "claim_coverage": metrics.get("claim_coverage"),
                "faithfulness": metrics.get("faithfulness"),
            },
        )

        # --- Pairwise (ranker) vs author/arXiv summary if available ---
        arxiv_summary = paper.get("summary") or ""
        if arxiv_summary.strip():
            author_metrics = self._compute_metrics(
                arxiv_summary, paper.get("abstract", ""), arxiv_summary
            )
            prefer_baseline = metrics["overall"] > author_metrics["overall"]

            pos_text = summary if prefer_baseline else arxiv_summary
            neg_text = arxiv_summary if prefer_baseline else summary

            self.memory.training_events.add_pairwise(
                model_key=self.model_key_ranker,
                dimension="alignment",
                query_text=title,
                pos_text=pos_text,
                neg_text=neg_text,
                weight=0.5,
                trust=0.3,
                goal_id=context.get("goal",{}).get("id"),
                pipeline_run_id=context.get("pipeline_run_id"),
                agent_name=self.name,
                source="track1_baseline",
                meta={
                    "stage": "track1",
                    "baseline_score": metrics["overall"],
                    "author_score": author_metrics["overall"],
                    "prefer_baseline": prefer_baseline,
                },
            )

    def _maybe_pipeline_run_id(self) -> Optional[str]:
        try:
            return getattr(self, "pipeline_run_id", None) or None
        except Exception:
            return None

    # ---------- parsing / validation ----------

    def _parse_model_output(self, text: str) -> Dict[str, str]:
        """
        Supports either:
          ## Summary
          ...
          ## Blog post introduction
          ...
        OR the older variant including Score/Rationale (will ignore for persistence).
        """

        def grab(section: str) -> str:
            m = re.search(
                rf"^##\s*{section}\s*\n(.+?)(?=^##|\Z)", text, re.S | re.M
            )
            return (m.group(1).strip() if m else "").strip()

        # Try new format
        summary = grab("Summary")
        intro = grab("Blog post introduction")

        # If missing, try legacy keys
        if not summary:
            summary = grab(
                "Blog post introduction for the paper"
            )  # fallback (rare)
        # Optional legacy score + rationale
        score_raw = grab("Score") or ""
        rationale = grab("Rational") or grab("Rationale") or ""

        return {
            "summary": summary,
            "intro": intro,
            "score_self": score_raw,
            "rationale": rationale,
        }

    def _validate_output(
        self, summary: str, min_sents: int, max_sents: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        if not summary:
            return False, "Missing 'Summary' section", {}

        sents = [
            s
            for s in re.split(r"(?<=[.!?])\s+", summary)
            if len(s.strip()) > 1
        ]
        if not (min_sents <= len(sents) <= max_sents):
            return (
                False,
                f"Summary must have {min_sents}-{max_sents} sentences (found {len(sents)})",
                {"sentence_count": len(sents)},
            )

        # lightweight hallucination markers
        hallucination_markers = [
            ("not specified", 0.4),
            ("we propose", 0.6),
            ("novel approach", 0.6),
        ]
        hscore = sum(
            w for t, w in hallucination_markers if t in summary.lower()
        )
        if hscore > 1.0:
            return (
                False,
                "Summary contains hallucination markers",
                {"hallucination_score": hscore},
            )

        return (
            True,
            "ok",
            {"sentence_count": len(sents), "hallucination_score": hscore},
        )

    # ---------- metrics (cheap + deterministic) ----------

    def _compute_metrics(
        self, summary: str, abstract: str, arxiv_summary: str
    ) -> Dict[str, float]:
        # 1) Claim coverage from abstract sentences (first 2–3 + numeric lines)
        claims = self._extract_key_claims(abstract)
        covered = sum(
            1 for claim in claims if self._contains_concept(summary, claim)
        )
        claim_coverage = covered / max(1, len(claims))

        # 2) Faithfulness via cosine on embeddings (summary vs sources)
        abstract_sim = (
            self._cosine_similarity(summary, abstract) if abstract else 0.0
        )
        author_sim = (
            self._cosine_similarity(summary, arxiv_summary)
            if arxiv_summary
            else 0.0
        )
        faithfulness = 0.7 * abstract_sim + 0.3 * author_sim

        # 3) Structure (problem→approach→results→implications heuristic)
        structure_score = self._evaluate_structure(summary)

        # 4) Hallucination: count sentences with verbs + not present in sources by fuzzy sim
        hallucination_issues = self._detect_hallucinations(
            summary, abstract, arxiv_summary
        )
        sent_count = max(
            1, len([s for s in re.split(r"(?<=[.!?])", summary) if s.strip()])
        )
        hallucination_rate = min(1.0, len(hallucination_issues) / sent_count)

        overall = (
            (claim_coverage * 0.4)
            + ((1 - hallucination_rate) * 0.4)
            + (structure_score * 0.2)
        )
        return {
            "claim_coverage": float(claim_coverage),
            "faithfulness": float(faithfulness),
            "structure": float(structure_score),
            "hallucination_rate": float(hallucination_rate),
            "sentence_count": sent_count,
            "tokens": len(summary.split()),
            "overall": float(overall),
        }

    # ---------- small helpers used by metrics ----------

    def _fetch_abstract(self, doc_id) -> str:
        try:
            sections = self.memory.document_sections.get_by_document(doc_id)
            for s in sections:
                sd = s.to_dict()
                if (
                    sd.get("section_name") or ""
                ).lower().strip() == "abstract":
                    return sd.get("section_text", "") or ""
        except Exception as e:
            self.logger.log(
                "AbstractFetchFailed", {"doc_id": doc_id, "error": str(e)}
            )
        return ""

    def _cosine_similarity(self, a: str, b: str) -> float:
        try:
            va = self.memory.embedding.get_or_create(a)
            vb = self.memory.embedding.get_or_create(b)
        except Exception as e:
            self.logger.log("EmbedSimFallback", {"error": str(e)})
            return 0.0
        # cosine
        dot = sum(x * y for x, y in zip(va, vb))
        na = max(1e-8, sum(x * x for x in va) ** 0.5)
        nb = max(1e-8, sum(y * y for y in vb) ** 0.5)
        return float(dot / (na * nb))

    def _extract_key_claims(self, abstract: str) -> List[str]:
        if not abstract:
            return []
        sents = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", abstract)
            if len(s.strip()) > 0
        ]
        # take first 2–3 + any sentence with numbers
        key = sents[:3]
        key += [s for s in sents[3:] if re.search(r"\d", s)]
        # unique & trimmed to ~3–5
        uniq = []
        for s in key:
            if s not in uniq:
                uniq.append(s)
        return uniq[:5]

    def _contains_concept(self, text: str, claim: str) -> bool:
        # quick semantic+lexical check
        if self._cosine_similarity(text, claim) >= 0.65:
            return True
        # fallback lexical overlap
        t = set(re.findall(r"[a-z0-9]+", text.lower()))
        c = set(re.findall(r"[a-z0-9]+", claim.lower()))
        if not c:
            return False
        overlap = len(t & c) / len(c)
        return overlap >= 0.25

    def _evaluate_structure(self, summary: str) -> float:
        # look for cues: problem/approach/results/implications
        s = summary.lower()
        cues = 0
        cues += 1 if re.search(r"(problem|challenge|gap|motivation)", s) else 0
        cues += (
            1
            if re.search(
                r"(we|the paper|the authors|method|approach|model|framework)",
                s,
            )
            else 0
        )
        cues += (
            1
            if re.search(
                r"(results|experiments|evaluation|improv(e|ement)|accuracy|performance)",
                s,
            )
            else 0
        )
        cues += (
            1
            if re.search(
                r"(implication|impact|application|future work|limitations)", s
            )
            else 0
        )
        return min(1.0, cues / 4.0)

    def _detect_hallucinations(
        self, summary: str, abstract: str, arxiv_summary: str
    ) -> List[str]:
        # any sentence far from both sources is suspicious
        issues = []
        for sent in [
            s.strip()
            for s in re.split(r"(?<=[.!?])", summary)
            if len(s.strip()) > 0
        ]:
            sim_a = (
                self._cosine_similarity(sent, abstract) if abstract else 0.0
            )
            sim_b = (
                self._cosine_similarity(sent, arxiv_summary)
                if arxiv_summary
                else 0.0
            )
            if max(sim_a, sim_b) < 0.45 and re.search(r"[A-Za-z]", sent):
                issues.append(sent)
        return issues
``n

## File: paper_section_processor.py

`python
# stephanie/agents/thought/paper_section_processor.py
from __future__ import annotations
import json
import time
import uuid
from typing import Dict, Any
import traceback
from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.thought.paper_blog import SimplePaperBlogAgent
from stephanie.models.casebook import CaseBookORM
from stephanie.scoring.scorable_factory import TargetType

class PaperSectionProcessorAgent(BaseAgent):
    """
    Processes each section of a document individually, generates summaries,
    and logs the process to a case book for future reference.
    
    This agent:
    - Takes structured document data from DocumentProfilerAgent
    - Processes each section individually with the PaperSummarizer
    - Saves each section's input, summary, and metrics to a case book
    - Creates a comprehensive case book for the entire paper
    - Tracks section-by-section processing for future analysis
    
    # PRODUCTION-READY CASEBOOK MANAGEMENT
    #
    # This agent implements a structured casebook system that:
    # 1. Models a "blog casebook" explicitly with a structured naming convention
    # 2. Defines a tight case taxonomy (roles)
    # 3. Uses consistent scoring schema
    # 4. Implements global indexing strategy (not per-casebook)
    # 5. Manages entities and claims pipeline
    # 6. Tracks lineage between cases for provenance
    # 7. Supports governance & publishing gates
    #
    # Key principles:
    # - One casebook per blog post (not per paper)
    # - Cases are immutable; new iterations create new cases
    # - Text stored in CaseScorable rows, not in case meta
    # - Global index with metadata filters (not per-casebook indices)
    # - Clear role taxonomy for cases
    # - Lineage tracking between cases
    """
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.summarizer = SimplePaperBlogAgent(cfg, memory, container, logger)
        self.goal_template = cfg.get("goal_template", "academic_summary")
        self.min_section_length = cfg.get("min_section_length", 100)
        self.max_sections = cfg.get("max_sections", 10)
        
        # [CASEBOOK MANAGEMENT] Define casebook naming convention
        # Format: "blog::<paper_id>::<post_slug>"
        # Paper ID is the arXiv ID or document ID
        # Post slug is a normalized version of the post title
        self.casebook_name_template = cfg.get("casebook_name_template", "blog::{paper_id}::{post_slug}")
        
        # [CASEBOOK MANAGEMENT] Define case taxonomy (roles)
        # These are standardized roles for different types of cases
        self.case_roles = {
            "input_section": "raw section text from document",
            "summary_baseline": "Track A baseline summary",
            "summary_sharpened": "Track B sharpened summary",
            "summary_verified": "Track C verified summary",
            "critique": "hallucination/coverage notes",
            "citation_check": "citation verification results",
            "figure_grounding": "figure/table grounding results",
            "edit_patch": "diffs between draft versions",
            "final_section": "final approved section",
            "seo_meta": "SEO metadata for section",
            "social_snippet": "social media snippet for section",
            "claim": "extracted atomic claim",
            "entity": "linked entity information",
            "chat_turn": "conversation turn related to this section"
        }
        
        # [SCORING] Define consistent scoring schema
        # All metrics should be in [0,1] range with standardized names
        self.metrics_schema = [
            "overall", "coverage", "faithfulness", "coherence", "structure",
            "hallucination_rate", "knowledge_verification", "figure_grounding",
            "readability", "style_fit", "citation_support", "novelty", "stickiness"
        ]
        
        self.logger.info("PaperSectionProcessorAgent initialized", {
            "goal_template": self.goal_template,
            "min_section_length": self.min_section_length,
            "max_sections": self.max_sections,
            "case_roles": self.case_roles,
            "metrics_schema": self.metrics_schema
        })
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.report({
            "event": "start",
            "step": "PaperSectionProcessor",
            "details": "Processing document sections"
        })
        
        # Get structured document data from document_profiler
        documents = context.get(self.input_key, [])
        processed_sections = []
        # [CASEBOOK MANAGEMENT] Create casebook with structured naming
        casebook = self._get_casebook(context)
        
        for doc in documents:
            doc_id = doc.get("id")
            structured_data = doc.get("structured_data", {})
            title = doc.get("title", "")
            paper_id = doc.get("paper_id", doc_id)
            arxiv_id = doc.get("arxiv_id", "")
            
            if not structured_data:
                self.logger.warning(f"No structured data for document {doc_id}")
                continue
                
            self.logger.info(f"Processing document {doc_id} with {len(structured_data)} sections")
            
            # Process sections in order of importance
            section_order = ["abstract", "methods", "results", "conclusions", "introduction", "title"]
            sorted_sections = sorted(
                structured_data.items(),
                key=lambda x: section_order.index(x[0]) if x[0] in section_order else len(section_order)
            )
            
            for section_name, section_text in sorted_sections[:self.max_sections]:
                # Skip very short sections
                if len(section_text) < self.min_section_length:
                    self.logger.debug(f"Skipping short section '{section_name}' for doc {doc_id}")
                    continue
                    
                try:
                    # [CASEBOOK MANAGEMENT] Create context with proper casebook reference
                    section_context = {
                        "id": f"{doc_id}_{section_name}",
                        "title": title,
                        "summary": section_text,  # This is the section text
                        "section_name": section_name,
                        "goal_template": self.goal_template,
                        "paper_id": paper_id,
                        "arxiv_id": arxiv_id,
                        "pipeline_run_id": context.get("pipeline_run_id"),
                        "casebook_id": casebook.id,
                        "source": "document_profiler"
                    }
                    
                    # [CASEBOOK MANAGEMENT] Run summarizer on this section
                    section_summary = await self.summarizer.run(section_context)
                    
                    # [CASEBOOK MANAGEMENT] Save to case book with proper roles
                    self._save_section_to_casebook(
                        casebook, 
                        doc_id, 
                        section_name, 
                        section_text, 
                        section_summary,
                        context
                    )
                    
                    # [CASEBOOK MANAGEMENT] Track lineage and versioning
                    processed_sections.append({
                        "doc_id": doc_id,
                        "section_name": section_name,
                        "summary": section_summary.get("summary", ""),
                        "metrics": section_summary.get("metrics", {}),
                        "valid": section_summary.get("valid", False),
                        "case_id": section_summary.get("case_id"),  # Track the case ID
                        "version": section_summary.get("version", 1)  # Track version
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing section {section_name} for doc {doc_id}: {str(e)}")
                    traceback.print_exc()
                    
        context[self.output_key] = {
            "processed_sections": processed_sections,
            "casebook_id": casebook.id,
            "casebook_name": casebook.name  # Track the actual casebook name
        }
        
        self.report({
            "event": "end",
            "step": "PaperSectionProcessor",
            "details": f"Processed {len(processed_sections)} sections across {len(documents)} documents"
        })
        
        return context
        
    def _get_casebook(self, context: Dict[str, Any]) -> CaseBookORM:
        """Get or create a casebook for this processing run with structured naming
        
        # [CASEBOOK MANAGEMENT] Casebook naming convention:
        # Format: "blog::<paper_id>::<post_slug>"
        # Where:
        # - paper_id: arXiv ID or document ID
        # - post_slug: normalized version of the post title
        #
        # This ensures each blog post has its own dedicated casebook
        # with clear boundaries and provenance.
        """
        # Get paper metadata from context
        paper_id = context.get("paper_id")
        post_title = context.get("post_title", "blog")
        
        # Create normalized post slug (e.g., "my-paper-title" from "My Paper Title")
        post_slug = self._normalize_slug(post_title)
        
        # Build casebook name using template
        casebook_name = self.casebook_name_template.replace("{paper_id}", paper_id).replace("{post_slug}", post_slug)
        
        # [CASEBOOK MANAGEMENT] Create casebook with metadata
        casebook = self.memory.casebooks.get_casebook_by_name(casebook_name)
        if not casebook:
            casebook = self.memory.casebooks.create_casebook(
                name=casebook_name,
                description=f"Blog post casebook for paper {paper_id}",
                tag="blog_post",
                meta={
                    "paper_id": paper_id,
                    "post_title": post_title,
                    "post_slug": post_slug,
                    "created_at": time.time(),
                    "status": "draft",
                    "version": 1
                }
            )
        return casebook
    
    def _normalize_slug(self, title: str) -> str:
        """Convert title to URL-friendly slug"""
        return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    
    def _save_section_to_casebook(
        self,
        casebook: CaseBookORM,
        doc_id: str,
        section_name: str,
        section_text: str,
        section_summary: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Save a section and its summary to the case book with proper taxonomy
        
        # [CASEBOOK MANAGEMENT] Case taxonomy & storage layout:
        # - Text stored in CaseScorable rows (not in case meta)
        # - Different roles for different types of content
        # - Metrics stored as separate scorables
        # - Lineage tracking between cases
        # - Immutable cases (new iterations create new cases)
        #
        # Roles:
        # - "input_section": raw section text
        # - "summary_baseline": Track A summary
        # - "summary_sharpened": Track B summary
        # - "summary_verified": Track C summary
        # - "metrics": structured metrics
        # - "critique": critique notes
        # - "edit_patch": diffs between versions
        # - "final_section": final approved version
        # - "claim": extracted claims
        # - "entity": linked entities
        #
        # All cases are immutable - new iterations create new cases
        """
        # [CASEBOOK MANAGEMENT] Create case with proper role and lineage
        # Get current version from context if available
        version = section_summary.get("version", 1)
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=context.get("goal", {}).get("id"),
            prompt_text=json.dumps({
                "doc_id": doc_id,
                "section_name": section_name,
                "goal_template": self.goal_template,
                "version": version
            }),
            agent_name=self.name,
            role="input_section",  # This is the input section case
            meta={
                "type": "section_processing",
                "doc_id": doc_id,
                "section_name": section_name,
                "timestamp": time.time()
            }
        )
        
        # Save section text as scorable
        section_scorable = self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            scorable_type=TargetType.DOCUMENT_SECTION,
            text=section_text,
            role="input_section",
            meta={
                "doc_id": doc_id,
                "section_name": section_name,
                "length": len(section_text)
            }
        )
        
        # Save summary as scorable
        summary_text = section_summary.get("summary", "")
        if not summary_text:
            # Try to get summary from the summary object
            if "summary_v0" in section_summary and doc_id in section_summary["summary_v0"]:
                summary_text = section_summary["summary_v0"][doc_id].get("summary", "")
        
        summary_scorable = self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            scorable_type=TargetType.DYNAMIC,
            text=summary_text,
            role="summary",
            meta={
                "doc_id": doc_id,
                "section_name": section_name,
                "metrics": section_summary.get("metrics", {})
            }
        )
        
        # Save metrics as scorable
        metrics_text = json.dumps(section_summary.get("metrics", {}), indent=2)
        metrics_scorable = self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            scorable_type=TargetType.METRICS,
            text=metrics_text,
            role="metrics",
            meta={
                "doc_id": doc_id,
                "section_name": section_name
            }
        )
        
        # Log to case book
        self.logger.log("SectionProcessed", {
            "case_id": case.id,
            "doc_id": doc_id,
            "section_name": section_name,
            "summary_length": len(summary_text),
            "metrics": section_summary.get("metrics", {})
        })
        
        # Log to KnowledgeBus for tracking
        if hasattr(self.memory, "bus") and self.memory.bus:
            self.memory.bus.publish("section.processed", {
                "case_id": case.id,
                "doc_id": doc_id,
                "section_name": section_name,
                "summary_length": len(summary_text),
                "metrics": section_summary.get("metrics", {})
            })
``n

## File: sharpened_paper_summarizer.py

`python
# stephanie/agents/thought/sharpened_paper_summarizer.py
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.thought.paper_blog import \
    SimplePaperBlogAgent
from stephanie.scoring.scorable_factory import TargetType

MAX_ITERS_DEFAULT = 4
MIN_GAIN_DEFAULT = 0.02
MIN_OVERALL_DEFAULT = 0.75
TARGET_CONFIDENCE_DEFAULT = 0.85
MIN_FIGURE_SCORE_DEFAULT = 0.70
SENTS_MIN_DEFAULT = 4
SENTS_MAX_DEFAULT = 20


class SharpenedPaperSummarizerAgent(BaseAgent):
    """
    Track B: Super Sharpening (GROWS → CRITIC → REFLECT loop)

    Reads Track A baseline summaries from dynamic_scorables using provenance:
      source='paper_summarizer', source_scorable_type='document', source_scorable_id=<doc_id>

    Writes refined dynamic_scorables with provenance pointing to the baseline dynamic scorable.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Sharpening loop parameters
        self.max_iters = int(cfg.get("max_iters", MAX_ITERS_DEFAULT))
        self.min_gain = float(cfg.get("min_gain", MIN_GAIN_DEFAULT))
        self.min_overall = float(cfg.get("min_overall", MIN_OVERALL_DEFAULT))
        self.target_confidence = float(cfg.get("target_confidence", TARGET_CONFIDENCE_DEFAULT))
        self.min_figure_score = float(cfg.get("min_figure_score", MIN_FIGURE_SCORE_DEFAULT))

        # Sentence window (match Track A)
        self.min_sents = int(cfg.get("min_sents", SENTS_MIN_DEFAULT))
        self.max_sents = int(cfg.get("max_sents", SENTS_MAX_DEFAULT All right))

        # Training model keys
        self.model_key_ranker = cfg.get("model_key_ranker", "ranker.sicql.v1")
        self.model_key_retriever = cfg.get("model_key_retriever", "retriever.mrq.v1")

        # Metrics helper (reuse Track A deterministic metrics)
        self.metrics = SimplePaperBlogAgent(cfg, memory, container, logger)

        # Optional scoring service
        self.scoring = container.get("scoring")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.report({"event": "start", "step": "SuperSharpening", "details": "Track B sharpening loop"})

        documents = context.get("documents", []) or context.get(self.input_key, [])
        out_v1: Dict[Any, Dict[str, Any]] = {}

        for doc in documents:
            doc_id = doc.get("paper_id") or doc.get("id")
            if doc_id is None:
                self.logger.log("SharpenSkipNoDocId", {"doc": str(doc)[:200]})
                continue

            # 1) Pull baseline summary created by Track A via provenance
            baseline_obj = None
            try:
                sid = int(doc_id)
                baseline_obj = self.memory.dynamic_scorables.get_latest_by_source_pointer(
                    source="paper_summarizer",
                    source_scorable_type="document",
                    source_scorable_id=sid,
                )
            except Exception:
                # Non-numeric doc_id: optionally fall back by meta.paper_id
                baseline_obj = self.memory.dynamic_scorables.get_latest_by_source_and_meta(
                    source="paper_summarizer",
                    meta_key="paper_id",
                    meta_value=str(doc_id),
                )

            if not baseline_obj:
                self.logger.log("SharpenBaselineMissing", {
                    "doc_id": doc_id,
                    "hint": "Ensure Track A saved dynamic_scorables with source_scorable_id=doc_id and type='document', or meta.paper_id"
                })
                continue

            # Extract baseline summary text (prefer the ## Summary section, fallback to meta.summary)
            baseline_text = baseline_obj.text or ""
            baseline_summary = self._extract_summary_from_text(baseline_text) or (baseline_obj.meta or {}).get("summary", "")
            if not baseline_summary:
                self.logger.log("SharpenBaselineParseFailed", {
                    "doc_id": doc_id,
                    "scorable_id": getattr(baseline_obj, "id", None),
                })
                continue

            # 2) Sharpen
            enhanced = self._enhance_summary(
                doc_id=str(doc_id),
                baseline_summary=baseline_summary,
                paper_data=doc,
                context=context,
            )
            out_v1[doc_id] = enhanced

            # 3) Persist refined scorable with provenance to the baseline scorable
            refined_obj = None
            try:
                refined_text = f"## Summary\n{enhanced['summary']}".strip()
                abstract = self._fetch_abstract(doc.get("id") or doc_id)
                refined_obj = self.memory.dynamic_scorables.add(
                    pipeline_run_id=context.get("pipeline_run_id"),
                    scorable_type=TargetType.DYNAMIC,
                    source=self.name, 
                    text=refined_text,
                    source_scorable_id=int(baseline_obj.id),
                    source_scorable_type="dynamic",
                    meta={
                        "paper_id": doc_id,
                        "title": doc.get("title", ""),
                        "abstract": abstract,
                        "arxiv_summary": doc.get("summary", ""),
                        "text": refined_text,
                        "summary": enhanced.get("summary", ""),
                        "metrics": enhanced.get("metrics", {}),
                        "hallucination_issues": enhanced.get("hallucination_issues", []),
                        "figure_results": enhanced.get("figure_results", {}),
                        "origin": "track_b_super_sharpening",
                        "iterations": enhanced.get("iterations", []),
                        "processing_time": enhanced.get("processing_time", 0),
                        "passes_guardrails": enhanced.get("passes_guardrails", False),
                        "converged": enhanced.get("converged", False),
                    },
                )
                # ensure embedding
                self.memory.embedding.get_or_create(refined_text)
            except Exception as e:
                self.memory.session.rollback()
                self.logger.log("SharpenRefinedPersistError", {"doc_id": doc_id, "error": str(e)})

            # 4) Emit training events
            try:
                # Build paper bundle for metrics/grounding
                paper_bundle = {
                    "paper_id": doc_id,
                    "title": doc.get("title", ""),
                    "abstract": self._fetch_abstract(doc.get("id") or doc_id),
                    "arxiv_summary": doc.get("summary", ""),
                }

                # Baseline metrics (if we have them in baseline_obj.meta), else recompute quickly
                baseline_metrics = (baseline_obj.meta or {}).get("metrics")
                if not baseline_metrics:
                    baseline_metrics = self._score_summary(
                        baseline_summary, paper_bundle["abstract"], paper_bundle["arxiv_summary"]
                    )

                # Emit only if our enhanced pass actually exists
                if enhanced and "metrics" in enhanced:
                    self._emit_training_events(
                        paper=paper_bundle,
                        baseline_summary=baseline_summary,
                        enhanced_summary=enhanced["summary"],
                        baseline_metrics=baseline_metrics,
                        enhanced_metrics=enhanced["metrics"],
                        context=context,
                    )
            except Exception as e:
                self.memory.session.rollback()
                self.logger.log("SharpenTrainingEventError", {"doc_id": doc_id, "error": str(e)})

        context.setdefault("summary_v1", {})
        context["summary_v1"] = out_v1
        return context

    # ---------- Core sharpening prompt ----------
    def _build_super_sharpen_prompt(self, *, title: str, abstract: str, summary: str,
                                    min_sents: int, max_sents: int) -> str:
        """
        A single 'super' sharpening prompt that wraps GROWS + CRITIC + REFLECT.
        Keeps output format to exactly the improved summary paragraph.
        """
        abstract_snip = (abstract or "")[:1000]
        return f"""
You are an expert science editor. Improve the paper summary below using a combined **GROWS + CRITIC + REFLECT** loop.

GROWS:
- Generate alternatives, Review against abstract, Optimize for clarity/flow, Work again on weak spots, Stop when optimal.

CRITIC:
- Find assumptions, spot gaps/overclaims, propose precise fixes grounded in the abstract, then rewrite.

REFLECT:
- Double-check factuality and faithfulness to the abstract; remove speculation and marketing language.

Constraints:
- Output **one paragraph** of {min_sents}-{max_sents} sentences.
- Use ONLY facts present in the abstract; if a detail is missing, prefer generic phrasing over guessing.
- Avoid first person, questions, citations/links, and equations.
- If you mention numbers/metrics, only keep those clearly present in the abstract context.

Paper Title: {title}

Abstract:
\"\"\"
{abstract_snip}
\"\"\"

Current summary:
\"\"\"{summary}\"\"\"

Rewrite now (one paragraph, {min_sents}-{max_sents} sentences):
""".strip()

    def _enhance_summary(self, doc_id: str, baseline_summary: str, paper_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        title = paper_data.get("title", "")
        abstract = self._fetch_abstract(paper_data.get("id") or doc_id)
        arxiv_summary = paper_data.get("summary", "")

        best_summary = baseline_summary.strip()
        best_metrics = self._score_summary(best_summary, abstract, arxiv_summary)

        iterations: List[Dict[str, Any]] = []
        no_gain = 0

        for i in range(self.max_iters):
            prompt = self._build_super_sharpen_prompt(
                title=title,
                abstract=abstract,
                summary=best_summary,
                min_sents=self.min_sents,
                max_sents=self.max_sents,
            )
            candidate = self.call_llm(prompt, context=context).strip()
            candidate = self._extract_summary_from_text(candidate)

            cand_metrics = self._score_summary(candidate, abstract, arxiv_summary)
            gain = cand_metrics["overall"] - best_metrics["overall"]

            iterations.append({
                "iteration": i + 1,
                "candidate_overall": cand_metrics["overall"],
                "current_best": best_metrics["overall"],
                "gain": gain,
            })

            if cand_metrics["overall"] >= self.min_overall and gain >= self.min_gain:
                best_summary, best_metrics = candidate, cand_metrics
                no_gain = 0
            else:
                no_gain += 1

            if best_metrics["overall"] >= self.target_confidence or no_gain >= 2:
                break

        ok_hall, hallucinations = self._verify_hallucinations(best_summary, abstract, arxiv_summary)
        fig_check = self._verify_figure_grounding(best_summary, paper_data)
        passes = ok_hall and (fig_check["overall_figure_score"] >= self.min_figure_score)

        return {
            "summary": best_summary,
            "metrics": best_metrics,
            "iterations": iterations,
            "processing_time": time.time() - start,
            "hallucination_issues": hallucinations,
            "figure_results": fig_check,
            "passes_guardrails": passes,
            "converged": best_metrics["overall"] >= self.target_confidence,
        }

    # ---------- helpers reused from Track A ----------
    def _extract_summary_from_text(self, text: str) -> str:
        m = re.search(r"(?mi)^##\s*Summary\s*\n(.+?)(?=^##|\Z)", text, re.S)
        return (m.group(1).strip() if m else text.strip())

    def _score_summary(self, summary: str, abstract: str, arxiv_summary: str) -> Dict[str, float]:
        return self.metrics._compute_metrics(summary, abstract, arxiv_summary)

    def _verify_hallucinations(self, summary: str, abstract: str, arxiv_summary: str) -> Tuple[bool, List[str]]:
        issues = self.metrics._detect_hallucinations(summary, abstract, arxiv_summary)
        return len(issues) == 0, issues

    def _verify_figure_grounding(self, summary: str, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify figure/table grounding with precise claim matching."""
        # Find quantitative claims with context
        quant_claims = []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
        
        for sent in sentences:
            matches = re.findall(
                r"(\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b)", 
                sent, 
                flags=re.I
            )
            if matches:
                quant_claims.append({
                    "claim": sent,
                    "value": matches[0][0],
                    "metric": matches[0][1],
                    "has_citation": any(marker in sent.lower() for marker in ["figure", "fig.", "table", "as shown", "see"])
                })
        
        # Check if citations match paper content
        properly_cited = 0
        for claim in quant_claims:
            if claim["has_citation"]:
                # In production, this would check if the citation actually supports the claim
                # For now, simple heuristic
                properly_cited += 1
        
        citation_rate = properly_cited / max(1, len(quant_claims))
        
        return {
            "total_claims": len(quant_claims),
            "properly_cited": properly_cited,
            "citation_rate": citation_rate,
            "overall_figure_score": citation_rate,
            "claims": quant_claims
        }

    def _fetch_abstract(self, doc_id) -> str:
        try:
            sections = self.memory.document_sections.get_by_document(doc_id)
            for s in sections:
                sd = s.to_dict()
                if (sd.get("section_name") or "").lower().strip() == "abstract":
                    return sd.get("section_text", "") or ""
        except Exception as e:
            self.logger.log("AbstractFetchFailed", {"doc_id": doc_id, "error": str(e)})
        return ""

    def _emit_training_events(
        self,
        paper: Dict[str, Any],
        baseline_summary: str,
        enhanced_summary: str,
        baseline_metrics: Dict[str, float],
        enhanced_metrics: Dict[str, float],
        context: Dict[str, Any],
    ):
        title = paper.get("title", "paper")
        gain = float(enhanced_metrics.get("overall", 0.0) - (baseline_metrics or {}).get("overall", 0.0))
        w = max(0.1, min(1.0, gain + 0.3))

        # pointwise enhanced
        self.memory.training_events.add_pointwise(
            model_key=self.model_key_retriever,
            dimension="alignment",
            query_text=title,
            cand_text=enhanced_summary,
            label=1,
            weight=float(enhanced_metrics.get("overall", 0.7)),
            trust=float(enhanced_metrics.get("overall", 0.7)),
            goal_id=context.get("goal", {}).get("id"),
            pipeline_run_id=context.get("pipeline_run_id"),
            agent_name=self.name,
            source="track_b",
            meta={"stage": "track_b", "gain": gain},
        )

        # pairwise enhanced vs baseline
        self.memory.training_events.add_pairwise(
            model_key=self.model_key_ranker,
            dimension="alignment",
            query_text=title,
            pos_text=enhanced_summary,
            neg_text=baseline_summary,
            weight=w,
            trust=w * 0.6,
            goal_id=context.get("goal", {}).get("id"),
            pipeline_run_id=context.get("pipeline_run_id"),
            agent_name=self.name,
            source="track_b",
            meta={
                "stage": "track_b",
                "enhanced_score": enhanced_metrics.get("overall"),
                "baseline_score": (baseline_metrics or {}).get("overall"),
                "gain": gain,
            },
        )

        # pairwise vs author summary (optional)
        arxiv_summary = paper.get("arxiv_summary", "") or ""
        if arxiv_summary.strip():
            author_metrics = self._score_summary(arxiv_summary, paper.get("abstract", ""), arxiv_summary)
            prefer_enhanced = enhanced_metrics["overall"] > author_metrics["overall"]
            pos = enhanced_summary if prefer_enhanced else arxiv_summary
            neg = arxiv_summary if prefer_enhanced else enhanced_summary

            self.memory.training_events.add_pairwise(
                model_key=self.model_key_ranker,
                dimension="alignment",
                query_text=title,
                pos_text=pos,
                neg_text=neg,
                weight=0.5,
                trust=0.3,
                goal_id=context.get("goal", {}).get("id"),
                pipeline_run_id=context.get("pipeline_run_id"),
                agent_name=self.name,
                source="track_b",
                meta={
                    "stage": "track_b",
                    "enhanced_score": enhanced_metrics["overall"],
                    "author_score": author_metrics["overall"],
                    "prefer_enhanced": prefer_enhanced,
                },
            )
``n

## File: sot_v01_agent.py

`python
# stephanie/agents/thought/sot_v01_agent.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from .sot_v01_dataset_builder import SoTV01DatasetBuilder
from .sot_v01_trainer import SoTV01Trainer

class SoTV01Agent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.chat_store = memory.chats
        self.embedding_service = container.get("embedding")
        self.dataset_path = cfg.get("dataset_path", "data/sot_v01_dataset.jsonl")
        self.model_path = cfg.get("model_path", "models/sot_v01")

    async def run(self, context: dict) -> dict:
        # Step 1: Build the dataset
        builder = SoTV01DatasetBuilder(self.chat_store, self.embedding_service, self.logger)
        builder.build_dataset(self.dataset_path, max_conversations=500)

        # Step 2: Train the model
        trainer = SoTV01Trainer()
        trainer.train(self.dataset_path, self.model_path, epochs=3, batch_size=4)

        context["sot_v01_trained"] = True
        return context
``n

## File: sot_v01_collator.py

`python
# stephanie/agents/thought/sot_v01_collator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import torch

@dataclass
class SoTDataCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Each item in batch should have fields pre-assembled:
        #   prompt_text, target_text, move_label (int)
        prompts = [b["prompt_text"] for b in batch]
        targets = [b["target_text"] + self.tokenizer.eos_token for b in batch]

        # Tokenize separately to compute prompt lengths
        tok_prompt = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        tok_full   = self.tokenizer([p+t for p,t in zip(prompts, targets)],
                                    padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        input_ids = tok_full.input_ids
        attention_mask = tok_full.attention_mask

        labels = input_ids.clone()
        # mask prompt part
        for i, plen in enumerate(tok_prompt.input_ids.shape[1] for _ in prompts):
            # recompute exact prompt lengths per sample from non-pad count
            plen = int((tok_prompt.attention_mask[i] == 1).sum().item())
            labels[i, :plen] = -100

        prompt_lengths = []
        for i in range(len(prompts)):
            plen = int((tok_prompt.attention_mask[i] == 1).sum().item())
            prompt_lengths.append(plen)
        prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long)

        move_labels = torch.tensor([b["move_label"] for b in batch], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "move_labels": move_labels,
            "prompt_lengths": prompt_lengths
        }
``n

## File: sot_v01_dataset_builder_pg.py

`python
# stephanie/agents/thought/sot_v01_dataset_builder_pg.py
from __future__ import annotations

import json
from sqlalchemy.orm import Session
from stephanie.memory.chat_store import ChatStore
from stephanie.trajectory.pgvector_ann import PgVectorANN  # from earlier


class SoTV01DatasetBuilderPg:
    """
    Build SoT v0.1 dataset using pgvector ANN and span expansion.
    Each example:
      - query (user text)
      - retrieved_turns (few examples from spans)
      - response (assistant text)
      - predicted_move (from meta, if present)
    """

    def __init__(self, session: Session, memory, logger=None, window: int = 2):
        self.session = session
        self.store = ChatStore(session)
        self.memory = memory
        self.logger = logger
        self.window = window
        self.ann = PgVectorANN(
            session
        )  # table/cols default to your earlier names

        # batch embed via your memory (ad-hoc namespace)
        import numpy as np

        def embed_batch(texts):
            vecs = []
            for t in texts:
                v = self.memory.get_or_create(t, {"namespace": "ad_hoc"})
                import numpy as _np

                v = _np.array(v, dtype=float).reshape(-1)
                v /= float(_np.linalg.norm(v)) + 1e-12
                vecs.append(v)
            return np.vstack(vecs) if vecs else np.zeros((0, 1))

        self.embed_batch = embed_batch

    def _reasoning_move(self, turn) -> str:
        meta = (
            (turn.assistant_message.meta or {})
            if turn.assistant_message
            else {}
        )
        return (meta.get("reasoning_move") or "VOICE").upper()

    def _user_text(self, turn) -> str:
        return (
            (turn.user_message.text or "").strip() if turn.user_message else ""
        )

    def _assistant_text(self, turn) -> str:
        return (
            (turn.assistant_message.text or "").strip()
            if turn.assistant_message
            else ""
        )

    def _retrieve_spans_fast(
        self,
        section_text: str,
        exclude_conv_id: int,
        exclude_turn_id: int,
        k: int = 20,
        top_spans: int = 3,
    ):
        # ANN → expand spans → score mean sim → exclude same conversation vicinity
        import numpy as np

        q = self.embed_batch([section_text])[0]
        hits = self.ann.search(q, k=k)  # [(turn_id, sim)]

        out, seen_convs = [], set()
        for turn_id, _ in hits:
            t = self.store.get_turn_by_id(turn_id)
            if not t:
                continue
            # exclude same conversation within +/- window to avoid leakage
            if t.conversation_id == exclude_conv_id:
                # find index to compare distance
                turns = self.store.get_turns_for_conversation(exclude_conv_id)
                idx = next(
                    (
                        i
                        for i, x in enumerate(turns)
                        if x.id == exclude_turn_id
                    ),
                    None,
                )
                jdx = next(
                    (j for j, x in enumerate(turns) if x.id == turn_id), None
                )
                if (
                    idx is not None
                    and jdx is not None
                    and abs(idx - jdx) <= self.window
                ):
                    continue

            # expand span around the hit
            turns = self.store.get_turns_for_conversation(t.conversation_id)
            center = next(
                (i for i, x in enumerate(turns) if x.id == turn_id), 0
            )
            lo, hi = (
                max(0, center - self.window),
                min(len(turns) - 1, center + self.window),
            )
            span = turns[lo : hi + 1]

            # mean sim across span
            texts = []
            for s in span:
                u = (s.user_message.text or "") if s.user_message else ""
                a = (
                    (s.assistant_message.text or "")
                    if s.assistant_message
                    else ""
                )
                texts.append(f"USER: {u}\nYOU: {a}")
            V = self.embed_batch(texts) if texts else None
            span_sim = (
                float(np.mean(V @ q))
                if (V is not None and len(V) > 0)
                else 0.0
            )

            if t.conversation_id in seen_convs:
                continue
            seen_convs.add(t.conversation_id)

            out.append(
                {
                    "score": span_sim,
                    "conversation_id": t.conversation_id,
                    "span": span,
                }
            )
            if len(out) >= top_spans:
                break

        out.sort(key=lambda d: d["score"], reverse=True)
        return out

    def build_dataset(
        self,
        output_path: str,
        max_conversations: int = 1000,
        top_spans: int = 3,
    ):
        top = self.store.get_top_conversations(
            limit=max_conversations, by="turns"
        )
        total = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for conv, _ in top:
                turns = self.store.get_turns_for_conversation(conv.id)
                for t in turns:
                    q = self._user_text(t)
                    a = self._assistant_text(t)
                    if not q or not a:
                        continue
                    spans = self._retrieve_spans_fast(
                        q,
                        exclude_conv_id=conv.id,
                        exclude_turn_id=t.id,
                        top_spans=top_spans,
                    )
                    ex = {
                        "query": q,
                        "retrieved_turns": [
                            {
                                "user": (s.user_message.text or "")
                                if s.user_message
                                else "",
                                "assistant": (s.assistant_message.text or "")
                                if s.assistant_message
                                else "",
                                "conversation_id": s.conversation_id,
                            }
                            for block in spans
                            for s in block["span"]
                        ],
                        "response": a,
                        "predicted_move": self._reasoning_move(t),
                        "conversation_id": conv.id,
                        "turn_id": t.id,
                    }
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    total += 1
        if self.logger:
            self.logger.info(
                f"[SoTV01DatasetBuilderPg] wrote {total} examples to {output_path}"
            )
``n

## File: sot_v01_dataset_builder.py

`python
# stephanie/agents/thought/sot_v01_dataset_builder.py
from __future__ import annotations

from typing import List, Dict, Any
from tqdm import tqdm
import json

class SoTV01DatasetBuilder:
    """
    Builds a dataset of thought trajectories for SoT v0.1.
    Each example is: (query, retrieved_turns, your_response, predicted_move)
    """

    def __init__(self, chat_store, embedding_service, logger=None):
        self.chat_store = chat_store
        self.embedding_service = embedding_service
        self.logger = logger

    def build_dataset(self, output_path: str, max_conversations: int = 1000):
        """
        Build the dataset and save it as a JSONL file.
        """
        # Get top conversations (by turns or messages)
        top_convs = self.chat_store.get_top_conversations(limit=max_conversations, by="turns")
        
        dataset = []
        for conv, _ in tqdm(top_convs, desc="Building SoT v0.1 Dataset"):
            turns = self.chat_store.get_turns_for_conversation(conv.id)
            for turn in turns:
                if not turn.user_message or not turn.assistant_message:
                    continue

                user_query = turn.user_message.text.strip()
                your_response = turn.assistant_message.text.strip()

                if not user_query or not your_response:
                    continue

                # Retrieve 1-2 most similar past turns (excluding this one)
                retrieved_turns = self._retrieve_similar_turns(user_query, current_turn_id=turn.id, top_k=2)

                # Get the predicted move (from meta)
                predicted_move = self._get_predicted_move(turn)

                # Create training example
                example = {
                    "query": user_query,
                    "retrieved_turns": [
                        {
                            "user": rt.user_message.text.strip() if rt.user_message else "",
                            "assistant": rt.assistant_message.text.strip() if rt.assistant_message else "",
                            "conversation_title": rt.conversation.title if rt.conversation else "Unknown",
                            "similarity_score": getattr(rt, 'similarity_score', 0.0)
                        }
                        for rt in retrieved_turns
                    ],
                    "response": your_response,
                    "predicted_move": predicted_move,
                    "conversation_id": conv.id,
                    "turn_id": turn.id
                }

                dataset.append(example)

        # Save dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        if self.logger:
            self.logger.info(f"[SoTV01DatasetBuilder] Dataset saved to {output_path}. Total examples: {len(dataset)}")

    def _retrieve_similar_turns(self, query: str, current_turn_id: int, top_k: int = 2) -> List[Any]:
        """
        Simple semantic search: find the top_k most similar turns to the query.
        Excludes the current turn.
        """
        query_embedding = self.embedding_service.get_or_create(query)
        all_turns = self._get_all_turns_excluding(current_turn_id)

        similarities = []
        for turn in all_turns:
            if not turn.user_message or not turn.user_message.text:
                continue
            turn_embedding = self.embedding_service.get_or_create(turn.user_message.text)
            similarity = self._cosine_similarity(query_embedding, turn_embedding)
            similarities.append((similarity, turn))

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_turns = [turn for _, turn in similarities[:top_k]]
        for i, turn in enumerate(top_turns):
            turn.similarity_score = similarities[i][0]
        return top_turns

    def _get_all_turns_excluding(self, turn_id_to_exclude: int) -> List[Any]:
        """Fetch all ChatTurnORMs except the one with the given ID."""
        all_conversations = self.chat_store.get_all(limit=1000)
        all_turns = []
        for conv in all_conversations:
            turns = self.chat_store.get_turns_for_conversation(conv.id)
            for turn in turns:
                if turn.id != turn_id_to_exclude:
                    all_turns.append(turn)
        return all_turns

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        import numpy as np
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    @staticmethod
    def _get_predicted_move(turn) -> str:
        """Extract the predicted move from the turn's meta."""
        meta = turn.assistant_message.meta if turn.assistant_message else {}
        return meta.get("reasoning_move", "VOICE")  # Default to "VOICE"
``n

## File: sot_v01_dataset.py

`python
# stephanie/agents/thought/sot_v01_dataset.py
from __future__ import annotations

import json
from torch.utils.data import Dataset

class SoTV01Dataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line)
                # skip empties
                if not ex.get("query") or not ex.get("response"): 
                    continue
                self.examples.append(ex)

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        move = (ex.get("predicted_move") or "VOICE").upper()
        move_label = MOVE2ID.get(move, MOVE2ID["VOICE"])

        prompt = self._prompt(ex["query"], ex.get("retrieved_turns", []), move)
        target = ex["response"]

        return {
            "prompt_text": prompt,
            "target_text": target,
            "move_label": move_label
        }

    def _prompt(self, query: str, retrieved_turns: list, move: str) -> str:
        # Few-shot style like your composer
        parts = [f"[PREDICTED MOVE: {move}]",
                 "[RETRIEVED CONTEXT]"]
        # cap to ~3 examples worth of lines
        lines = []
        for rt in retrieved_turns[:12]:  # 6 user/assistant pairs ~ 3 examples
            u = (rt.get("user") or "").strip()
            a = (rt.get("assistant") or "").strip()
            if u or a:
                lines.append(f"User: {u}\nYou: {a}")
        if lines:
            parts.append("\n\n".join(lines))
        parts.append("[END RETRIEVED CONTEXT]\n")
        parts.append(f"User: {query}\nYou: ")
        return "\n".join(parts)
``n

## File: sot_v01_multitask.py

`python
# stephanie/agents/thought/sot_v01_multitask.py
from __future__ import annotations

import torch
import torch.nn as nn

MOVE_LABELS = ["VOICE","OUTLINE","CRITIQUE","CODE","REFACTOR","PLAN","MATH","DERIVE","SEARCH","RETRIEVE"]
MOVE2ID = {m:i for i,m in enumerate(MOVE_LABELS)}

class SoTMultiTaskWrapper(nn.Module):
    """
    Wraps a Causal LM with a small classifier head for move prediction.
    - lm: AutoModelForCausalLM
    """
    def __init__(self, lm, hidden_size:int, num_moves:int=len(MOVE_LABELS), move_loss_weight:float=0.2):
        super().__init__()
        self.lm = lm
        self.move_head = nn.Linear(hidden_size, num_moves)
        self.move_loss_weight = move_loss_weight

    def forward(self, input_ids=None, attention_mask=None, labels=None, move_labels=None, prompt_lengths=None):
        # LM forward
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lm_loss = out.loss

        # Move classification: take the hidden state at the last prompt token per sample
        # prompt_lengths: (B,) tensor with prompt length in tokens for each sample
        hidden_states = out.hidden_states[-1]     # (B, T, H)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        idxs = torch.clamp(prompt_lengths-1, min=0)
        reps = hidden_states[batch_idx, idxs]     # (B, H)

        logits = self.move_head(reps)             # (B, num_moves)
        move_loss = nn.functional.cross_entropy(logits, move_labels) if move_labels is not None else 0.0

        loss = lm_loss + self.move_loss_weight * move_loss
        return type("Out", (), {"loss": loss, "lm_loss": lm_loss, "move_loss": move_loss, "logits": out.logits, "move_logits": logits})
``n

## File: sot_v01_trainer.py

`python
# stephanie/agents/thought/sot_v01_trainer.py
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from .sot_v01_dataset import SoTV01Dataset  # You'll need to create this (see below)

class SoTV01Trainer:
    def __init__(self, model_name: str = "Qwen/Qwen1.5-0.5B", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def train(self, train_dataset_path: str, output_dir: str, epochs: int = 3, batch_size: int = 4):
        train_dataset = SoTV01Dataset(train_dataset_path, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            report_to="none",
            save_total_limit=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            fp16=True if self.device == "cuda" else False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")
``n

## File: verify_pipeline.py

`python
# stephanie/agents/thought/verify_pipeline.py
from __future__ import annotations

import asyncio
import os
import re
import json
import time
import math
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----- External deps from your codebase -----
from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.summary.paper_summarizer import SimplePaperBlogAgent
from stephanie.knowledge.anti_hallucination import AntiHallucination
from stephanie.knowledge.figure_grounding import FigureGrounding
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.utils.json_sanitize import sanitize_for_json
from stephanie.models.strategy import StrategyProfile
from stephanie.utils.casebook_utils import generate_casebook_name


# ==========================
# 1) Config
# ==========================
@dataclass
class VerifierConfig:
    max_iters: int = 5
    min_gain: float = 0.015
    min_overall: float = 0.80
    target_confidence: float = 0.95
    min_figure_score: float = 0.80
    verification_threshold: float = 0.90
    convergence_window: int = 2
    knowledge_graph_conf: float = 0.70
    sents_min: int = 4
    sents_max: int = 20
    cbr_cases: int = 3
    hrm_weight: float = 0.10
    use_cbr: bool = True
    use_hrm: bool = True
    use_zeromodel: bool = True
    use_descendants_metric: bool = False
    strategy_scope: str = "track_c"
    report_dir: str = "reports/track_c"
    vpm_dir: str = "reports/vpm"
    enable_audit_report: bool = True

    model_key_ranker: str = "ranker.sicql.v1"
    model_key_retriever: str = "retriever.mrq.v1"


# ==========================
# 2) Visualization (VPM) Emitter
# ==========================
class VPMEmitter:
    """
    Emits VPM images. Uses your zero_model_service if available;
    otherwise falls back to matplotlib PNG/GIFs.
    """
    def __init__(self, logger, zeromodel_service, out_dir: str = "reports/vpm"):
        self.logger = logger
        self.zm = zeromodel_service
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def emit_abc_tile(self, doc_id: str, metrics_a: dict, metrics_b: dict, metrics_c: dict) -> Optional[str]:
        try:
            if self.zm and hasattr(self.zm, "generate_summary_vpm_tiles"):
                payload = {
                    "vpm_data": {
                        "doc_id": str(doc_id),
                        "title": "",
                        "metrics": {
                            "A": self._pack(metrics_a),
                            "B": self._pack(metrics_b),
                            "C": self._pack(metrics_c),
                        },
                        "iterations": [],
                        "timestamp": time.time(),
                    },
                    "output_dir": self.out_dir,
                }
                res = self.zm.generate_summary_vpm_tiles(**payload) or {}
                return res.get("quality_tile_path")
            # fallback to simple matplotlib strip
            return self._matplotlib_abc_tile(doc_id, metrics_a, metrics_b, metrics_c)
        except Exception as e:
            self.logger.log("VPMEmitABCTileError", {"doc_id": doc_id, "error": str(e)})
            return None

    def emit_iteration_timeline(self, doc_id: str, iterations: List[Dict[str, Any]]) -> Optional[str]:
        try:
            return self._matplotlib_iteration_line(doc_id, iterations)
        except Exception as e:
            self.logger.log("VPMEmitIterTimelineError", {"doc_id": doc_id, "error": str(e)})
            return None

    def emit_panel_heatmap(self, doc_id: str, panel_detail: Dict[str, Any]) -> Optional[str]:
        """
        Heatmap rows = roles (skeptic/editor/risk), cols = key sub-metrics,
        values = normalized 0..1 for quick visual compare.
        """
        try:
            return self._matplotlib_panel_heatmap(doc_id, panel_detail or {})
        except Exception as e:
            self.logger.log("VPMEmitPanelHeatmapError", {"doc_id": doc_id, "error": str(e)})
            return None

    def emit_knowledge_progress(self, doc_id: str, iterations: List[Dict[str, Any]]) -> Optional[str]:
        """
        Line plot of knowledge_verification (or claim_coverage/evidence_strength) across iterations.
        """
        try:
            return self._matplotlib_knowledge_progress(doc_id, iterations)
        except Exception as e:
            self.logger.log("VPMEmitKnowledgeProgressError", {"doc_id": doc_id, "error": str(e)})
            return None

    # ---- helpers ----
    def _pack(self, m: dict) -> dict:
        return {
            "overall": float(m.get("overall", 0.0)),
            "coverage": float(m.get("claim_coverage", m.get("coverage", 0.0))),
            "faithfulness": float(m.get("faithfulness", 0.0)),
            "structure": float(m.get("structure", 0.0)),
            "no_halluc": float(1.0 - m.get("hallucination_rate", 1.0)),
            "figure_ground": float(
                (m.get("figure_results", {}) or {}).get("overall_figure_score", 0.0)
            ) if isinstance(m.get("figure_results"), dict) else 0.0,
        }

    def _matplotlib_abc_tile(self, doc_id: str, A: dict, B: dict, C: dict) -> Optional[str]:
        try:
            import matplotlib
            if matplotlib.get_backend().lower() != "agg":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            self.logger.log("MatplotlibMissing", {"for": "abc_tile"})
            return None

        names = ["overall", "coverage", "faithfulness", "structure", "no_halluc", "figure_ground"]
        mat = np.array([
            [self._pack(A)[k] for k in names],
            [self._pack(B)[k] for k in names],
            [self._pack(C)[k] for k in names],
        ], dtype=np.float32)

        fig, ax = plt.subplots(figsize=(8, 2.6))
        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_yticks([0,1,2], labels=["A", "B", "C"])
        ax.set_xticks(range(len(names)), labels=names, rotation=20, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out = os.path.join(self.out_dir, f"{doc_id}_abc.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def _matplotlib_iteration_line(self, doc_id: str, iters: List[Dict[str, Any]]) -> Optional[str]:
        if not iters:
            return None
        try:
            import matplotlib
            if matplotlib.get_backend().lower() != "agg":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            self.logger.log("MatplotlibMissing", {"for": "iteration_timeline"})
            return None

        xs = [it["iteration"] for it in iters]
        cs = [float(it.get("current_score", 0.0)) for it in iters]
        ys = [float(it.get("best_candidate_score", 0.0)) for it in iters]

        fig, ax = plt.subplots(figsize=(8.6, 4.0))
        ax.plot(xs, cs, linewidth=2, label="current score")
        ax.plot(xs, ys, linewidth=2, label="candidate score")
        ax.set_title("Per-Iteration Scores (Track C)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Overall")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        out = os.path.join(self.out_dir, f"{doc_id}_timeline.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def _matplotlib_panel_heatmap(self, doc_id: str, panel_detail: Dict[str, Any]) -> Optional[str]:
        panel = panel_detail.get("panel") or []
        if not panel:
            return None
        try:
            import matplotlib
            if matplotlib.get_backend().lower() != "agg":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            self.logger.log("MatplotlibMissing", {"for": "panel_heatmap"})
            return None

        roles = [p.get("role","?") for p in panel]
        cols = ["overall", "claim_coverage", "faithfulness", "structure", "hallucination_rate"]
        M = []
        for p in panel:
            m = p.get("metrics", {}) or {}
            row = [
                float(m.get("overall", 0.0)),
                float(m.get("claim_coverage", m.get("coverage", 0.0))),
                float(m.get("faithfulness", 0.0)),
                float(m.get("structure", 0.0)),
                1.0 - float(m.get("hallucination_rate", 1.0)),
            ]
            M.append(row)
        M = np.array(M, dtype=np.float32)

        fig, ax = plt.subplots(figsize=(7, 2.2 + 0.3*len(roles)))
        im = ax.imshow(M, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_yticks(range(len(roles)), labels=roles)
        ax.set_xticks(range(len(cols)), labels=cols, rotation=20, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("PACS Panel Metrics")
        out = os.path.join(self.out_dir, f"{doc_id}_panel.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def _matplotlib_knowledge_progress(self, doc_id: str, iters: List[Dict[str, Any]]) -> Optional[str]:
        try:
            import matplotlib
            if matplotlib.get_backend().lower() != "agg":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            self.logger.log("MatplotlibMissing", {"for": "knowledge_progress"})
            return None

        xs = [it["iteration"] for it in iters]
        kv = []
        es = []
        for it in iters:
            kv.append(float(it.get("claim_coverage", it.get("knowledge_verification", 0.0))))
            es.append(float(it.get("evidence_strength", 0.0)))

        fig, ax = plt.subplots(figsize=(8.6, 4.0))
        ax.plot(xs, kv, linewidth=2, label="claim/evidence coverage")
        ax.plot(xs, es, linewidth=2, label="evidence strength")
        ax.set_title("Knowledge Verification Progress")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        out = os.path.join(self.out_dir, f"{doc_id}_knowledge.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out


# ==========================
# 3) Knowledge Graph Builder
# ==========================
class KnowledgeGraphBuilder:
    def __init__(self, container, logger):
        self.container = container
        self.logger = logger

    async def build(self, doc_id: str, paper_text: str, chat_corpus: List[dict], trajectories: List[dict], domains: List[dict]) -> Dict[str, Any]:
        def _empty() -> Dict[str, Any]:
            return {
                "nodes": [], "relationships": [], "claims": [],
                "claim_coverage": 0.0, "evidence_strength": 0.0,
                "temporal_coherence": 0.0, "domain_alignment": 0.0,
                "knowledge_gaps": [], "meta": {"paper_id": str(doc_id)}
            }

        svc = self.container.get("knowledge_graph")
        if not (svc and hasattr(svc, "build_tree")):
            self.logger.log("KGMissingBuildTree", {"doc_id": doc_id})
            return _empty()

        try:
            kg = await asyncio.to_thread(
                svc.build_tree,
                paper_text=paper_text or "",
                paper_id=str(doc_id),
                chat_corpus=chat_corpus or [],
                trajectories=trajectories or [],
                domains=domains or [],
            )
            if not isinstance(kg, dict):
                kg = {}
            kg = kg.get("knowledge_graph") or kg
            for k, v in _empty().items():
                kg.setdefault(k, v)
            kg["meta"].setdefault("paper_id", str(doc_id))
            return kg
        except Exception as e:
            self.logger.log("KnowledgeGraphBuildFailed", {"doc_id": doc_id, "error": str(e), "traceback": traceback.format_exc()})
            return _empty()


# ==========================
# 4) CBR Retriever
# ==========================
class CBRRetriever:
    def __init__(self, cbr_service, logger):
        self.cbr = cbr_service
        self.logger = logger

    def retrieve(self, goal_text: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.cbr:
            return []
        try:
            cases = self.cbr.retrieve(goal_text=goal_text, top_k=k) or []
            out = []
            for c in cases:
                out.append({
                    "title": (c.get("goal_text") or "")[:160],
                    "why_it_won": (c.get("scores", {}).get("winner_rationale") or "")[:240],
                    "patch": (c.get("lessons") or "")[:240],
                    "summary": (c.get("best_text") or c.get("summary") or "")[:400],
                })
            return out
        except Exception as e:
            self.logger.log("CBRRetrieveError", {"error": str(e)})
            return []


# ==========================
# 5) Prompt Builder
# ==========================
class PromptBuilder:
    def __init__(self, logger):
        self.logger = logger

    def build(self, *, current_summary: str, claims: List[dict], title: str,
              domain: str, kb_ctx: Dict[str, Any], sents_min: int, sents_max: int,
              case_pack: Optional[List[dict]] = None) -> str:

        claims_text = "\n".join(f"- {c.get('text','').strip()}" for c in (claims or [])[:5] if c.get("text"))
        tmpl_text = ""
        if (kb_ctx or {}).get("templates"):
            bullets = []
            for t in kb_ctx["templates"]:
                bullets.append("- " + " ".join(t.get("outline", [])[:3]))
            tmpl_text = "\n\nTemplates that worked before:\n" + "\n".join(bullets)

        hints_text = ""
        if (kb_ctx or {}).get("hints"):
            hints_text = "\n\nStrategy hints:\n" + "\n".join(f"- {h}" for h in kb_ctx["hints"])

        examples = ""
        if case_pack:
            ex_lines = []
            for ex in case_pack[:3]:
                ex_lines.append(f"- Lesson: {ex.get('patch','')}\n  Why it won: {ex.get('why_it_won','')}")
            if ex_lines:
                examples = "\n\nPrior improvements to emulate:\n" + "\n".join(ex_lines)

        return f"""
You are a verification expert checking this academic paper summary against the paper's key claims.

Title: {title}   (domain: {domain})

Key Claims:
{claims_text}{examples}{tmpl_text}{hints_text}

Current summary:
\"\"\"{current_summary}\"\"\"

Improve the summary by:
1) Ensuring all key claims are accurately represented
2) Citing figures/tables for quantitative claims when warranted
3) Removing unsupported statements 
4) Preserving clarity and neutrality

Constraints:
- Keep to {sents_min}-{sents_max} sentences
- Use ONLY facts present in the paper and allowed context
- Do not invent numbers or facts

Verified summary:
""".strip()


# ==========================
# 6) PACS Refiner
# ==========================
class PACSRefiner:
    def __init__(self, agent: BaseAgent, metrics_calc: SimplePaperBlogAgent, figure_grounding: FigureGrounding, logger):
        self.agent = agent
        self.metrics_calc = metrics_calc
        self.figure_grounding = figure_grounding
        self.logger = logger

    def refine(self, candidate: str, abstract: str, context: Dict[str, Any],
               paper_data: Dict[str, Any], knowledge_tree: Dict[str, Any],
               pacs_weights: Dict[str, float], sents_min: int, sents_max: int,
               kbase_ctx: Dict[str, Any] | None = None,
               return_panel: bool = False) -> str | Tuple[str, Dict[str, Any]]:

        roles = [
            ("skeptic", "remove speculation; flag ungrounded claims"),
            ("editor", f"tighten structure; keep {sents_min}-{sents_max} sentences"),
            ("risk",   "require figure/table citation for any numeric claim"),
        ]
        panel: List[Dict[str, Any]] = []

        for role, brief in roles:
            prompt = f"""Role: {role.title()}. Brief: {brief}
Abstract:
\"\"\"{abstract[:1000]}\"\"\"

Text to review:
\"\"\"{candidate}\"\"\"

Return ONLY the revised paragraph."""
            try:
                out = self.agent.call_llm(prompt, context=context)
                if not out:
                    continue
                text = out.strip()
                m = self.metrics_calc._compute_metrics(text, abstract, "")
                if role == "risk":
                    m["figure_results"] = self._figure_score(text, paper_data, knowledge_tree)
                panel.append({"role": role, "text": text, "metrics": m})
            except Exception as e:
                self.logger.log("PACSRoleError", {"role": role, "error": str(e)})

        if not panel:
            return (candidate, {}) if return_panel else candidate

        # choose best by role-weighted score
        best_text, best_score, best_entry = candidate, -1.0, None
        for entry in panel:
            score = self._role_weighted_score(entry["role"], entry["metrics"], pacs_weights)
            entry["score"] = score
            if score > best_score:
                best_text, best_score, best_entry = entry["text"], score, entry

        detail = {
            "panel": panel,
            "weights_used": dict(pacs_weights or {}),
            "kb_hints": (kbase_ctx or {}).get("hints", []),
            "kb_templates_count": len((kbase_ctx or {}).get("templates", [])),
        }
        return (best_text, detail) if return_panel else best_text

    def _role_weighted_score(self, role: str, m: Dict[str, float], w: Dict[str, float]) -> float:
        skeptic_focus = 0.6 * (1.0 - float(m.get("hallucination_rate", 0.0))) + 0.4 * float(m.get("faithfulness", 0.0))
        editor_focus  = 0.5 * float(m.get("coherence", 0.0)) + 0.5 * float(m.get("structure", 0.0))
        risk_focus    = float((m.get("figure_results", {}) or {}).get("overall_figure_score", 0.0)) if isinstance(m.get("figure_results"), dict) else 0.0
        base          = float(m.get("overall", 0.0))
        alpha         = w.get(role, 0.33)
        role_focus    = skeptic_focus if role == "skeptic" else editor_focus if role == "editor" else risk_focus
        return alpha * (0.5 * base + 0.5 * role_focus)

    def _figure_score(self, text: str, paper_data: Dict[str, Any], knowledge_tree: Dict[str, Any]) -> Dict[str, Any]:
        # quick heuristic (same as your prior)
        quant_claims = []
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
        for sent in sents:
            matches = re.findall(r"(\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b)", sent, flags=re.I)
            if matches:
                quant_claims.append({
                    "claim": sent,
                    "value": matches[0][0],
                    "metric": matches[0][1],
                    "has_citation": any(k in sent.lower() for k in ["figure","fig.","table","as shown","see"]),
                })
        cited = sum(1 for c in quant_claims if c["has_citation"])
        rate = cited / max(1, len(quant_claims))
        return {"total_claims": len(quant_claims), "properly_cited": cited, "citation_rate": rate, "overall_figure_score": rate, "claims": quant_claims}


# ==========================
# 7) Metrics Scorer (base + knowledge + HRM)
# ==========================
class MetricsScorer:
    def __init__(self, metrics_calc: SimplePaperBlogAgent, scoring_service, logger, hrm_weight: float = 0.10):
        self.calc = metrics_calc
        self.scoring = scoring_service
        self.logger = logger
        self.hrm_weight = hrm_weight

    def score(self, summary: str, abstract: str, author_summary: str,
              knowledge_tree: Dict[str, Any], goal_title: Optional[str], context: Optional[Dict[str, Any]],
              verification_threshold: float) -> Dict[str, float]:

        base = self.calc._compute_metrics(summary, abstract, author_summary)
        ver  = self._verify_against_knowledge(summary, knowledge_tree, verification_threshold)

        hrm_score = None
        if self.scoring:
            try:
                scorable = ScorableFactory.from_dict({"text": summary, "goal": goal_title or "", "type": "document"})
                bundle = self.scoring.score("hrm", context=context, scorable=scorable, dimensions=["alignment"])
                res = getattr(bundle, "results", {}).get("alignment")
                if getattr(res, "score", None) is not None:
                    hs = float(res.score)
                    hrm_score = 1.0/(1.0+math.exp(-hs)) if hs < 0 or hs > 1 else hs
            except Exception as e:
                self.logger.log("HRMScoreError", {"error": str(e)})

        overall = 0.8 * base.get("overall", 0.0) + 0.2 * ver
        if hrm_score is not None:
            overall = (1.0 - self.hrm_weight) * overall + self.hrm_weight * hrm_score

        out = dict(base)
        out["knowledge_verification"] = float(ver)
        if hrm_score is not None:
            out["hrm_score"] = float(hrm_score)
        out["overall"] = float(overall)
        return out

    def _verify_against_knowledge(self, summary: str, tree: Dict[str, Any], threshold: float) -> float:
        if not tree:
            return 0.5
        claims = tree.get("claims", []) or []
        covered = sum(1 for c in claims if c.get("text") and self.calc._contains_concept(summary, c["text"]))
        claim_cov = covered / max(1, len(claims))
        rels = tree.get("relationships", []) or []
        strong = [r for r in rels if float(r.get("confidence", 0.0)) >= threshold]
        evidence = len(strong)/max(1, len(rels))
        return 0.7*claim_cov + 0.3*evidence


# ==========================
# 8) Guardrails
# ==========================
class Guardrails:
    def __init__(self, anti_hallucination: AntiHallucination, figure_grounding: FigureGrounding, logger):
        self.anti = anti_hallucination
        self.fig = figure_grounding
        self.logger = logger

    def hallucinations(self, summary: str, abstract: str, author_summary: str, tree: Dict[str, Any]) -> Tuple[bool, List[str]]:
        try:
            ok, issues = self.anti.verify_section(summary, tree, {"abstract": abstract, "summary": author_summary})
            return bool(ok), (issues or [])
        except Exception as e:
            self.logger.log("AntiHallucinationError", {"error": str(e)})
            return True, ["anti_hallucination_failed_soft"]


# ==========================
# 9) Strategy Manager
# ==========================
class StrategyManager:
    def __init__(self, strategy_store, agent_name: str, scope: str, logger):
        self.store = strategy_store
        self.agent_name = agent_name
        self.scope = scope
        self.logger = logger

    def load(self) -> StrategyProfile:
        if self.store:
            return self.store.load(agent_name=self.agent_name, scope=self.scope)
        return StrategyProfile()

    def save(self, profile: StrategyProfile):
        if self.store:
            self.store.save(agent_name=self.agent_name, profile=profile, scope=self.scope)


# ==========================
# 10) Persistence (casebooks + scorables + signals)
# ==========================
class Persistence:
    def __init__(self, memory, logger):
        self.memory = memory
        self.logger = logger

    def save_case_and_scorable(self, *, doc_id: str, paper_title: str, track_b_id: Any,
                               prompt_text: str, raw_llm: str, candidate: str,
                               best_summary: str, best_metrics: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:

        out: Dict[str, Any] = {}
        try:
            # ensure blog casebook (action_type='blog')
            casebook_name = generate_casebook_name("blog", paper_title)
            casebook = self.memory.casebooks.ensure_casebook(name=casebook_name, tag="blog",
                                                             meta={"paper_id": str(doc_id), "title": paper_title})
            case = self.memory.casebooks.add_case(
                casebook_id=casebook.id,
                goal_id=casebook.goal_id,
                prompt_text=prompt_text,
                agent_name=context.get("agent_name") or "KnowledgeInfusedVerifier",
                response_texts=[raw_llm, candidate],
                meta={},
            )
            out["case_id"] = getattr(case, "id", None)

            # dynamic scorable
            safe_meta = sanitize_for_json({
                "paper_id": str(doc_id),
                "title": paper_title,
                "metrics": best_metrics,
                "origin": "track_c_verified",
                "origin_ids": [track_b_id],
            })
            scorable = self.memory.dynamic_scorables.add(
                pipeline_run_id=context.get("pipeline_run_id"),
                scorable_type=TargetType.DYNAMIC,
                source=context.get("agent_name") or "KnowledgeInfusedVerifier",
                text=best_summary,
                meta=safe_meta,
                source_scorable_id=track_b_id,
                source_scorable_type="dynamic",
            )
            out["scorable_id"] = getattr(scorable, "id", None)

            # link scorable to case
            try:
                self.memory.casebooks.add_scorable(
                    case_id=out["case_id"],
                    pipeline_run_id=context.get("pipeline_run_id"),
                    role="text",
                    scorable_id=out["scorable_id"],
                    text=best_summary,
                    scorable_type=TargetType.DYNAMIC,
                    meta={},
                )
            except Exception:
                pass

        except Exception as e:
            self.logger.log("PersistenceError", {"doc_id": doc_id, "error": str(e)})
        return out

    def capture_signal(self, *, paper_id: str, domain: str, strategy: StrategyProfile,
                       metrics: Dict[str, Any], iterations: List[Dict[str, Any]]):
        payload = {
            "paper_id": str(paper_id),
            "domain": domain,
            "strategy_version": int(getattr(strategy, "strategy_version", 0)),
            "verification_threshold": float(getattr(strategy, "verification_threshold", 0.0)),
            "pacs_weights": dict(getattr(strategy, "pacs_weights", {})),
            "final_quality": float(metrics.get("overall", 0.0)),
            "knowledge_verification": float(metrics.get("knowledge_verification", 0.0)),
            "iterations": len(iterations or []),
            "first_iter_score": float((iterations or [{}])[0].get("current_score", 0.0)) if iterations else None,
            "last_iter_score": float((iterations or [{}])[-1].get("best_candidate_score", 0.0)) if iterations else None,
            "ts": time.time(),
        }
        try:
            if hasattr(self.memory, "calibration_events"):
                self.memory.calibration_events.add({
                    "domain": domain or "general",
                    "query": f"{paper_id}:{domain}",
                    "raw_similarity": payload["final_quality"],
                    "is_relevant": bool(payload["final_quality"] >= 0.80),
                    "scorable_id": str(paper_id),
                    "scorable_type": "paper",
                    "entity_type": "summary_verification",
                    "features": {k: payload.get(k) for k in ["final_quality","knowledge_verification","iterations"]},
                })
        except Exception:
            pass
        try:
            if hasattr(self.memory, "casebooks") and hasattr(self.memory.casebooks, "add"):
                self.memory.casebooks.add(
                    casebook_name="verification_signals",
                    case_id=str(paper_id),
                    role="signal",
                    text=json.dumps(payload),
                    meta={"domain": domain, "timestamp": payload["ts"]},
                )
        except Exception:
            pass


# ==========================
# 11) Audit Reporter
# ==========================
class AuditReporter:
    def __init__(self, report_dir: str, logger):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self.logger = logger

    def write(self, *, doc_id: str, title: str, baseline: Dict[str, Any],
              final: Dict[str, Any], iterations: List[Dict[str, Any]],
              images: Dict[str, Optional[str]], strategy_before: Dict[str, Any], strategy_after: Dict[str, Any]) -> str:

        def f(x):
            try:
                return f"{float(x):.3f}"
            except Exception:
                return str(x)

        lines = []
        lines.append(f"# Verification Report — {title or doc_id}\n")
        if images.get("abc_tile"):
            lines.append(f"![ABC tile]({os.path.relpath(images['abc_tile'], self.report_dir)})\n")

        lines.append("## Overview (Baseline → Final)\n")
        rows = [
            ("overall", baseline.get("overall"), final.get("overall")),
            ("knowledge_verification", baseline.get("knowledge_verification"), final.get("knowledge_verification")),
            ("coverage", baseline.get("claim_coverage", baseline.get("coverage")), final.get("claim_coverage", final.get("coverage"))),
            ("faithfulness", baseline.get("faithfulness"), final.get("faithfulness")),
            ("structure", baseline.get("structure"), final.get("structure")),
            ("hallucination_rate (↓)", baseline.get("hallucination_rate"), final.get("hallucination_rate")),
            ("figure_grounding", (baseline.get("figure_results") or {}).get("overall_figure_score") if isinstance(baseline.get("figure_results"), dict) else None,
                                  (final.get("figure_results") or {}).get("overall_figure_score") if isinstance(final.get("figure_results"), dict) else None),
        ]
        lines.append("| metric | baseline | final |\n|---|---:|---:|")
        for k, b, c in rows:
            lines.append(f"| {k} | {f(b)} | {f(c)} |")
        lines.append("")

        if images.get("iter_timeline"):
            lines.append("## Iteration Timeline\n")
            lines.append(f"![Iteration scores]({os.path.relpath(images['iter_timeline'], self.report_dir)})\n")

        if images.get("panel_heatmap"):
            lines.append("## PACS Panel Snapshot\n")
            lines.append(f"![Panel]({os.path.relpath(images['panel_heatmap'], self.report_dir)})\n")

        if images.get("knowledge_progress"):
            lines.append("## Knowledge Progress\n")
            lines.append(f"![Knowledge]({os.path.relpath(images['knowledge_progress'], self.report_dir)})\n")

        lines.append("## Strategy\n")
        lines.append(f"- Before: `{json.dumps(strategy_before)}`")
        lines.append(f"- After:  `{json.dumps(strategy_after)}`\n")

        out_md = os.path.join(self.report_dir, f"{doc_id}.md")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return out_md


# ==========================
# 12) Orchestrator (the new Agent)
# ==========================
class KnowledgeInfusedVerifier(BaseAgent):
    """
    Thin orchestrator that wires all components, runs the loop,
    and emits VPM images for each doc.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = VerifierConfig(
            max_iters=int(cfg.get("max_iters", 5)),
            min_gain=float(cfg.get("min_gain", 0.015)),
            min_overall=float(cfg.get("min_overall", 0.80)),
            target_confidence=float(cfg.get("target_confidence", 0.95)),
            min_figure_score=float(cfg.get("min_figure_score", 0.80)),
            verification_threshold=float(cfg.get("verification_threshold", 0.90)),
            convergence_window=int(cfg.get("convergence_window", 2)),
            knowledge_graph_conf=float(cfg.get("knowledge_graph_conf", 0.70)),
            sents_min=int(cfg.get("min_sents", 4)),
            sents_max=int(cfg.get("max_sents", 20)),
            cbr_cases=int(cfg.get("cbr_cases", 3)),
            hrm_weight=float(cfg.get("hrm_weight", 0.10)),
            use_cbr=bool(cfg.get("use_cbr", True)),
            use_hrm=bool(cfg.get("use_hrm", True)),
            use_zeromodel=bool(cfg.get("use_zeromodel", True)),
            use_descendants_metric=bool(cfg.get("use_descendants_metric", False)),
            strategy_scope=cfg.get("strategy_scope", "track_c"),
            report_dir=str(cfg.get("audit_report_dir", "reports/track_c")),
            vpm_dir=str(cfg.get("vpm_dir", "reports/vpm")),
            enable_audit_report=bool(cfg.get("enable_audit_report", True)),
            model_key_ranker=cfg.get("model_key_ranker", "ranker.sicql.v1"),
            model_key_retriever=cfg.get("model_key_retriever", "retriever.mrq.v1"),
        )

        # services
        self.metrics_calc = SimplePaperBlogAgent(cfg, memory, container, logger)
        self.kg_builder = KnowledgeGraphBuilder(container, logger)
        self.prompt_builder = PromptBuilder(logger)
        self.pacs_refiner = PACSRefiner(agent=self, metrics_calc=self.metrics_calc,
                                        figure_grounding=FigureGrounding(logger), logger=logger)
        self.guardrails = Guardrails(AntiHallucination(logger), FigureGrounding(logger), logger)
        self.scorer = MetricsScorer(self.metrics_calc, container.get("scoring") if self.cfg.use_hrm else None,
                                    logger, hrm_weight=self.cfg.hrm_weight)

        self.cbr = CBRRetriever(container.get("cbr") if self.cfg.use_cbr else None, logger)
        self.strategy_mgr = StrategyManager(container.get("strategy"), self.name, self.cfg.strategy_scope, logger)
        self.strategy = self.strategy_mgr.load()

        self.vpm = VPMEmitter(logger, container.get("zeromodel") if self.cfg.use_zeromodel else None, out_dir=self.cfg.vpm_dir)
        self.persist = Persistence(memory, logger)
        self.reporter = AuditReporter(self.cfg.report_dir, logger)

        self.model_key_ranker = self.cfg.model_key_ranker
        self.model_key_retriever = self.cfg.model_key_retriever

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        documents = context.get("documents", []) or context.get(self.input_key, [])
        chat_corpus = context.get("chat_corpus", [])
        out: Dict[Any, Dict[str, Any]] = {}

        for doc in documents:
            doc_id = doc.get("id") or doc.get("paper_id")
            if not doc_id:
                continue

            # Load Track A/B artifacts (optional: keep your existing getters)
            track_a, track_b = self._load_tracks(doc_id)
            if not track_b:
                self.logger.log("TrackBMissing", {"doc_id": doc_id})
                continue
            a_meta = self._safe_meta(track_a) if track_a else {}
            b_meta = self._safe_meta(track_b)
            title = doc.get("title", "") or a_meta.get("title","")
            abstract = a_meta.get("abstract") or b_meta.get("abstract") or self._fetch_abstract(doc_id)
            author_sum = a_meta.get("arxiv_summary") or b_meta.get("arxiv_summary") or (doc.get("summary","") or "")
            b_text = (getattr(track_b, "text", "") or "").strip()
            baseline = self._extract_summary(b_text) or (b_meta.get("summary") or b_text)
            baseline_metrics = b_meta.get("metrics") or self.scorer.calc._compute_metrics(baseline, abstract, author_sum)

            # Build knowledge graph
            kg = await self.kg_builder.build(
                doc_id=str(doc_id),
                paper_text=(doc.get("text") or ""),
                chat_corpus=chat_corpus,
                trajectories=context.get("conversation_trajectories", []),
                domains=context.get("domains", []),
            )

            # Iterative loop
            best_summary, best_metrics, iterations, panel_detail, prompt_used = await self._loop(
                doc_id=str(doc_id),
                title=title,
                baseline=baseline,
                abstract=abstract,
                author_summary=author_sum,
                knowledge_graph=kg,
                context=context
            )

            # Guardrails
            ok, issues = self.guardrails.hallucinations(best_summary, abstract, author_sum, kg)
            figure = self.pacs_refiner._figure_score(best_summary, doc, kg)
            passes = bool(ok) and (figure.get("overall_figure_score", 0.0) >= self.cfg.min_figure_score)
            best_metrics["figure_results"] = figure
            out_doc = {
                "summary": best_summary,
                "metrics": best_metrics,
                "iterations": iterations,
                "passes_guardrails": passes,
                "knowledge_graph": kg,
            }

            # Persist + VPM images + report
            images = {
                "abc_tile": self.vpm.emit_abc_tile(str(doc_id), a_meta.get("metrics",{}), baseline_metrics, best_metrics),
                "iter_timeline": self.vpm.emit_iteration_timeline(str(doc_id), iterations),
                "panel_heatmap": self.vpm.emit_panel_heatmap(str(doc_id), panel_detail or {}),
                "knowledge_progress": self.vpm.emit_knowledge_progress(str(doc_id), iterations),
            }

            try:
                saved = self.persist.save_case_and_scorable(
                    doc_id=str(doc_id),
                    paper_title=title,
                    track_b_id=getattr(track_b, "id", None),
                    prompt_text=prompt_used,
                    raw_llm=iterations[0].get("raw_llm","") if iterations else "",
                    candidate=iterations[0].get("candidate","") if iterations else "",
                    best_summary=best_summary,
                    best_metrics=best_metrics,
                    context={"pipeline_run_id": context.get("pipeline_run_id"),
                             "agent_name": self.name},
                )
                out_doc.update(saved)
            except Exception:
                pass

            if self.cfg.enable_audit_report:
                try:
                    report_md = self.reporter.write(
                        doc_id=str(doc_id),
                        title=title,
                        baseline=baseline_metrics,
                        final=best_metrics,
                        iterations=iterations,
                        images=images,
                        strategy_before=self.strategy.to_dict(),
                        strategy_after=self.strategy.to_dict(),  # updated below if changed
                    )
                    out_doc["report_md"] = report_md
                except Exception:
                    pass

            # Signals
            self.persist.capture_signal(
                paper_id=str(doc_id),
                domain=self._domain(context),
                strategy=self.strategy,
                metrics=best_metrics,
                iterations=iterations
            )

            out[doc_id] = out_doc

        context.setdefault("summary_v2", {})
        context["summary_v2"].update(out)
        return context

    # ----- core loop -----
    async def _loop(self, *, doc_id: str, title: str, baseline: str, abstract: str,
                    author_summary: str, knowledge_graph: Dict[str, Any], context: Dict[str, Any]) \
                    -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], str]:

        domain = self._domain(context)
        # kbase context (optional)
        kbase = self.container.get("kbase")
        kb_ctx = kbase.context_for_paper(title=title, abstract=abstract, domain=domain) if kbase else {}

        current = baseline
        current_m = self.scorer.score(current, abstract, author_summary, knowledge_graph, title, context, self.cfg.verification_threshold)
        best_s, best_m = current, current_m
        iterations: List[Dict[str, Any]] = []
        panel_detail: Dict[str, Any] = {}
        no_improve = 0

        for i in range(self.cfg.max_iters):
            case_pack = self.cbr.retrieve(goal_text=title, k=self.cfg.cbr_cases) if self.cfg.use_cbr else []
            prompt = PromptBuilder(self.logger).build(
                current_summary=current,
                claims=knowledge_graph.get("claims", []),
                title=title,
                domain=domain,
                kb_ctx=kb_ctx,
                sents_min=self.cfg.sents_min,
                sents_max=self.cfg.sents_max,
                case_pack=case_pack
            )

            raw_llm = self.call_llm(prompt, context=context) or current
            candidate, detail = self.pacs_refiner.refine(
                raw_llm, abstract, context, {"title": title}, knowledge_graph,
                pacs_weights=self.strategy.pacs_weights, sents_min=self.cfg.sents_min, sents_max=self.cfg.sents_max,
                kbase_ctx=kb_ctx, return_panel=True
            )
            panel_detail = detail or {}

            cand_m = self.scorer.score(candidate, abstract, author_summary, knowledge_graph, title, context, self.cfg.verification_threshold)
            gain = cand_m["overall"] - current_m["overall"]

            iterations.append({
                "iteration": i+1,
                "current_score": current_m["overall"],
                "best_candidate_score": cand_m["overall"],
                "gain": gain,
                "claim_coverage": knowledge_graph.get("claim_coverage", 0.0),
                "evidence_strength": knowledge_graph.get("evidence_strength", 0.0),
                "raw_llm": raw_llm,
                "candidate": candidate,
            })

            if cand_m["overall"] >= self.cfg.min_overall and gain >= self.cfg.min_gain:
                current, current_m = candidate, cand_m
                if cand_m["overall"] > best_m["overall"]:
                    best_s, best_m = current, current_m
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if best_m["overall"] >= self.cfg.target_confidence or no_improve >= 2:
                break

        # (Optional) strategy nudge when we truly improved
        if best_m["overall"] >= baseline and (best_m["overall"] - baseline) >= self.cfg.min_gain:
            try:
                new_w = dict(self.strategy.pacs_weights)
                # tiny heuristic
                if float(best_m.get("hallucination_rate", 1.0)) > 0.2:
                    new_w["skeptic"] = min(0.4, new_w.get("skeptic", 0.33) + 0.03)
                self.strategy.update(pacs_weights=new_w, verification_threshold=min(0.99, self.strategy.verification_threshold + 0.01))
                StrategyManager(self.container.get("strategy"), self.name, self.cfg.strategy_scope, self.logger).save(self.strategy)
            except Exception:
                pass

        return best_s, best_m, iterations, panel_detail, prompt

    # ----- misc helpers -----
    def _domain(self, context: Dict[str, Any]) -> str:
        doms = context.get("domains") or []
        if doms and isinstance(doms, list):
            d = doms[0]
            return str((d.get("domain") if isinstance(d, dict) else d) or "unknown")
        return "unknown"

    def _load_tracks(self, doc_id: Any):
        a = b = None
        try:
            a = self.memory.dynamic_scorables.get_latest_by_source_pointer(
                source="paper_summarizer", source_scorable_type="document", source_scorable_id=int(doc_id)
            )
        except Exception:
            pass
        try:
            if a:
                b = self.memory.dynamic_scorables.get_latest_by_source_pointer(
                    source="sharpened_paper_summarizer", source_scorable_type="dynamic", source_scorable_id=int(a.id)
                )
        except Exception:
            pass
        return a, b

    def _safe_meta(self, obj) -> dict:
        meta = getattr(obj, "meta", {}) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return meta

    def _fetch_abstract(self, doc_id) -> str:
        try:
            sections = self.memory.document_sections.get_by_document(doc_id)
            for s in sections:
                sd = s.to_dict()
                if (sd.get("section_name") or "").lower().strip() == "abstract":
                    return sd.get("section_text", "") or ""
        except Exception:
            pass
        return ""

    def _extract_summary(self, text: str) -> str:
        m = re.search(r"(?mi)^##\s*Summary\s*\n(.+?)(?=^##|\Z)", text or "", re.S)
        return m.group(1).strip() if m else (text or "").strip()
``n
