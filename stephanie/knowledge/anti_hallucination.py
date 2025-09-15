# stephanie/verification/anti_hallucination.py
"""
AntiHallucination
-----------------
Hard guardrails against unsupported claims in generated content.
Fails verification for sections with hallucinated content.
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple, Any
import logging

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
    
    def _is_claim_supported(self, claim: str, tree: Dict[str, Any]) -> bool:
        """Check if a claim is supported by the knowledge tree."""
        # Check against paper claims
        for paper_claim in tree.get("claims", []):
            if self._text_similarity(claim, paper_claim["text"]) > 0.6:
                return True
                
        # Check against verified insights
        for insight in tree.get("insights", []):
            if self._text_similarity(claim, insight["text"]) > 0.6:
                return True
                
        # Check for entity support
        entities = tree.get("entities", [])
        claim_entities = self._extract_entities(claim)
        
        # If claim has entities that appear in the knowledge tree
        for entity in entities:
            for claim_entity in claim_entities:
                if entity["text"].lower() in claim_entity.lower():
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