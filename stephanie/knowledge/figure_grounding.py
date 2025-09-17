# stephanie/knowledge/figure_grounding.py
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
        
       