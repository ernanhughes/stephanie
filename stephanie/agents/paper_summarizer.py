# stephanie/agents/paper_summarizer.py
import re
from typing import Dict, Any, Tuple
from stephanie.agents.base_agent import BaseAgent  # your updated BaseAgent(cfg, memory, container, logger)

SENTS_MIN_DEFAULT = 4
SENTS_MAX_DEFAULT = 5

class SimplePaperSummarizerAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.min_sents = int(cfg.get("min_sents", SENTS_MIN_DEFAULT))
        self.max_sents = int(cfg.get("max_sents", SENTS_MAX_DEFAULT))
        self.scoring = container.get("scoring")    # optional (for cosine/rouge helpers) Good job

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        documents = context.get(self.input_key, [])

        # Start profiling process
        self.report({
            "event": "start",
            "step": "DocumentProfiler",
            "details": f"Profiling {len(documents)} documents",
        })

        out = {}
        for doc in documents:
            doc_id = doc["id"]
            title = doc.get("title", "")
            summary = doc.get("summary", "")
            existing_sections = self.memory.document_sections.get_by_document(doc_id)
            abstract = ""
            for sec in existing_sections:
                sec_dict = sec.to_dict()
                if sec_dict.get("section_name") == "abstract":
                    abstract = sec_dict.get("section_text", "")
                    break

            merged_context = {
                "title": title,
                "summary": summary,
                "min_sents": self.min_sents,
                "max_sents": self.max_sents,
                "abstract": abstract,
                **context
            }
            prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)
            response = self.call_llm(prompt, context=context)

            self.report({
                "event": "reasoning_generated",
                "step": "paper_summarizer",
                "prompt": prompt,
                "response": response,
            })

            context[self.output_key] = response
            out[doc_id] = response


        context.setdefault(self.output_key, {})
        context[self.output_key]["summary_v0"] = out
        return context

    # ---------- helpers ----------


    def _validate_output(self, output: str, min_sents: int, max_sents: int) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate that output meets all constraints."""
        # Extract summary section
        summary_match = re.search(r"## Summary\n(.+?)(?=##|$)", output, re.DOTALL)
        if not summary_match:
            return False, "Missing '## Summary' section", {}
        
        summary = summary_match.group(1).strip()
        
        # Check sentence count
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", summary) if len(s.strip()) > 10]
        if not (min_sents <= len(sentences) <= max_sents):
            return False, f"Summary must have {min_sents}-{max_sents} sentences (found {len(sentences)})", {
                "sentence_count": len(sentences)
            }
        
        # Check for hallucination markers
        hallucination_markers = [
            ("not specified", 0.5),  # Should be rare in good summaries
            ("we propose", 0.7),     # First-person language
            ("significantly", 0.8),  # Unsupported adverbs
            ("novel approach", 0.9)  # Marketing language
        ]
        
        hallucination_score = 0.0
        for marker, weight in hallucination_markers:
            if marker in summary.lower():
                hallucination_score += weight
        
        if hallucination_score > 1.0:
            return False, "Summary contains hallucination markers", {
                "hallucination_score": hallucination_score
            }
        
        return True, "Valid output", {
            "sentence_count": len(sentences),
            "hallucination_score": hallucination_score
        }

    def _compute_metrics(self, summary: str, abstract: str, author_summary: str) -> Dict[str, float]:
        """Compute objective metrics for scientific comparison across tracks."""
        # 1. Claim coverage (vs abstract)
        claims = self._extract_key_claims(abstract)
        covered = sum(1 for claim in claims if self._contains_concept(summary, claim))
        claim_coverage = covered / max(1, len(claims))
        
        # 2. Faithfulness (vs abstract + author summary)
        abstract_similarity = self._cosine_similarity(summary, abstract) if abstract else 0.0
        author_similarity = self._cosine_similarity(summary, author_summary) if author_summary else 0.0
        faithfulness = (abstract_similarity * 0.7) + (author_similarity * 0.3)
        
        # 3. Structure compliance (problem → approach → results → implications)
        structure_score = self._evaluate_structure(summary)
        
        # 4. Hallucination detection
        hallucination_issues = self._detect_hallucinations(summary, abstract, author_summary)
        hallucination_rate = len(hallucination_issues) / max(1, len([s for s in re.split(r"(?<=[.!?])", summary) if s.strip()]))
        
        return {
            "claim_coverage": claim_coverage,
            "faithfulness": faithfulness,
            "structure": structure_score,
            "hallucination_rate": hallucination_rate,
            "sentence_count": len([s for s in re.split(r"(?<=[.!?])", summary) if s.strip()]),
            "tokens": len(summary.split()),
            "overall": self._calculate_overall(claim_coverage, faithfulness, structure_score, hallucination_rate)
        }

    def _calculate_overall(self, claim_coverage: float, faithfulness: float, 
                        structure: float, hallucination_rate: float) -> float:
        """Calculate overall score using weights aligned with your rubric."""
        # Coverage (40%): claim_coverage * 0.4
        # Faithfulness (40%): (1 - hallucination_rate) * 0.4
        # Clarity/Structure (20%): structure * 0.2
        return (claim_coverage * 0.4) + ((1 - hallucination_rate) * 0.4) + (structure * 0.2)
    
    def _persist_summary(self, paper: Dict[str, Any], text: str):
        # Write to your scorables/documents store; keep it simple and DB-safe
        try:
            scorable_id = self.memory.scorables.create_document({
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "text": text,
                "summary": text,
                "meta": {"source": "baseline_local_llm"}
            })
            # force embedding creation
            _ = self.memory.embedding.get_or_create(text)
            self.logger.log("BaselineSummaryCreated", {
                "paper_id": paper.get("paper_id"), "scorable_id": scorable_id, "length": len(text)
            })
            return scorable_id
        except Exception as e:
            self.logger.log("BaselinePersistFailed", {"error": str(e)})
            return None

def _emit_training_events(self, paper: Dict[str, Any], summary: str, metrics: Dict[str, float]):
    """Only emit training events when quality meets threshold."""
    # Only use high-quality outputs for training
    if metrics["overall"] < 0.75 or metrics["hallucination_rate"] > 0.1:
        self.logger.log("TrainingEventSkipped", {
            "reason": "low_quality",
            "overall": metrics["overall"],
            "hallucination_rate": metrics["hallucination_rate"]
        })
        return
    
    try:
        # Pointwise event with objective score
        self.memory.training_events.add_pointwise(
            model_key="retriever.mrq.v1",
            dimension="alignment",
            query_text=paper.get("title", "paper"),
            cand_text=summary,
            label=1, 
            weight=metrics["overall"],
            trust=metrics["overall"],
            source="baseline",
            meta={
                "stage": "track1",
                "claim_coverage": metrics["claim_coverage"],
                "faithfulness": metrics["faithfulness"]
            }
        )
        
        # Only create pairwise events if author summary exists
        if paper.get("author_summary"):
            self._emit_pairwise_events(paper, summary, metrics)
            
    except Exception as e:
        self.logger.log("TrainingEventEmitError", {"error": str(e)})

    def _emit_pairwise_events(self, paper: Dict[str, Any], summary: str, metrics: Dict[str, float]):
        """Create pairwise events comparing baseline to author summary."""
        author_summary = paper["author_summary"]
        # Determine preference based on objective metrics
        author_metrics = self._compute_metrics(author_summary, paper["abstract"], author_summary)
        prefer_baseline = metrics["overall"] > author_metrics["overall"]
        
        pos_text = summary if prefer_baseline else author_summary
        neg_text = author_summary if prefer_baseline else summary
        
        self.memory.training_events.add_pairwise(
            model_key="ranker.sicql.v1",
            dimension="alignment",
            query_text=paper.get("title", "paper"),
            pos_text=pos_text,
            neg_text=neg_text,
            weight=0.5,
            trust=0.3,
            source="baseline",
            meta={
                "stage": "track1",
                "baseline_score": metrics["overall"],
                "author_score": author_metrics["overall"],
                "prefer_baseline": prefer_baseline
            }
        )
