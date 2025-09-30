# stephanie/agents/dspy/learning_from_learning.py
from __future__ import annotations
import dspy
import json
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.learning.strategy_manager import StrategyManager
from stephanie.agents.learning.corpus_retriever import CorpusRetriever
from stephanie.agents.learning.scoring import Scoring
from stephanie.agents.learning.summarizer import Summarizer
from stephanie.agents.learning.knowledge_arena import KnowledgeArena
from stephanie.agents.learning.persistence import Persistence
from stephanie.agents.learning.evidence import Evidence
from stephanie.utils.json_sanitize import dumps_safe
import time


class BaselineSummarySignature(dspy.Signature):
    """DSPy signature for generating baseline summaries."""
    title: str = dspy.InputField(desc="Paper title")
    abstract: str = dspy.InputField(desc="Paper abstract")
    section_name: str = dspy.InputField(desc="Section name being summarized")
    section_text: str = dspy.InputField(desc="Text of the section to summarize")
    hints: str = dspy.InputField(desc="Optional hints from previous knowledge")
    
    summary: str = dspy.OutputField(desc="Generated baseline summary")


class ImproveSummarySignature(dspy.Signature):
    """DSPy signature for improving summaries with verification."""
    title: str = dspy.InputField(desc="Paper title")
    section_name: str = dspy.InputField(desc="Section name being summarized")
    section_text: str = dspy.InputField(desc="Text of the section to summarize")
    current_summary: str = dspy.InputField(desc="Current summary to improve")
    weaknesses: str = dspy.InputField(desc="Identified weaknesses in current summary")
    skeptic_weight: float = dspy.InputField(desc="Weight for skeptic component")
    editor_weight: float = dspy.InputField(desc="Weight for editor component")
    risk_weight: float = dspy.InputField(desc="Weight for risk component")
    
    improved_summary: str = dspy.OutputField(desc="Improved summary with knowledge applied")
    knowledge_applied: bool = dspy.OutputField(desc="Whether knowledge was applied")


class StrategyVerificationSignature(dspy.Signature):
    """DSPy signature for verification scoring."""
    summary: str = dspy.InputField(desc="Summary to verify")
    section_text: str = dspy.InputField(desc="Original section text")
    section_name: str = dspy.InputField(desc="Section name")
    
    overall: float = dspy.OutputField(desc="Overall verification score (0.0-1.0)")
    knowledge: float = dspy.OutputField(desc="Knowledge score (0.0-1.0)")
    clarity: float = dspy.OutputField(desc="Clarity score (0.0-1.0)")
    grounding: float = dspy.OutputField(desc="Grounding score (0.0-1.0)")
    weaknesses: str = dspy.OutputField(desc="List of weaknesses as JSON array")


class BaselineSummaryModule(dspy.Module):
    """DSPy module for generating baseline summaries."""
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__()
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        
        # Initialize with DSPy's ChainOfThought for better reasoning
        self.generate = dspy.ChainOfThought(BaselineSummarySignature)
    
    def forward(self, paper: Dict[str, Any], section: Dict[str, Any], 
                critical_msgs: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        # Prepare inputs
        hints = "\n".join((m.get("assistant_text") or m.get("text") or "")
                         for m in (critical_msgs[:6] if critical_msgs else []))
        
        # Call DSPy module
        try:
            result = self.generate(
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                section_name=section.get("section_name", ""),
                section_text=section.get("section_text", "")[:5000],
                hints=hints
            )
            return result.summary
        except Exception as e:
            self.logger.error(f"BaselineSummaryModule failed: {str(e)}")
            # Fallback to original implementation
            return Summarizer(
                self.cfg, self.memory, self.container, self.logger,
                strategy=StrategyManager(self.cfg, self.memory, self.container, self.logger),
                scoring=Scoring(self.cfg, self.memory, self.container, self.logger),
                prompt_loader=self.container.get("prompt_loader"),
                call_llm=self.container.get("call_llm")
            ).baseline(paper, section, critical_msgs, context)


class ImproveSummaryModule(dspy.Module):
    """DSPy module for improving summaries with verification."""
    
    def __init__(self, cfg, memory, container, logger, strategy):
        super().__init__()
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.strategy = strategy
        
        # Initialize with DSPy's ChainOfThought
        self.improve = dspy.ChainOfThought(ImproveSummarySignature)
        self.verify = dspy.ChainOfThought(StrategyVerificationSignature)
    
    def forward(self, paper: Dict[str, Any], section: Dict[str, Any],
                current_summary: str, context: Dict[str, Any],
                return_attribution: bool = False) -> Dict[str, Any]:
        # Improve the summary
        try:
            improvement = self.improve(
                title=paper.get("title", ""),
                section_name=section.get("section_name", ""),
                section_text=section.get("section_text", "")[:6000],
                current_summary=current_summary,
                weaknesses=json.dumps(context.get("weaknesses", [])),
                skeptic_weight=self.strategy.state.skeptic_weight,
                editor_weight=self.strategy.state.editor_weight,
                risk_weight=self.strategy.state.risk_weight
            )
            
            # Verify the improved summary
            verification = self.verify(
                summary=improvement.improved_summary,
                section_text=section.get("section_text", ""),
                section_name=section.get("section_name", "")
            )
            
            # Prepare result
            result = {
                "text": improvement.improved_summary,
                "metrics": {
                    "overall": verification.overall,
                    "knowledge": verification.knowledge,
                    "clarity": verification.clarity,
                    "grounding": verification.grounding,
                    "weaknesses": json.loads(verification.weaknesses) if verification.weaknesses else []
                },
                "knowledge_applied": improvement.knowledge_applied
            }
            
            # Add attribution if requested
            if return_attribution and improvement.knowledge_applied:
                # This would connect to your attribution system
                result["attribution"] = self._get_attribution(
                    improvement.improved_summary, context
                )
                
            return result
            
        except Exception as e:
            self.logger.error(f"ImproveSummaryModule failed: {str(e)}")
            # Fallback to original implementation
            return Summarizer(
                self.cfg, self.memory, self.container, self.logger,
                self.strategy,
                Scoring(self.cfg, self.memory, self.container, self.logger),
                self.container.get("prompt_loader"),
                self.container.get("call_llm")
            ).improve_once(paper, section, current_summary, context, return_attribution)
    
    def _get_attribution(self, summary: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Connect to your attribution system."""
        # This would integrate with your existing attribution logic
        try:
            # Simplified version - would connect to your real attribution system
            claims = self._extract_claim_sentences(summary)
            rpool = (context.get("retrieval_items") or []) + (context.get("arena_initial_pool") or [])
            th = float(self.cfg.get("applied_knowledge", {}).get("attr_sim_threshold", 0.75))
            
            # In a real implementation, you'd use your embedding system here
            # This is a placeholder for demonstration
            return [
                {
                    "claim": claim,
                    "support": {
                        "text": "Relevant supporting text snippet...",
                        "origin": "chat_corpus",
                        "variant": "c123"
                    },
                    "similarity": 0.87
                }
                for claim in claims[:3]  # Just first 3 claims for demo
            ]
        except Exception:
            return []


class VerifyAndImproveModule(dspy.Module):
    """DSPy module that handles the full verification and improvement loop."""
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__()
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        
        # Initialize strategy manager
        self.strategy = StrategyManager(cfg, memory, container, logger)
        
        # Initialize submodules
        self.baseline = BaselineSummaryModule(cfg, memory, container, logger)
        self.improve = ImproveSummaryModule(cfg, memory, container, logger, self.strategy)
    
    def forward(self, baseline: str, paper: Dict[str, Any], 
                section: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        current = baseline
        iterations = []
        max_iter = int(self.cfg.get("max_iterations", 3))
        last_knowledge_applied = False
        first_score_with_knowledge = None
        
        for i in range(1, max_iter + 1):
            # Get verification metrics
            verification = self.improve.verify(
                summary=current,
                section_text=section.get("section_text", ""),
                section_name=section.get("section_name", "")
            )
            
            # Record iteration
            iteration = {
                "iteration": i,
                "score": verification.overall,
                "metrics": {
                    "overall": verification.overall,
                    "knowledge": verification.knowledge,
                    "clarity": verification.clarity,
                    "grounding": verification.grounding
                },
                "knowledge_applied": last_knowledge_applied
            }
            iterations.append(iteration)
            
            # Track first knowledge application
            if last_knowledge_applied and first_score_with_knowledge is None:
                first_score_with_knowledge = verification.overall
            
            # Check if we've met the threshold
            if verification.overall >= self.strategy.state.verification_threshold:
                break
                
            # Improve the summary
            improvement = self.improve(
                paper=paper,
                section=section,
                current_summary=current,
                context={**context, "weaknesses": json.loads(verification.weaknesses)},
                return_attribution=True
            )
            
            current = improvement["text"]
            last_knowledge_applied = improvement.get("knowledge_applied", False)
        
        # Calculate knowledge metrics
        k_applied_iters = sum(1 for it in iterations if it.get("knowledge_applied"))
        k_applied_lift = (verification.overall - first_score_with_knowledge) if first_score_with_knowledge is not None else 0.0
        
        # Update strategy
        self.strategy.evolve(iterations, context)
        
        return {
            "summary": current,
            "metrics": {
                **iterations[-1]["metrics"],
                "knowledge_applied_iters": k_applied_iters,
                "knowledge_applied_lift": k_applied_lift
            },
            "iterations": iterations
        }


class LearningFromLearningDSPyAgent(dspy.Module):
    """True DSPy implementation of the LearningFromLearningAgent."""
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__()
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        
        # Initialize DSPy modules
        self.corpus = CorpusRetriever(cfg, memory, container, logger)
        self.verify_improve = VerifyAndImproveModule(cfg, memory, container, logger)
        self.persist = Persistence(cfg, memory, container, logger)
        self.evidence = Evidence(cfg, memory, container, logger)
        
        # Initialize compiler for optimization
        self._initialize_compiler()
    
    def _initialize_compiler(self):
        """Initialize DSPy compiler for optimization."""
        # Define metric for optimization
        def lm_accuracy_metric(example, pred, trace=None):
            # This would be your actual metric calculation
            # For demonstration, we'll use a simplified version
            try:
                return float(pred.metrics.get("overall", 0.0)) >= 0.85
            except:
                return False
        
        # Create compiler
        self.compiler = dspy.Advise(
            metric=lm_accuracy_metric,
            teacher_settings=dict(max_bootstrapped_demos=3, max_labeled_demos=5),
            student=self.verify_improve,
            max_rounds=3
        )
    
    def optimize(self, trainset: List[Dict[str, Any]]):
        """Optimize the DSPy modules using the compiler."""
        try:
            # Convert trainset to DSPy examples
            dspy_examples = [
                dspy.Example(
                    baseline=ex["baseline"],
                    paper=ex["paper"],
                    section=ex["section"],
                    context=ex["context"],
                    summary=ex["target_summary"],
                    metrics=ex["target_metrics"]
                ).with_inputs("baseline", "paper", "section", "context")
                for ex in trainset
            ]
            
            # Run optimization
            self.verify_improve = self.compiler.compile(
                self.verify_improve,
                trainset=dspy_examples
            )
            
            self.logger.info("DSPy modules optimized successfully")
            return True
        except Exception as e:
            self.logger.error(f"DSPy optimization failed: {str(e)}")
            return False
    
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Declarative implementation using DSPy modules."""
        documents = context.get("documents", [])
        results = []
        t0 = time.time()
        
        for paper in documents:
            casebook, goal, sections = self.persist.prepare_casebook_goal_sections(paper, context)
            
            for section in sections:
                if not self.persist.section_is_large_enough(section):
                    continue
                
                # Retrieve relevant knowledge
                corpus_items = self.corpus.fetch(section["section_text"])
                
                # Build candidates for arena
                candidates = [
                    {"origin": "corpus", "variant": f"c{it['id']}", "text": it["assistant_text"]}
                    for it in corpus_items
                    if (it.get("assistant_text") or "").strip()
                ]
                
                # Generate baseline (using DSPy module)
                baseline = self.verify_improve.baseline(
                    paper=paper,
                    section=section,
                    critical_msgs=corpus_items,
                    context=context
                )
                
                # Verify and improve (using DSPy module)
                verify = self.verify_improve(
                    baseline=baseline,
                    paper=paper,
                    section=section,
                    context=context
                )
                
                # Persist results
                saved_case = self.persist.save_section(
                    casebook=casebook,
                    paper=paper,
                    section=section,
                    verify=verify,
                    baseline=baseline,
                    goal_id=goal["id"],
                    context=context
                )
                
                # Track strategy evolution
                self.verify_improve.strategy.track_section(saved_case, verify["iterations"], context)
                
                # Collect results
                results.append({
                    "section_name": section["section_name"],
                    "summary": verify["summary"],
                    "metrics": verify["metrics"],
                    "iterations": verify["iterations"]
                })
        
        # Generate evidence report
        longitudinal = self.evidence.collect_longitudinal(context=context)
        cross = self.evidence.cross_episode(context=context)
        report_md = self.evidence.report(longitudinal, cross, context=context)
        
        return {
            **context,
            "results": results,
            "strategy": self.verify_improve.strategy.as_dict(),
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "cross_episode": cross,
            "longitudinal_metrics": longitudinal,
            "evidence_report_md": report_md,
        }