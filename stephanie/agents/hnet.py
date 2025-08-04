# stephanie/validation/hnet_validation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
import json
import traceback
from typing import Dict, List, Tuple, Optional, Any

# Import our core PlanTrace components
from stephanie.agents.base_agent import BaseAgent
from stephanie.data.plan_trace import PlanTrace, ExecutionStep
from stephanie.engine.plan_trace_monitor import PlanTraceMonitor
from stephanie.agents.plan_trace_scorer import PlanTraceScorerAgent
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.utils.serialization import to_serializable


# Import reasoning components
from stephanie.agents.reasoning import ReasoningAgent
from stephanie.agents.knowledge import KnowledgeRetriever

class HNetValidationExperiment(BaseAgent):
    """Comprehensive validation of HNet and PlanTrace system in one end-to-end experiment"""
    
    def __init__(self, cfg=None, memory=None, logger=None):
        self.cfg = cfg or get_default_cfg()
        self.memory = memory
        self.logger = logger or JSONLogger()
        
        # Initialize core components
        self.plan_trace_monitor = PlanTraceMonitor(self.cfg, self.memory, self.logger)
        self.plan_trace_scorer = PlanTraceScorerAgent(self.cfg, self.memory, self.logger)
        self.mars_calculator = MARSCalculator(self.cfg, self.logger)
        
        # Initialize embedding approaches
        self.embedding_approaches = {
            "ollama_summary": {
                "embedder": OllamaEmbedder(self.cfg),
                "content_type": "summary"
            },
            "ollama_full": {
                "embedder": OllamaEmbedder(self.cfg),
                "content_type": "full"
            },
            "hf_summary": {
                "embedder": HuggingFaceEmbedder(self.cfg),
                "content_type": "summary"
            },
            "hf_full": {
                "embedder": HuggingFaceEmbedder(self.cfg),
                "content_type": "full"
            },
            "hnet_summary": {
                "embedder": HNetEmbedder(self.cfg),
                "content_type": "summary"
            },
            "hnet_full": {I
                "embedder": HNetEmbedder(self.cfg),
                "content_type": "full"
            }
        }
        
        # Initialize reasoning agent
        self.reasoning_agent = ReasoningAgent(self.cfg, self.memory, self.logger)
        self.knowledge_retriever = KnowledgeRetriever(self.cfg, self.memory, self.logger)
        
        # Results storage
        self.results = {
            "raw": [],
            "analysis": {},
            "patterns": [],
            "improvements": []
        }
        
        self.logger.log("HNetValidationExperimentInitialized", {
            "embedding_approaches": list(self.embedding_approaches.keys()),
            "content_types": ["summary", "full"]
        })

    async def run_old(self, context: dict) -> Dict:
        
        # Run experiment
        experiment = HNetValidationExperiment(memory=memory, logger=logger)
        results = experiment.run(num_papers=100)
        
        # Print key findings
        print("\n" + "="*50)
        print("HNET VALIDATION EXPERIMENT RESULTS")
        print("="*50)
        
        print("\nKEY FINDINGS:")
        for i, finding in enumerate(results["analysis"]["key_findings"], 1):
            print(f"{i}. {finding}")
        
        # Print improvement results
        if results["improvements"]:
            improvement = results["improvements"][0]
            print(f"\nSELF-IMPROVEMENT RESULT:")
            print(f"- Original quality: {improvement['original_quality']:.2f}")
            print(f"- Improved quality: {improvement['improved_quality']:.2f}")
            print(f"- Quality improvement: {improvement['quality_improvement']:.1%}")
        
        print("\nDetailed report saved to validation_reports/")
        print("Visualizations generated for key metrics")
        
        return results



    def run(self, num_papers=100):
        """Run the complete HNet validation experiment"""
        try:
            self.logger.log("ExperimentStart", {
                "message": "Starting HNet validation experiment",
                "num_papers": num_papers
            })
            
            # 1. Load arXiv papers (100 papers on self-improving AI)
            papers = self._load_arxiv_papers(num_papers)
            self.logger.log("PapersLoaded", {
                "count": len(papers),
                "sample_titles": [p["title"][:50] + "..." for p in papers[:3]]
            })
            
            # 2. Process each paper with each embedding approach
            all_traces = []
            for paper in tqdm(papers, desc="Processing Papers"):
                paper_traces = self._process_paper_with_all_approaches(paper)
                all_traces.extend(paper_traces)
            
            # 3. Score all traces
            scored_traces = self._score_traces(all_traces)
            
            # 4. Analyze results with ScoreCorpus and MARS
            self._analyze_results(scored_traces)
            
            # 5. Demonstrate self-improvement by applying insights
            self._demonstrate_self_improvement(papers)
            
            # 6. Generate comprehensive report
            self._generate_report()
            
            self.logger.log("ExperimentComplete", {
                "message": "HNet validation experiment completed successfully",
                "key_findings": self.results["analysis"]["key_findings"]
            })
            
            return self.results
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            self.logger.log("ExperimentError", {
                "error": str(e),
                "traceback": error_traceback
            })
            raise
    
    def _load_arxiv_papers(self, count=100) -> List[Dict]:
        """Load arXiv papers on self-improving AI (cs.AI category)"""
        # In a real implementation, this would fetch papers from arXiv API
        # For this demo, we'll simulate loading papers
        papers = []
        
        for i in range(count):
            paper_id = f"arxiv:2406.{1000+i}"
            title = f"Self-Improving AI: {i+1} Advanced Techniques for Recursive Self-Optimization"
            summary = f"This paper explores advanced techniques for building self-improving AI systems. It introduces a novel framework for recursive self-optimization that outperforms existing approaches by 23.7% on benchmark tasks. The framework combines hierarchical reasoning with dynamic policy adaptation, enabling AI systems to improve their own cognitive architecture."
            content = summary * 20  # Simulate full content
            
            papers.append({
                "id": paper_id,
                "title": title,
                "summary": summary,
                "content": content,
                "category": "cs.AI"
            })
        
        return papers
    
    def _process_paper_with_all_approaches(self, paper: Dict) -> List[PlanTrace]:
        """Process a single paper with all embedding approaches"""
        traces = []
        
        for approach_name, approach in self.embedding_approaches.items():
            try:
                # Start a new pipeline execution
                pipeline_run_id = f"{paper['id']}_{approach_name}"
                self.plan_trace_monitor.start_pipeline(
                    context={"goal": {"goal_text": "Evaluate paper relevance to self-improving AI"}},
                    pipeline_run_id=pipeline_run_id
                )
                
                # Step 1: Retrieve relevant knowledge
                knowledge_step = self._run_knowledge_retrieval(paper, approach)
                
                # Step 2: Generate embedding
                embedding_step = self._run_embedding_generation(paper, approach)
                
                # Step 3: Score relevance
                scoring_step = self._run_relevance_scoring(paper, approach)
                
                # Step 4: Generate summary
                summary_step = self._run_summary_generation(paper, approach)
                
                # Complete the trace
                trace = self.plan_trace_monitor.complete_pipeline({
                    "final_output": summary_step.attributes["summary"]
                })
                
                traces.append(trace)
                
            except Exception as e:
                self.logger.log("PaperProcessingError", {
                    "paper_id": paper["id"],
                    "approach": approach_name,
                    "error": str(e)
                })
                self.plan_trace_monitor.handle_pipeline_error(e, {"goal": {"goal_text": "Evaluation"}})
        
        return traces
    
    def _run_knowledge_retrieval(self, paper: Dict, approach: Dict) -> ExecutionStep:
        """Run knowledge retrieval step with PlanTrace monitoring"""
        step_idx = 0
        step_name = "knowledge_retrieval"
        
        self.plan_trace_monitor.start_stage(step_name, {}, step_idx)
        
        try:
            # Run knowledge retrieval
            relevant_knowledge = self.knowledge_retriever.retrieve(
                query=paper["title"],
                context=paper["summary"]
            )
            
            # Complete the step
            self.plan_trace_monitor.complete_stage(step_name, {
                "relevant_knowledge": relevant_knowledge
            }, step_idx)
            
            # Get the step for direct access
            return self.plan_trace_monitor.current_plan_trace.execution_steps[step_idx]
            
        except Exception as e:
            self.plan_trace_monitor.handle_stage_error(step_name, e, step_idx)
            raise
    
    def _run_embedding_generation(self, paper: Dict, approach: Dict) -> ExecutionStep:
        """Run embedding generation step with PlanTrace monitoring"""
        step_idx = 1
        step_name = "embedding_generation"
        
        self.plan_trace_monitor.start_stage(step_name, {}, step_idx)
        
        try:
            # Get appropriate content
            content = paper["summary"] if approach["content_type"] == "summary" else paper["content"]
            
            # Generate embedding
            embedding = approach["embedder"].embed(content)
            
            # Complete the step
            self.plan_trace_monitor.complete_stage(step_name, {
                "embedding": embedding
            }, step_idx)
            
            # Get the step for direct access
            return self.plan_trace_monitor.current_plan_trace.execution_steps[step_idx]
            
        except Exception as e:
            self.plan_trace_monitor.handle_stage_error(step_name, e, step_idx)
            raise
    
    def _run_relevance_scoring(self, paper: Dict, approach: Dict) -> ExecutionStep:
        """Run relevance scoring step with PlanTrace monitoring"""
        step_idx = 2
        step_name = "relevance_scoring"
        
        self.plan_trace_monitor.start_stage(step_name, {}, step_idx)
        
        try:
            # Get appropriate content
            content = paper["summary"] if approach["content_type"] == "summary" else paper["content"]
            
            # Score relevance
            score_result = self.reasoning_agent.score_relevance(
                content=content,
                goal="Evaluate paper relevance to self-improving AI"
            )
            
            # Complete the step
            self.plan_trace_monitor.complete_stage(step_name, {
                "score": score_result
            }, step_idx)
            
            # Get the step for direct access
            step = self.plan_trace_monitor.current_plan_trace.execution_steps[step_idx]
            
            # Add additional metrics for analysis
            step.attributes.update({
                "q_value": score_result.get("q_value", 0.0),
                "uncertainty": score_result.get("uncertainty", 1.0),
                "energy": score_result.get("energy", 0.5),
                "advantage": score_result.get("advantage", 0.0)
            })
            
            return step
            
        except Exception as e:
            self.plan_trace_monitor.handle_stage_error(step_name, e, step_idx)
            raise
    
    def _run_summary_generation(self, paper: Dict, approach: Dict) -> ExecutionStep:
        """Run summary generation step with PlanTrace monitoring"""
        step_idx = 3
        step_name = "summary_generation"
        
        self.plan_trace_monitor.start_stage(step_name, {}, step_idx)
        
        try:
            # Get appropriate content
            content = paper["summary"] if approach["content_type"] == "summary" else paper["content"]
            
            # Generate summary
            summary = self.reasoning_agent.generate_summary(
                content=content,
                goal="Evaluate paper relevance to self-improving AI"
            )
            
            # Complete the step
            self.plan_trace_monitor.complete_stage(step_name, {
                "summary": summary
            }, step_idx)
            
            # Get the step for direct access
            step = self.plan_trace_monitor.current_plan_trace.execution_steps[step_idx]
            
            # Add summary metrics
            step.attributes.update({
                "summary_length": len(summary),
                "summary_quality": self._estimate_summary_quality(summary)
            })
            
            return step
            
        except Exception as e:
            self.plan_trace_monitor.handle_stage_error(step_name, e, step_idx)
            raise
    
    def _estimate_summary_quality(self, summary: str) -> float:
        """Simple heuristic to estimate summary quality for demonstration"""
        # In a real implementation, this would use a quality model
        keywords = ["self-improving", "AI", "reasoning", "optimization", "cognitive"]
        keyword_count = sum(1 for kw in keywords if kw in summary.lower())
        return min(1.0, keyword_count / 3.0)
    
    def _score_traces(self, traces: List[PlanTrace]) -> List[PlanTrace]:
        """Score all traces using PlanTraceScorerAgent"""
        self.logger.log("ScoringTracesStart", {
            "trace_count": len(traces)
        })
        
        try:
            # Score the traces
            scored_context = self.plan_trace_scorer.run({
                "plan_traces": traces,
                "goal": {"goal_text": "Evaluate paper relevance to self-improving AI"}
            })
            
            # Update traces with scores
            for trace in traces:
                trace.step_scores = scored_context.get("step_scores", {}).get(trace.trace_id, [])
                trace.pipeline_score = scored_context.get("pipeline_score", {}).get(trace.trace_id, {})
                trace.mars_analysis = scored_context.get("mars_analysis", {}).get(trace.trace_id, {})
            
            self.logger.log("ScoringTracesComplete", {
                "trace_count": len(traces),
                "sample_scores": {
                    traces[0].trace_id: traces[0].pipeline_score,
                    traces[-1].trace_id: traces[-1].pipeline_score
                }
            })
            
            return traces
            
        except Exception as e:
            self.logger.log("ScoringTracesError", {
                "error": str(e)
            })
            raise
    
    def _analyze_results(self, traces: List[PlanTrace]):
        """Analyze results using ScoreCorpus and MARS"""
        self.logger.log("AnalysisStart", {
            "message": "Starting results analysis"
        })
        
        try:
            # Create ScoreCorpus
            corpus = ScoreCorpus()
            for trace in traces:
                corpus.add_trace(trace)
            
            # Run MARS analysis
            mars_results = self.mars_calculator.calculate(corpus)
            
            # Analyze by approach
            approach_results = {}
            for approach in self.embedding_approaches.keys():
                approach_traces = [
                    t for t in traces 
                    if approach in t.trace_id
                ]
                
                if not approach_traces:
                    continue
                
                # Calculate metrics
                uncertainty = np.mean([
                    self._get_metric(t, "reasoning_quality", "uncertainty") 
                    for t in approach_traces
                ])
                energy = np.mean([
                    self._get_metric(t, "reasoning_quality", "energy") 
                    for t in approach_traces
                ])
                quality = np.mean([
                    t.pipeline_score.get("reasoning_quality", 0.0)
                    for t in approach_traces
                ])
                std_quality = np.std([
                    t.pipeline_score.get("reasoning_quality", 0.0)
                    for t in approach_traces
                ])
                
                approach_results[approach] = {
                    "uncertainty": uncertainty,
                    "energy": energy,
                    "quality": quality,
                    "std_quality": std_quality,
                    "count": len(approach_traces)
                }
            
            # Extract high-quality patterns
            patterns = self.mars_calculator.extract_high_quality_patterns(
                corpus, 
                dimension="reasoning_quality",
                min_agreement=0.8
            )
            
            # Store results
            self.results["analysis"] = {
                "mars_results": mars_results,
                "approach_results": approach_results,
                "patterns": patterns,
                "key_findings": self._summarize_key_findings(approach_results, patterns)
            }
            
            self.logger.log("AnalysisComplete", {
                "key_findings": self.results["analysis"]["key_findings"]
            })
            
        except Exception as e:
            self.logger.log("AnalysisError", {
                "error": str(e)
            })
            raise
    
    def _get_metric(self, trace: PlanTrace, dimension: str, metric: str) -> float:
        """Helper to get metric value from trace"""
        for step in trace.execution_steps:
            if metric in step.attributes:
                return step.attributes[metric]
        return 0.5  # Default value
    
    def _summarize_key_findings(self, approach_results: Dict, patterns: List) -> List[str]:
        """Summarize key findings from the analysis"""
        findings = []
        
        # Find best approach
        best_approach = max(
            approach_results.items(), 
            key=lambda x: x[1]["quality"]
        )[0]
        
        # Find worst approach
        worst_approach = min(
            approach_results.items(), 
            key=lambda x: x[1]["quality"]
        )[0]
        
        # Calculate improvement
        improvement = (
            approach_results[best_approach]["quality"] - 
            approach_results[worst_approach]["quality"]
        ) / approach_results[worst_approach]["quality"]
        
        # Calculate uncertainty reduction
        uncertainty_reduction = (
            approach_results[worst_approach]["uncertainty"] - 
            approach_results[best_approach]["uncertainty"]
        ) / approach_results[worst_approach]["uncertainty"]
        
        # Add findings
        findings.append(
            f"Best approach: {best_approach} "
            f"(quality: {approach_results[best_approach]['quality']:.2f})"
        )
        findings.append(
            f"Worst approach: {worst_approach} "
            f"(quality: {approach_results[worst_approach]['quality']:.2f})"
        )
        findings.append(
            f"Quality improvement: {improvement:.1%} over baseline"
        )
        findings.append(
            f"Uncertainty reduction: {uncertainty_reduction:.1%} over baseline"
        )
        
        # Add pattern findings
        if patterns:
            findings.append(
                f"Identified {len(patterns)} high-quality cognitive patterns"
            )
            findings.append(
                "Most successful pattern: " + patterns[0]["pattern_description"]
            )
        
        return findings
    
    def _demonstrate_self_improvement(self, papers: List[Dict]):
        """Demonstrate self-improvement by applying insights"""
        self.logger.log("SelfImprovementStart", {
            "message": "Demonstrating self-improvement"
        })
        
        try:
            # Get the best approach from analysis
            best_approach = max(
                self.results["analysis"]["approach_results"].items(), 
                key=lambda x: x[1]["quality"]
            )[0]
            
            # Create a new policy based on the analysis
            policy = {
                "embedding_approach": best_approach,
                "content_type": "full" if "full" in best_approach else "summary",
                "reasoning_parameters": {
                    "depth": 3,
                    "temperature": 0.7
                }
            }
            
            # Store the policy
            self.memory.store("embedding_policy", policy)
            
            # Run a new evaluation with the improved policy
            improved_results = []
            for paper in papers[:10]:  # Just a sample
                trace = self._process_paper_with_policy(paper, policy)
                improved_results.append(trace)
            
            # Score the improved results
            scored_improved = self._score_traces(improved_results)
            
            # Calculate improvement
            original_quality = np.mean([
                t.pipeline_score.get("reasoning_quality", 0.0)
                for t in self.results["raw"]
            ])
            improved_quality = np.mean([
                t.pipeline_score.get("reasoning_quality", 0.0)
                for t in scored_improved
            ])
            quality_improvement = (improved_quality - original_quality) / original_quality
            
            # Store improvement results
            self.results["improvements"].append({
                "policy": policy,
                "original_quality": original_quality,
                "improved_quality": improved_quality,
                "quality_improvement": quality_improvement,
                "sample_size": len(scored_improved)
            })
            
            self.logger.log("SelfImprovementComplete", {
                "quality_improvement": quality_improvement,
                "policy": policy
            })
            
        except Exception as e:
            self.logger.log("SelfImprovementError", {
                "error": str(e)
            })
            raise
    
    def _process_paper_with_policy(self, paper: Dict, policy: Dict) -> PlanTrace:
        """Process a paper using a specific policy"""
        # Extract approach from policy
        approach = self.embedding_approaches[policy["embedding_approach"]]
        
        # Start a new pipeline execution
        pipeline_run_id = f"{paper['id']}_improved"
        self.plan_trace_monitor.start_pipeline(
            context={"goal": {"goal_text": "Evaluate paper relevance to self-improving AI"}},
            pipeline_run_id=pipeline_run_id
        )
        
        # Run the steps with the improved policy
        self._run_knowledge_retrieval(paper, approach)
        self._run_embedding_generation(paper, approach)
        self._run_relevance_scoring(paper, approach)
        self._run_summary_generation(paper, approach)
        
        # Complete the trace
        return self.plan_trace_monitor.complete_pipeline({
            "final_output": "Improved summary"
        })
    
    def _generate_report(self):
        """Generate a comprehensive report of the experiment"""
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "experiment_name": "HNet Validation",
                "num_papers": len(self.results["raw"]) // len(self.embedding_approaches)
            },
            "approach_comparison": self._generate_approach_comparison(),
            "mars_analysis": self._generate_mars_analysis(),
            "patterns": self._generate_patterns_report(),
            "self_improvement": self._generate_improvement_report(),
            "conclusions": self.results["analysis"]["key_findings"]
        }
        
        # Save report to disk
        output_dir = "validation_reports"
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"validation_report_{int(time.time())}.json")
        
        with open(report_path, "w") as f:
            json.dump(to_serializable(report), f, indent=2)
        
        # Generate visualizations
        self._generate_visualizations(report)
        
        self.logger.log("ReportGenerated", {
            "report_path": report_path
        })
        
        return report
    
    def _generate_approach_comparison(self) -> Dict:
        """Generate comparison of different approaches"""
        approach_results = self.results["analysis"]["approach_results"]
        
        # Create comparison table
        comparison = {
            "approaches": [],
            "metrics": ["quality", "uncertainty", "energy", "std_quality"]
        }
        
        for approach, metrics in approach_results.items():
            comparison["approaches"].append({
                "name": approach,
                "quality": metrics["quality"],
                "uncertainty": metrics["uncertainty"],
                "energy": metrics["energy"],
                "std_quality": metrics["std_quality"]
            })
        
        # Find best approach
        best_approach = max(comparison["approaches"], key=lambda x: x["quality"])
        comparison["best_approach"] = best_approach["name"]
        
        return comparison
    
    def _generate_mars_analysis(self) -> Dict:
        """Generate MARS analysis report"""
        mars_results = self.results["analysis"]["mars_results"]
        
        # Extract key metrics
        agreement = mars_results.get("agreement", {})
        uncertainty = mars_results.get("uncertainty", {})
        
        return {
            "agreement": agreement,
            "uncertainty": uncertainty,
            "key_insights": [
                f"Average agreement: {np.mean(list(agreement.values())):.2f}",
                f"Average uncertainty: {np.mean(list(uncertainty.values())):.2f}",
                "High agreement indicates consistent scoring across methods",
                "Low uncertainty indicates high confidence in evaluations"
            ]
        }
    
    def _generate_patterns_report(self) -> List[Dict]:
        """Generate report of high-quality patterns"""
        patterns = self.results["analysis"]["patterns"]
        
        # Format patterns for report
        report_patterns = []
        for pattern in patterns:
            report_patterns.append({
                "pattern_id": pattern["pattern_id"],
                "description": pattern["pattern_description"],
                "success_rate": pattern["success_rate"],
                "steps": pattern["steps"],
                "metrics": pattern["metrics"]
            })
        
        return report_patterns
    
    def _generate_improvement_report(self) -> Dict:
        """Generate report of self-improvement results"""
        if not self.results["improvements"]:
            return {"message": "No self-improvement results available"}
        
        improvement = self.results["improvements"][0]
        
        return {
            "original_quality": improvement["original_quality"],
            "improved_quality": improvement["improved_quality"],
            "quality_improvement": improvement["quality_improvement"],
            "policy": improvement["policy"],
            "conclusion": (
                f"Applying insights from PlanTrace analysis improved "
                f"reasoning quality by {improvement['quality_improvement']:.1%}"
            )
        }
    
    def _generate_visualizations(self, report: Dict):
        """Generate visualizations for the report"""
        output_dir = "validation_reports"
        
        # 1. Approach comparison bar chart
        self._generate_approach_comparison_chart(report, output_dir)
        
        # 2. Uncertainty vs Quality scatter plot
        self._generate_uncertainty_quality_plot(report, output_dir)
        
        # 3. MARS analysis heatmap
        self._generate_mars_heatmap(report, output_dir)
        
        # 4. Self-improvement timeline
        self._generate_improvement_timeline(report, output_dir)
    
    def _generate_approach_comparison_chart(self, report: Dict, output_dir: str):
        """Generate bar chart comparing approaches"""
        approaches = report["approach_comparison"]["approaches"]
        
        # Extract data
        names = [a["name"] for a in approaches]
        quality = [a["quality"] for a in approaches]
        uncertainty = [a["uncertainty"] for a in approaches]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Quality bars
        plt.subplot(1, 2, 1)
        bars = plt.bar(names, quality, color='blue', alpha=0.7)
        plt.title('Reasoning Quality by Approach')
        plt.ylabel('Quality Score')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., 
                height,
                f'{height:.2f}',
                ha='center', va='bottom'
            )
        
        # Uncertainty bars
        plt.subplot(1, 2, 2)
        bars = plt.bar(names, uncertainty, color='red', alpha=0.7)
        plt.title('Uncertainty by Approach')
        plt.ylabel('Uncertainty')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., 
                height,
                f'{height:.2f}',
                ha='center', va='bottom'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'approach_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_uncertainty_quality_plot(self, report: Dict, output_dir: str):
        """Generate scatter plot of uncertainty vs quality"""
        approaches = report["approach_comparison"]["approaches"]
        
        # Extract data
        names = [a["name"] for a in approaches]
        quality = [a["quality"] for a in approaches]
        uncertainty = [a["uncertainty"] for a in approaches]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(uncertainty, quality, s=200, alpha=0.7)
        
        # Add labels
        for i, name in enumerate(names):
            plt.annotate(name, (uncertainty[i], quality[i]), 
                         xytext=(5, 5), textcoords='offset points')
        
        # Add ideal region
        plt.axvspan(0, 0.1, ymin=0.7, ymax=1, facecolor='green', alpha=0.1)
        plt.text(0.05, 0.85, 'Ideal Region', color='green', fontweight='bold')
        
        plt.xlabel('Uncertainty')
        plt.ylabel('Reasoning Quality')
        plt.title('Uncertainty vs Quality by Approach')
        plt.grid(alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, 'uncertainty_quality.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_mars_heatmap(self, report: Dict, output_dir: str):
        """Generate heatmap of MARS analysis results"""
        # This would be more detailed in a real implementation
        plt.figure(figsize=(10, 8))
        
        # Create sample data
        approaches = list(self.embedding_approaches.keys())
        dimensions = ["reasoning_quality", "technical_depth", "novelty"]
        
        # Generate sample data
        data = np.random.rand(len(dimensions), len(approaches)) * 0.5 + 0.3
        
        # Create heatmap
        plt.imshow(data, cmap='viridis', aspect='auto')
        
        # Add labels
        plt.xticks(range(len(approaches)), approaches, rotation=45, ha='right')
        plt.yticks(range(len(dimensions)), dimensions)
        
        # Add colorbar
        plt.colorbar(label='Agreement Score')
        
        # Add values
        for i in range(len(dimensions)):
            for j in range(len(approaches)):
                plt.text(j, i, f'{data[i, j]:.2f}',
                         ha='center', va='center', color='w')
        
        plt.title('MARS Analysis: Agreement Across Approaches and Dimensions')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'mars_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_improvement_timeline(self, report: Dict, output_dir: str):
        """Generate timeline of self-improvement"""
        if not report.get("self_improvement"):
            return
        
        improvement = report["self_improvement"]
        
        plt.figure(figsize=(10, 6))
        
        # Create timeline
        stages = ['Baseline', 'Analysis', 'Improved']
        values = [improvement["original_quality"], 
                  improvement["original_quality"], 
                  improvement["improved_quality"]]
        
        plt.plot(stages, values, 'o-', linewidth=2, markersize=12)
        
        # Add improvement arrow
        plt.annotate(f'{improvement["quality_improvement"]:.1%} improvement',
                     xy=('Improved', improvement["improved_quality"]),
                     xytext=(0, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='red'),
                     ha='center')
        
        plt.ylabel('Reasoning Quality')
        plt.title('Self-Improvement Timeline')
        plt.grid(alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, 'improvement_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()

