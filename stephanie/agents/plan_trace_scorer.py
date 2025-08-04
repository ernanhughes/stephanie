# stephanie/agents/knowledge/plan_trace_scorer.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.data.plan_trace import PlanTrace, ExecutionStep
from stephanie.scoring.scorable_factory import ScorableFactory
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.scoring.hrm_scorer import HRMScorer
from stephanie.scoring.sicql_scorer import SICQLScorer
from stephanie.scoring.contrastive_ranker_scorer import ContrastiveRankerScorer
from typing import Dict, List, Any, Optional
import time
from tqdm import tqdm
from statistics import mean
from stephanie.utils.trace_utils import load_plan_traces_from_export_dir


class PlanTraceScorerAgent(BaseAgent):
    """
    Scores pipeline execution traces at multiple levels:
    - Individual execution steps (granular reasoning quality)
    - Complete pipeline execution (overall quality)
    - Step relationships and flow patterns
    
    Uses HRM as primary reasoning quality scorer with MARS meta-analysis
    to enable self-tuning of pipeline execution patterns.
    """
    
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", [])
        self.include_mars = cfg.get("include_mars", True)
        
        # Configure which scorers to use
        self.scorer_types = cfg.get("scorer_types", [
            "hrm", "sicql", "contrastive_ranker"
        ])
        
        # Initialize scorers
        self.scorers = self._initialize_scorers()
        
        # Initialize MARS calculator
        dimension_config = cfg.get("dimension_config", {})
        self.mars_calculator = MARSCalculator(dimension_config, self.logger)
        
        # Pattern extraction parameters
        self.high_agreement_threshold = cfg.get("high_agreement_threshold", 0.8)
        self.low_uncertainty_threshold = cfg.get("low_uncertainty_threshold", 0.2)
        self.pattern_min_count = cfg.get("pattern_min_count", 3)
        
        self.export_dir = cfg.get("export_dir", "exports/plan_traces")

        self.logger.log("PlanTraceScorerInitialized", {
            "dimensions": self.dimensions,
            "scorers": self.scorer_types,
            "high_agreement_threshold": self.high_agreement_threshold,
            "low_uncertainty_threshold": self.low_uncertainty_threshold
        })

    def _initialize_scorers(self) -> Dict[str, Any]:
        """Initialize all configured scorers"""
        scorers = {}
        
        if "hrm" in self.scorer_types:
            scorers["hrm"] = HRMScorer(self.cfg.scorer.hrm, memory=self.memory, logger=self.logger)
        if "sicql" in self.scorer_types:
            scorers["sicql"] = SICQLScorer(self.cfg.scorer.sicql, memory=self.memory, logger=self.logger)
        if "contrastive_ranker" in self.scorer_types:
            scorers["contrastive_ranker"] = ContrastiveRankerScorer(
                self.cfg.scorer.contrastive_ranker, memory=self.memory, logger=self.logger
            )
            
        return scorers

    async def run(self, context: dict) -> dict:
        """Score pipeline execution traces with self-tuning capability"""
        start_time = time.time()
        
        # --- 1. Load and Prepare Training Data
        raw_traces_data = context.get("plan_traces", [])
        if not raw_traces_data:
            # If no traces are provided, try loading from export directory
            self.logger.log(
                "EpistemicPlanHRMTrainingNoTraces",
                {
                    "message": "No plan traces found in context['plan_traces']. Attempting to load from export directory.",
                    "export_dir": self.export_dir,
                }, 
            ) 
            raw_traces_data = load_plan_traces_from_export_dir(self.export_dir)

        for raw_trace in raw_traces_data:
            # Convert raw trace data to PlanTrace object
            if isinstance(raw_trace, dict):
                # If raw_trace is a dict, convert it to PlanTrace
                plan_trace = PlanTrace.from_dict(raw_trace)
            elif isinstance(raw_trace, PlanTrace):
                plan_trace = raw_trace
            if not plan_trace.execution_steps:
                self.logger.log("EmptyPlanTrace", {"trace_id": plan_trace.trace_id})
                continue
            
            # Score individual execution steps
            step_results = []
            all_step_bundles = {}  # step_id -> ScoreBundle
            
            # Process steps with progress tracking
            pbar = tqdm(
                plan_trace.execution_steps,
                desc="Scoring Steps",
                disable=not self.cfg.get("progress", True)
            )
            
            for step in pbar:
                # Create scorable for this step
                scorable = ScorableFactory.from_plan_trace(
                    plan_trace, 
                    mode="single_step",
                    step=step
                )
                
                # Score the step
                step_bundle = self._score_scorable(scorable, plan_trace.goal_text)
                all_step_bundles[step.step_id] = step_bundle
                
                # Prepare results for reporting
                step_scores = {
                    dim: {
                        "score": result.score,
                        "rationale": result.rationale,
                        "source": result.source
                    } for dim, result in step_bundle.results.items()
                }
                
                step_results.append({
                    "step_id": step.step_id,
                    "step_order": step.step_order,
                    "step_type": step.step_type,
                    "agent": step.agent_name,
                    "description": step.description,
                    "scores": step_scores
                })
                
                # Update progress bar
                pbar.set_postfix({"steps": f"{len(step_results)}/{len(plan_trace.execution_steps)}"})
            
            # Score the complete pipeline
            full_scorable = ScorableFactory.from_plan_trace(plan_trace, mode="full_trace")
            full_bundle = self._score_scorable(full_scorable, plan_trace.goal_text)
            
            # Create ScoreCorpus for MARS analysis
            corpus = ScoreCorpus(bundles=all_step_bundles)
            
            # Run MARS analysis across all steps
            mars_results = {}
            if self.include_mars:
                mars_results = self.mars_calculator.calculate(corpus)
                
                # Log MARS analysis metrics
                self.logger.log("MARSAnalysisCompleted", {
                    "trace_id": plan_trace.trace_id,
                    "step_count": len(plan_trace.execution_steps),
                    "dimensions": list(mars_results.keys()),
                    "overall_agreement": self.mars_calculator.get_aggregate_score(mars_results)
                })
                
                # Identify high-quality patterns for self-tuning
                self._update_self_tuning_patterns(corpus, mars_results, plan_trace)
            
            # Save results to context
            context["step_scores"] = step_results
            context["pipeline_score"] = {dim: result.score for dim, result in full_bundle.results.items()}
            context["mars_analysis"] = mars_results
            context["scoring_time"] = time.time() - start_time
            context["score_corpus"] = corpus.to_dict()
            
            self.logger.log("PlanTraceScoringComplete", {
                "trace_id": plan_trace.trace_id,
                "step_count": len(plan_trace.execution_steps),
                "dimensions": self.dimensions,
                "scorers": len(self.scorers)
            })
            
            return context

    def _score_scorable(self, scorable, goal_text) -> ScoreBundle:
        """Score a single scorable with all configured scorers"""
        score_results = {}
        
        for scorer_name, scorer in self.scorers.items():
            try:
                # Score with this scorer
                score_bundle = scorer.score(
                    goal={"goal_text": goal_text},
                    scorable=scorable,
                    dimensions=self.dimensions,
                )
                
                # Add results (prefer HRM for reasoning quality)
                for dim, result in score_bundle.results.items():
                    # If HRM is available for reasoning quality, prefer it
                    if dim == "reasoning_quality" and scorer_name == "hrm":
                        score_results[dim] = result
                    # For other dimensions, use the first available scorer
                    elif dim not in score_results:
                        score_results[dim] = result
            
            except Exception as e:
                self.logger.log("ScorerError", {
                    "scorer": scorer_name,
                    "error": str(e)
                })
                continue
        
        return ScoreBundle(results=score_results)

    def _update_self_tuning_patterns(self, corpus: ScoreCorpus, 
                                  mars_results: Dict, 
                                  plan_trace: PlanTrace):
        """Update self-tuning patterns based on high-quality pipeline executions"""
        # Find high-quality steps (high agreement, low uncertainty)
        high_quality_steps = []
        pattern_metrics = {}
        
        for dimension, results in mars_results.items():
            # Get steps with high agreement and low uncertainty
            agreement_threshold = results.get("agreement_score", 0.0) * 0.9
            high_agreement_steps = corpus.get_high_disagreement_scorables(
                dimension, 
                threshold=1.0 - agreement_threshold
            )
            
            # Get steps with low uncertainty
            low_uncertainty_steps = []
            if "uncertainty" in corpus.metrics:
                uncertainty_matrix = corpus.get_metric_matrix(dimension, "uncertainty")
                low_uncertainty_steps = uncertainty_matrix[
                    uncertainty_matrix.mean(axis=1) < self.low_uncertainty_threshold
                ].index.tolist()
            
            # Intersection: steps that are both high agreement AND low uncertainty
            high_quality_for_dim = list(set(high_agreement_steps) & set(low_uncertainty_steps))
            high_quality_steps.extend(high_quality_for_dim)
            
            # Track metrics for pattern extraction
            pattern_metrics[dimension] = {
                "high_agreement_steps": high_agreement_steps,
                "low_uncertainty_steps": low_uncertainty_steps,
                "high_quality_steps": high_quality_for_dim
            }
        
        # Remove duplicates
        high_quality_steps = list(set(high_quality_steps))
        
        if high_quality_steps:
            # Extract patterns from high-quality steps
            patterns = self._extract_patterns(high_quality_steps, corpus, plan_trace)
            
            # Store patterns for future pipeline construction
            self.memory.pipeline_patterns.store_patterns(patterns)
            
            self.logger.log("SelfTuningPatternsUpdated", {
                "pattern_count": len(patterns),
                "step_count": len(high_quality_steps),
                "trace_id": plan_trace.trace_id
            })
            
            # Generate recommendations for immediate improvement
            recommendations = self._generate_immediate_recommendations(
                corpus, mars_results, high_quality_steps
            )
            self.logger.log("SelfTuningRecommendations", {
                "trace_id": plan_trace.trace_id,
                "recommendations": recommendations
            })

    def _extract_patterns(self, step_ids: List[str], 
                         corpus: ScoreCorpus, 
                         plan_trace: PlanTrace) -> List[Dict]:
        """Extract patterns from high-quality steps for self-tuning"""
        patterns = []
        
        # Map step IDs to step objects for quick lookup
        step_map = {step.step_id: step for step in plan_trace.execution_steps}
        
        for step_id in step_ids:
            step = step_map.get(step_id)
            if not step:
                continue
                
            # Extract pattern features
            pattern = {
                "step_type": step.step_type,
                "agent": step.agent_name,
                "input_type": step.input_type,
                "output_type": step.output_type,
                "success_metrics": {}
            }
            
            # Add success metrics from MARS analysis
            for dimension in self.dimensions:
                # Get metric values for this dimension
                uncertainty_values = corpus.get_metric_values(dimension, "hrm", ["uncertainty"])
                if step_id in uncertainty_values["uncertainty"]:
                    pattern["success_metrics"][dimension] = {
                        "uncertainty": uncertainty_values["uncertainty"][step_id],
                        "agreement_score": corpus.get_dimension_matrix(dimension).std().mean()
                    }
            
            # Add contextual information
            pattern["context"] = {
                "previous_step_type": self._get_previous_step_type(step, plan_trace),
                "next_step_type": self._get_next_step_type(step, plan_trace),
                "position_in_pipeline": step.step_order / len(plan_trace.execution_steps)
            }
            
            patterns.append(pattern)
        
        return patterns

    def _get_previous_step_type(self, step: ExecutionStep, plan_trace: PlanTrace) -> Optional[str]:
        """Get the type of the previous step in the pipeline"""
        if step.step_order > 1:
            prev_step = next(
                (s for s in plan_trace.execution_steps if s.step_order == step.step_order - 1), 
                None
            )
            return prev_step.step_type if prev_step else None
        return None

    def _get_next_step_type(self, step: ExecutionStep, plan_trace: PlanTrace) -> Optional[str]:
        """Get the type of the next step in the pipeline"""
        if step.step_order < len(plan_trace.execution_steps):
            next_step = next(
                (s for s in plan_trace.execution_steps if s.step_order == step.step_order + 1), 
                None
            )
            return next_step.step_type if next_step else None
        return None

    def _generate_immediate_recommendations(self, 
                                         corpus: ScoreCorpus, 
                                         mars_results: Dict, 
                                         high_quality_steps: List[str]) -> List[str]:
        """Generate recommendations for immediate pipeline improvement"""
        recommendations = []
        
        # 1. Identify problematic dimensions
        for dimension, results in mars_results.items():
            if results["agreement_score"] < 0.7:
                recommendations.append(
                    f"⚠️ Low agreement in {dimension} scoring. "
                    "Consider reviewing pipeline steps for consistency."
                )
            
            if results["high_disagreement"]:
                primary_conflict = results["primary_conflict"]
                recommendations.append(
                    f"⚠️ Significant conflict between {primary_conflict[0]} and {primary_conflict[1]} "
                    f"in {dimension} scoring (Δ={results['delta']:.3f}). "
                    "This may indicate ambiguous pipeline steps."
                )
        
        # 2. Identify unreliable scorers
        scorer_reliability = {}
        for dimension in self.dimensions:
            reliability = corpus.analyze_scorer_reliability(dimension)
            for scorer, score in reliability.items():
                if scorer not in scorer_reliability:
                    scorer_reliability[scorer] = []
                scorer_reliability[scorer].append(score)
        
        # Average reliability across dimensions
        avg_reliability = {
            scorer: mean(scores) for scorer, scores in scorer_reliability.items()
        }
        
        # Find least reliable scorer
        if avg_reliability:
            least_reliable = min(avg_reliability, key=avg_reliability.get)
            if avg_reliability[least_reliable] < 0.6:
                recommendations.append(
                    f"⚠️ {least_reliable} shows low reliability across dimensions. "
                    "Consider retraining or adjusting its configuration."
                )
        
        # 3. Identify opportunities for improvement
        if high_quality_steps:
            # Find common patterns in high-quality steps
            step_types = [step.step_type for step_id, step in self._get_steps_by_id(high_quality_steps)]
            common_step_type = max(set(step_types), key=step_types.count)
            
            recommendations.append(
                f"💡 High-quality steps frequently use {common_step_type} pattern. "
                "Consider applying this pattern to similar pipeline sections."
            )
        
        return recommendations

    def _get_steps_by_id(self, step_ids: List[str]) -> Dict[str, ExecutionStep]:
        """Get step objects by their IDs"""
        # This would be implemented based on your memory structure
        # For now, return a mock implementation
        return {step_id: ExecutionStep(
            step_id=step_id,
            step_order=0,
            step_type="unknown",
            description="",
            output_text="",
            scores=None
        ) for step_id in step_ids}