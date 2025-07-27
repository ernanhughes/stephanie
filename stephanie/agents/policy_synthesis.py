# stephanie/agents/policy_synthesis.py

import os
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict


from stephanie.agents.base_agent import BaseAgent

class PolicySynthesisAgent(BaseAgent):
    """
    Agent to synthesize multi-layered analysis results from ScoreComparisonAgent
    and ScoreEnergyComparisonAgent. Generates comprehensive policy health reports
    and prepares structured data/signals for GILD-based self-improvement.
    This is Step 5: Policy Synthesis and GILD Signal Preparation.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        
        # Configuration
        self.output_dir = cfg.get("report_output_dir", "logs/policy_synthesis_reports")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Thresholds for identifying issues (can be made configurable)
        self.high_error_threshold = cfg.get("high_error_threshold", 0.5) # Placeholder, e.g., high MAE relative to range
        self.misleading_uncertainty_correlation_threshold = cfg.get("misleading_uncertainty_correlation_threshold", -0.2) # Negative correlation
        self.low_correlation_threshold = cfg.get("low_correlation_threshold", 0.3)

    async def run(self, context: dict) -> dict:
        """
        Main execution logic for the agent.
        """
        try:
            self.logger.log("PolicySynthesisStarted", {})

            # --- 1. Get Input Data from Context ---
            score_comparison_data = context.get('score_comparison_data', [])
            score_analysis_results = context.get('score_analysis_results', {})
            score_energy_analysis_results = context.get('score_energy_analysis_results', {})
            
            if not score_comparison_data or not score_analysis_results:
                self.logger.log("PolicySynthesisWarning", {"message": "Missing core analysis data in context. Skipping synthesis."})
                context['policy_synthesis_results'] = {}
                return context

            # --- 2. Synthesize Insights ---
            # Combine findings from all layers of analysis
            synthesis_report = self._synthesize_policy_insights(
                score_comparison_data, 
                score_analysis_results, 
                score_energy_analysis_results
            )

            # --- 3. Prepare GILD Signals ---
            # Extract and structure data needed for GILD training
            gild_signals = self._prepare_gild_signals(score_comparison_data, score_energy_analysis_results)

            # --- 4. Store Results in Context ---
            synthesis_output = {
                "synthesis_report": synthesis_report,
                "gild_signals_summary": {
                    "total_training_examples": len(gild_signals.get('sicql_advantages', [])),
                    "sources_included": list(gild_signals.get('sources', [])),
                    "dimensions_covered": list(gild_signals.get('dimensions', []))
                },
                "gild_signals": gild_signals # This might be large, consider if storing full data is needed
            }
            context['policy_synthesis_results'] = synthesis_output
            context['policy_synthesis_metadata'] = {
                "synthesis_timestamp": datetime.now().isoformat(),
                "input_data_summary": {
                    "score_comparison_points": len(score_comparison_data),
                    "analysis_results_keys": list(score_analysis_results.keys())[:5], # Sample
                    "energy_analysis_available": bool(score_energy_analysis_results)
                }
            }

            # --- 5. Generate Comprehensive Report ---
            self._generate_policy_synthesis_report(synthesis_report, context['policy_synthesis_metadata'])

            self.logger.log("PolicySynthesisCompleted", {
                "report_sections": len(synthesis_report) if synthesis_report else 0,
                "gild_signals_prepared": synthesis_output['gild_signals_summary']
            })

            return context

        except Exception as e:
            error_msg = f"PolicySynthesisAgent failed: {str(e)}"
            self.logger.log("PolicySynthesisFailed", {"error": str(e), "context_keys": list(context.keys())})
            raise

    def _safe_format_float(self, value, precision: int = 4) -> str:
        """
        Safely formats a float value to a string with specified precision.
        Returns 'N/A' if the value is None.
        
        Args:
            value: The value to format (float, int, or None).
            precision (int): The number of decimal places. Defaults to 4.
            
        Returns:
            str: Formatted string or 'N/A'.
        """
        if value is None:
            return "N/A"
        try:
            # Ensure it's a number before formatting
            numeric_value = float(value)
            return f"{numeric_value:.{precision}f}"
        except (ValueError, TypeError):
            # If conversion fails, return N/A
            return "N/A"

    def _synthesize_policy_insights(self, comparison_data: List[Dict], analysis_results: Dict, energy_results: Dict) -> Dict[str, Any]:
        """
        Combines all analysis layers to create a holistic policy health report.
        """
        report = {
            "executive_summary": {},
            "model_performance_diagnostics": {},
            "internal_state_analysis": {},
            "cross_model_comparison": {},
            "refinement_recommendations": []
        }

        try:
            # --- 1. Executive Summary (based on statistical analysis) ---
            # Highlight top-level performance issues
            report["executive_summary"]["performance_overview"] = {}
            for key, metrics in analysis_results.items():
                source = metrics.get("source")
                dimension = metrics.get("dimension")
                mae = metrics.get("mae")
                rmse = metrics.get("rmse")
                corr = metrics.get("correlation")
                
                if source and dimension:
                    if source not in report["executive_summary"]["performance_overview"]:
                        report["executive_summary"]["performance_overview"][source] = {}
                    
                    is_high_error = mae > 40 if mae is not None else False # Example threshold
                    is_low_correlation = (corr is not None and abs(corr) < self.low_correlation_threshold)
                    
                    report["executive_summary"]["performance_overview"][source][dimension] = {
                        "mae": mae,
                        "rmse": rmse,
                        "correlation_with_llm": corr,
                        "issues": []
                    }
                    if is_high_error:
                        report["executive_summary"]["performance_overview"][source][dimension]["issues"].append("High MAE/RMSE")
                    if is_low_correlation:
                        report["executive_summary"]["performance_overview"][source][dimension]["issues"].append("Low correlation with LLM")

            # --- 2. Model Performance Diagnostics ---
            # Detail issues per model/dimension
            report["model_performance_diagnostics"] = analysis_results # Direct inclusion for detail

            # --- 3. Internal State Analysis (from energy/attribute analysis) ---
            # Summarize findings from ScoreEnergyComparisonAgent
            insights_from_energy = energy_results.get("analysis_insights", [])
            report["internal_state_analysis"]["key_insights"] = insights_from_energy

            # Highlight specific calibration issues
            report["internal_state_analysis"]["calibration_issues"] = []
            for insight in insights_from_energy:
                insight_type = insight.get("type", "")
                source = insight.get("source", "")
                dimension = insight.get("dimension", "")
                corr_value = insight.get("value") # For correlation insights
                p_value = insight.get("p_value")
                
                # Check for negative correlation between uncertainty/energy and error
                # This indicates poor calibration (high confidence = high error)
                if "uncertainty_vs_abs_delta_correlation" in insight_type or "energy_vs_abs_delta_correlation" in insight_type:
                    if corr_value is not None and corr_value < self.misleading_uncertainty_correlation_threshold and p_value is not None and p_value < 0.05:
                        report["internal_state_analysis"]["calibration_issues"].append({
                            "model": source,
                            "dimension": dimension,
                            "issue": f"Poorly calibrated {'uncertainty' if 'uncertainty' in insight_type else 'energy'}",
                            "correlation": corr_value,
                            "p_value": p_value,
                            "description": "Model's confidence metric inversely predicts accuracy."
                        })

            # --- 4. Cross-Model Comparison ---
            # Compare overall performance and characteristics
            # Group stats by dimension for comparison
            stats_by_dimension = defaultdict(lambda: defaultdict(dict))
            for key, metrics in analysis_results.items():
                source = metrics.get("source")
                dimension = metrics.get("dimension")
                if source and dimension:
                    stats_by_dimension[dimension][source] = metrics
            
            comparison_summary = {}
            for dimension, source_stats in stats_by_dimension.items():
                comparison_summary[dimension] = {
                    "models": dict(source_stats),
                    # --- Corrected lines ---
                    # Handle None values explicitly in the key function
                    "best_mae_model": min(
                        source_stats.items(),
                        key=lambda x: x[1].get('mae', float('inf')) if x[1].get('mae') is not None else float('inf'),
                        default=(None, {})
                    )[0],
                    "best_correlation_model": max(
                        source_stats.items(),
                        # --- Key Fix: Check for None explicitly ---
                        key=lambda x: x[1].get('correlation') if x[1].get('correlation') is not None else -float('inf'),
                        default=(None, {})
                    )[0],
                    # --- End of corrected lines ---
                }
            report["cross_model_comparison"] = comparison_summary

            # --- 5. Refinement Recommendations ---
            # Based on synthesis, recommend actions
            recommendations = []
            
            # Check for high error and poor calibration
            for source, dims in report["executive_summary"]["performance_overview"].items():
                for dimension, metrics in dims.items():
                    issues = metrics.get("issues", [])
                    has_high_error = "High MAE/RMSE" in issues
                    # Check if this source/dim has a calibration issue
                    is_poorly_calibrated = any(
                        issue.get("model") == source and issue.get("dimension") == dimension 
                        for issue in report["internal_state_analysis"]["calibration_issues"]
                    )
                    
                    if has_high_error:
                        priority = "High" if is_poorly_calibrated else "Medium"
                        reason = f"{source} shows high error (MAE={self._safe_format_float(metrics.get('mae'))}) on '{dimension}'"
                        if is_poorly_calibrated:
                            reason += " and its confidence metric is poorly calibrated."
                        
                        recommendations.append({
                            "priority": priority,
                            "action": f"Retrain/Refine {source} policy for dimension '{dimension}'",
                            "reason": reason,
                            "suggested_approach": "Use GILD with advantage weighting, potentially filtering examples based on error/confidence."
                        })
            
            # Check for models with good correlation but potentially other issues
            # (This is a placeholder for more nuanced logic)
            # ...

            report["refinement_recommendations"] = recommendations

        except Exception as e:
             self.logger.log("PolicySynthesisInsightGenerationFailed", {"error": str(e)})

        return report

    def _prepare_gild_signals(self, comparison_data: List[Dict], energy_results: Dict) -> Dict[str, Any]:
        """
        Extracts and structures data needed for GILD training.
        Core signal is SICQL advantage (Q-V), weighted potentially by performance/error.
        """
        gild_data = {
            "sicql_advantages": [], # List of dicts with advantage data and context
            "training_contexts": [], # Corresponding contexts (target info, dimension, etc.)
            "performance_weights": [], # Optional: weights based on delta or other metrics
            "sources": set(), # Track which models' data is included
            "dimensions": set() # Track dimensions covered
        }

        try:
            # Get the enriched data map from energy results
            # Assuming it's stored in a way we can access, e.g., as a list
            enriched_data_list = energy_results.get("enriched_data_sample", []) 
            # If it's not a list, we might need to process the map differently
            # Let's assume for now it's a list of enriched data points
            
            # If enriched_data_list is empty, fall back to using comparison_data
            # and fetching attributes on the fly (less efficient)
            data_source = enriched_data_list if enriched_data_list else comparison_data

            for key, data_point  in data_source:
                source = data_point.get('source')
                dimension = data_point.get('dimension')
                target_id = data_point.get('target_id')
                target_type = data_point.get('target_type')
                
                gild_data["sources"].add(source)
                gild_data["dimensions"].add(dimension)

                # --- Focus on SICQL for GILD signals ---
                if source == 'sicql':
                    advantage = data_point.get('advantage')
                    q_value = data_point.get('q_value')
                    v_value = data_point.get('v_value')
                    uncertainty = data_point.get('uncertainty')
                    entropy = data_point.get('entropy')
                    delta = data_point.get('delta') # Error signal
                    
                    # Ensure we have the core components
                    if advantage is not None and q_value is not None and v_value is not None:
                        # Prepare the advantage data point for GILD
                        advantage_record = {
                            "target_id": target_id,
                            "target_type": target_type,
                            "dimension": dimension,
                            "advantage": float(advantage), # The core GILD weighting signal
                            "q_value": float(q_value),
                            "v_value": float(v_value),
                            "uncertainty": float(uncertainty) if uncertainty is not None else None,
                            "entropy": float(entropy) if entropy is not None else None
                        }
                        gild_data["sicql_advantages"].append(advantage_record)
                        
                        # Context for this training example
                        context_record = {
                            "target_id": target_id,
                            "target_type": target_type,
                            "dimension": dimension,
                            # Could include more context if needed (e.g., target metadata)
                        }
                        gild_data["training_contexts"].append(context_record)

                        # Optional: Performance weight (e.g., inverse of error magnitude)
                        # This can be used to focus GILD more on examples where the policy was wrong
                        weight = 1.0
                        if delta is not None:
                            # Example: Higher weight for larger errors (focus on fixing mistakes)
                            # Or lower weight for larger errors (don't overfit to outliers)
                            # Let's use a simple inverse relationship, capped
                            abs_delta = abs(delta)
                            # Avoid division by zero and cap weight
                            weight = min(10.0, 1.0 / (abs_delta + 1e-5)) if abs_delta > 1e-5 else 1.0
                        gild_data["performance_weights"].append(weight)

            # Convert sets to lists for JSON serialization
            gild_data["sources"] = list(gild_data["sources"])
            gild_data["dimensions"] = list(gild_data["dimensions"])

            self.logger.log("GILDSignalsPrepared", {
                "sicql_advantage_points": len(gild_data["sicql_advantages"]),
                "sources": gild_data["sources"],
                "dimensions": gild_data["dimensions"]
            })

        except Exception as e:
             self.logger.log("GILDSignalPreparationFailed", {"error": str(e)})
             # Return partially filled or empty data on error
             gild_data = {k: (list(v) if isinstance(v, set) else v) for k, v in gild_data.items()} # Ensure sets are lists

        return gild_data

    def _generate_policy_synthesis_report(self, synthesis_report: Dict, metadata: Dict):
        """
        Generates a comprehensive markdown report from the synthesis.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            report_filename = f"policy_synthesis_report_{timestamp}.md"
            report_path = os.path.join(self.output_dir, report_filename)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Policy Synthesis & Health Report\n\n")
                f.write(f"**Generated:** {metadata.get('synthesis_timestamp', 'N/A')}\n\n")
                f.write("---\n\n")

                if not synthesis_report:
                    f.write("## Report Generation Failed\n\n")
                    f.write("No synthesis data available to generate report.\n")
                    return

                # --- Executive Summary ---
                f.write("## Executive Summary\n\n")
                perf_overview = synthesis_report.get("executive_summary", {}).get("performance_overview", {})
                if perf_overview:
                    f.write("### Model Performance at a Glance\n\n")
                    for source, dims in perf_overview.items():
                        f.write(f"#### Model: `{source}`\n")
                        for dim, metrics in dims.items():
                            issues = metrics.get("issues", [])
                            f.write(f"- **Dimension `{dim}`**:\n")
                            
                            # Use the helper function for safe formatting
                            mae_str = self._safe_format_float(metrics.get('mae'))
                            rmse_str = self._safe_format_float(metrics.get('rmse'))
                            corr_str = self._safe_format_float(metrics.get('correlation_with_llm'))

                            f.write(f"  - MAE: `{mae_str}`\n")
                            f.write(f"  - RMSE: `{rmse_str}`\n")
                            f.write(f"  - Correlation with LLM: `{corr_str}`\n")
                            if issues:
                                f.write(f"  - **Issues**: {', '.join(issues)}\n")
                        f.write("\n")
                else:
                    f.write("Performance overview unavailable.\n\n")

                # --- Internal State Analysis ---
                f.write("## Internal State Analysis\n\n")
                cal_issues = synthesis_report.get("internal_state_analysis", {}).get("calibration_issues", [])
                if cal_issues:
                    f.write("### Calibration Issues Detected\n\n")
                    for issue in cal_issues:
                        f.write(f"- **{issue.get('model', 'N/A')} ({issue.get('dimension', 'N/A')})**: {issue.get('issue', 'N/A')}\n")
                        
                        # Use the helper function for safe formatting of correlation and p-value
                        corr_str_issue = self._safe_format_float(issue.get('correlation'))
                        p_val_issue = issue.get('p_value')
                        p_str_issue = f"{p_val_issue:.2e}" if p_val_issue is not None else "N/A" # Scientific notation for p-value

                        f.write(f"  - Correlation: `{corr_str_issue}` (p={p_str_issue})\n")
                        f.write(f"  - Description: {issue.get('description', 'N/A')}\n\n")
                
                general_insights = synthesis_report.get("internal_state_analysis", {}).get("key_insights", [])
                if general_insights:
                    f.write("### Other Key Insights\n\n")
                    for insight in general_insights:
                        f.write(f"- **{insight.get('type', 'Insight')}** ({insight.get('source', 'N/A')}/{insight.get('dimension', 'N/A')}):\n")
                        
                        # Handle value formatting based on type if needed, or just use safe_format_float
                        value_to_format = insight.get('value')
                        if isinstance(value_to_format, (int, float)) and not isinstance(value_to_format, bool):
                            value_str = self._safe_format_float(value_to_format)
                        else:
                            value_str = str(value_to_format) if value_to_format is not None else "N/A"

                        metric_str = str(insight.get('metric', 'N/A')) # Metric name is likely a string
                        
                        f.write(f"  - Metric: `{metric_str}`\n")
                        f.write(f"  - Value: `{value_str}`\n")
                        
                        # Handle p-value for general insights
                        p_val_general = insight.get('p_value')
                        if p_val_general is not None:
                            p_str_general = f"{p_val_general:.2e}" if isinstance(p_val_general, (int, float)) and not isinstance(p_val_general, bool) else str(p_val_general)
                            f.write(f"  - P-Value: `{p_str_general}`\n")
                        if 'interpretation' in insight:
                            f.write(f"  - Interpretation: {insight.get('interpretation', 'N/A')}\n")
                        f.write("\n")
                if not cal_issues and not general_insights:
                    f.write("No specific internal state issues or insights identified.\n\n")

                # --- Cross-Model Comparison ---
                f.write("## Cross-Model Comparison\n\n")
                comparisons = synthesis_report.get("cross_model_comparison", {})
                if comparisons:
                    for dimension, data in comparisons.items():
                        f.write(f"### Dimension: `{dimension}`\n")
                        f.write("| Model | MAE | RMSE | Correlation | Best in Category |\n")
                        f.write("| :--- | ---: | ---: | ---: | :---: |\n")
                        models_data = data.get("models", {})
                        best_mae = data.get("best_mae_model")
                        best_corr = data.get("best_correlation_model")
                        for model, metrics in models_data.items():
                            # Use the helper function for safe formatting in the table
                            mae_str = self._safe_format_float(metrics.get('mae'))
                            rmse_str = self._safe_format_float(metrics.get('rmse'))
                            corr_str = self._safe_format_float(metrics.get('correlation'))
                            
                            is_best_mae = "✅" if model == best_mae else ""
                            is_best_corr = "✅" if model == best_corr else ""
                            best_marker = f"{is_best_mae} {is_best_corr}".strip()
                            f.write(f"| {model} | {mae_str} | {rmse_str} | {corr_str} | {best_marker} |\n")
                        f.write("\n")
                else:
                    f.write("Cross-model comparison data unavailable.\n\n")

                # --- Refinement Recommendations ---
                f.write("## Refinement Recommendations\n\n")
                recommendations = synthesis_report.get("refinement_recommendations", [])
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        f.write(f"{i}. **{rec.get('action', 'No action specified')}**\n")
                        f.write(f"   - **Priority**: {rec.get('priority', 'N/A')}\n")
                        f.write(f"   - **Reason**: {rec.get('reason', 'N/A')}\n")
                        if rec.get('suggested_approach'):
                            f.write(f"   - **Suggested Approach**: {rec.get('suggested_approach', 'N/A')}\n")
                        f.write("\n")
                else:
                    f.write("No specific refinement recommendations generated.\n\n")

            self.logger.log("PolicySynthesisReportSaved", {"path": report_path})

        except Exception as e:
            self.logger.log("PolicySynthesisReportGenerationFailed", {"error": str(e)})
