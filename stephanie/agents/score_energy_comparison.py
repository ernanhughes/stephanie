# stephanie/agents/score_energy_comparison.py

import os
import json
from datetime import datetime
from typing import List, Dict, Any

from sqlalchemy import text
import sqlalchemy
import numpy as np
from scipy.stats import pearsonr

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.models.evaluation_attribute import EvaluationAttributeORM # Import the new ORM

class ScoreEnergyComparisonAgent(BaseAgent):
    """
    Agent to perform deep analysis on score data by fetching and analyzing
    rich attributes from EvaluationAttributeORM (e.g., SICQL's Q/V/uncertainty,
    EBT's energy). Consumes data from ScoreComparisonAgent.
    This is Step 2 (Deep Analysis): Leveraging detailed model internals.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", [])
        
        # Configuration for sources to analyze (focus on those with rich attributes)
        # Default to SICQL and EBT, as they are primary sources of rich data
        # SVM, MRQ might have some, but SICQL and EBT are richest.
        self.sources_for_deep_analysis = cfg.get("sources_for_deep_analysis", ["sicql", "ebt"])
        
        # Output directory for reports
        self.output_dir = cfg.get("report_output_dir", "logs/deep_analysis_reports")
        os.makedirs(self.output_dir, exist_ok=True)
        self.session = self.memory.session  # Get the database session

    async def run(self, context: dict) -> dict:
        """
        Main execution logic for the agent.
        """
        try:
            self.logger.log("ScoreEnergyComparisonStarted", {
                "sources_for_analysis": self.sources_for_deep_analysis
            })

            # --- 1. Get Input Data from Context ---
            # This agent relies on the output of ScoreComparisonAgent
            score_comparison_data = context.get('score_comparison_data', [])
            score_comparison_metadata = context.get('score_comparison_metadata', {})
            
            if not score_comparison_data:
                self.logger.log("ScoreEnergyComparisonWarning", {"message": "No score_comparison_data found in context. Skipping deep analysis."})
                context['score_energy_analysis_results'] = {}
                context['score_energy_analysis_metadata'] = {"status": "skipped_no_input_data"}
                return context

            # --- 2. Extract Target Info for Deep Analysis ---
            # We need to identify the specific evaluations (by ID) that correspond
            # to the scores we want to analyze deeply.
            # The score_comparison_data contains target_id, target_type, dimension, source, score
            # We need to map this back to the EvaluationORM ID to fetch attributes.
            
            # Let's build a query to get the relevant EvaluationORM IDs and link them
            # to the score data for the sources we are interested in.
            
            # Group comparison data by pipeline_run_id (if available) and source for efficient querying
            # The comparison data should have been linked to pipeline runs.
            pipeline_run_ids_from_comparison = score_comparison_metadata.get('pipeline_run_ids', [])
            
            if not pipeline_run_ids_from_comparison:
                 self.logger.log("ScoreEnergyComparisonWarning", {"message": "No pipeline_run_ids found in comparison metadata. Analysis might be incomplete."})
                 # We can still try to fetch attributes, but it's less efficient.

            # --- 3. Fetch Deep Attribute Data ---
            # We need to get EvaluationAttributeORM records that match the scores
            # analyzed by ScoreComparisonAgent for the specified sources.
            # The most robust way is to join back through EvaluationORM and ScoreORM
            # using target_id, target_type, dimension, and source.
            
            deep_analysis_results = self._fetch_deep_attributes(
                pipeline_run_ids=pipeline_run_ids_from_comparison,
                sources=self.sources_for_deep_analysis,
                # We could pass dimensions, but let's fetch all for the specified sources/runs initially
                # and filter in Python if needed based on score_comparison_data
            )

            # --- 4. Correlate Deep Attributes with Comparison Data ---
            # Link the fetched deep attributes with the score_comparison_data
            # to create a richer dataset for analysis.
            # Key: (target_id, target_type, dimension, source) -> Value: comparison + attribute data
            enriched_data_map = self._enrich_comparison_data(score_comparison_data, deep_analysis_results)

            # --- 5. Perform Deep Analysis ---
            # Analyze the relationship between attributes (e.g., SICQL uncertainty)
            # and the comparison metrics (delta, score variance).
            analysis_insights = self._perform_deep_analysis(enriched_data_map)

            # --- 6. Store Results in Context ---
            context['score_energy_analysis_results'] = {
                "enriched_data_sample": dict(list(enriched_data_map.items())[:3]), # Sample for context
                "full_enriched_data_count": len(enriched_data_map),
                "analysis_insights": analysis_insights
            }
            context['score_energy_analysis_metadata'] = {
                "analysis_timestamp": datetime.now().isoformat(),
                "sources_analyzed": self.sources_for_deep_analysis,
                "pipeline_run_ids": pipeline_run_ids_from_comparison,
                "total_attributes_fetched": len(deep_analysis_results)
            }

            # --- 7. Generate Detailed Report ---
            self._generate_deep_analysis_report(analysis_insights, context['score_energy_analysis_metadata'])

            self.logger.log("ScoreEnergyComparisonCompleted", {
                "total_attributes_processed": len(deep_analysis_results),
                "enriched_data_points": len(enriched_data_map),
                "insights_generated": len(analysis_insights) if analysis_insights else 0
            })

            
            return context

        except Exception as e:
            error_msg = f"ScoreEnergyComparisonAgent failed: {str(e)}"
            self.logger.log("ScoreEnergyComparisonFailed", {"error": str(e), "context_keys": list(context.keys())})
            raise

    def _fetch_deep_attributes(self, pipeline_run_ids: List[int], sources: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches detailed EvaluationAttributeORM data for specified sources and pipeline runs.
        Joins with EvaluationORM and ScoreORM to get context.
        """
        try:
            if not sources:
                return []

            # Base query to fetch attributes with context
            # We join EvaluationAttributeORM with EvaluationORM to get target info and run info
            # We also join with ScoreORM to ensure the attribute corresponds to an actual score record
            # and to get the score value if needed directly from the attribute query.
            query_text = """
            SELECT
                e.id AS evaluation_id,
                e.target_id,
                e.target_type,
                e.pipeline_run_id,
                e.source, -- This should match the 'source' in attributes and score_comparison_data
                s.dimension,
                s.score AS score_from_score_table, -- Score from ScoreORM
                -- EvaluationAttributeORM fields
                ea.raw_score,
                ea.energy,
                ea.q_value,
                ea.v_value,
                ea.advantage,
                ea.pi_value,
                ea.entropy,
                ea.uncertainty,
                ea.td_error,
                ea.expected_return
                -- ea.policy_logits -- Consider if including JSON is efficient here
            FROM evaluation_attributes ea
            JOIN evaluations e ON ea.evaluation_id = e.id
            JOIN scores s ON (
                s.evaluation_id = e.id 
                AND s.dimension = ea.dimension 
            )
            WHERE e.source IN :sources
            """
            
            params = {
                "sources": tuple(sources)
            }

            if pipeline_run_ids:
                query_text += " AND e.pipeline_run_id IN :pipeline_run_ids\n"
                params["pipeline_run_ids"] = tuple(pipeline_run_ids)
            
            if self.dimensions:
                query_text += " AND s.dimension IN :dimensions\n"
                params["dimensions"] = tuple(self.dimensions)

            # Order might help with consistency, though not strictly necessary for processing
            query_text += " ORDER BY e.target_type, e.target_id, s.dimension, e.evaluator_name;"

            result = self.session.execute(text(query_text), params).fetchall()

            attributes_data = [dict(row._mapping) for row in result]
            
            self.logger.log("DeepAttributesFetched", {
                "sources": sources,
                "pipeline_run_ids": pipeline_run_ids,
                "count": len(attributes_data)
            })
            return attributes_data

        except sqlalchemy.exc.SQLAlchemyError as sae:
            self.logger.log("DeepAttributeFetchDatabaseError", {"error": f"SQLAlchemy Error: {str(sae)}"})
            return []
        except Exception as e:
            self.logger.log("DeepAttributeFetchFailed", {"error": str(e)})
            return []

    def _enrich_comparison_data(self, comparison_data: List[Dict], attribute_data: List[Dict]) -> Dict[tuple, Dict[str, Any]]:
        """
        Combines score comparison data with fetched deep attributes.
        Creates a map keyed by (target_id, target_type, dimension, source).
        """
        # Create a lookup map for attributes for fast joining
        # Key: (target_id, target_type, dimension, source) -> Value: attribute dict
        attribute_lookup = {}
        for attr in attribute_data:
            key = (
                attr['target_id'],
                attr['target_type'],
                attr['dimension'],
                attr['source'] # This should match evaluator_name from EvaluationORM
            )
            # There might be multiple attributes per key if joins are not perfect,
            # but we'll take the first one or handle duplicates if necessary.
            # Ideally, the join should be unique.
            if key not in attribute_lookup:
                attribute_lookup[key] = attr
            else:
                # Log a warning if duplicates are found, might indicate a data issue
                # For now, we keep the first one.
                pass 

        # Enrich the comparison data
        enriched_map = {}
        for comp_item in comparison_data:
            # Only enrich data for sources we are analyzing
            if comp_item.get('source') not in self.sources_for_deep_analysis:
                continue

            comp_key = (
                comp_item['target_id'],
                comp_item['target_type'],
                comp_item['dimension'],
                comp_item['source']
            )
            
            enriched_item = comp_item.copy() # Start with comparison data
            if comp_key in attribute_lookup:
                # Merge attribute data
                attr_data = attribute_lookup[comp_key]
                # Add all attribute fields, potentially overwriting if names clash
                # (though unlikely as comparison data keys are different)
                enriched_item.update(attr_data)
            else:
                 # Log if attribute data is missing for a comparison item
                 # This is expected for sources not in sources_for_deep_analysis
                 # or if the attribute wasn't saved for some reason.
                 pass

            key_str = f"{enriched_item['target_id']}|{enriched_item['target_type']}|{enriched_item['dimension']}|{enriched_item['source']}" # Use a separator unlikely to be in your data
            enriched_map[key_str] = enriched_item

        self.logger.log("DataEnrichmentCompleted", {
            "comparison_items": len(comparison_data),
            "attribute_items": len(attribute_data),
            "enriched_items": len(enriched_map)
        })
        return enriched_map

    def _perform_deep_analysis(self, enriched_data_map: Dict[tuple, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyzes the enriched data to find relationships and insights.
        E.g., correlation between SICQL uncertainty and delta.
        """
        insights = []
        try:
            if not enriched_data_map:
                return insights

            # Group data by source and dimension for analysis
            from collections import defaultdict
            grouped_data = defaultdict(list)
            for item in enriched_data_map.values():
                key = (item.get('source'), item.get('dimension'))
                if key[0] and key[1]: # Ensure source and dimension are present
                    grouped_data[key].append(item)

            # --- Analysis per Source/Dimension ---
            for (source, dimension), items in grouped_data.items():
                if not items:
                    continue

                # --- SICQL Specific Analysis ---
                if source == 'sicql':
                    # 1. Uncertainty vs Delta (Error)
                    uncertainties = np.array([item['uncertainty'] for item in items if item.get('uncertainty') is not None])
                    deltas = np.array([item['delta'] for item in items if item.get('delta') is not None])
                    
                    # Filter to common indices where both values exist
                    common_indices = np.intersect1d(
                        np.where(~np.isnan(uncertainties))[0],
                        np.where(~np.isnan(deltas))[0]
                    )
                    if len(common_indices) > 1:
                        filtered_uncertainties = uncertainties[common_indices]
                        filtered_deltas = deltas[common_indices]
                        
                        if np.std(filtered_uncertainties) > 1e-10 and np.std(filtered_deltas) > 1e-10:
                            try:
                                corr_result = pearsonr(filtered_uncertainties, np.abs(filtered_deltas)) # Correlate with abs error
                                corr_coef = corr_result.statistic if hasattr(corr_result, 'statistic') else corr_result[0]
                                corr_p_value = corr_result.pvalue if hasattr(corr_result, 'pvalue') else corr_result[1]
                                
                                insights.append({
                                    "type": "sicql_uncertainty_vs_abs_delta_correlation",
                                    "source": source,
                                    "dimension": dimension,
                                    "description": "Correlation between SICQL uncertainty (|Q-V|) and absolute error (|delta|).",
                                    "metric": "Pearson Correlation Coefficient",
                                    "value": float(corr_coef),
                                    "p_value": float(corr_p_value),
                                    "sample_size": len(common_indices),
                                    "interpretation": "Positive correlation suggests high uncertainty predicts high error."
                                })
                            except Exception as e:
                                self.logger.log("SICQLUncertaintyCorrelationFailed", {"error": str(e), "source": source, "dimension": dimension})

                    # 2. Advantage (Q-V) Analysis
                    advantages = np.array([item['advantage'] for item in items if item.get('advantage') is not None])
                    if len(advantages) > 0 and not np.isnan(advantages).all():
                        mean_advantage = np.nanmean(advantages)
                        std_advantage = np.nanstd(advantages)
                        insights.append({
                            "type": "sicql_advantage_stats",
                            "source": source,
                            "dimension": dimension,
                            "description": "Mean and standard deviation of SICQL advantage (Q-V).",
                            "metric": "Mean Advantage",
                            "value": float(mean_advantage),
                            "std_dev": float(std_advantage),
                            "sample_size": len(advantages) - np.count_nonzero(np.isnan(advantages))
                        })

                    # 3. Entropy Analysis
                    entropies = np.array([item['entropy'] for item in items if item.get('entropy') is not None])
                    if len(entropies) > 0 and not np.isnan(entropies).all():
                        mean_entropy = np.nanmean(entropies)
                        insights.append({
                            "type": "sicql_entropy_stats",
                            "source": source,
                            "dimension": dimension,
                            "description": "Mean entropy of SICQL policy distribution.",
                            "metric": "Mean Entropy",
                            "value": float(mean_entropy),
                            "sample_size": len(entropies) - np.count_nonzero(np.isnan(entropies))
                        })

                # --- EBT Specific Analysis ---
                elif source == 'ebt':
                    # 1. Energy vs Delta (Error)
                    energies = np.array([item['energy'] for item in items if item.get('energy') is not None])
                    deltas = np.array([item['delta'] for item in items if item.get('delta') is not None])
                    
                    common_indices_ebt = np.intersect1d(
                        np.where(~np.isnan(energies))[0],
                        np.where(~np.isnan(deltas))[0]
                    )
                    if len(common_indices_ebt) > 1:
                        filtered_energies = energies[common_indices_ebt]
                        filtered_deltas = deltas[common_indices_ebt]
                        
                        if np.std(filtered_energies) > 1e-10 and np.std(filtered_deltas) > 1e-10:
                            try:
                                corr_result_ebt = pearsonr(filtered_energies, np.abs(filtered_deltas))
                                corr_coef_ebt = corr_result_ebt.statistic if hasattr(corr_result_ebt, 'statistic') else corr_result_ebt[0]
                                corr_p_value_ebt = corr_result_ebt.pvalue if hasattr(corr_result_ebt, 'pvalue') else corr_result_ebt[1]
                                
                                insights.append({
                                    "type": "ebt_energy_vs_abs_delta_correlation",
                                    "source": source,
                                    "dimension": dimension,
                                    "description": "Correlation between EBT energy and absolute error (|delta|).",
                                    "metric": "Pearson Correlation Coefficient",
                                    "value": float(corr_coef_ebt),
                                    "p_value": float(corr_p_value_ebt),
                                    "sample_size": len(common_indices_ebt),
                                    "interpretation": "Positive correlation suggests high energy predicts high error (less confidence)."
                                })
                            except Exception as e:
                                self.logger.log("EBTEnergyCorrelationFailed", {"error": str(e), "source": source, "dimension": dimension})

                # --- General Analysis for any source ---
                # Correlation between model's own score and LLM score (redundant check, but using attribute data)
                # This might be slightly different if raw_score in attributes differs from score in comparison data
                model_scores_attr = np.array([item['raw_score'] for item in items if item.get('raw_score') is not None])
                llm_scores_attr = np.array([item['llm_score'] for item in items if item.get('llm_score') is not None])
                
                common_indices_general = np.intersect1d(
                    np.where(~np.isnan(model_scores_attr))[0],
                    np.where(~np.isnan(llm_scores_attr))[0]
                )
                if len(common_indices_general) > 1:
                    filtered_model_scores = model_scores_attr[common_indices_general]
                    filtered_llm_scores = llm_scores_attr[common_indices_general]
                    
                    if np.std(filtered_model_scores) > 1e-10 and np.std(filtered_llm_scores) > 1e-10:
                        try:
                            corr_result_gen = pearsonr(filtered_model_scores, filtered_llm_scores)
                            corr_coef_gen = corr_result_gen.statistic if hasattr(corr_result_gen, 'statistic') else corr_result_gen[0]
                            corr_p_value_gen = corr_result_gen.pvalue if hasattr(corr_result_gen, 'pvalue') else corr_result_gen[1]
                            
                            insights.append({
                                "type": "model_vs_llm_score_correlation",
                                "source": source,
                                "dimension": dimension,
                                "description": "Correlation between model's raw score (from attributes) and LLM score.",
                                "metric": "Pearson Correlation Coefficient",
                                "value": float(corr_coef_gen),
                                "p_value": float(corr_p_value_gen),
                                "sample_size": len(common_indices_general)
                            })
                        except Exception as e:
                            self.logger.log("GeneralScoreCorrelationFailed", {"error": str(e), "source": source, "dimension": dimension})


            self.logger.log("DeepAnalysisCompleted", {"insights_generated": len(insights)})
            return insights

        except Exception as e:
            self.logger.log("DeepAnalysisFailed", {"error": str(e)})
            return insights # Return any insights generated before the error

    def _generate_deep_analysis_report(self, analysis_insights: List[Dict], metadata: Dict):
        """
        Generates a detailed markdown report summarizing the deep analysis insights.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pipeline_ids_str = "_".join(map(str, metadata.get('pipeline_run_ids', ['unknown'])))
            report_filename = f"score_energy_deep_analysis_{pipeline_ids_str}_{timestamp}.md"
            report_path = os.path.join(self.output_dir, report_filename)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# Deep Score Analysis Report (Attributes)\n\n")
                f.write(f"**Generated:** {metadata.get('analysis_timestamp', 'N/A')}\n\n")
                f.write(f"**Pipeline Runs Analyzed:** {metadata.get('pipeline_run_ids', 'N/A')}\n\n")
                f.write(f"**Sources Analyzed:** {', '.join(metadata.get('sources_analyzed', []))}\n\n")
                f.write("---\n\n")

                if not analysis_insights:
                    f.write("## No Insights Generated\n\n")
                    f.write("No significant relationships or statistics were found in the deep attribute analysis.\n")
                else:
                    f.write("## Key Insights from Model Attributes\n\n")
                    for insight in analysis_insights:
                        f.write(f"### {insight.get('type', 'Unnamed Insight').replace('_', ' ').title()}\n")
                        f.write(f"- **Source:** `{insight.get('source', 'N/A')}`\n")
                        f.write(f"- **Dimension:** `{insight.get('dimension', 'N/A')}`\n")
                        f.write(f"- **Description:** {insight.get('description', 'N/A')}\n")
                        f.write(f"- **Metric:** `{insight.get('metric', 'N/A')}`\n")
                        f.write(f"- **Value:** `{insight.get('value', 'N/A')}`\n")
                        if 'p_value' in insight:
                            f.write(f"- **P-Value:** `{insight.get('p_value', 'N/A')}`\n")
                        if 'std_dev' in insight:
                            f.write(f"- **Std Dev:** `{insight.get('std_dev', 'N/A')}`\n")
                        if 'sample_size' in insight:
                            f.write(f"- **Sample Size:** `{insight.get('sample_size', 'N/A')}`\n")
                        if 'interpretation' in insight:
                            f.write(f"- **Interpretation:** {insight.get('interpretation', 'N/A')}\n")
                        f.write("\n---\n\n")
            
            self.logger.log("DeepAnalysisReportSaved", {"path": report_path})

        except Exception as e:
            self.logger.log("DeepAnalysisReportGenerationFailed", {"error": str(e)})


# --- Explanation of the Agent's Structure ---

# 1.  __init__: Sets up configuration, especially `sources_for_deep_analysis` (defaulting to `sicql` and `ebt`)
#     and the output directory. Gets the database session.

# 2.  run: The main orchestration method.
#     - Retrieves `score_comparison_data` and `score_comparison_metadata` from the `context`.
#     - Calls `_fetch_deep_attributes` to get `EvaluationAttributeORM` data for the relevant
#       pipeline runs and sources.
#     - Calls `_enrich_comparison_data` to merge the attribute data with the comparison data
#       based on `target_id`, `target_type`, `dimension`, and `source`.
#     - Calls `_perform_deep_analysis` on the enriched data to calculate correlations and statistics.
#     - Stores the results and metadata in the `context` under `score_energy_analysis_results`
#       and `score_energy_analysis_metadata`.
#     - Calls `_generate_deep_analysis_report` to create a markdown report.

# 3.  _fetch_deep_attributes: Executes a SQL query that joins `evaluation_attributes`, `evaluations`,
#     and `scores` to fetch the rich attribute data for the specified sources and pipeline runs.
#     It selects key fields like `energy`, `q_value`, `v_value`, `uncertainty`, `entropy`, `advantage`.

# 4.  _enrich_comparison_data: Takes the flat list of comparison data and the flat list of attribute data.
#     It creates a lookup map for attributes and then iterates through the comparison data,
#     finding the matching attribute record (if it exists) and merging the two dictionaries.
#     The result is a map keyed by `(target_id, target_type, dimension, source)` for easy access.

# 5.  _perform_deep_analysis: This is the core analysis logic.
#     - It groups the enriched data by `source` and `dimension`.
#     - For `sicql`, it calculates:
#         - Correlation between `uncertainty` (|Q-V|) and `abs(delta)` (absolute error).
#           This directly tests if SICQL's internal uncertainty estimate is a good predictor of its actual error.
#         - Mean and standard deviation of `advantage` (Q-V).
#         - Mean `entropy` of the policy distribution.
#     - For `ebt`, it calculates:
#         - Correlation between `energy` and `abs(delta)`. High energy often means low confidence,
#           so a positive correlation would mean high energy (low confidence) predicts high error.
#     - For all sources, it recalculates the correlation between the model's score (from attributes)
#       and the LLM score, as a sanity check or alternative view.
#     - It returns a list of structured insight dictionaries.

# 6.  _generate_deep_analysis_report: Creates a detailed markdown report summarizing all the
#     insights generated by `_perform_deep_analysis`, making the findings easily readable.
