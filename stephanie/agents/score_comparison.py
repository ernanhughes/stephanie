# stephanie/agents/score_comparison.py
import csv
import os
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy import text
import sqlalchemy
from stephanie.models import EvaluationORM, ScoreORM  # Assuming these are the ORM models used

import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict

from stephanie.agents.base_agent import BaseAgent
# If ScoringStore methods aren't directly usable, we might need to adapt them or use session directly

class ScoreComparisonAgent(BaseAgent):
    """
    Agent to aggregate and compare scores from multiple sources (SICQL, MRQ, SVM, EBT, LLM)
    across specified pipeline runs. Handles asynchronous LLM scoring by fetching latest
    LLM scores for targets evaluated by pipeline-run-linked scorers.
    This is Step 1: Comprehensive Score Aggregation and Comparison.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", [])  # Default dimensions, can be overridden in config
        
        # Configuration for sources to compare
        # Default to common scorers. Can be overridden in config.
        self.sources_to_compare = cfg.get("sources_to_compare", ["sicql", "mrq", "svm", "ebt"])
        self.ground_truth_source = cfg.get("ground_truth_source", "llm") # Typically "llm"
        
        # Ensure ground truth is included if not already in the list
        if self.ground_truth_source not in self.sources_to_compare:
             self.sources_to_compare.append(self.ground_truth_source)

        # Output directory for reports (optional)
        self.output_dir = cfg.get("report_output_dir", "logs/comparison_reports")
        os.makedirs(self.output_dir, exist_ok=True)

        # Get session from memory
        if memory and hasattr(memory, 'session'):
            self.session = memory.session
        else:
            raise ValueError("ScoreComparisonAgent requires a memory object with a session attribute.")

        # Initialize ScoringStore if it's the preferred way to interact
        # self.scoring_store = ScoringStore(self.session, logger) # Optional, if methods are adapted

    async def run(self, context: dict) -> dict:
        """
        Main execution logic for the agent.
        """
        try:
            # --- 1. Get Input Parameters ---
            pipeline_run_ids = context.get("pipeline_run_ids", [4148])
            # Fallback to single ID if list isn't provided
            single_pipeline_run_id = context.get("pipeline_run_id")
            if single_pipeline_run_id and not pipeline_run_ids:
                 pipeline_run_ids = [single_pipeline_run_id]
            
            if not pipeline_run_ids:
                self.logger.log("ScoreComparisonWarning", {"message": "No pipeline_run_id(s) provided. Analysis might be limited or empty."})
                # Decide if we should proceed or raise an error
                # For now, let's proceed but log it.

            dimensions = context.get("dimensions", self.dimensions) # Get from context or config
            # If still empty, we might fetch all available dimensions or use a default set
            # Let's assume ScoringStore/load_gild_examples handles this by fetching all if none specified implicitly

            self.logger.log("ScoreComparisonStarted", {
                "pipeline_run_ids": pipeline_run_ids,
                "dimensions": self.dimensions,
                "sources": self.sources_to_compare,
                "ground_truth": self.ground_truth_source
            })

            # --- 2. Fetch Scores from Pipeline-Linked Sources ---
            # We need scores linked to specific pipeline runs for SICQL, MRQ, SVM, EBT
            # We'll adapt the logic from PolicyAnalyzer._get_sicql_data/_get_mrq_data etc.
            # Or, if we modify ScoringStore, we could use a new method like:
            # local_scores_data = self.scoring_store.get_scores_for_pipeline_runs(
            #     pipeline_run_ids=pipeline_run_ids, 
            #     sources=[s for s in self.sources_to_compare if s != self.ground_truth_source],
            #     dimensions=dimensions
            # )
            
            # For now, let's implement the fetching logic directly using session
            # similar to PolicyAnalyzer methods.
            local_scores_data = self._fetch_local_scores(pipeline_run_ids, self.dimensions)

            # --- 3. Identify Targets for Ground Truth Lookup ---
            # Extract unique (target_id, dimension) combinations from local scores
            # Assuming target_type is consistent or handled, or we fetch it too.
            target_info_set = set()
            for score_record in local_scores_data:
                 # Adjust key names based on actual data structure from _fetch_local_scores
                 target_info_set.add((score_record.get('target_id'), score_record.get('dimension')))

            target_info_list = [{"target_id": tid, "dimension": dim} for tid, dim in target_info_set if tid is not None and dim is not None]

            self.logger.log("ScoreComparisonTargetsIdentified", {
                "target_count": len(target_info_list),
                "sample_targets": list(target_info_list)[:5] # Log first 5 for sanity check
            })

            # --- 4. Fetch Ground Truth (LLM) Scores ---
            # Fetch latest LLM scores for the identified targets, regardless of pipeline run
            # Adapted from PolicyAnalyzer._get_llm_data logic
            llm_scores_data = self._fetch_latest_ground_truth_scores(target_info_list, self.dimensions)

            # --- 5. Merge and Calculate Deltas ---
            # Create a lookup for LLM scores: {(target_id, dimension): score}
            llm_score_lookup = {(item['target_id'], item['dimension']): item['score'] for item in llm_scores_data}

            # Augment local scores with LLM score and delta
            aggregated_results = []
            for local_score in local_scores_data:
                target_id = local_score.get('target_id')
                dimension = local_score.get('dimension')
                source = local_score.get('source')
                local_score_value = local_score.get('score')

                llm_score_for_target = llm_score_lookup.get((target_id, dimension))
                delta = None
                if local_score_value is not None and llm_score_for_target is not None:
                     delta = local_score_value - llm_score_for_target

                # Add LLM score and delta to the local score record
                augmented_record = local_score.copy()
                augmented_record['llm_score'] = llm_score_for_target
                augmented_record['delta'] = delta
                aggregated_results.append(augmented_record)

            # --- 6. Store Results in Context ---
            context['score_comparison_data'] = aggregated_results
            context['score_comparison_metadata'] = {
                "pipeline_run_ids": pipeline_run_ids,
                "sources_compared": self.sources_to_compare,
                "ground_truth_source": self.ground_truth_source,
                "dimensions": dimensions,
                "comparison_timestamp": datetime.now().isoformat()
            }

            # --- 7. (Optional) Basic Reporting ---
            # Generate a simple summary or export
            self._generate_basic_report(aggregated_results, context['score_comparison_metadata'])

            self.logger.log("ScoreComparisonCompleted", {
                "total_scores_processed": len(aggregated_results),
                # Add more summary stats if needed
            })


            # --- 7. (Optional) Basic Reporting ---
            # Generate a simple summary or export
            self._generate_basic_report(aggregated_results, context['score_comparison_metadata'])

            # --- 8. NEW: Save Raw CSV ---
            self._save_comparison_csv(aggregated_results, context['score_comparison_metadata'])

            # --- 9. Log completion and return ---
            self.logger.log("ScoreComparisonCompleted", {
                "total_scores_processed": len(aggregated_results),
                # Add more summary stats if needed
            })


            # --- 9. NEW: Perform Statistical Analysis ---
            analysis_results = self._perform_statistical_analysis(aggregated_results)
            context['score_analysis_results'] = analysis_results
            context['score_analysis_metadata'] = {
                "analysis_timestamp": datetime.now().isoformat(),
                "sources_analyzed": self.sources_to_compare,
                "ground_truth_source": self.ground_truth_source,
                # You could add more metadata here if needed
            }

            # --- 10. NEW: Generate Detailed Analysis Report ---
            self._generate_analysis_report(analysis_results, context['score_comparison_metadata']) # Use comparison metadata for context

            # --- 11. Log completion and return ---
            self.logger.log("ScoreComparisonCompleted", {
                "total_scores_processed": len(aggregated_results),
                "analysis_results_generated": len(analysis_results) > 0,
                # Add more summary stats if needed
            })

            return context

        except Exception as e:
            error_msg = f"ScoreComparisonAgent failed: {str(e)}"
            self.logger.log("ScoreComparisonFailed", {"error": str(e), "context": str(context)})
            # Depending on requirements, you might want to re-raise or handle gracefully
            raise # Re-raise for now to halt the pipeline on critical failure


    def _fetch_local_scores(self, pipeline_run_ids: List[int], dimensions: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches scores for specified sources linked to specific pipeline runs.
        Uses a SQL query with ROW_NUMBER() and pivoting for efficient retrieval
        of the latest score per target/dimension/source combination.
        """
        try:
            if not pipeline_run_ids:
                self.logger.log("LocalScoreFetchWarning", {"message": "No pipeline_run_ids provided. Returning empty list."})
                return []

            # 1. Build the list of sources to filter by (excluding ground truth for now)
            non_gt_sources = [s for s in self.sources_to_compare if s != self.ground_truth_source]
            
            # Handle case where only GT source is requested
            if not non_gt_sources:
                self.logger.log("LocalScoreFetchInfo", {"message": "No non-ground-truth sources to fetch. Returning empty list."})
                return []

            # 2. Create placeholders for the IN clauses in the SQL query
            # Note: Using tuple() for IN clauses in SQLAlchemy text queries
            pipeline_ids_tuple = tuple(pipeline_run_ids) if pipeline_run_ids else (None,) # Prevent empty tuple error
            sources_tuple = tuple(non_gt_sources) if non_gt_sources else (None,)
            dimensions_tuple = tuple(dimensions) if dimensions else None # Will handle NULL check in SQL

            # 3. Define the SQL query using text()
            # We'll build the CASE statements dynamically based on sources
            case_statements = []
            for source in non_gt_sources:
                # Normalize source name for column alias (e.g., 'sicql_scorer' -> 'sicql_score')
                # Adjust this normalization logic if needed based on your exact evaluator names
                # The key change: Use the column name 'source' directly, not 's.source'
                case_statements.append(f"MAX(CASE WHEN source = '{source}' THEN score END) AS {source}_score")
            
            case_part = ",\n        ".join(case_statements)

            # 4. Base query - CORRECTED: Removed 's.' prefix in the grouped_scores CTE
            query_text = f"""
            WITH pipeline_scores AS (
                SELECT
                    e.target_type,
                    e.target_id,
                    s.dimension,
                    s.source, -- Column alias 'source'
                    s.score,  -- Column alias 'score'
                    ROW_NUMBER() OVER (
                        PARTITION BY e.target_type, e.target_id, s.dimension, s.source
                        ORDER BY e.created_at DESC
                    ) AS row_num
                FROM scores s
                JOIN evaluations e ON s.evaluation_id = e.id
                WHERE e.pipeline_run_id IN :pipeline_run_ids
                AND s.source IN :sources
                -- Filter by dimensions if provided
                AND (:dimensions IS NULL OR s.dimension IN :dimensions)
            ),
            latest_scores AS (
                SELECT *
                FROM pipeline_scores
                WHERE row_num = 1
            ),
            grouped_scores AS (
                SELECT
                    target_type,
                    target_id,
                    dimension,
                    {case_part} -- Uses 'source' and 'score' from latest_scores
                FROM latest_scores
                GROUP BY target_type, target_id, dimension
            )
            SELECT *
            FROM grouped_scores
            ORDER BY dimension, target_type, target_id;
            """

            # 5. Log the query for debugging (optional, remove in production)
            # self.logger.log("DebugSQLQuery", {"query": query_text, "params": {
            #     "pipeline_run_ids": pipeline_ids_tuple,
            #     "sources": sources_tuple,
            #     "dimensions": dimensions_tuple
            # }})

            # 6. Execute the query with parameters
            result = self.session.execute(
                text(query_text),
                {
                    "pipeline_run_ids": pipeline_ids_tuple,
                    "sources": sources_tuple,
                    "dimensions": dimensions_tuple
                }
            )

            # 7. Process the results
            # The result will have columns like: target_type, target_id, dimension, sicql_score, mrq_score, ...
            raw_rows = result.fetchall()

            formatted_scores = []
            for row in raw_rows:
                row_dict = row._mapping # Convert Row to dict-like object
                
                target_type = row_dict.get("target_type")
                target_id = row_dict.get("target_id")
                dimension = row_dict.get("dimension")

                # Iterate through the dynamically created score columns
                for source_alias in non_gt_sources: # e.g., 'sicql', 'mrq', 'svm', 'ebt'
                    # The column name in the result set matches the alias used in CASE
                    column_name = f"{source_alias}_score" 
                    
                    score_value = row_dict.get(column_name)
                    
                    # Only add an entry if a score was found for this source
                    if score_value is not None:
                        formatted_scores.append({
                            # Evaluation ID is not directly available in this pivoted format.
                            "target_id": target_id,
                            "target_type": target_type,
                            "dimension": dimension,
                            "source": source_alias, # Use the original source name
                            "score": float(score_value), # Ensure it's a native Python type
                        })

            self.logger.log("LocalScoresFetched", {
                "requested_pipeline_runs": pipeline_run_ids,
                "requested_sources": non_gt_sources,
                "requested_dimensions": dimensions,
                "fetched_record_count": len(raw_rows), # Number of grouped rows
                "expanded_score_count": len(formatted_scores) # Number of individual score entries
            })
            return formatted_scores

        except sqlalchemy.exc.SQLAlchemyError as sae:
            # More specific error handling for database issues
            self.logger.log("LocalScoreFetchDatabaseError", {"error": f"SQLAlchemy Error: {str(sae)}", "query": query_text if 'query_text' in locals() else "Query construction failed"})
            return []
        except Exception as e:
            self.logger.log("LocalScoreFetchFailed", {"error": f"General Error: {str(e)}", "pipeline_run_ids": pipeline_run_ids, "dimensions": dimensions})
            return [] # Return empty list on error to allow pipeline to potentially continue

    def _fetch_latest_ground_truth_scores(self, target_info_list: List[Dict[str, Any]], dimensions: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches the latest scores from the ground truth source (e.g., LLM) for given targets.
        Adapted from PolicyAnalyzer._get_llm_data.
        """

        if not target_info_list:
             return []

        try:
            # We need to get the LATEST score for each (target_id, dimension) pair where source is LLM
            # This is trickier than a simple filter. We can use a subquery or window function.
            # Let's use a common approach: join with a subquery that finds the max created_at per group.

            # Subquery to find the latest evaluation_id for each (target_id, dimension) for LLM
            latest_eval_subq = (
                self.session.query(
                    EvaluationORM.target_id,
                    ScoreORM.dimension,
                    # Using func.max might not directly give us the id, so we use a window function approach
                    # Or, simpler, get the latest EvaluationORM.id per group and join back
                )
                .join(ScoreORM, ScoreORM.evaluation_id == EvaluationORM.id)
                .filter(EvaluationORM.evaluator_name == self.ground_truth_source)
                .filter(EvaluationORM.target_id.in_([t['target_id'] for t in target_info_list]))
                .filter(ScoreORM.dimension.in_(dimensions) if dimensions else True)
                # Group by target and dimension
                .group_by(EvaluationORM.target_id, ScoreORM.dimension)
                # This approach with group_by alone won't give the latest id directly
                # Let's use a more robust method with a correlated subquery or distinct on
                # Or, use the logic from ScoringStore.load_gild_examples which handles "latest"
            )

            # Simpler and more aligned with existing patterns: Use a modified version of the logic
            # that gets latest scores for a specific source, similar to how `load_gild_examples` works
            # but filtered for LLM and specific targets/dimensions.

            # Let's adapt the CTE logic from ScoringStore.load_gild_examples for just LLM
            from sqlalchemy import text
            # This is a simplified version focusing only on LLM
            # Note: This assumes target_type is consistent or handled, or we filter it out if not needed here
            cte_query_text = """
            WITH ranked_llm_scores AS (
                SELECT
                    s.dimension,
                    s.score,
                    e.target_id,
                    e.id as evaluation_id, -- Include evaluation_id for join if needed
                    e.created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.target_id, s.dimension
                        ORDER BY e.created_at DESC
                    ) AS rank
                FROM scores s
                JOIN evaluations e ON e.id = s.evaluation_id
                WHERE e.evaluator_name = :evaluator_name -- 'llm'
                AND e.target_id IN :target_ids
                AND s.dimension IN :dimensions
                -- Add target_type filter if strictly needed
            )
            SELECT
                target_id,
                dimension,
                score,
                created_at
            FROM ranked_llm_scores
            WHERE rank = 1
            """

            # Prepare parameters
            target_ids = [t['target_id'] for t in target_info_list]
            dims = dimensions if dimensions else [t['dimension'] for t in target_info_list] # Fallback if needed

            if not target_ids or not dims: # Safety check
                 return []

            result = self.session.execute(
                text(cte_query_text),
                {
                    "evaluator_name": self.ground_truth_source,
                    "target_ids": tuple(target_ids),
                    "dimensions": tuple(dims)
                }
            ).fetchall()

            llm_scores = [dict(row._mapping) for row in result]
            
            self.logger.log("GroundTruthScoresFetched", {"count": len(llm_scores)})
            return llm_scores

        except Exception as e:
             self.logger.log("GroundTruthScoreFetchFailed", {"error": str(e)})
             return []

    def _generate_basic_report(self, aggregated_data: List[Dict], metadata: Dict):
        """
        Generates a simple summary report of the comparison.
        """
        try:
            if not aggregated_data:
                 report_content = "# Score Comparison Report (Empty)\n\nNo data found for comparison.\n"
                 self.logger.log("EmptyComparisonReportGenerated", {})
            else:
                # Simple aggregation: count, average delta per source
                from collections import defaultdict
                import statistics

                source_stats = defaultdict(lambda: {"count": 0, "avg_delta": 0, "deltas": []})
                
                for item in aggregated_data:
                     source = item.get('source')
                     delta = item.get('delta')
                     if source: # Ensure source is present
                          source_stats[source]["count"] += 1
                          if delta is not None:
                               source_stats[source]["deltas"].append(delta)
                
                # Calculate average deltas
                for source, stats in source_stats.items():
                     if stats["deltas"]:
                          stats["avg_delta"] = statistics.mean(stats["deltas"])
                          # Could add stddev, min, max etc.
                     del stats["deltas"] # Remove raw list for cleaner output

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                report_filename = f"score_comparison_summary_{timestamp}.md"
                report_path = os.path.join(self.output_dir, report_filename)

                with open(report_path, 'w') as f:
                     f.write("# Score Comparison Summary Report\n\n")
                     f.write(f"**Generated:** {metadata.get('comparison_timestamp', 'N/A')}\n\n")
                     f.write(f"**Pipeline Runs Analyzed:** {metadata.get('pipeline_run_ids', 'N/A')}\n\n")
                     f.write(f"**Sources Compared:** {', '.join(metadata.get('sources_compared', []))}\n\n")
                     f.write(f"**Ground Truth Source:** {metadata.get('ground_truth_source', 'N/A')}\n\n")
                     f.write(f"**Dimensions:** {', '.join(metadata.get('dimensions', []))}\n\n")
                     f.write("## Summary Statistics (vs Ground Truth)\n\n")
                     f.write("| Source | Count | Avg Delta (Model - LLM) |\n")
                     f.write("| :--- | :--- | :--- |\n")
                     for source, stats in sorted(source_stats.items()):
                          f.write(f"| {source} | {stats['count']} | {stats['avg_delta']:.4f} |\n")
                
                self.logger.log("ComparisonSummaryReportSaved", {"path": report_path})

        except Exception as e:
             self.logger.log("ComparisonReportGenerationFailed", {"error": str(e)})


    def _save_comparison_csv(self, aggregated_data: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """
        Saves the aggregated score comparison data to a CSV file.
        """
        try:
            if not aggregated_data:
                self.logger.log("SaveComparisonCSVWarning", {"message": "No data to save to CSV. Skipping."})
                return

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Create a descriptive filename
            pipeline_ids_str = "_".join(map(str, metadata.get('pipeline_run_ids', ['unknown'])))
            filename = f"score_comparison_raw_{pipeline_ids_str}_{timestamp}.csv"
            file_path = os.path.join(self.output_dir, filename)

            # Define the fieldnames for the CSV. These should match the keys in your aggregated data dicts.
            # Based on the current structure in run() and _fetch_local_scores/_fetch_latest_ground_truth_scores
            fieldnames = [
                "target_id",
                "target_type",
                "dimension",
                "source", # The model that produced the score
                "score",  # The score from the model
                "llm_score", # The ground truth LLM score for the same target/dimension
                "delta"  # The difference (score - llm_score)
                # Add other fields like 'embedding_type', 'created_at' if they are included in aggregated_data
            ]
            # Dynamically add any other keys found in the first data point (for robustness)
            if aggregated_data:
                sample_keys = set(aggregated_data[0].keys())
                for key in sample_keys:
                    if key not in fieldnames:
                        fieldnames.append(key)
            # Ensure consistent order, put standard ones first
            standard_fields = [f for f in fieldnames if f in ['target_id', 'target_type', 'dimension', 'source', 'score', 'llm_score', 'delta']]
            other_fields = [f for f in fieldnames if f not in standard_fields]
            ordered_fieldnames = standard_fields + sorted(other_fields)


            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=ordered_fieldnames)
                writer.writeheader()
                # Sort data for consistent output (optional)
                sorted_data = sorted(aggregated_data, key=lambda x: (x.get('dimension', ''), x.get('target_type', ''), x.get('target_id', 0), x.get('source', '')))
                for row in sorted_data:
                    # Handle potential serialization issues (e.g., datetime objects)
                    # Although our current data should be basic types, this is good practice.
                    safe_row = {k: v if isinstance(v, (str, int, float, type(None))) else str(v) for k, v in row.items()}
                    writer.writerow(safe_row)


            self.logger.log("ComparisonCSVSaved", {"path": file_path, "record_count": len(aggregated_data)})
            
        except Exception as e:
            self.logger.log("SaveComparisonCSVFailed", {"error": str(e), "output_dir": self.output_dir})



    def _perform_statistical_analysis(self, aggregated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Performs statistical analysis on the aggregated score comparison data.
        Calculates MAE, RMSE, Correlation, Bias, and Variance of model scores
        per source and dimension.
        """
        try:

            # --- 1. Organize Data ---
            # Group data by (source, dimension) for metric calculation
            grouped_data = defaultdict(list)
            for item in aggregated_data:
                # Only analyze non-ground-truth scores that have a corresponding LLM score/delta
                if item.get('source') and item.get('source') != self.ground_truth_source and item.get('delta') is not None:
                    key = (item['source'], item['dimension'])
                    grouped_data[key].append(item)
                
                # Also collect model scores for calculating variance of scores produced by the model
                # This uses the 'score' field for the model itself
                if item.get('source') and item.get('source') != self.ground_truth_source and item.get('score') is not None:
                    # Use a distinct key structure for model scores
                    score_key = (item['source'], item['dimension'], 'model_scores') 
                    grouped_data[score_key].append(item['score']) # Store just the score value

            # --- 2. Calculate Metrics ---
            results = {}

            # --- Calculate Main Metrics (MAE, RMSE, Correlation, Bias) ---
            for key, items in grouped_data.items():
                # Process only the main metric keys: (source, dimension)
                # Skip keys with 'model_scores' marker: (source, dimension, 'model_scores')
                if isinstance(key, tuple) and len(key) == 3 and key[2] == 'model_scores':
                    continue # This will be handled later for score variance

                if isinstance(key, tuple) and len(key) == 2:
                    source, dimension = key
                else:
                    # Handle unexpected key format gracefully
                    self.logger.log("StatisticalAnalysisWarning", {
                        "message": "Skipping unexpected key format in grouped data",
                        "key": str(key), "key_type": str(type(key))
                    })
                    continue

                if not items:
                    continue

                # Extract arrays for calculations
                deltas = np.array([item['delta'] for item in items if item['delta'] is not None])
                model_scores = np.array([item['score'] for item in items if item['score'] is not None])
                llm_scores = np.array([item['llm_score'] for item in items if item['llm_score'] is not None])

                # Ensure we have data to calculate metrics
                if len(deltas) == 0 or len(model_scores) == 0 or len(llm_scores) == 0:
                    self.logger.log("StatisticalAnalysisWarning", {
                        "message": "Insufficient data for metric calculation",
                        "source": source, "dimension": dimension,
                        "delta_count": len(deltas), "model_score_count": len(model_scores), "llm_score_count": len(llm_scores)
                    })
                    continue

                # --- Core Metrics ---
                mae = np.mean(np.abs(deltas))
                rmse = np.sqrt(np.mean(deltas**2))
                
                # Correlation: Check if variance is sufficient for meaningful correlation
                corr_coef = None
                corr_p_value = None
                if np.std(model_scores) > 1e-10 and np.std(llm_scores) > 1e-10:
                    try:
                        # Use scipy.stats.pearsonr
                        corr_result = pearsonr(model_scores, llm_scores)
                        # Handle different scipy versions
                        if hasattr(corr_result, 'statistic'):
                            corr_coef = corr_result.statistic
                            corr_p_value = corr_result.pvalue
                        else: # Older scipy versions return a tuple
                            corr_coef, corr_p_value = corr_result
                    except Exception as e:
                        self.logger.log("CorrelationCalculationWarning", {
                            "error": str(e), "source": source, "dimension": dimension
                        })

                bias = np.mean(deltas) # Average difference (Model - LLM)

                # --- Store Results ---
                result_key = f"{source}_{dimension}"
                results[result_key] = {
                    "source": source,
                    "dimension": dimension,
                    "count": len(deltas),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "correlation": float(corr_coef) if corr_coef is not None else None,
                    "correlation_p_value": float(corr_p_value) if corr_p_value is not None else None,
                    "bias": float(bias), # Positive bias = Model tends to score higher than LLM
                }

            # --- 3. Calculate Variance of Model Scores ---
            # This is the standard deviation of scores *produced by each model* for a dimension
            for key, scores_list in grouped_data.items():
                # Process only the score variance keys: (source, dimension, 'model_scores')
                if not (isinstance(key, tuple) and len(key) == 3 and key[2] == 'model_scores'):
                    continue # Skip main metric keys

                source, dimension, _ = key # Unpack the 3-tuple key

                if not scores_list:
                    continue
                
                scores_array = np.array(scores_list)
                if len(scores_array) > 1: # Need more than one value for std dev
                    score_variance = float(np.std(scores_array))
                    # Add this to the existing result dict or create a new entry if it doesn't exist
                    result_key = f"{source}_{dimension}"
                    if result_key in results:
                        results[result_key]["score_std_dev"] = score_variance
                    else:
                        # Less likely, but handle if main metrics weren't calculated for some reason
                        results[result_key] = {
                            "source": source,
                            "dimension": dimension,
                            "count": len(scores_array),
                            "score_std_dev": score_variance
                            # Other metrics will be missing
                        }
                # else: Not enough data for variance, leave it out or set to 0?

            self.logger.log("StatisticalAnalysisCompleted", {
                "unique_source_dimension_combinations_analyzed": len(results)
            })
            
            return results

        except Exception as e:
            self.logger.log("StatisticalAnalysisFailed", {"error": str(e)})
            # Depending on robustness needs, return empty dict or re-raise
            return {} # Return empty dict on error to allow pipeline to potentially continue

    def _generate_analysis_report(self, analysis_results: Dict[str, Any], comparison_metadata: Dict[str, Any]):
        """
        Generates a detailed markdown report summarizing the statistical analysis results.
        """
        try:
            if not analysis_results:
                report_content = "# Score Analysis Report (Empty)\n\nNo analysis data found.\n"
                self.logger.log("EmptyAnalysisReportGenerated", {})
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                pipeline_ids_str = "_".join(map(str, comparison_metadata.get('pipeline_run_ids', ['unknown'])))
                report_filename = f"score_analysis_detailed_{pipeline_ids_str}_{timestamp}.md"
                report_path = os.path.join(self.output_dir, report_filename)

                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Detailed Score Analysis Report\n\n")
                    f.write(f"**Generated:** {comparison_metadata.get('comparison_timestamp', 'N/A')}\n\n") # Use timestamp from comparison
                    f.write(f"**Analysis Performed:** {datetime.now().isoformat()}\n\n")
                    f.write(f"**Pipeline Runs Analyzed:** {comparison_metadata.get('pipeline_run_ids', 'N/A')}\n\n")
                    f.write(f"**Sources Compared:** {', '.join(comparison_metadata.get('sources_compared', []))}\n\n")
                    f.write(f"**Ground Truth Source:** {comparison_metadata.get('ground_truth_source', 'N/A')}\n\n")
                    f.write(f"**Dimensions:** {', '.join(comparison_metadata.get('dimensions', []))}\n\n")
                    f.write("---\n\n")

                    # Group results by dimension for better organization
                    from collections import defaultdict
                    results_by_dimension = defaultdict(list)
                    for key, metrics in analysis_results.items():
                        dim = metrics.get('dimension', 'unknown')
                        results_by_dimension[dim].append(metrics)

                    # Sort dimensions and sources for consistent output
                    sorted_dimensions = sorted(results_by_dimension.keys())

                    for dimension in sorted_dimensions:
                        f.write(f"## Analysis for Dimension: `{dimension}`\n\n")
                        f.write("| Source | Count | MAE | RMSE | Correlation (p-value) | Bias | Score Std Dev |\n")
                        f.write("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n")

                        sorted_sources_for_dim = sorted(results_by_dimension[dimension], key=lambda x: x.get('source', ''))
                        for metrics in sorted_sources_for_dim:
                            source = metrics.get('source', 'N/A')
                            count = metrics.get('count', 0)
                            mae = f"{metrics.get('mae', 0):.4f}" if metrics.get('mae') is not None else "N/A"
                            rmse = f"{metrics.get('rmse', 0):.4f}" if metrics.get('rmse') is not None else "N/A"
                            
                            corr = metrics.get('correlation')
                            p_val = metrics.get('correlation_p_value')
                            if corr is not None:
                                corr_str = f"{corr:.4f}"
                                if p_val is not None:
                                    corr_str += f" ({p_val:.2e})" # Add p-value in scientific notation
                            else:
                                corr_str = "N/A"

                            bias = f"{metrics.get('bias', 0):.4f}" if metrics.get('bias') is not None else "N/A"
                            std_dev = f"{metrics.get('score_std_dev', 0):.4f}" if metrics.get('score_std_dev') is not None else "N/A"

                            f.write(f"| {source} | {count} | {mae} | {rmse} | {corr_str} | {bias} | {std_dev} |\n")
                        f.write("\n") # Space between dimensions

                self.logger.log("DetailedAnalysisReportSaved", {"path": report_path})

        except Exception as e:
            self.logger.log("DetailedAnalysisReportGenerationFailed", {"error": str(e)})
        