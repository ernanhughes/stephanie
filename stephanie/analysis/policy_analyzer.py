# stephanie/analysis/policy_analyzer.py
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from stephanie.models.evaluation import EvaluationORM
from stephanie.models.evaluation_attribute import EvaluationAttributeORM
from stephanie.models.score import ScoreORM


class PolicyAnalyzer:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.uncertainty_threshold = 0.3  # From config

    def analyze_dimension(self, dimension: str, pipeline_run_id: int = None) -> Dict[str, Any]:
        try:
            sicql_data = self._get_sicql_data(dimension, pipeline_run_id)
            mrq_data = self._get_mrq_data(dimension, pipeline_run_id)
            svm_data = self._get_svm_data(dimension, pipeline_run_id)
            ebt_data = self._get_ebt_data(dimension, pipeline_run_id)
            llm_data = self._get_llm_data(dimension, pipeline_run_id)

            comparison_results = [
                self._compare_with_source(sicql_data, mrq_data, "mrq"),
                self._compare_with_source(sicql_data, svm_data, "svm"),
                self._compare_with_source(sicql_data, ebt_data, "ebt"),
                # self._compare_with_source(sicql_data, llm_data, "llm"),
            ]

            results = {
                "dimension": dimension,
                "policy_stats": self._analyze_policy_patterns(sicql_data),
                "comparisons": comparison_results,
                "uncertainty_cases": self._find_high_uncertainty(sicql_data),
                "policy_entropy": self._calculate_entropy(sicql_data),
                "policy_drift": self._detect_policy_drift(sicql_data)
            }

            self.logger.log("PolicyAnalysis", results)
            return results

        except Exception as e:
            self.logger.log("PolicyAnalysisFailed", {"error": str(e)})
            raise

    def _get_sicql_data(self, dimension: str, pipeline_run_id: int = None) -> List[Dict]:
        """Get SICQL policy data from database with optional pipeline run filter, including target matching keys"""
        query = (
            self.session.query(EvaluationAttributeORM, EvaluationORM)
            .join(EvaluationORM)
            .filter(
                EvaluationAttributeORM.dimension == dimension,
                EvaluationORM.source.contains("sicql"),
                EvaluationAttributeORM.q_value.isnot(None),
                EvaluationAttributeORM.v_value.isnot(None),
                EvaluationAttributeORM.policy_logits.isnot(None)
            )
        )

        if pipeline_run_id:
            query = query.filter(EvaluationORM.pipeline_run_id == pipeline_run_id)

        results = query.all()

        if not results:
            self.logger.log("NoSICQLDataFound", {
                "dimension": dimension,
                "pipeline_run_id": pipeline_run_id,
            })

        formatted = []
        for attr, eval_obj in results:
            formatted.append({
                "evaluation_id": attr.evaluation_id,
                "policy_logits": json.loads(attr.policy_logits),
                "q_value": attr.q_value,
                "v_value": attr.v_value,
                "uncertainty": attr.uncertainty,
                "dimension": attr.dimension,
                "timestamp": attr.created_at,
                "target_type": eval_obj.target_type,
                "target_id": eval_obj.target_id,
            })

        return formatted

    def _get_mrq_data(self, dimension: str, pipeline_run_id: int = None) -> List[Dict]:
        """Get MRQ scores for comparison, including target matching keys"""
        query = (
            self.session.query(ScoreORM, EvaluationORM)
            .join(EvaluationORM)
            .filter(
                ScoreORM.dimension == dimension,
                EvaluationORM.source.contains("mrq"),
            )
        )

        if pipeline_run_id:
            query = query.filter(EvaluationORM.pipeline_run_id == pipeline_run_id)

        results = query.all()

        return [self._format_score(score, eval_obj) for score, eval_obj in results]

    def _get_svm_data(self, dimension: str, pipeline_run_id: int = None) -> List[Dict]:
        """Get SVM scores for comparison with target identifiers included"""
        query = (
            self.session.query(ScoreORM, EvaluationORM)
            .join(EvaluationORM)
            .filter(
                ScoreORM.dimension == dimension,
                EvaluationORM.source.contains("svm"),
            )
        )

        if pipeline_run_id:
            query = query.filter(EvaluationORM.pipeline_run_id == pipeline_run_id)

        results = query.all()

        return [
            {
                "evaluation_id": score.evaluation_id,
                "score": score.score,
                "dimension": score.dimension,
                "source": "svm",
                "target_type": eval.target_type,
                "target_id": eval.target_id
            }
            for score, eval in results
        ]


    def _get_ebt_data(self, dimension: str, pipeline_run_id: int = None) -> List[Dict]:
        """Get EBT scores with target IDs for comparison"""
        query = (
            self.session.query(ScoreORM, EvaluationORM)
            .join(EvaluationORM)
            .filter(
                ScoreORM.dimension == dimension,
                EvaluationORM.source.contains("ebt"),
            )
        )
        if pipeline_run_id:
            query = query.filter(EvaluationORM.pipeline_run_id == pipeline_run_id)

        results = query.all()
        return [self._format_score(score, eval_obj) for score, eval_obj in results]

    def _get_llm_data(self, dimension: str, pipeline_run_id: int = None) -> List[Dict]:
        """Get LLM-based evaluation scores"""
        query = (
            self.session.query(ScoreORM, EvaluationORM)
            .join(EvaluationORM)
            .filter(
                ScoreORM.dimension == dimension,
                EvaluationORM.source.contains("llm")
            )
        )
        if pipeline_run_id:
            query = query.filter(EvaluationORM.pipeline_run_id == pipeline_run_id)

        results = query.all()
        return [self._format_score(score, eval_obj) for score, eval_obj in results]

    def _format_attribute(self, attr: EvaluationAttributeORM) -> Dict:
        return {
            "evaluation_id": attr.evaluation_id,
            "policy_logits": json.loads(attr.policy_logits),  # Should be 2D
            "q_value": attr.q_value,
            "v_value": attr.v_value,
            "uncertainty": attr.uncertainty,
            "dimension": attr.dimension,
            "timestamp": attr.created_at
        }

    def _format_score(self, score: ScoreORM, eval_obj: EvaluationORM) -> Dict:
        return {
            "evaluation_id": score.evaluation_id,
            "score": score.score,
            "dimension": score.dimension,
            "source": eval_obj.source,
            "target_type": eval_obj.target_type,
            "target_id": eval_obj.target_id,
        }

    def _analyze_policy_patterns(self, sicql_data: List[Dict]) -> Dict[str, Any]:
        """Analyze policy patterns across samples"""
        valid_logits = [d["policy_logits"] for d in sicql_data if d["policy_logits"] is not None and len(d["policy_logits"]) > 1]
        
        if not valid_logits:
            return {"available": False}

        all_logits = np.stack(valid_logits)

        from scipy.special import softmax
        action_probs = softmax(all_logits, axis=1)

        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1)
        most_probable_actions = np.argmax(action_probs, axis=1)
        action_counts = np.bincount(most_probable_actions, minlength=action_probs.shape[1])

        return {
            "available": True,
            "action_distribution": action_counts.tolist(),
            "avg_entropy": float(np.mean(entropy)),
            "action_probabilities": action_probs.tolist(),
            "policy_consistency": self._calculate_policy_consistency(sicql_data)
        }

    def _calculate_policy_consistency(self, sicql_data: List[Dict]) -> Dict[str, Any]:
        """Measure policy consistency over time"""
        if len(sicql_data) < 2:
            return {"consistent": None}
            
        # Compare consecutive policy outputs
        actions = np.argmax([d["policy_logits"] for d in sicql_data if d["policy_logits"] is not None], axis=1)
        policy_changes = np.sum(actions[1:] != actions[:-1]) / len(actions) if len(actions) > 1 else 0
        
        # Compare Q/V alignment
        q_values = np.array([d["q_value"] for d in sicql_data if d["q_value"] is not None])
        v_values = np.array([d["v_value"] for d in sicql_data if d["v_value"] is not None])
        
        if len(q_values) > 1:
            if np.std(q_values) == 0 or np.std(v_values) == 0:
                q_v_correlation = None
            else:
                q_v_correlation = np.corrcoef(q_values, v_values)[0,1]
        else:
            q_v_correlation = None
            
        return {
            "policy_stability": 1 - policy_changes,
            "q_v_correlation": float(q_v_correlation) if q_v_correlation is not None else None,
            "sample_count": len(sicql_data)
        }

    def _compare_with_source(self, sicql_data: List[Dict], reference_data: List[Dict], label: str) -> Dict[str, Any]:
        """Compare SICQL policy with reference data using document_id and dimension for matching"""
        if not sicql_data or not reference_data:
            return {
                "source": label,
                "comparable": False,
                "score_correlation": None,
                "avg_score_deviation": None,
                "sample_count": 0
            }
        
        # Create lookup by (document_id, dimension)
        ref_by_key = {}
        for d in reference_data:
            # Use target_id and dimension as the key
            key = (d.get("target_id"), d.get("dimension"))
            if None not in key:  # Skip if either is None
                ref_by_key[key] = d
        
        # Find matching records
        matched_data = []
        for sicql in sicql_data:
            key = (sicql.get("target_id"), sicql.get("dimension"))
            if key in ref_by_key:
                matched_data.append((sicql, ref_by_key[key]))
        
        if not matched_data:
            return {
                "source": label,
                "comparable": False,
                "score_correlation": None,
                "avg_score_deviation": None,
                "sample_count": 0
            }
        
        # Calculate correlation
        sicql_scores = [d[0]["q_value"] for d in matched_data]
        ref_scores = [d[1]["score"] for d in matched_data]
        
        if len(sicql_scores) < 5:
            self.logger.warning("Low sample count for correlation", extra={
                "source": label,
                "sample_count": len(sicql_scores),
                "dimension": sicql_data[0].get("dimension", "unknown")
            })
        
        if len(sicql_scores) < 2:
            score_correlation = None
        else:
            score_correlation = np.corrcoef(sicql_scores, ref_scores)[0, 1]
        
        # Calculate average deviation
        score_deviation = np.mean([abs(a - b) for a, b in zip(sicql_scores, ref_scores)]) if sicql_scores else None
        
        return {
            "source": label,
            "comparable": True,
            "score_correlation": float(score_correlation) if score_correlation is not None else None,
            "avg_score_deviation": float(score_deviation) if score_deviation is not None else None,
            "sample_count": len(sicql_scores)
        }

    def _find_high_uncertainty(self, sicql_data: List[Dict]) -> List[Dict]:
        """Find cases where policy was uncertain"""
        return [
            d for d in sicql_data 
            if d["uncertainty"] and d["uncertainty"] > self.uncertainty_threshold
        ]

    def _calculate_entropy(self, sicql_data: List[Dict]) -> Dict[str, float]:
        """Calculate entropy statistics"""
        valid_logits = [d["policy_logits"] for d in sicql_data if d["policy_logits"] is not None]
        if not valid_logits:
            return {}
            
        action_probs = np.exp(valid_logits) / np.exp(valid_logits).sum(axis=1, keepdims=True)
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1)
        
        return {
            "avg_entropy": float(np.mean(entropy)),
            "std_entropy": float(np.std(entropy)),
            "high_entropy": int(np.sum(entropy > np.median(entropy) + np.std(entropy)))
        }

    def _detect_policy_drift(self, sicql_data: List[Dict]) -> Dict[str, Any]:
        """Detect policy drift over time"""
        if len(sicql_data) < 2:
            return {"drift_detected": False}
            
        # Sort by timestamp
        sorted_data = sorted(sicql_data, key=lambda x: x["timestamp"])
        actions = np.argmax([d["policy_logits"] for d in sorted_data if d["policy_logits"] is not None], axis=1)
        
        if len(actions) < 2:
            return {"drift_detected": False}
            
        # Calculate action changes over time
        changes = np.sum(actions[1:] != actions[:-1]) / len(actions)
        
        # Detect clusters of similar policy outputs
        from sklearn.cluster import KMeans
        action_probs = np.exp(actions) / np.exp(actions).sum()
        
        try:
            unique_actions = np.unique(actions)
            if len(unique_actions) < 3:
                cluster_changes = 0
            else:
                kmeans = KMeans(n_clusters=3)
                clusters = kmeans.fit_predict(action_probs.reshape(-1, 1))
                cluster_changes = np.sum(clusters[1:] != clusters[:-1]) / len(clusters)
        except:
            cluster_changes = 0
            
        return {
            "drift_detected": changes > 0.3,
            "action_drift_rate": float(changes),
            "cluster_drift_rate": float(cluster_changes),
            "sample_count": len(actions)
        }

    def generate_policy_report(self, dimension: str, pipeline_run_id: int = None) -> Dict[str, Any]:
        """Generate comprehensive policy report with flattened structure"""
        analysis = self.analyze_dimension(dimension, pipeline_run_id=pipeline_run_id)

        # Extract policy stats
        policy_stats = analysis.get("policy_stats", {})
        policy_consistency = policy_stats.get("policy_consistency", {})
        
        # Extract comparison stats for each source
        comparison_dict = {
            comp["source"]: {
                "score_correlation": comp["score_correlation"],
                "score_deviation": comp["avg_score_deviation"],
                "sample_count": comp["sample_count"]
            }
            for comp in analysis.get("comparisons", [])
        }
        
        # Extract other stats
        uncertainty_cases = analysis.get("uncertainty_cases", [])
        policy_entropy = analysis.get("policy_entropy", {})
        
        report = {
            "dimension": dimension,
            "policy_available": policy_stats.get("available", False),
            "policy_stability": policy_consistency.get("policy_stability", None),
            "q_v_correlation": policy_consistency.get("q_v_correlation", None),
            "policy_entropy_avg": policy_entropy.get("avg_entropy", None),
            "policy_entropy_std": policy_entropy.get("std_entropy", None),
            "uncertainty_count": len(uncertainty_cases),
            "policy_drift": policy_consistency.get("drift_detected", False),
            "action_drift_rate": policy_consistency.get("action_drift_rate", 0.0),
            **{
                f"{src}_correlation": comparison_dict.get(src, {}).get("score_correlation")
                for src in ["mrq", "svm", "ebt", "llm"]
            },
            **{
                f"{src}_deviation": comparison_dict.get(src, {}).get("score_deviation")
                for src in ["mrq", "svm", "ebt", "llm"]
            },
            **{
                f"{src}_samples": comparison_dict.get(src, {}).get("sample_count")
                for src in ["mrq", "svm", "ebt", "llm"]
            },
            "insights": self._generate_insights(analysis)
        }

        return report

    def _generate_insights(self, stats: Dict) -> List[str]:
        """Generate actionable insights from flattened policy report"""
        insights = []
        
        # 1. Uncertainty detection
        uncertainty_count = stats.get("uncertainty_count", 0)
        sample_count = stats.get("sample_count", 1)  # Avoid division by 0
        uncertainty_ratio = uncertainty_count / sample_count
        
        if uncertainty_count > 0:
            insights.append(
                f"Found {uncertainty_count} high-uncertainty cases ({uncertainty_ratio:.1%} of samples). "
                "Consider retraining for this dimension."
            )

        # Check for weak Q/V alignment
        if stats.get("qv_correlation") is not None and stats["qv_correlation"] < 0.3:
            insights.append("⚠️ Weak alignment between Q and V — the policy may lack internal consistency.")

        # Check for poor alignment with MRQ
        # Multi-source model comparisons
        for src in ["mrq", "svm", "ebt", "llm"]:
            corr = stats.get(f"{src}_correlation")
            dev = stats.get(f"{src}_deviation")
            samples = stats.get(f"{src}_samples", 0)

            if corr is not None and abs(corr) < 0.3:
                insights.append(
                    f"⚠️ Low correlation with {src.upper()} model ({corr:.2f}) — consider alignment or retraining."
                )
            if dev is not None and dev > 20:
                insights.append(
                    f"⚠️ Score deviation from {src.upper()} exceeds 20 units ({dev:.1f}) — calibration may be needed."
                )
            if samples < 5:
                insights.append(
                    f"⚠️ Fewer than 5 samples available for {src.upper()} comparison — not enough for statistical confidence."
                )

        # High uncertainty
        uncertainty_cases = stats.get("uncertainty_cases", 0)
        if isinstance(uncertainty_cases, list):
            uncertainty_count = len(uncertainty_cases)
        else:
            uncertainty_count = uncertainty_cases

        if uncertainty_count > 50:
            insights.append("⚠️ Model is highly uncertain — consider revisiting training or reward signal.")

        # Low policy stability
        if stats.get("policy_stability", 1.0) < 0.5:
            insights.append("⚠️ Policy behavior is unstable — high action variance detected.")



        
        # 2. Policy stability check
        policy_stability = stats.get("policy_stability")
        if policy_stability is not None:
            if policy_stability < 0.7:
                insights.append(
                    f"Policy shows instability (stability: {policy_stability:.2f}). "
                    "Consider policy smoothing or additional training data."
                )
            elif policy_stability < 0.9:
                insights.append(
                    f"Moderate policy stability ({policy_stability:.2f}) - watch for drift."
                )
        
        # 3. Score alignment analysis
        score_correlation = stats.get("score_correlation")
        score_deviation = stats.get("score_deviation")
        
        if score_correlation is not None and score_deviation is not None:
            if score_correlation < 0.5:
                insights.append(
                    f"Low correlation ({score_correlation:.2f}) between SICQL and MRQ scores. "
                    "Policy may be misaligned with expected scoring patterns."
                )
            if score_deviation > 5.0:
                insights.append(
                    f"Large score deviation ({score_deviation:.1f}) from MRQ baseline. "
                    "Consider recalibration or investigating outlier cases."
                )
        
        # 4. Entropy analysis
        policy_entropy_avg = stats.get("policy_entropy_avg")
        if policy_entropy_avg is not None:
            if policy_entropy_avg > 2.0:
                insights.append(
                    f"High policy entropy ({policy_entropy_avg:.2f}) - system is exploring broadly. "
                    "Consider focusing training on high-reward regions."
                )
            elif policy_entropy_avg < 0.5:
                insights.append(
                    f"Low policy entropy ({policy_entropy_avg:.2f}) - system is exploiting known strategies. "
                    "Consider adding diversity to training data."
                )
        
        # 5. Policy drift detection
        policy_drift = stats.get("policy_drift", False)
        action_drift_rate = stats.get("action_drift_rate", 0.0)
        
        if policy_drift and action_drift_rate > 0.2:
            insights.append(
                f"Policy drift detected (drift rate: {action_drift_rate:.2f}). "
                "Consider retraining or policy regularization."
            )
 
        if not insights:
            insights.append("✅ No issues detected.")

        return insights
        
    def visualize_policy(self, dimension: str, output_path: str = "logs/policy_visualization"):
        """Generate policy visualization for a dimension"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sicql_data = self._get_sicql_data(dimension)
            if not sicql_data or not any(d["policy_logits"] for d in sicql_data):
                return None
                
            # Create DataFrame
            df = pd.DataFrame({
                "timestamp": [d["timestamp"] for d in sicql_data],
                "q_value": [d["q_value"] for d in sicql_data],
                "uncertainty": [d["uncertainty"] for d in sicql_data],
                "action": np.argmax(
                    [d["policy_logits"] for d in sicql_data if d["policy_logits"] is not None], 
                    axis=1
                ).tolist()
            })
            
            # Policy visualization
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df, x="timestamp", y="q_value", label="Q-value")
            sns.lineplot(data=df, x="timestamp", y="uncertainty", label="Uncertainty")
            
            # Highlight high-uncertainty periods
            high_uncertainty = df[df["uncertainty"] > self.uncertainty_threshold]
            for _, row in high_uncertainty.iterrows():
                plt.axvspan(row["timestamp"], row["timestamp"], alpha=0.2, color='red')
                
            plt.title(f"Policy Behavior for {dimension.capitalize()}")
            plt.xlabel("Time")
            plt.ylabel("Score")
            plt.legend()
            plt.savefig(f"{output_path}_behavior.png")
            plt.close()
            
            # Action distribution
            plt.figure(figsize=(8, 6))
            sns.histplot(df["action"], bins=np.arange(-0.5, 3.5, 1), discrete=True)
            plt.title(f"Action Distribution for {dimension.capitalize()}")
            plt.xlabel("Selected Action")
            plt.ylabel("Frequency")
            plt.savefig(f"{output_path}_distribution.png")
            plt.close()
            
            return {
                "behavior_plot": f"{output_path}_behavior.png",
                "distribution_plot": f"{output_path}_distribution.png"
            }
            
        except Exception as e:
            self.logger.log("PolicyVisualizationFailed", {"error": str(e)})
            return None
    
    def _generate_visualization_guidance(self, report):
        if report["policy_drift"]:
            return "policy_drift"
        if report["uncertainty_count"] > 20:
            return "uncertainty_analysis"
        return "standard_view"
    

    def generate_markdown_summary(self, analysis: Dict[str, Any]) -> str:
        lines = [f"# Policy Report for Dimension: **{analysis['dimension']}**\n"]
        
        lines.append("## Summary Metrics:")
        lines.append(f"- Policy Stability: `{analysis.get('policy_stability')}`")
        lines.append(f"- Q/V Correlation: `{analysis.get('q_v_correlation')}`")
        lines.append(f"- Score Correlation (vs MRQ): `{analysis.get('score_correlation')}`")
        lines.append(f"- Score Deviation: `{analysis.get('score_deviation')}`")
        lines.append(f"- Entropy (avg): `{analysis.get('policy_entropy_avg')}`")
        lines.append(f"- Uncertainty Cases: `{analysis.get('uncertainty_count')}`")
        lines.append(f"- Policy Drift Detected: `{analysis.get('policy_drift')}`")
        lines.append(f"- Action Drift Rate: `{analysis.get('action_drift_rate')}`")
        
        lines.append("\n## Insights:")
        insights = analysis.get("insights", [])
        if insights:
            for insight in insights:
                lines.append(f"- {insight}")
        else:
            lines.append("- No insights generated.")

        return "\n".join(lines)
