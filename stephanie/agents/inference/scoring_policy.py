import torch
from sqlalchemy import text

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.inference.ebt_inference import EBTInferenceAgent
from stephanie.agents.inference.llm_inference import LLMInferenceAgent
from stephanie.agents.inference.mrq_inference import MRQInferenceAgent
from stephanie.memcubes.memcube import MemCube
from stephanie.memcubes.memcube_factory import MemCubeFactory
from stephanie.scoring.ebt.buffer import EBTTrainingBuffer
from stephanie.scoring.ebt.refinement_trainer import EBTRefinementTrainer
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType

DEFAULT_DIMENSIONS = [
    "alignment",
    "implementability",
    "clarity",
    "relevance",
    "novelty",
]


class ScoringPolicyAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.source = cfg.get("source", "mrq")

        self.dimensions = cfg.get("dimensions", DEFAULT_DIMENSIONS)

        self.ebt_refine_threshold = cfg.get("ebt_refinement_threshold", 0.7)
        self.llm_fallback_threshold = cfg.get("llm_fallback_threshold", 0.9)
        self.steps = cfg.get("optimization_steps", 10)
        self.step_size = cfg.get("step_size", 0.05)

        self.ebt = EBTInferenceAgent(self.cfg, memory, logger)
        self.mrq = MRQInferenceAgent(self.cfg, memory, logger)
        self.llm = LLMInferenceAgent(self.cfg, memory, logger)

        self.training_buffer_path = cfg.get(
            "training_buffer_path", "ebt_buffer.json"
        )
        self.training_buffer = EBTTrainingBuffer(
            self.logger, self.training_buffer_path
        )

    async def run(self, context: dict) -> dict:
        goal_text = context["goal"]["goal_text"]
        docs = context[self.input_key]
        results = []
        event_ids = []
        self.ebt.load_models(self.dimensions)
        self.mrq.load_models(self.dimensions)


        for doc in docs:
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            # 1. Initial MRQ score
            mrq_scores = self.mrq.score(context, scorable)
            llm_scores = self.llm.score(context, scorable)
            self.logger.log(
                "MRQScoresCalculated",
                {
                    "scorable": scorable.id,
                    "scores": mrq_scores,
                },
            )

            # Step 2: Estimate uncertainty using EBT energy
            ebt_energy = self.ebt.get_energy(goal_text, scorable.text)
            self.logger.log(
                "EBTEnergyCalculated",
                {"scorable": scorable.id, "energy": ebt_energy},
            )
            uncertainty_by_dim = {
                dim: torch.sigmoid(torch.tensor(raw)).item()
                for dim, raw in ebt_energy.items()
            }

            self.logger.log(
                "UncertaintyEstimated",
                {
                    "scorable": scorable.id,
                    "uncertainty_by_dimension": uncertainty_by_dim,
                },
            )

            refined = False
            refined_text = None
            # Step 3: Optional refinement (if any dimension exceeds EBT refine threshold)
            if any(
                u > self.ebt_refine_threshold
                for u in uncertainty_by_dim.values()
            ):
                refined = True
                refined_result = self.ebt.optimize(goal_text, scorable.text)
                refined_text = refined_result.get("refined_text")
                refined_scorable = ScorableFactory.from_text(refined_text, scorable.target_type)
                mrq_scores = self.mrq.score(context, refined_scorable)
                self.logger.log(
                    "DocumentRefinedWithEBT", {"document_id": scorable.id}
                )
                refined_score = refined_result.get("final_energy")

                self._log_memcubes_for_srft(
                    scorable,  
                    refined_text,
                    goal_text,
                    mrq_scores,
                    llm_scores,
                    ebt_energy,
                    uncertainty_by_dim,
                    refined=refined,
                ) 
                # Log disagreement for retraining
                self.training_buffer.maybe_add(
                    context=goal_text,
                    candidate=scorable.text,
                    llm_score=refined_score,
                    ebt_score=mrq_scores["alignment"],
                    threshold=self.cfg.get("disagreement_threshold", 0.15),
                    metadata={"dimension": "alignment"},
                )

            # Step 4: Optional LLM fallback (if any dimension exceeds fallback threshold)
            if any(
                u > self.llm_fallback_threshold
                for u in uncertainty_by_dim.values()
            ):
                llm_scores = self.llm.score(context, scorable)
                final_scores = llm_scores
                source = "llm"
                self.logger.log(
                    "LLMFallbackUsed", {"document_id": scorable.id}
                )
            else:
                final_scores = mrq_scores
                source = "mrq"

            # Log raw data for analysis
            result_entry = {
                "document_id": scorable.id,
                "original_text": scorable.text,
                "refined_text": refined_text if refined else scorable.text,
                "mrq_scores": mrq_scores,
                "ebt_energy": ebt_energy,
                "uncertainty_by_dimension": uncertainty_by_dim,
                "used_refinement": any(
                    u > self.ebt_refine_threshold
                    for u in uncertainty_by_dim.values()
                ),
                "used_llm_fallback": any(
                    u > self.llm_fallback_threshold
                    for u in uncertainty_by_dim.values()
                ),
                "final_source": source,
                # "steps_used": len(refinement_trace) if refined else 0,
                # "converged": refinement_trace[-1] - refinement_trace[0] < 0.05 if refined else None
            }

            # Log to database    # Inside run() after processing a document
            event_id = self._log_to_database(result_entry)
            event_ids.append(event_id)
            self.logger.log("ScoringEvent", result_entry)
            results.append(result_entry)

            self.logger.log(
                "ScoringPolicyCompleted",
                {
                    "document_id": scorable.id,
                    "final_scores": final_scores,
                    "source": source,
                },
            )

        context[self.output_key] = results
        context["event_ids"] = event_ids

        self.plot_score_distributions(results)
        self.generate_summary_report(results)
        return context

    def calculate_agreement(self, results):
        """Compare MRQ and LLM scores where both used"""
        import pandas as pd

        llm_mrq = []
        for result in results:
            if result["used_llm_fallback"]:
                for dim in self.dimensions:
                    llm_mrq.append(
                        {
                            "dimension": dim,
                            "mrq_score": result["mrq_scores"][dim],
                            "llm_score": result["scores"][dim],
                        }
                    )
        return pd.DataFrame(llm_mrq)

    # Helper method
    def _log_to_database(self, entry):
        # Insert into scoring_events table
        insert_event_sql = """
        INSERT INTO scoring_events (
            document_id,
            goal_text,
            original_text,
            refined_text,
            final_source,
            used_refinement,
            refinement_steps,
            used_llm_fallback
        )
        VALUES (
            :document_id,
            :goal_text,
            :original_text,
            :refined_text,
            :final_source,
            :used_refinement,
            :refinement_steps,
            :used_llm_fallback
        )
        RETURNING id
        """

        event_params = {
            "document_id": entry["document_id"],
            "goal_text": entry.get("goal_text", "UNKNOWN"),
            "original_text": entry.get("original_text"),
            "refined_text": entry.get("refined_text"),
            "final_source": entry["final_source"],
            "used_refinement": entry["used_refinement"],
            "refinement_steps": entry.get("steps_used", 0),
            "used_llm_fallback": entry["used_llm_fallback"],
        }

        event_id = (
            self.memory.session.execute(text(insert_event_sql), event_params)
            .fetchone()
            .id
        )

        # Insert into scoring_dimensions table
        insert_dim_sql = """
        INSERT INTO scoring_dimensions (
            event_id,
            dimension,
            mrq_score,
            ebt_energy,
            uncertainty_score,
            final_score
        )
        VALUES (
            :event_id,
            :dimension,
            :mrq_score,
            :ebt_energy,
            :uncertainty_score,
            :final_score
        )
        """

        for dim in self.dimensions:
            dim_params = {
                "event_id": event_id,
                "dimension": dim,
                "mrq_score": entry["mrq_scores"].get(dim),
                "ebt_energy": entry["ebt_energy"].get(dim),
                "uncertainty_score": entry["uncertainty_by_dimension"].get(
                    dim
                ),
                "final_score": entry["mrq_scores"].get(dim),
            }
            self.memory.session.execute(text(insert_dim_sql), dim_params)

        return event_id

    # In scoring_policy.py
    def plot_uncertainty_map(self, uncertainties, doc_id):
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 4))
        sns.heatmap(
            [uncertainties],  # single row for this document
            annot=True,
            cmap="YlOrRd",
            yticklabels=[doc_id],
            xticklabels=self.dimensions,
        )
        plt.title("Uncertainty Across Dimensions")
        plt.tight_layout()
        plt.savefig(f"uncertainty_maps/{doc_id}.png")
        plt.close()

        # After processing all documents

    def plot_score_distributions(self, results):
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        df = pd.DataFrame(
            [
                {
                    "dimension": dim,
                    "source": result["final_source"],
                    "score": result["mrq_scores"][dim],
                }
                for result in results
                for dim in self.dimensions
            ]
        )

        plt.figure(figsize=(12, 6))
        sns.violinplot(x="dimension", y="score", hue="source", data=df)
        plt.xticks(rotation=45)
        plt.title("Score Distributions by Dimension and Source")
        plt.savefig("score_distributions.png")
        plt.close()

    def generate_summary_report(self, results):
        import json

        total = len(results)
        refined_count = sum(1 for r in results if r["used_refinement"])
        llm_fallback_count = sum(1 for r in results if r["used_llm_fallback"])

        summary = {
            "total_documents": total,
            "refined_documents": refined_count,
            "llm_fallback_rate": llm_fallback_count / total,
            "average_uncertainty": {
                dim: sum(r["uncertainty_by_dimension"][dim] for r in results)
                / total
                for dim in self.dimensions
            },
            "dimension_refinement_rate": {
                dim: sum(
                    1
                    for r in results
                    if r["uncertainty_by_dimension"][dim]
                    > self.ebt_refine_threshold
                )
                / total
                for dim in self.dimensions
            },
        }

        # Save to file
        with open("policy_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def analyze_refinement_impact(self, results):
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        impacts = []
        for result in results:
            if result["used_refinement"]:
                for dim in self.dimensions:
                    impact = result["scores"][dim] - result["mrq_scores"][dim]
                    impacts.append(
                        {
                            "dimension": dim,
                            "impact": impact,
                            "before": result["mrq_scores"][dim],
                            "after": result["scores"][dim],
                        }
                    )

        # Calculate average impact
        impacts_df = pd.DataFrame(impacts)
        avg_impact = impacts_df.groupby("dimension")["impact"].mean()

        print("Average Score Improvement from Refinement:")
        print(avg_impact)

        # Plot improvement distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="dimension", y="impact", data=impacts_df)
        plt.title("Refinement Impact by Dimension")
        plt.xticks(rotation=45)
        plt.savefig("refinement_impact.png")
        plt.close()

    def export_results(self, results):
        """Ex Where is the training myself port results to CSV for external analysis"""
        import pandas as pd

        rows = []
        for result in results:
            base_row = {
                "document_id": result["document_id"],
                "used_refinement": result["used_refinement"],
                "used_llm_fallback": result["used_llm_fallback"],
            }

            # Add per-dimension data
            for dim in self.dimensions:
                base_row[f"{dim}_uncertainty"] = result[
                    "uncertainty_by_dimension"
                ][dim]
                base_row[f"{dim}_score"] = result["scores"][dim]
                base_row[f"{dim}_energy"] = result["ebt_energy"][dim]

            rows.append(base_row)

        # Save to CSV
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv("scoring_results.csv", index=False)
        return df

    # In your self-tuning agent
    def update_ebt(self):
        """Periodically update EBT models from refinement history"""
        examples = self._fetch_recent_refinements()
        if examples:
            trainer = EBTRefinementTrainer(self.cfg.ebt_refinement)
            trainer.run(examples)
            self.logger.log("EBTModelRetrained", {
                "total_examples": len(examples),
                "dimensions": list(set(e["dimension"] for e in examples))
            })

    def extract_srft_examples(self, scoring_events: list[dict]) -> list[dict]:
        return [{
            "context": e["context"],
            "original": e["original"],
            "refined": e["refined"],
            "dimension": e["dimension"],
            "original_score": e["original_mrq_score"],
            "refined_score": e["final_mrq_score"],
            "original_energy": e["original_ebt_energy"],
            "refined_energy": e["final_ebt_energy"],
            "llm_score": e.get("llm_score"),
            "uncertainty": e.get("original_ebt_energy"),
        } for e in scoring_events if "refined" in e]


    def score_with_memcube(self, goal: str, memcube: MemCube):
        # Check access policy
        if not memcube.apply_governance(self.user, "read"):
            raise PermissionError(f"User {self.user} cannot read MemCube {memcube.id}")
        
        # Get raw Scorable
        scorable = memcube.scorable
        
        # Score using EBT/MRQ
        scores = self.ebt.score(goal, scorable.text)
        mrq_scores = self.mrq.score(goal, scorable.text)
        
        # Decide which scores to use
        if any(u > self.cfg.get("ebt_refine_threshold", 0.7) for u in scores["uncertainty_by_dimension"].values()):
            refined = self.refine_document(goal, memcube)
            refined_scores = self.ebt.score(goal, refined.scorable.text)
            return refined_scores
        return scores
    

    def _log_memcubes_for_srft(self, scorable, refined_text, goal_text, mrq_scores, llm_scores, ebt_energy, uncertainty_by_dim, refined):
        for dim in self.dimensions:
            memcube = MemCube.from_dict({           
                "scorable": scorable.to_dict(),
                "dimension": dim,
                "memcube_type": "refinement",
                "context": goal_text,
                "original": scorable.text,
                "refined": refined_text if refined else scorable.text,
                "original_score": ebt_energy.get(dim),
                "refined_score": mrq_scores.get(dim) if refined else None,
                "llm_score": llm_scores.get(dim) if self.source == "llm" else None,
                "uncertainty": uncertainty_by_dim.get(dim),
                "source": self.name
            })
            self.memory.memcube.save_memcube(memcube)