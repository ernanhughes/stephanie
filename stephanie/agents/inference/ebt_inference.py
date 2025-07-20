# stephanie/agents/inference/document_ebt_inference.py
import os
from typing import Optional

import torch

from stephanie.agents.base_agent import BaseAgent
from stephanie.memcubes.memcube_factory import MemCubeFactory
from stephanie.models.score import ScoreORM
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_utils import (discover_saved_dimensions,
                                         get_model_path)


class EBTInferenceAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = "ebt"
        self.evaluator = "ebt"
        self.model_path = cfg.get("model_path", "models")
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")
        self.embedding_type = self.memory.embedding.type

        self.dimensions = cfg.get("dimensions", [])
        self.models = {}
        self.model_meta = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.dimensions:
            self.dimensions = discover_saved_dimensions(
                model_type=self.model_type, target_type=self.target_type
            )

        self.logger.log(
            "DocumentEBTInferenceAgentInitialized",
            {
                "model_type": self.model_type,
                "target_type": self.target_type,
                "dimensions": self.dimensions,
                "device": str(self.device),
            },
        )

        self.load_models(self.dimensions)
        self.logger.log("AllEBTModelsLoaded", {"dimensions": self.dimensions})

    def load_models(self, dimensions):
        """
        Load EBT models for specified dimensions.
        If dimensions are not provided, load all available models.
        """
        if not dimensions:
            dimensions = self.dimensions

        for dim in dimensions:
            if dim not in self.models:
                model_path = get_model_path(
                    self.model_path,
                    self.model_type,
                    self.target_type,
                    dim,
                    self.model_version,
                    self.embedding_type
                )
                infer_path = f"{model_path}/{dim}.pt"
                meta_path = f"{model_path}/{dim}.meta.json"

                self.logger.log("LoadingEBTModel", {"dimension": dim, "path": infer_path})
                model = self._load_model(infer_path)
                self.models[dim] = model

                if os.path.exists(meta_path):
                    self.model_meta[dim] = load_json(meta_path)
                else:
                    self.model_meta[dim] = {"min": 40, "max": 100}

    def _load_model(self, path):
        model = EBTModel().to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model


    def get_model_name(self) -> str:
        return f"{self.target_type}_{self.model_type}_{self.model_version}"

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        results = []

        for doc in context.get(self.input_key, []):
            doc_id = doc.get("id")
            self.logger.log("EBTScoringStarted", {"document_id": doc_id})
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            memcube = MemCubeFactory.from_scorable(scorable, version="auto")
            memcube.extra_data["pipeline"] = "ebt_inference"

            ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal_text)).to(
                self.device
            )
            doc_emb = torch.tensor(
                self.memory.embedding.get_or_create(scorable.text)
            ).to(self.device)

            dimension_scores = {}
            score_results = []

            for dim, model in self.models.items():
                with torch.no_grad():
                    raw_energy = model(ctx_emb, doc_emb).squeeze().cpu().item()
                    normalized_score = torch.sigmoid(torch.tensor(raw_energy)).item()
                    meta = self.model_meta.get(dim, {"min": 40, "max": 100})
                    real_score = (
                        normalized_score * (meta["max"] - meta["min"]) + meta["min"]
                    )
                    final_score = round(real_score, 4)
                    dimension_scores[dim] = final_score

                    normalized_score = torch.sigmoid(torch.tensor(raw_energy)).item()
                    confidence_penalty = 1 - abs(normalized_score - 0.5) * 2
                    uncertainty = confidence_penalty * (1 / (1 + abs(raw_energy)))


                    score_results.append(
                        ScoreResult(
                            dimension=dim,
                            score=final_score,
                            rationale=f"Energy={round(raw_energy, 4)}",
                            weight=1.0,
                            energy = raw_energy,     
                            uncertainty = uncertainty, 
                            source=self.name,
                            target_type=scorable.target_type,
                            prompt_hash=ScoreORM.compute_prompt_hash(goal_text, scorable),
                        )
                    )

                    self.logger.log(
                        "EBTScoreComputed",
                        {
                            "document_id": doc_id,
                            "dimension": dim,
                            "raw_energy": round(raw_energy, 4),
                            "final_score": final_score,
                        },
                    ) 

            score_bundle = ScoreBundle(results={r.dimension: r for r in score_results})

            ScoringManager.save_score_to_memory(
                score_bundle,
                scorable,
                context,
                self.cfg,
                self.memory,
                self.logger,
                source=self.model_type,
                model_name=self.get_model_name(),
            )

            results.append(
                {
                    "scorable": scorable.to_dict(),
                    "scores": dimension_scores,
                    "score_bundle": score_bundle.to_dict(),
                }
            )

            self.logger.log(
                "EBTScoringFinished",
                {
                    "document_id": doc_id,
                    "scores": dimension_scores,
                    "dimensions_scored": list(dimension_scores.keys()),
                },
            )

        context[self.output_key] = results
        self.logger.log(
            "EBTInferenceCompleted", {"total_documents_scored": len(results)}
        )
        return context

    def get_energy(self, goal: str, text: str) -> dict[str, float]:
        """
        Returns raw (unnormalized) energy scores per dimension for a document-goal pair.
        This is useful for estimating uncertainty.

        Returns:
            A dictionary like: { "relevance": 3.42, "accuracy": 5.13, ... }
        """
        ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal)).to(self.device)
        doc_emb = torch.tensor(self.memory.embedding.get_or_create(text)).to(self.device)

        energy_by_dimension = {}

        for dim in self.dimensions:
            model = self.models.get(dim)
            if model is None:
                self.logger.log("EBTModelMissing", {"dimension": dim})
                continue

            try:
                with torch.no_grad():
                    raw_energy = model(ctx_emb, doc_emb).squeeze().cpu().item()
                    energy_by_dimension[dim] = raw_energy
            except Exception as e:
                self.logger.log("EBTEnergyComputationError", {
                    "dimension": dim,
                    "error": str(e),
                    "goal": goal[:100],
                    "text": text[:100]
                })

        return energy_by_dimension

    def optimize(self, goal: str, text: str, dimension: str = None, 
                steps: int = 10, step_size: float = 0.1) -> dict:
        """
        Optimize document text with energy trace and uncertainty estimation
        
        Returns:
            dict: {
                "refined_text": str,
                "energy_trace": List[float],
                "uncertainty_trace": List[float],
                "converged": bool
            }
        """
        # Setup
        target_dim = dimension or next(iter(self.models.keys()))
        model = self.models[target_dim]
        
        # Get base embeddings
        ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal)).to(self.device)
        doc_emb = torch.tensor(self.memory.embedding.get_or_create(text)).to(self.device)
        
        # Make document embedding differentiable
        doc_tensor = doc_emb.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([doc_tensor], lr=step_size)
        
        energy_trace = []
        uncertainty_trace = []
        for _ in range(steps):
            optimizer.zero_grad()
            energy = model(ctx_emb, doc_tensor)
            energy.backward()
            
            # Save trace data
            energy_trace.append(energy.item())
            uncertainty_trace.append(torch.norm(doc_tensor.grad).item() if doc_tensor.grad is not None else 0.0)
            
            # Update document embedding
            optimizer.step()
        
        # Generate refined document
        refined_emb = doc_tensor.detach()
        refined_text = self._embedding_to_text(refined_emb, goal, text)
        
        # Final evaluation
        final_energy = model(ctx_emb, refined_emb).item()
        uncertainty = torch.norm(refined_emb.grad).item() if refined_emb.grad is not None else 0.0
        
        # Create new MemCube version
        from stephanie.memcubes.memcube_factory import MemCubeFactory
        scorable = Scorable(id=hash(refined_text), text=refined_text, target_type=TargetType.DOCUMENT)
        memcube = MemCubeFactory.from_scorable(scorable, version="auto")
        memcube.extra_data["refinement_trace"] = energy_trace
        memcube.sensitivity = "internal"  # Upgrade sensitivity on refinement
        
        # Save new version
        self.memory.memcube.save_memcube(memcube)

        return {
            "refined_text": refined_text,
            "energy_trace": [round(e, 4) for e in energy_trace],
            "uncertainty_trace": [round(u, 4) for u in uncertainty_trace],
            "final_energy": final_energy,
            "converged": abs(energy_trace[-1] - energy_trace[0]) < 0.05,
            "uncertainty": uncertainty,
            "dimension": target_dim,
            "steps_used": len(energy_trace)
        }
    
    def is_uncertain(self, goal: str, text: str, dimension: str, threshold: float = 0.75) -> bool:
        """
        Determine if energy-based prediction is uncertain.
        Uses energy magnitude and gradient stability.
        """
        target_dim = dimension or next(iter(self.models.keys()))
        
        # Get base energy
        base_energy = self.get_energy(goal, text)[target_dim]
        
        # Get energy after small perturbation
        ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal)).to(self.device)
        doc_emb = torch.tensor(self.memory.embedding.get_or_create(text)).to(self.device)
        
        doc_tensor = doc_emb.clone().detach().requires_grad_(True)
        doc_tensor.retain_grad()
        
        # Forward pass
        energy = self.models[dimension](ctx_emb, doc_tensor)
        energy.backward()
        
        # Calculate uncertainty from gradient magnitude
        grad_norm = torch.norm(doc_tensor.grad).item()
        
        self.logger.log("EBTUncertaintyEstimate", {
            "dimension": target_dim,
            "base_energy": round(base_energy, 4),
            "gradient_norm": round(grad_norm, 4),
            "uncertainty_score": round(abs(base_energy) + grad_norm, 4)
        })
        
        return abs(base_energy) > threshold or grad_norm > threshold
    

    def _embedding_to_text(self, embedding, original_goal, original_text):
        """
        Convert a refined embedding back to text.
        Currently uses nearest neighbor search; could be replaced with a generator model.
        """
        # Option 1: Nearest neighbor search in embedding store
        neighbors = self.memory.embedding.find_neighbors(embedding, k=5)
        
        # Option 2: Hybrid approach with LLM
        if self.cfg.get("use_llm_refinement", False):
            from stephanie.agents.inference import LLMGenerator
            llm = LLMGenerator(self.cfg, self.memory, self.logger)
            
            prompt = f"""Improve the following text to better align with this goal:
            
            Goal: {original_goal}
            Original Text: {original_text}**** ****
            
            Generate an improved version that maintains content while optimizing for alignment."""
            
            return llm.generate(prompt)
        
        # Fallback: Use nearest neighbor from embedding database
        return neighbors[0]
    

    def plot_refinement_trace(self, trace, title):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        plt.plot(trace, marker="o")
        plt.title(title)
        plt.xlabel("Optimization Step")
        plt.ylabel("Energy Value")
        plt.grid(True)
        
        # Save for analysis
        os.makedirs("energy_traces", exist_ok=True)
        plt.savefig(f"energy_traces/{title}.png")
        plt.close()

    def is_unstable(self, goal: str, text: str, dimension: Optional[str] = None) -> bool:
        target_dim = dimension or next(iter(self.models))
        model = self.models[target_dim]
        meta = self.model_meta.get(target_dim, {"min": 40, "max": 100})
        dim_threshold = self.cfg.get(f"{target_dim}_uncertainty_threshold", self.uncertainty_threshold)

        # Prepare embeddings
        ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal)).to(self.device)
        doc_emb = torch.tensor(self.memory.embedding.get_or_create(text)).to(self.device)
        doc_tensor = doc_emb.clone().detach().requires_grad_(True)

        # Forward pass
        energy = model(ctx_emb, doc_tensor).squeeze()
        
        # Normalize energy using known bounds
        normalized_energy = (energy.item() - meta["min"]) / (meta["max"] - meta["min"])

        # Compute gradient
        energy.backward()
        grad = doc_tensor.grad
        grad_norm = torch.norm(grad).item() if grad is not None else 0.0

        # Combine to get uncertainty
        uncertainty_score = normalized_energy + grad_norm

        self.logger.log("EBTUncertaintyCheck", {
            "dimension": target_dim,
            "energy": round(energy.item(), 4),
            "normalized_energy": round(normalized_energy, 4),
            "grad_norm": round(grad_norm, 4),
            "uncertainty_score": round(uncertainty_score, 4),
            "threshold": dim_threshold,
        })
