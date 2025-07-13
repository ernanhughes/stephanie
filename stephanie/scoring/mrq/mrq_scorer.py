# stephanie/scoring/mrq/mrq_scorer.py
import json
import os
import re

import torch

from stephanie.evaluator.hypothesis_value_predictor import \
    HypothesisValuePredictor
from stephanie.evaluator.mrq_trainer import MRQTrainer
from stephanie.models.sharpening_prediction import SharpeningPredictionORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.model import MRQModel
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class MRQScorer(BaseScorer):
    def __init__(self, cfg: dict, memory, logger, dimensions=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.device = cfg.get("device", "cpu")

        self.dimensions = dimensions or ["mrq"]
        self.models = {}  # dim -> (encoder, predictor)
        self.trainers = {}
        self.min_score_by_dim = {}
        self.max_score_by_dim = {}
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)
        self.encoder = TextEncoder().to(self.device)
        self.regression_tuners = {}

        # Initialize model + tuner for each dimension
        for dim in self.dimensions:
            self.regression_tuners[dim] = RegressionTuner(
                dimension=dim, logger=self.logger
            )
            trainer = MRQTrainer(
                memory=memory,
                logger=logger,
                value_predictor=self.value_predictor,
                encoder=self.encoder,
                device=self.device,
            )
            self.models[dim] = (self.encoder, self.value_predictor)
            self.trainers[dim] = trainer
            self.min_score_by_dim[dim] = 0.0
            self.max_score_by_dim[dim] = 1.0

    @property
    def name(self) -> str:
        return "mrq"

    def score(
        self, goal: dict, scorable: Scorable, dimensions: list[str]
    ) -> ScoreBundle:
        """
        Predicts scores for given dimensions using MR.Q and applies tuning if available.
        """
        results = []
        for dimension_name in dimensions:
            score = self._estimate_score(goal, scorable, dimension_name)
            rationale = f"MRQ estimated score for {dimension_name}."
            self.logger.log(
                "MRQDimensionEvaluated",
                {"dimension": dimension_name, "score": score, "rationale": rationale},
            )
            results.append(
                ScoreResult(
                    dimension=dimension_name,
                    score=score,
                    rationale=rationale,
                    weight=1.0,
                    source="mrq",
                    target_type=scorable.target_type,  # Default target for MR.Q scores
                )
            )
        return ScoreBundle(results={r.dimension: r for r in results})

    def _estimate_score(self, goal: dict, scorable: Scorable, dimension: str) -> float:
        """
        Core logic: compute embeddings, run prediction, apply optional regression tuner.
        """
        # Initialize dimension on demand
        if dimension not in self.models:
            self._initialize_dimension(dimension)

        raw_score = self.models[dimension].predict(
            goal.get("goal_text"),
            scorable.text,
            embedding_store=self.memory.embedding,
        )

        norm_score = self.normalize_score(raw_score, dimension)

        # Optionally apply tuner
        tuner = self.regression_tuners.get(dimension)
        if tuner:
            tuned = tuner.transform(norm_score)
            min_score = self.min_score_by_dim.get(dimension, 0.0)
            max_score = self.max_score_by_dim.get(dimension, 100.0)
            tuned = max(min_score, min(max_score, tuned))  # clamp to [0, 100]
            self.logger.log(
                "MRQTunedScore",
                {"dimension": dimension, "raw": norm_score, "tuned": tuned},
            )
            return tuned
        return norm_score

    def _initialize_dimension(self, dimension):
        self.regression_tuners[dimension] = RegressionTuner(
            dimension=dimension, logger=self.logger
        )
        self.trainers[dimension] = MRQTrainer(
            memory=self.memory,
            logger=self.logger,
            value_predictor=self.value_predictor,
            encoder=self.encoder,
            device=self.device,
        )
        self.models[dimension] = MRQModel(
            self.encoder,
            self.value_predictor,
            self.memory.embedding,
            device=self.device,
        )
        self.min_score_by_dim[dimension] = 0.0
        self.max_score_by_dim[dimension] = 100.0
        self.logger.log("MRQModelInitializing", {"dimension": dimension})

    def align_to_best_llm_neighbour(self, goal, hypothesis, dimension):
        """
        Fetch similar hypotheses that already have high LLM scores.
        Then align MR.Q prediction to the best of them.
        """
        llm_scores = self.get_closest_llm_scores(hypothesis["text"], dimension)
        if llm_scores:
            self.align_with_llm_score(dimension, goal, hypothesis, max(llm_scores))

    def get_closest_llm_scores(
        self, hypothesis_text: str, dimension: str, top_k: int = 5
    ) -> list[float]:
        """
        Finds the top_k LLM scores for hypotheses most similar to the given one.
        """
        query_emb = self.memory.embedding.get_or_create(hypothesis_text)
        similar_items = self.memory.embedding.search_similar_prompts_with_scores(
            query_emb, top_k
        )

        scores = []
        for item in similar_items:
            matched_text = item.get("text")
            score_entry = self.memory.score.find_by_text_and_dimension(
                matched_text, dimension=dimension, source="llm"
            )
            if score_entry:
                scores.append(score_entry.score)
        return scores

    def align_with_llm_score(self, dimension, goal, hypothesis, llm_score):
        mrq_score = self._estimate_score(goal, hypothesis, dimension)
        self.logger.log(
            "MRQAligningToLLM",
            {
                "goal": goal.get("goal_text"),
                "hypothesis": hypothesis.get("text"),
                "dimension": dimension,
                "mrq_raw": mrq_score,
                "llm_target": llm_score,
            },
        )
        self.regression_tuners[dimension].add_example(mrq_score, llm_score)
        self.logger.log(
            "MRQAlignmentAdded",
            {
                "dimension": dimension,
                "example_count": len(self.regression_tuners[dimension].examples),
            },
        )

    def evaluate(self, prompt: str, response: str) -> ScoreBundle:
        """
        Scores a prompt-response pair across all dimensions, and saves it.
        """
        results = []
        for dim, (encoder, predictor) in self.models.items():
            prompt_emb = torch.tensor(
                self.memory.embedding.get_or_create(prompt), device=self.device
            ).unsqueeze(0)
            output_emb = torch.tensor(
                self.memory.embedding.get_or_create(response), device=self.device
            ).unsqueeze(0)
            zsa = encoder(prompt_emb, output_emb)
            value = predictor(zsa).item()
            norm_score = self.normalize_score(value, dim)

            results.append(
                ScoreResult(
                    dimension=dim,
                    score=norm_score,
                    weight=1.0,
                    rationale=f"MR.Q model trained for {dim}",
                    source="mrq",
                )
            )

        bundle = ScoreBundle(results={r.dimension: r for r in results})
        ScoringManager.save_score_to_memory(
            bundle,
            response,
            cfg=self.cfg,
            memory=self.memory,
            logger=self.logger,
            source="mrq",
        )
        return bundle

    def normalize_score(self, raw, dim):
        return round(raw, 2)

    def judge(self, goal, prompt, output_a, output_b):
        """
        Compares two outputs via MR.Q and returns the preferred one.
        """
        dim = self.dimensions[0]
        model = self.models[dim]
        encoder = model.encoder
        predictor = model.predictor

        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt), device=self.device
        ).unsqueeze(0)
        a_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_a), device=self.device
        ).unsqueeze(0)
        b_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_b), device=self.device
        ).unsqueeze(0)

        value_a = predictor(encoder(prompt_emb, a_emb)).item()
        value_b = predictor(encoder(prompt_emb, b_emb)).item()
        preferred = output_a if value_a >= value_b else output_b

        # Optionally log sharpening example
        if self.memory.mrq.log_evaluations():
            pred = SharpeningPredictionORM(
                id=None,
                goal_id=-1,
                prompt_text=prompt,
                output_a=output_a,
                output_b=output_b,
                preferred="a" if value_a >= value_b else "b",
                predicted="a" if value_a >= value_b else "b",
                value_a=value_a,
                value_b=value_b,
            )
            self.memory.sharpening.insert_sharpening_prediction(pred.to_dict(), goal)

        return preferred, {"value_a": value_a, "value_b": value_b}

    def train_from_database(self, cfg: dict):
        all_samples = self.memory.mrq.get_training_pairs_by_dimension()
        for dim, samples in all_samples.items():
            if not samples:
                self.logger.log("MRQNoTrainingSamples", {"dimension": dim})
                continue

            self.align_mrq_with_llm_scores_from_pairs(samples, dimension=dim)

            self.logger.log(
                "MRQTrainingStart", {"dimension": dim, "sample_count": len(samples)}
            )

            if dim not in self.trainers:
                self.trainers[dim] = MRQTrainer(
                    memory=self.memory,
                    logger=self.logger,
                    encoder=self.encoder,
                    value_predictor=self.value_predictor,
                    device=self.device,
                )

            self.update_score_bounds_from_data(samples, dim)
            dataloader = self.trainers[dim].prepare_training_data(samples)
            self.trainers[dim].train(dataloader, cfg)

            self.logger.log("MRQTrainingComplete", {"dimension": dim})

    def train_from_context(self, context: dict, cfg: dict):
        dim_samples = context.get("mrq_training_pairs_by_dimension", {})
        for dim, samples in dim_samples.items():
            if not samples:
                self.logger.log("MRQNoTrainingFromContext", {"dimension": dim})
                continue

            self.logger.log(
                "MRQContextTrainingStart",
                {"dimension": dim, "sample_count": len(samples)},
            )

            self.update_score_bounds_from_data(samples, dim)
            dataloader = self.trainers[dim].prepare_training_data(samples)
            self.trainers[dim].train(dataloader, cfg)

            self.logger.log("MRQContextTrainingComplete", {"dimension": dim})

    def update_score_bounds_from_data(self, samples: list, dim: str):
        values = []
        for s in samples:
            if "value_a" in s and "value_b" in s:
                values.extend([s["value_a"], s["value_b"]])
            elif "value" in s:
                values.append(s["value"])
        if values:
            min_score = min(values)
            max_score = max(values)
            self.min_score_by_dim[dim] = min_score
            self.max_score_by_dim[dim] = max_score
            self.logger.log(
                "MRQScoreBoundsUpdated",
                {
                    "dimension": dim,
                    "min_score": min_score,
                    "max_score": max_score,
                    "example_count": len(values),
                },
            )

    def align_mrq_with_llm_scores_from_pairs(
        self, pair_samples: list[dict], dimension: str, log_prefix: str = "MRQAlignment"
    ):
        for pair in pair_samples:
            prompt = pair["prompt"]
            for side in ["a", "b"]:
                hyp = pair[f"output_{side}"]
                llm_score = pair[f"value_{side}"]

                # Predict MRQ score dynamically
                mrq_score = self.score(
                    {"goal_text": prompt}, Scorable(text=hyp), [dimension]
                )

                # Log the alignment
                self.logger.log(
                    f"{log_prefix}Dynamic",
                    {
                        "prompt_hash": hash(prompt),
                        "hypothesis_hash": hash(hyp),
                        "dimension": dimension,
                        "llm_score": llm_score,
                        "predicted_mrq": mrq_score,
                    },
                )

                # Pass the pair into the regression tuner
                if mrq_score is not None and llm_score is not None:
                    self.regression_tuners[dimension].train_single(
                        mrq_score=mrq_score.results[dimension].score,
                        llm_score=llm_score,
                    )

    def save_models(self):
        base_dir = self.cfg.get("scoring", {}).get("model_dir", "models/mrq/")
        os.makedirs(base_dir, exist_ok=True)

        for dim, (encoder, predictor) in self.models.items():
            dim_dir = os.path.join(base_dir, dim)
            os.makedirs(dim_dir, exist_ok=True)

            # Save encoder and predictor
            torch.save(encoder.state_dict(), os.path.join(dim_dir, "encoder.pt"))
            torch.save(predictor.state_dict(), os.path.join(dim_dir, "predictor.pt"))

            # Save tuner
            self.regression_tuners[dim].save(os.path.join(dim_dir, "tuner.json"))

            # Save metadata
            meta = {
                "min_score": self.min_score_by_dim[dim],
                "max_score": self.max_score_by_dim[dim],
            }
            with open(os.path.join(dim_dir, "meta.json"), "w") as f:
                json.dump(meta, f)

            self.logger.log("MRQModelSaved", {"dimension": dim, "path": dim_dir})

    def get_available_mrq_dimensions_flat(
        self, model_dir: str, prefix="document_rm_"
    ) -> list[str]:
        dimensions = set()
        pattern = re.compile(rf"{re.escape(prefix)}(.+?)\.(pt|json)")
        for filename in os.listdir(model_dir):
            match = pattern.match(filename)
            if match:
                dimensions.add(
                    match.group(1).replace("_tuner", "")
                )  # normalize tuner file
        return sorted(dimensions)

    def load_models(self):
        base_dir = self.cfg.get("scoring", {}).get("model_dir", "models/mrq/")

        if not os.path.exists(base_dir):
            self.logger.log("MRQModelDirNotFound", {"path": base_dir})
            return

        # Get all subdirectories as dimensions
        self.dimensions = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        for dim in self.dimensions:
            dim_dir = os.path.join(base_dir, dim)

            # Ensure dimension is initialized
            if dim not in self.models:
                self._initialize_dimension(dim)

            model = self.models[dim]
            encoder = model.encoder
            predictor = model.predictor

            try:
                # Load encoder and predictor
                encoder_path = os.path.join(dim_dir, "encoder.pt")
                predictor_path = os.path.join(dim_dir, "predictor.pt")
                if not os.path.exists(encoder_path) or not os.path.exists(
                    predictor_path
                ):
                    self.logger.log("MRQModelFilesMissing", {"dimension": dim})
                    continue

                encoder.load_state_dict(
                    torch.load(encoder_path, map_location=self.device)
                )
                predictor.load_state_dict(
                    torch.load(predictor_path, map_location=self.device)
                )

                # Load tuner
                tuner_path = os.path.join(dim_dir, "tuner.json")
                if os.path.exists(tuner_path):
                    self.regression_tuners[dim].load(tuner_path)

                # Load metadata
                meta_path = os.path.join(dim_dir, "meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                        self.min_score_by_dim[dim] = meta.get("min_score", 0.0)
                        self.max_score_by_dim[dim] = meta.get("max_score", 100.0)

                self.logger.log("MRQModelLoaded", {"dimension": dim})

            except Exception as e:
                self.logger.log(
                    "MRQModelLoadError", {"dimension": dim, "error": str(e)}
                )

    def load_models_with_path(self):
        base_dir = self.cfg.get("scoring", {}).get("model_dir", "models/mrq/")

        for dim in self.dimensions:
            dim_dir = os.path.join(base_dir, dim)
            if not os.path.exists(dim_dir):
                self.logger.log("MRQLoadMissing", {"dimension": dim})
                continue

            model = self.models[dim]
            encoder = model.encoder
            predictor = model.predictor

            encoder.load_state_dict(torch.load(os.path.join(dim_dir, "encoder.pt")))
            predictor.load_state_dict(torch.load(os.path.join(dim_dir, "predictor.pt")))

            self.regression_tuners[dim].load(os.path.join(dim_dir, "tuner.json"))

            with open(os.path.join(dim_dir, "meta.json")) as f:
                meta = json.load(f)
                self.min_score_by_dim[dim] = meta["min_score"]
                self.max_score_by_dim[dim] = meta["max_score"]

            self.logger.log("MRQModelLoaded", {"dimension": dim})

    def predict_score_from_prompt(
        self, prompt: str, dimension: str = "mrq", top_k: int = 5
    ) -> float:
        """
        Predicts a score for a new prompt using MR.Q-style reverse scoring.
        Works with flattened row data: one (score, dimension, source) per row.
        """
        try:
            nearest = self.memory.embedding.search_similar_prompts_with_scores(
                prompt, top_k=top_k
            )

            llm_scores = []
            mrq_scores = []

            for item in nearest:
                if item.get("dimension") != dimension:
                    continue

                score = item.get("score")
                source = item.get("source")

                if score is None:
                    continue

                if source == "llm":
                    llm_scores.append(score)
                elif source == "mrq":
                    mrq_scores.append(score)

            if not llm_scores:
                if self.logger:
                    self.logger.log(
                        "MRQPromptScorerNoLLMScoresFound",
                        {"prompt": prompt[:100], "dimension": dimension},
                    )
                return 0.5

            avg_llm = sum(llm_scores) / len(llm_scores)

            # Apply regression tuner to MR.Q estimates if available
            if mrq_scores and dimension in self.regression_tuners:
                avg_mrq = sum(mrq_scores) / len(mrq_scores)
                aligned_score = self.regression_tuners[dimension].transform(avg_mrq)
            else:
                aligned_score = avg_llm

            final_score = max(0.0, min(1.0, aligned_score))

            if self.logger:
                self.logger.log(
                    "MRQPromptScorePredicted",
                    {
                        "prompt": prompt[:100],
                        "score": final_score,
                        "dimension": dimension,
                        "neighbors_found": len(nearest),
                        "used_alignment": bool(mrq_scores),
                    },
                )

            return final_score

        except Exception as e:
            if self.logger:
                self.logger.log(
                    "MRQPromptScoreError",
                    {"error": str(e), "prompt": prompt[:100], "dimension": dimension},
                )
            return 0.5

    def save_metadata(self, base_dir):
        for dim in self.dimensions:
            dim_dir = os.path.join(base_dir, dim)
            os.makedirs(dim_dir, exist_ok=True)
            meta_path = os.path.join(dim_dir, "meta.json")
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "min_score": self.min_score_by_dim.get(dim, 0.0),
                        "max_score": self.max_score_by_dim.get(dim, 1.0),
                    },
                    f,
                )
