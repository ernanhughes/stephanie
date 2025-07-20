# stephanie/scoring/mrq/core_scoring.py
import torch

from stephanie.models.score import ScoreORM
from stephanie.models.sharpening_prediction import SharpeningPredictionORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.scoring_manager import ScoringManager


class MRQCoreScoring:
    def evaluate(self, prompt, response):
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

    def judge(self, goal, prompt, output_a, output_b):
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

    def predict_score_from_prompt(self, prompt, dimension="mrq", top_k=5):
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

    def estimate_score(self, goal, scorable, dimension):
        """
        Core logic: compute embeddings, run prediction, apply optional regression tuner.
        """
        if dimension not in self.models:
            self.initialize_dimension(dimension)

        raw_score = self.models[dimension].predict(
            goal.get("goal_text"),
            scorable.text
        )

        norm_score = self.normalize_score(raw_score, dimension)

        tuner = self.regression_tuners.get(dimension)
        if tuner:
            tuned = tuner.transform(norm_score)
            tuned = max(
                self.min_score_by_dim.get(dimension, 0.0),
                min(self.max_score_by_dim.get(dimension, 100.0), tuned),
            )
            self.logger.log(
                "MRQTunedScore",
                {
                    "dimension": dimension,
                    "raw": norm_score,
                    "tuned": tuned,
                },
            )
            return tuned

        return norm_score
