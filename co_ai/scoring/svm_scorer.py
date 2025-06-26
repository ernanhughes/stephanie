# co_ai/scoring/svm_scorer.py

from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from co_ai.scoring.base_scorer import BaseScorer
from co_ai.scoring.score_bundle import ScoreBundle
from co_ai.scoring.score_result import ScoreResult
from co_ai.scoring.transforms.regression_tuner import RegressionTuner
import os
import json
import torch

class SVMScorer(BaseScorer):
    def __init__(self, cfg: dict, memory, logger, dimensions=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.dimensions = dimensions or ["alignment"]
        self.models = {dim: SVR() for dim in self.dimensions}
        self.scalers = {dim: StandardScaler() for dim in self.dimensions}
        self.trained = {dim: False for dim in self.dimensions}
        self.regression_tuners = {}  
        for dim in self.dimensions:
            self._initialize_dimension(dim)

    def _initialize_dimension(self, dim):
        self.models[dim] = SVR()
        self.scalers[dim] = StandardScaler()
        self.trained[dim] = False
        self.regression_tuners[dim] = RegressionTuner(dimension=dim, logger=self.logger)

    def train(self, samples_by_dim: dict[str, list[dict]]):
        """
        Train per-dimension SVM from labeled LLM/MRQ training data
        """
        for dim, samples in samples_by_dim.items():
            x = []
            y = []
            for sample in samples:
                prompt = sample["prompt"]
                hyp = sample["output"]
                score = sample["value"]
                feat = self._build_feature_vector({"goal_text": prompt}, {"text": hyp})
                x.append(feat)
                y.append(score)

            x = np.array(x)
            y = np.array(y)
            self.scalers[dim].fit(x)
            x_scaled = self.scalers[dim].transform(x)

            self.models[dim].fit(x_scaled, y)
            self.trained[dim] = True

            self.logger.log("SVMTrainingComplete", {
                "dimension": dim,
                "samples": len(samples),
                "score_min": float(np.min(y)),
                "score_max": float(np.max(y)),
            })

    def _build_feature_vector(self, goal: dict, hypothesis: dict):
        """
        Basic feature vector: concat prompt + hypothesis embeddings + MRQ raw score (if available)
        """
        emb_goal = self.memory.embedding.get_or_create(goal["goal_text"])
        emb_hyp = self.memory.embedding.get_or_create(hypothesis["text"])
        vec = emb_goal + emb_hyp

        # Optional MRQ bridge feature
        mrq = self.memory.score.find_by_text_and_dimension(
            hypothesis["text"], dimension="alignment", source="mrq"
        )
        if mrq:
            vec.append(mrq.score / 100.0)  # normalized to [0,1]
        else:
            vec.append(0.5)  # neutral if no MRQ score

        return vec

    def train_from_database(self, cfg:dict):
        pair_samples = self.memory.mrq.get_training_pairs_by_dimension()
        samples_by_dim = self.convert_mrq_pairs_to_supervised_examples(pair_samples)

        for dim, examples in samples_by_dim.items():
            self.train_for_dimension(dim, examples)


    def convert_mrq_pairs_to_supervised_examples(self, pair_samples: list[dict]) -> dict[str, list[dict]]:
        """
        Converts MRQ-style contrastive training pairs into a flat list of (prompt, output, value)
        entries per dimension, suitable for supervised regression training.
        """
        per_dimension = defaultdict(list)
        for pair in pair_samples:
            dim = pair.get("dimension", "default")

            for side in ["a", "b"]:
                output = pair.get(f"output_{side}")
                score = pair.get(f"value_{side}")
                if output is not None and score is not None:
                    per_dimension[dim].append({
                        "prompt": pair["prompt"],
                        "output": output,
                        "value": score
                    })

        self.logger.log("SVMConvertedMRQPacks", {
            "dimensions": list(per_dimension.keys()),
            "total_samples": sum(len(v) for v in per_dimension.values())
        })

        return per_dimension

    def train_for_dimension(self, dimension: str, examples: list[dict]):
        X = []
        y = []
        for ex in examples:
            prompt_vec = self.memory.embedding.get_or_create(ex["prompt"])
            output_vec = self.memory.embedding.get_or_create(ex["output"])
            pair_vec = np.array(prompt_vec + output_vec)
            X.append(pair_vec)
            y.append(ex["value"])

        X = np.array(X)
        y = np.array(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = SVR(kernel="linear")  # you can adjust kernel if needed
        model.fit(X_scaled, y)

        self.models[dimension] = (scaler, model)

        self.logger.log("SVMModelTrained", {
            "dimension": dimension,
            "num_samples": len(y)
        })

    def score(self, goal: dict, hypothesis: dict, dimensions: list[str]) -> ScoreBundle:
        results = {}
        for dim in dimensions:
            vec = self._build_feature_vector(goal, hypothesis)

            # Dynamic training if needed
            if not self.trained[dim]:
                self._try_train_on_dimension(dim)

            if not self.trained[dim]:
                score = 50.0
                rationale = f"SVM not trained for {dim}, returning neutral."
            else:
                x = self.scalers[dim].transform([vec])
                raw_score = self.models[dim].predict(x)[0]
                tuned_score = self.regression_tuners[dim].transform(raw_score)
                score = tuned_score
                rationale = f"SVM predicted and aligned score for {dim}"

            self.logger.log("SVMScoreComputed", {
                "dimension": dim,
                "score": score,
                "hypothesis": hypothesis.get("text"),
            })

            results[dim] = ScoreResult(
                dimension=dim,
                score=score,
                rationale=rationale,
                weight=1.0,
                source="svm",
            )

        return ScoreBundle(results=results)

    def _try_train_on_dimension(self, dim):
        samples_by_dim = self.memory.mrq.get_training_pairs_by_dimension()
        samples = samples_by_dim.get(dim, [])
        if not samples:
            self.logger.log("SVMNoSamples", {"dimension": dim})
            return

        X, y = [], []
        for s in samples:
            for side in ["a", "b"]:
                prompt = s["prompt"]
                hypothesis = s[f"output_{side}"]
                llm_score = s.get(f"value_{side}")
                if prompt and hypothesis and llm_score is not None:
                    vec = self._build_feature_vector({"goal_text": prompt}, {"text": hypothesis})
                    X.append(vec)
                    y.append(llm_score)
                    self.regression_tuners[dim].add_example(llm_score, llm_score)  # no-op, self-alignment fallback

        if not X:
            return

        X_scaled = self.scalers[dim].fit_transform(X)
        self.models[dim].fit(X_scaled, y)
        self.trained[dim] = True

        self.logger.log("SVMTrainingComplete", {
            "dimension": dim,
            "samples": len(X)
        })

        # Align the scores using same logic as MRQ
        self._align_with_llm(samples, dim)

    def _align_with_llm(self, samples, dim):
        for s in samples:
            for side in ["a", "b"]:
                prompt = s["prompt"]
                hypothesis = s[f"output_{side}"]
                llm_score = s.get(f"value_{side}")
                if llm_score is None:
                    continue

                vec = self._build_feature_vector({"goal_text": prompt}, {"text": hypothesis})
                x = self.scalers[dim].transform([vec])
                raw_score = self.models[dim].predict(x)[0]

                self.regression_tuners[dim].train_single(mrq_score=raw_score, llm_score=llm_score)

                self.logger.log("SVMAlignmentDynamic", {
                    "dimension": dim,
                    "mrq_score": raw_score,
                    "llm_score": llm_score
                })    


    def _train_dimension(self, dim: str):
        pairs_by_dim = self.memory.mrq.get_training_pairs_by_dimension()
        samples = pairs_by_dim.get(dim, [])
        if not samples:
            self.logger.log("SVMNoTrainingData", {"dimension": dim})
            self.trained[dim] = False
            return

        X = []
        y = []
        for sample in samples:
            goal = {"goal_text": sample["prompt"]}
            for side in ["a", "b"]:
                hyp = {"text": sample[f"output_{side}"]}
                label = sample.get(f"value_{side}")
                if label is not None:
                    vec = self._build_feature_vector(goal, hyp)
                    X.append(vec)
                    y.append(label)

        if len(X) < 5:
            self.logger.log("SVMInsufficientTrainingData", {"dimension": dim, "count": len(X)})
            self.trained[dim] = False
            return

        X_scaled = self.scalers[dim].fit_transform(X)
        self.models[dim].fit(X_scaled, y)
        self.trained[dim] = True
        self.logger.log("SVMTrained", {"dimension": dim, "samples": len(X)})

    def save_models(self):
        base_dir = self.cfg.get("scoring", {}).get("model_dir", "models/mrq/")
        os.makedirs(base_dir, exist_ok=True)

        for dim, (encoder, predictor) in self.models.items():
            dim_dir = os.path.join(base_dir, dim)
            os.makedirs(dim_dir, exist_ok=True)

            torch.save(encoder.state_dict(), os.path.join(dim_dir, "encoder.pt"))
            torch.save(predictor.state_dict(), os.path.join(dim_dir, "predictor.pt"))

            # Save tuner weights
            self.regression_tuners[dim].save(os.path.join(dim_dir, "tuner.json"))

            # Save metadata
            meta = {
                "min_score": self.min_score_by_dim[dim],
                "max_score": self.max_score_by_dim[dim],
            }
            with open(os.path.join(dim_dir, "meta.json"), "w") as f:
                json.dump(meta, f)