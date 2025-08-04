# stephanie/agents/unified_mrq.py
import os
import pickle
from collections import defaultdict

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.evaluator.mrq_trainer import MRQTrainer
from stephanie.models.unified_mrq import UnifiedMRQModelORM
from stephanie.utils.similarity_utils import compute_similarity_matrix


class UnifiedMRQAgent(BaseAgent):
    """
    Unified Multidimensional MR.Q Agent Right sure Much yep and a man get there Yeah yeah I'd love it but I can't guess
    - Collects scores across all pipelines and dimensions.
    - Builds contrastive training pairs.
    - Trains a multidimensional preference model.
    - Saves models and logs metadata to DB.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.target_dimensions = cfg.get(
            "target_dimensions", ["correctness", "originality", "clarity", "relevance"]
        )
        self.similarity_threshold = cfg.get("similarity_threshold", 0.85)
        self.top_k_similar = cfg.get("top_k_similar", 20)
        self.min_value_difference = cfg.get("min_value_difference", 10)
        self.output_dir = cfg.get("model_output_dir", "mrq_models")
        self.trainer = MRQTrainer(memory, logger)

    async def run(self, context: dict) -> dict:
        self.logger.log("UnifiedMRQStarted", {})

        # Step 1: Load hypotheses and scores
        hypotheses = self.get_hypotheses(context)
        if not hypotheses:
            self.logger.log("NoHypothesesFound", {})
            return context

        hypothesis_ids = [h["id"] for h in hypotheses]
        evaluations = self.memory.evaluations.get_by_hypothesis_ids(hypothesis_ids)
        evaluation_ids = [e.id for e in evaluations]
        scores = self.memory.scores.get_by_evaluation_ids(evaluation_ids)

        # Step 2: Embed and index hypotheses
        embedded = self._index_embeddings(hypotheses)

        print(f"Embedded: {[(k, v[1][:5]) for k, v in embedded.items()]}")

        # Step 3: Collect dimension-wise scores
        score_map = self._group_scores(scores)

        print("Score map keys:", list(score_map.keys()))
        print("Example score entry:", next(iter(score_map.items()), None))

        # Step 4: Generate contrast pairs
        contrast_pairs = self._generate_contrast_pairs(embedded, score_map, context)

        # Step 5: Train model per dimension
        trained_models = self.trainer.train_multidimensional_model(contrast_pairs)
        self.logger.log(
            "UnifiedMRQTrained",
            {
                "pair_count": len(contrast_pairs),
                "dimensions": list(trained_models.keys()),
            },
        )

        # Step 6: Save and log to DB
        os.makedirs(self.output_dir, exist_ok=True)
        for dim, model in trained_models.items():
            path = os.path.join(self.output_dir, f"{dim}_mrq.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)

            pair_count = len([p for p in contrast_pairs if p["dimension"] == dim])
            self.memory.session.add(
                UnifiedMRQModelORM(
                    dimension=dim,
                    model_path=path,
                    pair_count=pair_count,
                    trainer_version="v1.0",
                    context={
                        "similarity_threshold": self.similarity_threshold,
                        "min_value_diff": self.min_value_difference,
                    },
                )
            )

        self.memory.session.commit()
        self.logger.log(
            "UnifiedMRQModelsSaved", {"dimensions": list(trained_models.keys())}
        )
        context["unified_mrq_model_paths"] = {
            dim: os.path.join(self.output_dir, f"{dim}_mrq.pkl")
            for dim in trained_models
        }

        return context

    def _index_embeddings(self, hypotheses):
        index = {}
        for hyp in hypotheses:
            text = hyp.get("text")
            if not text:
                continue

            vector = self.memory.embedding.get_or_create(text)
            if vector is not None:
                index[hyp["id"]] = (hyp, np.array(vector))

        return index

    def _group_scores(self, scores):
        grouped = defaultdict(dict)
        for s in scores:
            hypothesis_id = getattr(s.evaluation, "hypothesis_id", None)
            if hypothesis_id and s.dimension:
                grouped[hypothesis_id][s.dimension] = s.score
        return grouped

    def _generate_contrast_pairs(
        self, embedded: dict, score_map: dict, context: dict
    ) -> list[dict]:
        """
        Given a map of hypothesis_id -> (hypothesis_dict, embedding), and a score_map,
        return all valid contrast pairs where two hypotheses have scores for the same dimensions.
        """
        contrast_pairs = []
        dim_seen = set()

        all_ids = list(embedded.keys())
        self.logger.log(
            "ContrastPairGenerationStart",
            {
                "total_hypotheses": len(all_ids),
                "score_map_keys": list(score_map.keys())[:10],
            },
        )

        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                id_a, id_b = all_ids[i], all_ids[j]

                if id_a not in score_map or id_b not in score_map:
                    continue

                scores_a = score_map[id_a]
                scores_b = score_map[id_b]

                shared_dims = set(scores_a.keys()) & set(scores_b.keys())

                for dim in shared_dims:
                    score_a = scores_a[dim]
                    score_b = scores_b[dim]

                    # Skip if scores are equal
                    if score_a == score_b:
                        continue

                    dim_seen.add(dim)

                    # Get embedding vectors
                    emb_a = embedded[id_a][1]
                    emb_b = embedded[id_b][1]

                    if emb_a is None or emb_b is None:
                        self.logger.log(
                            "MissingEmbeddingInContrast",
                            {"id_a": id_a, "id_b": id_b, "dim": dim},
                        )
                        continue

                    preferred = "a" if score_a > score_b else "b"
                    pair = {
                        "dimension": dim,
                        "prompt": context.get("goal").get(
                            "goal_text"
                        ),  # Optional: use goal or reasoning task if desired
                        "output_a": embedded[id_a][0]["text"],
                        "output_b": embedded[id_b][0]["text"],
                        "preferred": preferred,
                    }
                    contrast_pairs.append(pair)

        self.logger.log(
            "ContrastPairGenerationComplete",
            {
                "pairs_generated": len(contrast_pairs),
                "dimensions_covered": list(dim_seen),
            },
        )

        return contrast_pairs
