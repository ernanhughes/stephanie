# stephanie/scoring/mrq/training.py


class MRQTraining:
    def train_from_database(self, cfg):
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
                self.trainers[dim] = self._build_trainer(dim)

            self.update_score_bounds_from_data(samples, dim)
            dataloader = self.trainers[dim].prepare_training_data(samples)
            self.trainers[dim].train(dataloader, cfg)

            self.logger.log("MRQTrainingComplete", {"dimension": dim})

    def train_from_context(self, context, cfg):
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

    def align_mrq_with_llm_scores_from_pairs(
        self, pair_samples, dimension, log_prefix="MRQAlignment"
    ):
        for pair in pair_samples:
            prompt = pair["prompt"]
            for side in ["a", "b"]:
                hyp = pair[f"output_{side}"]
                llm_score = pair[f"value_{side}"]

                mrq_score = self.score(
                    {"goal_text": prompt}, self.Scorable(text=hyp), [dimension]
                )

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

                if mrq_score and llm_score is not None:
                    self.regression_tuners[dimension].train_single(
                        mrq_score=mrq_score.results[dimension].score,
                        llm_score=llm_score,
                    )

    def update_score_bounds_from_data(self, samples, dim):
        values = []
        for s in samples:
            if "value_a" in s and "value_b" in s:
                values.extend([s["value_a"], s["value_b"]])
            elif "value" in s:
                values.append(s["value"])
        if values:
            min_value = min(values)
            max_value = max(values)
            self.min_value_by_dim[dim] = min_value
            self.max_value_by_dim[dim] = max_value
            self.logger.log(
                "MRQScoreBoundsUpdated",
                {
                    "dimension": dim,
                    "min_value": min_value,
                    "max_value": max_value,
                    "example_count": len(values),
                },
            )
