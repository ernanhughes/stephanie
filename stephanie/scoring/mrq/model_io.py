# stephanie/scoring/mrq/model_io.py
import json
import os

import torch


class MRQModelIO:
    def save_models(self):
        base_dir = self.cfg.get("scoring", {}).get("model_dir", "models/mrq/")
        os.makedirs(base_dir, exist_ok=True)

        for dim, (encoder, predictor) in self.models.items():
            dim_dir = os.path.join(base_dir, dim)
            os.makedirs(dim_dir, exist_ok=True)

            torch.save(encoder.state_dict(), os.path.join(dim_dir, "encoder.pt"))
            torch.save(predictor.state_dict(), os.path.join(dim_dir, "predictor.pt"))

            self.regression_tuners[dim].save(os.path.join(dim_dir, "tuner.json"))

            meta = {
                "min_score": self.min_score_by_dim[dim],
                "max_score": self.max_score_by_dim[dim],
            }
            with open(os.path.join(dim_dir, "meta.json"), "w") as f:
                json.dump(meta, f)

            self.logger.log("MRQModelSaved", {"dimension": dim, "path": dim_dir})

    def load_models(self):
        base_dir = self.cfg.get("scoring", {}).get("model_dir", "models/mrq/")

        if not os.path.exists(base_dir):
            self.logger.log("MRQModelDirNotFound", {"path": base_dir})
            return

        self.dimensions = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        for dim in self.dimensions:
            dim_dir = os.path.join(base_dir, dim)

            if dim not in self.models:
                self._initialize_dimension(dim)

            model = self.models[dim]
            encoder = model.encoder
            predictor = model.predictor

            try:
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

                tuner_path = os.path.join(dim_dir, "tuner.json")
                if os.path.exists(tuner_path):
                    self.regression_tuners[dim].load(tuner_path)

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
