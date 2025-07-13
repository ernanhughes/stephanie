# stephanie/scoring/mrq/scorer.py

from .core_scoring import MRQCoreScoring
from .initializer import initialize_dimension
from .model_io import MRQModelIO
from .training import MRQTraining


class MRQScorer(MRQCoreScoring, MRQTraining, MRQModelIO):
    def __init__(self, cfg, memory, logger, dimensions=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.device = cfg.get("device", "cpu")
        self.dimensions = dimensions or ["mrq"]
        self.models = {}
        self.trainers = {}
        self.min_score_by_dim = {}
        self.max_score_by_dim = {}
        self.regression_tuners = {}
        self.value_predictor = None
        self.encoder = None

        for dim in self.dimensions:
            initialize_dimension(self, dim)

        # Bind methods to self
        self.estimate_score = lambda goal, scorable, dim: self.estimate_score(
            self, goal, scorable, dim
        )
        self.evaluate = lambda prompt, response: self.evaluate(self, prompt, response)
        self.judge = lambda goal, prompt, a, b: self.judge(self, goal, prompt, a, b)
        self.predict_score_from_prompt = (
            lambda prompt, dim="mrq", top_k=5: self.predict_score_from_prompt(
                self, prompt, dim, top_k
            )
        )

        self.train_from_database = lambda cfg: self.train_from_database(self, cfg)
        self.train_from_context = lambda ctx, cfg: self.train_from_context(self, ctx, cfg)
        self.align_mrq_with_llm_scores_from_pairs = (
            lambda samples,
            dim,
            prefix="MRQAlignment": self.align_mrq_with_llm_scores_from_pairs(
                self, samples, dim, prefix
            )
        )
        self.update_score_bounds_from_data = (
            lambda samples, dim: self.update_score_bounds_from_data(self, samples, dim)
        )

        self.save_models = lambda: self.save_models(self)
        self.load_models = lambda: self.load_models(self)
        self.load_models_with_path = lambda: self.load_models_with_path(self)
        self.save_metadata = lambda base_dir: self.save_metadata(self, base_dir)
