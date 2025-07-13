# stephanie/scoring/proximity_scorer.py
import re

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class ProximityScorer(BaseScorer):
    def __init__(self, cfg, memory, logger, prompt_loader=None, dimensions=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.prompt_loader = prompt_loader

        # Dynamically pull dimensions from scoring config
        self.dimensions_config = cfg.get("dimensions", [])
        self.dimensions = dimensions or [d["name"] for d in self.dimensions_config]

        self.models = {}
        self.scalers = {}
        self.trained = dict.fromkeys(self.dimensions, False)
        self.force_rescore = cfg.get("force_rescore", False)
        self.regression_tuners = {}

        for dim in self.dimensions:
            self._initialize_dimension(dim)

    @property
    def name(self) -> str:
        return "proximity"

    def evaluate(self, prompt: str, response: str) -> ScoreBundle:
        if not response:
            return self._fallback("No proximity response available.")

        try:
            themes = self._extract_block(response, "Common Themes Identified")
            grafts = self._extract_block(response, "Grafting Opportunities")
            directions = self._extract_block(response, "Strategic Directions")

            themes_score = 10.0 * len(themes)
            grafts_score = 10.0 * len(grafts)
            directions_score = 20.0 * len(directions)

            results = {
                "proximity_themes": ScoreResult(
                    dimension="proximity_themes",
                    score=min(100.0, themes_score),
                    weight=0.3,
                    rationale=f"{len(themes)} theme(s) identified",
                    source="proximity",
                ),
                "proximity_grafts": ScoreResult(
                    dimension="proximity_grafts",
                    score=min(100.0, grafts_score),
                    weight=0.3,
                    rationale=f"{len(grafts)} grafting suggestion(s)",
                    source="proximity",
                ),
                "proximity_directions": ScoreResult(
                    dimension="proximity_directions",
                    score=min(100.0, directions_score),
                    weight=0.4,
                    rationale=f"{len(directions)} strategic direction(s)",
                    source="proximity",
                ),
            }

            return ScoreBundle(results=results)

        except Exception as e:
            return self._fallback(f"Failed to parse proximity response: {str(e)}")

    def _extract_block(self, text: str, section_title: str) -> list:
        pattern = rf"# {re.escape(section_title)}\n((?:- .+\n?)*)"
        match = re.search(pattern, text)
        if not match:
            return []
        block = match.group(1).strip()
        return [line.strip("- ").strip() for line in block.splitlines() if line.strip()]

    def _fallback(self, message: str) -> ScoreBundle:
        results = {
            "proximity_themes": ScoreResult(
                "proximity_themes", 0.0, message, 0.3, source="proximity"
            ),
            "proximity_grafts": ScoreResult(
                "proximity_grafts", 0.0, message, 0.3, source="proximity"
            ),
            "proximity_directions": ScoreResult(
                "proximity_directions", 0.0, message, 0.4, source="proximity"
            ),
        }
        return ScoreBundle(results=results)

    def _initialize_dimension(self, dim: str):
        self.models[dim] = SVR()
        self.scalers[dim] = StandardScaler()
        self.regression_tuners[dim] = RegressionTuner(dimension=dim, logger=self.logger)
        self.trained[dim] = False

    def _try_train_on_dimension(self, dim: str):
        training_data = self.memory.training.get_training_data(dimension=dim)
        if not training_data:
            return

        X = [self._build_feature_vector(g, s) for g, s in training_data]
        y = [score for (_, _, score) in training_data]

        self.scalers[dim].fit(X)
        X_scaled = self.scalers[dim].transform(X)

        self.models[dim].fit(X_scaled, y)
        self.trained[dim] = True

    def _build_feature_vector(self, goal: dict, scorable: Scorable):
        goal_vec = self.memory.embedding.get_or_create(goal.get("goal_text", ""))
        text_vec = self.memory.embedding.get_or_create(scorable.text)
        return [g - t for g, t in zip(goal_vec, text_vec)]

    def score(
        self, goal: dict, scorable: Scorable, dimensions: list[str]
    ) -> ScoreBundle:
        results = {}

        for dim in dimensions:
            vec = self._build_feature_vector(goal, scorable)

            if not self.trained[dim]:
                self._try_train_on_dimension(dim)

            if not self.trained[dim]:
                score = 50.0
                rationale = f"SVM not trained for {dim}, returning neutral."
            else:
                x = self.scalers[dim].transform([vec])
                raw_score = self.models[dim].predict(x)[0]
                score = self.regression_tuners[dim].transform(raw_score)
                rationale = f"SVM predicted and aligned score for {dim}"

            # Lookup weight from config
            weight = next(
                (
                    d.get("weight", 1.0)
                    for d in self.dimensions_config
                    if d["name"] == dim
                ),
                1.0,
            )

            self.logger.log(
                "ProximityScoreComputed",
                {
                    "dimension": dim,
                    "score": score,
                    "hypothesis": scorable.text,
                },
            )

            results[dim] = ScoreResult(
                dimension=dim,
                score=score,
                rationale=rationale,
                weight=weight,
                source="svm",
            )

        return ScoreBundle(results=results)

    def parse_from_response(self, response: str) -> ScoreBundle:
        return self.evaluate(prompt="", response=response)
