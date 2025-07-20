# stephanie/scoring/document_mrq_trainer.py

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.trainer_engine import MRQTrainerEngine
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class DocumentMRQTrainer:
    def __init__(
        self, memory, logger, encoder=None, value_predictor=None, device="cpu"
    ):
        self.memory = memory
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim
        self.logger = logger
        self.device = device
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim


        self.encoder = encoder.to(device) if encoder else TextEncoder().to(device)
        self.value_predictor = (
            value_predictor.to(device)
            if value_predictor
            else ValuePredictor(self.dim, self.hdim).to(device)
        )
        self.regression_tuners = {}
        self.engine = MRQTrainerEngine(memory, logger, device)

    def train_multidimensional_model(self, contrast_pairs: List[dict], cfg=None):
        return self.engine.train_all(contrast_pairs, cfg or {})

    def align_to_best_llm_neighbour(self, goal, hypothesis, dimension):
        """
        Fetch similar hypotheses that already have high LLM scores.
        Then align MR.Q prediction to the best of them.
        """
        llm_scores = self.get_closest_llm_scores(hypothesis["text"], dimension)
        if llm_scores:
            self.align_with_llm_score(dimension, goal, hypothesis, max(llm_scores))
