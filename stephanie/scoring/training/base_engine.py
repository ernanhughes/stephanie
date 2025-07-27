import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from stephanie.models.training_stats import TrainingStatsORM


class BaseTrainingEngine:
    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = cfg.get("batch_size", 32)
        self.epochs = cfg.get("epochs", 10)
        self.lr = cfg.get("lr", 1e-4)
        self.gamma = cfg.get("gamma", 0.99)
        self.use_early_stopping = cfg.get("early_stopping", True)
        self.patience = cfg.get("patience", 3)
        self.min_delta = cfg.get("min_delta", 0.001)
        self.early_stop_counter = 0
        self.best_loss = float('inf')
    
    def _create_dataloader(self, samples):
        """Convert samples to PyTorch DataLoader"""
        context_embs, doc_embs, scores = [], [], []
        
        for item in samples:
            context = item.get("title", "")
            doc_text = item.get("output", "")
            
            context_emb = torch.tensor(self.memory.embedding.get_or_create(context))
            doc_emb = torch.tensor(self.memory.embedding.get_or_create(doc_text))
            score = float(item.get("score", 0.5))
            
            context_embs.append(context_emb)
            doc_embs.append(doc_emb)
            scores.append(score)
        
        # Convert to tensors
        context_tensors = torch.stack(context_embs).to(self.device)
        doc_tensors = torch.stack(doc_embs).to(self.device)
        score_tensors = torch.tensor(scores).float().to(self.device)
        
        return DataLoader(
            torch.utils.data.TensorDataset(
                context_tensors, doc_tensors, score_tensors
            ),
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def _should_stop_early(self, losses):
        """Early stopping logic with validation"""
        if not self.use_early_stopping or len(losses) < self.patience:
            return False
        
        # Get recent losses
        recent_losses = losses[-self.patience:]
        avg_recent = sum(recent_losses) / len(recent_losses)
        
        # Check for improvement
        if avg_recent < self.best_loss - self.min_delta:
            self.best_loss = avg_recent
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
        
        # Return early stopping condition
        return self.early_stop_counter >= self.patience
    
    def _log_training_stats(self, stats):
        """Log training stats to database"""
        training_stats = TrainingStatsORM(**stats)
        self.memory.session.add(training_stats)
        self.memory.session.commit()