# stephanie/agents/maintenance/hard_reset_manager.py

import os
import shutil
from datetime import datetime
import json

from stephanie.agents.base_agent import BaseAgent


class HardResetManager(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.reset_thresholds = cfg.get("hard_reset_thresholds", {
            "ethics": 0.2,
            "system_instability": 0.4,
            "alignment_loss": 0.3,
        })
        self.backup_dir = cfg.get("hard_reset_backup_dir", "backups/hard_reset")
        self.model_dir = cfg.get("model_dir", "models")

    def _fetch_recent_scores(self):
        """Query recent scoring results for key dimensions."""
        query = """
        SELECT dimension, AVG(transformed_score) as avg_score
        FROM scoring_history
        WHERE created_at > NOW() - INTERVAL '1 day'
        GROUP BY dimension
        """
        results = self.memory.session.execute(query).fetchall()
        return {r.dimension: r.avg_score for r in results}

    def _ethics_failure(self, scores: dict) -> bool:
        ethics_score = scores.get("ethics", 1.0)
        if ethics_score < self.reset_thresholds["ethics"]:
            self.logger.log("HardResetEthicsFailure", {"ethics_score": ethics_score})
            return True
        return False

    def _instability_detected(self, scores: dict) -> bool:
        # 1. Alignment drift (compared to historical averages)
        if self._alignment_drift(scores.get("alignment", 1.0)):
            return True
            
        # 2. Score volatility (high variance in recent scores)
        if self._score_volatility():
            return True
            
        # 3. Consistency check (model vs LLM agreement)
        if self._consistency_failure():
            return True
            
        return False

    def _restore_backup(self):
        """Restores the model directory from the hard reset backup."""
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        shutil.copytree(self.backup_dir, self.model_dir)
        self.logger.log("HardResetRestore", {
            "from": self.backup_dir,
            "to": self.model_dir
        })

    def create_backup(self):
        """Creates a versioned backup with metadata"""
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        backup_path = os.path.join(self.backup_dir, backup_id)
        
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        
        # Copy models
        shutil.copytree(self.model_dir, backup_path)
        
        # Save metadata
        metadata = {
            "timestamp": str(datetime.utcnow()),
            "model_versions": self._get_current_versions(),
            "description": "Hard reset baseline"
        }
        
        with open(os.path.join(backup_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
            
        self.logger.log("HardResetBackupCreated", {
            "backup_id": backup_id,
            "model_versions": metadata["model_versions"]
        })

    def _get_current_versions(self):
        """Get active model versions from DB"""
        query = """
        SELECT model_type, target_type, dimension, version 
        FROM model_versions WHERE active = TRUE
        """
        results = self.memory.session.execute(query).fetchall()
        return {
            f"{r.model_type}/{r.target_type}/{r.dimension}": r.version 
            for r in results
        }


    def _alignment_drift(self, current_score):
        """Check against historical alignment performance"""
        historical = self._get_historical_avg("alignment")
        if current_score < historical * 0.7:  # 30% drop
            self.logger.log("AlignmentDriftDetected", {
                "current_score": current_score,
                "historical_avg": historical
            })
            return True
        return False

    def _score_volatility(self):
        """Detect high variance in recent scores"""
        query = """
        SELECT dimension, STDDEV_POP(transformed_score) as volatility
        FROM scoring_history
        WHERE created_at > NOW() - INTERVAL '1 hour'
        GROUP BY dimension
        """
        results = self.memory.session.execute(query).fetchall()
        
        for r in results:
            if r.volatility > self.reset_thresholds.get("volatility", 0.5):
                self.logger.log("ScoreVolatilityDetected", {
                    "dimension": r.dimension,
                    "volatility": r.volatility
                })
                return True
        return False
    
    def check_for_reset(self, dry_run=False):
        """Evaluate system state with optional dry run"""
        recent_scores = self._fetch_recent_scores()
        
        if self._ethics_failure(recent_scores) or self._instability_detected(recent_scores):
            self.logger.log("HardResetTriggered", {
                "timestamp": str(datetime.utcnow()),
                "dry_run": dry_run
            })
            
            if not dry_run:
                self._restore_backup()
                self._notify_admins()
                self._log_failure_details(recent_scores)
                
            return True
        return False