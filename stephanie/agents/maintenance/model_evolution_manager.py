# stephanie/agents/maintenance/model_evolution_manager.py

import json

from sqlalchemy import text

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.ebt.refinement_trainer import EBTRefinementTrainer


class ModelEvolutionManager(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_dir = cfg.get("model_dir", "models")
        self.min_improvement = cfg.get(
            "min_improvement", 0.05
        )  # 5% improvement threshold

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text", None)

        # Retrieve distinct scoring contexts from history
        query = """
        SELECT DISTINCT model_type, target_type, dimension
        FROM scoring_history
        """
        results = self.memory.session.execute(query).fetchall()

        summary = []

        for row in results:
            model_type = row.model_type
            target_type = row.target_type
            dimension = row.dimension

            # Get current best model
            current = self.get_best_model(model_type, target_type, dimension)

            # Simulate training — replace with actual model training logic
            new_version = (
                f"auto_{self._generate_version(model_type, target_type, dimension)}"
            )
            validation_metrics = {
                "validation_loss": 0.20,  # placeholder
                "accuracy": 0.87,  # placeholder
            }

            # Log the new model version
            model_id = self.log_model_version(
                model_type=model_type,
                target_type=target_type,
                dimension=dimension,
                version=new_version,
                performance=validation_metrics,
            )

            # Compare and promote if better
            if self.check_model_performance(
                validation_metrics, current["performance"] if current else {}
            ):
                self.promote_model_version(model_id)
                status = "promoted"
            else:
                status = "not promoted"

            summary.append(
                {
                    "model_type": model_type,
                    "target_type": target_type,
                    "dimension": dimension,
                    "new_version": new_version,
                    "status": status,
                }
            )

        self.logger.log("ModelEvolutionRun", {"summary": summary})
        return {"status": "completed", "summary": summary}

    def get_best_model(self, model_type: str, target_type: str, dimension: str):
        """Returns the current best model version for a dimension"""
        query = """
        SELECT version, performance 
        FROM model_versions 
        WHERE model_type = :model_type
          AND target_type = :target_type
          AND dimension = :dimension
          AND active = TRUE
        ORDER BY created_at DESC
        LIMIT 1
        """
        result = self.memory.session.execute(
            text(query),
            {
                "model_type": model_type,
                "target_type": target_type,
                "dimension": dimension,
            },
        ).fetchone()

        if result:
            print(f"Pefoorrmance {result.peformance}")
            performance = result.performance or "{}"
            return {"version": result.version, "performance": json.loads(performance)}
        return None

    def log_model_version(
        self,
        model_type: str,
        target_type: str,
        dimension: str,
        version: str,
        performance: dict,
    ):
        """Record a new model version in the registry"""
        query = """
        INSERT INTO model_versions (
            model_type, target_type, dimension, version, performance, active
        ) VALUES (
            :model_type, :target_type, :dimension, :version, :performance, FALSE
        ) RETURNING id
        """
        result = self.memory.session.execute(
            text(query),
            {
                "model_type": model_type,
                "target_type": target_type,
                "dimension": dimension,
                "version": version,
                "performance": json.dumps(performance),
            },
        ).fetchone()

        self.logger.log(
            "ModelVersionLogged",
            {
                "model_type": model_type,
                "dimension": dimension,
                "version": version,
                "performance": performance,
            },
        )
        return result.id

    def promote_model_version(self, model_id: int):
        """Mark a model as active and deprecate previous active models"""
        query = """
        UPDATE model_versions 
        SET active = FALSE 
        WHERE id != :id 
          AND model_type = (SELECT model_type FROM model_versions WHERE id = :id)
          AND target_type = (SELECT target_type FROM model_versions WHERE id = :id)
          AND dimension = (SELECT dimension FROM model_versions WHERE id = :id)
        """
        self.memory.session.execute(text(query), {"id": model_id})

        query = """
        UPDATE model_versions 
        SET active = TRUE 
        WHERE id = :id
        """
        self.memory.session.execute(text(query), {"id": model_id})

        self.logger.log("ModelVersionPromoted", {"model_id": model_id})

    def check_model_performance(self, new_perf: dict, old_perf: dict) -> bool:
        """Compare two model versions to see if new one is better"""
        if not old_perf:
            return True  # no baseline, accept new model

        # Compare based on metrics (e.g., lower loss = better)
        new_loss = new_perf.get("validation_loss", float("inf"))
        old_loss = old_perf.get("validation_loss", float("inf"))

        # Accept if improvement exceeds threshold
        return (old_loss - new_loss) / old_loss > self.min_improvement

    # In model_evolution_manager.py
    def retrain_ebt_from_refinement(self):
        """Check if EBT needs retraining from recent refinements"""
        examples = self._get_recent_refinements()
        
        if examples:
            trainer = EBTRefinementTrainer(self.cfg.ebt_refinement)
            trained_models = trainer.run(examples)
            
            # Promote better models
            for dim, new_model in trained_models.items():
                old_model = self.ebt.models[dim]
                improvement = self._evaluate_improvement(old_model, new_model, dim)
                
                if improvement > self.cfg.get("min_improvement", 0.05):
                    self.ebt.models[dim] = new_model
                    self.logger.log("EBTModelPromoted", {
                        "dimension": dim,
                        "improvement": improvement
                    })

    # model_evolution_manager.py
    def get_best_model(self, model_type: str, target_type: str, dimension: str):
        """Get the current best model for a dimension"""
        query = """
        SELECT version, performance 
        FROM model_versions 
        WHERE model_type = :model_type
        AND target_type = :target_type
        AND dimension = :dimension
        AND active = TRUE
        ORDER BY created_at DESC
        LIMIT 1
        """
        result = self.memory.db.execute(query, {
            "model_type": model_type,
            "target_type": target_type,
            "dimension": dimension
        }).fetchone()
        
        return {
            "version": result.version,
            "performance": json.loads(result.performance)
        } if result else {"version": "v0"}

    def _trigger_ebt_retraining(self, cfg, dimension: str):
        from stephanie.agents.maintenance.ebt_trainer import EBTTrainerAgent
        """Trigger EBT retraining for a dimension"""
        query = """
        SELECT * FROM scoring_history
        WHERE dimension = :dim
        AND created_at > NOW() - INTERVAL '7 days'
        """
        recent_data = self.memory.db.execute(text(query), {"dim": dimension}).fetchall()
        
        if recent_data:
            self.logger.log("EBTRetrainTriggered", {
                "dimension": dimension,
                "examples": len(recent_data)
            })
            # Run trainer with new data
            trainer = EBTTrainerAgent(cfg.ebt_trainer)
            trainer.run(recent_data)