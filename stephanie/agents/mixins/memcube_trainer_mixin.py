from sqlalchemy import text

class MemCubeTrainerMixin:
    """
    Mixin that allows agents to invoke SRFT-style training using MemCube-managed refinement examples.
    """

    def train_with_memcube(self, dimension: str, model_cls, model_cfg: dict, trainer_cls, memcube_type="refinement"):
        """
        Trigger SRFT-style training using MemCube-managed examples.

        Args:
            dimension (str): Dimension to train (e.g., "ethics", "usefulness").
            model_cls (class): Model class to train (e.g., EBTModel).
            model_cfg (dict): Model configuration parameters.
            trainer_cls (class): Trainer to use (e.g., SRFTRefinementTrainer).
            memcube_type (str): Type of MemCube entries to pull (default: "refinement").
        """
        # 1. Fetch training examples
        examples = self._fetch_memcube_examples(dimension, memcube_type=memcube_type)
        if not examples:
            self.logger.log("MemCubeTrainingSkipped", {
                "dimension": dimension,
                "reason": "No training data found"
            })
            return

        # 2. Train
        trainer = trainer_cls(cfg=model_cfg, memory=self.memory, logger=self.logger)
        trainer.train_srft_model(dimension=dimension, examples=examples)

        # 3. Log
        self.logger.log("MemCubeTrainingComplete", {
            "dimension": dimension,
            "model_class": model_cls.__name__,
            "examples_used": len(examples)
        })

    def What_fetch_memcube_examples(self, dimension: str, memcube_type="refinement"):
        """
        Query valid MemCube entries for a given scoring dimension and training type.
        """
        query = """
        SELECT * FROM memcubes
        WHERE dimension = %s
          AND type = %s
          AND valid = true
          AND created_at > NOW() - INTERVAL '14 days'
        """
        results = self.memory.session.execute(text(query), (dimension, memcube_type)).fetchall()

        return [{
            "context": r.context,
            "original": r.original,
            "refined": r.refined,
            "dimension": r.dimension,
            "llm_score": r.llm_score,
            "mrq_reward": r.reward_delta,
            "original_energy": r.energy_before,
            "refined_energy": r.energy_after,
        } for r in results]
