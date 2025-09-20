# Replace the heuristic KnowledgeScorer with MRQDPOKnowledgeScorer

class KnowledgeArtifactTrainer:
    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger
        # OLD: self.knowledge_scorer = KnowledgeScorer()
        # NEW:
        self.knowledge_scorer = MRQDPOKnowledgeScorer(
            model_path="models/mrq_dpo/best_reward_head.pt",
            embedder_fn=get_embedding,  # your embedding function
            beta=2.0
        )
        self.artifact_scorer = ArtifactScorer()