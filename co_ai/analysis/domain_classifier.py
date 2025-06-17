import yaml
from sklearn.metrics.pairwise import cosine_similarity

class DomainClassifier:
    def __init__(self, memory, logger, config_path="config/domain/seeds.yaml"):
        self.memory = memory
        self.logger = logger
        self.logger.log("DomainClassifierInit", {"config_path": config_path})

        with open(config_path, "r") as f:
            self.domain_config = yaml.safe_load(f)
        
        self.domains = self.domain_config.get("domains", {})
        self.logger.log("DomainConfigLoaded", {"num_domains": len(self.domains)})
        
        self._prepare_seed_embeddings()

    def _prepare_seed_embeddings(self):
        self.embeddings = []
        self.labels = []
        total_seeds = 0

        for domain, details in self.domains.items():
            seeds = details.get("seeds", [])
            total_seeds += len(seeds)
            for seed in seeds:
                embedding = self.memory.embedding.get_or_create(seed)
                self.embeddings.append(embedding)
                self.labels.append(domain)
        
        self.logger.log("SeedEmbeddingsPrepared", {
            "total_seeds": total_seeds,
            "domains": list(self.domains.keys())
        })

    def classify(self, text: str, top_k: int = 3, min_score: float = 0.7):
        embedding = self.memory.embedding.get_or_create(text)
        scores = []
        for domain, seed_embedding in zip(self.labels, self.embeddings):
            score = float(cosine_similarity([embedding], [seed_embedding])[0][0])
            if score >= min_score:
                scores.append((domain, score))

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
