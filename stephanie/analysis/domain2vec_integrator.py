# stephanie/analysis/domain2vec_integrator.py
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.models.embedding import EmbeddingORM
from stephanie.scoring.model.ner_retriever import NERRetrieverEmbedder


class Domain2VecNERIntegrator:
    """
    Integrates Domain2Vec with NER Retriever for enhanced domain-aware entity retrieval.
    
    This class:
    - Uses Domain2Vec to identify relevant domains for a query
    - Uses NER Retriever to find entities within those domains
    - Combines both signals for more accurate retrieval
    """
    
    def __init__(self, memory, ner_retriever: NERRetrieverEmbedder, domain_classifier: ScorableClassifier):
        self.memory = memory
        self.ner_retriever = ner_retriever
        self.domain_classifier = domain_classifier
        self.domain_embeddings = self._build_domain_embeddings()
    
    def _build_domain_embeddings(self) -> Dict[str, np.ndarray]:
        """Build embeddings for each domain using seed examples"""
        domain_embeddings = {}
        
        for domain, details in self.domain_classifier.domains.items():
            seeds = details.get("seeds", [])
            if not seeds:
                continue
                
            # Get embeddings for seed examples
            seed_embeddings = []
            for seed in seeds:
                try:
                    # Use H-Net for domain seed embeddings
                    embedding = self.memory.embedding.get_or_create(seed)
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    seed_embeddings.append(embedding)
                except Exception as e:
                    continue
            
            # Average to get domain embedding
            if seed_embeddings:
                domain_embeddings[domain] = np.mean(seed_embeddings, axis=0)
        
        return domain_embeddings
    
    def find_relevant_domains(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find domains most relevant to the query.
        
        Args:
            query: User query
            top_k: Number of top domains to return
            
        Returns:
            List of (domain, score) tuples
        """
        # Get query embedding from H-Net
        query_embedding = self.memory.embedding.get_or_create(query)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        # Calculate similarity with each domain
        scores = []
        for domain, domain_embedding in self.domain_embeddings.items():
            sim = cosine_similarity([query_embedding], [domain_embedding])[0][0]
            scores.append((domain, float(sim)))
        
        # Sort and return top domains
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def retrieve_domain_entities(
        self, 
        query: str, 
        k: int = 5,
        min_similarity: float = 0.6
    ) -> List[Dict]:
        """
        Retrieve entities relevant to the query within relevant domains.
        
        Args:
            query: User query (type description)
            k: Number of results per domain
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching entities with domain information
        """
        # Find relevant domains
        relevant_domains = self.find_relevant_domains(query)
        self.memory.logger.log("DomainEntitiesRetrieval", {
            "query": query,
            "relevant_domains": [d[0] for d in relevant_domains]
        })
        
        # Retrieve entities for each relevant domain
        all_results = []
        for domain, domain_score in relevant_domains:
            # Use NER Retriever to find entities
            results = self.ner_retriever.retrieve_entities(
                query,
                k=k,
                min_similarity=min_similarity
            )
            
            # Add domain information to results
            for result in results:
                result["domain"] = domain
                result["domain_score"] = float(domain_score)
                all_results.append(result)
        
        # Sort by combined score
        all_results.sort(
            key=lambda x: x["similarity"] * x["domain_score"], 
            reverse=True
        )
        
        # Return top results
        return all_results[:k]