import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


class RubricClusterer:
    def __init__(self, memory):
        """
        Parameters:
        - memory: allos us to call the embedding object.
        """
        self.memory = memory

    def embed_rubrics(self, rubrics):
        """Embed each rubric using the provided embedding function."""
        embedded = []
        for r in rubrics:
            text = r["rubric"]
            vec = self.memory.embedding.get_or_create(text)
            embedded.append({
                "text": text,
                "dimension": r.get("dimension", "Unknown"),
                "vector": vec
            })
        return embedded

    def cluster_rubrics(self, embedded_rubrics, num_clusters=6):
        vectors = np.array([r["vector"] for r in embedded_rubrics])
        clustering = AgglomerativeClustering(n_clusters=num_clusters)
        labels = clustering.fit_predict(vectors)
        for i, label in enumerate(labels):
            embedded_rubrics[i]["cluster"] = int(label)
        return embedded_rubrics

    def summarize_clusters(self, clustered_rubrics):
        """Pick the most central rubric in each cluster as representative."""
        df = pd.DataFrame(clustered_rubrics)
        summaries = []

        for cluster_id in sorted(df["cluster"].unique()):
            items = df[df["cluster"] == cluster_id]
            vectors = np.stack(items["vector"])
            centroid = np.mean(vectors, axis=0)
            sims = cosine_similarity([centroid], vectors)[0]
            best_idx = np.argmax(sims)
            rep = items.iloc[best_idx]

            summaries.append({
                "cluster": int(cluster_id),
                "representative_rubric": rep["text"],
                "dimension": rep["dimension"],
                "count": len(items)
            })

        return summaries
