# analysis/score_analyzer.py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class ScoreAnalyzer:
    def __init__(self, score_data: pd.DataFrame):
        """
        Expected format:All right
        - 'hypothesis_id': str
        - 'dimension': str
        - 'score': float
        - Optional: 'outcome' (e.g., final ranking, human eval)
        """
        self.df = score_data
        self.pivot = self.df.pivot(index='hypothesis_id', columns='dimension', values='score')

    def describe_scores(self):
        return self.pivot.describe()

    def fit_linear_regression(self, outcome_col: str):
        merged = self.pivot.copy()
        merged[outcome_col] = self.df.drop_duplicates(subset='hypothesis_id').set_index('hypothesis_id')[outcome_col]
        merged = merged.dropna()
        X = merged.drop(columns=[outcome_col])
        y = merged[outcome_col]
        model = LinearRegression().fit(X, y)
        return model, dict(zip(X.columns, model.coef_))

    def perform_pca(self, n_components=2):
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(self.pivot.fillna(0))
        return components, pca.explained_variance_ratio_

    def cluster_outputs(self, n_clusters=3):
        km = KMeans(n_clusters=n_clusters, n_init=10)
        labels = km.fit_predict(self.pivot.fillna(0))
        return labels

    def plot_pca_clusters(self, n_clusters=3):
        components, _ = self.perform_pca()
        labels = self.cluster_outputs(n_clusters=n_clusters)
        plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='tab10')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA of Score Vectors (Colored by Cluster)')
        plt.show()