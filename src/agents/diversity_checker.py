from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
import os


class DiversityCheckResult(BaseModel):
    cluster_scores: Dict[str, float]
    minimum_cluster_score: float
    diversity_score: float
    is_diverse: bool
    cluster_assignments: Dict[int, List[int]]
    recommendations: List[str]


class DiversityChecker:
    DEFAULT_MIN_CLUSTER_SCORE = 0.60
    DEFAULT_DIVERSITY_THRESHOLD = 0.50
    DEFAULT_NUM_CLUSTERS = 3

    def __init__(
        self,
        min_cluster_score: float = DEFAULT_MIN_CLUSTER_SCORE,
        diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
        num_clusters: int = DEFAULT_NUM_CLUSTERS,
    ):
        self.min_cluster_score = min_cluster_score
        self.diversity_threshold = diversity_threshold
        self.num_clusters = num_clusters
        self.model = None

    def _get_model(self):
        if self.model is None:
            model_name = os.getenv("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2")
            self.model = SentenceTransformer(model_name)
        return self.model

    def _extract_text_features(self, data: List[Dict[str, Any]]) -> List[str]:
        features = []
        for item in data:
            text_parts = []
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, str):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, (int, float)):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, list):
                        text_parts.append(f"{key}: {len(value)} items")
                if text_parts:
                    features.append(" | ".join(text_parts))
                else:
                    features.append(str(item))
            else:
                features.append(str(item))
        return features

    def _compute_cluster_score(
        self,
        cluster_indices: List[int],
        data: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        evaluate_func,
    ) -> float:
        if not cluster_indices:
            return 0.0

        cluster_data = [data[i] for i in cluster_indices]
        cluster_labels = (
            [ground_truths[i] for i in cluster_indices]
            if ground_truths
            else [None] * len(cluster_data)
        )

        return evaluate_func(cluster_data, cluster_labels)

    def check_diversity(
        self,
        data: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        evaluate_func,
        cluster_assignments: Optional[List[int]] = None,
    ) -> DiversityCheckResult:
        if len(data) < 3:
            return DiversityCheckResult(
                cluster_scores={},
                minimum_cluster_score=0.0,
                diversity_score=1.0,
                is_diverse=True,
                cluster_assignments={},
                recommendations=["Insufficient data for diversity check"],
            )

        model = self._get_model()
        text_features = self._extract_text_features(data)

        if all(f.strip() == "" for f in text_features):
            text_features = [f"item_{i}" for i in range(len(data))]

        embeddings = model.encode(text_features)

        actual_clusters = min(self.num_clusters, len(data) // 2, len(data))
        if actual_clusters < 2:
            actual_clusters = 2

        if cluster_assignments is None or len(cluster_assignments) != len(data):
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            cluster_assignments = kmeans.fit_predict(embeddings).tolist()

        cluster_to_indices = {}
        for i, cluster_id in enumerate(cluster_assignments):
            if cluster_id not in cluster_to_indices:
                cluster_to_indices[cluster_id] = []
            cluster_to_indices[cluster_id].append(i)

        cluster_scores = {}
        for cluster_id, indices in cluster_to_indices.items():
            score = self._compute_cluster_score(
                indices, data, ground_truths, evaluate_func
            )
            cluster_scores[f"cluster_{cluster_id}"] = score

        min_cluster_score = min(cluster_scores.values()) if cluster_scores else 0.0

        cluster_embeddings = {}
        for cluster_id, indices in cluster_to_indices.items():
            cluster_embeddings[cluster_id] = np.mean(
                [embeddings[i] for i in indices], axis=0
            )

        inter_cluster_distances = []
        cluster_ids = list(cluster_embeddings.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                dist = np.linalg.norm(
                    cluster_embeddings[cluster_ids[i]]
                    - cluster_embeddings[cluster_ids[j]]
                )
                inter_cluster_distances.append(dist)

        avg_inter_cluster_dist = (
            np.mean(inter_cluster_distances) if inter_cluster_distances else 0.0
        )

        intra_cluster_variances = []
        for cluster_id, indices in cluster_to_indices.items():
            if len(indices) > 1:
                cluster_embs = [embeddings[i] for i in indices]
                centroid = np.mean(cluster_embs, axis=0)
                variance = np.mean([np.linalg.norm(e - centroid) for e in cluster_embs])
                intra_cluster_variances.append(variance)

        avg_intra_variance = (
            np.mean(intra_cluster_variances) if intra_cluster_variances else 0.0
        )

        diversity_score = avg_inter_cluster_dist / (avg_intra_variance + 1e-6)
        diversity_score = min(1.0, diversity_score / 2.0)

        is_diverse = (
            min_cluster_score >= self.min_cluster_score
            and diversity_score >= self.diversity_threshold
        )

        recommendations = []
        if min_cluster_score < self.min_cluster_score:
            weakest_cluster = min(cluster_scores, key=cluster_scores.get)
            recommendations.append(
                f"Weak cluster ({weakest_cluster}, score={cluster_scores[weakest_cluster]:.2%}). Add more diverse examples for this cluster type."
            )
        if diversity_score < self.diversity_threshold:
            recommendations.append(
                f"Low diversity score ({diversity_score:.2%}). Examples may be too similar. Consider adding varied examples."
            )
        if not recommendations:
            recommendations.append("Good diversity across examples.")

        return DiversityCheckResult(
            cluster_scores=cluster_scores,
            minimum_cluster_score=min_cluster_score,
            diversity_score=diversity_score,
            is_diverse=is_diverse,
            cluster_assignments={k: v for k, v in cluster_to_indices.items()},
            recommendations=recommendations,
        )


def check_diversity(
    data: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    evaluate_func,
    min_cluster_score: float = 0.60,
    diversity_threshold: float = 0.50,
) -> DiversityCheckResult:
    checker = DiversityChecker(min_cluster_score, diversity_threshold)
    return checker.check_diversity(data, ground_truths, evaluate_func)
