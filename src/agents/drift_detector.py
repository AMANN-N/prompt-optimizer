from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np


class DriftDetectionResult(BaseModel):
    drift_detected: bool
    drift_severity: str
    drift_score: float
    drifted_fields: List[str]
    mahalanobis_distances: Dict[int, float]
    recommendations: List[str]


class TrainingDistribution(BaseModel):
    field_embeddings: Dict[str, Dict[str, float]]
    field_means: Dict[str, float]
    field_stds: Dict[str, float]
    overall_mean: List[float]
    overall_std: List[float]
    created_at: str


class DriftDetector:
    SEVERITY_NONE = "none"
    SEVERITY_LOW = "low"
    SEVERITY_MEDIUM = "medium"
    SEVERITY_HIGH = "high"

    DEFAULT_DRIFT_THRESHOLD = 3.0
    DEFAULT_DRIFT_SEVERITY_THRESHOLDS = {
        SEVERITY_LOW: 3.0,
        SEVERITY_MEDIUM: 5.0,
        SEVERITY_HIGH: 8.0,
    }

    def __init__(
        self, drift_threshold: float = None, storage_path: Optional[str] = None
    ):
        self.drift_threshold = drift_threshold or self.DEFAULT_DRIFT_THRESHOLD
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "training_distribution.json"
        )
        self.training_distribution = None
        self.model = None

    def _get_model(self):
        if self.model is None:
            model_name = os.getenv("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2")
            self.model = SentenceTransformer(model_name)
        return self.model

    def _ensure_storage_dir(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def save_training_distribution(self, training_predictions: List[Dict[str, Any]]):
        self._ensure_storage_dir()

        model = self._get_model()

        field_embeddings = {}
        field_values = {}

        for pred in training_predictions:
            if not isinstance(pred, dict):
                continue

            for key, value in pred.items():
                if key not in field_values:
                    field_values[key] = []
                field_values[key].append(str(value) if value is not None else "")

        field_means = {}
        field_stds = {}

        for field, values in field_values.items():
            if len(values) > 1:
                embeddings = model.encode(values)
                field_embeddings[field] = {
                    "values": values,
                    "embeddings": embeddings.tolist(),
                }
                field_means[field] = float(np.mean(embeddings))
                field_stds[field] = (
                    float(np.std(embeddings)) if np.std(embeddings) > 0 else 1.0
                )

        all_embeddings = []
        for embeddings_dict in field_embeddings.values():
            all_embeddings.extend(embeddings_dict.get("embeddings", []))

        if all_embeddings:
            overall_mean = np.mean(all_embeddings, axis=0).tolist()
            overall_std = np.std(all_embeddings, axis=0).tolist()
        else:
            overall_mean = []
            overall_std = []

        from datetime import datetime

        self.training_distribution = TrainingDistribution(
            field_embeddings=field_embeddings,
            field_means=field_means,
            field_stds=field_stds,
            overall_mean=overall_mean,
            overall_std=overall_std,
            created_at=datetime.now().isoformat(),
        )

        with open(self.storage_path, "w") as f:
            json.dump(self.training_distribution.model_dump(), f, indent=2)

        return self.training_distribution

    def load_training_distribution(self) -> Optional[TrainingDistribution]:
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self.training_distribution = TrainingDistribution(**data)
                    return self.training_distribution
        except Exception as e:
            print(f"Warning: Failed to load training distribution: {e}")
        return None

    def _compute_mahalanobis_distance(
        self, embedding: List[float], mean: List[float], std: List[float]
    ) -> float:
        if not mean or not std:
            return 0.0

        embedding = np.array(embedding)
        mean = np.array(mean)
        std = np.array(std)

        std = np.where(std == 0, 1e-6, std)

        diff = embedding - mean
        normalized = diff / std

        return float(np.sqrt(np.sum(normalized**2)))

    def _extract_field_values(self, prediction: Dict[str, Any]) -> Dict[str, str]:
        result = {}
        for key, value in prediction.items():
            if isinstance(value, (int, float)):
                result[key] = str(value)
            elif value is not None:
                result[key] = str(value)
            else:
                result[key] = ""
        return result

    def detect_drift(
        self, new_predictions: List[Dict[str, Any]]
    ) -> DriftDetectionResult:
        if self.training_distribution is None:
            self.load_training_distribution()

        if self.training_distribution is None:
            return DriftDetectionResult(
                drift_detected=False,
                drift_severity=self.SEVERITY_NONE,
                drift_score=0.0,
                drifted_fields=[],
                mahalanobis_distances={},
                recommendations=[
                    "No training distribution saved. Cannot detect drift."
                ],
            )

        model = self._get_model()

        mahalanobis_distances = {}
        field_drift_scores = {}

        for idx, prediction in enumerate(new_predictions):
            if not isinstance(prediction, dict):
                continue

            field_values = self._extract_field_values(prediction)

            max_distance = 0.0
            drifted_fields = []

            for field, value in field_values.items():
                if field in self.training_distribution.field_means:
                    mean = self.training_distribution.field_means[field]
                    std = self.training_distribution.field_stds[field]

                    embedding = model.encode([value]).tolist()[0]

                    distance = self._compute_mahalanobis_distance(embedding, mean, std)
                    mahalanobis_distances[idx] = (
                        mahalanobis_distances.get(idx, 0) + distance
                    )

                    if distance > self.drift_threshold:
                        drifted_fields.append(field)
                        max_distance = max(max_distance, distance)

            if drifted_fields:
                field_drift_scores[idx] = {
                    "max_distance": max_distance,
                    "drifted_fields": drifted_fields,
                }

        if not mahalanobis_distances:
            return DriftDetectionResult(
                drift_detected=False,
                drift_severity=self.SEVERITY_NONE,
                drift_score=0.0,
                drifted_fields=[],
                mahalanobis_distances={},
                recommendations=["No comparable fields found for drift detection."],
            )

        avg_distance = np.mean(list(mahalanobis_distances.values()))
        drift_score = avg_distance

        drift_severity = self.SEVERITY_NONE
        if drift_score >= self.DEFAULT_DRIFT_SEVERITY_THRESHOLDS[self.SEVERITY_HIGH]:
            drift_severity = self.SEVERITY_HIGH
        elif (
            drift_score >= self.DEFAULT_DRIFT_SEVERITY_THRESHOLDS[self.SEVERITY_MEDIUM]
        ):
            drift_severity = self.SEVERITY_MEDIUM
        elif drift_score >= self.DEFAULT_DRIFT_SEVERITY_THRESHOLDS[self.SEVERITY_LOW]:
            drift_severity = self.SEVERITY_LOW

        drift_detected = drift_severity != self.SEVERITY_NONE

        all_drifted_fields = []
        for info in field_drift_scores.values():
            all_drifted_fields.extend(info["drifted_fields"])
        all_drifted_fields = list(set(all_drifted_fields))

        recommendations = []
        if drift_detected:
            recommendations.append(
                f"Drift detected (severity: {drift_severity}, score: {drift_score:.2f})"
            )
            if all_drifted_fields:
                recommendations.append(
                    f"Drifted fields: {', '.join(all_drifted_fields[:5])}"
                )
            recommendations.append(
                "Consider retraining with new examples or adding drift handling."
            )
        else:
            recommendations.append(
                "No significant drift detected. Current model should perform well."
            )

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            drift_score=drift_score,
            drifted_fields=all_drifted_fields,
            mahalanobis_distances=mahalanobis_distances,
            recommendations=recommendations,
        )

    def should_pause_for_drift(self, result: DriftDetectionResult) -> tuple:
        if result.drift_detected and result.drift_severity in [
            self.SEVERITY_MEDIUM,
            self.SEVERITY_HIGH,
        ]:
            return (
                True,
                f"Drift detected ({result.drift_severity}). {result.recommendations[0]}",
            )
        return False, ""


def detect_drift(
    new_predictions: List[Dict[str, Any]], storage_path: Optional[str] = None
) -> DriftDetectionResult:
    detector = DriftDetector(storage_path=storage_path)
    return detector.detect_drift(new_predictions)
