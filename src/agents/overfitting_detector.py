from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import numpy as np


class OverfittingDetectionResult(BaseModel):
    risk_level: str
    gini_coefficient: float
    fields_at_risk: List[str]
    recommendation: str
    field_accuracy_distribution: Dict[str, Dict[str, float]]


class OverfittingDetector:
    RISK_LOW = "low"
    RISK_MEDIUM = "medium"
    RISK_HIGH = "high"
    DEFAULT_GINI_THRESHOLD = 0.3
    DEFAULT_ACCURACY_VARIANCE_THRESHOLD = 0.25

    def __init__(
        self,
        gini_threshold: float = DEFAULT_GINI_THRESHOLD,
        accuracy_variance_threshold: float = DEFAULT_ACCURACY_VARIANCE_THRESHOLD,
    ):
        self.gini_threshold = gini_threshold
        self.accuracy_variance_threshold = accuracy_variance_threshold

    def _compute_gini(self, values: List[float]) -> float:
        if not values or len(values) < 2:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        cumulative = 0.0
        for i, v in enumerate(sorted_values):
            cumulative += (2 * (i + 1) - n - 1) * v

        return cumulative / (n * sum(values)) if sum(values) > 0 else 0.0

    def _compute_field_accuracy_variance(
        self, predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        field_accuracies = {}

        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            if not isinstance(pred, dict) or not isinstance(gt, dict):
                continue

            self._extract_field_accuracies(pred, gt, "", field_accuracies, i)

        return field_accuracies

    def _extract_field_accuracies(
        self,
        pred: Any,
        gt: Any,
        prefix: str,
        field_accuracies: Dict[str, List[float]],
        example_idx: int,
    ):
        if isinstance(gt, dict):
            for key, value in gt.items():
                full_key = f"{prefix}.{key}" if prefix else key
                pred_val = pred.get(key) if isinstance(pred, dict) else None
                if isinstance(value, dict):
                    self._extract_field_accuracies(
                        pred_val, value, full_key, field_accuracies, example_idx
                    )
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        list_key = f"{full_key}[{i}]"
                        pred_list_val = (
                            pred_val[i]
                            if isinstance(pred_val, list) and i < len(pred_val)
                            else None
                        )
                        if isinstance(item, dict):
                            self._extract_field_accuracies(
                                pred_list_val,
                                item,
                                list_key,
                                field_accuracies,
                                example_idx,
                            )
                        else:
                            accuracy = 1.0 if pred_list_val == item else 0.0
                            if list_key not in field_accuracies:
                                field_accuracies[list_key] = []
                            while len(field_accuracies[list_key]) <= example_idx:
                                field_accuracies[list_key].append(0.0)
                            field_accuracies[list_key][example_idx] = accuracy
                else:
                    accuracy = 1.0 if pred_val == value else 0.0
                    if full_key not in field_accuracies:
                        field_accuracies[full_key] = []
                    while len(field_accuracies[full_key]) <= example_idx:
                        field_accuracies[full_key].append(0.0)
                    field_accuracies[full_key][example_idx] = accuracy

    def detect(
        self, predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]
    ) -> OverfittingDetectionResult:
        if not predictions or not ground_truths:
            return OverfittingDetectionResult(
                risk_level=self.RISK_LOW,
                gini_coefficient=0.0,
                fields_at_risk=[],
                recommendation="Insufficient data for overfitting detection",
                field_accuracy_distribution={},
            )

        field_accuracies = self._compute_field_accuracy_variance(
            predictions, ground_truths
        )

        if not field_accuracies:
            return OverfittingDetectionResult(
                risk_level=self.RISK_LOW,
                gini_coefficient=0.0,
                fields_at_risk=[],
                recommendation="No comparable fields found for overfitting detection",
                field_accuracy_distribution={},
            )

        field_mean_accuracies = {}
        for field, accuracies in field_accuracies.items():
            valid_accuracies = [a for a in accuracies if a > 0 or len(accuracies) > 0]
            field_mean_accuracies[field] = (
                sum(valid_accuracies) / len(valid_accuracies)
                if valid_accuracies
                else 0.0
            )

        accuracy_values = list(field_mean_accuracies.values())
        gini = self._compute_gini(accuracy_values)

        fields_at_risk = []
        field_accuracy_distribution = {}

        for field, mean_acc in field_mean_accuracies.items():
            distribution = {
                "mean_accuracy": mean_acc,
                "min_accuracy": min(field_accuracies.get(field, []))
                if field_accuracies.get(field)
                else 0.0,
                "max_accuracy": max(field_accuracies.get(field, []))
                if field_accuracies.get(field)
                else 0.0,
                "variance": np.var(field_accuracies.get(field, []))
                if field_accuracies.get(field)
                else 0.0,
            }
            field_accuracy_distribution[field] = distribution

            if mean_acc > 0.95:
                min_acc = min(field_accuracies.get(field, []))
                if min_acc < 0.60:
                    fields_at_risk.append(field)

        risk_level = self.RISK_LOW
        if gini > 0.5:
            risk_level = self.RISK_HIGH
        elif gini > self.gini_threshold:
            risk_level = self.RISK_MEDIUM

        recommendation = ""
        if risk_level == self.RISK_HIGH:
            recommendation = f"HIGH OVERFITTING RISK (Gini={gini:.2f}). The prompt is too specific to certain examples. Add more diverse examples and simplify rules."
        elif risk_level == self.RISK_MEDIUM:
            recommendation = f"Medium overfitting risk (Gini={gini:.2f}). Some fields have uneven performance. Review and generalize instructions for: {', '.join(fields_at_risk[:3])}"
        else:
            recommendation = (
                "Low overfitting risk. Performance is consistent across examples."
            )

        return OverfittingDetectionResult(
            risk_level=risk_level,
            gini_coefficient=gini,
            fields_at_risk=fields_at_risk,
            recommendation=recommendation,
            field_accuracy_distribution=field_accuracy_distribution,
        )

    def should_pause_for_overfitting(self, result: OverfittingDetectionResult) -> tuple:
        if result.risk_level == self.RISK_HIGH:
            return True, f"High overfitting risk detected. {result.recommendation}"
        if result.risk_level == self.RISK_MEDIUM and len(result.fields_at_risk) > 0:
            return (
                True,
                f"Medium overfitting risk. Fields at risk: {', '.join(result.fields_at_risk)}. {result.recommendation}",
            )
        return False, ""


def detect_overfitting(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    gini_threshold: float = 0.3,
) -> OverfittingDetectionResult:
    detector = OverfittingDetector(gini_threshold)
    return detector.detect(predictions, ground_truths)
