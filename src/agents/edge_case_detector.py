from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import re


class EdgeCaseClassification(BaseModel):
    example_index: int
    is_edge_case: bool
    edge_case_type: Optional[str]
    severity: str
    reasoning: str
    recommended_action: str


class EdgeCaseDetectionResult(BaseModel):
    edge_cases: List[EdgeCaseClassification]
    normal_count: int
    edge_case_count: int
    outlier_count: int
    edge_case_types: Dict[str, int]
    recommendations: List[str]


class EdgeCaseDetector:
    EDGE_CASE_TYPES = {
        "missing_fields": ["missing", "na", "n/a", "null", "none", ""],
        "unusual_format": ["format", "style", "layout"],
        "poor_quality": ["unclear", "blur", "fade", "low resolution"],
        "unusual_value": ["unusual", "unexpected", "outlier"],
        "complex_structure": ["complex", "nested", "multi"],
        "language_variant": ["language", "regional", "local"],
    }

    SEVERITY_HIGH = "high"
    SEVERITY_MEDIUM = "medium"
    SEVERITY_LOW = "low"

    def __init__(
        self, missing_threshold: float = 0.3, format_variance_threshold: float = 0.5
    ):
        self.missing_threshold = missing_threshold
        self.format_variance_threshold = format_variance_threshold

    def _classify_edge_case(
        self,
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any],
        example_index: int,
    ) -> EdgeCaseClassification:
        missing_fields = []
        unusual_formats = []
        unusual_values = []

        for key, gt_val in ground_truth.items():
            pred_val = prediction.get(key) if isinstance(prediction, dict) else None

            if gt_val is None or str(gt_val).lower() in [
                "",
                "na",
                "n/a",
                "null",
                "none",
            ]:
                if pred_val is not None and str(pred_val).strip() != "":
                    unusual_values.append(
                        f"Non-null prediction for missing field: {key}"
                    )

            elif pred_val is None or str(pred_val).lower() in [
                "",
                "na",
                "n/a",
                "null",
                "none",
            ]:
                missing_fields.append(key)

            else:
                pred_str = str(pred_val)
                gt_str = str(gt_val)

                if len(pred_str) > 2 * len(gt_str) or len(pred_str) < 0.5 * len(gt_str):
                    unusual_formats.append(
                        f"Length variance in {key}: pred={len(pred_str)}, gt={len(gt_str)}"
                    )

                if not any(c.isdigit() for c in gt_str) and any(
                    c.isdigit() for c in pred_str
                ):
                    if "date" not in key.lower() and "amount" not in key.lower():
                        unusual_formats.append(f"Unexpected digits in {key}")

        total_fields = len(ground_truth) if ground_truth else 1
        missing_ratio = len(missing_fields) / total_fields

        edge_case_type = None
        severity = self.SEVERITY_LOW
        reasoning_parts = []

        if missing_ratio > self.missing_threshold:
            edge_case_type = "missing_fields"
            severity = (
                self.SEVERITY_MEDIUM if missing_ratio < 0.7 else self.SEVERITY_HIGH
            )
            reasoning_parts.append(f"High missing field ratio: {missing_ratio:.1%}")

        if len(unusual_formats) > 2:
            if edge_case_type is None:
                edge_case_type = "unusual_format"
            severity = self.SEVERITY_MEDIUM
            reasoning_parts.append(
                f"Multiple format anomalies: {len(unusual_formats)} issues"
            )

        if len(unusual_values) > 2:
            edge_case_type = edge_case_type or "unusual_value"
            if severity == self.SEVERITY_LOW:
                severity = self.SEVERITY_MEDIUM
            reasoning_parts.append(
                f"Multiple unusual values: {len(unusual_values)} detected"
            )

        is_edge_case = edge_case_type is not None
        is_outlier = missing_ratio > 0.7 and len(unusual_formats) > 3

        if is_outlier:
            severity = self.SEVERITY_HIGH
            if not reasoning_parts:
                reasoning_parts.append(
                    "Multiple severe issues detected - likely outlier"
                )

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Normal example"

        if is_outlier:
            recommended_action = "FLAG FOR HUMAN REVIEW - This appears to be an outlier that the prompt cannot handle"
        elif is_edge_case:
            recommended_action = (
                f"Add generic handling for {edge_case_type} in error_handling section"
            )
        else:
            recommended_action = "No action needed"

        return EdgeCaseClassification(
            example_index=example_index,
            is_edge_case=is_edge_case,
            edge_case_type=edge_case_type,
            severity=severity,
            reasoning=reasoning,
            recommended_action=recommended_action,
        )

    def detect(
        self, predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]
    ) -> EdgeCaseDetectionResult:
        if not predictions or not ground_truths:
            return EdgeCaseDetectionResult(
                edge_cases=[],
                normal_count=0,
                edge_case_count=0,
                outlier_count=0,
                edge_case_types={},
                recommendations=["Insufficient data for edge case detection"],
            )

        edge_cases = []
        edge_case_types = {}
        outlier_count = 0
        normal_count = 0
        edge_case_count = 0

        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            classification = self._classify_edge_case(pred, gt, i)
            edge_cases.append(classification)

            if classification.is_edge_case:
                edge_case_count += 1
                et = classification.edge_case_type
                edge_case_types[et] = edge_case_types.get(et, 0) + 1

                if classification.severity == self.SEVERITY_HIGH:
                    outlier_count += 1
            else:
                normal_count += 1

        recommendations = []
        if outlier_count > 0:
            recommendations.append(
                f"Found {outlier_count} outliers. Consider excluding from training or adding specific handling."
            )
        if edge_case_types:
            top_type = max(edge_case_types, key=edge_case_types.get)
            recommendations.append(
                f"Most common edge case: {top_type} ({edge_case_types[top_type]} occurrences)"
            )
        if edge_case_count > len(predictions) * 0.3:
            recommendations.append(
                "High edge case ratio (>30%). Consider collecting more representative examples."
            )
        if not recommendations:
            recommendations.append("Good distribution of normal examples.")

        return EdgeCaseDetectionResult(
            edge_cases=edge_cases,
            normal_count=normal_count,
            edge_case_count=edge_case_count,
            outlier_count=outlier_count,
            edge_case_types=edge_case_types,
            recommendations=recommendations,
        )

    def should_flag_for_review(self, result: EdgeCaseDetectionResult) -> tuple:
        if result.outlier_count > 0:
            return (
                True,
                f"Found {result.outlier_count} outliers that require human review",
            )
        if result.edge_case_count > len(result.edge_cases) * 0.5:
            return (
                True,
                f"High edge case ratio ({result.edge_case_count}/{len(result.edge_cases)}). Consider adding handling.",
            )
        return False, ""


def detect_edge_cases(
    predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]
) -> EdgeCaseDetectionResult:
    detector = EdgeCaseDetector()
    return detector.detect(predictions, ground_truths)
