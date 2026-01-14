from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.agents.base import BaseAgent
from src.core.llm_client import LLMClient
import json


class FieldUncertainty(BaseModel):
    field_name: str
    values: List[Any]
    variance_score: float
    is_uncertain: bool
    entropy: float


class UncertaintyEstimationResult(BaseModel):
    overall_uncertainty_score: float
    field_uncertainties: Dict[str, FieldUncertainty]
    uncertain_fields: List[str]
    confidence_intervals: Dict[str, Any]
    is_reliable: bool


class UncertaintyEstimator(BaseAgent):
    DEFAULT_NUM_RUNS = 3
    DEFAULT_TEMPERATURE = 0.3
    UNCERTAINTY_THRESHOLD = 0.3

    def __init__(self, llm_client: LLMClient, num_runs: int = DEFAULT_NUM_RUNS):
        super().__init__(llm_client)
        self.num_runs = num_runs

    def _normalize_for_comparison(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip().lower()

    def _compute_variance_score(self, values: List[Any]) -> float:
        normalized = [self._normalize_for_comparison(v) for v in values]
        unique = list(set(normalized))

        if len(unique) == 1:
            return 0.0

        if len(values) <= 1:
            return 0.0

        variance = len(unique) / len(values)
        return min(1.0, variance)

    def _compute_entropy(self, values: List[Any]) -> float:
        normalized = [self._normalize_for_comparison(v) for v in values]
        value_counts = {}
        for val in normalized:
            value_counts[val] = value_counts.get(val, 0) + 1

        total = len(values)
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * (p.bit_length() - 1)

        return entropy

    def _extract_field_values(
        self, predictions: List[Dict[str, Any]], prefix: str = ""
    ) -> Dict[str, List[Any]]:
        field_values = {}

        for pred_idx, prediction in enumerate(predictions):
            if not isinstance(prediction, dict):
                continue

            self._extract_recursive(prediction, prefix, field_values, pred_idx)

        return field_values

    def _extract_recursive(
        self, obj: Any, prefix: str, field_values: Dict[str, List[Any]], pred_idx: int
    ):
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    self._extract_recursive(value, full_key, field_values, pred_idx)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        list_key = f"{full_key}[{i}]"
                        if isinstance(item, dict):
                            self._extract_recursive(
                                item, list_key, field_values, pred_idx
                            )
                        else:
                            if list_key not in field_values:
                                field_values[list_key] = []
                            while len(field_values[list_key]) <= pred_idx:
                                field_values[list_key].append(None)
                            if len(field_values[list_key]) == pred_idx:
                                field_values[list_key].append(item)
                            else:
                                field_values[list_key][pred_idx] = item
                else:
                    if full_key not in field_values:
                        field_values[full_key] = []
                    while len(field_values[full_key]) <= pred_idx:
                        field_values[full_key].append(None)
                    if len(field_values[full_key]) == pred_idx:
                        field_values[full_key].append(value)
                    else:
                        field_values[full_key][pred_idx] = value

    def estimate(
        self, predictions: List[Dict[str, Any]]
    ) -> UncertaintyEstimationResult:
        if len(predictions) < 2:
            return UncertaintyEstimationResult(
                overall_uncertainty_score=0.0,
                field_uncertainties={},
                uncertain_fields=[],
                confidence_intervals={},
                is_reliable=True,
            )

        field_values = self._extract_field_values(predictions)

        field_uncertainties = {}
        uncertain_fields = []
        total_uncertainty = 0.0
        field_count = 0

        for field_name, values in field_values.items():
            values = [v for v in values if v is not None]

            if len(values) < 2:
                continue

            variance_score = self._compute_variance_score(values)
            entropy = self._compute_entropy(values)
            is_uncertain = variance_score >= self.UNCERTAINTY_THRESHOLD

            field_uncertainties[field_name] = FieldUncertainty(
                field_name=field_name,
                values=values,
                variance_score=variance_score,
                is_uncertain=is_uncertain,
                entropy=entropy,
            )

            if is_uncertain:
                uncertain_fields.append(field_name)

            total_uncertainty += variance_score
            field_count += 1

        overall_uncertainty = (
            total_uncertainty / field_count if field_count > 0 else 0.0
        )
        is_reliable = overall_uncertainty < self.UNCERTAINTY_THRESHOLD

        confidence_intervals = {}
        for field_name, uncertainty in field_uncertainties.items():
            values = uncertainty.values
            if values and all(isinstance(v, (int, float)) for v in values):
                mean_val = sum(values) / len(values)
                std_val = (
                    sum((v - mean_val) ** 2 for v in values) / len(values)
                ) ** 0.5
                confidence_intervals[field_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "ci_95": (mean_val - 1.96 * std_val, mean_val + 1.96 * std_val),
                }

        return UncertaintyEstimationResult(
            overall_uncertainty_score=overall_uncertainty,
            field_uncertainties=field_uncertainties,
            uncertain_fields=uncertain_fields,
            confidence_intervals=confidence_intervals,
            is_reliable=is_reliable,
        )


def run_uncertainty_estimation(
    predictions: List[Dict[str, Any]],
) -> UncertaintyEstimationResult:
    estimator = UncertaintyEstimator(llm_client=None)
    return estimator.estimate(predictions)
