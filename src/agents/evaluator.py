from src.agents.base import BaseAgent
from src.core.config import (
    EnhancedOverallEvaluationReport,
    EnhancedEvaluationResult,
    TaskConfig,
)
from src.agents.semantic_scorer import batch_semantic_score
from src.agents.partial_credit_scorer import batch_partial_credit
from src.agents.confidence_calibrator import ConfidenceCalibrator
from typing import List, Dict, Any
import json


class Evaluator(BaseAgent):
    DEFAULT_WEIGHTS = {
        "exact_match": 0.30,
        "semantic": 0.30,
        "partial_credit": 0.25,
        "confidence": 0.15,
    }

    def __init__(self, llm_client, weights: Dict[str, float] = None):
        super().__init__(llm_client)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.calibrator = ConfidenceCalibrator()

    def _compute_exact_match_score(
        self, predictions: List[Any], ground_truths: List[Any]
    ) -> float:
        if not predictions or not ground_truths:
            return 1.0

        exact_matches = 0
        total_fields = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_str = json.dumps(pred, sort_keys=True) if pred else ""
            gt_str = json.dumps(gt, sort_keys=True) if gt else ""

            if pred_str == gt_str:
                exact_matches += 1

        return exact_matches / len(predictions) if predictions else 1.0

    def _extract_field_scores(
        self, semantic_results: Dict, partial_results: Dict
    ) -> Dict[str, Dict[str, float]]:
        field_scores = {}

        for example_key, sem_result in semantic_results.items():
            field_scores[example_key] = {}
            for field_score in sem_result.field_scores:
                field_scores[example_key][field_score.field_name] = {
                    "semantic_similarity": field_score.cosine_similarity,
                    "is_exact": field_score.is_exact_match,
                }

        for example_key, partial_result in partial_results.items():
            if example_key not in field_scores:
                field_scores[example_key] = {}
            for credit in partial_result.field_credits:
                if credit.field_name not in field_scores[example_key]:
                    field_scores[example_key][credit.field_name] = {}
                field_scores[example_key][credit.field_name]["partial_credit"] = (
                    credit.score
                )
                field_scores[example_key][credit.field_name]["match_type"] = (
                    credit.match_type
                )

        return field_scores

    def _aggregate_field_scores(
        self, field_scores: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        aggregated = {}

        for example_key, scores in field_scores.items():
            for field_name, metrics in scores.items():
                if field_name not in aggregated:
                    aggregated[field_name] = {}

                for metric, value in metrics.items():
                    if metric not in aggregated[field_name]:
                        aggregated[field_name][metric] = []
                    if isinstance(value, (int, float)):
                        aggregated[field_name][metric].append(value)

        final_scores = {}
        for field_name, metrics in aggregated.items():
            final_scores[field_name] = {}
            for metric, values in metrics.items():
                if values:
                    final_scores[field_name][metric] = sum(values) / len(values)

        return final_scores

    def run(
        self,
        predictions: List[Any],
        ground_truths: List[Any],
        rules: List[str],
        task_config: TaskConfig = None,
        confidence_scores: List[float] = None,
    ) -> EnhancedOverallEvaluationReport:
        exact_match_score = self._compute_exact_match_score(predictions, ground_truths)

        semantic_results = batch_semantic_score(
            [p if isinstance(p, dict) else {} for p in predictions],
            [g if isinstance(g, dict) else {} for g in ground_truths],
        )

        semantic_scores = [
            r.overall_similarity_score for r in semantic_results.values()
        ]
        avg_semantic_score = (
            sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
        )

        partial_results = batch_partial_credit(
            [p if isinstance(p, dict) else {} for p in predictions],
            [g if isinstance(g, dict) else {} for g in ground_truths],
        )

        partial_scores = [r.overall_credit_score for r in partial_results.values()]
        avg_partial_score = (
            sum(partial_scores) / len(partial_scores) if partial_scores else 0.0
        )

        calibrated_scores = []
        if confidence_scores:
            for conf in confidence_scores:
                calib_result = self.calibrator.calibrate(conf)
                calibrated_scores.append(calib_result.calibrated_confidence)
        else:
            calibrated_scores = [0.5] * len(predictions)

        avg_confidence_score = (
            sum(calibrated_scores) / len(calibrated_scores)
            if calibrated_scores
            else 0.5
        )

        overall_score = (
            self.weights["exact_match"] * exact_match_score
            + self.weights["semantic"] * avg_semantic_score
            + self.weights["partial_credit"] * avg_partial_score
            + self.weights["confidence"] * avg_confidence_score
        )

        uncertainty_flags = {}
        uncertain_fields = []
        for example_key, result in semantic_results.items():
            for field_score in result.field_scores:
                if field_score.cosine_similarity < 0.7:
                    uncertainty_flags[field_score.field_name] = True
                    if field_score.field_name not in uncertain_fields:
                        uncertain_fields.append(field_score.field_name)

        field_scores = self._extract_field_scores(semantic_results, partial_results)

        results = []
        for i in range(len(predictions)):
            example_key = f"example_{i}"
            sem_result = semantic_results.get(example_key)
            partial_result = partial_results.get(example_key)

            example_uncertainty_flags = {}
            if sem_result:
                for field_score in sem_result.field_scores:
                    if field_score.cosine_similarity < 0.7:
                        example_uncertainty_flags[field_score.field_name] = True

            results.append(
                EnhancedEvaluationResult(
                    exact_match_score=1.0
                    if json.dumps(predictions[i], sort_keys=True)
                    == json.dumps(ground_truths[i], sort_keys=True)
                    else 0.0,
                    semantic_similarity_score=sem_result.overall_similarity_score
                    if sem_result
                    else 0.0,
                    partial_credit_score=partial_result.overall_credit_score
                    if partial_result
                    else 0.0,
                    confidence_calibrated_score=calibrated_scores[i]
                    if i < len(calibrated_scores)
                    else 0.5,
                    uncertainty_flags=example_uncertainty_flags,
                    field_scores={},
                    rule_violations=[],
                    hallucinations_detected=False,
                    format_validity=True,
                )
            )

        report = EnhancedOverallEvaluationReport(
            results=results,
            exact_match_score=exact_match_score,
            semantic_similarity_score=avg_semantic_score,
            partial_credit_score=avg_partial_score,
            confidence_calibrated_score=avg_confidence_score,
            overall_score=overall_score,
            uncertainty_detected=len(uncertain_fields) > 0,
            uncertain_fields=uncertain_fields,
            failure_patterns={},
        )

        return report
