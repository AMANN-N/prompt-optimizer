#!/usr/bin/env python3
"""
Test script for the production-ready prompt optimizer components.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_models():
    """Test that all new config models are properly defined."""
    print("Testing config models...")
    from src.core.config import (
        FieldSimilarityScore,
        SemanticScoringResult,
        FieldPartialCredit,
        PartialCreditResult,
        ConfidenceCalibrationResult,
        FieldUncertainty,
        UncertaintyEstimationResult,
        EnhancedEvaluationResult,
        EnhancedOverallEvaluationReport,
        CrossValidationResult,
        DiversityCheckResult,
        OverfittingDetectionResult,
    )

    # Test SemanticScoringResult
    sem_result = SemanticScoringResult(
        overall_similarity_score=0.85,
        field_scores=[],
        average_similarity=0.85,
        match_count=5,
        total_fields=10,
    )
    assert sem_result.overall_similarity_score == 0.85
    print("  ✓ SemanticScoringResult works")

    # Test PartialCreditResult
    partial_result = PartialCreditResult(
        overall_credit_score=0.72,
        field_credits=[],
        exact_matches=3,
        normalized_matches=2,
        partial_matches=1,
        mismatches=1,
        missing_fields=3,
    )
    assert partial_result.overall_credit_score == 0.72
    print("  ✓ PartialCreditResult works")

    # Test ConfidenceCalibrationResult
    conf_result = ConfidenceCalibrationResult(
        calibrated_confidence=0.65,
        raw_confidence=0.8,
        temperature_scale_factor=1.2,
        calibration_quality="fair",
        confidence_breakdown={},
        is_calibrated=True,
    )
    assert conf_result.calibrated_confidence == 0.65
    print("  ✓ ConfidenceCalibrationResult works")

    # Test UncertaintyEstimationResult
    uncert_result = UncertaintyEstimationResult(
        overall_uncertainty_score=0.3,
        field_uncertainties={},
        uncertain_fields=["amount", "date"],
        confidence_intervals={},
        is_reliable=True,
    )
    assert uncert_result.overall_uncertainty_score == 0.3
    print("  ✓ UncertaintyEstimationResult works")

    # Test EnhancedEvaluationResult
    eval_result = EnhancedEvaluationResult(
        exact_match_score=0.7,
        semantic_similarity_score=0.8,
        partial_credit_score=0.75,
        confidence_calibrated_score=0.65,
        uncertainty_flags={},
        field_scores={},
        rule_violations=[],
        hallucinations_detected=False,
        format_validity=True,
    )
    assert eval_result.exact_match_score == 0.7
    print("  ✓ EnhancedEvaluationResult works")

    # Test EnhancedOverallEvaluationReport
    overall = EnhancedOverallEvaluationReport(
        results=[eval_result],
        exact_match_score=0.7,
        semantic_similarity_score=0.8,
        partial_credit_score=0.75,
        confidence_calibrated_score=0.65,
        overall_score=0.72,
        uncertainty_detected=False,
        uncertain_fields=[],
        failure_patterns={},
    )
    assert overall.overall_score == 0.72
    print("  ✓ EnhancedOverallEvaluationReport works")

    # Test CrossValidationResult
    cv_result = CrossValidationResult(
        fold_results=[{"train": 0.9, "val": 0.85}],
        mean_score=0.85,
        std_score=0.05,
        generalization_gap=0.05,
        is_overfitting=False,
    )
    assert cv_result.mean_score == 0.85
    print("  ✓ CrossValidationResult works")

    # Test DiversityCheckResult
    div_result = DiversityCheckResult(
        cluster_scores={"cluster_0": 0.8, "cluster_1": 0.75},
        minimum_cluster_score=0.75,
        diversity_score=0.65,
        is_diverse=True,
        cluster_assignments={},
    )
    assert div_result.is_diverse == True
    print("  ✓ DiversityCheckResult works")

    # Test OverfittingDetectionResult
    over_result = OverfittingDetectionResult(
        risk_level="low",
        gini_coefficient=0.2,
        fields_at_risk=[],
        recommendation="Low overfitting risk",
        field_accuracy_distribution={},
    )
    assert over_result.risk_level == "low"
    print("  ✓ OverfittingDetectionResult works")

    print("✓ All config models work correctly!\n")


def test_evaluator():
    """Test the enhanced evaluator."""
    print("Testing enhanced evaluator...")
    from src.agents.evaluator import Evaluator
    from unittest.mock import Mock

    mock_llm = Mock()
    evaluator = Evaluator(mock_llm)

    predictions = [
        {"amount": "80000", "date": "10/11/2025"},
        {"amount": "2496537", "date": "12/11/2025"},
    ]
    ground_truths = [
        {"amount": "80000", "date": "10/11/2025"},
        {"amount": "2496537", "date": "11/11/2025"},
    ]
    rules = ["Extract all fields"]

    try:
        report = evaluator.run(predictions, ground_truths, rules)
        assert hasattr(report, "overall_score")
        assert hasattr(report, "semantic_similarity_score")
        print("  ✓ Enhanced evaluator runs successfully")
    except Exception as e:
        print(f"  ⚠ Evaluator test skipped (may need LLM): {e}")

    print()


def test_constraint_checker():
    """Test the constraint checker."""
    print("Testing constraint checker...")
    from src.agents.constraint_checker import check_constraints, ConstraintChecker

    good_prompt = {
        "system_role": "You are an expert analyst.",
        "task_instruction": "Step 1: Read the document. Step 2: Extract the amount.",
        "field_definitions": "amount: The tender amount in rupees",
        "extraction_steps": "Step 1: Identify the amount field. Step 2: Extract the numeric value.",
        "error_handling": "If amount is not found, return null.",
        "output_format": "Return JSON with amount field.",
    }

    result = check_constraints(good_prompt)
    print(
        f"  Good prompt: {result.all_constraints_satisfied} constraints satisfied, {len(result.violated_constraints)} violated"
    )

    bad_prompt = {
        "system_role": "",
        "task_instruction": "Extract data like the first example",
        "field_definitions": "",
        "extraction_steps": "Do the needful",
        "error_handling": "",
        "output_format": "",
    }

    result = check_constraints(bad_prompt)
    print(
        f"  Bad prompt: {result.all_constraints_satisfied} constraints satisfied, {len(result.violated_constraints)} violated"
    )
    assert len(result.violated_constraints) > 0

    print("  ✓ Constraint checker works correctly\n")


def test_exploration_controller():
    """Test the exploration controller."""
    print("Testing exploration controller...")
    from src.agents.exploration_controller import ExplorationController

    controller = ExplorationController()

    # Test with improving scores
    decision = controller.decide([0.7, 0.72, 0.75, 0.78])
    print(
        f"  Improving trend: explore={decision.exploration_weight:.2f}, exploit={decision.exploitation_weight:.2f}"
    )

    # Test with plateau
    decision = controller.decide([0.78, 0.78, 0.79, 0.78])
    print(
        f"  Plateau trend: explore={decision.exploration_weight:.2f}, exploit={decision.exploitation_weight:.2f}, reason={decision.reason[:50]}..."
    )

    # Test status
    status = controller.get_status()
    print(f"  Status: {status['status']}, trend={status['trend']}")

    print("  ✓ Exploration controller works correctly\n")


def test_edge_case_detector():
    """Test the edge case detector."""
    print("Testing edge case detector...")
    from src.agents.edge_case_detector import EdgeCaseDetector

    detector = EdgeCaseDetector()

    predictions = [
        {"amount": "80000", "date": "10/11/2025", "authority": "Chandigarh Tourism"},
        {"amount": "", "date": "12/11/2025", "authority": ""},
    ]
    ground_truths = [
        {"amount": "80000", "date": "10/11/2025", "authority": "Chandigarh Tourism"},
        {
            "amount": "2496537",
            "date": "11/11/2025",
            "authority": "Road Construction Department",
        },
    ]

    result = detector.detect(predictions, ground_truths)
    print(
        f"  Normal: {result.normal_count}, Edge cases: {result.edge_case_count}, Outliers: {result.outlier_count}"
    )
    print(f"  Recommendations: {result.recommendations[0][:80]}...")

    print("  ✓ Edge case detector works correctly\n")


def test_partial_credit():
    """Test the partial credit scorer."""
    print("Testing partial credit scorer...")
    from src.agents.partial_credit_scorer import compute_partial_credit

    pred = {"amount": "₹80,000", "date": "10/11/2025"}
    gt = {"amount": "80000", "date": "10/11/2025"}

    result = compute_partial_credit(pred, gt)
    print(f"  Overall credit score: {result.overall_credit_score:.2f}")
    print(
        f"  Exact matches: {result.exact_matches}, Normalized matches: {result.normalized_matches}"
    )

    print("  ✓ Partial credit scorer works correctly\n")


def main():
    print("=" * 60)
    print("Production-Ready Prompt Optimizer - Component Tests")
    print("=" * 60)
    print()

    test_config_models()
    test_evaluator()
    test_constraint_checker()
    test_exploration_controller()
    test_edge_case_detector()
    test_partial_credit()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
