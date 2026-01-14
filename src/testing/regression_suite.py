from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import os
from datetime import datetime


class RegressionTestResult(BaseModel):
    test_passed: bool
    new_score: float
    old_score: float
    score_delta: float
    tolerance: float
    passed_tests: int
    failed_tests: int
    total_tests: int
    details: List[Dict[str, Any]]


class RegressionReport(BaseModel):
    test_results: List[RegressionTestResult]
    overall_passed: bool
    deployment_blocked: bool
    recommendations: List[str]
    tested_at: str


class RegressionSuite:
    DEFAULT_TOLERANCE = 0.05

    def __init__(self, storage_path: Optional[str] = None, tolerance: float = None):
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "regression_history.json"
        )
        self.tolerance = tolerance or self.DEFAULT_TOLERANCE
        self.history = self._load_history()

    def _load_history(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"prompts": {}}

    def _save_history(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def _get_prompt_key(self, prompt: str) -> str:
        import hashlib

        return hashlib.md5(prompt.encode()).hexdigest()[:8]

    def run_regression_test(
        self,
        prompt_version_id: str,
        new_predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        evaluate_func,
        old_prompt_version_id: Optional[str] = None,
    ) -> RegressionTestResult:
        if old_prompt_version_id is None:
            if prompt_version_id in self.history["prompts"]:
                old_prompt_version_id = self.history["prompts"][prompt_version_id].get(
                    "previous_version"
                )
            else:
                old_prompt_version_id = None

        new_score = evaluate_func(new_predictions, ground_truths)

        if old_prompt_version_id and old_prompt_version_id in self.history["prompts"]:
            old_score = self.history["prompts"][old_prompt_version_id].get(
                "last_score", 0.0
            )
        else:
            old_score = new_score

        score_delta = new_score - old_score
        test_passed = score_delta >= -self.tolerance

        return RegressionTestResult(
            test_passed=test_passed,
            new_score=new_score,
            old_score=old_score,
            score_delta=score_delta,
            tolerance=self.tolerance,
            passed_tests=1 if test_passed else 0,
            failed_tests=0 if test_passed else 1,
            total_tests=1,
            details=[
                {
                    "prompt_version": prompt_version_id,
                    "old_prompt_version": old_prompt_version_id,
                    "score_delta": score_delta,
                    "tolerance": self.tolerance,
                    "message": "PASSED"
                    if test_passed
                    else f"REGRESSION: Score dropped by {abs(score_delta):.2%}",
                }
            ],
        )

    def run_full_suite(
        self,
        prompt_version_id: str,
        new_predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        evaluate_func,
        test_sets: List[Dict[str, Any]],
    ) -> RegressionReport:
        results = []
        total_passed = 0
        total_failed = 0

        for test_set in test_sets:
            test_name = test_set.get("name", "unnamed")
            test_predictions = test_set.get("predictions", new_predictions)
            test_ground_truths = test_set.get("ground_truths", ground_truths)

            result = self.run_regression_test(
                prompt_version_id, test_predictions, test_ground_truths, evaluate_func
            )
            result.details[0]["test_name"] = test_name
            results.append(result)

            if result.test_passed:
                total_passed += 1
            else:
                total_failed += 1

        overall_passed = total_failed == 0
        deployment_blocked = not overall_passed

        recommendations = []
        if overall_passed:
            recommendations.append(
                "All regression tests passed. Deployment can proceed."
            )
        else:
            recommendations.append(
                f"REGRESSION DETECTED: {total_failed}/{len(results)} tests failed"
            )
            for result in results:
                if not result.test_passed:
                    recommendations.append(
                        f"- {result.details[0].get('test_name', 'unknown')}: {result.details[0].get('message', '')}"
                    )
            recommendations.append("Do NOT deploy. Review and fix failing tests first.")

        return RegressionReport(
            test_results=results,
            overall_passed=overall_passed,
            deployment_blocked=deployment_blocked,
            recommendations=recommendations,
            tested_at=datetime.now().isoformat(),
        )

    def save_result(
        self, prompt_version_id: str, score: float, prompt_content: str = None
    ):
        prompt_key = self._get_prompt_key(prompt_version_id)

        if prompt_key not in self.history["prompts"]:
            self.history["prompts"][prompt_key] = {"versions": [], "last_score": 0.0}

        version_entry = {
            "version_id": prompt_version_id,
            "score": score,
            "timestamp": datetime.now().isoformat(),
        }

        self.history["prompts"][prompt_key]["versions"].append(version_entry)

        if len(self.history["prompts"][prompt_key]["versions"]) > 1:
            prev_version = self.history["prompts"][prompt_key]["versions"][-2]
            self.history["prompts"][prompt_key]["previous_version"] = prev_version[
                "version_id"
            ]

        self.history["prompts"][prompt_key]["last_score"] = score
        self._save_history()

    def get_prompt_history(self, prompt_version_id: str) -> List[Dict[str, Any]]:
        prompt_key = self._get_prompt_key(prompt_version_id)
        if prompt_key in self.history["prompts"]:
            return self.history["prompts"][prompt_key]["versions"]
        return []


def run_regression_test(
    prompt_version_id: str,
    new_predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    evaluate_func,
    storage_path: Optional[str] = None,
    tolerance: float = 0.05,
) -> RegressionTestResult:
    suite = RegressionSuite(storage_path=storage_path, tolerance=tolerance)
    return suite.run_regression_test(
        prompt_version_id, new_predictions, ground_truths, evaluate_func
    )
