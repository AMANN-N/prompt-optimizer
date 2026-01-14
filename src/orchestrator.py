from src.core.llm_client import LLMClient
from src.core.config import (
    TaskConfig,
    PromptObject,
    EnhancedOverallEvaluationReport,
    FailureAnalysis,
)
from src.agents.task_formalizer import TaskFormalizer
from src.agents.base_prompt_generator import BasePromptGenerator
from src.agents.prompt_executor import PromptExecutor, ExecutionResult
from src.agents.evaluator import Evaluator
from src.agents.failure_analyzer import FailureAnalyzer
from src.agents.prompt_optimizer import PromptOptimizer
from src.agents.prompt_freezer import PromptFreezer
from src.agents.prompt_selector import PromptSelector
from src.agents.variant_generator import PromptVariantGenerator
from src.core.cross_validator import CrossValidator
from src.agents.diversity_checker import DiversityChecker
from src.agents.overfitting_detector import OverfittingDetector
from src.agents.edge_case_detector import EdgeCaseDetector
from typing import List, Dict, Any


class Orchestrator:
    def __init__(
        self,
        llm_provider="mock",
        api_key=None,
        model_name="gemini-2.0-flash-lite-preview-02-05",
    ):
        self.llm = LLMClient(
            provider=llm_provider, api_key=api_key, model_name=model_name
        )

        self.task_formalizer = TaskFormalizer(self.llm)
        self.base_generator = BasePromptGenerator(self.llm)
        self.executor = PromptExecutor(self.llm)
        self.evaluator = Evaluator(self.llm)
        self.analyzer = FailureAnalyzer(self.llm)
        self.optimizer = PromptOptimizer(self.llm)
        self.freezer = PromptFreezer(self.llm)
        self.selector = PromptSelector(self.llm)
        self.variant_generator = PromptVariantGenerator(self.llm)

        self.cross_validator = CrossValidator(
            k_folds=5, generalization_gap_threshold=0.15
        )
        self.diversity_checker = DiversityChecker()
        self.overfitting_detector = OverfittingDetector()
        self.edge_case_detector = EdgeCaseDetector()

    def optimize_prompt(
        self,
        user_intent: str,
        demo_data: List[Dict[str, Any]],
        max_iterations: int = 3,
        base_prompt_context: str = None,
    ):
        print(f"🚀 Starting Optimization for intent: {user_intent[:50]}...")
        if base_prompt_context:
            print(f"ℹ️ Base Prompt Context provided ({len(base_prompt_context)} chars)")

        task_config = self.task_formalizer.run(user_intent)
        print(f"Task Config: {task_config.task_name}")

        current_prompt = self.base_generator.run(
            task_config, context=base_prompt_context
        )

        predictions = []
        ground_truths = [item["output"] for item in demo_data]
        confidence_scores = []

        best_score = 0
        best_prompt = current_prompt
        score_history = []

        for i in range(max_iterations):
            print(f"\n🔄 Iteration {i + 1}/{max_iterations}")

            predictions = []
            confidence_scores = []

            for item in demo_data:
                input_data = item["input"]
                result = self.executor.run(
                    current_prompt, input_data, return_metadata=True
                )

                if isinstance(result, ExecutionResult):
                    predictions.append(result.parsed_output)
                    confidence_scores.append(result.confidence_score)
                else:
                    try:
                        pred = self.llm.parse_json(result)
                        predictions.append(pred)
                    except:
                        predictions.append({})
                    confidence_scores.append(0.5)

            report = self.evaluator.run(
                predictions,
                ground_truths,
                task_config.hard_rules,
                task_config,
                confidence_scores,
            )

            current_score = report.overall_score
            score_history.append(current_score)
            print(f"Score: {current_score:.4f}")

            if current_score >= 95:
                print("✅ Threshold met!")
                best_prompt = current_prompt
                best_score = current_score
                break

            if current_score > best_score:
                best_score = current_score
                best_prompt = current_prompt

            should_stop, stop_reason = self._check_early_stopping(
                score_history, report, predictions, ground_truths
            )
            if should_stop:
                print(f"🛑 Early stopping: {stop_reason}")
                break

            failures = self.analyzer.run(report)

            if not failures:
                print("No obvious failures found, stopping.")
                break

            overfitting_result = self.overfitting_detector.detect(
                predictions, ground_truths
            )
            print(f"Overfitting risk: {overfitting_result.risk_level}")

            diversity_result = self.diversity_checker.check_diversity(
                predictions,
                ground_truths,
                lambda p, g: self._quick_evaluate(p, g, task_config),
            )
            print(
                f"Diversity check: {'Passed' if diversity_result.is_diverse else 'Failed'}"
            )

            opt_result = self.optimizer.run(
                current_prompt,
                failures,
                overfitting_info=overfitting_result.model_dump(),
                diversity_info=diversity_result.model_dump(),
            )

            current_prompt = opt_result.updated_prompt
            print(f"Changes: {opt_result.change_log[:200]}...")

        edge_case_result = self.edge_case_detector.detect(predictions, ground_truths)
        if edge_case_result.outlier_count > 0:
            print(f"⚠️ Edge cases detected: {edge_case_result.outlier_count} outliers")

        print("\n❄️ Step 9: Freezing Best Prompt...")
        final_result = self.freezer.run(best_prompt)
        print("🎉 Optimization Complete!")

        return {
            "frozen_prompt": final_result.get("frozen_prompt", ""),
            "best_score": best_score,
            "score_history": score_history,
            "overfitting_risk": overfitting_result.risk_level
            if "overfitting_result" in dir()
            else "unknown",
            "diversity_check": diversity_result.is_diverse
            if "diversity_result" in dir()
            else True,
            "edge_cases": edge_case_result.outlier_count
            if "edge_case_result" in dir()
            else 0,
        }

    def optimize_with_variants(
        self,
        user_intent: str,
        demo_data: List[Dict[str, Any]],
        max_iterations: int = 3,
        base_prompt_context: str = None,
        num_variants: int = 3,
    ):
        print(
            f"🚀 Starting Multi-Variant Optimization for intent: {user_intent[:50]}..."
        )

        task_config = self.task_formalizer.run(user_intent)
        current_prompt = self.base_generator.run(
            task_config, context=base_prompt_context
        )

        ground_truths = [item["output"] for item in demo_data]

        best_score = 0
        best_prompt = current_prompt
        score_history = []
        variant_scores = {}

        for i in range(max_iterations):
            print(f"\n🔄 Iteration {i + 1}/{max_iterations}")

            variants = self.variant_generator.run(current_prompt, n=num_variants)
            all_prompts = [current_prompt] + variants

            variant_predictions = {f"base": []}
            for v in variants:
                variant_predictions[f"variant_{variants.index(v)}"] = []

            for item in demo_data:
                input_data = item["input"]

                for j, prompt in enumerate(all_prompts):
                    key = "base" if j == 0 else f"variant_{j - 1}"
                    result = self.executor.run(prompt, input_data, return_metadata=True)

                    if isinstance(result, ExecutionResult):
                        variant_predictions[key].append(result.parsed_output)
                    else:
                        try:
                            pred = self.llm.parse_json(result)
                            variant_predictions[key].append(pred)
                        except:
                            variant_predictions[key].append({})

            for key, preds in variant_predictions.items():
                report = self.evaluator.run(
                    preds, ground_truths, task_config.hard_rules, task_config
                )
                variant_scores[key] = report.overall_score
                print(f"  {key}: {report.overall_score:.4f}")

            best_key = max(variant_scores, key=variant_scores.get)
            best_score = variant_scores[best_key]

            if best_score >= 95:
                print("✅ Threshold met!")
                best_prompt = (
                    all_prompts[0]
                    if best_key == "base"
                    else variants[int(best_key.split("_")[1])]
                )
                break

            if best_score > score_history[-1] if score_history else 0:
                current_prompt = (
                    all_prompts[0]
                    if best_key == "base"
                    else variants[int(best_key.split("_")[1])]
                )

            score_history.append(best_score)

            should_stop, stop_reason = self._check_early_stopping(
                score_history, variant_predictions[best_key], ground_truths, task_config
            )
            if should_stop:
                print(f"🛑 Early stopping: {stop_reason}")
                break

            if best_key != "base":
                failures = self.analyzer.run(
                    self.evaluator.run(
                        variant_predictions[best_key],
                        ground_truths,
                        task_config.hard_rules,
                        task_config,
                    )
                )

                overfitting_result = self.overfitting_detector.detect(
                    variant_predictions[best_key], ground_truths
                )

                opt_result = self.optimizer.run(
                    current_prompt,
                    failures,
                    overfitting_info=overfitting_result.model_dump(),
                )
                current_prompt = opt_result.updated_prompt

        print("\n❄️ Freezing Best Prompt...")
        final_result = self.freezer.run(best_prompt)
        print("🎉 Multi-Variant Optimization Complete!")

        return {
            "frozen_prompt": final_result.get("frozen_prompt", ""),
            "best_score": best_score,
            "score_history": score_history,
            "variant_scores": variant_scores,
        }

    def _quick_evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        task_config: TaskConfig,
    ) -> float:
        if not predictions or not ground_truths:
            return 0.0

        exact_matches = 0
        for pred, gt in zip(predictions, ground_truths):
            if pred == gt:
                exact_matches += 1

        return exact_matches / len(predictions)

    def _check_early_stopping(
        self,
        score_history: List[float],
        report: EnhancedOverallEvaluationReport,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
    ) -> tuple:
        if len(score_history) < 3:
            return False, ""

        recent_scores = score_history[-3:]
        improvement = recent_scores[-1] - recent_scores[0]

        if improvement < 0.01:
            return True, "No significant improvement in last 3 iterations"

        if hasattr(report, "uncertainty_detected") and report.uncertainty_detected:
            if (
                len(report.uncertainty_fields) > len(predictions[0]) * 0.5
                if predictions and isinstance(predictions[0], dict)
                else False
            ):
                return True, "High uncertainty detected across predictions"

        return False, ""
