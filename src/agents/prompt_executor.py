from src.agents.base import BaseAgent
from src.core.config import PromptObject
from typing import Optional
import os
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    raw_output: str
    parsed_output: dict
    confidence_score: float
    uncertainty_flags: dict
    execution_time_ms: float
    model_metadata: dict


class PromptExecutor(BaseAgent):
    def run(
        self,
        prompt_object: PromptObject,
        input_data: str,
        return_metadata: bool = False,
    ) -> ExecutionResult:
        import time

        start_time = time.time()

        full_system_prompt = (
            f"{prompt_object.system_role}\n\n"
            f"{prompt_object.task_instruction}\n\n"
            f"Field Definitions:\n{prompt_object.field_definitions}\n\n"
            f"Extraction Steps:\n{prompt_object.extraction_steps}\n\n"
            f"Error Handling:\n{prompt_object.error_handling}\n\n"
            f"Output Format:\n{prompt_object.output_format}"
        )

        image_path = None
        user_message = f"Input Data:\n{input_data}"

        if os.path.exists(input_data) and input_data.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            image_path = input_data
            user_message = "Extract data from this image."

        raw_output = self.llm.complete(
            prompt=user_message, system_prompt=full_system_prompt, image_path=image_path
        )

        execution_time_ms = (time.time() - start_time) * 1000

        try:
            parsed_output = self.llm.parse_json(raw_output)
        except:
            parsed_output = {"error": "Failed to parse output"}

        confidence_score = self._estimate_confidence(raw_output, parsed_output)
        uncertainty_flags = self._detect_uncertainty(parsed_output)

        model_metadata = {
            "execution_time_ms": execution_time_ms,
            "has_image": image_path is not None,
            "output_length": len(raw_output),
        }

        if return_metadata:
            return ExecutionResult(
                raw_output=raw_output,
                parsed_output=parsed_output,
                confidence_score=confidence_score,
                uncertainty_flags=uncertainty_flags,
                execution_time_ms=execution_time_ms,
                model_metadata=model_metadata,
            )

        return raw_output

    def _estimate_confidence(self, raw_output: str, parsed_output: dict) -> float:
        confidence_indicators = []

        if not raw_output or len(raw_output.strip()) < 10:
            confidence_indicators.append(0.3)
        else:
            confidence_indicators.append(0.7)

        if parsed_output and not parsed_output.get("error"):
            confidence_indicators.append(0.2)

        uncertainty_words = [
            "uncertain",
            "unclear",
            "might",
            "possibly",
            "could be",
            "not sure",
        ]
        raw_lower = raw_output.lower()
        uncertainty_count = sum(1 for word in uncertainty_words if word in raw_lower)
        if uncertainty_count > 0:
            confidence_indicators.append(-0.1 * uncertainty_count)

        null_count = 0
        total_fields = 0
        if parsed_output:

            def count_fields(obj, path=""):
                nonlocal null_count, total_fields
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        count_fields(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for item in obj:
                        count_fields(item)
                else:
                    total_fields += 1
                    if obj is None or (
                        isinstance(obj, str) and obj.lower() in ["null", "n/a", ""]
                    ):
                        null_count += 1

            count_fields(parsed_output)

        if total_fields > 0:
            null_ratio = null_count / total_fields
            if null_ratio > 0.5:
                confidence_indicators.append(-0.15)
            elif null_ratio < 0.2:
                confidence_indicators.append(0.1)

        base_confidence = 0.5
        for adjustment in confidence_indicators:
            base_confidence += adjustment

        return max(0.0, min(1.0, base_confidence))

    def _detect_uncertainty(self, parsed_output: dict) -> dict:
        uncertainty_flags = {}

        if not parsed_output or parsed_output.get("error"):
            uncertainty_flags["parse_error"] = True
            return uncertainty_flags

        null_fields = []

        def check_uncertainty(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_uncertainty(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_uncertainty(item, f"{path}[{i}]")
            else:
                if obj is None or (
                    isinstance(obj, str) and obj.lower() in ["null", "n/a", ""]
                ):
                    null_fields.append(path)

        check_uncertainty(parsed_output)

        if null_fields:
            uncertainty_flags["null_fields"] = null_fields

        return uncertainty_flags
