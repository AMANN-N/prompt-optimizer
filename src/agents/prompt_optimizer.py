from src.agents.base import BaseAgent
from src.core.config import PromptObject, FailureAnalysis, OptimizationResult
from typing import List
import yaml


class PromptOptimizer(BaseAgent):
    PROMPT_TEMPLATE = """
You are a Prompt Optimization Agent.

Given:
- The current prompt (YAML):
{current_prompt}

- A structured failure summary:
{failure_summary}

- Anti-Overfitting Guidelines:
{anti_overfitting_guidelines}

Your task:
- Improve the prompt to reduce failures
- Modify ONLY necessary sections
- Preserve parts that already work
- Ensure the prompt GENERALIZES across diverse examples

CRITICAL ANTI-OVERFITTING RULES:
- The prompt must work on diverse examples, not just similar ones
- Avoid example-specific phrases; use general patterns
- If a rule is needed for only 1/10 examples, it's likely overfitting
- Prefer rules that generalize across 80%+ of examples
- Add explicit instructions for handling edge cases generically
- Avoid hard-coding specific formats; describe patterns instead

Return YAML with:
- updated_prompt (complete prompt object)
- change_log (what changed and why)
- generalization_notes (how changes improve generalization)
"""

    def __init__(self, llm_client, anti_overfitting_weight: float = 0.1):
        super().__init__(llm_client)
        self.anti_overfitting_weight = anti_overfitting_weight

    def run(
        self,
        current_prompt: PromptObject,
        failures: List[FailureAnalysis],
        overfitting_info: dict = None,
        diversity_info: dict = None,
    ) -> OptimizationResult:
        failures_str = json.dumps([f.model_dump() for f in failures], indent=2)

        anti_overfitting_guidelines = self._build_anti_overfitting_guidelines(
            overfitting_info, diversity_info
        )

        prompt = self.PROMPT_TEMPLATE.format(
            current_prompt=current_prompt.to_yaml(),
            failure_summary=failures_str,
            anti_overfitting_guidelines=anti_overfitting_guidelines,
        )

        response = self.llm.complete(prompt)
        data = self.llm.parse_yaml(response)

        updated_prompt_data = data.get("updated_prompt", {})
        change_log = data.get("change_log", "No changes logged")
        generalization_notes = data.get("generalization_notes", "")

        if isinstance(change_log, list):
            change_log = "; ".join([str(item) for item in change_log])

        return OptimizationResult(
            updated_prompt=PromptObject(**updated_prompt_data),
            change_log=f"{change_log}\n\nGeneralization: {generalization_notes}",
        )

    def _build_anti_overfitting_guidelines(
        self, overfitting_info: dict = None, diversity_info: dict = None
    ) -> str:
        guidelines = []

        if overfitting_info:
            risk_level = overfitting_info.get("risk_level", "unknown")
            if risk_level == "high":
                guidelines.append("⚠️ HIGH OVERFITTING RISK DETECTED")
                guidelines.append("- Simplify complex rules that may be too specific")
                guidelines.append("- Add more general pattern descriptions")
                guidelines.append("- Remove example-specific formatting instructions")
            elif risk_level == "medium":
                guidelines.append("⚠️ Medium overfitting risk")
                guidelines.append("- Review fields with uneven performance")
                guidelines.append("- Add fallback instructions for edge cases")

        if diversity_info:
            is_diverse = diversity_info.get("is_diverse", True)
            if not is_diverse:
                guidelines.append("⚠️ Low diversity detected in examples")
                guidelines.append(
                    "- Create rules that work across multiple example types"
                )
                guidelines.append("- Avoid optimizing for a single example pattern")

        guidelines.append("- Ensure new rules apply to 80%+ of examples")
        guidelines.append("- Use pattern-based instructions, not specific examples")
        guidelines.add("- Add explicit uncertainty handling for ambiguous cases")

        return (
            "\n".join(guidelines)
            if guidelines
            else "No specific anti-overfitting guidelines."
        )


import json
