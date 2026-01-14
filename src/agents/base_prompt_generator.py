from src.agents.base import BaseAgent
from src.core.config import TaskConfig, PromptObject
import yaml

class BasePromptGenerator(BaseAgent):
    PROMPT_TEMPLATE = """
You are a Prompt Generator.

Create an initial extraction prompt for the task below.

Task Intent:
{task_intent}

Context / Base Instructions:
{context}

Input Modality:
{input_modality}

Output Schema:
{output_schema}

Rules:
{rules}

Generate a prompt with these sections:
- system_role
- task_instruction
- field_definitions
- extraction_steps
- error_handling
- output_format

Constraints:
- Be explicit and deterministic
- Assume noisy real-world data
- Instruct model to return null if uncertain
- Output as YAML
"""

    def run(self, task_config: TaskConfig, context: str = None) -> PromptObject:
        rules_combined = task_config.hard_rules + task_config.soft_rules
        rules_str = "\n".join([f"- {r}" for r in rules_combined])
        
        prompt = self.PROMPT_TEMPLATE.format(
            task_intent=task_config.intent,
            context=context or "None provided",
            input_modality=task_config.input_modality,
            output_schema=json.dumps(task_config.output_schema, indent=2),
            rules=rules_str
        )
        
        response = self.llm.complete(prompt)
        data = self.llm.parse_yaml(response)
        return PromptObject(**data)
import json
