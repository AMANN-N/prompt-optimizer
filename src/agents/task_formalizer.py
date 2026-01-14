from src.agents.base import BaseAgent
from src.core.config import TaskConfig
import json

class TaskFormalizer(BaseAgent):
    PROMPT_TEMPLATE = """
You are a Task Formalization Agent.

Your job is to convert a human intent into a precise, machine-readable task definition.

Input:
- User intent: {user_intent}
- Fields to extract: {fields}
- Constraints: {constraints}

Output a JSON with:
1. task_name (snake_case)
2. intent (1-2 lines, unambiguous)
3. input_modality
4. output_schema (strict JSON schema)
5. hard_rules (must never be violated)
6. soft_rules (quality preferences)
7. evaluation_hints (what success looks like)

Rules:
- Do NOT invent fields
- Do NOT add explanations
- Output valid JSON only
"""

    def run(self, user_intent: str, fields: list = None, constraints: list = None) -> TaskConfig:
        fields_str = ", ".join(fields) if fields else "implied from intent"
        constraints_str = ", ".join(constraints) if constraints else "none"
        
        prompt = self.PROMPT_TEMPLATE.format(
            user_intent=user_intent,
            fields=fields_str,
            constraints=constraints_str
        )
        
        response = self.llm.complete(prompt)
        data = self.llm.parse_json(response)
        return TaskConfig(**data)
