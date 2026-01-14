from src.agents.base import BaseAgent
from src.core.config import PromptObject
import yaml
import json

class PromptFreezer(BaseAgent):
    PROMPT_TEMPLATE = """
You are a Prompt Freezing Agent.

Given the best-performing prompt:
{prompt}

- Validate it against all rules
- Confirm stability across cases
- Mark it production-ready

Output JSON:
- frozen_prompt (the prompt content)
- version_id (string)
- known_limitations (list)
"""

    def run(self, prompt_object: PromptObject) -> dict:
        prompt = self.PROMPT_TEMPLATE.format(
            prompt=prompt_object.to_yaml()
        )
        
        response = self.llm.complete(prompt)
        return self.llm.parse_json(response)
