from src.agents.base import BaseAgent
from src.core.config import PromptObject
from typing import List
import yaml

class PromptVariantGenerator(BaseAgent):
    PROMPT_TEMPLATE = """
You are a Prompt Variant Generator.

Generate {N} alternative prompts by varying:
- instruction phrasing
- field definitions
- extraction steps

Constraints:
- All prompts must obey the same schema
- Do not weaken rules
- Keep outputs deterministic

Base Prompt:
{base_prompt}

Return a list of prompt variants in YAML.
"""

    def run(self, base_prompt: PromptObject, n: int = 3) -> List[PromptObject]:
        prompt = self.PROMPT_TEMPLATE.format(
            N=n,
            base_prompt=base_prompt.to_yaml()
        )
        
        response = self.llm.complete(prompt)
        data = self.llm.parse_yaml(response) or []
        
        # Expecting a list of prompt objects
        variants = []
        if isinstance(data, list):
            for item in data:
                try:
                    variants.append(PromptObject(**item))
                except:
                    continue
        return variants
