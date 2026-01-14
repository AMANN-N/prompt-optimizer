from src.agents.base import BaseAgent
from src.core.config import PromptObject
from typing import List, Dict, Any
import json


class PromptSelector(BaseAgent):
    PROMPT_TEMPLATE = """
You are a Prompt Selection Agent.

Given:
- Prompt versions and their evaluation scores.

{candidates}

Select:
- best_prompt_index
- backup_prompt_index
- explanation

Output JSON only.
"""

    def run(self, candidates: List[PromptObject], reports: List) -> Dict[str, Any]:
        candidates_info = []
        for i, (p, r) in enumerate(zip(candidates, reports)):
            score = (
                r.overall_score
                if hasattr(r, "overall_score")
                else r.get("overall_score", 0)
            )
            failures = (
                r.failure_patterns
                if hasattr(r, "failure_patterns")
                else r.get("failure_patterns", {})
            )
            candidates_info.append({"index": i, "score": score, "failures": failures})

        prompt = self.PROMPT_TEMPLATE.format(
            candidates=json.dumps(candidates_info, indent=2)
        )

        response = self.llm.complete(prompt)
        return self.llm.parse_json(response)
