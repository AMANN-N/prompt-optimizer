from src.agents.base import BaseAgent
from src.core.config import FailureAnalysis
from typing import List, Any, Dict
import json


class FailureAnalyzer(BaseAgent):
    PROMPT_TEMPLATE = """
You are a Failure Analysis Agent.

Analyze the evaluation report.

Report:
{report}

Your task:
1. Identify dominant failure modes
2. Classify them into:
   - instruction ambiguity
   - missing constraints
   - field confusion
   - format errors
   - hallucinations
3. For each failure mode, explain the root cause
4. Suggest which prompt section should change

Output JSON (list of objects) with:
- failure_type
- frequency
- root_cause
- affected_prompt_section
"""

    def run(self, report: Any) -> List[FailureAnalysis]:
        if hasattr(report, "model_dump_json"):
            report_json = report.model_dump_json(indent=2)
        elif hasattr(report, "model_dump"):
            report_json = json.dumps(report.model_dump(), indent=2, default=str)
        elif isinstance(report, dict):
            report_json = json.dumps(report, indent=2, default=str)
        else:
            report_json = str(report)

        prompt = self.PROMPT_TEMPLATE.format(report=report_json)

        response = self.llm.complete(prompt)
        data = self.llm.parse_json(response)

        # Ensure it's a list
        if isinstance(data, dict) and "failures" in data:
            data = data["failures"]

        final_list = []
        if isinstance(data, list):
            for item in data:
                # Sanitize frequency
                freq = item.get("frequency")
                if isinstance(freq, str):
                    freq_lower = freq.lower()
                    if "high" in freq_lower:
                        item["frequency"] = 3
                    elif "medium" in freq_lower:
                        item["frequency"] = 2
                    elif "low" in freq_lower:
                        item["frequency"] = 1
                    else:
                        # Try to parse int, default to 1
                        try:
                            item["frequency"] = int(freq)
                        except:
                            item["frequency"] = 1

                final_list.append(item)

        return [FailureAnalysis(**item) for item in final_list]
