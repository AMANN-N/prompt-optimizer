from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import re


class ConstraintCheckResult(BaseModel):
    all_constraints_satisfied: bool
    satisfied_constraints: List[str]
    violated_constraints: List[str]
    constraint_details: Dict[str, Dict[str, str]]


class ConstraintChecker:
    CONSTRAINTS = [
        {
            "name": "system_role_non_empty",
            "check": lambda p: p.get("system_role", "").strip() != "",
            "error": "system_role must be non-empty",
            "severity": "critical",
        },
        {
            "name": "task_instruction_non_empty",
            "check": lambda p: p.get("task_instruction", "").strip() != "",
            "error": "task_instruction must be non-empty",
            "severity": "critical",
        },
        {
            "name": "field_definitions_non_empty",
            "check": lambda p: p.get("field_definitions", "").strip() != "",
            "error": "field_definitions must be non-empty",
            "severity": "critical",
        },
        {
            "name": "extraction_steps_ordered",
            "check": lambda p: bool(
                re.search(r"step\s*\d", p.get("extraction_steps", ""), re.IGNORECASE)
            ),
            "error": "extraction_steps should have ordered steps (Step 1, Step 2, etc.)",
            "severity": "warning",
        },
        {
            "name": "error_handling_mentions_null",
            "check": lambda p: "null" in p.get("error_handling", "").lower()
            or "none" in p.get("error_handling", "").lower(),
            "error": "error_handling should mention how to handle null/missing values",
            "severity": "warning",
        },
        {
            "name": "output_format_specified",
            "check": lambda p: p.get("output_format", "").strip() != "",
            "error": "output_format must be specified",
            "severity": "critical",
        },
        {
            "name": "no_example_specific_phrases",
            "check": lambda p: not bool(
                re.search(
                    r"as shown in|like the (first|second|third) example|from (image|file) \d",
                    p.get("task_instruction", ""),
                    re.IGNORECASE,
                )
            ),
            "error": "prompt should not contain example-specific references",
            "severity": "warning",
        },
        {
            "name": "deterministic_instructions",
            "check": lambda p: not bool(
                re.search(
                    r"might|perhaps|possibly|usually",
                    p.get("task_instruction", ""),
                    re.IGNORECASE,
                )
            ),
            "error": "instructions should be deterministic, avoid hedging language",
            "severity": "warning",
        },
    ]

    def __init__(self, custom_constraints: List[Dict[str, Any]] = None):
        self.constraints = self.CONSTRAINTS.copy()
        if custom_constraints:
            self.constraints.extend(custom_constraints)

    def check(self, prompt_dict: Dict[str, Any]) -> ConstraintCheckResult:
        satisfied = []
        violated = []
        details = {}

        for constraint in self.constraints:
            name = constraint["name"]
            check_func = constraint["check"]
            error_msg = constraint["error"]
            severity = constraint.get("severity", "warning")

            try:
                passed = check_func(prompt_dict)
                if passed:
                    satisfied.append(name)
                    details[name] = {
                        "status": "passed",
                        "message": "",
                        "severity": severity,
                    }
                else:
                    violated.append(name)
                    details[name] = {
                        "status": "failed",
                        "message": error_msg,
                        "severity": severity,
                    }
            except Exception as e:
                violated.append(name)
                details[name] = {
                    "status": "error",
                    "message": str(e),
                    "severity": severity,
                }

        critical_violated = [
            c for c in violated if details[c]["severity"] == "critical"
        ]
        all_satisfied = len(critical_violated) == 0

        return ConstraintCheckResult(
            all_constraints_satisfied=all_satisfied,
            satisfied_constraints=satisfied,
            violated_constraints=violated,
            constraint_details=details,
        )

    def should_accept_prompt(self, result: ConstraintCheckResult) -> tuple:
        critical_violated = [
            c
            for c in result.violated_constraints
            if result.constraint_details[c]["severity"] == "critical"
        ]

        if critical_violated:
            return (
                False,
                f"Critical constraints violated: {', '.join(critical_violated)}",
            )
        if len(result.violated_constraints) > 3:
            return (
                False,
                f"Too many constraint violations ({len(result.violated_constraints)}). Review and fix.",
            )
        return True, "Constraints satisfied"


def check_constraints(prompt_dict: Dict[str, Any]) -> ConstraintCheckResult:
    checker = ConstraintChecker()
    return checker.check(prompt_dict)
