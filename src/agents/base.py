from typing import Any
from src.core.llm_client import LLMClient

class BaseAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError
