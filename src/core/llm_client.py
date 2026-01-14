import os
import json
import yaml
import re
from typing import Optional, Any, Dict, Union
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging
import sys

# Configure logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-lite"):
        self.provider = provider
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Token Tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
        # Pricing for Gemini 2.0 Flash Lite (per 1M tokens)
        self.input_price_per_1m = 0.075
        self.output_price_per_1m = 0.30
        self.usd_to_inr = 87.0  # Approx conversion rate
        
        if self.provider == "gemini":
            if not self.api_key:
                print("Warning: GEMINI_API_KEY not set. Calls might fail if not using default auth.")
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = model_name
        
    def _track_usage(self, usage_metadata):
        if not usage_metadata:
            return
            
        in_tokens = getattr(usage_metadata, "prompt_token_count", 0)
        out_tokens = getattr(usage_metadata, "candidates_token_count", 0)
        
        self.total_input_tokens += in_tokens
        self.total_output_tokens += out_tokens
        
        # Determine pricing based on model name (simplified logic)
        # Using the lite pricing user provided as default
        cost_usd = (in_tokens / 1_000_000 * self.input_price_per_1m) + \
                   (out_tokens / 1_000_000 * self.output_price_per_1m)
        self.total_cost += cost_usd
        
        cost_inr = cost_usd * self.usd_to_inr
        total_inr = self.total_cost * self.usd_to_inr
        
        print(f"💰 [Cost] Call: ${cost_usd:.6f} (₹{cost_inr:.4f}) | Total: ${self.total_cost:.6f} (₹{total_inr:.4f}) | Tokens: In={self.total_input_tokens}, Out={self.total_output_tokens}")

    def complete(self, prompt: Union[str, list], system_prompt: Optional[str] = None, image_path: Optional[str] = None) -> str:
        if self.provider == "mock":
            return self._mock_response(prompt if isinstance(prompt, str) else str(prompt))
        
        if self.provider == "gemini":
            if image_path:
                return self._gemini_image_call(image_path, prompt if isinstance(prompt, str) else str(prompt), system_prompt)
            else:
                return self._gemini_text_call(prompt, system_prompt)
        
        raise ValueError(f"Unknown provider: {self.provider}")

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _gemini_text_call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        config = types.GenerateContentConfig(
            temperature=0,
            system_instruction=[types.Part.from_text(text=system_prompt)] if system_prompt else None
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=config
        )
        
        self._track_usage(getattr(response, "usage_metadata", None))
        return response.text

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _gemini_image_call(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        with open(image_path, "rb") as image_file:
            img_content = image_file.read()

        sys_instr = [types.Part.from_text(text=system_prompt)] if system_prompt else []

        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            system_instruction=sys_instr,
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        data=img_content,
                        mime_type="image/jpeg"
                    ),
                    types.Part.from_text(text=prompt)
                ],
            ),
        ]

        full_text = ""
        last_chunk = None
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                full_text += chunk.text
            last_chunk = chunk
            
        if last_chunk:
            self._track_usage(getattr(last_chunk, "usage_metadata", None))

        return self._clean_json_response(full_text.strip())

    def _clean_json_response(self, response_text: str) -> str:
        # 1. Try direct load
        try:
            json.loads(response_text)
            return response_text
        except json.JSONDecodeError:
            pass
        
        # 2. Extract from code blocks
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
             try:
                 json.loads(match)
                 return match
             except: continue

        # 3. Extract from first { to last } (heuristic)
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
             candidate = response_text[start:end+1]
             try:
                 json.loads(candidate)
                 return candidate
             except: pass
             
        # 4. Fallback cleanup
        cleaned = (
            response_text.replace("```json", "")
                         .replace("```", "")
                         .strip()
        )
        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            pass

        return response_text

    def _mock_response(self, prompt: str) -> str:
        # Mock logic preserved
        return "{}"

    def parse_json(self, response: str) -> Dict[str, Any]:
        cleaned = self._clean_json_response(response)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {response[:100]}...")
            return {}

    def parse_yaml(self, response: str) -> Dict[str, Any]:
        # 1. Try generic code block extraction
        # Matches ```yaml ... ``` or just ``` ... ```
        pattern = r"```(?:yaml)?\s*(.*?)\s*```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            clean_text = match.group(1)
            try:
                return yaml.safe_load(clean_text)
            except: pass
        
        # 2. Try raw parse (e.g. if model didn't use code blocks but returned valid YAML)
        try:
            return yaml.safe_load(response)
        except: pass
            
        print(f"Failed to parse YAML from: {response[:200]}...")
        return {}
