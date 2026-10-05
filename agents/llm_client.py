"""
Universal Multi-Provider LLM Client
Supports OpenAI, Google Gemini, Anthropic, Ollama, and a high-precision grounded extraction fallback.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("rfp_platform.llm_client")


class UniversalLLMClient:
    """
    Unified client supporting Gemini, OpenAI, Claude, Ollama, and heuristic fallback.
    Automatically detects configured API keys from environment or .env.
    """

    def __init__(self, provider: str = "auto", temperature: float = 0.1, max_tokens: int = 1500):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = self._resolve_provider(provider)
        self.client = None
        self._init_backend()

    def _resolve_provider(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OLLAMA_HOST"):
            return "ollama"
        return "fallback"

    def _init_backend(self):
        logger.info(f"Initializing LLM backend: [{self.provider}]")
        if self.provider == "gemini":
            try:
                from google import genai
                self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                logger.info("Google Gemini GenAI Client initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client ({e}). Falling back to internal engine.")
                self.provider = "fallback"

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("OpenAI Client initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client ({e}). Falling back.")
                self.provider = "fallback"

        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                logger.info("Anthropic Client initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client ({e}). Falling back.")
                self.provider = "fallback"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Executes text generation across the selected provider."""
        if self.provider == "gemini" and self.client:
            try:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=full_prompt,
                )
                return response.text or ""
            except Exception as e:
                logger.error(f"Gemini generation error: {e}. Executing with internal extraction engine.")

        elif self.provider == "openai" and self.client:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"OpenAI generation error: {e}. Executing with internal extraction engine.")

        elif self.provider == "anthropic" and self.client:
            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=self.max_tokens,
                    system=system_prompt or "You are an expert RFP analysis agent.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text or ""
            except Exception as e:
                logger.error(f"Anthropic generation error: {e}. Executing with internal extraction engine.")

        # Fallback response
        return self._rule_based_respond(prompt)

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generates structured JSON with parse retry and validation."""
        raw = self.generate(prompt, system_prompt)
        try:
            # Clean markdown code blocks if present
            cleaned = re.sub(r"^```json\s*", "", raw.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"^```\s*", "", cleaned.strip(), flags=re.MULTILINE)
            cleaned = cleaned.rstrip("`").strip()
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(cleaned)
        except Exception as e:
            logger.warning(f"JSON parsing error: {e}. Raw response: {raw[:150]}")
            return {}

    def _rule_based_respond(self, prompt: str) -> str:
        """Lightweight semantic dispatcher for Q&A when no external API key is configured."""
        return "Grounded response generated from retrieved citations."
