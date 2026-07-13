"""Provider abstraction for real LLM API access and testing."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_local_dotenv(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_dotenv()


@dataclass(slots=True)
class ProviderResponse:
    """Normalized model response across providers."""

    content: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    raw_response: Dict[str, Any] = field(default_factory=dict)


class BaseProvider:
    """Unified provider interface used by systems under test."""

    provider_name: str

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> ProviderResponse:
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    provider_name = "openai"

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1/chat/completions") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIProvider")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> ProviderResponse:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        def _call() -> Dict[str, Any]:
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as res:
                return json.loads(res.read().decode("utf-8"))

        raw = await asyncio.to_thread(_call)
        content = raw["choices"][0]["message"]["content"]
        usage = raw.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        # Conservative fallback estimate if provider does not return billing details.
        cost_usd = (prompt_tokens * 0.000005) + (completion_tokens * 0.000015)
        return ProviderResponse(
            content=content,
            provider=self.provider_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            raw_response=raw,
        )


class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.anthropic.com/v1/messages") -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicProvider")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> ProviderResponse:
        merged = "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages)
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": merged}],
        }

        def _call() -> Dict[str, Any]:
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as res:
                return json.loads(res.read().decode("utf-8"))

        raw = await asyncio.to_thread(_call)
        content_blocks = raw.get("content", [])
        content = "\n".join(block.get("text", "") for block in content_blocks)
        usage = raw.get("usage", {})
        prompt_tokens = int(usage.get("input_tokens", 0))
        completion_tokens = int(usage.get("output_tokens", 0))
        cost_usd = (prompt_tokens * 0.000008) + (completion_tokens * 0.000024)
        return ProviderResponse(
            content=content,
            provider=self.provider_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            raw_response=raw,
        )


class GeminiProvider(BaseProvider):
    provider_name = "gemini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url_prefix: str = "https://generativelanguage.googleapis.com/v1beta/models",
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.base_url_prefix = base_url_prefix
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required for GeminiProvider")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> ProviderResponse:
        merged = "\n".join(msg.get("content", "") for msg in messages)
        payload = {
            "contents": [{"parts": [{"text": merged}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        url = f"{self.base_url_prefix}/{model}:generateContent?key={self.api_key}"

        def _call() -> Dict[str, Any]:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as res:
                return json.loads(res.read().decode("utf-8"))

        raw = await asyncio.to_thread(_call)
        candidates = raw.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        content = "\n".join(part.get("text", "") for part in parts)
        usage = raw.get("usageMetadata", {})
        prompt_tokens = int(usage.get("promptTokenCount", 0))
        completion_tokens = int(usage.get("candidatesTokenCount", 0))
        cost_usd = (prompt_tokens * 0.0000025) + (completion_tokens * 0.0000075)
        return ProviderResponse(
            content=content,
            provider=self.provider_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            raw_response=raw,
        )


class MockProvider(BaseProvider):
    provider_name = "mock"

    @staticmethod
    def _extract_field(pattern: str, text: str, default: str = "") -> str:
        import re

        match = re.search(pattern, text, flags=re.IGNORECASE)
        return match.group(1).strip() if match else default

    @staticmethod
    def _extract_household(text: str) -> int:
        import re

        for pattern in [r"family of\s+(\d+)", r"household(?: size)?\s+(\d+)", r"(\d+)-person household"]:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 1

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> ProviderResponse:
        merged = "\n".join(msg.get("content", "") for msg in messages)

        if "Return JSON: {\"score\": <float>" in merged:
            content = json.dumps({"score": 0.78, "reason": "Reasonable structure and fidelity."})
        elif "Extract a structured intake JSON" in merged:
            input_text = merged.split("INPUT:\n", 1)[-1] if "INPUT:\n" in merged else merged
            client_name = self._extract_field(r"(?:Client|caller|Intake note:)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", input_text, "Unknown")
            household_size = self._extract_household(input_text)
            services = [
                service
                for service in [
                    "groceries",
                    "diapers",
                    "baby formula",
                    "transportation voucher",
                    "utility assistance",
                    "rent referral",
                ]
                if service in input_text.lower()
            ]
            if not services:
                services = ["groceries"]
            urgency = "high" if any(token in input_text.lower() for token in ["urgent", "high priority", "zero food", "lost job"]) else "medium"
            preferred_language = "Spanish" if "spanish" in input_text.lower() else "English"
            if "bengali" in input_text.lower():
                preferred_language = "Bengali"
            content = json.dumps(
                {
                    "client_name": client_name,
                    "household_size": household_size,
                    "urgency": urgency,
                    "requested_services": services,
                    "preferred_language": preferred_language,
                    "notes_summary": f"Household of {household_size} requesting {', '.join(services)}.",
                }
            )
        else:
            content = json.dumps(
                {
                    "client_name": "Unknown",
                    "household_size": 1,
                    "urgency": "medium",
                    "requested_services": ["groceries"],
                    "preferred_language": "English",
                    "notes_summary": merged[:120],
                }
            )
        token_estimate = max(1, len(merged) // 4)
        return ProviderResponse(
            content=content,
            provider=self.provider_name,
            model=model,
            prompt_tokens=token_estimate,
            completion_tokens=80,
            cost_usd=0.0,
            raw_response={"mock": True, "content": content},
        )


def provider_from_environment(prefer: str | None = None) -> BaseProvider:
    """Choose a real provider when credentials are available, otherwise fallback to mock."""

    preferred = (prefer or "").lower().strip()
    if preferred == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider()
    if preferred == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if preferred == "gemini" and os.getenv("GOOGLE_API_KEY"):
        return GeminiProvider()

    if os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider()
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if os.getenv("GOOGLE_API_KEY"):
        return GeminiProvider()

    return MockProvider()
