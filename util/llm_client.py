"""
LLM Client abstraction using official provider SDKs.

Unified interface for multiple LLM providers (OpenAI, Anthropic, Google, DeepSeek, OpenRouter,
llama.cpp, vLLM, SGLang) with automatic retry, error handling, and structured output support.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import LLMSettings, get_settings
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any = None

    def parse_json(self) -> dict[str, Any]:
        """Parse response content as JSON."""
        try:
            content = self.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e

    def parse_model(self, model_class: type[BaseModel]) -> BaseModel:
        """Parse response content into a Pydantic model."""
        data = self.parse_json()
        return model_class.model_validate(data)


@dataclass
class TokenTracker:
    """Accumulates token usage across multiple LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def record(self, usage: dict[str, int]) -> None:
        """Add a single LLM response's usage to running totals."""
        if not usage:
            return
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.total_tokens += int(usage.get("total_tokens", 0))

    def totals(self) -> dict[str, int]:
        """Return aggregated token usage."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


DEEPSEEK_THINKING_ENABLED = False

_THINKING_TOKEN_RE = re.compile(r"<\|channel\|>.*?<\|end\|>", re.DOTALL)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)
_STRAY_TAGS_RE = re.compile(r"<\|(channel|message|end)\|>")
_STRAY_THINK_RE = re.compile(r"</?think>")


def strip_thinking_tokens(text: str) -> str:
    """Remove model thinking tokens from content."""
    text = _THINKING_TOKEN_RE.sub("", text)
    text = _THINK_TAG_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    text = _STRAY_TAGS_RE.sub("", text)
    text = _STRAY_THINK_RE.sub("", text)
    return text.strip()


class LLMClient:
    """Unified LLM client with provider abstraction using official SDKs."""

    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or get_settings().llm
        self._openai_client: AsyncOpenAI | None = None
        self._deepseek_client: AsyncOpenAI | None = None
        self._openrouter_client: AsyncOpenAI | None = None
        self._llama_cpp_client: AsyncOpenAI | None = None
        self._vllm_client: AsyncOpenAI | None = None
        self._sglang_client: AsyncOpenAI | None = None
        self._anthropic_client: AsyncAnthropic | None = None
        self._google_configured: bool = False
        self._configure_clients()

    _LOCAL_PROVIDERS = {"llama_cpp", "vllm", "sglang"}

    def _configure_clients(self) -> None:
        """Configure provider SDK clients based on available API keys."""
        local_timeout = None
        api_timeout = self.settings.timeout

        if self.settings.openai_api_key:
            kwargs_oai: dict[str, Any] = {
                "api_key": self.settings.openai_api_key,
                "timeout": api_timeout,
            }
            if self.settings.openai_base_url:
                kwargs_oai["base_url"] = self.settings.openai_base_url
            self._openai_client = AsyncOpenAI(**kwargs_oai)

        if self.settings.deepseek_api_key:
            self._deepseek_client = AsyncOpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
                timeout=api_timeout,
            )

        if self.settings.openrouter_api_key:
            self._openrouter_client = AsyncOpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
                timeout=api_timeout,
            )

        if self.settings.provider == "llama_cpp":
            self._llama_cpp_client = AsyncOpenAI(
                api_key=self.settings.llama_cpp_api_key,
                base_url=self.settings.llama_cpp_base_url,
                timeout=local_timeout,
            )

        if self.settings.provider == "vllm":
            self._vllm_client = AsyncOpenAI(
                api_key=self.settings.vllm_api_key,
                base_url=self.settings.vllm_base_url,
                timeout=local_timeout,
            )

        if self.settings.provider == "sglang":
            self._sglang_client = AsyncOpenAI(
                api_key=self.settings.sglang_api_key,
                base_url=self.settings.sglang_base_url,
                timeout=local_timeout,
            )

        if self.settings.anthropic_api_key:
            kwargs: dict[str, Any] = {
                "api_key": self.settings.anthropic_api_key,
                "timeout": self.settings.timeout,
            }
            if self.settings.anthropic_base_url:
                kwargs["base_url"] = self.settings.anthropic_base_url
            self._anthropic_client = AsyncAnthropic(**kwargs)

        if self.settings.google_api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.settings.google_api_key)
            self._google_configured = True

    def _get_model_name(self, model: str | None = None) -> str:
        return model or self.settings.model

    def _get_openai_compatible_client(self) -> AsyncOpenAI:
        """Get the appropriate OpenAI-compatible client for the current provider."""
        if self.settings.provider == "llama_cpp":
            if not self._llama_cpp_client:
                raise RuntimeError(
                    f"llama.cpp client not available. Check LLAMA_CPP_BASE_URL ({self.settings.llama_cpp_base_url})."
                )
            return self._llama_cpp_client

        if self.settings.provider == "vllm":
            if not self._vllm_client:
                raise RuntimeError(
                    f"vLLM client not available. Check VLLM_BASE_URL ({self.settings.vllm_base_url})."
                )
            return self._vllm_client

        if self.settings.provider == "sglang":
            if not self._sglang_client:
                raise RuntimeError(
                    f"SGLang client not available. Check SGLANG_BASE_URL ({self.settings.sglang_base_url})."
                )
            return self._sglang_client

        if self.settings.provider == "deepseek":
            if not self._deepseek_client:
                raise RuntimeError(
                    "DeepSeek API key not configured. Set DEEPSEEK_API_KEY environment variable."
                )
            return self._deepseek_client

        if self.settings.provider == "openrouter":
            if not self._openrouter_client:
                raise RuntimeError(
                    "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."
                )
            return self._openrouter_client

        if not self._openai_client:
            raise RuntimeError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
            )
        return self._openai_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a completion request to the LLM."""
        model_name = self._get_model_name(model)
        provider = self.settings.provider
        temperature = temperature if temperature is not None else self.settings.temperature
        max_tokens = max_tokens or self.settings.max_tokens

        logger.info(
            "Sending LLM request",
            model=model_name,
            provider=provider,
            prompt_length=len(prompt),
            json_mode=json_mode,
        )

        try:
            if provider in ("openai", "deepseek", "openrouter", "llama_cpp", "vllm", "sglang"):
                return await self._complete_openai_compatible(
                    prompt=prompt,
                    system=system,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )

            elif provider == "anthropic":
                return await self._complete_anthropic(
                    prompt=prompt,
                    system=system,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )

            elif provider == "google":
                return await self._complete_google(
                    prompt=prompt,
                    system=system,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            logger.error(
                "LLM request failed",
                model=model_name,
                provider=provider,
                error=str(e),
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a multi-turn chat request."""
        model_name = self._get_model_name(model)
        provider = self.settings.provider
        temperature = temperature if temperature is not None else self.settings.temperature
        max_tokens = max_tokens or self.settings.max_tokens

        if provider not in ("openai", "deepseek", "openrouter", "llama_cpp", "vllm", "sglang"):
            raise NotImplementedError(f"chat() not supported for provider '{provider}'")

        return await self._chat_openai_compatible(
            messages=messages,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    async def _chat_openai_compatible(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        """Handle multi-turn chat for OpenAI-compatible providers."""
        client = self._get_openai_compatible_client()

        clean_messages = []
        for msg in messages:
            c = msg.get("content", "")
            if msg["role"] == "assistant" and ("<|" in c or "<think>" in c):
                clean_messages.append({**msg, "content": strip_thinking_tokens(c)})
            else:
                clean_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": clean_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.settings.provider == "deepseek" and "v4" in model_name and not DEEPSEEK_THINKING_ENABLED:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if self.settings.provider == "openrouter":
            extra = kwargs.get("extra_body", {}) or {}
            extra["provider"] = {"only": ["SiliconFlow", "AtlasCloud", "NovitaAI"]}
            kwargs["extra_body"] = extra

        response = await client.chat.completions.create(**kwargs)

        content = strip_thinking_tokens(response.choices[0].message.content or "")
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        logger.info(
            "LLM chat response received",
            model=model_name,
            tokens=usage["total_tokens"],
            turns=len(messages),
        )

        return LLMResponse(
            content=content,
            model=model_name,
            provider=self.settings.provider,
            usage=usage,
            raw_response=response,
        )

    async def _complete_openai_compatible(
        self,
        prompt: str,
        system: str | None,
        model_name: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        """Handle completion for OpenAI-compatible providers."""
        client = self._get_openai_compatible_client()

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.settings.provider == "deepseek" and "v4" in model_name and not DEEPSEEK_THINKING_ENABLED:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if self.settings.provider == "openrouter":
            extra = kwargs.get("extra_body", {}) or {}
            extra["provider"] = {"only": ["SiliconFlow", "AtlasCloud", "NovitaAI"]}
            kwargs["extra_body"] = extra

        response = await client.chat.completions.create(**kwargs)

        content = strip_thinking_tokens(response.choices[0].message.content or "")
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        logger.info(
            "LLM response received",
            model=model_name,
            tokens=usage["total_tokens"],
        )

        return LLMResponse(
            content=content,
            model=model_name,
            provider=self.settings.provider,
            usage=usage,
            raw_response=response,
        )

    async def _complete_anthropic(
        self,
        prompt: str,
        system: str | None,
        model_name: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        """Handle completion for Anthropic provider."""
        if not self._anthropic_client:
            raise RuntimeError(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."
            )

        effective_system = system or ""
        if json_mode and "JSON" not in effective_system.upper():
            effective_system = (
                effective_system + "\n\nRespond with valid JSON only, no additional text."
            ).strip()

        response = await self._anthropic_client.messages.create(
            model=model_name,
            system=effective_system if effective_system else "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content = block.text
                    break
        usage = {
            "prompt_tokens": response.usage.input_tokens if response.usage else 0,
            "completion_tokens": response.usage.output_tokens if response.usage else 0,
            "total_tokens": (
                (response.usage.input_tokens + response.usage.output_tokens)
                if response.usage
                else 0
            ),
        }

        logger.info(
            "LLM response received",
            model=model_name,
            tokens=usage["total_tokens"],
        )

        return LLMResponse(
            content=content,
            model=model_name,
            provider=self.settings.provider,
            usage=usage,
            raw_response=response,
        )

    async def _complete_google(
        self,
        prompt: str,
        system: str | None,
        model_name: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        """Handle completion for Google Generative AI provider."""
        if not self._google_configured:
            raise RuntimeError(
                "Google API key not configured. Set GOOGLE_API_KEY environment variable."
            )

        import google.generativeai as genai

        def _run_google() -> Any:
            generation_config: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            if json_mode:
                generation_config["response_mime_type"] = "application/json"

            model_obj = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system if system else None,
            )

            return model_obj.generate_content(
                prompt,
                generation_config=generation_config,
            )

        response = await asyncio.to_thread(_run_google)

        content = response.text or ""
        usage_meta = getattr(response, "usage_metadata", None)
        usage = {
            "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0,
            "completion_tokens": (
                getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0
            ),
            "total_tokens": getattr(usage_meta, "total_token_count", 0) if usage_meta else 0,
        }

        logger.info(
            "LLM response received",
            model=model_name,
            tokens=usage["total_tokens"],
        )

        return LLMResponse(
            content=content,
            model=model_name,
            provider=self.settings.provider,
            usage=usage,
            raw_response=response,
        )

